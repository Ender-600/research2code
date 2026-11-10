#!/usr/bin/env python3
"""Run generated code one step at a time inside an OS-level sandbox.

R2C installs, imports and executes code an LLM wrote from an untrusted paper.
The opencode server, the model key and this driver stay outside; every step
that touches generated code runs through Anthropic's Sandbox Runtime (`srt`:
Seatbelt on macOS, bubblewrap on Linux) with only what that step needs:

    step                 can write                network
    pip install          the run's own venv       PyPI only
    everything else      the run directory        none

No step sees the provider keys: the environment is built from an allowlist.
No step can read the home directory, other runs or the repo's .env.
Every step also gets a HOME that is deleted afterwards and the run's cache
directory, which is kept so that caches are built once per run.

The run's venv sits next to the run directory, not in it, because the driver
snapshots the whole run tree around every agent dispatch. It chains to the
driver's site-packages through a .pth file, so installed packages are reused
and a generated package's own dependencies never touch the driver's
environment. The driver never executes anything from that venv itself.

Limits: a step can still forge its own output (the probe battery and the
generated code share a process), and srt does not limit memory or CPU.

Set R2C_EXEC_SANDBOX=off to run without it. Inside the Docker sandbox
(R2C_SANDBOX=1) the default is off, since srt would have to run in a weakened
nested mode there.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import site
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from urllib.parse import urlsplit

MODE_ENV = "R2C_EXEC_SANDBOX"
SRT_BIN_ENV = "R2C_SRT_BIN"
EXTRA_READ_ENV = "R2C_SANDBOX_EXTRA_READ"
SMOKE_DOMAINS_ENV = "R2C_SANDBOX_SMOKE_DOMAINS"
PIP_DOMAINS_ENV = "R2C_SANDBOX_PIP_DOMAINS"
KERNEL_TRANSPORT_ENV = "R2C_KERNEL_TRANSPORT"
INSIDE_ENV = "R2C_INSIDE_EXEC_SANDBOX"

SRT_INSTALL_HINT = "npm install -g @anthropic-ai/sandbox-runtime@0.0.81"
VENVS_DIRNAME = ".sandbox_venvs"
CACHE_DIRNAME = "cache"
PIP_DOMAINS = ("pypi.org", "files.pythonhosted.org")

# Exit status a step reports when the sandbox itself could not start.
SANDBOX_FAILURE_EXIT = 126

_PASSED_ENV = (
    "LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "TERM", "TZ", "USER",
    "LOGNAME", "COLUMNS", "LINES",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK",
    "R2C_DRY_RUN_TIMEOUT_S",
)
_PROXY_ENV = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
)
_READ_DENIED_ROOTS = ("/Users", "/home", "/root", "/Volumes", "/mnt", "/media")


class SandboxError(RuntimeError):
    """The sandbox could not be set up, so the step must not run."""


@dataclass(frozen=True)
class Step:
    name: str
    base_domains: tuple[str, ...] = ()
    domains_env: str | None = None
    writes_run_dir: bool = True
    writes_venv: bool = False
    # .pipeline/ entries this step legitimately rewrites. Every other entry
    # that exists when the step starts is read-only to it.
    pipeline_outputs: frozenset[str] = frozenset()


PIP_STEP = Step("pip_install", base_domains=PIP_DOMAINS,
                domains_env=PIP_DOMAINS_ENV,
                writes_run_dir=False, writes_venv=True)
ELEMENT_TEST_STEP = Step("element_test")
# The post-delivery battery writes under details/; .pipeline/ stays frozen.
REVALIDATE_STEP = Step("revalidate")

# Every script that imports or runs the generated package.
# tests/test_exec_sandbox.py fails when a new one is not listed here.
SCRIPT_STEPS: dict[str, Step] = {
    "validate_method_coder_output.py": Step("validate_method_coder_output"),
    "validate_package_imports.py": Step("validate_package_imports"),
    "validate_arch_contract_runtime.py": Step("validate_arch_contract_runtime"),
    "scale_calibration_2x.py": Step(
        "scale_calibration",
        pipeline_outputs=frozenset({"scale_calibration.json"})),
    "budget_sufficiency.py": Step(
        "budget_sufficiency",
        pipeline_outputs=frozenset({"budget_assessment.json"})),
    "smoke_run_notebook.py": Step("smoke", domains_env=SMOKE_DOMAINS_ENV),
    "fix_data_signal.py": Step("fix_data_signal"),
    "run_probes.py": Step(
        "probes",
        pipeline_outputs=frozenset({"probe_report.json", "claims_ledger.json"})),
}


def mode(environ: dict[str, str] | None = None) -> str:
    """"srt" or "off". An unrecognized value keeps the sandbox on."""
    env = os.environ if environ is None else environ
    raw = env.get(MODE_ENV, "").strip().lower()
    if raw in {"off", "0", "false", "no"}:
        return "off"
    if raw:
        return "srt"
    return "off" if env.get("R2C_SANDBOX") else "srt"


def enabled() -> bool:
    return mode() == "srt"


def step_for_args(args: list[str]) -> Step | None:
    """The sandbox step for a `python <args>` invocation, if it needs one."""
    if args[:2] == ["-m", "pip"]:
        return PIP_STEP
    script = next((a for a in args if not a.startswith("-")), "")
    return SCRIPT_STEPS.get(Path(script).name)


def run_dir_from_args(args: list[str]) -> Path:
    """The run directory a step works on, read from its own arguments."""
    for flag, derive in (("--run-dir", lambda p: p), ("-r", lambda p: p.parent)):
        if flag in args and args.index(flag) + 1 < len(args):
            return derive(Path(args[args.index(flag) + 1])).resolve()
    raise SandboxError(f"cannot tell which run this step belongs to: {args}")


def find_srt(environ: dict[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    explicit = env.get(SRT_BIN_ENV, "").strip()
    if explicit:
        return explicit if os.access(explicit, os.X_OK) else None
    found = shutil.which("srt", path=env.get("PATH"))
    if found:
        return found
    # npm's global bin directory is often not on PATH.
    npm = shutil.which("npm", path=env.get("PATH"))
    if npm is None:
        return None
    try:
        prefix = subprocess.run([npm, "prefix", "-g"], capture_output=True,
                                text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    candidate = Path(prefix) / "bin" / "srt"
    return str(candidate) if os.access(candidate, os.X_OK) else None


def run_venv_dir(run_dir: Path) -> Path:
    return run_dir.parent / VENVS_DIRNAME / run_dir.name


def run_site_packages(venv: Path) -> Path:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return venv / "lib" / version / "site-packages"


def _driver_site_dirs() -> list[str]:
    dirs = list(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        dirs.append(site.getusersitepackages())
    return [str(Path(d).resolve()) for d in dirs if Path(d).is_dir()]


def ensure_run_venv(run_dir: Path) -> Path:
    """Create the run's venv if needed and return its interpreter's path."""
    venv = run_venv_dir(run_dir)
    python = venv / "bin" / "python"
    if not python.exists():
        venv.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", "--without-pip", "--symlinks",
                 str(venv)],
                check=True, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", "") or exc
            raise SandboxError(f"could not create {venv}: {detail}") from exc
    site_dir = run_site_packages(venv)
    if not site_dir.is_dir():
        # A step started by hand can use a different `python` from the driver.
        raise SandboxError(
            f"{venv} was built by a different Python version than "
            f"{sys.executable}. Use the Python that ran the pipeline.")
    lines = [f"import site; site.addsitedir({d!r})\n"
             for d in _driver_site_dirs()]
    (site_dir / "_r2c_driver_env.pth").write_text(
        "".join(lines), encoding="utf-8")
    return python


def ensure_run_cache(venv: Path) -> Path:
    """The run's cache directory, which every step of the run can write.

    HOME is new for each step, so without this every step that imports
    matplotlib would spend about 20 s rebuilding its font list. The pip step
    can write the venv, so a package's build script could swap this path for
    a symlink; that must never become a writable root for the next step."""
    cache = venv / CACHE_DIRNAME
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        cache.unlink()
    cache.mkdir(exist_ok=True)
    return cache


def installed_version(run_dir: Path, name: str) -> str:
    """A distribution's version as the run's sandboxed steps will see it.

    Reads metadata only: nothing from the run's venv is imported or executed
    in the driver. Raises metadata.PackageNotFoundError when absent."""
    site_dir = run_site_packages(run_venv_dir(run_dir))
    for dist in metadata.distributions(name=name, path=[str(site_dir)]):
        return dist.version
    return metadata.version(name)


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,\s]+", value) if item.strip()]


def step_domains(step: Step, environ: dict[str, str]) -> list[str]:
    domains = list(step.base_domains)
    if step.domains_env:
        domains += _split_list(environ.get(step.domains_env, ""))
    index_url = environ.get("PIP_INDEX_URL", "").strip()
    if step is PIP_STEP and index_url:
        parts = urlsplit(index_url)
        if parts.username or parts.password:
            raise SandboxError(
                "PIP_INDEX_URL embeds credentials, and package build scripts "
                "run inside this step and could read them. Use R2C_WHEELHOUSE "
                f"or set {MODE_ENV}=off.")
        if parts.hostname:
            domains.append(parts.hostname)
    return sorted(set(domains))


def _siblings(directory: Path, keep: set[str]) -> list[str]:
    if not directory.is_dir():
        return []
    return [str(entry) for entry in directory.iterdir() if entry.name not in keep]


def build_settings(
    step: Step,
    *,
    run_dir: Path,
    repo_root: Path,
    venv: Path,
    tmp: Path,
    domains: list[str],
    environ: dict[str, str],
    platform: str = sys.platform,
) -> dict:
    """The srt policy for one step."""
    runs_root = venv.parent.parent
    deny_read = {str(Path.home()), str(Path("/tmp").resolve()),
                 str(repo_root / ".env"), f"{repo_root}/.env.*",
                 str(repo_root / "input_papers")}
    deny_read.update(root for root in _READ_DENIED_ROOTS if Path(root).exists())
    # Every other run, one entry at a time: a deny nested inside an allow
    # outranks an allow nested inside that deny, so denying all of r2c_runs
    # and re-opening this run would leave this run unreadable.
    deny_read.update(_siblings(runs_root, keep={venv.parent.name, venv.name}))
    deny_read.update(_siblings(venv.parent, keep={venv.name}))

    allow_read = {str(repo_root), str(run_dir), str(venv), str(tmp),
                  str(Path(sys.prefix).resolve()),
                  str(Path(sys.base_prefix).resolve()),
                  str(Path(sys.executable).resolve().parent.parent),
                  *_driver_site_dirs(),
                  *_split_list(environ.get(EXTRA_READ_ENV, ""))}
    wheelhouse = environ.get("R2C_WHEELHOUSE", "").strip()
    if wheelhouse:
        allow_read.add(str(Path(wheelhouse).expanduser().resolve()))

    allow_write = [str(tmp), str(venv / CACHE_DIRNAME)]
    deny_write: list[str] = []
    if step.writes_run_dir:
        allow_write.append(str(run_dir))
        deny_write.append(str(run_dir / ".git"))
        pipeline_dir = run_dir / ".pipeline"
        if pipeline_dir.is_dir():
            deny_write += sorted(
                str(entry) for entry in pipeline_dir.iterdir()
                if entry.name not in step.pipeline_outputs)
    if step.writes_venv:
        allow_write.append(str(venv))

    network: dict = {"allowedDomains": domains, "deniedDomains": []}
    if platform == "darwin":
        # The notebook kernel talks over Unix sockets in the step's own temp
        # directory. TCP would need allowLocalBinding, which on macOS also
        # opens every localhost port, the opencode server's included. Linux
        # steps get a private loopback in their own network namespace.
        network["allowUnixSockets"] = [str(tmp)]

    settings: dict = {
        "network": network,
        "filesystem": {
            "denyRead": sorted(deny_read),
            "allowRead": sorted(allow_read),
            "allowWrite": allow_write,
            "denyWrite": deny_write,
        },
    }
    if platform == "darwin":
        # IPython runs a notebook's `%pip` cell under a pseudo-terminal.
        settings["allowPty"] = True
    return settings


def build_env(
    *,
    venv: Path,
    tmp: Path,
    repo_root: Path,
    run_dir: Path,
    domains: list[str],
    environ: dict[str, str],
    extra_env: dict[str, str] | None = None,
    platform: str = sys.platform,
) -> dict[str, str]:
    """The step's whole environment, from an allowlist: no provider keys."""
    env = {key: environ[key] for key in _PASSED_ENV if key in environ}
    untrusted_roots = (str(repo_root), str(run_dir.parent))
    path = [entry for entry in environ.get("PATH", "").split(os.pathsep)
            if entry.startswith("/") and not entry.startswith(untrusted_roots)]
    env.update({
        "PATH": os.pathsep.join([str(venv / "bin"), *path]),
        "VIRTUAL_ENV": str(venv),
        "HOME": str(tmp / "home"),
        "TMPDIR": str(tmp),
        "CLAUDE_CODE_TMPDIR": str(tmp),  # srt sets the child's TMPDIR from this
        "TMP": str(tmp),
        "TEMP": str(tmp),
        "XDG_CACHE_HOME": str(venv / CACHE_DIRNAME),
        "MPLCONFIGDIR": str(venv / CACHE_DIRNAME / "matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_NO_CACHE_DIR": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        INSIDE_ENV: "1",
    })
    if domains:
        env.update({key: environ[key] for key in _PROXY_ENV if key in environ})
        if environ.get("PIP_INDEX_URL", "").strip():
            env["PIP_INDEX_URL"] = environ["PIP_INDEX_URL"].strip()
    else:
        env["R2C_OFFLINE"] = "1"
    if platform == "darwin":
        env[KERNEL_TRANSPORT_ENV] = "ipc"
        # pip's default TLS check goes through the macOS trust daemon, which
        # the sandbox blocks; its bundled CA list works without it.
        env["PIP_USE_DEPRECATED"] = "legacy-certs"
    env.update(extra_env or {})
    return env


def describe(step: Step) -> str:
    domains = step_domains(step, dict(os.environ))
    network = ", ".join(domains) if domains else "none"
    writes = "run venv" if step.writes_venv else "run directory"
    return f"step={step.name} writes={writes} network={network}"


def _kill_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def run(
    step: Step,
    args: list[str],
    *,
    run_dir: Path,
    repo_root: Path,
    cwd: Path,
    timeout: float | None = None,
    extra_env: dict[str, str] | None = None,
    venv_run_dir: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run `python <args>` in the sandbox, like subprocess.run(capture, text).

    Raises SandboxError if the sandbox cannot start and TimeoutExpired on
    timeout, after killing the step's whole process group. `venv_run_dir`
    names the run whose venv to use when `run_dir` is a temporary copy."""
    environ = dict(os.environ)
    srt = find_srt(environ)
    if srt is None:
        raise SandboxError(f"srt is not installed; run: {SRT_INSTALL_HINT}")
    run_dir = Path(run_dir).resolve()
    repo_root = Path(repo_root).resolve()
    venv_owner = Path(venv_run_dir).resolve() if venv_run_dir else run_dir
    python = ensure_run_venv(venv_owner)
    venv = run_venv_dir(venv_owner)
    ensure_run_cache(venv)
    domains = step_domains(step, environ)

    # Short and under /tmp: macOS caps Unix socket paths at 104 bytes.
    tmp = Path(tempfile.mkdtemp(prefix="r2c-", dir=Path("/tmp").resolve()))
    settings_fd, settings_name = tempfile.mkstemp(
        prefix="r2c-srt-", suffix=".json")
    process: subprocess.Popen | None = None
    try:
        (tmp / "home").mkdir()
        settings = build_settings(step, run_dir=run_dir, repo_root=repo_root,
                                  venv=venv, tmp=tmp, domains=domains,
                                  environ=environ)
        with os.fdopen(settings_fd, "w", encoding="utf-8") as handle:
            json.dump(settings, handle)
        env = build_env(venv=venv, tmp=tmp, repo_root=repo_root,
                        run_dir=run_dir, domains=domains, environ=environ,
                        extra_env=extra_env)
        argv = [srt, "--settings", settings_name, "--", str(python), *args]
        process = subprocess.Popen(
            argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                argv, timeout, output=stdout, stderr=stderr) from None
        return subprocess.CompletedProcess(
            argv, process.returncode, stdout, stderr)
    finally:
        if process is not None:
            # Nothing a step started outlives it.
            _kill_group(process.pid, signal.SIGKILL)
        shutil.rmtree(tmp, ignore_errors=True)
        Path(settings_name).unlink(missing_ok=True)


def run_by_hand(args: list[str]) -> int:
    """Run one step started outside the driver and return its exit code.

    `args` is what would follow `python` on the command line. Relative paths
    are made absolute because the step runs from the repo root, as it does
    under the driver: the caller's own directory may be unreadable in there."""
    args = [str(Path(arg).resolve())
            if not arg.startswith("-") and Path(arg).exists() else arg
            for arg in args]
    if not enabled():
        return subprocess.run([sys.executable, *args]).returncode
    step = step_for_args(args)
    if step is None:
        print(f"FAIL: not a sandboxed step: {' '.join(args)}", file=sys.stderr)
        return 2
    repo_root = Path(__file__).resolve().parent.parent
    try:
        run_dir = run_dir_from_args(args)
        if not run_dir.is_dir():
            raise SandboxError(f"run directory not found: {run_dir}")
        proc = run(step, args, run_dir=run_dir, repo_root=repo_root,
                   cwd=repo_root)
    except SandboxError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return SANDBOX_FAILURE_EXIT
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


def relaunch_in_sandbox(script: str) -> None:
    """Send a step started by hand into its sandbox, then exit.

    The run companion's post-edit gates start steps directly. Run bare they
    would execute generated code outside any sandbox and miss the packages in
    the run's venv. Returns only when the step belongs in this process: the
    sandbox is off, this already is the sandbox, or argparse has to answer
    (`--help`, a missing argument)."""
    if enabled() and not os.environ.get(INSIDE_ENV) and "--run-dir" in sys.argv:
        sys.exit(run_by_hand([script, *sys.argv[1:]]))


def preflight() -> str | None:
    """Why generated code could not be sandboxed on this machine, or None."""
    if not enabled():
        return None
    srt = find_srt()
    if srt is None:
        return (f"srt is not installed.\n  Install it: {SRT_INSTALL_HINT}\n"
                f"  If it is installed elsewhere, set {SRT_BIN_ENV} to its path.")
    settings = {
        "network": {"allowedDomains": [], "deniedDomains": []},
        "filesystem": {"denyRead": [], "allowRead": [], "allowWrite": [],
                       "denyWrite": []},
    }
    with tempfile.NamedTemporaryFile(
            "w", prefix="r2c-srt-", suffix=".json") as handle:
        json.dump(settings, handle)
        handle.flush()
        try:
            proc = subprocess.run(
                [srt, "--settings", handle.name, "--", "/bin/sh", "-c", "exit 0"],
                capture_output=True, text=True, timeout=60,
                stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as exc:
            return f"srt could not start: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[-800:]
        return f"srt could not start a sandbox on this machine:\n{detail}"
    return None


if __name__ == "__main__":
    # With arguments: run one step, e.g. `-m pip install -r <run>/requirements.txt`.
    if sys.argv[1:]:
        sys.exit(run_by_hand(sys.argv[1:]))
    problem = preflight()
    if problem is not None:
        print(problem, file=sys.stderr)
        sys.exit(1)
