"""scripts/exec_sandbox.py: the per-step sandbox around generated code.

These tests need no srt: they check the policy each step gets, that the driver
routes every generated-code step through it, and that a sandbox failure stops
the step. tests/test_exec_sandbox_srt.py runs the policy for real.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

import exec_sandbox as es
import run_pipeline

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
RUN = "this-run"
PROBES = es.SCRIPT_STEPS["run_probes.py"]
SMOKE = es.SCRIPT_STEPS["smoke_run_notebook.py"]

SENSITIVE_ENV = {
    "ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "x", "GEMINI_API_KEY": "x",
    "GOOGLE_GENERATIVE_AI_API_KEY": "x", "OPENCODE_SERVER_PASSWORD": "x",
    "AWS_SECRET_ACCESS_KEY": "x", "GITHUB_TOKEN": "x", "R2C_PORT": "4096",
    "R2C_SERVER_URL": "http://127.0.0.1:4096",
}

# What marks a script as running the generated package.
_EXECUTES = re.compile(
    r"package_loader|NotebookClient\(|exec\(compile\(|runpy\.run_path"
    r'|\[sys\.executable, "-c"|"-m", "pytest"'
)
# Listed with the reason each needs no sandbox of its own.
_NOT_A_STEP = {
    "element_tests.py": "its test runner calls exec_sandbox.run directly",
}


@pytest.fixture
def layout(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / ".env").write_text("KEY=secret\n")
    (repo / "input_papers").mkdir()
    run_dir = repo / "r2c_runs" / RUN
    (run_dir / ".pipeline" / "judge_parts").mkdir(parents=True)
    for name in ("driver_state.json", "probe_report.json", "claims_ledger.json",
                 "stage_2a.complete"):
        (run_dir / ".pipeline" / name).write_text("{}")
    (repo / "r2c_runs" / "other-run" / ".pipeline").mkdir(parents=True)
    venv = repo / "r2c_runs" / es.VENVS_DIRNAME / RUN
    venv.mkdir(parents=True)
    (repo / "r2c_runs" / es.VENVS_DIRNAME / "other-run").mkdir()
    tmp = tmp_path / "steptmp"
    tmp.mkdir()
    return {"repo": repo, "run_dir": run_dir, "venv": venv, "tmp": tmp}


def _settings(step, layout, platform="darwin", environ=None, domains=None):
    environ = environ if environ is not None else {}
    return es.build_settings(
        step, run_dir=layout["run_dir"], repo_root=layout["repo"],
        venv=layout["venv"], tmp=layout["tmp"],
        domains=es.step_domains(step, environ) if domains is None else domains,
        environ=environ, platform=platform)


def _env(layout, environ, domains=()):
    return es.build_env(
        venv=layout["venv"], tmp=layout["tmp"], repo_root=layout["repo"],
        run_dir=layout["run_dir"], domains=list(domains), environ=environ,
        platform="darwin")


# -- network ---------------------------------------------------------------

def test_only_pip_reaches_the_network():
    assert es.step_domains(es.PIP_STEP, {}) == sorted(es.PIP_DOMAINS)
    assert es.step_domains(es.ELEMENT_TEST_STEP, {}) == []
    for name, step in es.SCRIPT_STEPS.items():
        assert es.step_domains(step, {}) == [], name


def test_smoke_network_is_an_explicit_opt_in():
    environ = {es.SMOKE_DOMAINS_ENV: "www.cs.toronto.edu, ossci-datasets.s3.amazonaws.com"}
    assert es.step_domains(SMOKE, environ) == [
        "ossci-datasets.s3.amazonaws.com", "www.cs.toronto.edu"]
    assert es.step_domains(PROBES, environ) == []


def test_custom_index_host_is_allowed_for_pip_only():
    environ = {"PIP_INDEX_URL": "https://mirror.example.org/simple/"}
    assert "mirror.example.org" in es.step_domains(es.PIP_STEP, environ)
    assert es.step_domains(SMOKE, environ) == []


def test_index_url_with_credentials_is_refused():
    environ = {"PIP_INDEX_URL": "https://user:tok@mirror.example.org/simple/"}
    with pytest.raises(es.SandboxError, match="credentials"):
        es.step_domains(es.PIP_STEP, environ)


def test_macos_kernel_uses_unix_sockets_not_localhost(layout):
    # allowLocalBinding on macOS opens every localhost port, including the
    # opencode server's, which has no password.
    network = _settings(SMOKE, layout)["network"]
    assert "allowLocalBinding" not in network
    assert network["allowUnixSockets"] == [str(layout["tmp"])]
    assert _env(layout, {})[es.KERNEL_TRANSPORT_ENV] == "ipc"


def test_linux_never_allows_unix_sockets(layout):
    # The seccomp filter is what keeps host sockets (docker.sock) out of reach.
    network = _settings(SMOKE, layout, platform="linux")["network"]
    assert not {"allowUnixSockets", "allowAllUnixSockets"} & set(network)


# -- filesystem -------------------------------------------------------------

def test_a_step_writes_only_its_run_its_cache_and_temp(layout):
    fs = _settings(SMOKE, layout)["filesystem"]
    assert sorted(fs["allowWrite"]) == sorted(
        [str(layout["tmp"]), str(layout["run_dir"]),
         str(layout["venv"] / es.CACHE_DIRNAME)])
    assert str(layout["venv"]) not in fs["allowWrite"]


def test_pip_writes_only_the_venv_and_temp(layout):
    fs = _settings(es.PIP_STEP, layout)["filesystem"]
    assert sorted(fs["allowWrite"]) == sorted(
        [str(layout["tmp"]), str(layout["venv"]),
         str(layout["venv"] / es.CACHE_DIRNAME)])
    assert str(layout["run_dir"]) not in fs["allowWrite"]


def test_earlier_evidence_is_read_only_except_the_steps_own_outputs(layout):
    denied = _settings(PROBES, layout)["filesystem"]["denyWrite"]
    pipeline = layout["run_dir"] / ".pipeline"
    assert str(layout["run_dir"] / ".git") in denied
    for name in ("driver_state.json", "stage_2a.complete", "judge_parts"):
        assert str(pipeline / name) in denied, name
    for name in ("probe_report.json", "claims_ledger.json"):
        assert str(pipeline / name) not in denied, name
    # The same report is evidence, not an output, for every other step.
    assert str(pipeline / "probe_report.json") in (
        _settings(SMOKE, layout)["filesystem"]["denyWrite"])


def test_secrets_and_other_runs_are_unreadable(layout):
    fs = _settings(SMOKE, layout)["filesystem"]
    repo, runs = layout["repo"], layout["repo"] / "r2c_runs"
    for path in (Path.home(), repo / ".env", repo / "input_papers",
                 runs / "other-run", runs / es.VENVS_DIRNAME / "other-run"):
        assert str(path) in fs["denyRead"], path
    assert str(layout["run_dir"]) not in fs["denyRead"]
    assert str(layout["venv"]) not in fs["denyRead"]
    assert str(layout["run_dir"]) in fs["allowRead"]


def test_no_deny_wraps_the_run(layout):
    # Regression: a deny on r2c_runs inside the allowed repo outranks the
    # allow on this run, and every step then failed to read its own files.
    denied = _settings(SMOKE, layout)["filesystem"]["denyRead"]
    runs = layout["repo"] / "r2c_runs"
    assert str(runs) not in denied
    assert str(runs / es.VENVS_DIRNAME) not in denied


# -- environment ------------------------------------------------------------

def test_environment_carries_no_keys_or_server_address(layout):
    env = _env(layout, {**SENSITIVE_ENV, "PATH": "/usr/bin", "LANG": "C"})
    assert not set(SENSITIVE_ENV) & set(env)
    assert env["LANG"] == "C"


def test_environment_points_home_and_temp_inside_the_step(layout):
    env = _env(layout, {"HOME": "/Users/me", "TMPDIR": "/var/x", "PATH": "/usr/bin"})
    assert env["HOME"] == str(layout["tmp"] / "home")
    assert env["TMPDIR"] == str(layout["tmp"])
    assert env["CLAUDE_CODE_TMPDIR"] == str(layout["tmp"])


def test_caches_outlive_the_step_but_home_does_not(layout):
    env = _env(layout, {"XDG_CACHE_HOME": "/Users/me/.cache", "MPLCONFIGDIR": "/x"})
    cache = layout["venv"] / es.CACHE_DIRNAME
    assert env["XDG_CACHE_HOME"] == str(cache)
    assert Path(env["MPLCONFIGDIR"]).parent == cache
    assert not Path(env["HOME"]).is_relative_to(cache)


def test_run_cache_is_created_and_kept(layout):
    cache = es.ensure_run_cache(layout["venv"])
    (cache / "fontlist.json").write_text("{}")
    assert es.ensure_run_cache(layout["venv"]) == cache
    assert (cache / "fontlist.json").exists()


@pytest.mark.parametrize("plant", [
    lambda cache, target: cache.symlink_to(target, target_is_directory=True),
    lambda cache, target: cache.write_text("not a directory"),
])
def test_a_planted_cache_path_never_becomes_a_writable_root(layout, tmp_path, plant):
    target = tmp_path / "somewhere-else"
    target.mkdir()
    (target / "keep").write_text("x")
    cache = layout["venv"] / es.CACHE_DIRNAME
    plant(cache, target)
    es.ensure_run_cache(layout["venv"])
    assert cache.is_dir() and not cache.is_symlink()
    assert cache.resolve() == layout["venv"].resolve() / es.CACHE_DIRNAME
    assert (target / "keep").exists()


def test_offline_flag_follows_the_network(layout):
    assert _env(layout, {})["R2C_OFFLINE"] == "1"
    assert "R2C_OFFLINE" not in _env(layout, {}, domains=["pypi.org"])


def test_proxy_settings_reach_only_online_steps(layout):
    environ = {"HTTPS_PROXY": "http://proxy:1"}
    assert "HTTPS_PROXY" not in _env(layout, environ)
    assert "HTTPS_PROXY" in _env(layout, environ, domains=["pypi.org"])


def test_path_drops_repo_run_and_relative_entries(layout):
    repo = layout["repo"]
    environ = {"PATH": f"{repo}/bin:.:relative:{repo}/r2c_runs/x:/usr/bin"}
    entries = _env(layout, environ)["PATH"].split(os.pathsep)
    assert entries == [str(layout["venv"] / "bin"), "/usr/bin"]


# -- mode and step lookup ---------------------------------------------------

@pytest.mark.parametrize("environ, expected", [
    ({}, "srt"),
    ({es.MODE_ENV: "off"}, "off"),
    ({es.MODE_ENV: "OFF"}, "off"),
    ({es.MODE_ENV: "srt"}, "srt"),
    ({es.MODE_ENV: "typo"}, "srt"),
    ({"R2C_SANDBOX": "1"}, "off"),
    ({"R2C_SANDBOX": "1", es.MODE_ENV: "srt"}, "srt"),
])
def test_mode(environ, expected):
    assert es.mode(environ) == expected


def test_step_lookup():
    assert es.step_for_args(["-m", "pip", "install", "-r", "x"]) is es.PIP_STEP
    assert es.step_for_args(["scripts/run_probes.py", "--run-dir", "x"]) is PROBES
    assert es.step_for_args(["/abs/scripts/run_probes.py"]) is PROBES
    assert es.step_for_args(["scripts/scaffold_package.py", "--spec", "x"]) is None
    assert es.step_for_args(["-c", "print(1)"]) is None


def test_run_dir_comes_from_the_steps_own_arguments(tmp_path):
    assert es.run_dir_from_args(
        ["x.py", "--run-dir", str(tmp_path)]) == tmp_path.resolve()
    assert es.run_dir_from_args(
        ["-m", "pip", "install", "-r", str(tmp_path / "requirements.txt")]
    ) == tmp_path.resolve()
    with pytest.raises(es.SandboxError):
        es.run_dir_from_args(["x.py"])


def _uncovered(scripts: list[Path]) -> set[str]:
    return {
        path.name for path in scripts
        if _EXECUTES.search(path.read_text(encoding="utf-8"))
        and path.name not in es.SCRIPT_STEPS
        and path.name not in _NOT_A_STEP
        and path.name != "exec_sandbox.py"
    }


def test_every_script_that_runs_generated_code_is_a_sandboxed_step():
    assert _uncovered(sorted(SCRIPTS_DIR.glob("*.py"))) == set()


def test_the_coverage_check_catches_a_new_script(tmp_path):
    new = tmp_path / "new_validator.py"
    new.write_text("client = NotebookClient(nb)\n")
    assert _uncovered([new]) == {"new_validator.py"}


def test_listed_scripts_exist():
    for name in es.SCRIPT_STEPS:
        assert (SCRIPTS_DIR / name).is_file(), name


# -- driver routing ---------------------------------------------------------

@pytest.fixture
def sandbox_on(monkeypatch):
    monkeypatch.setenv(es.MODE_ENV, "srt")


def _record_runs(monkeypatch) -> list:
    calls: list = []

    def fake_run(step, args, **kwargs):
        calls.append((step, args, kwargs))
        return subprocess.CompletedProcess(args, 0, "out", "")

    monkeypatch.setattr(es, "run", fake_run)
    return calls


@pytest.mark.parametrize("args, step", [
    (["-m", "pip", "install", "-r", "/x/run/requirements.txt"], es.PIP_STEP),
    (["scripts/smoke_run_notebook.py", "--run-dir", "/x/run"], SMOKE),
    (["scripts/validate_package_imports.py", "--spec", "s", "--run-dir", "/x/run"],
     es.SCRIPT_STEPS["validate_package_imports.py"]),
])
def test_run_script_sends_generated_code_through_the_sandbox(
        monkeypatch, sandbox_on, args, step):
    calls = _record_runs(monkeypatch)

    def no_bare_run(*a, **k):
        raise AssertionError("generated code ran outside the sandbox")

    monkeypatch.setattr(run_pipeline.subprocess, "run", no_bare_run)
    proc = run_pipeline.run_script("stage_test", args, timeout=30)
    assert proc.returncode == 0
    assert calls[0][0] is step
    assert calls[0][1] == args
    assert calls[0][2]["timeout"] == 30
    assert calls[0][2]["run_dir"] == Path("/x/run").resolve()


def test_run_script_leaves_trusted_scripts_alone(monkeypatch, sandbox_on):
    calls = _record_runs(monkeypatch)
    proc = run_pipeline.run_script("stage_test", ["-c", "print('ok')"], timeout=30)
    assert proc.returncode == 0 and "ok" in proc.stdout
    assert calls == []


def test_a_sandbox_failure_fails_the_step_instead_of_running_bare(
        monkeypatch, sandbox_on):
    def broken(*a, **k):
        raise es.SandboxError("srt is not installed")

    def no_bare_run(*a, **k):
        raise AssertionError("fell back to running without the sandbox")

    monkeypatch.setattr(es, "run", broken)
    monkeypatch.setattr(run_pipeline.subprocess, "run", no_bare_run)
    proc = run_pipeline.run_script(
        "stage_test", ["scripts/run_probes.py", "--run-dir", "/x/run"], timeout=30)
    assert proc.returncode == es.SANDBOX_FAILURE_EXIT
    assert "srt is not installed" in proc.stderr


def test_off_mode_runs_without_the_sandbox(monkeypatch):
    monkeypatch.setenv(es.MODE_ENV, "off")
    calls = _record_runs(monkeypatch)
    monkeypatch.setattr(
        run_pipeline.subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "bare", ""))
    proc = run_pipeline.run_script(
        "stage_test", ["scripts/run_probes.py", "--run-dir", "/x/run"], timeout=30)
    assert proc.stdout == "bare" and calls == []


def test_element_tests_use_the_sandbox(monkeypatch, sandbox_on, tmp_path):
    import element_tests

    calls = _record_runs(monkeypatch)
    module = tmp_path / "test_x.py"
    module.write_text("def test_x(): pass\n")
    result = element_tests.run_test_module(tmp_path, module, venv_run_dir=tmp_path)
    assert result["status"] == "passed"
    assert calls[0][0] is es.ELEMENT_TEST_STEP
    assert calls[0][2]["venv_run_dir"] == tmp_path


# -- preflight and process handling, against a stub srt ---------------------

STUB_SRT = """#!/bin/bash
# Stands in for srt: drops `--settings FILE --` and runs the command.
echo "$2" > "$STUB_SETTINGS_RECORD"
shift 3
exec "$@"
"""


@pytest.fixture
def stub_srt(tmp_path, monkeypatch) -> Path:
    stub = tmp_path / "srt"
    stub.write_text(STUB_SRT)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    record = tmp_path / "settings-path"
    monkeypatch.setenv(es.SRT_BIN_ENV, str(stub))
    monkeypatch.setenv("STUB_SETTINGS_RECORD", str(record))
    monkeypatch.setattr(es, "_PASSED_ENV", (*es._PASSED_ENV, "STUB_SETTINGS_RECORD"))
    return record


def test_preflight_names_the_install_command(monkeypatch):
    monkeypatch.setenv(es.MODE_ENV, "srt")
    monkeypatch.setattr(es, "find_srt", lambda environ=None: None)
    problem = es.preflight()
    assert problem is not None and es.SRT_INSTALL_HINT in problem


def test_preflight_passes_with_a_working_srt(monkeypatch, stub_srt):
    monkeypatch.setenv(es.MODE_ENV, "srt")
    assert es.preflight() is None


def test_preflight_is_silent_when_off(monkeypatch):
    monkeypatch.setenv(es.MODE_ENV, "off")
    monkeypatch.setattr(es, "find_srt", lambda environ=None: None)
    assert es.preflight() is None


def test_run_refuses_without_srt(monkeypatch, layout):
    monkeypatch.setattr(es, "find_srt", lambda environ=None: None)
    with pytest.raises(es.SandboxError, match="not installed"):
        es.run(SMOKE, ["-c", "pass"], run_dir=layout["run_dir"],
               repo_root=layout["repo"], cwd=layout["run_dir"])


def _run_stub(layout, code, **kwargs):
    return es.run(SMOKE, ["-c", code], run_dir=layout["run_dir"],
                  repo_root=layout["repo"], cwd=layout["run_dir"], **kwargs)


def test_run_reports_output_and_cleans_up(layout, stub_srt):
    code = "import os,sys; print(os.environ['TMPDIR']); sys.exit(3)"
    proc = _run_stub(layout, code)
    assert proc.returncode == 3
    step_tmp = Path(proc.stdout.strip())
    assert step_tmp.name.startswith("r2c-")
    assert not step_tmp.exists()
    assert not Path(stub_srt.read_text().strip()).exists()


def test_run_venv_chains_to_the_driver_packages(layout, stub_srt):
    proc = _run_stub(layout, "import sys,pytest; print(sys.prefix)")
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip()).name == RUN


def test_timeout_kills_the_whole_process_tree(layout, stub_srt, tmp_path):
    pid_file = tmp_path / "child.pid"
    code = (
        "import subprocess,time\n"
        f"p = subprocess.Popen(['/bin/sleep','300'])\n"
        f"open({str(pid_file)!r},'w').write(str(p.pid))\n"
        "time.sleep(300)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _run_stub(layout, code, timeout=3)
    child = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail("a child of the timed-out step survived")


# Steps started outside the driver: the run companion's gates and people.


@pytest.fixture
def by_hand(tmp_path, monkeypatch) -> tuple[Path, list]:
    run_dir = tmp_path / "r2c_runs" / RUN
    run_dir.mkdir(parents=True)
    calls: list = []

    def fake_run(step, args, **kwargs):
        calls.append((step, args, kwargs))
        return subprocess.CompletedProcess(args, 7, "out\n", "err\n")

    monkeypatch.setattr(es, "run", fake_run)
    monkeypatch.setenv(es.MODE_ENV, "srt")
    monkeypatch.delenv(es.INSIDE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    return run_dir, calls


def _started_by_hand(monkeypatch, script: Path, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", [str(script), *args])
    es.relaunch_in_sandbox(str(script))


def test_step_started_by_hand_enters_its_sandbox(by_hand, monkeypatch, capsys):
    run_dir, calls = by_hand
    script = SCRIPTS_DIR / "smoke_run_notebook.py"
    with pytest.raises(SystemExit) as stopped:
        _started_by_hand(monkeypatch, script, "--run-dir", f"r2c_runs/{RUN}",
                         "--timeout", "180")
    assert stopped.value.code == 7
    step, args, kwargs = calls[0]
    assert step is SMOKE
    assert args == [str(script), "--run-dir", str(run_dir.resolve()),
                    "--timeout", "180"]
    assert kwargs["run_dir"] == run_dir.resolve()
    assert kwargs["cwd"] == kwargs["repo_root"] == SCRIPTS_DIR.parent
    assert capsys.readouterr() == ("out\n", "err\n")


@pytest.mark.parametrize("env, args", [
    ({es.MODE_ENV: "off"}, ["--run-dir", "x"]),
    ({es.INSIDE_ENV: "1"}, ["--run-dir", "x"]),
    ({}, ["--help"]),
])
def test_step_stays_in_this_process_when_it_should(by_hand, monkeypatch, env, args):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _started_by_hand(monkeypatch, SCRIPTS_DIR / "smoke_run_notebook.py", *args)
    assert by_hand[1] == []


def test_inside_marker_stops_a_second_relaunch(layout):
    assert _env(layout, {})[es.INSIDE_ENV] == "1"


def test_by_hand_pip_is_the_pip_step(by_hand):
    run_dir, calls = by_hand
    (run_dir / "requirements.txt").write_text("numpy\n")
    rc = es.run_by_hand(["-m", "pip", "install", "-r",
                         f"r2c_runs/{RUN}/requirements.txt"])
    assert rc == 7
    assert calls[0][0] is es.PIP_STEP
    assert calls[0][2]["run_dir"] == run_dir.resolve()


def test_by_hand_refuses_a_missing_run_without_creating_anything(by_hand, capsys):
    run_dir, calls = by_hand
    args = [str(SCRIPTS_DIR / "smoke_run_notebook.py"), "--run-dir",
            str(run_dir.parent / "typo")]
    assert es.run_by_hand(args) == es.SANDBOX_FAILURE_EXIT
    assert calls == []
    assert "not found" in capsys.readouterr().err
    assert not (run_dir.parent / es.VENVS_DIRNAME).exists()


def test_by_hand_fails_closed(by_hand, monkeypatch, capsys):
    run_dir, _calls = by_hand

    def broken(*args, **kwargs):
        raise es.SandboxError("srt is not installed")

    monkeypatch.setattr(es, "run", broken)
    bare = []
    monkeypatch.setattr(es.subprocess, "run", lambda *a, **k: bare.append(a))
    args = [str(SCRIPTS_DIR / "smoke_run_notebook.py"), "--run-dir", str(run_dir)]
    assert es.run_by_hand(args) == es.SANDBOX_FAILURE_EXIT
    assert bare == []
    assert "srt is not installed" in capsys.readouterr().err


def test_by_hand_rejects_anything_that_is_not_a_step(by_hand, capsys):
    assert es.run_by_hand(["-c", "print(1)"]) == 2
    assert by_hand[1] == []
    assert "not a sandboxed step" in capsys.readouterr().err


def test_by_hand_runs_bare_only_when_the_sandbox_is_off(by_hand, monkeypatch):
    run_dir, calls = by_hand
    monkeypatch.setenv(es.MODE_ENV, "off")
    bare = []

    def fake_bare(argv, **kwargs):
        bare.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(es.subprocess, "run", fake_bare)
    script = str(SCRIPTS_DIR / "smoke_run_notebook.py")
    assert es.run_by_hand([script, "--run-dir", str(run_dir)]) == 0
    assert calls == []
    assert bare == [[sys.executable, script, "--run-dir", str(run_dir.resolve())]]


def test_a_python_mismatch_is_reported_as_a_sandbox_problem(tmp_path):
    run_dir = tmp_path / "r2c_runs" / RUN
    run_dir.mkdir(parents=True)
    bin_dir = es.run_venv_dir(run_dir) / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text("")
    with pytest.raises(es.SandboxError, match="different Python"):
        es.ensure_run_venv(run_dir)


def test_every_companion_gate_enters_the_sandbox():
    companion = SCRIPTS_DIR.parent / ".opencode" / "agents" / "r2c-companion.md"
    named = set(re.findall(r"python scripts/(\w+\.py)", companion.read_text()))
    gates = named & set(es.SCRIPT_STEPS)
    assert {"validate_package_imports.py", "smoke_run_notebook.py"} <= gates
    for gate in gates:
        assert "relaunch_in_sandbox(__file__)" in (SCRIPTS_DIR / gate).read_text(), gate
