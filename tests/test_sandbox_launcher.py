"""sandbox/r2c-sandbox.sh builds the docker invocation the sandbox relies on.

Runs the launcher against a stub `docker` that records its arguments, in a
scratch copy of the repo layout, so no container runtime is needed. Checks
the properties sandbox/README.md promises: only the repo is mounted, the
paths the host later executes (.git hooks, .opencode, .env, the driver)
are read-only, privileges are dropped, and ports bind to host loopback.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "sandbox" / "r2c-sandbox.sh"

STUB = """#!/bin/bash
printf '%s\\n' "$@" > "$DOCKER_ARGS_FILE.$1"
exit 0
"""

MUST_BE_READ_ONLY = [
    ".git", ".opencode/agents", ".opencode/plugin", ".opencode/opencode.json",
    ".env", "scripts", "paradigms", "schemas",
    "docs", "tests", "sandbox", "r2c-start.sh", "requirements.lock",
]


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    repo = tmp_path / "repo"
    (repo / "sandbox").mkdir(parents=True)
    shutil.copy2(LAUNCHER, repo / "sandbox" / "r2c-sandbox.sh")
    for d in [".git", ".opencode/agents", ".opencode/plugin",
              ".opencode/node_modules", "scripts", "paradigms", "schemas",
              "docs", "tests", "r2c_runs"]:
        (repo / d).mkdir(parents=True, exist_ok=True)
    for f in ["r2c-start.sh", "requirements.lock", "requirements.txt",
              ".opencode/opencode.json", ".opencode/package.json"]:
        (repo / f).write_text("x\n")
    return repo


def _run(repo: Path, tmp_path: Path, *args: str) -> list[str]:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "docker"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    record = tmp_path / "docker-args"
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}",
               DOCKER_ARGS_FILE=str(record))
    subprocess.run(["bash", str(repo / "sandbox" / "r2c-sandbox.sh"), *args],
                   check=True, env=env, stdin=subprocess.DEVNULL,
                   capture_output=True, timeout=30)
    return (Path(f"{record}.run")).read_text().splitlines()


def _mounts(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]


def test_read_only_paths_and_writable_holes(fake_repo, tmp_path):
    argv = _run(fake_repo, tmp_path, "run", "true")
    mounts = _mounts(argv)
    for rel in MUST_BE_READ_ONLY:
        assert f"{fake_repo}/{rel}:/workspace/{rel}:ro" in mounts, rel
    assert f"{fake_repo}:/workspace" in mounts
    assert f"{fake_repo}/.opencode/scratch:/workspace/.opencode/scratch" in mounts
    assert "r2c-sandbox-home:/home/r2c" in mounts
    assert (fake_repo / ".opencode" / "scratch").is_dir()
    # opencode's own state lives in a container-only volume; host copies of
    # files the host opencode installs from are never mapped in.
    state = mounts.index("r2c-sandbox-opencode-state:/workspace/.opencode")
    assert state < mounts.index(f"{fake_repo}/.opencode/agents:/workspace/.opencode/agents:ro")
    for name in ("node_modules", "package.json"):
        assert not any(m.startswith(f"{fake_repo}/.opencode/{name}:") for m in mounts), name


def test_missing_env_file_is_created_so_it_can_be_read_only(fake_repo, tmp_path):
    # Regression: with no .env, the read-only mount was skipped and code in
    # the sandbox could create .env, which r2c-start.sh sources on the host.
    assert not (fake_repo / ".env").exists()
    argv = _run(fake_repo, tmp_path, "run", "true")
    assert (fake_repo / ".env").is_file()
    assert (fake_repo / ".env").stat().st_mode & 0o077 == 0
    assert f"{fake_repo}/.env:/workspace/.env:ro" in _mounts(argv)


def test_nothing_outside_the_repo_is_bind_mounted(fake_repo, tmp_path):
    argv = _run(fake_repo, tmp_path, "run", "true")
    home = str(Path.home())
    for spec in _mounts(argv):
        source = spec.split(":", 1)[0]
        if source.startswith("/"):
            assert source == str(fake_repo) or source.startswith(f"{fake_repo}/"), spec
        assert "docker.sock" not in spec
        assert not (source.startswith(home) and not source.startswith(str(fake_repo)))


def test_privileges_resources_and_ports(fake_repo, tmp_path):
    argv = _run(fake_repo, tmp_path, "run", "true")
    joined = " ".join(argv)
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    for flag in ("--memory", "--cpus", "--pids-limit"):
        assert flag in argv
    assert "--privileged" not in argv
    published = [argv[i + 1] for i, a in enumerate(argv) if a == "-p"]
    assert published and all(p.startswith("127.0.0.1:") for p in published)
    assert argv[-1] == "true"


def test_build_context_excludes_secrets():
    ignore = (REPO_ROOT / "sandbox" / "Dockerfile.dockerignore").read_text()
    rules = [ln.strip() for ln in ignore.splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert rules == ["*", "!requirements.lock"]
    dockerfile = (REPO_ROOT / "sandbox" / "Dockerfile").read_text()
    assert "--require-hashes -r /tmp/requirements.lock" in dockerfile
    assert "\nUSER r2c\n" in dockerfile
