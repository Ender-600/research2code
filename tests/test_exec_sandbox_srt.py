"""The per-step sandbox against the real srt: what generated code cannot do.

Skipped unless srt is installed on macOS. The Linux backend (bubblewrap) needs
user namespaces that most CI runners do not grant.
"""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import exec_sandbox as es

REPO_ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or es.find_srt() is None,
    reason="needs srt on macOS",
)

SMOKE = es.SCRIPT_STEPS["smoke_run_notebook.py"]

PAYLOAD = r"""
import json, os, socket, sys, urllib.request
from pathlib import Path

results = {}

def attempt(name, action):
    try:
        action()
        results[name] = "reached"
    except Exception as exc:
        results[name] = type(exc).__name__

def write(path):
    Path(path).write_text("x")

run = Path(os.environ["T_RUN"])
repo = Path(os.environ["T_REPO"])
attempt("read_home", lambda: os.listdir(os.environ["T_REAL_HOME"]))
attempt("read_env_file", lambda: (repo / ".env").read_text())
attempt("read_other_run", lambda: (run.parent / "other" / "secret").read_text())
attempt("write_repo", lambda: write(repo / "planted.py"))
attempt("write_other_run", lambda: write(run.parent / "other" / "x"))
attempt("write_venv", lambda: write(Path(os.environ["VIRTUAL_ENV"]) / "x"))
attempt("forge_evidence", lambda: write(run / ".pipeline" / "driver_state.json"))
attempt("delete_evidence", lambda: (run / ".pipeline" / "stage.complete").unlink())
attempt("https", lambda: urllib.request.urlopen("https://example.com", timeout=6))
attempt("raw_tcp", lambda: socket.create_connection(("1.1.1.1", 443), timeout=4))
attempt("localhost", lambda: socket.create_connection(
    ("127.0.0.1", int(os.environ["T_PORT"])), timeout=4))
attempt("write_run", lambda: write(run / "output.txt"))
attempt("write_tmp", lambda: write(Path(os.environ["TMPDIR"]) / "x"))
results["keys_in_env"] = sorted(k for k in os.environ if "API_KEY" in k)
print(json.dumps(results))
"""


@pytest.fixture
def run_tree(tmp_path: Path) -> Path:
    # Resolved: macOS tmp_path sits behind the /var -> /private/var symlink.
    runs = tmp_path.resolve() / "r2c_runs"
    run_dir = runs / "this"
    (run_dir / ".pipeline").mkdir(parents=True)
    (run_dir / ".pipeline" / "driver_state.json").write_text('{"real": true}')
    (run_dir / ".pipeline" / "stage.complete").write_text("")
    (runs / "other").mkdir()
    (runs / "other" / "secret").write_text("other run's data")
    return run_dir


def _attack(run_tree: Path, port: int, repo: Path) -> dict:
    (run_tree / "payload.py").write_text(PAYLOAD)
    proc = es.run(
        SMOKE, [str(run_tree / "payload.py")], run_dir=run_tree, repo_root=repo,
        cwd=run_tree, timeout=90,
        extra_env={"T_RUN": str(run_tree), "T_REPO": str(repo), "T_PORT": str(port),
                   "T_REAL_HOME": str(Path.home())},
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture
def attack_results(run_tree: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    repo = tmp_path.resolve() / "repo"
    repo.mkdir()
    (repo / ".env").write_text("ANTHROPIC_API_KEY=must-not-leak\n")

    class Server(http.server.BaseHTTPRequestHandler):
        hits: list[str] = []

        def do_GET(self):
            Server.hits.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield _attack(run_tree, server.server_address[1], repo), Server.hits
    finally:
        server.shutdown()


@pytest.mark.parametrize("attack", [
    "read_home", "read_env_file", "read_other_run", "write_repo",
    "write_other_run", "write_venv", "forge_evidence", "delete_evidence",
    "https", "raw_tcp", "localhost",
])
def test_generated_code_is_stopped(attack_results, attack):
    results, _hits = attack_results
    assert results[attack] != "reached", attack


def test_generated_code_can_still_do_its_job(attack_results):
    results, _hits = attack_results
    assert results["write_run"] == "reached"
    assert results["write_tmp"] == "reached"


def test_no_api_key_and_no_request_to_the_local_server(attack_results):
    results, hits = attack_results
    assert results["keys_in_env"] == []
    assert hits == []


def test_evidence_survives_the_attack(attack_results, run_tree):
    state = run_tree / ".pipeline" / "driver_state.json"
    assert json.loads(state.read_text()) == {"real": True}
    assert (run_tree / ".pipeline" / "stage.complete").exists()


def test_a_notebook_runs_in_the_sandbox(run_tree):
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("import os\nprint(os.getpid())"),
        nbformat.v4.new_code_cell("x = 1 + 1\nassert x == 2"),
    ])
    nbformat.write(notebook, run_tree / "notebook.ipynb")
    args = [str(REPO_ROOT / "scripts" / "smoke_run_notebook.py"),
            "--run-dir", str(run_tree)]
    proc = es.run(SMOKE, args, run_dir=run_tree, repo_root=REPO_ROOT,
                  cwd=REPO_ROOT, timeout=180)
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-800:]
    executed = nbformat.read(run_tree / "notebook.ipynb", as_version=4)
    assert executed.cells[0].outputs, "the kernel never ran the cell"


def test_element_tests_run_in_the_sandbox(run_tree, monkeypatch):
    # Regression: pytest collected from the repo root and died on .env.
    import element_tests

    monkeypatch.setenv(es.MODE_ENV, "srt")
    package = run_tree / "method"
    (package / "tests").mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 2\n")
    module = package / "tests" / "test_value.py"
    module.write_text("import method\n\ndef test_value():\n    assert method.VALUE == 2\n")
    result = element_tests.run_test_module(run_tree, module)
    assert result["status"] == "passed", result["detail"][-800:]


def test_pip_step_cannot_touch_the_run(run_tree, tmp_path):
    repo = tmp_path.resolve() / "repo"
    repo.mkdir()
    code = (
        "import os\n"
        "from pathlib import Path\n"
        "run, venv = Path(os.environ['T_RUN']), Path(os.environ['VIRTUAL_ENV'])\n"
        "(venv / 'installed').write_text('x')\n"
        "try:\n"
        "    (run / 'planted').write_text('x')\n"
        "except PermissionError:\n"
        "    print('blocked')\n"
    )
    proc = es.run(es.PIP_STEP, ["-c", code], run_dir=run_tree, repo_root=repo,
                  cwd=repo, timeout=60, extra_env={"T_RUN": str(run_tree)})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "blocked"
    assert not (run_tree / "planted").exists()
    assert (es.run_venv_dir(run_tree) / "installed").exists()


def test_gate_started_by_hand_uses_the_runs_own_packages(run_tree):
    # Regression: the companion's gates ran bare and could not import packages
    # that the sandboxed pip step had put in the run's venv.
    package = run_tree / "method"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import only_in_run_venv\n__all__ = ['VALUE']\nVALUE = 2\n")
    (run_tree / "requirements.txt").write_text("")
    spec = run_tree / ".pipeline" / "method_spec.json"
    spec.write_text("{}")
    es.ensure_run_venv(run_tree)
    site_dir = es.run_site_packages(es.run_venv_dir(run_tree))
    (site_dir / "only_in_run_venv.py").write_text("")
    with pytest.raises(ImportError):
        __import__("only_in_run_venv")
    env = {**os.environ, es.MODE_ENV: "srt"}
    env.pop(es.INSIDE_ENV, None)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "validate_package_imports.py"),
         "--spec", str(spec), "--run-dir", str(run_tree)],
        capture_output=True, text=True, timeout=120, env=env, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-800:]


def _step(run_tree, code, step=SMOKE, **extra_env):
    proc = es.run(step, ["-c", code], run_dir=run_tree, repo_root=run_tree.parent,
                  cwd=run_tree, timeout=90, extra_env=extra_env)
    assert proc.returncode == 0, proc.stderr[-800:]
    return proc.stdout.strip()


def test_cache_outlives_the_step_and_home_does_not(run_tree):
    write = (
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['XDG_CACHE_HOME'], 'built-once').write_text('x')\n"
        "Path(os.environ['HOME'], 'left-behind').write_text('x')\n"
        "print(os.environ['HOME'])\n"
    )
    first_home = _step(run_tree, write)
    read = (
        "import os\nfrom pathlib import Path\n"
        "print(Path(os.environ['XDG_CACHE_HOME'], 'built-once').exists(),"
        " Path(os.environ['HOME'], 'left-behind').exists())\n"
    )
    assert _step(run_tree, read) == "True False"
    assert not Path(first_home).exists()


def test_matplotlib_builds_its_font_list_once_per_run(run_tree):
    pytest.importorskip("matplotlib")
    code = ("import matplotlib\nmatplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\nplt.figure()\n"
            "print(matplotlib.get_cachedir())\n")
    cachedir = Path(_step(run_tree, code))
    assert cachedir.parent == es.run_venv_dir(run_tree) / es.CACHE_DIRNAME
    assert list(cachedir.glob("fontlist-*.json")), "no font list was cached"


def test_a_cache_symlink_planted_by_pip_gives_no_write_outside(run_tree, tmp_path):
    outside = tmp_path.resolve() / "outside"
    outside.mkdir()
    plant = (
        "import os, shutil\nfrom pathlib import Path\n"
        "cache = Path(os.environ['XDG_CACHE_HOME'])\n"
        "shutil.rmtree(cache)\n"
        "cache.symlink_to(os.environ['T_OUTSIDE'], target_is_directory=True)\n"
    )
    _step(run_tree, plant, step=es.PIP_STEP, T_OUTSIDE=str(outside))
    assert (es.run_venv_dir(run_tree) / es.CACHE_DIRNAME).is_symlink()
    attempt = (
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['XDG_CACHE_HOME'], 'via-cache').write_text('x')\n"
        "try:\n"
        "    Path(os.environ['T_OUTSIDE'], 'direct').write_text('x')\n"
        "    print('reached')\n"
        "except PermissionError:\n"
        "    print('blocked')\n"
    )
    assert _step(run_tree, attempt, T_OUTSIDE=str(outside)) == "blocked"
    assert list(outside.iterdir()) == []

