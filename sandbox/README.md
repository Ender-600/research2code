# R2C sandbox

R2C reads untrusted input (paper PDFs and markdown), has LLM agents act on
it, and then executes the code they generate: stage 2.d runs `pip install`
and imports the package, the smoke gate executes the notebook, and probes
run on its outputs. Without a sandbox, all of that runs as you, with access
to your home directory, SSH keys and cloud credentials. A prompt injection
hidden in a paper, a hallucinated dependency, or plain buggy code can read
or delete your files.

There are two layers. The per-step sandbox is on by default. The container is
optional and is mainly for shared machines.

## Per-step sandbox (default)

`scripts/exec_sandbox.py` runs each step that touches generated code through
[`srt`](https://github.com/anthropic-experimental/sandbox-runtime), Anthropic's
Sandbox Runtime (Seatbelt on macOS, bubblewrap on Linux). The opencode server,
the model key and the driver stay outside it.

```bash
npm install -g @anthropic-ai/sandbox-runtime@0.0.81
```

Linux also needs `bubblewrap`, `socat` and `ripgrep`. The driver checks that srt
works before the first model call and refuses to start if it does not.

| Step | Can write | Network |
|---|---|---|
| `pip install` of the generated requirements | the run's venv | PyPI only |
| import checks, runtime dry-run, scale and budget checks, element tests, notebook smoke run, data-signal check, probe battery | the run directory | none |

Every step also gets:

- **No secrets.** The environment is built from an allowlist, so no provider
  keys and no opencode address. `HOME` and `TMPDIR` are a throwaway directory.
- **No access to your files.** Your home directory, the repo's `.env`,
  `input_papers/` and every other run are unreadable.
- **No access to the opencode server.** On macOS the notebook kernel uses Unix
  sockets, so localhost TCP stays closed. On Linux each step has its own
  loopback.
- **Earlier evidence is read-only.** Everything already in `.pipeline/`, apart
  from the files that step itself produces, cannot be changed.
- **Its own Python packages.** Each run has a venv in
  `r2c_runs/.sandbox_venvs/<run>/` that reuses the driver's packages through a
  `.pth` file. What a paper installs never reaches the driver's environment.
- **A cache that lasts the run.** `HOME` is deleted after every step, so
  `XDG_CACHE_HOME` and matplotlib's config directory point at
  `r2c_runs/.sandbox_venvs/<run>/cache/` instead. Otherwise every step that
  imports matplotlib spends about 20 seconds rebuilding its font list; now only
  the first one in a run does.

Settings (environment or `.env`):

| Variable | Effect |
|---|---|
| `R2C_EXEC_SANDBOX=off` | Run generated code without the sandbox. |
| `R2C_SRT_BIN` | Path to srt, if it is not on `PATH` or in npm's global bin directory. |
| `R2C_SANDBOX_SMOKE_DOMAINS` | Hosts the notebook smoke run may reach. Default: none. |
| `R2C_SANDBOX_PIP_DOMAINS` | Extra hosts for `pip install`. `PIP_INDEX_URL`'s host is added automatically. |
| `R2C_SANDBOX_EXTRA_READ` | Extra directories every step may read. |

**Datasets.** The smoke run has no network, so a template that downloads MNIST
or CIFAR-10 through torchvision (`vision_transformer`, `active_learning`,
`knowledge_distillation` and its detection variants) fails with a message
saying so. Put the data in `method/example_data/`, or allow the hosts:
`R2C_SANDBOX_SMOKE_DOMAINS=ossci-datasets.s3.amazonaws.com,www.cs.toronto.edu`.
pip reads no `pip.conf` from your home directory, so use `PIP_INDEX_URL` or
`R2C_WHEELHOUSE` for a mirror.

**Steps started by hand.** The run companion's post-edit gates
(`validate_package_imports.py`, `smoke_run_notebook.py`, `revalidate_run.py`)
enter the same sandbox on their own, so they see the packages in the run's
venv. They have no network, so after changing a run's `requirements.txt`,
install it first:

```bash
python scripts/exec_sandbox.py -m pip install -r r2c_runs/<run>/requirements.txt
```

**Limits of this layer.**

- Only exercised on macOS so far. The Linux backend (bubblewrap) has not been
  run end to end, though the driver still checks that srt starts.
- macOS steps run on CPU. The sandbox blocks Metal, so `torch.mps` is
  unavailable.
- srt does not limit memory or CPU. Steps still have their timeouts.
- A step can forge its own output. The probe battery shares a process with the
  code it checks, so malicious code could write a false `probe_report.json`
  during that step.
- `.pipeline/` files that do not exist yet when a step starts can be created.
- With `R2C_EXEC_SANDBOX=off`, those gates run in the driver's environment and
  do not see packages an earlier sandboxed run installed in its venv.
- Not covered: the delivered probe harness (`run_checks.py`), other commands the
  run companion runs through its bash tool, and PDF parsing.

## Whole-pipeline container (optional)

`sandbox/r2c-sandbox.sh` runs the whole pipeline in one container: the
opencode server and TUI, every agent, the Python driver, and every
execution step.

## Quick start

```bash
sandbox/r2c-sandbox.sh build      # once; again after requirements.lock changes
sandbox/r2c-sandbox.sh            # same as ./r2c-start.sh, inside the sandbox
```

The live viewer is published on `http://127.0.0.1:8765/` on the host.

**Model credentials.** The container never sees your home directory, so
host `opencode auth login` sessions are not visible inside it. Either put
the provider key in the repo's `.env` (mounted read-only, sourced by
`r2c-start.sh` as usual), or log in once inside the sandbox; the login
persists in the `r2c-sandbox-home` volume:

```bash
sandbox/r2c-sandbox.sh run opencode auth login
```

Other entry points:

```bash
sandbox/r2c-sandbox.sh shell                                 # bash inside
sandbox/r2c-sandbox.sh run python -m pytest -m "not probe_runtime" -q
```

## What it protects

| Boundary | How |
|---|---|
| Your files and credentials | Only the repo is mounted. No host home, `~/.ssh`, `~/.aws`, browser profiles or Docker socket. The container's home is a separate volume (`r2c-sandbox-home`). |
| Persistence back to the host | The paths the host later executes or trusts are mounted read-only: `.git` (hooks), `.opencode` (plugin, agent permissions, commands), `scripts/`, `paradigms/`, `schemas/`, `docs/`, `tests/`, `sandbox/`, `r2c-start.sh`, `requirements*`, and `.env`, which `r2c-start.sh` sources as shell code. opencode's own `node_modules` go to a volume, not the host tree. |
| Privilege | Runs as an unprivileged user with your UID, all Linux capabilities dropped, `no-new-privileges`. |
| Resource exhaustion | Memory, CPU and process-count limits (`R2C_SANDBOX_MEMORY`, `R2C_SANDBOX_CPUS`, `R2C_SANDBOX_PIDS`). |
| Leftovers between runs | `--rm`. Packages stage 2.d installs live in the container's venv and are discarded when the container exits. |
| Exposed ports | The opencode server listens on container loopback only. The read-only viewer is published on host loopback only. |

Writable: `r2c_runs/`, `input_papers/`, `.opencode/scratch/`, and the rest of
the repo root. The driver's stray-write quarantine
(`scripts/repo_scope_boundary.py`) needs the root writable.

## What it does not protect (yet)

- **The model API key.** Agents need it, so it is inside the container,
  and the network is open (`bridge`). Generated code runs in this same
  container: the per-step sandbox defaults to off here, because srt would
  need its weaker nested mode. For the key to stay out of reach of generated
  code, run on the host with the per-step sandbox instead.
- **Kernel isolation.** Docker shares the host kernel. On a shared lab
  server, use gVisor (`--runtime runsc`) and one container per user and job.
- **PDF parsing.** The image does not include Marker or llama.cpp, so it
  takes markdown inputs. PDF parsing is untrusted-input processing too and
  belongs inside the sandbox; adding Marker to the image is a follow-up.
- **GPU.** Docker on macOS cannot use the Apple GPU. CPU is enough for
  smoke-scale runs. On Linux with NVIDIA, add `--gpus all` (NVIDIA
  Container Toolkit).

## Image contents

`node:22-bookworm-slim` plus Debian's Python 3.11, git, tini, opencode
`1.18.34` (the version the R2C plugin targets), the driver dependencies from
`requirements.lock` installed with `--require-hashes`, and CPU torch
(`INSTALL_TORCH=0` to skip). Only `requirements.lock` enters the build
context (`Dockerfile.dockerignore`), so `.env` and run outputs never land
in an image layer.
