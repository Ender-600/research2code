# R2C sandbox

R2C reads untrusted input (paper PDFs and markdown), has LLM agents act on
it, and then executes the code they generate: stage 2.d runs `pip install`
and imports the package, the smoke gate executes the notebook, and probes
run on its outputs. Without a sandbox, all of that runs as you, with access
to your home directory, SSH keys and cloud credentials. A prompt injection
hidden in a paper, a hallucinated dependency, or plain buggy code can read
or delete your files.

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
  and the network is open (`bridge`). A successful prompt injection could
  still exfiltrate the key or anything under `r2c_runs/`. Next step: run
  the execution steps (install, import, smoke, probes) in a second,
  per-run container with no key and `--network none`.
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
