#!/bin/bash
# Run R2C inside the sandbox container (see sandbox/README.md).
#
# Usage:
#   sandbox/r2c-sandbox.sh build          build the image (once, and after
#                                         requirements.lock changes)
#   sandbox/r2c-sandbox.sh [start]        opencode server + TUI, like ./r2c-start.sh
#   sandbox/r2c-sandbox.sh shell          interactive bash inside the sandbox
#   sandbox/r2c-sandbox.sh run CMD...     run one command inside the sandbox
#
# Tunables (environment):
#   R2C_SANDBOX_IMAGE    image tag              (default r2c-sandbox:local)
#   R2C_SANDBOX_MEMORY   memory limit           (default 8g)
#   R2C_SANDBOX_CPUS     CPU limit              (default 4)
#   R2C_SANDBOX_PIDS     process limit          (default 1024)
#   R2C_SANDBOX_NETWORK  docker network         (default bridge; the agents
#                                                need the model API)
#   R2C_WATCH_PORT       host port for the live viewer (default 8765)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${R2C_SANDBOX_IMAGE:-r2c-sandbox:local}"
WATCH_PORT="${R2C_WATCH_PORT:-8765}"

# Repo paths the sandbox may read but never write. Each one is something the
# host later executes or trusts, so a write from inside would persist past
# the container: git hooks, the driver and templates, and .env, which
# r2c-start.sh sources as shell code on the next host launch. .opencode is
# handled separately below. r2c_runs/, input_papers/ and .opencode/scratch/
# stay writable; everything else at the repo root is writable too, so the
# driver's stray-write quarantine keeps working.
READ_ONLY_PATHS=(
  .git .github .env .env.example
  scripts paradigms schemas docs tests sandbox
  r2c-start.sh r2c-watch requirements.txt requirements.lock pyproject.toml
  README.md CONTRIBUTING.md LICENSE.pdf
)

die() { echo "r2c-sandbox: $*" >&2; exit 1; }

command -v docker >/dev/null || die "docker is not installed"

build() {
  DOCKER_BUILDKIT=1 docker build \
    -f "${REPO}/sandbox/Dockerfile" \
    --build-arg "R2C_UID=$(id -u)" \
    --build-arg "R2C_GID=$(id -g)" \
    -t "${IMAGE}" "$@" "${REPO}"
}

run_container() {
  local tty_flags=("$@")
  shift $#
  docker image inspect "${IMAGE}" >/dev/null 2>&1 \
    || die "image ${IMAGE} not found; run: sandbox/r2c-sandbox.sh build"

  # Writable mount points that sit inside read-only mounts must exist on the
  # host first, or Docker creates them as root-owned directories.
  mkdir -p "${REPO}/r2c_runs" "${REPO}/input_papers" "${REPO}/.opencode/scratch"
  # .env must exist to be mounted read-only. Otherwise code in the sandbox
  # could create it, and r2c-start.sh would run it as shell on the host.
  if [ ! -e "${REPO}/.env" ]; then
    (umask 077 && : > "${REPO}/.env")
  fi

  local mounts=(-v "${REPO}:/workspace")
  local p
  for p in "${READ_ONLY_PATHS[@]}"; do
    [ -e "${REPO}/${p}" ] && mounts+=(-v "${REPO}/${p}:/workspace/${p}:ro")
  done
  # .opencode: opencode writes its own state at startup (.gitignore,
  # package.json, a lockfile, node_modules with the plugin SDK). The host's
  # opencode would install and load whatever those list, so they live in a
  # container-only volume. The R2C configuration from the host (agents and
  # their permissions, commands, the plugin, skills, opencode.json) is laid
  # over it read-only; only scratch/ maps back to the host.
  mounts+=(-v r2c-sandbox-opencode-state:/workspace/.opencode)
  local entry name
  for entry in "${REPO}"/.opencode/* "${REPO}"/.opencode/.[!.]*; do
    [ -e "${entry}" ] || continue
    name="$(basename "${entry}")"
    case "${name}" in
      scratch|node_modules|package.json|package-lock.json|bun.lock|bun.lockb|.gitignore) ;;
      *) mounts+=(-v "${entry}:/workspace/.opencode/${name}:ro") ;;
    esac
  done
  mounts+=(
    -v "${REPO}/.opencode/scratch:/workspace/.opencode/scratch"
    # Container-only home: opencode auth (`opencode auth login`), sessions
    # and caches. The host home, ~/.ssh and cloud credentials are never mounted.
    -v r2c-sandbox-home:/home/r2c
  )

  docker run --rm "${tty_flags[@]}" \
    --name "r2c-sandbox-$$" \
    --hostname r2c-sandbox \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --pids-limit "${R2C_SANDBOX_PIDS:-1024}" \
    --memory "${R2C_SANDBOX_MEMORY:-8g}" \
    --cpus "${R2C_SANDBOX_CPUS:-4}" \
    --network "${R2C_SANDBOX_NETWORK:-bridge}" \
    -p "127.0.0.1:${WATCH_PORT}:8765" \
    -e R2C_WATCH_PORT=8765 \
    "${mounts[@]}" \
    "${IMAGE}" "${RUN_ARGS[@]}"
}

cmd="${1:-start}"
[ $# -gt 0 ] && shift
case "${cmd}" in
  build) build "$@" ;;
  start) RUN_ARGS=(./r2c-start.sh "$@"); run_container -it ;;
  shell) RUN_ARGS=(bash); run_container -it ;;
  run)   [ $# -gt 0 ] || die "usage: r2c-sandbox.sh run CMD..."
         RUN_ARGS=("$@")
         if [ -t 0 ] && [ -t 1 ]; then run_container -it; else run_container -i; fi ;;
  -h|--help|help) sed -n '2,20p' "$0" ;;
  *) die "unknown command '${cmd}' (build | start | shell | run CMD...)" ;;
esac
