# R2C sandbox image.
#
# The whole pipeline runs in this container as an unprivileged user: the
# opencode server and TUI, every agent (including any tool calls a prompt
# injection in the paper text could trigger), the Python driver, and each
# step that executes generated code (stage 2.d pip install + import check,
# the smoke gate, probes). See sandbox/README.md for what this does and
# does not protect against.
#
# Build through sandbox/r2c-sandbox.sh build, which passes your UID/GID so
# files written into r2c_runs/ stay owned by you on Linux hosts.

FROM node:22-bookworm-slim

# opencode release the R2C plugin targets (.opencode/package.json pins
# @opencode-ai/plugin to the same version).
ARG OPENCODE_VERSION=1.18.34
# CPU torch, preinstalled so stage 2.d does not download it every run.
# Set INSTALL_TORCH=0 for a small image (method packages then install torch
# from PyPI at stage 2.d).
ARG INSTALL_TORCH=1
ARG TORCH_VERSION=2.14.1
ARG R2C_UID=1000
ARG R2C_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/r2c-venv \
    PATH=/opt/r2c-venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv git ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g --no-fund --no-audit "opencode-ai@${OPENCODE_VERSION}" \
    && npm cache clean --force

# Replace the base image's `node` user (UID 1000) with `r2c` at the host's
# UID/GID. -o tolerates IDs that already exist (macOS GID 20 is `dialout`).
RUN userdel -r node 2>/dev/null || true \
    && (getent group "${R2C_GID}" >/dev/null || groupadd -g "${R2C_GID}" r2c) \
    && useradd -o -m -u "${R2C_UID}" -g "${R2C_GID}" -s /bin/bash r2c \
    && git config --system --add safe.directory /workspace

# Python environment owned by r2c, so stage 2.d can install a generated
# package's requirements without root. Driver dependencies come from the
# hashed lock; --require-hashes refuses anything the lock does not vouch for.
COPY requirements.lock /tmp/requirements.lock
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock \
    && if [ "${INSTALL_TORCH}" = "1" ]; then \
         pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
             "torch==${TORCH_VERSION}"; \
       fi \
    && rm /tmp/requirements.lock \
    && chown -R "${R2C_UID}:${R2C_GID}" "${VIRTUAL_ENV}"

# Mount point for the .opencode state volume, owned by r2c: Docker seeds a
# new volume with the ownership of the image path it is mounted on.
RUN mkdir -p /workspace/.opencode \
    && chown -R "${R2C_UID}:${R2C_GID}" /workspace

USER r2c
WORKDIR /workspace

# The live viewer must listen on the container interface to be reachable
# through the published port; r2c-sandbox.sh publishes it on host loopback
# only. The opencode server itself stays on container loopback.
ENV R2C_WATCH_HOST=0.0.0.0 \
    R2C_SANDBOX=1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["./r2c-start.sh"]
