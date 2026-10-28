#!/usr/bin/env bash
# setup_env.sh [check_env.py options]
# Creates the uv virtual environment for one dependency group and runs the
# environment check. Use GROUP=train (default on x86_64) for training and
# GROUP=deploy (default on arm64) for running the exported model.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

info() { printf '\n>> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

info "checking prerequisites"
if [ -r /etc/nv_tegra_release ]; then
    printf '  board      : %s\n' "$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
    printf '  release    : %s\n' "$(head -1 /etc/nv_tegra_release)"
    printf '  memory     : %s\n' "$(free -g | awk '/^Mem:/ {print $2 " GiB, unified with the GPU"}')"
    printf '  note       : Tegra has no nvidia-smi and no GPU wheel, paddle runs on CPU\n'
elif command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
else
    printf 'error: nvidia-smi not found, the NVIDIA driver is missing or not loaded\n' >&2
    printf '       inspect the hardware with: bash tools/check_gpu.sh\n' >&2
    exit 1
fi
command -v uv >/dev/null || die "uv not found, install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"

[ -f vendors/paddet/ppdet/__init__.py ] || {
    info "initializing the paddet submodule"
    git submodule update --init --recursive
}

# /mnt is the Windows filesystem, far slower than ext4 for the thousands of
# small files a virtual environment contains. Keep the venv inside Linux when
# the repository itself lives on /mnt.
if [ -z "${UV_PROJECT_ENVIRONMENT:-}" ] && [[ "$ROOT" == /mnt/* ]]; then
    export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/$(basename "$ROOT")"
    info "virtual environment: $UV_PROJECT_ENVIRONMENT"
fi

info "selecting the dependency group"
if [ -n "${GROUP:-}" ]; then
    printf '  group      : %s (from $GROUP)\n' "$GROUP"
elif [ "$(uname -m)" = "x86_64" ]; then
    GROUP=train
    printf '  group      : train (x86_64 host, override with GROUP=deploy)\n'
else
    GROUP=deploy
    printf '  group      : deploy (non x86_64 host, override with GROUP=train)\n'
fi
[ "$GROUP" = "train" ] || [ "$GROUP" = "deploy" ] || die "GROUP must be train or deploy"

info "creating the environment with python $PYTHON_VERSION"
# uv downloads a managed CPython when the requested version is not installed,
# so the result does not depend on the system python. The train group pulls
# paddet from vendors/paddet in editable mode, see tool.uv.sources.
uv sync --python "$PYTHON_VERSION" --group "$GROUP"

info "running the environment check"
if [ "$(uname -m)" != "x86_64" ]; then
    set -- "$@" --cpu
fi
uv run --no-sync tools/check_env.py "$@"
