#!/usr/bin/env bash
set -euo pipefail
PGMT_XY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGMT_XY_ENV=/home/lzp/miniconda3/envs/pmt
PGMT_XY_LAB="$PGMT_XY_ENV/lib/python3.11/site-packages/isaaclab/source"
export OMNI_KIT_ACCEPT_EULA=YES
export PMT_REPO_ROOT="$PGMT_XY_ROOT"
export PYTHONPATH="$PGMT_XY_LAB/isaaclab:$PGMT_XY_LAB/isaaclab_assets:$PGMT_XY_LAB/isaaclab_contrib:$PGMT_XY_LAB/isaaclab_tasks:$PGMT_XY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
cd "$PGMT_XY_ROOT"
exec "$PGMT_XY_ENV/bin/python" scripts/eval_pgmt_xy.py --headless --device cuda:0 "$@"
