#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Use the installed environment directly: do not synchronize packages during cd01.
exec .venv/bin/python -u tools/ral_cd_nine_hour.py --machine 3090 "$@"
