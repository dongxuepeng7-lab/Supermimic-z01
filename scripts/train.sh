#!/bin/bash
# Train Z01 v1.6 with PartyMimic. See README.md for the 16 GB GPU recipe.
# Usage: bash scripts/train.sh <log_dir> --robot z01 --motion_dir <npz clips> [more train.py flags, e.g. --seed 1]
# Python: $PYTHON, default "uv run python". Extra flags are forwarded to train.py.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR=${1:?usage: bash scripts/train.sh <log_dir> [train.py flags]}; shift
mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}  # the reference runs used 4 CPU threads
PYTHONPATH=src ${PYTHON:-uv run python} -u -m partymimic.train --log_dir "$LOG_DIR" "$@" 2>&1 | tee "$LOG_DIR/stdout.log"
