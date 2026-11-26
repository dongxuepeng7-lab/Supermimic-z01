#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec uv run --locked python -u tools/ral_nine_hour.py judge "$@"
