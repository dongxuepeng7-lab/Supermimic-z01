#!/bin/bash
# Prepare the G1 LAFAN1 inputs required by the Z01 builder and arm diagnostics.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-uv run python}
$PYTHON tools/urdf_to_mjcf.py
$PYTHON tools/data/prep_motions.py "$@"
