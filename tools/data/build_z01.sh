#!/bin/bash
# Retarget the G1 LAFAN source to Z01 v1.6, then constrain and screen trajectories.
# HHT=/path/to/human-humanoid-tools bash tools/data/build_z01.sh [HHT flags: --clip name, --limit N, ...]
# Z01_RAW and Z01_OUT select separate output directories for truncated smoke runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
HHT=${HHT:-$(dirname "$ROOT")/human-humanoid-tools}
HHT=$(realpath "$HHT")
HHT_PYTHON=${HHT_PYTHON:-$HHT/.venv/bin/python}
PYTHON=${PYTHON:-uv run python}
P=${P:-4}
for option in "$@"; do
  case "$option" in --clip|--clip=*|--limit|--limit=*) P=1 ;; esac
done
[[ "$P" =~ ^[1-9][0-9]*$ ]] || { echo "P must be a positive integer" >&2; exit 1; }
RAW=$(realpath -m "${Z01_RAW:-data/raw/z01_v2}")
OUT=$(realpath -m "${Z01_OUT:-data/motions/z01_lafan_v2}")
IK_OUT=$(realpath -m "${OUT}_ik")
CLEAN_OUT=$(realpath -m "${Z01_CLEAN_OUT:-${OUT}_clean}")
[ -x "$HHT_PYTHON" ] && [ -f "$HHT/scripts/batch_r2r_retarget.py" ] || {
  echo "Set HHT to an installed human-humanoid-tools checkout (README: commit ed8b350)." >&2; exit 1;
}
$PYTHON tools/data/z01_data.py prepare
# Never let --skip-existing silently reuse CSVs from a different calibration/model/run.
mkdir -p "$RAW"
$PYTHON - "$RAW" "$HHT" "$P" "$@" <<'PY'
import hashlib, json, pathlib, subprocess, sys
raw, hht = map(pathlib.Path, sys.argv[1:3])
paths = sorted(p for robot in ('g1', 'z01') for p in pathlib.Path('assets/hhtools_robots', robot).rglob('*'))
paths += sorted(pathlib.Path('data/motions/g1/lafan').glob('*.npz'))
files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
manifest = dict(files=files, hht_commit=subprocess.check_output(['git', '-C', str(hht), 'rev-parse', 'HEAD'], text=True).strip(),
                workers=sys.argv[3], flags=sys.argv[4:])
path = raw / 'retarget_manifest.json'
if path.exists():
    previous = json.loads(path.read_text())
    previous['files'] = {k: v for k, v in previous['files'].items() if not k.startswith('assets/hhtools_robots/rp1/')}
    if previous != manifest:
        raise SystemExit('Retarget inputs changed. Use a new Z01_RAW directory; old CSVs are preserved.')
elif list((raw / 'csv').glob('*.csv')):
    raise SystemExit('Existing CSVs have no provenance manifest. Use a new Z01_RAW directory.')
path.write_text(json.dumps(manifest, indent=2))
PY
if ! compgen -G 'data/motions/g1/lafan/*.npz' > /dev/null; then
  echo "Missing G1 LAFAN npz: run bash tools/download_data.sh, then bash tools/data/prepare_source.sh." >&2; exit 1
fi
$PYTHON tools/data/hht_csv.py data/motions/g1/lafan "$RAW/g1csv_p$P" --shards "$P"
R="$ROOT/data/hhtools_robots"
for robot in g1 z01; do
  mkdir -p "$R/$robot"
  cp assets/hhtools_robots/$robot/* "$R/$robot/"
done
ln -sfn ../../g1_description/meshes "$R/g1/meshes"
ln -sfn ../../../Z01_v1.6_whole_body/meshes "$R/z01/meshes"
mkdir -p "$RAW/csv"
export HHTOOLS_ROBOT_PATH=$R PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
pids=()
stop_workers() { for pid in "${pids[@]}"; do kill -- "-$pid" 2>/dev/null || true; done; }
trap stop_workers INT TERM EXIT
for shard in "$RAW/g1csv_p$P"/shard_*; do
  name=$(basename "$shard")
  (
    cd "$HHT"
    exec setsid "$HHT_PYTHON" scripts/batch_r2r_retarget.py --source-robot g1 --target-robot z01 \
      --in "$shard" --out "$RAW/csv" \
      --profile mimic --backend newton --source-fps 50 --fps 50 --ik-iterations 24 \
      --skip-existing --in-process --failure-log "$RAW/failures_$name.jsonl" "$@"
  ) > "$RAW/retarget_$name.log" 2>&1 &
  pids+=("$!")
  echo "Retarget $name: worker ${pids[-1]}, log $RAW/retarget_$name.log"
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
trap - INT TERM EXIT
if [ "$status" != 0 ]; then
  echo "A retarget worker failed; inspect $RAW/retarget_shard_*.log before training." >&2
  exit 1
fi
$PYTHON tools/data/z01_data.py convert "$RAW/csv" "$IK_OUT"
$PYTHON tools/data/condition_z01.py "$IK_OUT" "$OUT" --clean-out "$CLEAN_OUT"
