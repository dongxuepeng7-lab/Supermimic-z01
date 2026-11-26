#!/usr/bin/env bash
# Development evaluation of existing night01 policies; no training or locked response-test access.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/ral_evaluate_existing_5080.sh [pilot|common] [--dry-run]
  pilot:  seeds 2,4; six configurations; nominal/identified (24 evaluations)
  common: seeds 2,4,6,8,10; same configurations (60 evaluations)
Environment:
  RAL_SESSION  existing complete session (default logs/ral_8h/night01_5080)
  RAL_EVAL_OUT new output directory (default timestamped under logs/ral_closed_loop)
These are development reference-library evaluations, not unseen-motion or held-out actuator tests.
The velocity metric currently returned is world-XYZ root velocity per-component RMSE, not forward-only RMSE.
EOF
}

mode="${1:-pilot}"
case "$mode" in
  -h|--help) usage; exit 0 ;;
  pilot) seeds=(2 4) ;;
  common) seeds=(2 4 6 8 10) ;;
  *) usage >&2; exit 2 ;;
esac
dry_run=false
if [[ "${2:-}" == --dry-run ]]; then dry_run=true
elif [[ $# -gt 1 ]]; then usage >&2; exit 2; fi
if [[ $# -gt 2 ]]; then usage >&2; exit 2; fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
session="${RAL_SESSION:-logs/ral_8h/night01_5080}"
output="${RAL_EVAL_OUT:-logs/ral_closed_loop/night01_${mode}_$(date +%Y%m%d_%H%M%S)}"
candidates=(A0 A Aplus B C_reference C_huber_1e2)
if [[ ! -d data/motions/z01_lafan_v2_clean || ! -f release/ral/params.csv ]]; then
  printf 'Missing real motion directory or parameter manifest.\n' >&2
  exit 2
fi
missing=0
for seed in "${seeds[@]}"; do
  for candidate in "${candidates[@]}"; do
    run="$session/jobs/s${seed}_${candidate}/run"
    for name in last.pt args.json run.json; do
      if [[ ! -f "$run/$name" ]]; then
        printf 'Missing input: %s\n' "$run/$name" >&2
        missing=1
      fi
    done
  done
done
if [[ "$missing" == 1 ]]; then
  printf 'Nothing started. Use the complete 5080 session; the 12-file selected bundle alone lacks A0/A/Aplus.\n' >&2
  exit 2
fi
if [[ -e "$output" ]]; then printf 'Output already exists: %s\n' "$output" >&2; exit 2; fi
if ! "$dry_run"; then
  mkdir -p "$(dirname "$output")"
  mkdir "$output"
  printf '%s\n' 'Development only. Fixed evaluation seed 0; one start per reference clip; no DR. Not held-out robustness.' > "$output/scope.txt"
fi
for seed in "${seeds[@]}"; do
  for candidate in "${candidates[@]}"; do
    run="$session/jobs/s${seed}_${candidate}/run"
    for plant in nominal identified; do
      name="s${seed}_${candidate}_${plant}"
      command=(uv run --locked python -u -m partymimic.evaluate
        --run_dir "$run" --ckpt "$run/last.pt"
        --motion_dir data/motions/z01_lafan_v2_clean
        --parameter_manifest release/ral/params.csv
        --seed 0 --plant "$plant" --out "$output/$name.json")
      if "$dry_run"; then printf '%q ' "${command[@]}"; printf '\n'; continue; fi
      printf 'Evaluating %s\n' "$name"
      printf '%q ' "${command[@]}" > "$output/$name.command.txt"
      printf '\n' >> "$output/$name.command.txt"
      if ! "${command[@]}" > "$output/$name.console.log" 2>&1; then
        printf 'Evaluation failed. Read %s/%s.console.log\n' "$output" "$name" >&2
        exit 1
      fi
    done
  done
done
printf 'Output: %s\n' "$output"
