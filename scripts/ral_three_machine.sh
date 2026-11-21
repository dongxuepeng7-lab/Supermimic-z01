#!/usr/bin/env bash
# Sequential, seed-paired pilot runs on the three experiment machines.
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
motion_dir="${RAL_MOTION_DIR:-data/motions/z01_lafan_v2_clean}"
params="release/ral/params.csv"
config="configs/ral/common.json"
responses="${RAL_RESPONSES:-data/ral_pairs_v1/manifest.json}"
protocol="${RAL_PROTOCOL:-release/ral/protocol_v1.json}"
attempt="${RAL_ATTEMPT:-pilot01}"

usage() {
  cat <<'EOF'
Run from a checkout containing the same code, assets and actual motion data:
  bash scripts/ral_three_machine.sh prepare             # 3090: pairs + protocol, once
  bash scripts/ral_three_machine.sh train4090           # seed 1: A0, A, A+, B, C
  bash scripts/ral_three_machine.sh train5080           # seed 2: A, A0, A+, B, C
  bash scripts/ral_three_machine.sh windows SEED TOLERANCE  # 3090: D and C-perm
  bash scripts/ral_three_machine.sh evaluate            # 3090: all 14 final policies

Copy the SAME pairs directory and frozen protocol to all machines after prepare.
Copy all five training run directories per seed to the 3090 before evaluate.
Before windows, copy that seed's complete C run directory to the 3090.
TOLERANCE must be a positive, prespecified calibration error threshold.
Current common.json is a 200-iteration PILOT; window branches use 100 updates.
Fresh attempts: RAL_ATTEMPT=pilot02 (no resume; never overwrite existing runs).
Alternative inputs: RAL_MOTION_DIR, RAL_RESPONSES, RAL_PROTOCOL.
EOF
}

ral() { uv run python -u tools/ral.py "$@"; }
require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'Missing required file: %s\n' "$1" >&2
    exit 2
  fi
}
run_dir() {
  local tag="${1//+/plus}"
  tag="${tag//-/}"
  printf 'logs/ral_%s_s%s_%s' "$tag" "$2" "$attempt"
}
check_shared_inputs() {
  require_file "$protocol"
  require_file "$responses"
  ral preflight --group C --params "$params" --motion_dir "$motion_dir" --responses "$responses"
}
train_sequence() {
  local seed="$1" group out
  shift
  check_shared_inputs
  # Reject a partly completed attempt before starting another member of its sequence.
  for group in "$@"; do
    out="$(run_dir "$group" "$seed")"
    if [[ -e "$out" ]]; then
      printf 'Attempt already exists: %s; choose a fresh RAL_ATTEMPT.\n' "$out" >&2
      exit 2
    fi
  done
  mkdir -p logs/ral_console
  for group in "$@"; do
    out="$(run_dir "$group" "$seed")"
    local response_args=()
    if [[ "$group" != A0 && "$group" != A ]]; then
      response_args=(--responses "$responses")
    fi
    ral train --group "$group" --seed "$seed" --config "$config" \
      --params "$params" --motion_dir "$motion_dir" --protocol "$protocol" \
      "${response_args[@]}" --out "$out" 2>&1 | tee "logs/ral_console/${out##*/}.log"
  done
}

case "${1:---help}" in
  -h|--help|help) usage ;;
  prepare)
    if [[ "${responses##*/}" != manifest.json ]]; then
      printf 'RAL_RESPONSES must end in manifest.json.\n' >&2
      exit 2
    fi
    ral preflight --group A0 --params "$params" --motion_dir "$motion_dir"
    if [[ -e "$(dirname "$responses")" || -e "$protocol" ]]; then
      printf 'Pairs/protocol output already exists. Distribute existing frozen inputs or choose new paths.\n' >&2
      exit 2
    fi
    ral pairs --params "$params" --motion_dir "$motion_dir" --horizon 32 \
      --blocks-per-clip 4 --epsilon 0.01 --out "$(dirname "$responses")"
    # The paired-data generator always writes manifest.json.
    require_file "$responses"
    mkdir -p "$(dirname "$protocol")"
    ral freeze --params "$params" --config "$config" --motion_dir "$motion_dir" \
      --responses "$responses" --out "$protocol"
    ;;
  train4090) train_sequence 1 A0 A 'A+' B C ;;
  train5080) train_sequence 2 A A0 'A+' B C ;;
  windows)
    seed="${2:?Usage: windows SEED TOLERANCE}"
    tolerance="${3:?Supply the prespecified positive calibration tolerance}"
    case "$seed" in 1|2) ;; *) printf 'Expected seed 1 or 2.\n' >&2; exit 2 ;; esac
    check_shared_inputs
    ckpt="$(run_dir C "$seed")/last.pt"
    cal="release/ral/window_calibration_s${seed}_${attempt}.json"
    starts="data/ral_starts_s${seed}_${attempt}"
    require_file "$ckpt"
    for target in "$cal" "$starts" "$(run_dir D "$seed")" "$(run_dir C-perm "$seed")"; do
      if [[ -e "$target" ]]; then
        printf 'Window attempt already exists: %s\n' "$target" >&2
        exit 2
      fi
    done
    ral calibrate --checkpoint "$ckpt" --params "$params" --responses "$responses" \
      --short 8 --long 32 --edges 0.5,1,2 --min-blocks 10 \
      --tolerance "$tolerance" --out "$cal"
    ral capture-starts --checkpoint "$ckpt" --params "$params" --responses "$responses" \
      --motion_dir "$motion_dir" --horizon 32 --steps 200 --out "$starts"
    mkdir -p logs/ral_console
    # Counterbalance order, while keeping each seed's two branches on the same GPU.
    groups=(D C-perm)
    if [[ "$seed" == 2 ]]; then groups=(C-perm D); fi
    for group in "${groups[@]}"; do
      out="$(run_dir "$group" "$seed")"
      ral window-train --group "$group" --checkpoint "$ckpt" --params "$params" \
        --motion_dir "$motion_dir" --calibration "$cal" --starts "$starts/manifest.json" \
        --seed "$seed" --iterations 100 --batch-size 64 --lr 0.0001 \
        --out "$out" 2>&1 | tee "logs/ral_console/${out##*/}.log"
    done
    ;;
  evaluate)
    check_shared_inputs
    for seed in 1 2; do
      for group in A0 A 'A+' B C C-perm D; do
        require_file "$(run_dir "$group" "$seed")/last.pt"
        require_file "$(run_dir "$group" "$seed")/args.json"
      done
    done
    for seed in 1 2; do
      for group in A0 A 'A+' B C C-perm D; do
        out="$(run_dir "$group" "$seed")"
        for plant in nominal identified; do
          uv run python -u -m partymimic.evaluate --run_dir "$out" --ckpt "$out/last.pt" \
            --motion_dir "$motion_dir" --parameter_manifest "$params" --seed 0 \
            --plant "$plant" --out "$out/eval_${plant}.json"
        done
      done
    done
    ;;
  *) usage >&2; exit 2 ;;
esac
