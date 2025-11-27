#!/usr/bin/env bash
set -euo pipefail

# ======== quick config (edit if needed) ========
base="${base:-$(cd .. && pwd)}"
echo $base
IMAGE="${IMAGE:-noalcohol/covid:3env-offline}"
MODE_ENV="${MODE_ENV:-covid}"

mkdir -p "$base/logs" "$base/saved_models"

# join helper
_join_dash() { local IFS=-; echo "$*"; }

# print help
_usage() {
  cat <<'EOF'
Usage:
  ./covid_runner.sh train <mods...> [-- extra train args]
  ./covid_runner.sh test  <mods...> [-- extra test args]
  ./covid_runner.sh all   <mods...> [-- extra train args]   # train then test
  ./covid_runner.sh combos                                  # train all non-empty combos

Notes:
  - <mods...> is any subset/order of: ct ehr cxr scrna
  - Everything after `--` goes straight to the Python script (epochs, batch size, lr, seed, etc.).
  - Outputs:  saved_models/<RUN_ID>-<tag>/
  - Logs:     logs/<RUN_ID>-<tag>/{train.log,test.log}
Env (optional):
  base=/path/to/project
  IMAGE=noalcohol/covid:3env-offline
  MODE_ENV=covid
  RUN_ID=YYYYmmdd-HHMMSS    # reuse/force a run id
Examples:
  ./covid_runner.sh train scrna
  ./covid_runner.sh train ct ehr -- --epochs 80 --batch-size 32 --lr 3e-5 --seed 777
  ./covid_runner.sh test  ct ehr
  ./covid_runner.sh all   ct cxr scrna -- --epochs 60
  RUN_ID=20251112-183015 ./covid_runner.sh test ct ehr
EOF
}

# core runner (shared)
_run() {
  local phase="$1"; shift
  # split mods vs extras (after --)
  local -a mods=() extra=()
  local in_extra=0
  for tok in "$@"; do
    if [ "$tok" = "--" ]; then in_extra=1; continue; fi
    if [ $in_extra -eq 0 ]; then mods+=("$tok"); else extra+=("$tok"); fi
  done
  if [ "${#mods[@]}" -eq 0 ]; then echo "Select at least one modality."; exit 1; fi

  IFS=$'\n' read -r -d '' -a sorted_mods < <(printf '%s\n' "${mods[@]}" | sort && printf '\0')
  local tag; tag="$(_join_dash "${sorted_mods[@]}")"
  local run_id="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

  local host_out="$base/saved_models/${run_id}-${tag}"
  mkdir -p "$host_out"

  local log_dir="$base/logs/${run_id}-${tag}"
  mkdir -p "$log_dir"
  local log_file="$log_dir/${phase}.log"

  local module="src.ablation_study_622.${phase}"

  echo ">>> [$phase] run_id=$run_id  mods=${mods[*]}  tag=$tag"
  echo ">>> [$phase] out=$host_out"
  echo ">>> [$phase] log=$log_file"

  docker run --rm --gpus all \
    -u "$(id -u)":"$(id -g)" \
    -w /app \
    -e MODE="$MODE_ENV" \
    -e PYTHONPATH=/app \
    -e OUTPUT_DIR="/saved_models" \
    -e RUN_ID="$run_id" \
    -v "$base/data":/data:ro \
    -v "$host_out":/saved_models:rw \
    -v "$base/src":/app/src:ro \
    "$IMAGE" \
    run_mode -m "$module" --modalities "${mods[@]}" "${extra[@]}" \
    | tee "$log_file"
}

train_cmd() { _run "train" "$@"; }
test_cmd()  { _run "test"  "$@"; }

# train then test with same RUN_ID
all_cmd() {
  local now="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
  RUN_ID="$now" train_cmd "$@"
  RUN_ID="$now" test_cmd  "$@"
}

# generate all non-empty combinations and train
combos_cmd() {
  local mods=(ct ehr cxr scrna)
  local n=${#mods[@]}
  local max=$(( (1<<n) - 1 ))
  for mask in $(seq 1 "$max"); do
    sel=()
    for i in $(seq 0 $((n-1))); do
      (( (mask>>i)&1 )) && sel+=("${mods[$i]}")
    done
    train_cmd "${sel[@]}"
  done
}

# dispatch
cmd="${1:-help}"; shift || true
case "$cmd" in
  train)  train_cmd "$@";;
  test)   test_cmd  "$@";;
  all)    all_cmd   "$@";;
  combos) combos_cmd;;
  help|-h|--help) _usage;;
  *) echo "Unknown command: $cmd"; _usage; exit 1;;
esac

