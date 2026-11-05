#!/usr/bin/env bash
# Learned baselines for Figure 3 (GPU): train Meta OT and UNOT for every cost and entropic regularisation, then
# predict the evaluation sets with the run chosen on the validation set (lowest validation MAPE per cost).
#   bash reproduce/run_learned_baselines.sh [gpu ...]          (default: 0 1; all 12 runs in parallel, the k-th eps
#                                                               value on the k-th GPU, round robin)
# Environments: env_metaot (baselines/metaot/requirements.txt), env_unot (baselines/unot/requirements.txt).
# Needs the sets built by reproduce/build_sets.py (sets/ and sets_val/ for the costs below).
# Other support sizes: N=100 CR=<root built with --n 100> OUT=experiments/baselines_n100 (MAX_PAR limits concurrency).
# 2D (6 x 6 grid, reproduce/build_sets_2d.py): DIM=2 N=36 CR=experiments/torot/camera_ready_2d OUT=experiments/baselines_2d
# (UNOT then uses its official image generator and log-domain Sinkhorn targets for every eps).
# Outputs: experiments/baselines/<method>/<cost>_eps<eps>/ (checkpoints, config, validation metrics),
#          <camera_ready>/preds/<set>__{metaot,unot}.npy (+ .timing.json), experiments/baselines/env_*.freeze.txt
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
CR=${CR:-$REPO/experiments/torot/camera_ready}
OUT=${OUT:-$REPO/experiments/baselines}
CONDA_ENVS=${CONDA_ENVS:-$(conda info --base 2>/dev/null)/envs}
PY_METAOT=${PY_METAOT:-$CONDA_ENVS/env_metaot/bin/python}
PY_UNOT=${PY_UNOT:-$CONDA_ENVS/env_unot/bin/python}
GPUS=("$@"); [ ${#GPUS[@]} -eq 0 ] && GPUS=(0 1)
N=${N:-30}                                          # support size of the sets in $CR (2D: number of bins, side^2)
DIM=${DIM:-1}                                       # 1 or 2 (2D sets of reproduce/build_sets_2d.py)
UNOT_EXTRA=(); [ "$DIM" = 2 ] && UNOT_EXTRA=(--log-targets)
W_OFFICIAL=$((4 * N))                               # UNOT's official width rule (4 x input size)
MAX_PAR=${MAX_PAR:-0}                               # > 0: at most this many training runs at once
throttle() { [ "$MAX_PAR" -gt 0 ] && while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do wait -n; done; return 0; }
read -r -a COSTS <<< "${COSTS:-sqeuclidean euclidean canberra sqrt_euclidean cubic_euclidean gaussian}"   # the six cost types of the corpus
EPS=(0.01 0.001)
SETS=('id_simplex_*' 'ood_paper_*'); [ "$DIM" = 2 ] && SETS=('id_simplex2d_*' 'ood2d_paper_*')
export XLA_PYTHON_CLIENT_PREALLOCATE=false          # several small JAX runs share a GPU
mkdir -p "$OUT/logs"

train() {  # method python gpu cost eps (skips finished runs)
  local method=$1 py=$2 gpu=$3 cost=$4 eps=$5 run=$OUT/$1/${4}_eps$5 log=$OUT/logs/${1}_${4}_eps$5.log
  [ -f "$run/metrics.json" ] && return 0
  echo "$(date +%F_%T) start $method $cost eps=$eps (GPU $gpu)"
  local extra=(); [ "$method" = unot ] && extra=("${UNOT_EXTRA[@]}")
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$py" -m baselines.$method.train --cost "$cost" --eps "$eps" --out "$run" \
    --val "$CR" --n "$N" --dim "$DIM" "${extra[@]}" > "$log" 2>&1 || echo "$(date +%F_%T) FAILED $method $cost eps=$eps"
  echo "$(date +%F_%T) end   $method $cost eps=$eps: $(grep '^done' "$log" | tail -1)"
}
for i in "${!EPS[@]}"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  for cost in "${COSTS[@]}"; do
    throttle; train metaot "$PY_METAOT" "$gpu" "$cost" "${EPS[$i]}" &
    throttle; train unot "$PY_UNOT" "$gpu" "$cost" "${EPS[$i]}" &
  done
done
wait
# UNOT with the official width rule (4 x n hidden units, 120 for n = 30); validation chooses between widths and eps
for i in "${!EPS[@]}"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  for cost in "${COSTS[@]}"; do
    run=$OUT/unot/${cost}_eps${EPS[$i]}_w$W_OFFICIAL; log=$OUT/logs/unot_${cost}_eps${EPS[$i]}_w$W_OFFICIAL.log
    [ -f "$run/metrics.json" ] && continue
    throttle
    ( CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY_UNOT" -m baselines.unot.train --cost "$cost" --eps "${EPS[$i]}" \
        --width $W_OFFICIAL --n "$N" --dim "$DIM" "${UNOT_EXTRA[@]}" --out "$run" --val "$CR" > "$log" 2>&1 || echo "$(date +%F_%T) FAILED unot $cost eps=${EPS[$i]} w$W_OFFICIAL" ) &
  done
done
wait

note=()
[ -n "${NOTE:-}" ] && note=(--note "$NOTE")
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY_METAOT" -m baselines.metaot.predict --root "$CR" --runs "$OUT/metaot/*" \
  --sets "${SETS[@]}" "${note[@]}" 2>&1 | grep -v "^[WEI]0000\|cuda_\|absl::\|hydra.main\|compatability\|assume defaults" | tee "$OUT/logs/metaot_predict.log"
CUDA_VISIBLE_DEVICES=${GPUS[-1]} "$PY_UNOT" -m baselines.unot.predict --root "$CR" --runs "$OUT/unot/*" \
  --sets "${SETS[@]}" "${note[@]}" 2>&1 | tee "$OUT/logs/unot_predict.log"
for env in env_metaot env_unot; do
  "$CONDA_ENVS/$env/bin/python" -m pip freeze > "$OUT/$env.freeze.txt"
done
echo "$(date +%F_%T) all done"
