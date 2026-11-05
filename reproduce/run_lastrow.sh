#!/usr/bin/env bash
# ToR-OT with the last row decoded by the model (torot.decode.decode_by_rows(last_row="model")) instead of the
# remaining capacity: every evaluation set (GPU A) and the 5M corpus' test splits (GPU B), one job per GPU.
#   bash reproduce/run_lastrow.sh [gpu_sets] [gpu_5m]          (default: 0 1)
# Outputs: <camera_ready>/preds/<set>__v2row_lastrow{,_sym}...  and  <camera_ready>/eval_row_M_5M/eval_<split>_lastrow.json
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
CR=${CR:-$REPO/experiments/torot/camera_ready}
EVAL=${EVAL:-$REPO/experiments/torot/row_M_5M}
CONDA_ENVS=${CONDA_ENVS:-$(conda info --base 2>/dev/null)/envs}
PY=${PY:-$CONDA_ENVS/env_torot/bin/python}
GPU_SETS=${1:-0}; GPU_5M=${2:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$CR/logs" "$CR/eval_row_M_5M"
( CUDA_VISIBLE_DEVICES=$GPU_SETS "$PY" -u reproduce/predict_torot.py "$CR" "$EVAL/best.pt" --last-row model \
    > "$CR/logs/predict_torot_lastrow.log" 2>&1; echo "$(date +%F_%T) sets done rc=$?" ) &
( for split in test_ood test_id test_extrap; do
    extra=(); [ "$split" = test_extrap ] && extra=(--max_instances 1000) && bs=64 || bs=128
    CUDA_VISIBLE_DEVICES=$GPU_5M "$PY" -m torot.evaluate --checkpoint "$EVAL/best.pt" --split "$split" --batch_size $bs \
      "${extra[@]}" --last_row model --out "$CR/eval_row_M_5M/eval_${split}_lastrow.json" \
      > "$CR/logs/eval_${split}_lastrow.log" 2>&1; echo "$(date +%F_%T) $split done rc=$?"
  done ) &
wait
echo "$(date +%F_%T) all done"
