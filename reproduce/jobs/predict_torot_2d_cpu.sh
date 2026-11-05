#!/bin/bash
# 2D results (Figure 4 C/D, Figure 5, Appendix F) on CPU: ToR-OT predictions of the 2D model (CKPT2D, default: the
# run of torot/configs/torot_2d.yaml), then the scores. Greedy decoding gives the same plans on CPU and GPU; CPU runs
# write no timing files (predict_torot_2d.sh, on GPUs, does).
#   bash reproduce/jobs/predict_torot_2d_cpu.sh            (env_torot; about 1 h with two processes x 32 threads)
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-python}                                  # env_torot
CR2D=${CR2D:-experiments/torot/camera_ready_2d}
CKPT2D=${CKPT2D:-experiments/torot/row_M_2d_mixed_support/best.pt}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${THREADS:-32} PYTHONUNBUFFERED=1
mkdir -p $CR2D/logs
predict() {   # $1: log name, then the costs
  local log=$1; shift; local pats=()
  for c in "$@"; do pats+=("id_simplex2d_$c" "ood2d_paper_$c"); done
  $PY reproduce/predict_torot.py $CR2D $CKPT2D "${pats[@]}" --last-row model --plain-only --device cpu > $CR2D/logs/$log 2>&1
}
predict predict_torot2d_cpu_a.log canberra cubic_euclidean euclidean &
predict predict_torot2d_cpu_b.log gaussian sqeuclidean sqrt_euclidean &
wait
$PY reproduce/score.py $CR2D > $CR2D/logs/score.log 2>&1
$PY reproduce/score_reference.py $CR2D --sets 'id_simplex2d_*' 'ood2d_paper_*' --methods v2row_lastrow --workers 16 \
  > $CR2D/logs/score_reference_torot2d.log 2>&1
tail -13 $CR2D/logs/score_reference_torot2d.log
