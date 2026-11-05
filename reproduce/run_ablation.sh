#!/usr/bin/env bash
# One ablation of the small testbed (torot/configs/ablation_n30.yaml): train on one GPU, then evaluate test_id and
# test_ood of the testbed corpus (last row decoded by the model).
#   bash reproduce/run_ablation.sh <name>[_s<seed>] [gpu]
# names: base, no_transpose, plain_ce, no_leftover, d64, d128, d384, d512, corpus10, corpus25, corpus50 (seed 0);
# <name>_s<k> trains seed k
# (base_seed1 is the first extra seed of base, kept under its original name).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
NAME=${1:?name}; GPU=${2:-}
ABL=${ABL:-$REPO/experiments/torot/ablations}
CONDA_ENVS=${CONDA_ENVS:-$(conda info --base 2>/dev/null)/envs}
PY=${PY:-$CONDA_ENVS/env_torot/bin/python}
SEED_OV=()
CONFIG=$NAME
if [[ $NAME =~ ^(.+)_s([0-9]+)$ ]]; then CONFIG=${BASH_REMATCH[1]}; SEED_OV=(run.seed=${BASH_REMATCH[2]}); fi
case $CONFIG in
  base)         OV=() ;;
  base_seed1)   OV=(run.seed=1) ;;                                            # seed-to-seed spread
  no_transpose) OV=(data.transpose_aug_prob=0.0) ;;                           # symmetry 6.1 off in training
  plain_ce)     OV=(loss.digit_weight_decay=1.0 loss.ordinal_eps=0.0) ;;      # plain token cross-entropy
  no_leftover)  OV=(data.residual_leftover_prob=0.0 data.residual_noise=0.0) ;;  # no capacity perturbation
  d128)         OV=(model.d_model=128 model.n_heads=4 model.d_ff=384) ;;      # ~2.5M parameters
  d512)         OV=(model.d_model=512 model.n_heads=8 model.d_ff=1408 optim.max_dec_tokens=65536) ;;   # ~38M
  corpus25)     OV=(data.repo_id=$ABL/corpus_n30_25pct) ;;                    # a quarter of the training instances
  corpus10)     OV=(data.repo_id=$ABL/corpus_n30_10pct) ;;                    # scaling curve: 10% of the training instances
  corpus50)     OV=(data.repo_id=$ABL/corpus_n30_50pct) ;;                    #                50%
  d64)          OV=(model.d_model=64 model.n_heads=4 model.d_ff=192) ;;       # scaling curve: ~0.7M parameters
  d384)         OV=(model.d_model=384 model.n_heads=6 model.d_ff=1024 optim.max_dec_tokens=65536) ;;   # ~22M
  *) echo "unknown ablation $NAME" >&2; exit 1 ;;
esac
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES=$GPU          # otherwise keep the caller's (e.g. the GPU queue)
mkdir -p "$ABL/$NAME"
if [ ! -f "$ABL/$NAME/best.pt" ] || ! grep -q '"step": 11000' "$ABL/$NAME/log.jsonl" 2>/dev/null; then
  "$CONDA_ENVS/env_torot/bin/torchrun" --nproc_per_node 1 --master_port $((29500 + RANDOM % 1000)) -m torot.train \
    --config torot/configs/ablation_n30.yaml run.out_dir="$ABL/$NAME" "${OV[@]}" "${SEED_OV[@]}" > "$ABL/$NAME/train.log" 2>&1
fi
for split in test_id test_ood; do
  "$PY" -m torot.evaluate --checkpoint "$ABL/$NAME/best.pt" --split $split --batch_size 64 --last_row model \
    --out "$ABL/$NAME/eval_$split.json" > "$ABL/$NAME/eval_$split.log" 2>&1
done
echo "$NAME done"
