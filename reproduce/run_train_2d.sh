#!/usr/bin/env bash
# Train the 2D ToR-OT model (same architecture and recipe as the paper's 1D model row_M_5M, 2D corpus) on two GPUs,
# sized to a wall-clock budget:
#   1. wait until the 2D corpus is completely on the Hub (all splits and the dataset card, which is uploaded last);
#   2. wait until the GPUs are free (less than 1 GB in use, by anyone);
#   3. timing run (300 steps) -> seconds per step;  4. full run with max_steps = HOURS * 3600 / (seconds per step).
#
#   DATA=ivanjaenm/torot_data_4M_2d HOURS=64 bash reproduce/run_train_2d.sh 2,3 experiments/torot/row_M_2d_mixed_support
# BATCH / MAX_TOKENS: rows / encoder tokens per GPU (default 256 / 32768, the 1D model's; see RECIPE).
# Progress of the full run: cat <out_dir>/progress.txt ; python -m torot.progress <out_dir>
set -uo pipefail
GPUS=${1:?usage: run_train_2d.sh <gpus, e.g. 2,3> <out_dir>}; OUT=${2:?out_dir}
DATA=${DATA:-ivanjaenm/torot_data_4M_2d}; HOURS=${HOURS:-24}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
CONDA_ENVS=${CONDA_ENVS:-$(conda info --base 2>/dev/null)/envs}
export PATH=$CONDA_ENVS/env_torot/bin:$PATH
PY=$CONDA_ENVS/env_torot/bin/python
# the paper model's recipe (experiments/torot/row_M_5M/config.yaml), 2D corpus. Batch per GPU as the 1D model: 256 rows /
# 32,768 tokens. It nearly fills a 40 GB A100 over a whole run: the 1D run peaked at 30.7 GB in its first 300 steps
# but 36.1 GB overall (the largest batches are rare), and the 2D corpus starts at the same 30.5 GB. A short run
# therefore understates the memory; 288 rows / 36,864 tokens (33.0 GB in 300 steps) would exceed the card later.
RECIPE=(model.decoding=row model.d_model=768 model.n_heads=12 model.d_ff=2048 optim.batch_size=${BATCH:-256}
        optim.max_tokens=${MAX_TOKENS:-32768} optim.max_dec_tokens=24576 data.repo_id=$DATA data.residual_leftover_prob=0.25
        data.residual_noise=0.01)
log() { echo "$(date +%F_%T) $*"; }

log "waiting for $DATA on the Hub"
until $PY - "$DATA" <<'EOF'
import sys
from huggingface_hub import HfApi, hf_hub_download
try:
    info = HfApi().dataset_info(sys.argv[1], files_metadata=False)
    files = [s.rfilename for s in info.siblings]
    card = open(hf_hub_download(sys.argv[1], "README.md", repo_type="dataset")).read()
    ok = any("test_ood" in f for f in files) and "synthetic optimal-transport corpus" in card
except Exception:
    ok = False
sys.exit(0 if ok else 1)
EOF
do sleep 300; done
log "dataset complete; downloading"
$PY -c "from datasets import load_dataset; d = load_dataset('$DATA'); print({k: len(v) for k, v in d.items()})"

# busy: any process (of any user) holds more than 1 GB on one of the GPUs (the run needs the whole card)
busy() { local g; for g in ${GPUS//,/ }; do
           [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")" -gt 1024 ] && return 0; done; return 1; }
log "waiting for GPUs $GPUS"; while busy; do sleep 60; done

log "timing run"
rm -rf "$OUT.timing"
torot/run_train.sh "$GPUS" "$OUT.timing" "${RECIPE[@]}" optim.max_steps=300 optim.warmup_steps=50 run.val_every=1000000
while kill -0 "$(cat "$OUT.timing/train.pid")" 2>/dev/null; do sleep 20; done
SPS=$($PY - "$OUT.timing/log.jsonl" <<'EOF'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1]) if '"step"' in l and '"elapsed_s"' in l]
r = [x for x in r if x["step"] >= 100]                       # skip warm-up / compilation
print((r[-1]["elapsed_s"] - r[0]["elapsed_s"]) / (r[-1]["step"] - r[0]["step"]))
EOF
)
STEPS=$($PY -c "print(int($HOURS * 3600 / $SPS) // 1000 * 1000)")
log "seconds/step $SPS -> max_steps $STEPS for $HOURS h"
torot/run_train.sh "$GPUS" "$OUT" "${RECIPE[@]}" optim.max_steps="$STEPS" optim.warmup_steps=2000 run.val_every=5000
log "full run started"
