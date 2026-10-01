#!/usr/bin/env bash
# Launch a ToR-OT training run detached from the terminal (survives SSH / VPN drops).
#
#   torot/run_train.sh <gpus> <out_dir> [extra config overrides...]
#
# e.g.  torot/run_train.sh 0,1,6,7 experiments/torot/base_25ep optim.max_steps=72000
#
# Progress:  cat <out_dir>/progress.txt          (one-glance status, updated every 50 steps)
#            python -m torot.progress <out_dir>  (validation table)
#            tail -f <out_dir>/train.log         (raw stdout/stderr)
# Stop:      kill $(cat <out_dir>/train.pid)
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <gpus> <out_dir> [config overrides...]" >&2
    exit 1
fi
GPUS=$1
OUT_DIR=$2
shift 2

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_DIR"
mkdir -p "$OUT_DIR"

NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PORT=$((29500 + RANDOM % 1000))

CUDA_VISIBLE_DEVICES=$GPUS setsid nohup torchrun --nproc_per_node "$NPROC" --master_port "$PORT" \
    -m torot.train --config torot/configs/default.yaml run.out_dir="$OUT_DIR" "$@" \
    > "$OUT_DIR/train.log" 2>&1 < /dev/null &

echo $! > "$OUT_DIR/train.pid"
echo "started pid $(cat "$OUT_DIR/train.pid") on GPUs $GPUS -> $OUT_DIR"
echo "  status:   cat $OUT_DIR/progress.txt"
echo "  details:  python -m torot.progress $OUT_DIR"
echo "  log:      tail -f $OUT_DIR/train.log"
