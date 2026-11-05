# Block A (on otherwise idle GPUs, for clean timings): latency vs n with the last row decoded by the model, then two more seeds of every
# ablation of the small testbed (seed 0 exists for all; base also has seed 1 as base_seed1).
#   bash reproduce/gpu_queue.sh reproduce/jobs/block_a.sh 2 3
mkdir -p "$CR/latency"
JOBS+=("latency_lastrow|PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -m torot.benchmark --checkpoint $EVAL/best.pt --last_row model --out $CR/latency/latency_row_M_5M_lastrow.json")
JOBS+=("ablation_base_s2|bash $REPO/reproduce/run_ablation.sh base_s2")
for seed in 1 2; do
  for name in no_leftover corpus25 no_transpose plain_ce d128 d512; do
    JOBS+=("ablation_${name}_s$seed|bash $REPO/reproduce/run_ablation.sh ${name}_s$seed")
  done
done
