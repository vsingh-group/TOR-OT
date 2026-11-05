# Extras after the main runs: GPU Sinkhorn baseline with clean timings (needs a GPU with no
# other job), then the scaling curves of the small testbed (corpus fraction, model width; 3 seeds each).
#   bash reproduce/gpu_queue.sh reproduce/jobs/extras_gpu.sh 2 3
X="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1"
CR100=${CR100:-$REPO/experiments/torot/camera_ready_n100}
JOBS+=("sinkhorn_gpu|$X $PY -m baselines.pot.sinkhorn_gpu --root $CR --sets 'id_simplex_*' 'ood_paper_*' && $X $PY -m baselines.pot.sinkhorn_gpu --root $CR100 --sets 'id_simplex_*' 'ood_paper_*'")
for seed in "" _s1 _s2; do
  for name in corpus10 corpus50 d64 d384; do
    JOBS+=("ablation_$name$seed|bash $REPO/reproduce/run_ablation.sh $name$seed")
  done
done
