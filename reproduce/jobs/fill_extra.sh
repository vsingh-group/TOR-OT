# Extra baseline work, run beside reproduce/jobs/fill_2d.sh on the same GPUs (small jobs, memory-guarded):
#   bash reproduce/gpu_queue.sh reproduce/jobs/fill_extra.sh 2 3     with ALLOW_SHARED=1 MIN_FREE_MB=5000
#   * n = 30: UNOT seeds 1 and 2 for |x-y|^2 and |x-y|^3 at the seed-0 setting (log-domain targets, chosen on validation
#     after the first seed runs; the other costs' seed runs already match and are reused), then their predictions
#   * Sinkhorn reference plans (entropic scoring of Meta OT / UNOT): all n = 200 sets
#   * n = 100 and n = 200: Meta OT and UNOT training seeds 1 and 2 (error bars of the scaling figure), 3 runs at a time
X="PYTHONUNBUFFERED=1"
CR200=${CR200:-$REPO/experiments/torot/camera_ready_n200}
JOBS+=("seeds_n30_unot1|bash reproduce/run_baseline_seeds.sh unot 1")
JOBS+=("seeds_n30_unot2|bash reproduce/run_baseline_seeds.sh unot 2")
JOBS+=("sinkhorn_gpu_n200|$X $PY -m baselines.pot.sinkhorn_gpu --root $CR200 --sets 'id_simplex_*' 'ood_paper_*'")
for n in 100 200; do
  for s in 1 2; do
    for m in metaot unot; do
      JOBS+=("seeds_n${n}_${m}$s|MAX_PAR=3 N=$n CR=$REPO/experiments/torot/camera_ready_n$n OUT=$REPO/experiments/baselines_n$n bash reproduce/run_baseline_seeds.sh $m $s")
    done
  done
done

