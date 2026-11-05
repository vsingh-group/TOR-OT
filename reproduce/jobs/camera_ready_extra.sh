# GPU queue after the learned baselines were trained (reproduce/run_learned_baselines.sh):
#   bash reproduce/gpu_queue.sh reproduce/jobs/camera_ready_extra.sh 0 1
# 1. clean timings of the learned baselines (their training runs shared the GPUs with other jobs);
# 2. learned baselines followed by k Sinkhorn iterations (their intended use), with cold-start Sinkhorn as reference;
# 3. NeuralOT on a subset of the same instances;
# 4. ablations of the small testbed (torot/configs/ablation_n30.yaml);
# 5. two more training seeds of each learned baseline at its selected setting.
X="XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1"
SETS="'id_simplex_*' 'ood_paper_*'"
JOBS+=("predict_metaot_clean|$X $PY_METAOT -m baselines.metaot.predict --root $CR --runs '$BASE/metaot/*' --sets $SETS")
JOBS+=("predict_unot_clean|$PY_UNOT -m baselines.unot.predict --root $CR --runs '$BASE/unot/*' --sets $SETS")
JOBS+=("refine_metaot|$X $PY_METAOT -m baselines.metaot.refine --root $CR --runs '$BASE/metaot/*' --sets 'id_simplex_*' 'ood_paper_*'")
JOBS+=("refine_unot|$PY_UNOT -m baselines.unot.refine --root $CR --runs '$BASE/unot/*' --sets 'id_simplex_*' 'ood_paper_*'")
JOBS+=("neuralot_id_simplex|$PY_NEURALOT -m baselines.neuralot --root $CR --sets 'id_simplex_*' --per-set 10 --workers 2")
JOBS+=("neuralot_ood_paper|$PY_NEURALOT -m baselines.neuralot --root $CR --sets 'ood_paper_*' --per-set 10 --workers 2")
for name in base base_seed1 no_transpose plain_ce no_leftover d128 d512 corpus25; do
  JOBS+=("ablation_$name|bash $REPO/reproduce/run_ablation.sh $name")
done
for seed in 1 2; do
  JOBS+=("seeds_metaot_$seed|bash $REPO/reproduce/run_baseline_seeds.sh metaot $seed")
  JOBS+=("seeds_unot_$seed|bash $REPO/reproduce/run_baseline_seeds.sh unot $seed")
done
