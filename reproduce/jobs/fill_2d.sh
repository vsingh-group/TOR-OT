# 2D baseline work (Figure 4 C/D, App. F):
#   * NeuralOT on all 1,000 2D test instances of |x-y|^2 (4 shards, index % 4)
#   * Meta OT and UNOT: training seeds 1 and 2 at the settings chosen on validation (error bars, as in Figure 3)
#   * NeuralOT on the 2D |x-y| sets (its other supported 2D cost): 49 OOD pairs + 200 test instances
#   ALLOW_SHARED=1 MIN_FREE_MB=14500 bash reproduce/gpu_queue.sh reproduce/jobs/fill_2d.sh 2 3
#   (shared: runs beside reproduce/jobs/fill_extra.sh; NeuralOT with 24 workers takes ~13 GB; finished jobs are skipped)
X="PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
CR2D=${CR2D:-$REPO/experiments/torot/camera_ready_2d}
B2D="DIM=2 N=36 CR=$CR2D OUT=$REPO/experiments/baselines_2d"
NOT="$X $PY_NEURALOT -m baselines.neuralot --root $CR2D --workers 24"
for i in 0 1 2 3; do
  JOBS+=("neuralot_2d_test_shard$i|$NOT --sets id_simplex2d_sqeuclidean --per-set all --shard $i/4")
  case $i in
    0) JOBS+=("seeds_2d_metaot1|$B2D bash reproduce/run_baseline_seeds.sh metaot 1") ;;
    1) JOBS+=("seeds_2d_metaot2|$B2D bash reproduce/run_baseline_seeds.sh metaot 2") ;;
    2) JOBS+=("seeds_2d_unot1|$B2D bash reproduce/run_baseline_seeds.sh unot 1") ;;
    3) JOBS+=("seeds_2d_unot2|$B2D bash reproduce/run_baseline_seeds.sh unot 2") ;;
  esac
done
JOBS+=("neuralot_2d_euclidean|$NOT --sets ood2d_paper_euclidean id_simplex2d_euclidean --per-set 200")

