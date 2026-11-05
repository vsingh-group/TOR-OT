# NeuralOT at n = 100 on all 1,000 test instances per cost (the 200 already solved by neuralot_n100.sh are reused from
# the cache), one job per cost; 24 instances in parallel per GPU (~80 instances/h per GPU). Without ALLOW_SHARED, a GPU
# that already runs another job of ours is skipped until it is free:
#   bash reproduce/gpu_queue.sh reproduce/jobs/neuralot_n100_full.sh 2 3
X="PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
CR100=${CR100:-$REPO/experiments/torot/camera_ready_n100}
for cost in sqeuclidean euclidean canberra sqrt_euclidean cubic_euclidean gaussian; do
  JOBS+=("neuralot_n100_full_$cost|$X $PY_NEURALOT -m baselines.neuralot --root $CR100 --sets id_simplex_$cost --per-set all --workers 24")
done
