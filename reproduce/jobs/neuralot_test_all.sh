# NeuralOT on all 1,000 test instances of every cost (the 500 already solved are reused from the cache), one job per
# cost so that the GPUs balance the load; 24 instances in parallel per GPU:
#   bash reproduce/gpu_queue.sh reproduce/jobs/neuralot_test_all.sh 2 3
X="PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
for cost in sqeuclidean euclidean canberra sqrt_euclidean cubic_euclidean gaussian; do
  JOBS+=("neuralot_test_all_$cost|$X $PY_NEURALOT -m baselines.neuralot --root $CR --sets id_simplex_$cost --per-set all --workers 24")
done
