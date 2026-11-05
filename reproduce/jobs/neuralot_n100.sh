# NeuralOT at n = 100 for the scaling figure: 200 evenly spaced test instances per cost (nested_subset) and all 144 OOD
# pairs, one job per cost so the two GPUs balance the load; 24 instances in parallel per GPU (~80 instances/h per GPU):
#   ALLOW_SHARED=1 bash reproduce/gpu_queue.sh reproduce/jobs/neuralot_n100.sh 2 3
X="PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
CR100=${CR100:-$REPO/experiments/torot/camera_ready_n100}
for cost in sqeuclidean euclidean canberra sqrt_euclidean cubic_euclidean gaussian; do
  JOBS+=("neuralot_n100_$cost|$X $PY_NEURALOT -m baselines.neuralot --root $CR100 --sets id_simplex_$cost ood_paper_$cost --per-set 200 --workers 24")
done
