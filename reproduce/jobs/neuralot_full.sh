# NeuralOT on every pair of the paper's OOD set for the six cost types (appendix E per-pair panels and heatmaps):
#   bash reproduce/gpu_queue.sh reproduce/jobs/neuralot_full.sh 2 3
# 8 instances per GPU (baselines/neuralot/ serialises their memory-heavy last step); resumable from its cache.
X="PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
JOBS+=("neuralot_ood_a|$X $PY_NEURALOT -m baselines.neuralot --root $CR --sets ood_paper_sqeuclidean ood_paper_euclidean ood_paper_canberra --per-set all --workers 8")
JOBS+=("neuralot_ood_b|$X $PY_NEURALOT -m baselines.neuralot --root $CR --sets ood_paper_sqrt_euclidean ood_paper_cubic_euclidean ood_paper_gaussian --per-set all --workers 8")
