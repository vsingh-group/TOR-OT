# Follow-up of camera_ready_extra2.sh: Meta OT timing on a GPU with no other job, then NeuralOT on the remaining
# OOD sets of the paper set.
#   bash reproduce/gpu_queue.sh reproduce/jobs/camera_ready_extra3.sh 0 1
X="XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
JOBS+=("predict_metaot_clean|$X $PY_METAOT -m baselines.metaot.predict --root $CR --runs '$BASE/metaot/*' --sets 'id_simplex_*' 'ood_paper_*'")
JOBS+=("neuralot_ood_paper_rest|$X $PY_NEURALOT -m baselines.neuralot --root $CR --sets ood_paper_canberra ood_paper_euclidean --per-set 10 --workers 2")
JOBS+=("neuralot_ood_paper_sqeuclidean|$X $PY_NEURALOT -m baselines.neuralot --root $CR --sets ood_paper_sqeuclidean --per-set 10 --workers 1")
