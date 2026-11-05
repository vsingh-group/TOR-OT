# Follow-up of camera_ready_extra.sh (run after it):
#   bash reproduce/gpu_queue.sh reproduce/jobs/camera_ready_extra2.sh 0 1
# * NeuralOT again with 2 instances per GPU: with 6 the legacy routine's final 1M-sample histogram (~13 GB each)
#   ran out of GPU memory;
# * Meta OT's timing again, on an otherwise idle GPU.
X="XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
JOBS+=("neuralot_id_simplex|$X $PY_NEURALOT -m baselines.neuralot --root $CR --sets 'id_simplex_*' --per-set 10 --workers 2")
JOBS+=("neuralot_ood_paper|$X $PY_NEURALOT -m baselines.neuralot --root $CR --sets 'ood_paper_*' --per-set 10 --workers 2")
JOBS+=("predict_metaot_clean|$X $PY_METAOT -m baselines.metaot.predict --root $CR --runs '$BASE/metaot/*' --sets 'id_simplex_*' 'ood_paper_*'")
