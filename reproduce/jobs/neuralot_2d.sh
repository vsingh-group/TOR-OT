# NeuralOT on the 2D sets (reproduce/build_sets_2d.py, 6 x 6 grid, |x-y|^2): all 49 OOD pairs and an evenly spaced
# subset of 200 of the 1,000 test instances (the rest can be added later with --per-set all; solved instances are
# cached), 24 instances in parallel per GPU:
#   bash reproduce/gpu_queue.sh reproduce/jobs/neuralot_2d.sh 3
X="PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
CR2D=${CR2D:-$REPO/experiments/torot/camera_ready_2d}
JOBS+=("neuralot_2d_sqeuclidean|$X $PY_NEURALOT -m baselines.neuralot --root $CR2D --sets ood2d_paper_sqeuclidean id_simplex2d_sqeuclidean --per-set 200 --workers 24")
