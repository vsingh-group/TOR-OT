# UNOT with log-domain targets at eps = 0.001, for n = 30, 100 and 200 (reproduce/run_unot_logtargets.sh):
#   bash reproduce/gpu_queue.sh reproduce/jobs/unot_logtargets.sh 2 3
JOBS+=("unot_logt_n30|N=30 CR=$CR OUT=$BASE bash $REPO/reproduce/run_unot_logtargets.sh")
JOBS+=("unot_logt_n100|N=100 CR=$REPO/experiments/torot/camera_ready_n100 OUT=$REPO/experiments/baselines_n100 bash $REPO/reproduce/run_unot_logtargets.sh")
JOBS+=("unot_logt_n200|N=200 CR=$REPO/experiments/torot/camera_ready_n200 OUT=$REPO/experiments/baselines_n200 bash $REPO/reproduce/run_unot_logtargets.sh")
