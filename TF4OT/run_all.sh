#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_TF4OT

# run all configs
export CUDA_VISIBLE_DEVICES=1,2,3
LOGDIR=./experiments/logs
mkdir -p "$LOGDIR"

# FIXED COST
# 100K data
python3 -m TF4OT.run_training  --config-name TF_100K_cost_fixed     > $LOGDIR/train_TF_100K_cost_fixed.log 2>&1 && \
python3 -m TF4OT.run_inference --config-name TF_100K_cost_fixed     > $LOGDIR/eval_TF_100K_cost_fixed.log  2>&1
# 1M data
#python3 -m TF4OT.run_training  --config-name TF_1M_cost_fixed       > $LOGDIR/train_TF_1M_cost_fixed.log 2>&1 && \
#python3 -m TF4OT.run_inference --config-name TF_1M_cost_fixed       > $LOGDIR/eval_TF_1M_cost_fixed.log  2>&1
# 10M data
#python3 -m TF4OT.run_training  --config-name TF_10M_cost_fixed      > $LOGDIR/train_TF_10M_cost_fixed.log 2>&1 && \
#python3 -m TF4OT.run_inference --config-name TF_10M_cost_fixed      > $LOGDIR/eval_TF_10M_cost_fixed.log  2>&1

# SEMIVARIABLE COST
# 100K data
#python3 -m TF4OT.run_training  --config-name TF_100K_cost_semivariable  > $LOGDIR/train_TF_100K_cost_semivariable.log 2>&1 && \
#python3 -m TF4OT.run_inference --config-name TF_100K_cost_semivariable  > $LOGDIR/eval_TF_100K_cost_semivariable.log  2>&1
# 1M data
# python3 -m TF4OT.run_training  --config-name TF_1M_cost_semivariable  > $LOGDIR/train_TF_1M_cost_semivariable.log 2>&1 && \
# python3 -m TF4OT.run_inference --config-name TF_1M_cost_semivariable  > $LOGDIR/eval_TF_1M_cost_semivariable.log  2>&1
# # 10M data
# python3 -m TF4OT.run_training  --config-name TF_10M_cost_semivariable  > $LOGDIR/train_TF_10M_cost_semivariable.log 2>&1 && \
# python3 -m TF4OT.run_inference --config-name TF_10M_cost_semivariable  > $LOGDIR/eval_TF_10M_cost_semivariable.log  2>&1

# VARIABLE COST
# 100K data
#python3 -m TF4OT.run_training  --config-name TF_100K_cost_variable  > $LOGDIR/train_TF_100K_cost_variable.log 2>&1 && \
#python3 -m TF4OT.run_inference --config-name TF_100K_cost_variable  > $LOGDIR/eval_TF_100K_cost_variable.log 2>&1
# 1M data
# python3 -m TF4OT.run_training  --config-name TF_1M_cost_variable    > $LOGDIR/train_TF_1M_cost_variable.log 2>&1 && \
# python3 -m TF4OT.run_inference --config-name TF_1M_cost_variable    > $LOGDIR/eval_TF_1M_cost_variable.log 2>&1
# 10M data
# python3 -m TF4OT.run_training  --config-name TF_10M_cost_variable   > $LOGDIR/train_TF_10M_cost_variable.log 2>&1 && \
# python3 -m TF4OT.run_inference --config-name TF_10M_cost_variable   > $LOGDIR/eval_TF_10M_cost_variable.log  2>&1