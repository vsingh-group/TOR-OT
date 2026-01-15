#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_TF4OT

# run all configs
export CUDA_VISIBLE_DEVICES=2,3,4,7
LOGDIR=./experiments/logs
mkdir -p "$LOGDIR"

# FIXED COST
# 100K data
#python3 -m TF4OT.run_training  --config-name TF_100K_cost_fixed     > $LOGDIR/train_TF_100K_cost_fixed.log 2>&1 && \
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

#python3 TF4OT/evals/NOT/NOT_toy_1D_cost_weak.py
python3 TF4OT/evals/NOT/NOT_toy_2D_cost_strong.py

# copy results from chtc condor
results_path="model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_minkowski"
rsync -ah --progress $results_path ivan@10.131.151.11:/home/ivan/repos/TF4OT/experiments/cost_fixed/

results_path="/staging/moros2/experiments/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_minkowski"
ssh $results_path ivan@ip:/home/ivan/repos/TF4OT/experiments/cost_fixed/
scp -r /staging/moros2/experiments/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_minkowski ivan@10.131.151.11:/home/ivan/repos/TF4OT/experiments/cost_fixed/