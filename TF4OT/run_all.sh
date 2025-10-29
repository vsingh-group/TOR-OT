#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_TF4OT

# run all configs
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

#python3 -m TF4OT.run_training  --config-name TF_100K_cost_fixed     > ./experiments/logs/train_TF_100K_cost_fixed.log 2>&1
#python3 -m TF4OT.run_inference --config-name TF_100K_cost_fixed     > ./experiments/logs/eval_TF_100K_cost_fixed.log 2>&1
#python3 -m TF4OT.run_training  --config-name TF_100K_cost_variable  > ./experiments/logs/train_TF_100K_cost_variable.log 2>&1
#python3 -m TF4OT.run_inference --config-name TF_100K_cost_variable  > ./experiments/logs/eval_TF_100K_cost_variable.log 2>&1

# python3 -m TF4OT.run_training  --config-name TF_1M_cost_fixed       > ./experiments/logs/train_TF_1M_cost_fixed.log 2>&1
# python3 -m TF4OT.run_inference --config-name TF_1M_cost_fixed       > ./experiments/logs/eval_TF_1M_cost_fixed.log 2>&1
python3 -m TF4OT.run_training  --config-name TF_1M_cost_variable    > ./experiments/logs/train_TF_1M_cost_variable.log 2>&1
#python3 -m TF4OT.run_inference --config-name TF_1M_cost_variable    > ./experiments/logs/eval_TF_1M_cost_variable.log 2>&1

#python3 -m TF4OT.run_training --config-name TF_10M_cost_fixed      > ./experiments/logs/train_TF_10M_cost_fixed.log 2>&1
python3 -m TF4OT.run_training  --config-name TF_10M_cost_variable   > ./experiments/logs/train_TF_10M_cost_variable.log 2>&1
#python3 -m TF4OT.run_inference --config-name TF_10M_cost_variable   > ./experiments/logs/eval_TF_10M_cost_variable.log 2>&1