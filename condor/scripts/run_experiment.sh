#!/bin/bash

experiment_name=$1
num_gpu_devices=$2
cluster_id=$3
username=moros2
#echo $username
echo "checking nvidia-smi:"
nvidia-smi

#export NETRC=/staging/jaenmarquez/.config/wandb/.netrc
#wandb login --relogin ***REMOVED***
export WANDB_API_KEY=***REMOVED***
#export HF_HOME=$staging_model_directory
export HF_TOKEN="***REMOVED***"
#export TRITON_CACHE_DIR="/staging/$username/.cache/triton_cache"
#export HF_DATASETS_CACHE="/staging/$username/.cache/huggingface/dataset"

# Run training only if it succeeds, copy results
if python3 -m TF4OT.run_training  --config-name "$experiment_name" then
    # copy results back to /staging
    experiment_dir_output="./experiments/cost*"
    experiment_dir_output_staging="/staging/$username/experiments"
    #mkdir -p "$experiment_dir_output_staging"
    echo "COPYING RESULTS BACK TO /STAGING:"
    echo "  FROM: $experiment_dir_output"
    ls -l $experiment_dir_output
    echo "  TO:   $experiment_dir_output_staging"
    rsync -ah --progress $experiment_dir_output/. "$experiment_dir_output_staging"
    ls -l "$experiment_dir_output_staging"
else
    echo "Training failed, not copying results to /staging" >&2
    exit 1
fi