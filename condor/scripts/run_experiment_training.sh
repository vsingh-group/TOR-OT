#!/bin/bash

experiment_name=$1
num_gpu_devices=$2
cluster_id=$3
staging_model_directory="/staging/jaenmarquez/models"

echo "checking nvidia-smi:"
nvidia-smi

#export NETRC=/staging/jaenmarquez/.config/wandb/.netrc
#wandb login --relogin ***REMOVED***
export WANDB_API_KEY=***REMOVED***
export HF_HOME=$staging_model_directory
export HF_TOKEN="***REMOVED***"
export TRITON_CACHE_DIR="/staging/jaenmarquez/.cache/triton_cache"
export HF_DATASETS_CACHE="/staging/jaenmarquez/.cache/huggingface/dataset"



# copy data files from /staging to local workspace

data_file_name=ot_sample_data.tar.gz
data_dir_orig=/staging/jaenmarquez/datasets/decoding4ot/$data_file_name
data_dir_dest=./experiments/singletarget/data
echo "COPYING TRAINING DATA FILES:"
echo "  FROM: $data_dir_orig"
echo "  TO:   $data_dir_dest"
mkdir -p $data_dir_dest
#cp -r $data_dir_orig $data_dir_dest
rsync -ah --progress $data_dir_orig $data_dir_dest
echo $data_dir_dest
ls -l $data_dir_dest


# uncompress data files
echo "UNCOMPRESSING DATA FILES:"
tar -xzf $data_dir_dest/$data_file_name -C $data_dir_dest
echo "UNCOMPRESSED DATA FILES:"
echo $data_dir_dest
ls -l $data_dir_dest

# run training job
echo "RUNNING TRAINING JOB:"
python3 -m decoding4ot.train_regression_singletarget_hydra --config-name=$experiment_name

# copy results back to /staging
experiment_dir_output="./experiments/singletarget/models"
experiment_dir_output_staging="/staging/jaenmarquez/models/decoding4ot/singletarget"
echo "COPYING RESULTS BACK TO /STAGING:"
echo "  FROM: $experiment_dir_output"
ls -l $experiment_dir_output
echo "  TO:   $experiment_dir_output_staging"
rsync -ah --progress $experiment_dir_output/. $experiment_dir_output_staging
ls -l $experiment_dir_output_staging