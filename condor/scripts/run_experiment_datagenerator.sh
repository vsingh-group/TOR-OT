#!/bin/bash

experiment_name=$1
staging_data_directory=/staging/jaenmarquez/datasets
num_cpus=$3
cluster_id=$4

export TRITON_CACHE_DIR="/staging/jaenmarquez/.cache/triton_cache"
export HF_DATASETS_CACHE="/staging/jaenmarquez/.cache/huggingface/dataset"
export HF_HUB_CACHE="/staging/jaenmarquez/.cache/huggingface/dataset/hub"

# call generate dataset script
echo "running dataset generation job:"
python3 -m optformer.decoding_regression.run_datagenerator
# copy results back to /staging
echo "Compressing and copying results back to /staging:"

SOURCE_DATA_DIR="./decoder4ot/experiments/singletarget/ot_sample_data/"
#SOURCE_PLOT_DIR="./decoder4ot/experiments/singletarget/ot_sample_plots/"
TARGET_DATA_DIR=${staging_data_directory}/decoding4ot/
mkdir -p $TARGET_DATA_DIR

# Compress data and plot directories
DATA_ARCHIVE="ot_sample_data.tar.gz"
#PLOT_ARCHIVE="ot_sample_plots.tar.gz"

echo "Creating compressed archives..."
tar -czf $DATA_ARCHIVE -C "$SOURCE_DATA_DIR" .
#tar -czf $PLOT_ARCHIVE -C "$SOURCE_PLOT_DIR" .
echo "Copying compressed archives to $TARGET_DATA_DIR"
cp -f $DATA_ARCHIVE $TARGET_DATA_DIR
#cp -f $PLOT_ARCHIVE $TARGET_DATA_DIR
echo "Done."

#copy the file from local to remote
#place the file into: /staging/jaenmarquez/datasets/decoding4ot/
scp ot_sample_data.tar.gz jaenmarquez@ap2002.chtc.wisc.edu:/staging/jaenmarquez/