# TF4OT: Transformers for Optimal Transport.
### Code for the paper: _"Transformers learn to approximately solve Optimal Transport"_.

## ⚙️ Environment Setup

### Setup python environment via conda
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create --name env_TF4OT python=3.11
conda activate env_TF4OT
pip3 install -e .
pip3 install -r requirements.txt
```

### Setup via docker
```bash
docker build -f image.dockerfile -t ivanjaenm/tf4ot:v1 . --push
```

## 🛢️ Synthetic data generator

### Run data generator (cpu)
```bash
python3 -m TF4OT.run_datagenerator --n_bins=10 --row_wise --metric='sqeuclidean' --cost_fixed  --dataset_size=100_000 --output_dir=./_otdata_100K_rowwise --upload --repo_name='ivanjaenm/otdata_100K_rowwise_cost_fixed'
python3 -m TF4OT.run_datagenerator --n_bins=10 --metric='sqeuclidean' --cost_fixed  --dataset_size=100_000 --output_dir=./_otdata_100K --upload --repo_name='ivanjaenm/otdata_100K_cost_fixed'
python3 -m TF4OT.run_datagenerator --n_bins=10 --row_wise --metric='sqeuclidean' --dataset_size=100_000 --output_dir=./_otdata_100K_rowwise_variableC --upload --repo_name='ivanjaenm/otdata_100K_rowwise_cost_variable'
python3 -m TF4OT.run_datagenerator --n_bins=10 --metric='sqeuclidean' --dataset_size=100_000 --output_dir=./_otdata_100K_variableC --upload --repo_name='ivanjaenm/otdata_100K_cost_variable'
```

## 🚀 Training Pipeline Overview

### Run training (gpu)
```bash
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_training --config-name TF_100K_cost_fixed
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m TF4OT.run_training --config-name TF_100K_cost_fixed
```

## 📊 Evaluation

### Run inference (gpu)
```bash
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_inference --config-name TF_100K_cost_fixed
```

### Run tests
```bash
python3 -m TF4OT.tokenizers_test
python3 -m TF4OT.vocabs_test
```

### Run all experiments in the background
```bash
./TF4OT/run_all.sh &
```

## 🌟 Citation
If you find this work is helpful to your research, please consider citing our paper:
```
@inproceedings{,
  title={Transformers learn to approximately solve Optimal Transport},
  author={},
  booktitle={ArXiv},
  year={2025}
}
```