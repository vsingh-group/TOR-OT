# TF4OT: Transformers for Optimal Transport.
### Code for the paper: _"Transformers learn to approximately solve Optimal Transport"_.

### Setup python environment via conda
```console
source ~/anaconda3/etc/profile.d/conda.sh
conda create --name env_TF4OT python=3.11
conda activate env_TF4OT
pip3 install -e .
pip3 install -r requirements.txt
```

### Run data generator (runs on cpu)
```console
python3 -m TF4OT.run_datagenerator --config-name small
python3 -m TF4OT.run_datagenerator --n_bins=10 --row_wise --metric='sqeuclidean' --cost_fixed  --dataset_size=100_000 --output_dir=./_otdata_100K_rowwise --upload --repo_name='ivanjaenm/otdata_100K_rowwise'
python3 -m TF4OT.run_datagenerator --n_bins=10 --metric='sqeuclidean' --cost_fixed  --dataset_size=100_000 --output_dir=./_otdata_100K --upload --repo_name='ivanjaenm/otdata_100K'
python3 -m TF4OT.run_datagenerator --n_bins=10 --row_wise --metric='sqeuclidean' --dataset_size=100_000 --output_dir=./_otdata_100K_rowwise_variableC --upload --repo_name='ivanjaenm/otdata_100K_rowwise_variableC'
python3 -m TF4OT.run_datagenerator --n_bins=10 --metric='sqeuclidean' --dataset_size=100_000 --output_dir=./_otdata_100K_variableC --upload --repo_name='ivanjaenm/otdata_100K_variableC'
```

### Run training (runs on gpu)
```console
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_training --config-name test_row_wise
CUDA_VISIBLE_DEVICES=2,5,6,7 python3 -m TF4OT.run_training --config-name TF_100K
```

### Run inference
```console
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_inference --config-name small
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_inference --config-name test_row_wise
```

### Run tests
```console
python3 -m TF4OT.tokenizers_test
python3 -m TF4OT.vocabs_test
```