import numpy as np
import os

path = "/home/ivan/repos/TF4OT/experiments/cost_semivariable/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data600K_rowwise_costsemivariable_prec4_bins30_DigitTokenizer_epochs25_batch128/eval_all_pairs_1d_euclidean_results_tot.npy"

if os.path.exists(path):
    data = np.load(path, allow_pickle=True)
    print(f"Type: {type(data)}")
    print(f"Shape: {data.shape}")
    if len(data) > 0:
        print(f"First element type: {type(data[0])}")
        print(f"First element keys: {data[0].keys()}")
        print(f"First element sample: {data[0]}")
else:
    print(f"File not found: {path}")
