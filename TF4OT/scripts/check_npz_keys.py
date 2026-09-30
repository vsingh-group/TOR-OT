import numpy as np
import os

path = "/home/ivan/repos/TF4OT/experiments/cost_semivariable/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data600K_rowwise_costsemivariable_prec4_bins30_DigitTokenizer_epochs25_batch128/eval_all_testset_1d_euclidean_results_tot.npz"

if os.path.exists(path):
    data = np.load(path)
    print("Keys found:", list(data.keys()))
    if 'all_max_violation' in data:
        print("Max violation shape:", data['all_max_violation'].shape)
        print("Max violation mean:", np.mean(data['all_max_violation']))
else:
    print(f"File not found: {path}")
