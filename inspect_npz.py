import numpy as np
import os

path1 = "TF4OT/experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean/eval_all_testset_1d_sqeuclidean_results_tot.npz"
path2 = "TF4OT/experiments/cost_semivariable/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data600K_rowwise_costsemivariable_prec4_bins30_DigitTokenizer_epochs25_batch128/eval_all_testset_1d_results_sqeuclidean.npz"

def inspect(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    try:
        data = np.load(path)
        print(f"\nKeys for {path}:")
        print(list(data.keys()))
        for key in list(data.keys())[:3]: # Print shape of first few items
             print(f"  {key}: {data[key].shape if hasattr(data[key], 'shape') else 'scalar'}")
             if key == 'results':
                 print(f"  Sample result item keys: {data[key][0].keys() if len(data[key]) > 0 else 'empty'}")

    except Exception as e:
        print(f"Error loading {path}: {e}")

inspect(path1)
inspect(path2)
