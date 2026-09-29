import numpy as np
import os

def load_and_print(path, method_name):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    data = np.load(path)
    if 'all_row_violation' in data and 'all_col_violation' in data:
        row_viol = data['all_row_violation']
        col_viol = data['all_col_violation']
        
        print(f"==== {method_name} ====")
        print(f"Mean Row Violation (Source Margin): {row_viol.mean():.6f}")
        print(f"Mean Col Violation (Target Margin): {col_viol.mean():.6f}")
        print(f"Asymmetry (Col - Row): {(col_viol.mean() - row_viol.mean()):.6f}")
    else:
        print(f"No violation arrays in {path}")

# Based on get_subfolder_name_from_cfg config
model_dir = "./experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean"

# I assume one is standard and one is sym. Let's check what's there.
npz_files = [f for f in os.listdir(model_dir) if f.endswith('.npz')]
print("Found files:", npz_files)

# Or maybe there's a specific naming convention?
for f in npz_files:
    load_and_print(os.path.join(model_dir, f), f)

