import nbformat
import numpy as np
import os

def extract_table_mean(nb_path, header_keyword):
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and cell.source.startswith('### ' + header_keyword):
            # Parse the markdown table
            lines = cell.source.strip().split('\n')
            
            # Find where the table starts
            data_start_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('|:---') or line.startswith('|---'):
                    data_start_idx = i + 1
                    break
            
            # Extract floats
            vals = []
            for line in lines[data_start_idx:]:
                if not line.strip() or not line.startswith('|'): continue
                # Split by | and ignore first (row name) and last (empty)
                cols = [c.strip() for c in line.split('|')[2:-1]]
                for c in cols:
                    try:
                        vals.append(float(c))
                    except ValueError:
                        pass
            return np.mean(vals)
    return None

model_dir = "experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean"
can_nb = os.path.join(model_dir, "eval_all_pairs_1d_sqeuclidean.ipynb")
sym_nb = os.path.join(model_dir, "eval_all_pairs_1d_sqeuclidean_sym.ipynb")

mape_can = extract_table_mean(can_nb, "MAPE (Sparse)")
mape_sym = extract_table_mean(sym_nb, "MAPE (Sparse)")

viol_can = extract_table_mean(can_nb, "Max Marginal Violation")
viol_sym = extract_table_mean(sym_nb, "Max Marginal Violation")

time_can = extract_table_mean(can_nb, "Run Time")
time_sym = extract_table_mean(sym_nb, "Run Time")

print("--- CANONICAL NOTEBOOK EXPORT ---")
print(f"MAPE: {mape_can}")
print(f"Viol: {viol_can}")
print(f"Time: {time_can}")

print("\n--- SYMMETRIC NOTEBOOK EXPORT ---")
print(f"MAPE: {mape_sym}")
print(f"Viol: {viol_sym}")
print(f"Time: {time_sym}")

print(f"\nAsymmetry Decrease: {(viol_can - viol_sym)*100/viol_can:.2f}%")
print(f"MAPE Decrease (Accuracy Lift): {(mape_can - mape_sym)*100/mape_can:.2f}%")
print(f"Time Multiplier: {time_sym/time_can:.2f}x")
