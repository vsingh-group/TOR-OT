import numpy as np
from tabulate import tabulate
import os

model_dir = "experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean"

# Load Testset NPZ
f_can = np.load(os.path.join(model_dir, "eval_all_testset_1d_sqeuclidean_results_tot.npz"))
f_sink = np.load(os.path.join(model_dir, "eval_all_testset_1d_sqeuclidean_results_tot_sinkhorn.npz"))

mape_can = f_can['all_mape_sparse'].mean()
mape_sink = f_sink['all_mape_sparse'].mean()

otdiff_can = f_can['all_otcost_diff'].mean()
otdiff_sink = f_sink['all_otcost_diff'].mean()

maxviol_can = f_can['all_max_violation'].mean()
maxviol_sink = f_sink['all_max_violation'].mean()

table = [
    ["ToR-OT (Canonical)", f"{mape_can:.3e}", f"{otdiff_can:.3e}", f"{maxviol_can:.3e}"],
    ["ToR-OT + Sinkhorn", f"{mape_sink:.3e}", f"{otdiff_sink:.3e}", f"{maxviol_sink:.3e}"]
]
print("=== In-Domain Testset ===")
print(tabulate(table, headers=["Model", "MAPE", "OT Cost Diff", "Max Marginal Violation"], tablefmt="grid"))

# Also for OOD Pairs
f_pairs_can = np.load(os.path.join(model_dir, "eval_all_pairs_1d_sqeuclidean_results_tot.npy"), allow_pickle=True)
f_pairs_sink = np.load(os.path.join(model_dir, "eval_all_pairs_1d_sqeuclidean_results_tot_sinkhorn.npy"), allow_pickle=True)

mape_pairs_can = np.mean([r['mape_sparse'] for r in f_pairs_can])
mape_pairs_sink = np.mean([r['mape_sparse'] for r in f_pairs_sink])

otdiff_pairs_can = np.mean([r['otcost_diff'] for r in f_pairs_can])
otdiff_pairs_sink = np.mean([r['otcost_diff'] for r in f_pairs_sink])

maxviol_pairs_can = np.mean([r['max_violation'] for r in f_pairs_can])
maxviol_pairs_sink = np.mean([r['max_violation'] for r in f_pairs_sink])

table_ood = [
    ["ToR-OT (Canonical)", f"{mape_pairs_can:.3e}", f"{otdiff_pairs_can:.3e}", f"{maxviol_pairs_can:.3e}"],
    ["ToR-OT + Sinkhorn", f"{mape_pairs_sink:.3e}", f"{otdiff_pairs_sink:.3e}", f"{maxviol_pairs_sink:.3e}"]
]
print("\n=== Out-of-Domain Pairs ===")
print(tabulate(table_ood, headers=["Model", "MAPE", "OT Cost Diff", "Max Marginal Violation"], tablefmt="grid"))
