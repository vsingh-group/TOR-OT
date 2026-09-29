import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import os
import time

model_dir = "experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean"

# Load Testset NPZ
print("Loading in-domain results...")
f_can = np.load(os.path.join(model_dir, "eval_all_testset_1d_sqeuclidean_results_tot.npz"))
f_sink = np.load(os.path.join(model_dir, "eval_all_testset_1d_sqeuclidean_results_tot_sinkhorn.npz"))

mape_can = float(f_can['all_mape_sparse'].mean())
mape_sink = float(f_sink['all_mape_sparse'].mean())

otdiff_can = float(f_can['all_otcost_diff'].mean())
otdiff_sink = float(f_sink['all_otcost_diff'].mean())

maxviol_can = float(f_can['all_max_violation'].mean())
maxviol_sink = float(f_sink['all_max_violation'].mean())

testset_speeds = np.load(os.path.join(model_dir, "testset_speeds.npy"), allow_pickle=True).item()
time_testset_can = float(testset_speeds['can'])
time_testset_sink = float(testset_speeds['sink'])

table = [
    ["ToR-OT (Canonical)", f"{mape_can:.3e}", f"{otdiff_can:.3e}", f"{maxviol_can:.3e}", f"{time_testset_can:.3e}"],
    ["ToR-OT + Sinkhorn", f"{mape_sink:.3e}", f"{otdiff_sink:.3e}", f"{maxviol_sink:.3e}", f"{time_testset_sink:.3e}"]
]
print("\n=== In-Domain Testset ===")
print(tabulate(table, headers=["Model", "MAPE", "OT Cost Diff", "Max Marginal Violation", "Run Time"], tablefmt="grid"))

# Also for OOD Pairs
print("\nLoading out-of-domain results...")
f_pairs_can = np.load(os.path.join(model_dir, "eval_all_pairs_1d_sqeuclidean_results_tot.npy"), allow_pickle=True)
f_pairs_sink = np.load(os.path.join(model_dir, "eval_all_pairs_1d_sqeuclidean_results_tot_sinkhorn.npy"), allow_pickle=True)

mape_pairs_can = float(np.mean([r['mape_sparse'] for r in f_pairs_can]))
mape_pairs_sink = float(np.mean([r['mape_sparse'] for r in f_pairs_sink]))

otdiff_pairs_can = float(np.mean([r['otcost_diff'] for r in f_pairs_can]))
otdiff_pairs_sink = float(np.mean([r['otcost_diff'] for r in f_pairs_sink]))

maxviol_pairs_can = float(np.mean([r['max_violation'] for r in f_pairs_can]))
maxviol_pairs_sink = float(np.mean([r['max_violation'] for r in f_pairs_sink]))

time_pairs_can = float(np.mean([r['run_time'] for r in f_pairs_can]))
time_pairs_sink = float(np.mean([r['run_time'] for r in f_pairs_sink]))

table_ood = [
    ["ToR-OT (Canonical)", f"{mape_pairs_can:.3e}", f"{otdiff_pairs_can:.3e}", f"{maxviol_pairs_can:.3e}", f"{time_pairs_can:.3e}"],
    ["ToR-OT + Sinkhorn", f"{mape_pairs_sink:.3e}", f"{otdiff_pairs_sink:.3e}", f"{maxviol_pairs_sink:.3e}", f"{time_pairs_sink:.3e}"]
]
print("\n=== Out-of-Domain Pairs ===")
print(tabulate(table_ood, headers=["Model", "MAPE", "OT Cost Diff", "Max Marginal Violation", "Run Time"], tablefmt="grid"))


print("\nGenerating plots...", flush=True)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

x = np.arange(2)
width = 0.35

# === Plot 1: Accuracy (MAPE) ===
ax1.bar(x - width/2, [mape_can, mape_pairs_can], width, label='ToR-OT (Canonical)', color='tab:blue')
ax1.bar(x + width/2, [mape_sink, mape_pairs_sink], width, label='ToR-OT + Sinkhorn', color='tab:orange')
ax1.set_xticks(x)
ax1.set_xticklabels(["In-Domain Testset", "Out-of-Domain Pairs"])
ax1.set_title('Accuracy (Prediction Error)')
ax1.set_ylabel('Mean Absolute Percentage Error (MAPE)')
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)
ax1.legend()


# === Plot 2: Violations ===
ax2.bar(x - width/2, [maxviol_can, maxviol_pairs_can], width, label='ToR-OT (Canonical)', color='tab:blue')
ax2.bar(x + width/2, [maxviol_sink, maxviol_pairs_sink], width, label='ToR-OT + Sinkhorn', color='tab:orange')
ax2.set_xticks(x)
ax2.set_xticklabels(["In-Domain Testset", "Out-of-Domain Pairs"])
ax2.set_title('Constraint Violations')
ax2.set_ylabel('Max Marginal Violation')
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3)
ax2.legend()


# === Plot 3: Execution Time ===
x_time = np.arange(2)
ax3.bar(x_time - width/2, [time_testset_can, time_pairs_can], width, label='ToR-OT (Canonical)', color='tab:blue')
ax3.bar(x_time + width/2, [time_testset_sink, time_pairs_sink], width, label='ToR-OT + Sinkhorn', color='tab:orange')
ax3.set_xticks(x_time)
ax3.set_xticklabels(["In-Domain Testset", "Out-of-Domain Pairs"])
ax3.set_title('Inference Speed')
ax3.set_ylabel('Execution Time per instance (seconds)')
ax3.grid(axis='y', alpha=0.3)
ax3.legend()

plt.tight_layout()
out_path = os.path.join(model_dir, "sinkhorn_comparison.png")
plt.savefig(out_path)
print("Plot saved to", out_path, flush=True)
