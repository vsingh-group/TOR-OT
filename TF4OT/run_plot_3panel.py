import numpy as np
import os
import time

print("Loading data...", flush=True)
model_dir = "experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean"

# Load In-Domain NPZ
f_can = np.load(os.path.join(model_dir, "eval_all_testset_1d_sqeuclidean_results_tot.npz"))
f_sink = np.load(os.path.join(model_dir, "eval_all_testset_1d_sqeuclidean_results_tot_sinkhorn.npz"))

mape_can = float(f_can['all_mape_sparse'].mean())
mape_sink = float(f_sink['all_mape_sparse'].mean())

maxviol_can = float(f_can['all_max_violation'].mean())
maxviol_sink = float(f_sink['all_max_violation'].mean())

# Load OOD NPY
f_pairs_can = np.load(os.path.join(model_dir, "eval_all_pairs_1d_sqeuclidean_results_tot.npy"), allow_pickle=True)
f_pairs_sink = np.load(os.path.join(model_dir, "eval_all_pairs_1d_sqeuclidean_results_tot_sinkhorn.npy"), allow_pickle=True)

mape_pairs_can = float(np.mean([r['mape_sparse'] for r in f_pairs_can]))
mape_pairs_sink = float(np.mean([r['mape_sparse'] for r in f_pairs_sink]))

maxviol_pairs_can = float(np.mean([r['max_violation'] for r in f_pairs_can]))
maxviol_pairs_sink = float(np.mean([r['max_violation'] for r in f_pairs_sink]))

time_pairs_can = float(np.mean([r['run_time'] for r in f_pairs_can]))
time_pairs_sink = float(np.mean([r['run_time'] for r in f_pairs_sink]))

print("Plotting...", flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

x = np.arange(2)
width = 0.35

# === Plot 1: Violations ===
ax1.bar(x - width/2, [maxviol_can, maxviol_pairs_can], width, label='ToR-OT (Canonical)', color='tab:blue')
ax1.bar(x + width/2, [maxviol_sink, maxviol_pairs_sink], width, label='ToR-OT + Sinkhorn', color='tab:orange')
ax1.set_xticks(x)
ax1.set_xticklabels(["In-Domain", "Out-of-Domain"])
ax1.set_title('Constraint Violations')
ax1.set_ylabel('Max Marginal Violation')
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)
ax1.legend()

# === Plot 2: Accuracy (MAPE) ===
ax2.bar(x - width/2, [mape_can, mape_pairs_can], width, label='ToR-OT (Canonical)', color='tab:blue')
ax2.bar(x + width/2, [mape_sink, mape_pairs_sink], width, label='ToR-OT + Sinkhorn', color='tab:orange')
ax2.set_xticks(x)
ax2.set_xticklabels(["In-Domain", "Out-of-Domain"])
ax2.set_title('Accuracy (Prediction Error)')
ax2.set_ylabel('Mean Absolute Percentage Error (MAPE)')
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3)
ax2.legend()

# === Plot 3: Execution Time ===
x_time = np.array([0])
ax3.bar(x_time - width/2, [time_pairs_can], width, label='ToR-OT (Canonical)', color='tab:blue')
ax3.bar(x_time + width/2, [time_pairs_sink], width, label='ToR-OT + Sinkhorn', color='tab:orange')
ax3.set_xticks(x_time)
ax3.set_xticklabels(["Out-of-Domain"])
ax3.set_title('Execution Time')
ax3.set_ylabel('Time per inference (seconds)')
ax3.grid(axis='y', alpha=0.3)
ax3.legend()

plt.tight_layout()
out_path = os.path.join(model_dir, "sinkhorn_comparison_3panel.png")
plt.savefig(out_path)
print("Done saving to:", out_path, flush=True)
