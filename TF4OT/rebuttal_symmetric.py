import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import os

model_dir = "experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean"

# Load OOD  for both Canonical and Symmetric
f_can = np.load(os.path.join(model_dir, "eval_all_pairs_1d_sqeuclidean_results_tot.npy"), allow_pickle=True)
f_sym = np.load(os.path.join(model_dir, "eval_all_pairs_1d_sqeuclidean_results_tot_sym.npy"), allow_pickle=True)

# Compute Testset speed (as baseline)
testset_speeds = np.load(os.path.join(model_dir, "testset_speeds.npy"), allow_pickle=True).item()
time_testset_can = float(testset_speeds['can'])

mape_can = float(np.mean([r['mape_sparse'] for r in f_can]))
mape_sym = float(np.mean([r['mape_sparse'] for r in f_sym]))

maxviol_can = float(np.mean([r['max_violation'] for r in f_can]))
maxviol_sym = float(np.mean([r['max_violation'] for r in f_sym]))

time_can = float(np.mean([r['run_time'] for r in f_can]))
time_sym = float(np.mean([r['run_time'] for r in f_sym]))

print("\nGenerating Rebuttal plots...", flush=True)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

x = np.arange(1)
width = 0.35

# Compute dynamic percentages
viol_pct = (maxviol_can - maxviol_sym) / maxviol_can * 100
mape_pct = (mape_can - mape_sym) / mape_can * 100
time_pct = (time_sym - time_can) / time_can * 100

# === Plot 1: Marginal Violations (Asymmetry Drop) ===
ax1.bar(x - width/2, [maxviol_can], width, label='Canonical (Single Pass)', color='tab:blue')
ax1.bar(x + width/2, [maxviol_sym], width, label='Symmetric (Dual Pass Avg)', color='tab:green')
ax1.set_xticks(x)
ax1.set_xticklabels(["Out-of-Domain"])
ax1.set_title(rf'Asymmetry (Violations)\n$-{viol_pct:.1f}\%$ Decrease', fontweight='bold')
ax1.set_ylabel('Max Marginal Violation')
ax1.grid(axis='y', alpha=0.3)
ax1.legend()


# === Plot 2: Accuracy (MAPE Drop) ===
ax2.bar(x - width/2, [mape_can], width, label='Canonical (Single Pass)', color='tab:blue')
ax2.bar(x + width/2, [mape_sym], width, label='Symmetric (Dual Pass Avg)', color='tab:green')
ax2.set_xticks(x)
ax2.set_xticklabels(["Out-of-Domain"])
ax2.set_title(rf'Accuracy Error (MAPE)\n$-{mape_pct:.1f}\%$ Improvement', fontweight='bold')
ax2.set_ylabel('Mean Absolute Percentage Error')
ax2.grid(axis='y', alpha=0.3)
ax2.legend()


# === Plot 3: Execution Time ===
ax3.bar(x - width/2, [time_can], width, label='Canonical (Single Pass)', color='tab:blue')
ax3.bar(x + width/2, [time_sym], width, label='Symmetric (Dual Pass Avg)', color='tab:green')
ax3.set_xticks(x)
ax3.set_xticklabels(["Out-of-Domain"])
ax3.set_title(rf'Inference Time\n$+{time_pct:.0f}\%$ Increase', fontweight='bold')
ax3.set_ylabel('Execution Time per instance (seconds)')
ax3.grid(axis='y', alpha=0.3)
ax3.legend()

plt.suptitle("Impact of Symmetric Averaging on Out-of-Domain Pairs", fontsize=16, y=1.05)
plt.tight_layout()
out_path = os.path.join(model_dir, "rebuttal_symmetric_comparison.png")
plt.savefig(out_path, bbox_inches='tight')
print("Plot saved to", out_path, flush=True)
