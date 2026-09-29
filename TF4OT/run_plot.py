import numpy as np
import os
import sys

print("Loading data...", flush=True)
model_dir = "experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean"

# Load Testset NPZ
f_can = np.load(os.path.join(model_dir, "eval_all_testset_1d_sqeuclidean_results_tot.npz"))
f_sink = np.load(os.path.join(model_dir, "eval_all_testset_1d_sqeuclidean_results_tot_sinkhorn.npz"))

mape_can = float(f_can['all_mape_sparse'].mean())
mape_sink = float(f_sink['all_mape_sparse'].mean())

maxviol_can = float(f_can['all_max_violation'].mean())
maxviol_sink = float(f_sink['all_max_violation'].mean())

# OOD
f_pairs_can = np.load(os.path.join(model_dir, "eval_all_pairs_1d_sqeuclidean_results_tot.npy"), allow_pickle=True)
f_pairs_sink = np.load(os.path.join(model_dir, "eval_all_pairs_1d_sqeuclidean_results_tot_sinkhorn.npy"), allow_pickle=True)

mape_pairs_can = float(np.mean([r['mape_sparse'] for r in f_pairs_can]))
mape_pairs_sink = float(np.mean([r['mape_sparse'] for r in f_pairs_sink]))

maxviol_pairs_can = float(np.mean([r['max_violation'] for r in f_pairs_can]))
maxviol_pairs_sink = float(np.mean([r['max_violation'] for r in f_pairs_sink]))

print("Plotting...", flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

x = np.arange(2)
width = 0.35

# In-Domain
ax1.bar(x - width/2, [mape_can, mape_sink], width, label='MAPE', color='tab:blue')
ax1.bar(x + width/2, [maxviol_can, maxviol_sink], width, label='Max Marginal Viol', color='tab:orange')
ax1.set_xticks(x)
ax1.set_xticklabels(["ToR-OT\n(Canonical)", "ToR-OT\n+ Sinkhorn"])
ax1.set_title('In-Domain Testset')
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)
ax1.legend()

# OOD
ax2.bar(x - width/2, [mape_pairs_can, mape_pairs_sink], width, label='MAPE', color='tab:blue')
ax2.bar(x + width/2, [maxviol_pairs_can, maxviol_pairs_sink], width, label='Max Marginal Viol', color='tab:orange')
ax2.set_xticks(x)
ax2.set_xticklabels(["ToR-OT\n(Canonical)", "ToR-OT\n+ Sinkhorn"])
ax2.set_title('Out-of-Domain Pairs')
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3)
ax2.legend()

plt.tight_layout()
out_path = os.path.join(model_dir, "sinkhorn_comparison.png")
plt.savefig(out_path)
print("Done!", flush=True)
