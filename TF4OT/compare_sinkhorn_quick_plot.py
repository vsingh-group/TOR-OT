import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

mae_sym = 0.002562
mae_sink = 0.00008167
maxviol_sym = 0.01197
maxviol_sink = 0.0001704

mae_pairs_sym = 0.003325
mae_pairs_sink = 0.0004531
maxviol_pairs_sym = 0.02185
maxviol_pairs_sink = 0.03173

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

x = np.arange(2)
width = 0.35

# In-Domain
ax1.bar(x - width/2, [mae_sym, mae_sink], width, label='MAE', color='tab:blue')
ax1.bar(x + width/2, [maxviol_sym, maxviol_sink], width, label='Max Marginal Viol', color='tab:orange')
ax1.set_xticks(x)
ax1.set_xticklabels(["ToR-OT\n(Symmetric)", "ToR-OT\n+ Sinkhorn"])
ax1.set_title('In-Domain Testset')
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)
ax1.legend()

# OOD
ax2.bar(x - width/2, [mae_pairs_sym, mae_pairs_sink], width, label='MAE', color='tab:blue')
ax2.bar(x + width/2, [maxviol_pairs_sym, maxviol_pairs_sink], width, label='Max Marginal Viol', color='tab:orange')
ax2.set_xticks(x)
ax2.set_xticklabels(["ToR-OT\n(Symmetric)", "ToR-OT\n+ Sinkhorn"])
ax2.set_title('Out-of-Domain Pairs')
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('sinkhorn_comparison.png')
print("Plot saved to sinkhorn_comparison.png")
