import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

model_dir = "experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean"

print("\nGenerating Rebuttal plots for Qwen 7B Comparison...", flush=True)

# Data given from prompt
models = ['Qwen2.5-7B-Instruct\n(5-shot ICL)', 'NeuralOT\n(Baseline)', 'ToR-OT\n(Single Pass)', 'ToR-OT\n(Joint / Sym Avg)']
mape_scores = [2.08, 1.35, 0.20, 0.10]
colors = ['#dc2626', '#f59e0b', '#3b82f6', '#10b981']

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(models, mape_scores, color=colors, edgecolor='black', zorder=3)

# Add numeric annotations on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Mean Absolute Percentage Error (MAPE)', fontsize=12, fontweight='bold')
ax.set_title('In-Context Learning vs. Native Finetuning\n(1D Test Set Performance)', fontsize=14, fontweight='bold')

# Y-axis tweaks
ax.set_ylim(0, 2.5)
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# Custom text box highlighting the 4-decimal precision limit parity
textstr = "ToR-OT models natively encode output tokens\nat identical 4-digit precision (prec4)"
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
out_path = os.path.join(model_dir, "rebuttal_llm_comparison.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print("Plot saved to", out_path, flush=True)
