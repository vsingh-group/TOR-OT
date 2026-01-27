
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import re

# ==========================================
# Configuration and Constants
# ==========================================

# Paths from Notebook
EXPERIMENT_DIR_INDIV = "../../../experiments/cost_fixed/"
EXPERIMENT_DIR_JOINT = "../../../experiments/cost_semivariable/"
# Adjust paths to be absolute or relative to where we run the script. 
# We run from repo root `/home/ivan/repos/TF4OT`.
# The notebook was in `TF4OT/evals/plots`.
# So `../../../experiments` translates to `experiments` from repo root?
# Let's verify: Notebook is in `TF4OT/evals/plots`
# `../../..` -> `TF4OT/evals/plots/../../..` = `(root)`
# Yes, `../../../` from notebook dir is the repo root.
# But when running python script from repo root, we should use `experiments/...`

# Correct paths relative to repo root:
EXPERIMENT_DIR_INDIV_ROOT = "experiments/cost_fixed/"
EXPERIMENT_DIR_JOINT_ROOT = "experiments/cost_semivariable/"

METRICS = ['cityblock', 'euclidean', 'sqeuclidean', 'minkowski', 'chebyshev', 'canberra']

MODEL_INDIV_TEMPLATE = "model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_{}"
MODEL_JOINT_NAME     = "model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data600K_rowwise_costsemivariable_prec4_bins30_DigitTokenizer_epochs25_batch128"

# LaTeX Labels
METRIC_LABELS = {
    'cityblock':   r'$L_1$',
    'euclidean':   r'$L_2$',
    'sqeuclidean': r'$L_2^2$',
    'minkowski':   r'$L_3$',
    'chebyshev':   r'$L_\infty$',
    'canberra':    r'Canberra',
}

METRIC_KEY = 'all_mape_sparse' 

summary_improvements = {}
detailed_data = {} # {metric: {'tot': [], 'not': []}}

print(f"\n{'='*40}\nLoading Data for All Metrics\n{'='*40}")

# -----------------------------\n
# 1. Load Data Loop\n
# -----------------------------\n
for metric in METRICS:
    # Paths (Updated to .npz and eval_all_testset from aggregated results)\n
    dir_joint = os.path.join(EXPERIMENT_DIR_JOINT_ROOT, MODEL_JOINT_NAME)
    path_tot = os.path.join(dir_joint, f"eval_all_testset_1d_{metric}_results_tot.npz")
    
    dir_indiv = os.path.join(EXPERIMENT_DIR_INDIV_ROOT, MODEL_INDIV_TEMPLATE.format(metric))
    path_not = os.path.join(dir_indiv, f"eval_all_testset_1d_{metric}_results_not.npz")

    if not os.path.exists(path_tot) or not os.path.exists(path_not):
        print(f"Skipping {metric}: File not found.\n  TOT: {path_tot}\n  NOT: {path_not}")
        continue

    try:
        # Load .npz
        with np.load(path_tot) as f_tot:
             keys = list(f_tot.keys())
             key_to_use = METRIC_KEY
             if key_to_use not in keys:
                 if 'mape' in keys: key_to_use = 'mape'
                 else:
                     print(f"Desired key {METRIC_KEY} not found in {keys}")
                     continue
             vals_tot = f_tot[key_to_use]
            
        with np.load(path_not) as f_not:
             key_to_use_not = METRIC_KEY
             keys_not = list(f_not.keys())
             if key_to_use_not not in keys_not:
                 if 'mape' in keys_not: key_to_use_not = 'mape'
                 else: continue
             vals_not = f_not[key_to_use_not]
            
        # Basic Validation\n
        if len(vals_tot) == 0:
            print(f"Skipping {metric}: Empirical data empty.")
            continue
            
        # Store for detailed plots\n
        detailed_data[metric] = {'tot': vals_tot, 'not': vals_not}
        
        # Calculate Mean Improvement\n
        mean_tot = np.mean(vals_tot)
        mean_not = np.mean(vals_not)
        
        # Improvement: (Baseline - New) / Baseline\n
        if mean_not > 1e-12:
            imp_pct = 100.0 * (mean_not - mean_tot) / mean_not
        else:
            imp_pct = 0.0
            
        summary_improvements[metric] = imp_pct
        print(f"Loaded {metric}: Improvement = {imp_pct:.2f}% (N={len(vals_tot)})")
        
    except Exception as e:
        print(f"Error loading {metric}: {e}")

# Global Font Settings for larger ticks
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20})

# -----------------------------\n
# 2. Summary Improvement Plot (Vertical)\n
# -----------------------------\n
if summary_improvements:
    plt.figure(figsize=(7, 5))

    metrics_sorted = sorted(summary_improvements.keys(),
                            key=lambda x: summary_improvements[x],
                            reverse=True)

    vals = [summary_improvements[m] for m in metrics_sorted]
    labels = [METRIC_LABELS.get(m, m) for m in metrics_sorted]

    colors = ['lightgreen' if v >= 0 else 'red' for v in vals]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(labels, vals, color=colors, width=0.5, edgecolor='black', linewidth=1.5)

    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Mean Improvement (%)', fontsize=22)
    # ax.set_title('Average Improvement by Cost Metric (MAPE)', fontsize=18)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

    plt.tight_layout()

    out_summary = "TF4OT/evals/plots/img/1d_improvement_summary_mape.png"
    os.makedirs(os.path.dirname(out_summary), exist_ok=True)
    plt.savefig(out_summary, bbox_inches="tight", dpi=300)
    print(f"Saved Summary Plot: {out_summary}")
    plt.close(fig)
else:
    print("No data available for summary plot.")

# -----------------------------\n
# 3. Detailed Visualizations\n
# -----------------------------\n
if detailed_data:
    N_metrics = len(detailed_data)
    
    # A. Comparative Box Plots\n
    plt.figure(figsize=(14, 8))
    
    plot_data = []
    for metric, d in detailed_data.items():
        label = METRIC_LABELS.get(metric, metric)
        for v in d['tot']:
            plot_data.append({'Metric': label, 'Error': v, 'Model': 'JOINT (TOT)'})
        for v in d['not']:
            plot_data.append({'Metric': label, 'Error': v, 'Model': 'INDIV (NOT)'})
            
    df_box = pd.DataFrame(plot_data)
    
    sns.boxplot(data=df_box, x='Metric', y='Error', hue='Model')
    plt.title(f'Error Distribution Comparison (MAPE)', fontsize=18)
    plt.ylabel('Error (Log Scale)', fontsize=16)
    plt.xlabel('Metric', fontsize=16)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    out_box = f"TF4OT/evals/plots/img/1d_boxplot_comparison_mape.png"
    plt.savefig(out_box, bbox_inches="tight", dpi=300)
    print(f"Saved Box Plot: {out_box}")
    plt.close()
    
    # B. Scatter and CDF Plots\n
    fig, axes = plt.subplots(N_metrics, 2, figsize=(14, 5 * N_metrics))
    if N_metrics == 1: axes = np.array([axes])
    elif len(axes.shape) == 1: pass 
    
    for idx, (metric, d) in enumerate(detailed_data.items()):
        ax_scatter = axes[idx][0]
        ax_cdf = axes[idx][1]
        
        label = METRIC_LABELS.get(metric, metric)
        
        x_not = d['not']
        y_tot = d['tot']
        
        # Ensure lengths match for scatter plot
        min_len = min(len(x_not), len(y_tot))
        if len(x_not) != len(y_tot):
            print(f"Warning: Length mismatch for {metric}. Truncating to {min_len}.")
            x_not = x_not[:min_len]
            y_tot = y_tot[:min_len]
        
        sns.scatterplot(x=x_not, y=y_tot, ax=ax_scatter, alpha=0.7, edgecolor='k')
        
        min_val = min(np.min(x_not), np.min(y_tot))
        max_val = max(np.max(x_not), np.max(y_tot))
        buffer = (max_val - min_val) * 0.05 if max_val != min_val else 0.1
        
        ax_scatter.plot([min_val-buffer, max_val+buffer], [min_val-buffer, max_val+buffer], 
                        'r--', label='Parity (y=x)')
        
        ax_scatter.set_xlabel(f'Baseline Error', fontsize=14)
        ax_scatter.set_ylabel(f'Joint Model Error', fontsize=14)
        ax_scatter.set_title(f'{label}: Direct Comparison', fontsize=16)
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.tick_params(axis='both', which='major', labelsize=12)
        
        # CDF Plot\n
        sns.ecdfplot(data=x_not, ax=ax_cdf, label='INDIV (NOT)', color='red', linewidth=2)
        sns.ecdfplot(data=y_tot, ax=ax_cdf, label='JOINT (TOT)', color='green', linewidth=2)
        ax_cdf.set_xlabel(f'Error', fontsize=14)
        ax_cdf.set_ylabel('Proportion', fontsize=14)
        ax_cdf.set_title(f'{label}: CDF Comparison', fontsize=16)
        ax_cdf.legend()
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.tick_params(axis='both', which='major', labelsize=12)
        
    plt.tight_layout()
    out_detailed = f"TF4OT/evals/plots/img/1d_detailed_distributions_mape.png"
    plt.savefig(out_detailed, bbox_inches="tight", dpi=300)
    print(f"Saved Detailed Plots: {out_detailed}")
    plt.close()
else:
    print("No detailed data to plot.")

# -----------------------------
# 4. Amortized Runtime Comparison
# -----------------------------
print(f"\n{'='*40}\nGenerating Amortized Runtime Plot\n{'='*40}")

T_TRAIN_HOURS = 72
T_TRAIN_SEC = T_TRAIN_HOURS * 3600

# User specified T_infer=1s for amortized method (TOT)
T_TOT_INFER_SEC = 1.0 

# Estimated runtime for Neural OT (NOT) based on max_steps=2501 in toy code.
# Assuming ~10-15ms per step => ~30 seconds per sample.
T_NOT_SEC = 30.0 


# -----------------------------
# 4. Amortized Runtime Comparison
# -----------------------------
print(f"\n{'='*40}\nGenerating Amortized Runtime Plot\n{'='*40}")

T_TRAIN_HOURS = 72
T_TRAIN_SEC = T_TRAIN_HOURS * 3600

# User specified T_infer=1s for amortized method (TOT)
T_TOT_INFER_SEC = 1.0 

# Estimated runtime for Neural OT (NOT) based on max_steps=2501 in toy code.
# Assuming ~10-15ms per step => ~30 seconds per sample.
T_NOT_SEC = 30.0 


# -----------------------------
# 4. Amortized Runtime Comparison
# -----------------------------
print(f"\n{'='*40}\nGenerating Amortized Runtime Plot\n{'='*40}")

T_TRAIN_HOURS = 72
T_TRAIN_SEC = T_TRAIN_HOURS * 3600

# User specified T_infer=1s for amortized method (TOT)
T_TOT_INFER_SEC = 1.0 

# Estimated runtime for Neural OT (NOT) based on max_steps=2501 in toy code.
# Assuming ~10-15ms per step => ~30 seconds per sample.
T_NOT_SEC = 2500.0 

# Determine break-even N
# T_TRAIN/N + T_TOT_INFER = T_NOT

# Determine break-even N
# T_TRAIN/N + T_TOT_INFER = T_NOT
if T_NOT_SEC > T_TOT_INFER_SEC:
    N_break_even = T_TRAIN_SEC / (T_NOT_SEC - T_TOT_INFER_SEC)
else:
    N_break_even = 1e6 # Fallback

# --- Plot 1: Log Scale ---
min_exp = 3  # 10^3
max_exp = int(np.log10(N_break_even)) + 2 
N_values_log = np.logspace(min_exp, max_exp, 100)

avg_time_tot_log = (T_TRAIN_SEC / N_values_log) + T_TOT_INFER_SEC
avg_time_not_log = np.full_like(N_values_log, T_NOT_SEC)

plt.figure(figsize=(8, 6))
plt.loglog(N_values_log, avg_time_not_log, label=f'Standard (NOT): {T_NOT_SEC}s/sample', linewidth=2, color='tab:orange', linestyle='--')
plt.loglog(N_values_log, avg_time_tot_log, label=f'Amortized (TOT): {T_TRAIN_HOURS}h train + {T_TOT_INFER_SEC}s/sample', linewidth=2, color='tab:blue')
plt.plot(N_break_even, T_NOT_SEC, 'ko', label=f'Break-even: N $\\approx$ {int(N_break_even):,}')

plt.xlabel('Number of Samples (N) [Log scale]', fontsize=14)
plt.ylabel('Average Time per Sample (Seconds) [Log scale]', fontsize=14)
plt.title('Amortized Runtime Comparison (Log Scale)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.grid(True, which="major", ls="-", alpha=0.5)
plt.tight_layout()

out_log = "TF4OT/evals/plots/img/amortized_runtime_comparison_log.png"
os.makedirs(os.path.dirname(out_log), exist_ok=True)
plt.savefig(out_log, bbox_inches="tight", dpi=300)
print(f"Saved Log Scale Plot: {out_log}")
plt.close()


# --- Plot 2: Linear Scale ---
max_N = int(N_break_even * 7) 
N_values_lin = np.linspace(100, max_N, 500)

avg_time_tot_lin = (T_TRAIN_SEC / N_values_lin) + T_TOT_INFER_SEC
avg_time_not_lin = np.full_like(N_values_lin, T_NOT_SEC)

plt.figure(figsize=(10, 4))
plt.plot(N_values_lin, avg_time_not_lin, label=f'NeuralOT', linewidth=4, color='darkred', linestyle='--')
plt.plot(N_values_lin, avg_time_tot_lin, label=f'ToR-OT', linewidth=4, color='darkgreen')
# plt.plot(N_break_even, T_NOT_SEC, 'ko', label=f'Break-even: N $\\approx$ {int(N_break_even):,}')

plt.xlabel('Number of Samples', fontsize=20)
plt.ylabel('Amortized time (Sec)', fontsize=20)
# plt.title('Amortized Runtime Comparison (Linear Scale)', fontsize=16)
plt.legend(fontsize=20, loc='upper right', framealpha=1)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.grid(True, which="major", ls="-", alpha=0.5)

plt.tight_layout()

out_lin = "TF4OT/evals/plots/img/amortized_runtime_comparison_linear.png"
plt.savefig(out_lin, bbox_inches="tight", dpi=300)
print(f"Saved Linear Scale Plot: {out_lin}")
plt.close()
