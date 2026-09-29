import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = (
    "experiments/cost_fixed/"
    "model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_"
    "flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_"
    "epochs50_batch192_sqeuclidean"
)
M = MODEL_DIR

def safe_mean(arr):
    a = np.array(arr, dtype=float)
    return float(np.nanmean(a)) if len(a) > 0 else None

# Test-set data
ts_tot = np.load(f"{M}/eval_all_testset_1d_sqeuclidean_results_tot.npz", allow_pickle=True)
ts_pot = np.load(f"{M}/eval_all_testset_1d_sqeuclidean_results_pot.npz", allow_pickle=True)
ts_sym = np.load(f"{M}/eval_all_testset_1d_sqeuclidean_results_tot_sym.npz", allow_pickle=True)

test_err = {
    "POT (exact)": 1e-12, # Technically 0, using 1e-12 for log plot
    "TOT (Ours)": safe_mean(ts_tot["all_otcost_diff"]),
    "ToR-OT (Sym)": safe_mean(ts_sym["all_otcost_diff"]),
    "NeuralOT": 8.310062e-05
}
test_time = {
    "POT (exact)": safe_mean(ts_pot["all_time_pot"]),
    "TOT (Ours)": safe_mean(ts_pot["all_time_tot"]),
    "ToR-OT (Sym)": None,
    "NeuralOT": 1.0 # Conservative estimate from logs
}

# OOD data
ood_tot = np.load(f"{M}/eval_all_pairs_1d_sqeuclidean_results_tot.npy", allow_pickle=True)
ood_pot = np.load(f"{M}/eval_all_pairs_1d_sqeuclidean_results_pot.npy", allow_pickle=True)
ood_sym = np.load(f"{M}/eval_all_pairs_1d_sqeuclidean_results_tot_sym.npy", allow_pickle=True)

def ood_val(data, key):
    if len(data) == 0: return None
    vals = [r[key] for r in data if key in r and isinstance(r[key], (int, float, np.floating))]
    return safe_mean(vals) if vals else None

ood_err = {
    "POT (exact)": 1e-12,
    "TOT (Ours)": ood_val(ood_tot, "otcost_diff"),
    "ToR-OT (Sym)": ood_val(ood_sym, "otcost_diff"),
    "NeuralOT": 1.5381e-02
}
ood_time = {
    "POT (exact)": ood_val(ood_pot, "run_time_pot"),
    "TOT (Ours)": ood_val(ood_tot, "run_time"),
    "ToR-OT (Sym)": ood_val(ood_sym, "run_time"),
    "NeuralOT": 1.0021e+00
}

# Plot
METHODS = ["POT (exact)", "TOT (Ours)", "ToR-OT (Sym)", "NeuralOT"]
COLORS = {"POT (exact)": "#4CAF50", "TOT (Ours)": "#2196F3", "ToR-OT (Sym)": "#FF9800", "NeuralOT": "#9C27B0"}
MARKERS = {"POT (exact)": "*", "TOT (Ours)": "o", "ToR-OT (Sym)": "s", "NeuralOT": "D"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=130)
fig.suptitle("Pareto Frontier: Inference Time vs Precision (OT Cost Δ)", fontsize=15, fontweight="bold")

for ax, m_time, m_err, title in zip(axes, [test_time, ood_time], [test_err, ood_err], ["Test Set (In-Domain)", "OOD Set (Out-of-Distribution)"]):
    for method in METHODS:
        # Check if we have both values
        t = m_time.get(method)
        e = m_err.get(method)
        if t is not None and e is not None:
            # For POT err is tiny, we just bounded it at 1e-12 for plotting log scale
            ax.scatter(t, e, color=COLORS[method], marker=MARKERS[method], s=250, label=method, zorder=5, alpha=0.9)
            # Annotate
            ax.annotate(method, (t, e), xytext=(8, -8), textcoords="offset points", fontsize=10, 
                        weight='bold', color=COLORS[method], va='top', ha='left')
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean Inference Time per Sample (Seconds) →\nLower is better", fontsize=11)
    ax.set_ylabel("OT Cost Δ (Precision Error) →\nLower is better", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Adding quadrants logic visually
    ax.axhline(1e-3, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(1e-1, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
out_path = os.path.join(M, "pareto_comparison.png")
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"✅ Pareto plot saved to {out_path}")
