"""
compare_pot_warm.py

Produces comparison tables and a figure evaluating POT vs POT Warmstart (with TOT/NOT context)
for BOTH the TestSet and the custom Out-Of-Distribution (OOD) pairs.
"""

import os
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

warnings.filterwarnings("ignore", category=RuntimeWarning)

M = (
    "experiments/cost_fixed/"
    "model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_"
    "flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_"
    "epochs50_batch192_sqeuclidean"
)

def safe_mean(arr):
    a = np.array(arr, dtype=float)
    return float(np.nanmean(a)) if len(a) > 0 else None

def fmt(v, suffix=""):
    return f"{v:.3e}{suffix}" if v is not None else "N/A"

# Load Test-Set Data
ts_tot = np.load(f"{M}/eval_all_testset_1d_sqeuclidean_results_tot.npz", allow_pickle=True)
ts_pot = np.load(f"{M}/eval_all_testset_1d_sqeuclidean_results_pot.npz", allow_pickle=True)

def ts_val(d, key):
    return safe_mean(d[key] if key in d else np.array([]))

TESTSET_ACC = [
    ("MAE",       "all_mae_sparse",      "all_mae_sparse_pot"),
    ("MAPE",      "all_mape_sparse",     "all_mape_sparse_pot"),
    ("Hellinger", "all_hellinger_sparse","all_hellinger_sparse_pot"),
    ("TV",        "all_tv_sparse",       "all_tv_sparse_pot"),
    ("OT Cost \u0394", "all_otcost_diff",     "all_otcost_diff_pot"),
]

ts_results = {"TOT": {}, "POT": {}, "POT Warm": {}, "NOT": {}}
for n, k_tot, k_pot in TESTSET_ACC:
    ts_results["TOT"][n] = ts_val(ts_tot, k_tot)
    ts_results["POT"][n] = ts_val(ts_pot, k_pot)
    ts_results["POT Warm"][n] = ts_val(ts_pot, k_pot + "_warm")

ts_results["TOT"]["Max Viol."] = ts_val(ts_tot, "all_max_violation")
ts_results["POT"]["Max Viol."] = None
ts_results["POT Warm"]["Max Viol."] = None

ts_results["TOT"]["Time (s)"] = ts_val(ts_pot, "all_time_tot")
ts_results["POT"]["Time (s)"] = ts_val(ts_pot, "all_time_pot")
ts_results["POT Warm"]["Time (s)"] = ts_val(ts_pot, "all_time_pot_warm")

ts_results["NOT"] = {
    "MAE": 1.378084e-03, "MAPE": 8.298525e-02, "Hellinger": 7.140481e-02,
    "TV": 4.149263e-02, "OT Cost \u0394": 8.310062e-05, "Max Viol.": None, "Time (s)": 1.0
}

all_metrics_ts = [n for n, *_ in TESTSET_ACC] + ["Max Viol.", "Time (s)"]

# Load OOD Data
ood_tot = np.load(f"{M}/eval_all_pairs_1d_sqeuclidean_results_pot.npy", allow_pickle=True)
ood_pot = np.load(f"{M}/eval_all_pairs_1d_sqeuclidean_results_pot.npy", allow_pickle=True)

def ood_val(data, key):
    if len(data) == 0: return None
    vals = [r[key] for r in data if key in r and isinstance(r[key], (int, float, np.floating))]
    return safe_mean(vals) if vals else None

OOD_ACC = ["mae_sparse", "mape_sparse", "hellinger_sparse", "tv_sparse", "otcost_diff", "max_violation"]
OOD_NAMES = ["MAE", "MAPE", "Hellinger", "TV", "OT Cost \u0394", "Max Viol."]

ood_results = {"TOT": {}, "POT": {}, "POT Warm": {}, "NOT": {}}

for name, key in zip(OOD_NAMES, OOD_ACC):
    ood_results["TOT"][name] = ood_val(ood_tot, key)
    if key == "mape_sparse":
        ood_results["POT"][name] = ood_val(ood_tot, key + "_pot")
        ood_results["POT Warm"][name] = ood_val(ood_tot, key + "_pot_warm")
    else:
        ood_results["POT"][name] = None
        ood_results["POT Warm"][name] = None

ood_results["TOT"]["Time (s)"] = ood_val(ood_tot, "run_time")
ood_results["POT"]["Time (s)"] = ood_val(ood_tot, "run_time_pot")
# IMPORTANT: My patch added run_time_pot_warm to the row dictionaries natively
ood_results["POT Warm"]["Time (s)"] = ood_val(ood_tot, "run_time_pot_warm")

ood_results["NOT"] = {
    "MAE": 9.9941e-03, "MAPE": 7.7170e-01, "Hellinger": 5.0861e-01, "TV": 3.8584e-01, "OT Cost \u0394": 1.5381e-02, "Max Viol.": None, "Time (s)": 1.0021
}
all_metrics_ood = OOD_NAMES + ["Time (s)"]


METHODS = ["POT (exact)", "POT Warmstart", "TOT (Ours)"]
METHOD_KEYS = ["POT", "POT Warm", "TOT"]
HEADERS = ["Metric"] + METHODS

def make_table(results, all_metrics):
    rows = []
    for m in all_metrics:
        row = [m]
        for key in METHOD_KEYS:
            v = results[key].get(m)
            row.append(fmt(v))
        rows.append(row)
    return rows

print("\n" + "="*90)
print("  TEST-SET WARMSTART COMPARISON  (1D, sqeuclidean, 50 examples)")
print("="*90)
print(tabulate(make_table(ts_results, all_metrics_ts), headers=HEADERS, tablefmt="grid"))

print("\n" + "="*90)
print("  OOD WARMSTART COMPARISON  (1D, Custom, 50 pairs)")
print("="*90)
print(tabulate(make_table(ood_results, all_metrics_ood), headers=HEADERS, tablefmt="grid"))

# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {"POT (exact)": "#4CAF50", "POT Warmstart": "#FFC107", 
          "TOT (Ours)": "#2196F3"}

# Using ONLY MAPE per User Request for error visualization
acc_names = ["MAPE"]

def acc_bar_data(results, metric_list):
    out = {m: [] for m in METHODS}
    for met in metric_list:
        for lbl, key in zip(METHODS, METHOD_KEYS):
            v = results[key].get(met)
            if v is None:
                val = 0.0
            elif v <= 0.0:
                val = 1e-7  # Avoid log(0) and force empty renders to be visually grounded
            else:
                val = v
            out[lbl].append(val)
    return out

ts_bar = acc_bar_data(ts_results, acc_names)
ood_bar = acc_bar_data(ood_results, acc_names)

fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=130)
fig.suptitle("Comparison: Exact Solver Warmstarting via ToR-OT vs Baselines\nOnly using MAPE",
             fontsize=15, fontweight="bold", y=1.03)

x = np.arange(len(acc_names))
n_methods = len(METHODS)
bar_w = 0.2

def _has_data(vals): return any(v > 0 for v in vals)

def plot_acc(ax, bar_data, title):
    for i, (method, color) in enumerate(COLORS.items()):
        vals = bar_data[method]
        if not _has_data(vals): continue
        offset = (i - (n_methods - 1) / 2) * bar_w
        ax.bar(x + offset, vals, bar_w, label=method, color=color, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(acc_names, rotation=0, fontsize=9)
    ax.set_ylabel("Mean Error (log scale)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

plot_acc(axes[0, 0], ts_bar, "Test-Set — MAPE")
plot_acc(axes[0, 1], ood_bar, "OOD — MAPE")

def plot_time(ax, results, title):
    methods_with_time = [(lbl, key) for lbl, key in zip(METHODS, METHOD_KEYS)
                         if results[key].get("Time (s)") is not None]
    labels = [lbl for lbl, _ in methods_with_time]
    vals   = [results[key]["Time (s)"] for _, key in methods_with_time]
    colors = [COLORS[lbl] for lbl in labels]

    bars = ax.bar(labels, vals, color=colors, alpha=0.9, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.3,
                f"{val:.2e}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Mean Time per Sample (s, log scale)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    pot_t = results["POT"].get("Time (s)")
    pot_w_t = results["POT Warm"].get("Time (s)")
    if pot_t and pot_w_t:
        speedup = pot_t / pot_w_t
        if speedup > 0:
            ax.text(0.97, 0.97, f"Warmstart Speedup: {speedup:.1f}x faster",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9, fontname='sans-serif', fontweight='bold',
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.9))

plot_time(axes[1, 0], ts_results, "Test-Set — Latency")
plot_time(axes[1, 1], ood_results, "OOD — Latency")

plt.tight_layout()
fig.savefig(f"{M}/compare_pot_warm.png")
print(f"\n\u2705 4-panel figure saved to {M}/compare_pot_warm.png\n")
