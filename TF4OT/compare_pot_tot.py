"""
compare_pot_tot.py

Collect results from saved .npz / .npy files and produce:
  - Printed comparison tables (test-set + OOD)
  - A matplotlib figure with accuracy bars and timing comparisons

NeuralOT (NOT) numbers are recovered from the nohup.out log of the January 2026
run where test_neural_ot=true was set. The NOT .npy/.npz result files are empty
because subsequent runs used test_neural_ot=false.
  - NOT testset: 2026-01-14 run, sqeuclidean, 1000 examples
  - NOT OOD:     2026-01-07 run, sqeuclidean, 144 distribution pairs

Usage (from repo root):
    python3 TF4OT/compare_pot_tot.py
"""

import os
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tabulate import tabulate

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = (
    "experiments/cost_fixed/"
    "model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_"
    "flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_"
    "epochs50_batch192_sqeuclidean"
)
M = MODEL_DIR

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def safe_mean(arr):
    a = np.array(arr, dtype=float)
    return float(np.nanmean(a)) if len(a) > 0 else None

def fmt(v, suffix=""):
    return f"{v:.3e}{suffix}" if v is not None else "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# 1. TEST-SET  (.npz files)
# ──────────────────────────────────────────────────────────────────────────────
ts_tot = np.load(f"{M}/eval_all_testset_1d_sqeuclidean_results_tot.npz",     allow_pickle=True)
ts_pot = np.load(f"{M}/eval_all_testset_1d_sqeuclidean_results_pot.npz",     allow_pickle=True)
ts_sym = np.load(f"{M}/eval_all_testset_1d_sqeuclidean_results_tot_sym.npz", allow_pickle=True)
ts_not = np.load(f"{M}/eval_all_testset_1d_sqeuclidean_results_not.npz",     allow_pickle=True)

def ts_val(d, key):
    arr = d[key] if key in d else np.array([])
    return safe_mean(arr)

# accuracy metric name → (tot_key, pot_key_in_pot_npz, not_key)
TESTSET_ACC = [
    ("MAE",       "all_mae_sparse",      "all_mae_sparse_pot",      "all_mae_sparse"),
    ("MAPE",      "all_mape_sparse",     "all_mape_sparse_pot",     "all_mape_sparse"),
    ("Hellinger", "all_hellinger_sparse","all_hellinger_sparse_pot","all_hellinger_sparse"),
    ("TV",        "all_tv_sparse",       "all_tv_sparse_pot",       "all_tv_sparse"),
    ("OT Cost Δ", "all_otcost_diff",     "all_otcost_diff_pot",     "all_otcost_diff"),
    ("Max Viol.", "all_max_violation",   None,                      None),
]

ts_results = {}  # method → metric → value
for method, data in [("TOT", ts_tot), ("ToR-OT", ts_sym)]:
    ts_results[method] = {name: ts_val(data, k_tot) for name, k_tot, _, _ in TESTSET_ACC}
    ts_results[method]["Max Viol."] = ts_val(data, "all_max_violation")

ts_results["POT"] = {name: ts_val(ts_pot, k_pot) if k_pot else None
                     for name, _, k_pot, _ in TESTSET_ACC}

ts_results["POT (Warm)"] = {name: ts_val(ts_pot, k_pot + "_warm") if k_pot else None
                            for name, _, k_pot, _ in TESTSET_ACC}
ts_results["POT"]["Max Viol."] = None
ts_results["POT (Warm)"]["Max Viol."] = None  # exact solver has no violation

ts_results["NOT"] = {
    # Recovered from nohup.out: 2026-01-14 run, sqeuclidean fixed cost, 1000 examples
    # (test_neural_ot=true, neural_ot_max_steps=2500)
    "MAE":       1.378084e-03,
    "MAPE":      8.298525e-02,
    "Hellinger": 7.140481e-02,
    "TV":        4.149263e-02,
    "OT Cost Δ": 8.310062e-05,
    "Max Viol.": None,          # NOT is exact by marginals, skip violation
    "Time (s)":  None,          # per-sample NOT time not logged in testset run
}

# Timing
time_tot_ts = ts_val(ts_pot, "all_time_tot")
time_pot_ts = ts_val(ts_pot, "all_time_pot")

ts_results["TOT"]["Time (s)"]    = time_tot_ts
ts_results["POT"]["Time (s)"]    = time_pot_ts
ts_results["POT (Warm)"]["Time (s)"] = ts_val(ts_pot, "all_time_pot_warm")
ts_results["ToR-OT"]["Time (s)"] = None
# NOT testset run-time: ~1.0s per sample (inferred from OOD run; tqdm not logged per-sample)
ts_results["NOT"]["Time (s)"]    = 1.0   # conservative per-sample estimate from OOD

all_metrics_ts = [n for n, *_ in TESTSET_ACC] + ["Time (s)"]

# ──────────────────────────────────────────────────────────────────────────────
# 2. OOD  (.npy - list of dicts)
# ──────────────────────────────────────────────────────────────────────────────
ood_tot = np.load(f"{M}/eval_all_pairs_1d_sqeuclidean_results_tot.npy",     allow_pickle=True)
ood_pot = np.load(f"{M}/eval_all_pairs_1d_sqeuclidean_results_pot.npy",     allow_pickle=True)
ood_sym = np.load(f"{M}/eval_all_pairs_1d_sqeuclidean_results_tot_sym.npy", allow_pickle=True)
ood_not = np.load(f"{M}/eval_all_pairs_1d_sqeuclidean_results_not.npy",     allow_pickle=True)

def ood_val(data, key):
    if len(data) == 0: return None
    vals = [r[key] for r in data if key in r and isinstance(r[key], (int, float, np.floating))]
    return safe_mean(vals) if vals else None

OOD_ACC = ["mae_sparse", "mape_sparse", "hellinger_sparse", "tv_sparse", "otcost_diff", "max_violation"]
OOD_NAMES = ["MAE", "MAPE", "Hellinger", "TV", "OT Cost Δ", "Max Viol."]

ood_results = {}
for method, data in [("TOT", ood_tot), ("ToR-OT", ood_sym)]:
    ood_results[method] = {name: ood_val(data, key) for name, key in zip(OOD_NAMES, OOD_ACC)}
    ood_results[method]["Time (s)"] = ood_val(data, "run_time")

# POT: exact — accuracy ~0 by definition; time from run_time_pot
ood_results["POT"] = {name: None for name in OOD_NAMES}
ood_results["POT"]["Time (s)"] = ood_val(ood_pot, "run_time_pot")

# NeuralOT OOD: recovered from nohup.out, 2026-01-07 run, sqeuclidean, 144 pairs
ood_results["NOT"] = {
    "MAE":       9.9941e-03,
    "MAPE":      7.7170e-01,
    "Hellinger": 5.0861e-01,
    "TV":        3.8584e-01,
    "OT Cost Δ": 1.5381e-02,
    "Max Viol.": None,
    "Time (s)":  1.0021e+00,   # mean run_time per OOD pair
}

all_metrics_ood = OOD_NAMES + ["Time (s)"]

# ──────────────────────────────────────────────────────────────────────────────
# 3. Print tables
# ──────────────────────────────────────────────────────────────────────────────
METHODS = ["POT (exact)", "TOT (Ours)", "ToR-OT (Sym)", "NeuralOT"]
METHOD_KEYS = ["POT", "TOT", "ToR-OT", "NOT"]
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

print("\n" + "="*80)
print("  TEST-SET COMPARISON  (1D, sqeuclidean, 1 000 examples)")
print("="*80)
print(tabulate(make_table(ts_results, all_metrics_ts), headers=HEADERS, tablefmt="grid"))

print("\n" + "="*80)
print("  OOD COMPARISON  (1D custom distributions, 144 pairs)")
print("="*80)
print(tabulate(make_table(ood_results, all_metrics_ood), headers=HEADERS, tablefmt="grid"))

print("\nℹ️  NeuralOT numbers are from the 2026-01 run (test_neural_ot=true, max_steps=2500).")
print("   TOT/ToR-OT/POT are from the 2026-03-30 run_inference_pot.py run.\n")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Plots
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {"POT (exact)": "#4CAF50", "TOT (Ours)": "#2196F3",
          "ToR-OT (Sym)": "#FF9800", "NeuralOT": "#9C27B0"}

acc_names = ["MAE", "MAPE", "Hellinger", "TV", "OT Cost Δ", "Max Viol."]

def acc_bar_data(results, metric_list):
    """Returns {method: [values]} with None → 0 for plotting."""
    out = {m: [] for m in METHODS}
    for met in metric_list:
        for lbl, key in zip(METHODS, METHOD_KEYS):
            v = results[key].get(met)
            out[lbl].append(v if v is not None else 0.0)
    return out

ts_bar = acc_bar_data(ts_results, acc_names)
ood_bar = acc_bar_data(ood_results, acc_names)

fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=130)
fig.suptitle("Comparison: TOT / ToR-OT / NeuralOT  vs  POT Exact Solver",
             fontsize=15, fontweight="bold", y=1.01)

x = np.arange(len(acc_names))
n_methods = len(METHODS)
bar_w = 0.2

def _has_data(vals):
    return any(v > 0 for v in vals)

def plot_acc(ax, bar_data, title):
    any_plotted = False
    for i, (method, color) in enumerate(COLORS.items()):
        vals = bar_data[method]
        if not _has_data(vals):
            continue
        offset = (i - (n_methods - 1) / 2) * bar_w
        ax.bar(x + offset, vals, bar_w, label=method, color=color, alpha=0.85)
        any_plotted = True
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(acc_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean Error (log scale)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=1e-12)

plot_acc(axes[0, 0], ts_bar,  "Test-Set — Accuracy Metrics")
plot_acc(axes[0, 1], ood_bar, "OOD — Accuracy Metrics")

def plot_time(ax, results, title):
    methods_with_time = [(lbl, key) for lbl, key in zip(METHODS, METHOD_KEYS)
                         if results[key].get("Time (s)") is not None]
    labels = [lbl for lbl, _ in methods_with_time]
    vals   = [results[key]["Time (s)"] for _, key in methods_with_time]
    colors = [COLORS[lbl] for lbl in labels]

    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.3,
                f"{val:.2e}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Mean Time per Sample (s, log scale)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(axis='x', labelsize=9)

    # Speedup annotations relative to POT
    pot_t = results["POT"].get("Time (s)")
    if pot_t:
        note = "\n".join(
            f"{lbl} is {results[key]['Time (s)'] / pot_t:.0f}× slower than POT"
            for lbl, key in zip(METHODS, METHOD_KEYS)
            if key != "POT" and results[key].get("Time (s)") is not None
        )
        ax.text(0.97, 0.97, note,
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

plot_time(axes[1, 0], ts_results,  "Test-Set — Execution Time")
plot_time(axes[1, 1], ood_results, "OOD — Execution Time")

plt.tight_layout()
out_path = os.path.join(MODEL_DIR, "compare_pot_tot.png")
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"✅  Figure saved → {out_path}")
