"""Every data-driven figure and table of the paper, from the results of the pipeline in reproduce/README.md (CPU).

    python reproduce/make_figures.py experiments/torot/camera_ready [--only <function> ...]        (env_torot)

Writes <results>/paper/ (or --out) <location>/<name>.{pdf,png | md,tex,csv}, one folder per place in the paper (main_text,
appendix_B_runtime, appendix_C_feasibility, appendix_D_interpretability, appendix_E_1d_results, appendix_F_2d_results,
appendix_G_scaling), and the numbers drawn in each figure in <location>/data/. ROUTES below maps every output to its
folder and file name; reproduce/README.md lists each paper element with its function and inputs.
Every function is run independently: one with a missing input is skipped and reported at the end.
Other result folders: <results>_2d (2D sets, --root-2d), <results>_n100 / _n200 (n = 100 / 200, --n100-root),
<results>/../ablations (small testbed, --ablations), the 1D training run (training log, --train-run).
"""
import argparse
import re, glob, json, os, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--train-hours", type=float, default=24.41,
                help="wall-clock training time of the 1D model, 8x A100 (row_M_5M: 0.55 h to step 5,000, then 23.86 h after a restart)")
ap.add_argument("--neuralot-seconds", type=float, default=2627.08, help="NeuralOT time per instance (paper, Table 2)")
ap.add_argument("--ablations", default=None, help="ablation runs of the small testbed (default: <root>/../ablations)")
ap.add_argument("--n100-root", default=None, help="results at n = 100 (default: <root>_n100)")
ap.add_argument("--root-2d", default=None, help="results on the 2D sets (default: <root>_2d)")
ap.add_argument("--train-run", default=None,
                help="training run of the 1D model, for its log.jsonl (default: <root>/train_row_M_5M if present, "
                     "else experiments/torot/row_M_5M)")
ap.add_argument("--style", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_style.yaml"),
                help="labels, colors, fonts and formats of every figure")
ap.add_argument("--out", default=None, help="output folder (default: <root>/paper)")
ap.add_argument("--only", nargs="*", default=None, help="make only these figure / table functions (names as listed below)")
args = ap.parse_args()
R = args.root
OUT = args.out or f"{R}/paper"                             # every figure and table of the paper, by where it appears

# Where each output goes: <results>/paper/<location>/<name>.{pdf,png | md,tex,csv}, the numbers behind a figure in
# <location>/data/. Rules are matched in order on the name the code below uses; an output without a rule is an error.
ROUTES = [
    (r"fig3_camera_ready(?P<rest>.*)",            "main_text",                   "fig3_evaluation{rest}"),
    (r"fig4_camera_ready(?P<rest>.*)",            "main_text",                   "fig4_results{rest}"),
    (r"fig5_camera_ready(?P<rest>.*)",            "main_text",                   "fig5_2d_example{rest}"),
    (r"table1_improvement",                       "main_text",                   "table1_improvement"),
    (r"table2_runtime",                           "appendix_B_runtime",          "table_appB_runtime"),
    (r"fig6_amortized_runtime",                   "appendix_B_runtime",          "fig_appB_amortized_runtime"),
    (r"fig_latency_vs_n",                         "appendix_B_runtime",          "fig_appB_latency_vs_n"),
    (r"fig_appC_warmstart_(?P<rest>.*)",          "appendix_C_feasibility",      "fig_appC_sinkhorn_refinement_{rest}"),
    (r"warmstart",                                "appendix_C_feasibility",      "fig_appC_sinkhorn_refinement"),
    (r"table3_sinkhorn_projection(?P<rest>.*)",   "appendix_C_feasibility",      "table_appC_sinkhorn_projection{rest}"),
    (r"table4_symmetric",                         "appendix_C_feasibility",      "table_appC_transpose_averaging_sets"),
    (r"table4_symmetric_5M",                      "appendix_C_feasibility",      "table_appC_transpose_averaging_5M_splits"),
    (r"fig_appD_interpretability",                "appendix_D_interpretability", "fig_appD_interpretability"),
    (r"appD_interpretability_summary",            "appendix_D_interpretability", "fig_appD_interpretability_summary"),
    (r"tableD_interpretability_all_sets",         "appendix_D_interpretability", "table_appD_interpretability_all_sets"),
    (r"(appendix_E/)?(?P<name>fig_appE_.*)",      "appendix_E_1d_results",       "{name}"),
    (r"(appendix_F/)?(?P<name>fig_appF_.*)",      "appendix_F_2d_results",       "{name}"),
    (r"(appendix_G/)?(?P<name>fig_appG_.*)",      "appendix_G_scaling",          "{name}"),
    (r"(?P<name>table_appG_.*)",                  "appendix_G_scaling",          "{name}"),
    (r"data_curve",                               "appendix_G_scaling",          "fig_appG_training_curve_points"),
    (r"table_ablations_n30(?P<rest>.*)",          "appendix_G_scaling",          "table_appG_ablations{rest}"),
    (r"fig_scaling_n100",                         "appendix_G_scaling",          "fig_appG_scaling_n100"),
    (r"table_scaling_n100",                       "appendix_G_scaling",          "table_appG_scaling_n100"),
    (r"fig_size_generalization",                  "appendix_G_scaling",          "fig_appG_size_generalization"),
]


def route(name):
    """(location folder, file name) of output ``name``."""
    for pattern, folder, new in ROUTES:
        m = re.fullmatch(pattern, name)
        if m:
            return folder, new.format(**{k: v or "" for k, v in m.groupdict().items()})
    raise KeyError(f"no output location for {name!r}: add a rule to ROUTES")


def data_file(name, ext):
    """Path of a data file (numbers behind a figure or table) for output ``name``: <location>/data/<file>.<ext>."""
    folder, new = route(name)
    os.makedirs(f"{OUT}/{folder}/data", exist_ok=True)
    return f"{OUT}/{folder}/data/{new}.{ext}"
ABL = args.ablations or os.path.join(os.path.dirname(os.path.abspath(R)), "ablations")
R100 = args.n100_root or os.path.abspath(R).rstrip("/") + "_n100"
R200 = os.path.abspath(R).rstrip("/") + "_n200"            # results at n = 200 (if present)
R2D = args.root_2d or os.path.abspath(R).rstrip("/") + "_2d"  # results on the 2D sets (Fig. 4C/D, App. F)
TRAIN_RUN = args.train_run or next((d for d in (f"{R}/train_row_M_5M", "experiments/torot/row_M_5M") if os.path.isdir(d)),
                                   f"{R}/train_row_M_5M")      # 1D training run (App. G training curve)
# ToR-OT's reported decoding: every row, the last one included, decoded by the model (reproduce/run_lastrow.sh)
TOROT = "v2row_lastrow"
os.makedirs(OUT, exist_ok=True)

import yaml
STYLE = yaml.safe_load(open(args.style))
plt.rcParams.update(STYLE.get("rc", {}))
TOROT_SYM = "v2row_sym_lastrow"                              # the same, transpose-averaged
EVAL_SUFFIX = "_lastrow"                                     # 5M-split evaluations, latency, data curve of that decoding
_m = {k: (v["label"], v["color"]) for k, v in STYLE["methods"].items()}
METHODS = {TOROT: _m["torot"], TOROT_SYM: _m["torot_sym"], "metaot": _m["metaot"], "unot": _m["unot"], "unot_sk10": _m["unot_sk10"],
           "sinkhorn_eps0.01": _m["sinkhorn_eps0.01"], "sinkhorn_eps0.001": _m["sinkhorn_eps0.001"], "pot": _m["pot"],
           "neuralot": _m["neuralot"], "sinkhorn_gpu_eps0.01": _m["sinkhorn_gpu_eps0.01"],
           "sinkhorn_gpu_eps0.001": _m["sinkhorn_gpu_eps0.001"]}   # key: (label, color)
CMAP = STYLE["colormaps"]


COSTS6 = ["sqeuclidean", "euclidean", "canberra", "sqrt_euclidean", "cubic_euclidean", "gaussian"]
PLAN_METRICS = [("mae", "MAE"), ("paper_mape", "MAPE"), ("hellinger", "Hellinger"),
                ("ot_cost_abs_diff", "|OT cost diff.|"), ("tv", "TV")]
FLOOR = 1e-10  # exact matches are drawn at this value on log axes
done, skipped = [], []


def save(fig, name, formats=None):
    """Figure ``name`` as <results>/paper/<location>/<file>.{pdf,png} (see ROUTES). The App. E and F figures are PNG
    only (as in the paper), unless ``formats`` says otherwise (App. G: PDF only)."""
    folder, new = route(name)
    os.makedirs(f"{OUT}/{folder}", exist_ok=True)
    for fmt in formats or (["png"] if name.startswith("appendix_") else STYLE.get("formats", ["pdf", "png"])):
        fig.savefig(f"{OUT}/{folder}/{new}.{fmt}", bbox_inches="tight", dpi=STYLE.get("dpi", 200))
    plt.close(fig); done.append(f"paper/{folder}/{new}")

def save_data(name, df):
    """The numbers drawn in figure ``name`` (re-plot them with any tool): <results>/paper/<location>/data/<file>.csv"""
    df.to_csv(data_file(name, "csv"), index=False)

def per_instance(set_name, method, root=None):
    path = f"{root or R}/scores/per_instance/{set_name}__{method}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None

def timing(set_name, method, root=None):
    path = f"{root or R}/preds/{set_name}__{method}.timing.json"
    return json.load(open(path)) if os.path.exists(path) else None


def write_table(name, df, caption):
    """Table ``name`` as <results>/paper/<location>/<file>.{md,tex,csv} (see ROUTES)."""
    folder, new = route(name)
    os.makedirs(f"{OUT}/{folder}", exist_ok=True)
    with open(f"{OUT}/{folder}/{new}.md", "w") as f:
        f.write(f"{caption}\n\n" + df.to_markdown(index=False) + "\n")
    df.to_latex(f"{OUT}/{folder}/{new}.tex", index=False, float_format="%.4g", caption=caption, label=f"tab:{new}")
    df.to_csv(f"{OUT}/{folder}/{new}.csv", index=False)
    done.append(f"paper/{folder}/{new}")

def step(fn):
    try:
        fn()
    except Exception as e:  # report and continue with the next figure
        skipped.append(f"{fn.__name__}: {type(e).__name__}: {e}")
        traceback.print_exc()
    return fn


# ----------------------------------------------------------------------------- improvement heatmaps (Fig. 4B/C, App. E/F)


def improvement_cmap(vs):
    """Diverging colormap of an improvement of ToR-OT over method ``vs``: the method's color where it is better
    (negative), white at 0, ToR-OT's color where ToR-OT is better (positive); the paper's colors for each method."""
    if vs is None:
        return CMAP["improvement"]
    return LinearSegmentedColormap.from_list(f"imp_{vs}", [METHODS[vs][1], "white", METHODS[TOROT][1]])


DIAG_GREY = "0.88"   # improvement heatmaps: background of the diagonal cells, which show an absolute difference


def draw_grid(ax, G, fams, title, kind, fig, vs=None, fontsize=5, colorbar=True, diag=None):
    """``diag`` (improvement heatmaps): per source = target pair, the absolute MAPE difference (baseline - ToR-OT), drawn
    on a grey cell instead of the relative improvement; there both errors can be close to 0 and a ratio of them is
    not meaningful."""
    if kind == "abs":
        im = ax.imshow(np.clip(G, 1e-4, None), cmap=CMAP["error"], norm=LogNorm())
    else:
        im = ax.imshow(np.clip(G, -100, 100), cmap=improvement_cmap(vs), norm=TwoSlopeNorm(vcenter=0, vmin=-100, vmax=100))
        for i in range(len(fams)):
            for j in range(len(fams)):
                if diag is not None and i == j:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=DIAG_GREY, edgecolor="none"))
                    t = f"{diag[i]:+.2f}"
                    ax.text(j, i, "0.00" if t in ("+0.00", "-0.00") else t, ha="center", va="center",
                            fontsize=fontsize * 0.85, color="0.15")
                elif np.isfinite(G[i, j]):
                    txt = f"{G[i, j]:.0f}" if abs(G[i, j]) < 1e4 else f"{G[i, j]:.0e}".replace("e+0", "e").replace("e+", "e")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize,
                            color="white" if abs(G[i, j]) > 65 else "0.15")
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=90, fontsize=7)
    ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=7)
    ax.set_xlabel("target"); ax.set_ylabel("source"); ax.set_title(title, fontsize=10)
    if colorbar: fig.colorbar(im, ax=ax, shrink=0.75)
    return im


def panel_letters(fig, axes, size=17, title_size=11, beside=False):
    """Bold panel letters in a box, each at the left edge of its panel's y-axis labels.
    Default: on one row above the panels, with the panels' left titles moved to that row. ``beside``: the top of the
    box aligned with the top of the panel (no space above the panels)."""
    fig.canvas.draw()
    r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    top = max(ax.get_position().y1 for ax in axes.values())
    pt = 1 / (fig.get_figheight() * 72)                                # 1 pt in figure coordinates (vertical)
    box = dict(boxstyle="round,pad=0.22,rounding_size=0.12", facecolor="white", edgecolor="black", linewidth=1.3)
    for letter, ax in axes.items():
        pos = ax.get_position()
        x0 = inv.transform(ax.yaxis.get_tightbbox(r).p0)[0]
        if beside:                                                     # text top = panel top minus the box padding
            t = fig.text(x0, pos.y1 - (0.22 * size + 0.65) * pt, letter, fontsize=size, fontweight="bold", ha="left",
                         va="top", bbox=box)
            fig.canvas.draw()                                          # if the box covers a tick label (e.g. the top
            bb = t.get_bbox_patch().get_window_extent(r)               # tick of a bar plot), move it left of them
            lo, hi = sorted(ax.get_ylim())
            hits = [lab.get_window_extent(r) for v, lab in zip(ax.get_yticks(), ax.get_yticklabels())
                    if lo <= v <= hi and lab.get_text() and lab.get_window_extent(r).overlaps(bb)]
            if hits:
                shift = bb.x1 - (min(e.x0 for e in hits) - 3 * fig.dpi / 72)
                t.set_x(x0 - shift / (fig.get_figwidth() * fig.dpi))
            continue
        y = top + 9 * pt                                               # 9 pt above the highest panel
        title = ax.get_title(loc="left")
        if title:
            ax.set_title(title, loc="left", fontsize=title_size, y=(y - pos.y0) / pos.height, pad=0, va="bottom")
        fig.text(x0, y, letter, fontsize=size, fontweight="bold", ha="left", va="bottom", bbox=box)


# ----------------------------------------------------------------------------- App. B: runtime table and amortized runtime
def fig6_and_table2_runtime():
    rows = []
    for s in ("id_simplex_sqeuclidean", "ood_paper_sqeuclidean"):
        for m in METHODS:
            t = timing(s, m)
            if t:
                rows.append({"set": s, "method": METHODS[m][0], "device": t.get("device", ""),
                             "single instance [s]": t["single_instance_s_median"], "batched, per instance [s]": t["batched_s_per_instance"]})
    if not rows:
        raise FileNotFoundError("no timing files")
    write_table("table2_runtime", pd.DataFrame(rows), "Time per instance (n = 30). GPU methods: single instance and batched; CPU solvers: one core.")
    t2 = timing("id_simplex_sqeuclidean", TOROT)
    k = np.arange(1, 2001); train = args.train_hours * 3600
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k, (train + k * t2["batched_s_per_instance"]) / k, color=METHODS[TOROT][1],
            label=f"ToR-OT (ours), incl. training ({args.train_hours:.1f} h, 8 GPUs)")
    ax.axhline(args.neuralot_seconds, color=METHODS["neuralot"][1], ls="--", label="NeuralOT (per instance)")
    tp = timing("id_simplex_sqeuclidean", "pot")
    if tp: ax.axhline(tp["single_instance_s_median"], color=METHODS["pot"][1], ls=":", label="POT (exact)")
    ax.set_yscale("log"); ax.xaxis.set_major_locator(mticker.MultipleLocator(250))
    ax.set_xlabel("Samples"); ax.set_ylabel("amortized time per instance [s]")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    save(fig, "fig6_amortized_runtime")


# ----------------------------------------------------------------------------- App. C.2: transpose averaging
def table4_symmetric():
    rows = []
    for s in sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{R}/sets/id_simplex_*.npz") + glob.glob(f"{R}/sets/ood_paper_*.npz")):
        for m in (TOROT, TOROT_SYM):
            df, t = per_instance(s, m), timing(s, m)
            if df is not None:
                rows.append({"set": s, "decoding": "transpose-averaged" if m == TOROT_SYM else "single pass",
                             "MAPE (mean)": df["paper_mape"].mean(), "max violation (median)": df["max_violation"].median(),
                             "time / instance [s]": t["batched_s_per_instance"] if t else np.nan})
    write_table("table4_symmetric", pd.DataFrame(rows), "Single-pass vs transpose-averaged decoding of ToR-OT.")


# ----------------------------------------------------------------------------- ToR-OT on the 5M corpus' test splits
def load_eval(split):
    path = f"{R}/eval_row_M_5M/eval_{split}{EVAL_SUFFIX}.json"
    return pd.DataFrame(json.load(open(path))["per_instance"]) if os.path.exists(path) else None


def fig_size_generalization():
    parts = [d for d in (load_eval("test_id"), load_eval("test_extrap")) if d is not None]
    if not parts: raise FileNotFoundError("eval_row_M_5M")
    df = pd.concat(parts); bins = np.arange(0, 210, 10)
    df["bin"] = pd.cut(df["n"], bins)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, lab in zip(axes, ("wmape", "support_recall", "max_violation"), ("wMAPE (median)", "support recall (mean)", "max violation (median)")):
        for c, name in enumerate(COSTS6):
            g = df[df["cost_type"] == c].groupby("bin", observed=True)[key]
            y = g.mean() if key == "support_recall" else g.median()
            ax.plot([b.mid for b in y.index], y.values, marker="o", ms=3, label=name)
        ax.axvspan(100, 205, color="0.9", zorder=0); ax.set_xlabel("support size n (shaded: beyond training range)"); ax.set_title(lab, fontsize=10)
        if key != "support_recall": ax.set_yscale("log")
    axes[-1].legend(fontsize=7, frameon=False); fig.tight_layout()
    save(fig, "fig_size_generalization")


def table4_symmetric_5M():
    rows = []
    for split in ("test_id", "test_ood", "test_extrap"):
        for tag, path in (("single pass", f"{R}/eval_row_M_5M/eval_{split}{EVAL_SUFFIX}.json"),
                          ("transpose-averaged", f"{R}/eval_row_M_5M/eval_{split}_sym{EVAL_SUFFIX}.json")):
            if os.path.exists(path):
                s = json.load(open(path))["summary"]
                rows.append({"split": split, "decoding": tag, "instances": s["instances"], "wMAPE (mean)": s["wmape/mean"],
                             "wMAPE (median)": s["wmape/median"], "support recall": s["support_recall/mean"],
                             "max violation (median)": s["max_violation/median"]})
    if not rows: raise FileNotFoundError("eval_row_M_5M")
    write_table("table4_symmetric_5M", pd.DataFrame(rows), "Single-pass vs transpose-averaged decoding on the 5M test splits "
                "(times: Table 2 / table4_symmetric, measured on an otherwise idle GPU; transpose averaging doubles them).")

def fig_latency_vs_n():
    d = json.load(open(f"{R}/latency/latency_row_M_5M{EVAL_SUFFIX}.json"))
    df = pd.DataFrame(d["sizes"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(df["n"], df["latency_single_s"], marker="o", label="ToR-OT, single instance")
    axes[0].plot(df["n"], 1.0 / df["throughput_inst_per_s"], marker="s", label=f"ToR-OT, batched ({d['batch_size']}) per instance")
    axes[0].plot(df["n"], df["pot_s"], marker="^", label="POT (exact, CPU)")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("support size n"); axes[0].set_ylabel("time per instance [s]")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].plot(df["n"], df["tokens_per_plan"], marker="o"); axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("support size n"); axes[1].set_ylabel("tokens per plan")
    slope = d.get("loglog_slope", {})
    fig.suptitle("log-log slopes: " + ", ".join(f"{k} {v:.2f}" for k, v in slope.items()), fontsize=9); fig.tight_layout()
    save(fig, "fig_latency_vs_n")

# Test vs OOD lines of one ToR-OT model (App. "Scaling and ablations"): both in ToR-OT's green, told apart by shade,
# line style and marker. key: (label, color, line style, marker)
SPLIT_STYLE = {"test_id": ("Test set", "#009e73", "-", "o"), "test_ood": ("OOD set", "#006d2c", "--", "s")}


def fig_appG_training_curve():
    """Checkpoints of the 1D model (25k-200k steps) on fixed subsets of its corpus' test splits (1,000 test_id
    instances, n = 5..100; the 864 named-distribution pairs of test_ood, n = 30): median MAPE against POT's plan,
    mean support recall and median largest marginal residual vs training rows seen."""
    log = [json.loads(l) for l in open(f"{TRAIN_RUN}/log.jsonl")]
    rows_per_step = 8 * np.mean([r["examples"] for r in log if "examples" in r])      # 8 GPUs (DDP)
    pts = []
    for path in sorted(glob.glob(f"{R}/data_curve/step*_test_id{EVAL_SUFFIX}.json")):
        step = int(os.path.basename(path)[4:10])
        for split in ("test_id", "test_ood"):
            p = path.replace("test_id", split)
            if os.path.exists(p):
                s = json.load(open(p))["summary"]
                pts.append({"step": step, "rows_seen_M": step * rows_per_step / 1e6, "split": split, **{k: s[k] for k in ("wmape/median", "wmape/mean", "support_recall/mean", "max_violation/median")}})
    if not pts: raise FileNotFoundError("data_curve")
    df = pd.DataFrame(pts); df.to_csv(data_file("data_curve", "csv"), index=False); save_data("fig_appG_training_curve", df)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
    panels = (("wmape/median", "MAPE vs POT (median)", True), ("support_recall/mean", "support recall (mean)", False),
              ("max_violation/median", "largest marginal residual\n(median)", True))
    for ax, (key, ylab, log_y) in zip(axes, panels):
        for split, (lab, color, ls, mk) in SPLIT_STYLE.items():
            d = df[df.split == split].sort_values("rows_seen_M")
            ax.plot(d["rows_seen_M"], d[key], color=color, ls=ls, marker=mk, ms=5, lw=1.6, label=lab)
        ax.set_xlabel("training rows seen (millions)"); ax.set_ylabel(ylab)
        if log_y:
            ax.set_yscale("log"); ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            lo, hi = ax.get_ylim()
            ticks = [m * 10.0 ** k for k in range(-6, 1) for m in (1, 2, 3, 5) if lo <= m * 10.0 ** k <= hi]
            ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(0, df["rows_seen_M"].max() * 1.05); closed_box(ax); ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, "appendix_G/fig_appG_training_curve", formats=["pdf"])


def draw_example(fig, gs_row, z, P, square=False):
    """Source / target histograms, exact plan (POT) and predicted plan of one instance (Fig. 3); returns the axes.
    square=True: the four panels are squares of the same size, bins are numbered 1..n and every bin axis has the same
    ticks (1, every 10 bins, and n, so the number of bins can be read off)."""
    n = len(z["a"])
    axes = []
    first = 1 if square else 0                                         # number of the first bin
    ticks = sorted({1, *range(10, n + 1, 10), n})
    for c, (vec, title) in enumerate(((z["a"], r"Source $\mu$ (1D)"), (z["b"], r"Target $\nu$ (1D)"))):
        axh = fig.add_subplot(gs_row[c]); axes.append(axh)
        axh.bar(np.arange(len(vec)) + first, vec, width=1.0, color=CMAP["histogram"], edgecolor="white", linewidth=0.4)
        axh.set_title(title, fontsize=10); axh.set_xlabel("Bin index", fontsize=9); axh.tick_params(labelsize=8)
        if c == 0: axh.set_ylabel("Probability", fontsize=9)
        if square:
            axh.set_box_aspect(1); axh.set_xlim(0.5, n + 0.5); axh.set_xticks(ticks)
    vmax = max(z["plan"].max(), P.max())
    for c, (M, title, cmap) in enumerate(((z["plan"], r"$\pi$ Ground Truth (POT)", CMAP["plans"]["exact"]),
                                          (P, r"$\hat\pi$ Ours (ToR-OT)", CMAP["plans"]["torot"]))):
        axp = fig.add_subplot(gs_row[2 + c]); axes.append(axp)
        axp.imshow(M, origin="lower", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest",
                   extent=(first - 0.5, first + n - 0.5, first - 0.5, first + n - 0.5))
        axp.set_title(title, fontsize=10); axp.set_xlabel("Target bins", fontsize=9); axp.tick_params(labelsize=8)
        if c == 0: axp.set_ylabel("Source bins", fontsize=9)
        if square:
            axp.set_box_aspect(1); axp.set_xticks(ticks); axp.set_yticks(ticks)
    return axes


# ----------------------------------------------------------------------------- shared constants
SAME_COSTS = list(STYLE["costs"].items())               # the corpus' six cost types: (key, label)
SEEDS = (1, 2)                                          # extra training seeds of the learned baselines (<method>_seed<s>)


# ----------------------------------------------------------------------------- App. C: learned baselines + k Sinkhorn iterations
def fig_warmstart():
    """MAPE of Meta OT / UNOT followed by k Sinkhorn iterations, and of Sinkhorn from its standard start, vs k and
    vs time per instance (batched, GPU); ToR-OT as a reference line / point."""
    files = sorted(glob.glob(f"{R}/warmstart/*.json"))
    if not files: raise FileNotFoundError("warmstart/")
    data = {}
    for f in files:
        d = json.load(open(f)); name = os.path.basename(f)[:-5]
        data[name.split("__")[0], d["method"]] = d
    style = {"metaot": (f"{METHODS['metaot'][0]} + k Sinkhorn", METHODS["metaot"][1], "-"), "unot": (f"{METHODS['unot'][0]} + k Sinkhorn", METHODS["unot"][1], "-"),
             "sinkhorn_cold_eps0.01": ("Sinkhorn ε=0.01, cold start", METHODS["sinkhorn_eps0.01"][1], "--"), "sinkhorn_cold_eps0.001": ("Sinkhorn ε=0.001, cold start", METHODS["sinkhorn_eps0.001"][1], ":")}
    rows = []
    for xkey, xlab, tag in (("k", "Sinkhorn iterations k", "iters"), ("batched_s_per_instance", "time per instance, batched GPU [s]", "time")):
        fig, axes = plt.subplots(2, len(SAME_COSTS), figsize=(3.0 * len(SAME_COSTS), 5.8), sharey="row")
        for r, prefix in enumerate(("id_simplex", "ood_paper")):
            for c, (cost, label) in enumerate(SAME_COSTS):
                ax, name = axes[r, c], f"{prefix}_{cost}"
                for m, (lab, color, ls) in style.items():
                    d = data.get((name, m))
                    if d is None: continue
                    x = [(b["k"] + 1) if xkey == "k" else b[xkey] for b in d["by_k"]]
                    ax.plot(x, [b["paper_mape_mean"] for b in d["by_k"]], ls, color=color, marker=".", ms=3, label=lab)
                    if tag == "iters":
                        rows += [{"set": name, "method": m, "eps": d["eps"], "k": b["k"], "MAPE": b["paper_mape_mean"],
                                  "time_batched_s": b["batched_s_per_instance"], "time_single_s": b["single_instance_s_median"]} for b in d["by_k"]]
                df = per_instance(name, TOROT)
                if df is not None:
                    y = float(df["paper_mape"].mean())
                    if xkey == "k":
                        ax.axhline(y, color=METHODS[TOROT][1], lw=1.2, label=METHODS[TOROT][0])
                    else:
                        t = timing(name, TOROT)
                        if t: ax.plot([t["batched_s_per_instance"]], [y], "*", color=METHODS[TOROT][1], ms=9, label=METHODS[TOROT][0])
                ax.set_xscale("log"); ax.set_yscale("log")
                if r == 0: ax.set_title(label, fontsize=10)
                if r == 1: ax.set_xlabel("k + 1" if xkey == "k" else "s / instance", fontsize=9)
                if c == 0: ax.set_ylabel(("test set" if prefix == "id_simplex" else "OOD (paper set)") + "\nMAPE", fontsize=10)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
        fig.suptitle(f"learned initialisation + Sinkhorn: MAPE vs {xlab}", fontsize=11); fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        save(fig, f"fig_appC_warmstart_{tag}")
    pd.DataFrame(rows).to_csv(data_file("warmstart", "csv"), index=False)


# ----------------------------------------------------------------------------- ablations of the small testbed
ABLATIONS = [("base", "base recipe"), ("base_seed1", "base recipe, seed 1"), ("no_transpose", "no transpose augmentation"),
             ("plain_ce", "plain cross-entropy (no digit weights, no ordinal smoothing)"), ("no_leftover", "no capacity perturbation"),
             ("d64", "model d = 64"), ("d128", "smaller model (d = 128)"), ("d384", "model d = 384"), ("d512", "larger model (d = 512)"),
             ("corpus10", "10% of the training corpus"), ("corpus25", "25% of the training corpus"), ("corpus50", "50% of the training corpus")]


def table_ablations():
    """Ablations of the small row-mode testbed (torot/configs/ablation_n30.yaml; n <= 30, 11k steps, one GPU), every
    configuration with up to three training seeds (<name>, <name>_s1, <name>_s2; base's seed 1 is base_seed1)."""
    rows = []
    copies = os.path.join(os.path.dirname(data_file("table_ablations_n30", "csv")), "ablations_n30")   # next to the table's data
    os.makedirs(copies, exist_ok=True)
    for name, what in ABLATIONS:
        if name == "base_seed1": continue
        for seed, run_name in ((0, name), (1, "base_seed1" if name == "base" else f"{name}_s1"), (2, f"{name}_s2")):
            run = f"{ABL}/{run_name}"
            if not os.path.exists(f"{run}/eval_test_ood.json"): continue
            params = next((l.split("parameters:")[1].split(",")[0].strip() for l in open(f"{run}/train.log") if "parameters:" in l), "")
            row = {"ablation": name, "change": what, "seed": seed, "parameters": params}
            for split in ("test_id", "test_ood"):
                v = json.load(open(f"{run}/eval_{split}.json"))["summary"]
                row.update({f"{split} wMAPE med.": v["wmape/median"], f"{split} wMAPE mean": v["wmape/mean"],
                            f"{split} recall": v["support_recall/mean"], f"{split} max viol. med.": v["max_violation/median"]})
                json.dump({"summary": v}, open(f"{copies}/{run_name}_{split}.json", "w"), indent=1)   # self-contained copy
            rows.append(row)
    if not rows: raise FileNotFoundError(ABL)
    df = pd.DataFrame(rows); df.to_csv(data_file("table_ablations_n30_runs", "csv"), index=False)
    metrics = [c for c in df.columns if c.startswith("test_")]
    agg = []
    for (name, what), d in df.groupby(["ablation", "change"], sort=False):
        row = {"ablation": name, "change": what, "seeds": len(d), "parameters": d["parameters"].iloc[0]}
        for c in metrics:
            row[c] = f"{d[c].mean():.4g} ± {d[c].std(ddof=1):.2g}" if len(d) > 1 else f"{d[c].iloc[0]:.4g}"
        agg.append(row)
    write_table("table_ablations_n30", pd.DataFrame(agg),
                "Ablations on the small testbed (row mode, n <= 30, 11k steps; last row decoded by the model): mean ± std "
                "over training seeds (per run: table_ablations_n30_runs.csv).")


# ----------------------------------------------------------------------------- plan metrics recomputed from predictions


def cost_matrix_for(s):
    from torot.data.cost import cost_matrix
    from torot.data.supports import grid_1d
    return cost_matrix(grid_1d(int(s["n"])), grid_1d(int(s["n"])), str(s["cost"]))


def score_paper_mape(R_, P):
    idx = np.union1d(np.flatnonzero(R_), np.flatnonzero(P))
    return float(np.abs(R_.ravel()[idx] - P.ravel()[idx]).sum() / max(np.abs(R_.ravel()[idx]).sum(), 1e-12))


# ----------------------------------------------------------------------------- shared helpers for the appendix figures
def plan_metric_arrays(z, idx, plans):
    """MAE / MAPE / Hellinger / |OT cost diff| / TV (as in score.py) of plans for instances idx of set z."""
    C = cost_matrix_for(z)
    out = {k: [] for k, _ in PLAN_METRICS}
    for k, P in zip(idx, plans):
        Rr = z["plan"][k]; d = Rr - P
        out["paper_mape"].append(score_paper_mape(Rr, P))
        out["mae"].append(float(np.abs(d[d != 0]).mean()) if (d != 0).any() else 0.0)
        out["hellinger"].append(float(np.sqrt(((np.sqrt(np.clip(Rr, 0, None)) - np.sqrt(np.clip(P, 0, None))) ** 2).sum()) / np.sqrt(2.0)))
        out["tv"].append(float(0.5 * np.abs(d).sum()))
        out["ot_cost_abs_diff"].append(abs(float((P * C).sum() - z["ot_cost"][k])))
    return {k: np.array(v) for k, v in out.items()}


def neuralot_metrics(name):
    """NeuralOT on a set: all instances (scores/) or the subset run (preds_subset/), else None."""
    df = per_instance(name, "neuralot")
    if df is not None:
        return {k: df[k].to_numpy() for k, _ in PLAN_METRICS}, len(df)
    f = f"{R}/preds_subset/{name}__neuralot.npz"
    if os.path.exists(f):
        zz = np.load(f)
        return plan_metric_arrays(np.load(f"{R}/sets/{name}.npz"), zz["idx"], zz["plans"]), len(zz["idx"])
    return None


def stat_rows(set_name, method, metric, values):
    """Box-plot statistics of one method / metric on one set (for figures/data/)."""
    v = np.asarray(values, dtype=float)
    return {"set": set_name, "method": method, "metric": metric, "instances": len(v), "mean": float(np.mean(v)),
            "median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75)),
            "min": float(np.min(v)), "max": float(np.max(v))}

def appendix_methods(name, root=None):
    """Fig. 3's methods that have reference scores on set ``name``, in Fig. 3's order (ToR-OT last)."""
    return [m for m in FIG3_METHODS if per_instance_ref(name, m, root) is not None]


# App. E metrics: every method against the solution of the problem it solves (score_reference.py), as Fig. 3
APPE_METRICS = [("mape_vs_reference", "MAPE"), ("mae_vs_reference", "MAE"), ("hellinger_vs_reference", "Hellinger"),
                ("tv_vs_reference", "TV"), ("ot_cost_diff_vs_reference", "|OT cost diff.|")]
APPE_SETS = [("id_simplex", "Test set"), ("ood_paper", "OOD set")]


def appx(app):
    """App. E (1D, n = 30) or App. F (2D, 6 x 6 grid, n = 36): results root, (set prefix, title), the OOD distributions
    (key, label) in the order drawn, the baseline of the improvement heatmaps (the strongest one), output folder, file tag."""
    if app == "E":
        return {"root": R, "sets": APPE_SETS, "dists": PAPER144, "vs": "metaot", "dir": "appendix_E", "tag": "appE"}
    return {"root": R2D, "sets": [("id_simplex2d", "Test set"), ("ood2d_paper", "OOD set")],
            "dists": OOD2D, "vs": "metaot", "dir": "appendix_F", "tag": "appF"}


OOD2D_EXCLUDE = list(STYLE.get("ood_2d_exclude") or [])
OOD2D = [(k, v) for k, v in STYLE["ood_2d"].items() if k not in OOD2D_EXCLUDE]   # 2D OOD distributions drawn


def ood_keep(set_name, root=None):
    """Mask of the instances of a 2D OOD set whose source and target are both drawn (plot_style.yaml: ood_2d minus
    ood_2d_exclude); None for every other set or when nothing is excluded."""
    if not set_name.startswith("ood2d_") or not OOD2D_EXCLUDE:
        return None
    z = np.load(f"{root or R}/sets/{set_name}.npz")
    return ~(np.isin(z["src"], OOD2D_EXCLUDE) | np.isin(z["tgt"], OOD2D_EXCLUDE))


def keep_rows(df, set_name, root=None):
    """``df`` (per-instance scores with an ``index`` column) without the excluded 2D OOD pairs."""
    keep = ood_keep(set_name, root)
    return df if df is None or keep is None else df[keep[df["index"].to_numpy()]].reset_index(drop=True)


def appE_label(m):
    return "ToR-OT (ours)" if m == TOROT else METHODS[m][0]


def appE_ref(set_name, method, root=None):
    """Reference scores with every App. E metric (without excluded 2D OOD pairs), or an error asking to rerun
    score_reference.py."""
    df = keep_rows(per_instance_ref(set_name, method, root), set_name, root)
    if df is not None and APPE_METRICS[1][0] not in df.columns:
        raise FileNotFoundError(f"{set_name}__{method}: no {APPE_METRICS[1][0]}; rerun reproduce/score_reference.py")
    return df


def appE_box(ax, values, positions, methods, width):
    """Box plots (quartiles, median line, whiskers at 1.5 IQR, no outliers) in each method's color."""
    bp = ax.boxplot([np.clip(v, FLOOR, None) for v in values], positions=positions, widths=width, patch_artist=True,
                    showfliers=False, medianprops={"color": "black", "lw": 1.1})
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(METHODS[m][1]); patch.set_edgecolor("0.2")


def closed_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("black")


# ----------------------------------------------------------------------------- App. C: Sinkhorn projection of ToR-OT's plans
def sinkhorn_project(P, a, b, k):
    """k alternating row / column rescalings of nonnegative plans [N, n, m] (zero entries stay zero)."""
    P = P.copy()
    for _ in range(k):
        r = P.sum(2, keepdims=True); P = P * np.where(r > 0, a[:, :, None] / np.maximum(r, 1e-300), 0.0)
        c = P.sum(1, keepdims=True); P = P * np.where(c > 0, b[:, None, :] / np.maximum(c, 1e-300), 0.0)
    return P


def table_appC_sinkhorn():
    """App. C, Table 3: ToR-OT's predicted plans followed by k Sinkhorn projections (support kept)."""
    rows = []
    for prefix in ("id_simplex", "ood_paper"):
        for cost, _ in SAME_COSTS:
            name = f"{prefix}_{cost}"
            if not os.path.exists(f"{R}/preds/{name}__{TOROT}.npy"): continue
            z, P0 = np.load(f"{R}/sets/{name}.npz"), np.load(f"{R}/preds/{name}__{TOROT}.npy")
            for k in (0, 1, 5, 10, 50):
                P = sinkhorn_project(P0, z["a"], z["b"], k)
                viol = np.maximum(np.abs(P.sum(2) - z["a"]).max(1), np.abs(P.sum(1) - z["b"]).max(1))
                met = plan_metric_arrays(z, range(len(P)), P)
                rows.append({"set": name, "Sinkhorn projections": k, "max marginal violation (mean)": viol.mean(),
                             "max marginal violation (median)": np.median(viol), "MAPE (mean)": met["paper_mape"].mean(),
                             "|OT cost diff.| (mean)": met["ot_cost_abs_diff"].mean()})
    if not rows: raise FileNotFoundError(f"{TOROT} predictions")
    df = pd.DataFrame(rows); df.to_csv(data_file("table3_sinkhorn_projection_all", "csv"), index=False)
    agg = df.groupby(["Sinkhorn projections"]).mean(numeric_only=True).reset_index()
    write_table("table3_sinkhorn_projection", agg, "App. C: ToR-OT followed by k Sinkhorn projections of its plan, mean over all "
                "test and OOD sets of the six costs (per set: table3_sinkhorn_projection_all.csv).")


# ----------------------------------------------------------------------------- App. E: per-cost summaries
def fig_appE_cost_summary(app="E"):
    """App. E.1-E.6: one figure per cost. Rows: test set (1,000 pairs), OOD set (144 pairs). Columns: five plan metrics
    (box plots over instances) and time per instance, for Fig. 3's methods, each scored against the solution of the
    problem it solves (score_reference.py); learned baselines: training seed 0. app="F": the same on the 2D sets."""
    cf = appx(app)
    for cost, label in FIG3_COSTS:
        sets = [(f"{p}_{cost}", t) for p, t in cf["sets"]]
        if not all(appendix_methods(s, cf["root"]) for s, _ in sets): continue
        ncol = len(APPE_METRICS) + 1
        fig, axes = plt.subplots(2, ncol, figsize=(2.75 * ncol, 5.4))
        shown, table = [], []
        for r, (s, title) in enumerate(sets):
            ms = appendix_methods(s, cf["root"])
            dfs = {m: appE_ref(s, m, cf["root"]) for m in ms}
            for c, (key, lab) in enumerate(APPE_METRICS):
                ax = axes[r, c]
                appE_box(ax, [dfs[m][key].to_numpy() for m in ms], range(1, len(ms) + 1), ms, 0.65)
                table += [stat_rows(s, m, key, dfs[m][key].to_numpy()) for m in ms]
                ax.set_yscale("log"); ax.set_xticks([]); closed_box(ax); ax.grid(axis="y", alpha=0.25)
                if r == 0: ax.set_title(lab, fontsize=10)
                if c == 0: ax.set_ylabel(f"{title}\n({len(dfs[ms[-1]]):,} pairs)", fontsize=10)
            ax = axes[r, -1]                                       # time per instance (NeuralOT: not comparable)
            for x, m in enumerate(ms):
                t = timing(s, m, cf["root"])
                if t and "not comparable" not in str(t.get("note", "")) and "do not report" not in str(t.get("note", "")):
                    ax.bar(x - 0.2, t["single_instance_s_median"], width=0.4, color=METHODS[m][1], edgecolor="0.2", lw=0.5)
                    ax.bar(x + 0.2, t["batched_s_per_instance"], width=0.4, color=METHODS[m][1], alpha=0.55, hatch="///",
                           edgecolor="0.2", lw=0.5)
                elif m == "neuralot":
                    ax.text(x, 0.5, "n/a", transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=8, color="0.3")
            ax.set_yscale("log"); ax.set_xticks([]); ax.set_xlim(-0.6, len(ms) - 0.4); closed_box(ax); ax.grid(axis="y", alpha=0.25)
            if r == 0: ax.set_title("time / instance [s]", fontsize=10)
            shown += [m for m in ms if m not in shown]
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=METHODS[m][1], edgecolor="0.2") for m in shown]
        handles += [plt.Rectangle((0, 0), 1, 1, facecolor="0.6", edgecolor="0.2"),
                    plt.Rectangle((0, 0), 1, 1, facecolor="0.6", alpha=0.55, hatch="///", edgecolor="0.2")]
        labels = [appE_label(m) for m in shown] + ["time: one instance", "time: per instance, batched"]
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
        fig.suptitle(f"Cost {label}" + (", 2D" if app == "F" else ""), fontsize=12); fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        save(fig, f"{cf['dir']}/fig_{cf['tag']}_summary_{cost}"); save_data(f"fig_{cf['tag']}_summary_{cost}", pd.DataFrame(table))


def fig_appE_pairs(app="E"):
    """App. E: the 144 OOD pairs (rows: source, columns: target) for every method, one page per method and cost.
    app="F": the 49 pairs of the 2D OOD set, in the order of App. F's distribution figure."""
    cf = appx(app)
    for cost, label in SAME_COSTS:
        s = f"{cf['sets'][1][0]}_{cost}"
        z = np.load(f"{cf['root']}/sets/{s}.npz")
        if app == "E":
            fams = list(dict.fromkeys(z["src"].tolist())); names = dict(zip(fams, fams))
        else:
            fams = [k for k, _ in cf["dists"]]; names = {k: f"{i + 1}: {t}" for i, (k, t) in enumerate(cf["dists"])}
        pos = {(a, b): k for k, (a, b) in enumerate(zip(z["src"].tolist(), z["tgt"].tolist()))}
        plans = [("exact", "POT (exact)", z["plan"], CMAP["plans"]["exact"])]
        for m, cmap in ((TOROT, CMAP["plans"]["torot"]), ("metaot", CMAP["plans"]["metaot"]), ("unot", CMAP["plans"]["unot"]),
                        ("neuralot", CMAP["plans"]["neuralot"])):                         # Fig. 3's methods
            if os.path.exists(f"{cf['root']}/preds/{s}__{m}.npy"):
                eps = (timing(s, m, cf["root"]) or {}).get("eps")  # entropic methods: say which regularisation they solve
                title = appE_label(m) + (f" (entropic OT, ε = {eps:g}: blurred by design)" if eps else "")
                plans.append((m, title, np.load(f"{cf['root']}/preds/{s}__{m}.npy"), cmap))
        size, fs = ((13, 13.5), 6) if app == "E" else ((10, 10.6), 8)
        for key, title, P, cmap in plans:
            fig, axes = plt.subplots(len(fams), len(fams), figsize=size)
            for i, a in enumerate(fams):
                for j, b in enumerate(fams):
                    ax = axes[i, j]; k = pos[(a, b)]
                    ax.imshow(P[k], origin="lower", cmap=cmap, vmin=0, vmax=max(P[k].max(), 1e-12), interpolation="nearest")
                    ax.set_xticks([]); ax.set_yticks([])
                    if j == 0: ax.set_ylabel(names[a], fontsize=fs, rotation=0, ha="right", va="center")
                    if i == len(fams) - 1: ax.set_xlabel(names[b], fontsize=fs, rotation=90)
            fig.suptitle(f"{title}: plans of the {len(fams) ** 2} OOD pairs, cost {label}" + (", 2D" if app == "F" else "")
                         + " (row: source, column: target)", fontsize=11)
            fig.subplots_adjust(left=0.1 if app == "E" else 0.15, right=0.99, top=0.96, bottom=0.08 if app == "E" else 0.13,
                                wspace=0.05, hspace=0.05)
            save(fig, f"{cf['dir']}/fig_{cf['tag']}_pairs_{cost}_{key}")


def fig_appE_allcosts(app="E"):
    """App. E.7: Fig. 3's methods on all six costs, one panel per plan metric (box plots over instances), one figure for
    the test set and one for the OOD set; each method against the solution of its own problem (score_reference.py)."""
    cf = appx(app)
    for prefix, _ in cf["sets"]:
        methods = [m for m in FIG3_METHODS if any(per_instance_ref(f"{prefix}_{k}", m, cf["root"]) is not None for k, _ in FIG3_COSTS)]
        fig, axes = plt.subplots(len(APPE_METRICS), 1, figsize=(12, 2.2 * len(APPE_METRICS)), sharex=True)
        width, table = 0.8 / len(methods), []
        for ax, (key, lab) in zip(axes, APPE_METRICS):
            for j, m in enumerate(methods):
                for g, (cost, _) in enumerate(FIG3_COSTS):
                    df = appE_ref(f"{prefix}_{cost}", m, cf["root"])
                    if df is None: continue
                    table.append(stat_rows(f"{prefix}_{cost}", m, key, df[key].to_numpy()))
                    appE_box(ax, [df[key].to_numpy()], [g + (j - (len(methods) - 1) / 2) * width], [m], width * 0.85)
            ax.set_yscale("log"); ax.set_ylabel(lab, fontsize=10); ax.grid(axis="y", alpha=0.25); closed_box(ax)
            for g in range(1, len(FIG3_COSTS)):
                ax.axvline(g - 0.5, color="0.85", lw=0.8, zorder=0)
        axes[-1].set_xlim(-0.5, len(FIG3_COSTS) - 0.5)
        axes[-1].set_xticks(range(len(FIG3_COSTS))); axes[-1].set_xticklabels([l for _, l in FIG3_COSTS], fontsize=10.5)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=METHODS[m][1], edgecolor="0.2") for m in methods]
        fig.legend(handles, [appE_label(m) for m in methods], loc="upper center", ncol=len(methods), frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        save(fig, f"{cf['dir']}/fig_{cf['tag']}_allcosts_{prefix}"); save_data(f"fig_{cf['tag']}_allcosts_{prefix}", pd.DataFrame(table))


def fig_appE_improvement(app="E"):
    """App. E.7: per cost, the relative MAPE improvement of ToR-OT over Meta OT (the strongest baseline on every cost)
    on the 144 OOD pairs, 100 (MAPE_MetaOT - MAPE_ToR-OT) / MAPE_MetaOT, each against its own reference; as Fig. 4B.
    One figure per cost and one with all six. app="F": the 49 pairs of the 2D OOD set."""
    cf = appx(app)
    vs, vs_label = cf["vs"], METHODS[cf["vs"]][0]
    names = [n for n, _ in cf["dists"]]
    ticks = [f"{i + 1}: {t}" for i, (_, t) in enumerate(cf["dists"])]
    small = len(names) < 10                                           # 2D: 7 x 7 pairs, larger numbers
    rows = []
    fig_all, axs = plt.subplots(2, 3, figsize=(19, 12) if not small else (15, 9.4))
    for ax_all, (cost, label) in zip(axs.ravel(), FIG3_COSTS):
        s = f"{cf['sets'][1][0]}_{cost}"
        o, v = mape_ref(s, vs, cf["root"]), mape_ref(s, TOROT, cf["root"])
        if o is None or v is None: raise FileNotFoundError(s)
        z = np.load(f"{cf['root']}/sets/{s}.npz")
        pos = {(a, b): k for k, (a, b) in enumerate(zip(z["src"].tolist(), z["tgt"].tolist()))}
        G = np.array([[100 * (o[pos[a, b]] - v[pos[a, b]]) / max(o[pos[a, b]], 1e-12) for b in names] for a in names])
        D = np.array([[o[pos[a, b]] - v[pos[a, b]] for b in names] for a in names])     # absolute: drawn on the diagonal
        rows += [{"cost": cost, "source": a, "target": b, f"improvement_over_{vs}_%": G[i, j], f"mape_{vs}_minus_torot": D[i, j]}
                 for i, a in enumerate(names) for j, b in enumerate(names)]
        title = f"{label}: ToR-OT better on {int((G > 0).sum())} of {G.size} pairs" + ("\n" if small else ", ") + f"median {np.median(G):.0f}%"
        fig, ax = plt.subplots(figsize=(7.4, 6.4) if not small else (6.2, 5.0))   # this cost alone
        im = draw_grid(ax, G, ticks, "", "imp", fig, vs=vs, fontsize=6.5 if not small else 9, colorbar=False, diag=np.diag(D))
        im_all = draw_grid(ax_all, G, ticks, "", "imp", fig_all, vs=vs, fontsize=5.5 if not small else 9, colorbar=False,
                           diag=np.diag(D))
        for a_ in (ax, ax_all):
            a_.set_xticklabels([str(i + 1) for i in range(len(names))], rotation=0)
            a_.set_title(title, fontsize=10, loc="left")
        if list(axs.ravel()).index(ax_all) % 3:                       # combined figure: names in the first column only
            ax_all.set_yticklabels([str(i + 1) for i in range(len(names))]); ax_all.set_ylabel("")
        fig.colorbar(im, ax=ax, shrink=0.8).set_label(f"improvement over {vs_label} [%]", fontsize=9)
        fig.tight_layout(); save(fig, f"{cf['dir']}/fig_{cf['tag']}_improvement_{cost}")
    fig_all.colorbar(im_all, ax=axs, shrink=0.5).set_label(f"improvement over {vs_label} [%]", fontsize=10)
    save(fig_all, f"{cf['dir']}/fig_{cf['tag']}_improvement_all"); save_data(f"fig_{cf['tag']}_improvement", pd.DataFrame(rows))


# ----------------------------------------------------------------------------- App. F: the same on the 2D sets
def fig_appF_cost_summary(): fig_appE_cost_summary("F")
def fig_appF_pairs(): fig_appE_pairs("F")
def fig_appF_allcosts(): fig_appE_allcosts("F")
def fig_appF_improvement(): fig_appE_improvement("F")


def fig_appE_distributions():
    """App. E: the 12 distributions of the 1D OOD set as 30-bin histograms, numbered as in the improvement heatmaps.
    Each pair is binned on its own grid (30 equal bins over the pooled 0.5-99.5 percentiles of the two
    standardized samples), so a histogram depends on the partner; each distribution is drawn as in its pair with
    itself (the diagonal pair), i.e. over its own range."""
    z = np.load(f"{R}/sets/ood_paper_sqeuclidean.npz")
    n = int(z["n"]); x = np.arange(1, n + 1)
    pos = {(a, b): k for k, (a, b) in enumerate(zip(z["src"].tolist(), z["tgt"].tolist()))}
    fig, axes = plt.subplots(2, 6, figsize=(14, 4.6), sharex=True)
    rows = []
    for i, (ax, (key, label)) in enumerate(zip(axes.ravel(), PAPER144)):
        a = z["a"][pos[(key, key)]]
        ax.bar(x, a, width=1.0, color=CMAP["histogram"], edgecolor="white", linewidth=0.3)
        ax.set_title(f"{i + 1}: {label}", fontsize=10); ax.set_xlim(0.5, n + 0.5); ax.set_ylim(0, None)
        ax.set_xticks([1, 10, 20, n]); ax.tick_params(labelsize=8)
        rows += [{"distribution": key, "bin": int(b), "mass": float(m)} for b, m in zip(x, a)]
    for ax in axes[-1]:
        ax.set_xlabel("bin", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("mass", fontsize=9)
    fig.tight_layout()
    save(fig, "appendix_E/fig_appE_distributions"); save_data("fig_appE_distributions", pd.DataFrame(rows))

def fig_appF_distributions():
    """App. F: the 7 distributions of the 2D OOD set as 6 x 6 histograms (the sources of the OOD pairs)."""
    cf = appx("F")
    z = np.load(f"{R2D}/sets/{cf['sets'][1][0]}_sqeuclidean.npz")
    xy = z["coords"]; xs, ys = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    ix, iy = np.searchsorted(xs, xy[:, 0]), np.searchsorted(ys, xy[:, 1])
    fig, axes = plt.subplots(1, len(cf["dists"]), figsize=(2.0 * len(cf["dists"]), 2.45))
    rows = []
    for i, (ax, (key, label)) in enumerate(zip(axes, cf["dists"])):
        a = z["a"][list(z["src"]).index(key)]
        H = np.zeros((len(ys), len(xs))); H[iy, ix] = a
        ax.imshow(H, origin="lower", cmap=CMAP["plans"]["exact"], vmin=0, interpolation="nearest",
                  extent=(xs[0] - (xs[1] - xs[0]) / 2, xs[-1] + (xs[1] - xs[0]) / 2, ys[0] - (ys[1] - ys[0]) / 2, ys[-1] + (ys[1] - ys[0]) / 2))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{i + 1}: {label}", fontsize=10)
        rows += [{"distribution": key, "x": float(xy[k, 0]), "y": float(xy[k, 1]), "mass": float(a[k])} for k in range(len(a))]
    fig.tight_layout()
    save(fig, "appendix_F/fig_appF_distributions"); save_data("fig_appF_distributions", pd.DataFrame(rows))


# ----------------------------------------------------------------------------- App. D: interpretability
def fig_appD_interpretability():
    """App. D: causal intervention, linear probes by layer, attention-head roles (reproduce/interpretability.py)."""
    d = f"{R}/interp"
    if not os.path.exists(f"{d}/probe.json"): raise FileNotFoundError(d)
    causal, probe, att = json.load(open(f"{d}/causal.json")), json.load(open(f"{d}/probe.json")), np.load(f"{d}/attention.npz")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2), gridspec_kw={"width_ratios": [1, 1.3, 1.2, 1.0]})
    keys = ["adapted (mass to j reduced)", "complete_avoidance", "mean_reduction", "respects_new_capacity"]
    for i, kind in enumerate(("control", "exhaust", "halve")):
        axes[0].bar(np.arange(len(keys)) + (i - 1) * 0.27, [causal[kind][k] for k in keys], width=0.27, label=kind)
    axes[0].set_xticks(range(len(keys))); axes[0].set_xticklabels(["adapted", "avoided", "mean\nreduction", "within new\ncapacity"], fontsize=8)
    axes[0].legend(fontsize=8, frameon=False); axes[0].set_title(f"(a) capacity intervention ({causal['interventions_per_kind']} rows)", fontsize=10)
    layers = list(probe["by_layer"])
    for t in ("row_budget", "placed_in_row", "capacity_j", "capacity_vector"):
        axes[1].plot(range(len(layers)), [probe["by_layer"][l][t] for l in layers], marker=".", label=t)
    for t, v in probe.get("control_from_capacity_j_only", {}).items():
        axes[1].axhline(v, ls=":", color="0.5", lw=0.8)
    axes[1].set_xticks(range(len(layers))); axes[1].set_xticklabels(["emb"] + [str(i) for i in range(len(layers) - 1)], fontsize=7)
    axes[1].set_xlabel("decoder layer"); axes[1].set_ylabel("held-out $R^2$"); axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_title("(b) linear probes (dotted: from $r_j$ alone)", fontsize=10)
    for ax, kind, title in ((axes[2], "cross", "(c) cross-attention"), (axes[3], "self", "(d) self-attention")):
        A = att[kind].mean(1)                                 # [layers, categories], averaged over heads
        im = ax.imshow(A.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(A.shape[1])); ax.set_yticklabels(att[f"{kind}_labels"], fontsize=8)
        ax.set_xlabel("decoder layer"); ax.set_title(title + ": share of attention (mass slots)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); save(fig, "fig_appD_interpretability")
    roles = {kind: {lab: int(c) for lab, c in zip(att[f"{kind}_labels"], np.bincount(att[kind][-1].argmax(-1), minlength=len(att[f"{kind}_labels"])))}
             for kind in ("cross", "self")}
    json.dump({"last_layer_head_roles (argmax category)": roles, "causal": causal,
               "probe_last_layer": probe["by_layer"][layers[-1]], "probe_embedding": probe["by_layer"]["embedding"]},
              open(data_file("appD_interpretability_summary", "json"), "w"), indent=1)


# ----------------------------------------------------------------------------- scaling with the number of bins (n = 100)
def per_instance_root(root, set_name, method):
    path = f"{root}/scores/per_instance/{set_name}__{method}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


def neuralot_root(root, name):
    """NeuralOT on set `name` of `root`: (per-instance metrics, True if only a subset of the instances), else (None, None)."""
    df = per_instance_root(root, name, "neuralot")
    if df is not None:
        return df, False
    f = f"{root}/preds_subset/{name}__neuralot.npz"
    if not os.path.exists(f):
        return None, None
    zz = np.load(f)
    return pd.DataFrame(plan_metric_arrays(np.load(f"{root}/sets/{name}.npz"), zz["idx"], zz["plans"])), True


def fig_scaling_n100():
    """MAPE of every method at n = 30, 100 and 200 (same construction of the sets; the learned baselines are retrained
    at every n, ToR-OT is the one model trained on n <= 100, so n = 200 is extrapolation for it). NeuralOT (trained per
    instance) where it was run; hatched bars: an evenly spaced subset of the instances (see table_neuralot_subset_n100)."""
    if not os.path.exists(f"{R100}/scores"): raise FileNotFoundError(R100)
    roots = [(30, os.path.abspath(R)), (100, R100)] + ([(200, R200)] if os.path.exists(f"{R200}/scores") else [])
    rows = []
    for prefix in ("id_simplex", "ood_paper"):
        for cost, _ in SAME_COSTS:
            for m in ("neuralot", "metaot", "unot", TOROT):
                for n, root in roots:
                    if m == "neuralot":
                        df, subset = neuralot_root(root, f"{prefix}_{cost}")
                    else:
                        df, subset = per_instance_root(root, f"{prefix}_{cost}", m), False
                    if df is not None:
                        rows.append({"split": prefix, "cost": cost, "method": m, "n": n, "instances": len(df), "subset": bool(subset),
                                     "MAPE (mean)": df["paper_mape"].mean(), "MAPE (median)": df["paper_mape"].median(),
                                     "|OT cost diff.| (mean)": df["ot_cost_abs_diff"].mean()})
    if not any(r["n"] == 100 for r in rows): raise FileNotFoundError(f"{R100}/scores (n = 100)")
    df = pd.DataFrame(rows); save_data("fig_scaling_n100", df)
    write_table("table_scaling_n100", df, "Every method at n = 30, 100 (and 200): learned baselines retrained at each n; ToR-OT trained "
                "on n <= 100 (n = 200: extrapolation).")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    ns_all = sorted(df.n.unique())
    combos = [(m, n) for m in dict.fromkeys(df.method) for n in ns_all if ((df.method == m) & (df.n == n)).any()]
    width = 0.84 / len(combos)                    # one slot per (method, n) with results; same slots in both panels
    costs = [c for c, _ in SAME_COSTS]
    for ax, prefix in zip(axes, ("id_simplex", "ood_paper")):
        d = df[df.split == prefix]
        for j, (m, n) in enumerate(combos):
            dm = d[(d.method == m) & (d.n == n)]
            if dm.empty: continue
            xs = [costs.index(c) + (j - (len(combos) - 1) / 2) * width for c in dm.cost]
            lab, color = METHODS.get(m, (m, "0.5"))
            q = ns_all.index(n)
            bars = ax.bar(xs, dm["MAPE (mean)"], width=width, color=color, alpha=0.35 + 0.65 * q / max(len(ns_all) - 1, 1),
                          label=f"{lab}, n = {n}")
            for bar, sub in zip(bars, dm["subset"]):
                if sub: bar.set_hatch("///")
        ax.set_xticks(range(len(SAME_COSTS))); ax.set_xticklabels([l for _, l in SAME_COSTS]); ax.set_yscale("log")
        title = "test set" if prefix == "id_simplex" else "OOD (paper set)"
        sub = d[d.subset]
        if len(sub):
            title += f"\n(hatched: {', '.join(sorted({METHODS[m][0] for m in sub.method}))} on {int(sub.instances.min())} of " \
                     f"{int(d[~d.subset].instances.max())} instances, n = {', '.join(str(n) for n in sorted(sub.n.unique()))})"
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("MAPE (mean)")
    handles, labels = axes[0].get_legend_handles_labels()
    for h, l in zip(*axes[1].get_legend_handles_labels()):
        if l not in labels: handles.append(h); labels.append(l)
    fig.legend(handles, labels, fontsize=7.5, ncol=min(len(labels), 6), frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.06 * ((len(labels) + 5) // 6), 1, 1)); save(fig, "fig_scaling_n100")


# ----------------------------------------------------------------------------- Table 1
PAPER144 = list(STYLE["ood_144"].items())  # all 12 distributions of the OOD set


def table1_improvement():
    """Table 1: relative improvement of ToR-OT's mean MAPE over each learned baseline, per cost, every comparison on
    identical instances (test: NeuralOT's subset for NeuralOT, all 1,000 instances otherwise; OOD: the 144 pairs)."""
    rows = []
    for cost, label in SAME_COSTS:
        row = {"cost": label}
        for prefix, split in (("id_simplex", "test"), ("ood_paper", "OOD")):
            name = f"{prefix}_{cost}"
            tor = per_instance(name, TOROT)
            if tor is None: continue
            v = tor["paper_mape"].to_numpy()
            row[f"{split}: ToR-OT MAPE"] = v.mean()
            for m in ("neuralot", "metaot", "unot"):
                o = per_instance(name, m)
                if o is not None:
                    o, vv = o["paper_mape"].to_numpy(), v
                elif m == "neuralot" and os.path.exists(f"{R}/preds_subset/{name}__neuralot.npz"):
                    idx = np.load(f"{R}/preds_subset/{name}__neuralot.npz")["idx"]
                    o, vv = neuralot_metrics(name)[0]["paper_mape"], v[idx]
                    row[f"{split}: NeuralOT instances"] = len(idx)
                else:
                    continue
                row[f"{split}: vs {METHODS[m][0]} [%]"] = 100 * (o.mean() - vv.mean()) / o.mean()
        rows.append(row)
    if not rows: raise FileNotFoundError("scores")
    write_table("table1_improvement", pd.DataFrame(rows),
                "Relative improvement of ToR-OT's mean MAPE over each learned baseline, per cost, on identical instances "
                "(|x-y|: L1 = L2 = L-inf in 1D; its optimal plans are not unique).")


# ----------------------------------------------------------------------------- App. D across sets, testbed scaling curves
def table_appD_all_sets():
    """Interpretability on every evaluation set (main run: interp/, others: interp_sets/<set>/)."""
    runs = [("ood_paper_sqeuclidean", f"{R}/interp")] + [(os.path.basename(d), d) for d in sorted(glob.glob(f"{R}/interp_sets/*"))]
    rows = []
    for name, d in runs:
        if not os.path.exists(f"{d}/probe.json"): continue
        c, pr = json.load(open(f"{d}/causal.json")), json.load(open(f"{d}/probe.json"))
        last = [k for k in pr["by_layer"] if k != "embedding"][-1]
        rows.append({"set": name, "exhaust: adapted": c["exhaust"]["adapted (mass to j reduced)"], "exhaust: avoided": c["exhaust"]["complete_avoidance"],
                     "exhaust: to adjacent columns": c["exhaust"]["share_of_removed_mass_to_adjacent_columns"],
                     "halve: mean reduction": c["halve"]["mean_reduction"], "control: unchanged": c["control"]["unchanged_row"],
                     "R2 row budget (embedding)": pr["by_layer"]["embedding"]["row_budget"], "R2 row budget (last layer)": pr["by_layer"][last]["row_budget"],
                     "R2 row budget (from r_j only)": pr["control_from_capacity_j_only"]["row_budget"],
                     "R2 capacity vector (last layer)": pr["by_layer"][last]["capacity_vector"]})
    if len(rows) < 2: raise FileNotFoundError("interp_sets/")
    write_table("tableD_interpretability_all_sets", pd.DataFrame(rows), "App. D on every evaluation set (row model, last row decoded).")


def table_appG_support_size():
    """Mean MAPE at n = 30, 100 and 200, each method against the solution of the problem it solves (score_reference.py,
    as Fig. 3); learned baselines: mean over their training seeds, retrained at every n; ToR-OT: the one model trained on
    n <= 100, so n = 200 is extrapolation. Also the plan length and generation time of ToR-OT vs n (latency/)."""
    roots = [(30, os.path.abspath(R)), (100, R100), (200, R200)]
    rows = []
    for prefix, split in (("id_simplex", "Test set"), ("ood_paper", "OOD set")):
        for cost, label in SAME_COSTS:
            for n, root in roots:
                for m in (TOROT, "metaot", "unot"):
                    mean, seeds = seed_mean(f"{prefix}_{cost}", m, root=root)
                    if mean is not None:
                        rows.append({"split": split, "cost": cost, "cost_label": label, "n": n, "method": m,
                                     "mape_vs_reference_mean": mean, "seeds": seeds})
    if not any(r["n"] == 100 for r in rows): raise FileNotFoundError(f"{R100}/scores/per_instance_ref")
    df = pd.DataFrame(rows); save_data("table_appG_support_size", df)
    wide = df.pivot_table(index=["split", "cost_label"], columns=["n", "method"], values="mape_vs_reference_mean", sort=False)
    write_table("table_appG_support_size", wide.reset_index(), "Mean MAPE against the solution of each method's own problem "
                "at n = 30, 100, 200 (learned baselines retrained at each n, mean over seeds; ToR-OT trained on n <= 100).")
    lat = json.load(open(f"{R}/latency/latency_row_M_5M{EVAL_SUFFIX}.json"))
    save_data("table_appG_latency", pd.DataFrame(lat["sizes"]))


def fig_appG_testbed():
    """Small testbed (torot/configs/ablation_n30.yaml: n <= 30, 11k steps, same training budget for every run): median
    MAPE against POT's plan vs the fraction of the training corpus and vs the number of parameters; line: mean over the
    three training seeds, error bar: std, dots: the seeds."""
    def runs(name):
        out = []
        for run in (name, "base_seed1" if name == "base" else f"{name}_s1", f"{name}_s2"):
            f = f"{ABL}/{run}/eval_test_ood.json"
            if os.path.exists(f):
                params = next((l.split("parameters:")[1].split(",")[0].strip() for l in open(f"{ABL}/{run}/train.log")
                               if "parameters:" in l), "")
                out.append((params, {sp: json.load(open(f"{ABL}/{run}/eval_{sp}.json"))["summary"]["wmape/median"]
                                     for sp in ("test_id", "test_ood")}))
        return out
    curves = {"corpus": [(10, "corpus10"), (25, "corpus25"), (50, "corpus50"), (100, "base")],
              "width": [(64, "d64"), (128, "d128"), (256, "base"), (384, "d384"), (512, "d512")]}
    rows = []
    for curve, pts in curves.items():
        for x, name in pts:
            for seed, (params, r) in enumerate(runs(name)):
                p = float(params.rstrip("M")) if params.endswith("M") else np.nan
                rows += [{"curve": curve, "x": x, "parameters_M": p, "run": name, "seed": seed, "split": sp, "wmape_median": v}
                         for sp, v in r.items()]
    if not rows: raise FileNotFoundError(ABL)
    df = pd.DataFrame(rows); save_data("fig_appG_testbed", df)
    df["x"] = df["x"].astype(float)
    df.loc[df.curve == "width", "x"] = df.loc[df.curve == "width", "parameters_M"]          # model size in parameters
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    xlabels = {"corpus": "training corpus (% of instances)", "width": "model parameters (millions)"}
    for ax, curve in zip(axes, curves):
        for sp, (lab, color, ls, mk) in SPLIT_STYLE.items():
            d = df[(df.curve == curve) & (df.split == sp)]
            g = d.groupby("x")["wmape_median"].agg(["mean", "std"]).reset_index()
            ax.scatter(d["x"], d["wmape_median"], s=10, color=color, alpha=0.35, lw=0, zorder=2)
            ax.errorbar(g["x"], g["mean"], yerr=g["std"].fillna(0), color=color, ls=ls, marker=mk, ms=5, lw=1.6, capsize=3,
                        label=lab, zorder=3)
        xs = sorted(df[df.curve == curve]["x"].unique())
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_locator(mticker.FixedLocator(xs)); ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{x:g}%" if curve == "corpus" else f"{x:g}" for x in xs]))
        ax.set_xlabel(xlabels[curve]); closed_box(ax); ax.grid(alpha=0.25)
    axes[0].yaxis.set_minor_formatter(mticker.NullFormatter())
    lo, hi = axes[0].get_ylim()
    axes[0].yaxis.set_major_locator(mticker.FixedLocator([m * 10.0 ** k for k in range(-3, 1) for m in (1, 2, 3, 5) if lo <= m * 10.0 ** k <= hi]))
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_ylabel("MAPE vs POT (median)"); axes[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, "appendix_G/fig_appG_testbed", formats=["pdf"])


# ----------------------------------------------------------------------------- Fig. 3
FIG3_METHODS = ["neuralot", "metaot", "unot", TOROT]          # drawn in this order; ToR-OT last
FIG3_COSTS = [(k, dict(SAME_COSTS)[k]) for k in                  # |x-y|^p by exponent, then the two others
              ("euclidean", "sqrt_euclidean", "sqeuclidean", "cubic_euclidean", "canberra", "gaussian")]


def mape_ref(set_name, method, root=None):
    """Per-instance MAPE against the solution of the problem the method solves (score_reference.py), else against
    POT's plan (score.py) for methods without a reference score."""
    df = per_instance_ref(set_name, method, root)
    if df is not None:
        return df["mape_vs_reference"].to_numpy()
    df = per_instance(set_name, method, root)
    return None if df is None else df["paper_mape"].to_numpy()


def per_instance_ref(set_name, method, root=None):
    """Per-instance MAPE against the solution of the problem the method solves (reproduce/score_reference.py)."""
    path = f"{root or R}/scores/per_instance_ref/{set_name}__{method}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


def fig3_camera_ready():
    """Figure 3: ToR-OT, NeuralOT, Meta OT and UNOT on the same instances (top: 1,000 test pairs per
    cost; bottom: the paper's OOD pairs), mean MAPE per cost on a log axis. Each method is scored against the solution
    of the problem it solves (score_reference.py): Meta OT and UNOT (entropic OT) against the converged Sinkhorn plan
    at their eps; ToR-OT and NeuralOT (exact OT) against the optimal plan closest to the prediction (POT's plan when
    the optimum is unique). Baseline bars: mean over their three training seeds (the seed std is in the data file).
    Right: source, target, POT plan and ToR-OT plan for the median-error ToR-OT instance of |x-y|^2."""
    # variant: (file tag, column of per_instance_ref, y label, value-label format); the same example in both
    for tag, col, ylab, fmt in (("mape", "mape_vs_reference", "MAPE", "{:.3f}"),
                                ("otcost_diff", "ot_cost_diff_vs_reference", "|OT cost diff.|", "{:.1e}")):
        rows, examples, bar_axes, example_axes = [], [], [], []
        # bars sorted from the highest to the lowest error (mean of log10 over every cost and both sets), in the same
        # order in every group and in the legend
        def log_mean(m):
            v = [seed_mean(f"{p}_{c}", m, col)[0] for p in ("id_simplex", "ood_paper") for c, _ in FIG3_COSTS]
            return np.mean([np.log10(x) for x in v if x])
        order = sorted(FIG3_METHODS, key=lambda m: -log_mean(m))
        fig = plt.figure(figsize=(17, 4.9))                            # rows about as tall as the square panels
        gs = fig.add_gridspec(2, 5, width_ratios=[3.4, 1, 1, 1, 1], wspace=0.32, hspace=0.6)
        width = 0.8 / len(FIG3_METHODS)
        for r, prefix in enumerate(("id_simplex", "ood_paper")):          # top: test set, bottom: OOD set
            ax = fig.add_subplot(gs[r, 0])
            drawn = []
            for j, m in enumerate(order):
                xs, hs, errs = [], [], []
                for g, (cost, _) in enumerate(FIG3_COSTS):
                    df = per_instance_ref(f"{prefix}_{cost}", m)
                    if df is None:
                        continue
                    seed_dfs = [df] + [d for d in (per_instance_ref(f"{prefix}_{cost}", f"{m}_seed{sd}") for sd in SEEDS) if d is not None]
                    per_seed = [float(d[col].mean()) for d in seed_dfs]
                    xs.append(g + (j - (len(FIG3_METHODS) - 1) / 2) * width); hs.append(float(np.mean(per_seed)))
                    errs.append(float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0)
                    rows.append({"split": prefix, "cost": cost, "method": m, "instances": len(df), "seeds": len(per_seed),
                                 f"{col}_mean": hs[-1], "seed_std": errs[-1], f"{col}_median_seed0": float(df[col].median()),
                                 "mape_vs_pot_mean_seed0": float(df["mape_vs_pot"].mean()),
                                 "references": "; ".join(sorted(set(d["reference"].iloc[0] for d in seed_dfs)))})
                drawn += hs
                label, color = METHODS[m]
                if m == TOROT:
                    label = "ToR-OT (ours)"
                ax.bar(xs, hs, width=width, color=color, label=label, edgecolor="white", linewidth=0.6)
                if m == TOROT:                                 # direct labels for ours; every value is in the data file
                    for x, h in zip(xs, hs):
                        ax.text(x, h * 1.12, fmt.format(h).replace("e-0", "e-"), ha="center", va="bottom",
                                fontsize=7.5, color="0.15", rotation=90)
            ax.set_yscale("log")
            ax.set_ylim(4e-3, 4) if tag == "mape" else ax.set_ylim(min(drawn) / 3, max(drawn) * 3)
            ax.set_ylabel(ylab, fontsize=11)
            ax.grid(axis="y", alpha=0.25, which="major"); ax.set_axisbelow(True)
            for spine in ax.spines.values():                           # closed black box, as the panels on the right
                spine.set_visible(True); spine.set_color("black")
            ax.set_xticks(range(len(FIG3_COSTS))); ax.set_xticklabels([l for _, l in FIG3_COSTS], fontsize=10.5)
            bar_axes.append(ax)
            if r == 0:
                legend = ax.get_legend_handles_labels()
            ex = f"{prefix}_sqeuclidean"
            z, P = np.load(f"{R}/sets/{ex}.npz"), np.load(f"{R}/preds/{ex}__{TOROT}.npy")
            err = per_instance_ref(ex, TOROT)["mape_vs_reference"].to_numpy()
            k = int(np.argsort(err)[len(err) // 2])
            examples.append({"set": ex, "index": k, "src": str(z["src"][k]), "tgt": str(z["tgt"][k]), "method": TOROT,
                             "mape_vs_reference": float(err[k])})
            example_axes.append(draw_example(fig, [gs[r, 1 + c] for c in range(4)],
                                             {key: z[key][k] for key in ("a", "b", "plan")}, P[k], square=True))
        fig.canvas.draw()                                              # places the square panels
        for ax, row_axes in zip(bar_axes, example_axes):               # bar chart exactly as tall as the squares
            bp, sq = ax.get_position(), row_axes[0].get_position()
            ax.set_position([bp.x0, sq.y0, bp.width, sq.height])
        fig.legend(*legend, fontsize=10, ncol=len(legend[1]), frameon=False, loc="lower left",
                   bbox_to_anchor=(bar_axes[0].get_position().x0, bar_axes[0].get_position().y1 + 0.01))
        save(fig, f"fig3_camera_ready_{tag}"); save_data(f"fig3_camera_ready_{tag}", pd.DataFrame(rows))
        pd.DataFrame(rows).to_csv(data_file(f"fig3_camera_ready_{tag}_bars", "csv"), index=False)
        json.dump(examples, open(data_file("fig3_camera_ready_examples", "json"), "w"), indent=1)


# ----------------------------------------------------------------------------- Fig. 4
def seed_mean(set_name, method, col="mape_vs_reference", root=None):
    """Mean of ``col`` over the method's training seeds (seed 0 and <method>_seed1/2 when present), as in Fig. 3;
    excluded 2D OOD pairs left out."""
    dfs = [keep_rows(d, set_name, root) for d in [per_instance_ref(set_name, method, root)]
           + [per_instance_ref(set_name, f"{method}_seed{sd}", root) for sd in SEEDS] if d is not None]
    return (float(np.mean([d[col].mean() for d in dfs])), len(dfs)) if dfs else (None, 0)


def fig4_camera_ready():
    """Figure 4, every method in its paper color; every method is scored against
    the solution of the problem it solves (score_reference.py), as in Fig. 3. (A) ToR-OT's marginal violations on the
    test set (|x-y|^2, 1,000 instances): row and column L1 residuals. (B) improvement of
    ToR-OT's MAPE over Meta OT, the strongest baseline, on the one-dimensional OOD pairs (|x-y|^2) of the
    distributions not in fig4b_exclude (plot_style.yaml: 10 distributions, 100 pairs; App. E shows all 144). (C) the same on the 49 two-dimensional OOD pairs (7 distributions, 6 x 6 grid, ||x-y||^2; the 2D model,
    <root>_2d), where Meta OT is also the strongest baseline. (D) 2D mean MAPE of every method, test set (1,000 pairs)
    and OOD pairs, ||x-y||^2; learned baselines: mean over their three training seeds."""
    s = "id_simplex_sqeuclidean"
    viol = per_instance(s, TOROT)
    fams_b = [(k, t) for k, t in PAPER144 if k not in STYLE.get("fig4b_exclude", [])]   # Fig. 4B rows / columns
    names = [n for n, _ in fams_b]
    def improvement(root, set_name, dists):
        z = np.load(f"{root}/sets/{set_name}.npz")
        pos = {(a, b): k for k, (a, b) in enumerate(zip(z["src"].tolist(), z["tgt"].tolist()))}
        o, v = mape_ref(set_name, "metaot", root), mape_ref(set_name, TOROT, root)
        if o is None or v is None: raise FileNotFoundError(f"{root}/scores/per_instance_ref/{set_name}")
        return (np.array([[100 * (o[pos[a, b]] - v[pos[a, b]]) / max(o[pos[a, b]], 1e-12) for b in dists] for a in dists]),
                np.array([[o[pos[a, b]] - v[pos[a, b]] for b in dists] for a in dists]))   # relative [%], absolute
    G1, D1 = improvement(R, "ood_paper_sqeuclidean", names)
    d2 = OOD2D; names2 = [n for n, _ in d2]                         # 2D: without the excluded distributions (as D)
    G2, D2 = improvement(R2D, "ood2d_paper_sqeuclidean", names2)
    bar_rows = []                                                     # (D) 2D mean MAPE per method and set
    for prefix, split in (("id_simplex2d", "Test set"), ("ood2d_paper", "OOD")):
        for m in FIG3_METHODS:
            mean, seeds = seed_mean(f"{prefix}_sqeuclidean", m, root=R2D)
            if mean is not None:
                bar_rows.append({"split": split, "method": m, "mape_vs_reference_mean": mean, "seeds": seeds,
                                 "instances": len(keep_rows(per_instance_ref(f"{prefix}_sqeuclidean", m, R2D),
                                                            f"{prefix}_sqeuclidean", R2D))})
    green = METHODS[TOROT][1]
    # Panels placed in inches, all of the same height h; the heatmaps B and C are h x h squares, so every panel fills its
    # space. Gaps hold the next panel's y-axis labels. No titles and no colorbars: the caption describes both.
    h, bottom, top = 3.6, 0.6, 0.1
    widths = [("gap", 0.75), ("A", 1.15), ("gap", 1.95), ("B", h), ("gap", 1.45), ("C", h), ("gap", 1.05), ("D", 2.0),
              ("gap", 0.1)]
    W, H = sum(w for _, w in widths), bottom + h + top
    fig = plt.figure(figsize=(W, H))
    ax_by, x = {}, 0.0
    for key, w in widths:
        if key != "gap":
            ax_by[key] = fig.add_axes([x / W, bottom / H, w / W, h / H])
        x += w
    axA, axB, axC, axD = (ax_by[k] for k in "ABCD")
    # (A) marginal violations
    cols = [("row_violation_l1", "Row"), ("col_violation_l1", "Column")]
    bp = axA.boxplot([np.clip(viol[c].to_numpy(), FLOOR, None) for c, _ in cols], widths=0.6, patch_artist=True,
                     showfliers=False, medianprops={"color": "white", "lw": 1.2}, whiskerprops={"color": green},
                     capprops={"color": green})
    for box in bp["boxes"]:
        box.set_facecolor(green); box.set_edgecolor(green)
    axA.set_xticks(range(1, len(cols) + 1)); axA.set_xticklabels([l for _, l in cols]); axA.set_yscale("log")
    lo, hi = axA.get_ylim()                                            # labels at 1, 2, 5 x 10^k only (the range is
    ticks = [m * 10.0 ** k for k in range(-6, 1) for m in (1, 2, 5) if lo <= m * 10.0 ** k <= hi]   # under a decade)
    axA.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    axA.yaxis.set_major_formatter(mticker.FixedFormatter(
        [rf"$10^{{{round(np.log10(t))}}}$" if round(t / 10.0 ** np.floor(np.log10(t))) == 1
         else rf"${round(t / 10.0 ** np.floor(np.log10(t)))}\times10^{{{int(np.floor(np.log10(t)))}}}$" for t in ticks]))
    axA.yaxis.set_minor_formatter(mticker.NullFormatter())
    axA.set_ylabel("violation magnitude (L1)")
    # (B) 1D, 144 OOD pairs; (C) 2D OOD pairs; both vs Meta OT. Rows: the distribution (with its parameters in B);
    # columns: its number, listed in the caption
    params = STYLE.get("ood_144_params", {})
    draw_grid(axB, G1, [params.get(k, t) for k, t in fams_b], "", "imp", fig, vs="metaot", fontsize=7,
              colorbar=False)                   # one metric everywhere: % improvement, diagonal included
    axB.set_xticklabels([str(i + 1) for i in range(len(names))], rotation=0)
    draw_grid(axC, G2, [t for _, t in d2], "", "imp", fig, vs="metaot", fontsize=9.5, colorbar=False)
    axC.set_xticklabels([f"{i + 1}'" for i in range(len(names2))], rotation=0)   # 1', 2', ...: told apart from B in the caption
    for ax in (axB, axC):
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=9); ax.tick_params(axis="x", labelsize=12)
        for lab in ax.get_xticklabels():                               # column numbers in bold
            lab.set_fontweight("bold")
    # (D) 2D mean MAPE, one bar per method and set
    splits = list(dict.fromkeys(r["split"] for r in bar_rows)); methods = [m for m in FIG3_METHODS if any(r["method"] == m for r in bar_rows)]
    # sorted from the highest to the lowest MAPE (mean over the two sets), in the same order in both groups and the legend
    methods.sort(key=lambda m: -np.mean([r["mape_vs_reference_mean"] for r in bar_rows if r["method"] == m]))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        xs = [splits.index(r["split"]) + (j - (len(methods) - 1) / 2) * w for r in bar_rows if r["method"] == m]
        hs = [r["mape_vs_reference_mean"] for r in bar_rows if r["method"] == m]
        axD.bar(xs, hs, width=w, color=METHODS[m][1], label=appE_label(m), edgecolor="white", linewidth=0.6)
    axD.set_xticks(range(len(splits))); axD.set_xticklabels(splits); axD.set_ylabel("MAPE")
    top_bar = max(r["mape_vs_reference_mean"] for r in bar_rows)
    if top_bar > 0.85: print(f"  fig4 (D): a bar reaches {top_bar:.2f}; check that it clears the legend (y axis 0..1)")
    axD.set_ylim(0, 1)                                                 # the values are in the data file and the text
    axD.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2, columnspacing=0.5, handlelength=1.0)
    for ax in (axA, axD):                                              # closed black box, as the heatmaps (and Fig. 3)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("black")
    panel_letters(fig, {"A": axA, "B": axB, "C": axC, "D": axD}, beside=True)
    save(fig, "fig4_camera_ready")
    save_data("fig4_camera_ready_D", pd.DataFrame(bar_rows))
    save_data("fig4_camera_ready_B", pd.DataFrame([{"source": a, "target": b, "improvement_over_metaot_%": G1[i, j],
                                                    "mape_metaot_minus_torot": D1[i, j]}
                                                   for i, a in enumerate(names) for j, b in enumerate(names)]))
    save_data("fig4_camera_ready_C", pd.DataFrame([{"source": a, "target": b, "improvement_over_metaot_%": G2[i, j],
                                                    "mape_metaot_minus_torot": D2[i, j]}
                                                   for i, a in enumerate(names2) for j, b in enumerate(names2)]))


# ----------------------------------------------------------------------------- Fig. 5
FIG5_PAIR = ("SwissRoll", "StandardNormal")                    # the pair shown in Fig. 5


def fig5_camera_ready():
    """Figure 5: one pair of the 2D OOD set (Swiss roll -> Standard normal; ||x-y||^2, 6 x 6 grid). Source and target histograms, the cost matrix, and the plans of POT (exact),
    ToR-OT, NeuralOT and Meta OT on one color scale. The data file has each method's MAPE on the pair (against the
    solution of the problem it solves, score_reference.py), its total mass and its rank among the drawn OOD pairs."""
    s = "ood2d_paper_sqeuclidean"
    z = np.load(f"{R2D}/sets/{s}.npz")
    k = [i for i, (a, b) in enumerate(zip(z["src"], z["tgt"])) if (a, b) == FIG5_PAIR][0]
    xy = z["coords"]; xs, ys = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    ix, iy = np.searchsorted(xs, xy[:, 0]), np.searchsorted(ys, xy[:, 1])
    def grid(vec):
        H = np.zeros((len(ys), len(xs))); H[iy, ix] = vec; return H
    eps_m = (timing(s, "metaot", R2D) or {}).get("eps")
    plans = [("exact", "$\\pi$ POT\n(exact)", z["plan"][k], CMAP["plans"]["exact"], None)]
    for m, title in ((TOROT, "$\\hat\\pi$ ToR-OT\n(ours)"), ("neuralot", "$\\hat\\pi$ NeuralOT\n(exact OT)"),
                     ("metaot", "$\\hat\\pi$ Meta OT\n" + (f"(entropic, $\\varepsilon$ = {eps_m:g})" if eps_m else "(entropic)"))):
        path = f"{R2D}/preds/{s}__{m}.npy"
        if os.path.exists(path):
            plans.append((m, title, np.load(path)[k], CMAP["plans"][{TOROT: "torot"}.get(m, m)],
                          float(per_instance_ref(s, m, R2D)["mape_vs_reference"].iloc[k])))
    names = dict(STYLE["ood_2d"])
    n = len(z["a"][k]); ticks = [1, 12, 24, n]
    # columns: source | target (+ colorbar) | gap | cost | plans ... (+ colorbar); one scale for the histograms, one for
    # the plans; the colorbars are as tall as the square panels
    fig = plt.figure(figsize=(2.0 * (3 + len(plans)) + 2.4, 2.75))
    gs = fig.add_gridspec(1, 5 + len(plans), width_ratios=[1, 1, 0.75, 1] + [1] * len(plans) + [0.35], wspace=0.14)
    axS, axT, axC = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])
    axP = [fig.add_subplot(gs[4 + i]) for i in range(len(plans))]
    caxH, caxP = axT.inset_axes([1.06, 0.0, 0.07, 1.0]), axP[-1].inset_axes([1.06, 0.0, 0.07, 1.0])
    hmax = max(z["a"][k].max(), z["b"][k].max())
    for ax, vec, title in ((axS, z["a"][k], "Source $\\mu$\n" + names[FIG5_PAIR[0]]),
                           (axT, z["b"][k], "Target $\\nu$\n" + names[FIG5_PAIR[1]])):
        imH = ax.imshow(grid(vec), origin="lower", cmap=CMAP["plans"]["exact"], vmin=0, vmax=hmax, interpolation="nearest",
                        extent=(0.5, len(xs) + 0.5, 0.5, len(ys) + 0.5))
        ax.set_title(title, fontsize=10.5); ax.set_xlabel("bin ($x$)", fontsize=10)
        ax.set_xticks(range(1, len(xs) + 1)); ax.set_yticks(range(1, len(ys) + 1)); ax.tick_params(labelsize=8.5)
    axS.set_ylabel("bin ($y$)", fontsize=10); axT.set_yticklabels([])
    fig.colorbar(imH, cax=caxH).ax.tick_params(labelsize=8)
    from torot.data.cost import cost_matrix
    C = cost_matrix(xy, xy, "sqeuclidean")
    axC.imshow(C, origin="lower", cmap=CMAP["plans"]["exact"], interpolation="nearest", extent=(0.5, n + 0.5, 0.5, n + 0.5))
    axC.set_title("Cost $C$\n($L_2^2$, max 1)", fontsize=10.5)
    vmax = max(P.max() for _, _, P, _, _ in plans)                   # one color scale for every plan
    rows = []
    for ax, (m, title, P, cmap, mape) in zip(axP, plans):
        imP = ax.imshow(P, origin="lower", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest",
                        extent=(0.5, n + 0.5, 0.5, n + 0.5))
        ax.set_title(title, fontsize=10.5)
        if mape is not None:
            v = keep_rows(per_instance_ref(s, m, R2D), s, R2D)["mape_vs_reference"].to_numpy()
            rows.append({"method": m, "pair": " -> ".join(FIG5_PAIR), "mape_vs_reference": mape, "total_mass": float(P.sum()),
                         "rank_among_drawn_ood_pairs": int((v < mape).sum()) + 1, "drawn_ood_pairs": len(v),
                         "median_mape_drawn_ood_pairs": float(np.median(v))})
    cb = fig.colorbar(imP, cax=caxP); cb.ax.tick_params(labelsize=8)
    cb.set_ticks(np.linspace(0, vmax, 4)); cb.set_ticklabels([f"{t:.2f}" for t in np.linspace(0, vmax, 4)])
    for ax in [axC] + axP:
        ax.set_xlabel("target bin", fontsize=10); ax.set_xticks(ticks); ax.set_yticks(ticks); ax.tick_params(labelsize=8.5)
        if ax is not axC: ax.set_yticklabels([])
    axC.set_ylabel("source bin", fontsize=10)
    save(fig, "fig5_camera_ready"); save_data("fig5_camera_ready", pd.DataFrame(rows))

for fn in (fig3_camera_ready, fig4_camera_ready, fig5_camera_ready, table1_improvement, table_appD_all_sets, fig_appG_testbed, fig_warmstart, table_ablations, table_appC_sinkhorn, fig_appE_cost_summary, fig_appE_pairs, fig_appE_allcosts, fig_appE_improvement,
           fig_appE_distributions, fig_appF_distributions, fig_appF_cost_summary, fig_appF_pairs, fig_appF_allcosts, fig_appF_improvement, fig_appD_interpretability, fig_scaling_n100, fig6_and_table2_runtime,
           table4_symmetric, table4_symmetric_5M, fig_size_generalization, fig_latency_vs_n,
           fig_appG_training_curve, table_appG_support_size):
    if args.only is None or fn.__name__ in args.only:
        step(fn)
print(f"\ngenerated {len(done)}:"); print("\n".join(f"  {d}" for d in done))
if skipped:
    print(f"\nskipped {len(skipped)}:"); print("\n".join(f"  {s}" for s in skipped))
