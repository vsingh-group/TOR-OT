import nbformat
import os

notebook_path = "TF4OT/evals/plots/comparison_box_plots_2d.ipynb"

def modify_notebook(nb_path):
    abs_path = os.path.abspath(nb_path)
    if not os.path.exists(abs_path):
        print(f"Skipping {nb_path} (not found)")
        return

    print(f"Processing {nb_path}...")
    with open(abs_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Update Keys definition cell
    keys_cell_index = -1
    for idx, cell in enumerate(nb.cells):
        if "TESTSET_KEYS = ['NOT_INDIV', 'TOT_INDIV']" in cell.source:
            keys_cell_index = idx
            # Add OOD keys definition
            if "OOD_KEYS" not in cell.source:
                cell.source += "\n# OOD Keys\nOOD_KEYS = ['NOT_OOD', 'TOT_OOD']"
                print("  Added OOD_KEYS definition.")
            break
            
    # 2. Update Data Loading Cell to include OOD
    loading_cell_index = -1
    for idx, cell in enumerate(nb.cells):
        if "all_testset_data[metric] = {}" in cell.source and "load_testset_npz" in cell.source:
            loading_cell_index = idx
            
            # We want to add OOD loading logic after the INDIV/JOINT loading logic is done for a metric
            # Or insert it alongside.
            
            if "# OOD Loading" not in cell.source:
                # Find the end of the loop body for the metric loop
                # This is a bit complex with regex replacement, so let's append the OOD loading block
                # inside the loop.
                
                # We'll look for the end of the specific metric block inside the loop
                # print("NOT_INDIV: ", val) appears near the end of the load block for a metric
                
                new_loading_code = """
    # OOD Loading
    # Assume OOD files follow: eval_all_ood_2d_{metric}_results_{model}.npy
    path_tot_ood = os.path.join(dir_joint, f"eval_all_ood_2d_{metric}_results_tot.npy")
    path_not_ood = os.path.join(dir_indiv, f"eval_all_ood_2d_{metric}_results_not.npy")
    
    val = load_testset_npz(path_tot_ood)
    if val: all_testset_data[metric]['TOT_OOD'] = val
    else: print(f"  No OOD TOT data found for {metric} at {path_tot_ood}")

    val = load_testset_npz(path_not_ood)
    if val: all_testset_data[metric]['NOT_OOD'] = val
    else: print(f"  No OOD NOT data found for {metric} at {path_not_ood}")
"""
                # Insert before the outer loop continues or finishes
                # We can replace 'print("NOT_INDIV: ", val)' with 'print("NOT_INDIV: ", val)\n' + new_loading_code
                
                if 'print("NOT_INDIV: ", val)' in cell.source:
                    cell.source = cell.source.replace(
                        'print("NOT_INDIV: ", val)', 
                        'print("NOT_INDIV: ", val)\n' + new_loading_code
                    )
                    print("  Injected OOD data loading logic.")
            break

    # 3. Add New Plotting Cell for OOD Comparison
    # We'll add this after the existing Testset Boxplot cell
    
    boxplot_cell_index = -1
    for idx, cell in enumerate(nb.cells):
        if "Generating Test Set Plot for" in cell.source and "ax.boxplot" in cell.source:
            boxplot_cell_index = idx
            
    if boxplot_cell_index != -1:
        # Create a new cell with similar logic but for OOD comparison
        # We want 4 bars per group: NOT_INDIV, TOT_INDIV, NOT_OOD, TOT_OOD
        
        new_cell_source = """
# OOD vs Test Set Comparison Boxplots
# Comparing: Testset (NOT, TOT) vs OOD (NOT, TOT)

PLOT_KEYS = ['NOT_INDIV', 'TOT_INDIV', 'NOT_OOD', 'TOT_OOD']
PLOT_COLORS = ['#FF9999', '#99FF99', '#FFCC99', '#99CCFF'] # Red, Green, Orange, Blue-ish
PLOT_LABELS = ['NOT (Test)', 'TOT (Test)', 'NOT (OOD)', 'TOT (OOD)']

for perf_metric in PERFORMANCE_METRICS:
    print(f"Generating OOD Comparison Plot for {perf_metric}...")
    
    plot_data = []
    valid_dist_metrics = [] 
    
    for dist_metric in METRICS:
        if dist_metric not in all_testset_data: continue
        if not all_testset_data[dist_metric]: continue
            
        # Check if we have OOD data specifically to make the plot meaningful?
        # Or just plot what we have (NaNs for missing)
        
        valid_dist_metrics.append(dist_metric)
        
        for key in PLOT_KEYS:
            model_data = all_testset_data[dist_metric].get(key, {})
            values = model_data.get(perf_metric, [])
            
            if isinstance(values, np.ndarray):
                values = values[~np.isnan(values)] if len(values) > 0 else []
            elif isinstance(values, list):
                 values = [v for v in values if v is not None]
                 
            plot_data.append(values)

    if not plot_data or all(len(x)==0 for x in plot_data):
        print(f"No valid data found for {perf_metric} (OOD Comparison)")
        continue

    # Setup Plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    num_groups = len(valid_dist_metrics)
    num_bars = len(PLOT_KEYS)
    group_width = 0.8
    bar_width = group_width / num_bars

    positions = []
    for i in range(num_groups):
        center = i + 1
        start_offset = -(num_bars - 1) * bar_width / 2.0
        for j in range(num_bars):
            offset = start_offset + (j * bar_width)
            positions.append(center + offset)

    # Boxplot
    bp = ax.boxplot(plot_data, positions=positions, widths=bar_width*0.8, 
                    patch_artist=True, medianprops=dict(color="black"), showfliers=False)

    # Colors
    for i, patch in enumerate(bp['boxes']):
        color_idx = i % num_bars
        patch.set_facecolor(PLOT_COLORS[color_idx])

    # X Actions
    ax.set_xticks(range(1, num_groups + 1))
    xtick_labels = [METRIC_LABELS.get(m, m) for m in valid_dist_metrics]
    ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=18)

    ax.set_ylabel(perf_metric, fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PLOT_COLORS[i], label=PLOT_LABELS[i])
        for i in range(len(PLOT_KEYS))
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout()
    
    if perf_metric in ['MAPE']:
        safe_metric = re.sub(r'[^a-z0-9]+', '_', perf_metric.lower()).strip('_')
        out_png = f"img/2d_ood_{safe_metric}.png"
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, bbox_inches='tight', dpi=300)
        print(f"Saved: {out_png}")
        
    plt.show()
"""
        new_cell = nbformat.v4.new_code_cell(new_cell_source)
        # Insert after the existing plot cell
        nb.cells.insert(boxplot_cell_index + 1, new_cell)
        print("  Added OOD comparison plot cell.")

    with open(abs_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Saved changes.")

if __name__ == "__main__":
    modify_notebook(notebook_path)
