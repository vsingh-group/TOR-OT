import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Visualize Constraint Violations")
    parser.add_argument("exp_dir", type=str, help="Experiment results directory")
    args = parser.parse_args()

    results_file = os.path.join(args.exp_dir, "eval_all_testset_1d_euclidean_results_tot.npz")
    
    if not os.path.exists(results_file):
        print(f"Error: File not found: {results_file}")
        return

    print(f"Loading results from: {results_file}")
    data = np.load(results_file)
    
    metrics = {
        'Row Violation': data['all_row_violation'],
        'Column Violation': data['all_col_violation'],
        'Max Violation': data['all_max_violation']
    }

    # Set up styling
    sns.set_theme(style="whitegrid")
    
    # Create histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Constraint Violation Distributions (Euclidean)', fontsize=16)

    for i, (name, values) in enumerate(metrics.items()):
        ax = axes[i]
        sns.histplot(values, kde=True, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(name)
        ax.set_xlabel('Violation Magnitude')
        ax.set_ylabel('Count')
        
        # Add summary stats
        mean_val = np.mean(values)
        max_val = np.max(values)
        stats_text = f"Mean: {mean_val:.2e}\nMax: {max_val:.2e}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    output_path = os.path.join(args.exp_dir, "constraint_violations_euclidean.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to: {output_path}")

if __name__ == "__main__":
    main()
