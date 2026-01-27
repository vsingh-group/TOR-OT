import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns
import pandas as pd

def plot_testset_analysis(data, output_dir):
    """Generates plots for the testset results."""
    print("Generating Testset Plots...")
    
    # Prepare DataFrame
    metrics = {
        'Row Violation': data['all_row_violation'],
        'Column Violation': data['all_col_violation'],
        'Max Violation': data['all_max_violation']
    }
    df = pd.DataFrame(metrics)
    df_melted = df.melt(var_name='Metric', value_name='Violation')

    # 1. Box Plot with Stripplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_melted, x='Metric', y='Violation', showfliers=False, palette='pastel')
    sns.stripplot(data=df_melted, x='Metric', y='Violation', color='black', alpha=0.1, jitter=True, size=2)
    plt.title('Distribution of Feasibility Violations (Testset)', fontsize=14)
    plt.ylabel('Violation Magnitude (L1 Error)')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'testset_feasibility_boxplot.png'), dpi=300)
    plt.close()

    # 2. Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_melted, x='Metric', y='Violation', palette='muted', inner='quartile')
    plt.title('Density of Feasibility Violations (Testset)', fontsize=14)
    plt.ylabel('Violation Magnitude')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'testset_feasibility_violinplot.png'), dpi=300)
    plt.close()

    # 3. ECDF Plot
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=df_melted, x='Violation', hue='Metric', linewidth=2)
    plt.title('Cumulative Distribution of Violations (Testset)', fontsize=14)
    plt.xlabel('Violation Magnitude')
    plt.ylabel('Proportion of Samples')
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'testset_feasibility_ecdf.png'), dpi=300)
    plt.close()


def plot_pairs_analysis(data_list, output_dir):
    """Generates plots for the all-pairs results."""
    print("Generating All-Pairs Plots...")
    
    df = pd.DataFrame(data_list)
    
    # Pivot for Heatmaps
    pivot_max = df.pivot(index='source', columns='target', values='max_violation')
    pivot_row = df.pivot(index='source', columns='target', values='row_violation')
    pivot_col = df.pivot(index='source', columns='target', values='col_violation')

    # 1. Max Violation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_max, annot=True, fmt=".2e", cmap="Reds", linewidths=.5, cbar_kws={'label': 'Max Violation'})
    plt.title('Max Constraint Violation by Distribution Pair', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairs_feasibility_heatmap_max.png'), dpi=300)
    plt.close()

    # 2. Aggregated Bar Chart (Average Max Violation by Source)
    avg_by_source = df.groupby('source')['max_violation'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_by_source.values, y=avg_by_source.index, palette='Reds_d')
    plt.title('Average Max Violation by Source Distribution', fontsize=14)
    plt.xlabel('Average Max Violation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairs_feasibility_barplot_source.png'), dpi=300)
    plt.close()
    
    # 3. Aggregated Bar Chart (Average Max Violation by Target)
    avg_by_target = df.groupby('target')['max_violation'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_by_target.values, y=avg_by_target.index, palette='Blues_d')
    plt.title('Average Max Violation by Target Distribution', fontsize=14)
    plt.xlabel('Average Max Violation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairs_feasibility_barplot_target.png'), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Comprehensive Feasibility Plots")
    parser.add_argument("exp_dir", type=str, help="Experiment results directory")
    args = parser.parse_args()

    # File paths
    testset_path = os.path.join(args.exp_dir, "eval_all_testset_1d_euclidean_results_tot.npz")
    pairs_path = os.path.join(args.exp_dir, "eval_all_pairs_1d_euclidean_results_tot.npy")

    # Process Testset
    if os.path.exists(testset_path):
        testset_data = np.load(testset_path)
        plot_testset_analysis(testset_data, args.exp_dir)
    else:
        print(f"Warning: Testset file not found: {testset_path}")

    # Process Pairs
    if os.path.exists(pairs_path):
        pairs_data = np.load(pairs_path, allow_pickle=True)
        # Handle if it's a 0-d array wrapping the list
        if pairs_data.shape == (): 
            pairs_data = pairs_data.item()
        
        # Ensure it's a list (npy might load as array of dicts)
        if isinstance(pairs_data, np.ndarray):
            pairs_data = pairs_data.tolist()
            
        plot_pairs_analysis(pairs_data, args.exp_dir)
    else:
        print(f"Warning: Pairs file not found: {pairs_path}")

    print("Done.")

if __name__ == "__main__":
    main()
