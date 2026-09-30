import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import ot

def make_1d_gauss(n, m, sigma):
    """Generate 1D Gaussian distributions."""
    x = np.arange(n, dtype=np.float64)
    a = np.exp(-(x - m)**2 / (2 * sigma**2))
    a /= a.sum()
    a /= a.sum()
    return a

# Configure matplotlib for nice LaTeX-style fonts
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False
})

def add_grid(ax, rows, cols):
    """Add grid lines to axes without showing standard spines/labels."""
    ax.axis('on')
    ax.tick_params(left=False, right=False, top=False, bottom=False,
                   labelleft=False, labeltop=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Grid lines at pixel boundaries
    ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle=':', linewidth=0.5)

def add_colorbar_arrow(cbar, color='black'):
    """Draw an upward arrow beside a colorbar, from just above the '0' tick
    to the top of the gradient, indicating increasing values."""
    ax = cbar.ax
    # Draw the colorbar (and its arrow) above the neighbouring matrix axes.
    ax.set_zorder(50)
    arrow = FancyArrowPatch(
        (1.85, 0.015), (1.85, 1.0),   # x in axes fraction: just right of the bar
        transform=ax.transAxes,
        arrowstyle='-|>', mutation_scale=55,
        linewidth=4, color=color,
        shrinkA=0, shrinkB=0,
        clip_on=False, zorder=60,
    )
    ax.add_patch(arrow)


def create_ot_plot():
    # 1. Generate data
    n = 30  # n bins
    
    # Source distribution: Gaussian mixture
    a = make_1d_gauss(n, m=7, sigma=2) + make_1d_gauss(n, m=20, sigma=2)
    a /= a.sum() # Normalize
    
    # Target distribution: Single Gaussian
    b = make_1d_gauss(n, m=n/2, sigma=5)
    
    # Cost matrix
    x = np.arange(n).reshape((n, 1))
    M = ot.dist(x, x).astype(np.float64)
    M /= M.max()

    # 2. Compute OT plan
    # Using EMD for sharp "matching" look
    pi = ot.emd(a, b, M)

    # 3. Plotting
    fig = plt.figure(figsize=(20, 10))
    
    # Grid Layout
    # Rows: [Top Stems, Top Hist, Main Matrix]
    # Cols: [Left Stems, Left Hist, Main Matrix, CbarPi, Spacer, CostMatrix, CbarCost]
    
    # Ratios
    # Height: Top Stems (smallish) : Top Hist (very thin) : Main (large)
    hr = [1.5, 0.19, 5]
    # Width: Left Stems, Left Hist, Main, CbarPi, Spacer, Cost, CbarCost
    wr = [1.5, 0.19, 5, 0.2, 1, 5, 0.2]
    gs = gridspec.GridSpec(3, 7, width_ratios=wr, height_ratios=hr, wspace=0.01, hspace=0.01)

    # --- Top Plot (Target beta) ---
    # Position: Row 0, Col 2
    ax_top = plt.subplot(gs[0, 2])
    # Stem plot
    markerline, stemlines, baseline = ax_top.stem(np.arange(n), b, markerfmt='o', basefmt=" ")
    plt.setp(stemlines, 'color', '#F39C12', linewidth=4)
    plt.setp(markerline, 'color', '#F39C12', 'markersize', 7)
    ax_top.set_xlim(-0.5, n-0.5)
    ax_top.axis('off')
    ax_top.text(n//2 + 10, b.max()*0.5, r'$\beta$', fontsize=75, color='#E67E22')

    # --- Top 1D Heatmap ---
    # Position: Row 1, Col 2
    ax_top_hist = plt.subplot(gs[1, 2])
    # b is shape (n,). Reshape to (1, n) for imshow
    ax_top_hist.imshow(b.reshape(1, -1), cmap='Greens', aspect='auto', interpolation='nearest')
    add_grid(ax_top_hist, 1, n)

    # --- Left Plot (Source alpha) ---
    # Position: Row 2, Col 0
    ax_left = plt.subplot(gs[2, 0])
    # Horizontal lines
    y_pos = np.arange(n)
    ax_left.hlines(y_pos, 0, a, colors='blue', lw=4)
    ax_left.plot(a, y_pos, 'b.', markersize=15)
    ax_left.set_ylim(n-0.5, -0.5) # Invert y to match matrix, 0 at top
    #ax_left.set_xlim(a.max()*1.1, 0) # 0 on the right
    xmax = a.max()
    ax_left.set_xlim(xmax * 1.1, -0.1 * xmax)   # small extra space past 0
    ax_left.axis('off')
    ax_left.text(a.max()*0.5, n//2, r'$\alpha$', fontsize=75, color='darkblue', ha='right')


    # Add a small gap between ax_left and ax_left_hist by shrinking ax_left
    pos = ax_left.get_position()
    ax_left.set_position([pos.x0, pos.y0, pos.width * 0.92, pos.height])  # 8% narrower

    # --- Left 1D Heatmap ---
    # Position: Row 2, Col 1
    ax_left_hist = plt.subplot(gs[2, 1])
    # a is shape (n,). Reshape to (n, 1) for imshow
    ax_left_hist.imshow(a.reshape(-1, 1), cmap='Greens', aspect='auto', interpolation='nearest')
    add_grid(ax_left_hist, n, 1)
    
    # --- Main Matrix (Transport Plan pi) ---
    # Position: Row 2, Col 2
    ax_main = plt.subplot(gs[2, 2])
    im = ax_main.imshow(pi, cmap='Greens', aspect='equal', interpolation='nearest')
    add_grid(ax_main, n, n)
    ax_main.text(n//2 + 4.5, n//2 - 5, r'$\pi$', fontsize=150, color='darkgreen')
    ax_main.text(n//2 + 11, n//2 - 7, '*', fontsize=80, color='darkgreen', weight='bold')

    # --- Colorbar for Pi ---
    # Position: Row 2, Col 3
    ax_cb = plt.subplot(gs[2, 3])
    cbar = plt.colorbar(im, cax=ax_cb)
    cbar.set_ticks([0])
    cbar.ax.set_yticklabels(['0'], fontsize=20, weight='bold')
    # Arrow
    add_colorbar_arrow(cbar)


    # --- Cost Matrix C ---
    # Position: Row 2, Col 5
    ax_cost = plt.subplot(gs[2, 5])
    im_c = ax_cost.imshow(M, cmap='Reds', aspect='equal', interpolation='nearest')
    add_grid(ax_cost, n, n)
    ax_cost.text(10, 10, r'$C$', fontsize=150, color='darkred', ha='left', va='top')
    
    # --- Colorbar for Cost ---
    # Position: Row 2, Col 6
    ax_cb_cost = plt.subplot(gs[2, 6])
    cbar_c = plt.colorbar(im_c, cax=ax_cb_cost)
    cbar_c.set_ticks([0])
    cbar_c.ax.set_yticklabels(['0'], fontsize=20, weight='bold')
    # Arrow
    add_colorbar_arrow(cbar_c)


    plt.tight_layout() # Might complain about gridspec but usually helps
    
    # Manually adjust subplots if tight_layout changes spacing too much?
    # Actually wspace/hspace in GridSpec handles it.
    
    output_path = 'ot_plot_generated.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.35, transparent=True)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    create_ot_plot()
