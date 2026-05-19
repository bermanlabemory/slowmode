"""Figure 2B: Lorenz attractor projections coloured by the slow
eigenvector phi_2 at beta=0.5, working lag set by the per-beta
characteristic timescale.

Renders the (x,y), (y,z), and (x,z) projections side-by-side as binned
mean-phi_2 density maps with a diverging colormap (RdBu_r), matching the
rendering grammar of the worm UMAP basin panels and the fly behavior-map
panels.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = LORENZ_DATA  # data location
d = np.load(os.path.join(OUT, 'lorenz_panel2B_data.npz'))
x = d['x']; y = d['y']; z = d['z']
phi2_raw = d['phi2_frame_raw']
cl_x = d['cluster_x']; cl_y = d['cluster_y']; cl_z = d['cluster_z']
cl_phi2 = d['cluster_phi2']; cl_n = d['cluster_count']
lag = int(d['lag_frames']); N_clust = int(d['N']); d_emb = int(d['d'])
print(f'Loaded {len(x)} frames + {N_clust} clusters; '
      f'lag={lag} frames, N={N_clust}, d={d_emb}')

# Symmetric color limits, slightly compressed (95th percentile) for saturation
vmax = float(np.nanpercentile(np.abs(phi2_raw), 95))
cmap = 'RdBu_r'

# Sort frames by |phi_2| ascending so strongly-colored points land on top
order_2d = np.argsort(np.abs(phi2_raw))
x2 = x[order_2d]; y2 = y[order_2d]; z2 = z[order_2d]; p2 = phi2_raw[order_2d]


# ---------------------------------------------------------------
# 2D projections, density-style: per-(x,y) bin mean phi_2 with RdBu_r
# (matches the rendering style of fig:worms C,D and fig:flies_bio A:
# diverging colormap on a binned density map).  This is the panel used
# in the published Fig. 2B.
# ---------------------------------------------------------------
def mean_phi2_grid(A, B, phi, n_bins=200, sigma=3.5, min_smoothed_count=2.0):
    """Smoothed per-bin mean of phi over (A, B) grid (matches Fig 5A grammar).

    Uses NG=200 grid + gaussian_filter sigma=2.0 on both the phi-weighted
    histogram and the count histogram, then divides. Bins where the smoothed
    count is below min_smoothed_count are masked.

    Returns (xedges, yedges, mean_grid_or_nan)."""
    A = np.asarray(A); B = np.asarray(B); phi = np.asarray(phi)
    a_lo, a_hi = np.percentile(A, [0.2, 99.8])
    b_lo, b_hi = np.percentile(B, [0.2, 99.8])
    pad = 0.04
    a_lo -= pad * (a_hi - a_lo); a_hi += pad * (a_hi - a_lo)
    b_lo -= pad * (b_hi - b_lo); b_hi += pad * (b_hi - b_lo)
    aedges = np.linspace(a_lo, a_hi, n_bins + 1)
    bedges = np.linspace(b_lo, b_hi, n_bins + 1)
    counts, _, _ = np.histogram2d(A, B, bins=[aedges, bedges])
    weighted, _, _ = np.histogram2d(A, B, bins=[aedges, bedges],
                                     weights=phi)
    counts_s = gaussian_filter(counts, sigma=sigma)
    weighted_s = gaussian_filter(weighted, sigma=sigma)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = weighted_s / counts_s
    mean = np.where(counts_s >= min_smoothed_count, mean, np.nan)
    return aedges, bedges, mean.T   # transpose for pcolormesh (y, x)

projections = [
    (x, y, r'$x$', r'$y$'),
    (y, z, r'$y$', r'$z$'),
    (x, z, r'$x$', r'$z$'),
]

# First pass: compute all three to get a global vmax for consistent color scale
all_grids = []
for A, B, _, _ in projections:
    aE, bE, M = mean_phi2_grid(A, B, phi2_raw)
    all_grids.append((aE, bE, M))
v_density = float(np.nanpercentile(np.abs(np.concatenate(
    [G[2][np.isfinite(G[2])].ravel() for G in all_grids])), 99))
print(f'Density-map vmax: {v_density:.4f} '
      f'(per-frame phi2 vmax: {vmax:.4f})')

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4),
                         gridspec_kw=dict(wspace=0.25))
for ax, (A, B, xlab, ylab), (aE, bE, M) in zip(axes, projections, all_grids):
    pc = ax.pcolormesh(aE, bE, M, cmap=cmap, vmin=-v_density, vmax=v_density,
                       shading='auto', rasterized=True)
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Light frame around the data extent (matches worm/fly basin panels)
    ax.add_patch(plt.Rectangle((aE[0], bE[0]),
                               aE[-1] - aE[0], bE[-1] - bE[0],
                               fill=False, edgecolor='0.4', linewidth=0.6))

cax = fig.add_axes([0.935, 0.18, 0.012, 0.66])
cb = fig.colorbar(pc, cax=cax)
cb.set_label(r'mean $\phi_2$ per bin', fontsize=12)
cb.ax.tick_params(labelsize=9)
plt.subplots_adjust(left=0.05, right=0.92, top=0.94, bottom=0.13)
save(fig, 'figure_2B')
plt.close(fig)
