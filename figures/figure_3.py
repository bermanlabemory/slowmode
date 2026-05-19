"""Fig 3 (worms) v7 -- panel F split into two side-by-side sub-panels.

Editor note (PRX Life pre-review): the v6 panel F overlays the two-basin
Costa scatter (24 points, two trend lines, statistics in legend) and reads
visually cluttered. v7 splits panel F into Run (left) and Pirouette (right)
sub-panels via nested GridSpec, with matched axis ranges, 95% bootstrap CI
bands on the trend lines, and the correlation/p-value statements moved to
the sub-panel titles.

Layout (2x3 grid, panel F holding a 1x2 nested grid):
  Row 1: A (PSD)  B (eigenvalues)  C (basin 1 UMAP)
  Row 2: D (basin 2 UMAP)  E (Costa plane)  F[Run | Pirouette]
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.signal import welch
from scipy.stats import pearsonr


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = WORMS_DATA  # data location
fr      = 16
TAU_S   = 3.0

# ---------- Caches ----------
gp = np.load(os.path.join(OUT, 'gpcca_worms_M2_tau3s.npz'))
chi      = gp['chi']
pi_w     = gp['pi']
eigvals  = gp['eigvals']

um = np.load(os.path.join(OUT, 'worms_umap_canonical_full.npz'))
gx       = um['x_edges']
gy       = um['y_edges']
dev      = um['dev']

cv = np.load(os.path.join(OUT, 'worms_costa_per_cluster.npz'))
om_mean  = cv['om_mean']; gp_mean = cv['gp_mean']
hard_M2  = cv['hard']

ln = np.load(os.path.join(OUT, 'lognormal_reanalysis_worms.npz'),
             allow_pickle=True)
sigma_slow_w = ln['sigma_slow']      # (12, 2)
std_logtau_w = ln['std_logtau']      # (12, 2)

# ==========================================================
# Layout: 2x3 figure-level grid; panel F (gs[1,2]) split 1x2.
# Widen figure slightly to give the F sub-panels room.
fig = plt.figure(figsize=(13.0, 7.6))
gs = GridSpec(2, 3, figure=fig,
              hspace=0.45, wspace=0.40,
              left=0.05, right=0.98, top=0.94, bottom=0.08)

# ---------- A: PSD ----------
axA = fig.add_subplot(gs[0, 0])
with open(os.path.join(WORMS_DATA, 'all_valid_segments_worms.pkl'), 'rb') as fEW:
    segs_raw = pickle.load(fEW)
ews = np.ma.compress_rows(segs_raw)
ews = np.asarray(ews, dtype=np.float64)
psd_freqs, psd = welch(ews.T, fs=fr, nperseg=2048, axis=-1)
for i in range(5):
    axA.loglog(psd_freqs[1:], psd[i, 1:], color=ARM_PALETTE[i % 4], lw=1.0,
               alpha=0.85, label=fr'$a_{i+1}$')
axA.axvline(0.1, color='red', lw=0.8, ls='--')
axA.axvline(8.0, color='red', lw=0.8, ls='--')
axA.set_xlabel('frequency (Hz)', fontsize=11, fontweight='bold')
axA.set_ylabel('PSD of eigenworm coefs', fontsize=11, fontweight='bold')
axA.tick_params(labelsize=8)
legA = axA.legend(prop={'size': 9, 'weight': 'bold'}, ncol=1,
                  loc='lower left', frameon=False,
                  handlelength=1.0, labelspacing=0.3, handletextpad=0.3)
axA.text(-0.18, 1.06, 'A', transform=axA.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='top')

# ---------- B: eigenvalue spectrum ----------
axB = fig.add_subplot(gs[0, 1])
k_axis = np.arange(2, 2 + len(eigvals))
axB.semilogy(k_axis, eigvals, 'o-', color='0.15', ms=5, lw=1.3)
M_sel = 2
axB.axvline(M_sel, color=ARM_PALETTE[0], lw=1.0, ls='--')
axB.set_xlabel(r'eigenvalue index $k$', fontsize=11, fontweight='bold')
axB.set_ylabel(r'$|\lambda_k|$', fontsize=11, fontweight='bold')
axB.set_xticks(k_axis); axB.tick_params(labelsize=8)
axB.text(-0.18, 1.06, 'B', transform=axB.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='top')

# ---------- C, D: basin enrichment on canonical UMAP ----------
vmax = float(np.nanpercentile(np.abs(dev), 99))
basin_titles  = ['Basin 1 (Pirouette)', 'Basin 2 (Run)']
basin_letters = ['C', 'D']
basin_chi_lab = [r'$\chi_1$ enrichment', r'$\chi_2$ enrichment']
basin_grid_pos = [(0, 2), (1, 0)]
for j in range(2):
    r, c = basin_grid_pos[j]
    ax = fig.add_subplot(gs[r, c])
    im = ax.pcolormesh(gx, gy, dev[j], cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='auto', rasterized=True)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(basin_chi_lab[j], fontsize=10, fontweight='bold')
    cb.ax.tick_params(labelsize=7)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(basin_titles[j], fontsize=11, pad=4,
                 color=ARM_PALETTE[j], fontweight='bold')
    ax.text(-0.05, 1.06, basin_letters[j], transform=ax.transAxes,
            fontsize=14, fontweight='bold', ha='left', va='top')

# ---------- E: Costa (omega, |theta|) plane ----------
axE = fig.add_subplot(gs[1, 1])
ok = np.isfinite(om_mean) & np.isfinite(gp_mean)
sizes = 18 + 280 * pi_w[ok] / pi_w[ok].max()
sc = axE.scatter(om_mean[ok], gp_mean[ok], c=chi[ok, 0],
                 cmap='RdBu_r', vmin=0, vmax=1,
                 s=sizes, edgecolor='0.25', linewidth=0.3, alpha=0.85)
axE.axvline(0, color='0.5', lw=0.5, ls='--')
axE.set_xlabel(r'cluster $\bar{\omega}$  (body wave/s)',
               fontsize=11, fontweight='bold')
axE.set_ylabel(r'cluster $\langle |\theta(s,t)|\rangle_{s,t}$  (rad)',
               fontsize=11, fontweight='bold')
axE.tick_params(labelsize=8)
axE.text(-0.18, 1.06, 'E', transform=axE.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='top')
basin_xy_off = [(-50, 8), (8, -18)]
for j, lab in enumerate(['pirouette', 'run']):
    sel_b = (hard_M2 == j) & ok
    cx = np.average(om_mean[sel_b], weights=pi_w[sel_b])
    cy = np.average(gp_mean[sel_b], weights=pi_w[sel_b])
    color = ARM_PALETTE[j]
    axE.scatter([cx], [cy], marker='X', s=140, c=color,
                edgecolor='black', linewidth=0.7, zorder=5)
    axE.annotate(f'basin {j+1}\n({lab})', xy=(cx, cy),
                 xytext=basin_xy_off[j], textcoords='offset points',
                 fontsize=8, fontweight='bold', color=color,
                 ha='left', va='bottom')
cb = fig.colorbar(sc, ax=axE, fraction=0.045, pad=0.02)
cb.set_label(r'$\chi_1$', fontsize=11, fontweight='bold')
cb.ax.tick_params(labelsize=7)

# ---------- F: per-worm Costa test split by basin ----------
# Nested 1x2 GridSpec within gs[1, 2].
gs_F = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 2],
                                wspace=0.42)

# Compute shared axis ranges across both basins (so the panels are
# directly comparable visually).
all_s = sigma_slow_w[np.isfinite(sigma_slow_w)]
all_y = std_logtau_w[np.isfinite(std_logtau_w)]
x_lo = np.nanmin(all_s) - 0.05 * (np.nanmax(all_s) - np.nanmin(all_s))
x_hi = np.nanmax(all_s) + 0.05 * (np.nanmax(all_s) - np.nanmin(all_s))
y_lo = np.nanmin(all_y) - 0.05 * (np.nanmax(all_y) - np.nanmin(all_y))
y_hi = np.nanmax(all_y) + 0.05 * (np.nanmax(all_y) - np.nanmin(all_y))

# Per-basin sub-panels (basin order: Run [j=1] left, Pirouette [j=0] right).
# Editor recommendation: lead with the positive effect (Run) on the left.
basin_order = [1, 0]
sub_titles  = ['Run Basin', 'Pirouette Basin']

rng = np.random.default_rng(0)
N_BOOT = 2000

for col, j in enumerate(basin_order):
    ax = fig.add_subplot(gs_F[0, col])
    s = sigma_slow_w[:, j]; y = std_logtau_w[:, j]
    ok_j = np.isfinite(s) & np.isfinite(y)
    sj = s[ok_j]; yj = y[ok_j]

    color = ARM_PALETTE[j]
    ax.scatter(sj, yj, c=color, s=45, alpha=0.85,
               edgecolors='0.2', linewidth=0.4, zorder=4)

    # Trend line + 95% bootstrap CI band (resample worms).
    if ok_j.sum() >= 3:
        m_j, b_j = np.polyfit(sj, yj, 1)
        xs = np.linspace(x_lo, x_hi, 80)
        ax.plot(xs, m_j * xs + b_j, color=color, lw=1.8, zorder=5)

        # Bootstrap regression line.
        boot_lines = np.zeros((N_BOOT, len(xs)))
        for b in range(N_BOOT):
            idx = rng.integers(0, len(sj), size=len(sj))
            m_b, b_b = np.polyfit(sj[idx], yj[idx], 1)
            boot_lines[b] = m_b * xs + b_b
        ci_lo = np.percentile(boot_lines, 2.5, axis=0)
        ci_hi = np.percentile(boot_lines, 97.5, axis=0)
        ax.fill_between(xs, ci_lo, ci_hi, color=color, alpha=0.18,
                        edgecolor='none', zorder=3)

        r_j, p_j = pearsonr(sj, yj)
        # p formatting: scientific for small p, decimal otherwise.
        if p_j < 0.001:
            p_str = f'p = {p_j:.1e}'
        elif p_j < 0.01:
            p_str = f'p = {p_j:.3f}'
        else:
            p_str = f'p = {p_j:.2f}'
        ax.set_title(f'{sub_titles[col]}\n' + r'$r = $' + f'{r_j:.2f}, {p_str}',
                     fontsize=11, pad=4, color=color, fontweight='bold')
    else:
        ax.set_title(sub_titles[col], fontsize=11, pad=4,
                     color=color, fontweight='bold')

    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$\sigma_{\mathrm{slow}}$', fontsize=11, fontweight='bold')
    if col == 0:
        ax.set_ylabel(r'$\sigma_{\log\tau}$', fontsize=11, fontweight='bold')
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=8)

    if col == 0:
        ax.text(-0.20, 1.10, 'F', transform=ax.transAxes, fontsize=14,
                fontweight='bold', ha='left', va='top')

save(plt.gcf(), 'figure_3')
