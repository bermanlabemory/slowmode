"""Fig 3 (worms) — C. elegans run/pirouette recovery.

Layout (2 rows):
  Row 1:  A (eigenworm PSD)              B (eigenvalue spectrum)
  Row 2:  C (Basin 1, Pirouette map)  D (Basin 2, Run map)  E (omega-|theta| plane)

Companion to Kaur, Jain, & Berman (2026).
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch


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

# ==========================================================
# Layout: 2 rows; row 1 holds A,B (wide); row 2 holds C,D,E.
fig = plt.figure(figsize=(13.0, 7.6))
gs = GridSpec(2, 6, figure=fig,
              hspace=0.45, wspace=0.95,
              left=0.06, right=0.97, top=0.94, bottom=0.08)

# ---------- A: PSD ----------
axA = fig.add_subplot(gs[0, 0:3])
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
axA.text(-0.12, 1.06, 'A', transform=axA.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='top')

# ---------- B: eigenvalue spectrum ----------
axB = fig.add_subplot(gs[0, 3:6])
k_axis = np.arange(2, 2 + len(eigvals))
axB.semilogy(k_axis, eigvals, 'o-', color='0.15', ms=5, lw=1.3)
M_sel = 2
axB.axvline(M_sel, color=ARM_PALETTE[0], lw=1.0, ls='--')
axB.set_xlabel(r'eigenvalue index $k$', fontsize=11, fontweight='bold')
axB.set_ylabel(r'$|\lambda_k|$', fontsize=11, fontweight='bold')
axB.set_xticks(k_axis); axB.tick_params(labelsize=8)
axB.text(-0.12, 1.06, 'B', transform=axB.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='top')

# ---------- C, D: basin enrichment on canonical UMAP ----------
vmax = float(np.nanpercentile(np.abs(dev), 99))
basin_titles  = ['Basin 1 (Pirouette)', 'Basin 2 (Run)']
basin_letters = ['C', 'D']
basin_chi_lab = [r'$\chi_1$ enrichment', r'$\chi_2$ enrichment']
basin_slices  = [gs[1, 0:2], gs[1, 2:4]]
for j in range(2):
    ax = fig.add_subplot(basin_slices[j])
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

# ---------- E: (omega, |theta|) plane ----------
axE = fig.add_subplot(gs[1, 4:6])
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

save(plt.gcf(), 'figure_3')
