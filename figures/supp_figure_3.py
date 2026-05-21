"""Supp. Fig. S3 (worms): biological correspondence and individual variability.

Five panels:
  Row 1:  A. UMAP coloured by phase velocity omega
          B. UMAP coloured by curvature magnitude |theta|
  Row 2:  C. Arms-and-hub G-PCCA M=2 scatter (phi_2, phi_3)
          D. M=3 G-PCCA basin centroids on the (omega, |theta|) plane
             (deeper hierarchy: pirouette stays intact; run splits in two)
  Row 3:  E. Leave-one-worm-out CV: held-out basin MI vs random 2-coloring null

Companion to Kaur, Jain, & Berman (2026).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = WORMS_DATA  # data location
fr   = 16

# ---- Caches ----
gp  = np.load(os.path.join(OUT, 'gpcca_worms_M2_tau3s.npz'))
um  = np.load(os.path.join(OUT, 'worms_umap_canonical_full.npz'))
cv  = np.load(os.path.join(OUT, 'worms_costa_per_cluster.npz'))
m34 = np.load(os.path.join(OUT, 'gpcca_worms_M3_M4_tau3s.npz'))
mt  = np.load(os.path.join(OUT, 'M_eq_1_vs_M_eq_2_test.npz'))

chi_M2  = gp['chi'];     pi_w   = gp['pi']
hub     = gp['hub'];     arm_centroids = gp['arm_centroids']

om_mean = cv['om_mean']; gp_mean = cv['gp_mean']
chi_M3  = m34['chi_M3']

eigs_full = np.load(os.path.join(WORMS_DATA, 'worm_eigs_tau3s.npz'))
pm        = eigs_full['phi_mt'][:, :3]

gx = um['x_edges']; gy = um['y_edges']
omega_field = um['omega_field']; theta_field = um['theta_field']

mi_obs_loo  = mt['heldout_mi_obs']
mi_null_loo = mt['heldout_mi_null']

# ==========================================================
fig = plt.figure(figsize=(11.0, 11.5))
gs = GridSpec(3, 6, figure=fig, hspace=0.45, wspace=0.85,
              left=0.07, right=0.96, top=0.95, bottom=0.06)

def _frame(ax):
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_anchor('W')
    for sp in ax.spines.values(): sp.set_visible(False)

# Common offset for panel letters (absolute, so letters align regardless
# of axes width).  -14 pt left of axes left edge, +6 pt above axes top.
def _letter_trans(ax):
    return ax.transAxes + ScaledTranslation(-14/72, 6/72,
                                            fig.dpi_scale_trans)

# ---- A: UMAP coloured by omega ----
axA = fig.add_subplot(gs[0, 0:3])
omax = float(np.nanpercentile(np.abs(omega_field), 98))
imA = axA.pcolormesh(gx, gy, omega_field, cmap='RdBu_r',
                     vmin=-omax, vmax=omax, shading='auto', rasterized=True)
cbA = fig.colorbar(imA, ax=axA, fraction=0.045, pad=0.02)
cbA.set_label(r'$\omega$ (body wave/s)', fontsize=10, fontweight='bold')
cbA.ax.tick_params(labelsize=6)
_frame(axA)
axA.text(0, 1, 'A', transform=_letter_trans(axA), fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- B: UMAP coloured by |theta| ----
axB = fig.add_subplot(gs[0, 3:6])
tlo = float(np.nanpercentile(theta_field, 2))
thi = float(np.nanpercentile(theta_field, 98))
imB = axB.pcolormesh(gx, gy, theta_field, cmap='viridis',
                     vmin=tlo, vmax=thi, shading='auto', rasterized=True)
cbB = fig.colorbar(imB, ax=axB, fraction=0.045, pad=0.02)
cbB.set_label(r'$\langle |\theta(s,t)|\rangle_{s}$ (rad)',
              fontsize=10, fontweight='bold')
cbB.ax.tick_params(labelsize=6)
_frame(axB)
axB.text(0, 1, 'B', transform=_letter_trans(axB), fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- C: arms-and-hub G-PCCA M=2 scatter ----
axC = fig.add_subplot(gs[1, 0:3])
assignments = chi_M2.argmax(1)
fuzzy = chi_M2.max(axis=1) < 0.5
axC.scatter(pm[fuzzy, 0], pm[fuzzy, 1], c='0.85', s=10, alpha=0.55,
            edgecolors='none', rasterized=True)
for j in range(2):
    m_ = (assignments == j) & ~fuzzy
    axC.scatter(pm[m_, 0], pm[m_, 1], c=ARM_PALETTE[j], s=22, alpha=0.85,
                edgecolors='none', rasterized=True, label=f'basin {j+1}')
hx, hy = hub[0], hub[1]
for j in range(2):
    cx, cy = arm_centroids[j, 0], arm_centroids[j, 1]
    L = 1.6
    axC.plot([hx, hx + (cx-hx)*L], [hy, hy + (cy-hy)*L],
             color=ARM_PALETTE[j], lw=4.0, zorder=10, solid_capstyle='round')
    axC.scatter([cx], [cy], s=100, marker='o', c=ARM_PALETTE[j], zorder=15,
                edgecolors='black', linewidths=0.6)
axC.scatter([hx], [hy], s=160, marker='*', c='red', zorder=20,
            edgecolors='black', linewidths=0.6)
axC.set_xlabel(r'$\phi_2$ (slow eigenvector)',
               fontsize=10, fontweight='bold')
axC.set_ylabel(r'$\phi_3$ (slow eigenvector)',
               fontsize=10, fontweight='bold')
axC.set_aspect('equal', adjustable='datalim')
axC.text(0, 1, 'C', transform=_letter_trans(axC), fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- D: M=3 hierarchy on (omega, |theta|) plane ----
axD = fig.add_subplot(gs[1, 3:6])
hard_M3 = chi_M3.argmax(axis=1)
ok = np.isfinite(om_mean) & np.isfinite(gp_mean)
sizes = 18 + 200 * pi_w[ok] / pi_w[ok].max()
M3_palette = [ARM_PALETTE[0], ARM_PALETTE[1], ARM_PALETTE[2]]
M3_labels = ['Pirouette', 'Slow Run', 'Fast Run']
for j in range(3):
    sel_b = (hard_M3 == j) & ok
    axD.scatter(om_mean[sel_b], gp_mean[sel_b],
                c=M3_palette[j], s=sizes[sel_b[ok]],
                alpha=0.75, edgecolor='0.25', linewidth=0.3)
    cx = np.average(om_mean[sel_b], weights=pi_w[sel_b])
    cy = np.average(gp_mean[sel_b], weights=pi_w[sel_b])
    axD.scatter([cx], [cy], marker='X', s=110, c=M3_palette[j],
                edgecolor='black', linewidth=0.7, zorder=5)
axD.axvline(0, color='0.5', lw=0.5, ls='--')
axD.set_xlabel(r'cluster $\bar{\omega}$  (body wave/s)',
               fontsize=10, fontweight='bold')
axD.set_ylabel(r'cluster $\langle|\theta|\rangle_{s,t}$  (rad)',
               fontsize=10, fontweight='bold')
axD.tick_params(labelsize=7)
D_handles = [
    Line2D([0], [0], marker='o', linestyle='none',
           markerfacecolor=M3_palette[j], markeredgecolor='0.25',
           markeredgewidth=0.4, markersize=8,
           label=f'Basin {j+1} ({M3_labels[j]})')
    for j in range(3)
]
leg_D = axD.legend(handles=D_handles, prop={'size': 8, 'weight': 'bold'},
                   loc='upper right', frameon=True, facecolor='white',
                   edgecolor='0.3', framealpha=1.0,
                   labelspacing=0.6, handletextpad=0.5, borderpad=0.5)
leg_D.get_frame().set_linewidth(0.8)
axD.text(0, 1, 'D', transform=_letter_trans(axD), fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- E: LOO-CV held-out basin MI vs random 2-coloring null ----
axE = fig.add_subplot(gs[2, 1:5])
xs = np.arange(len(mi_obs_loo))
axE.scatter(xs, mi_obs_loo, color=ARM_PALETTE[0], s=46, alpha=0.9,
            edgecolor='0.3', linewidth=0.4, label='Held-out')
axE.scatter(xs, mi_null_loo, color='0.55', s=22, alpha=0.85,
            marker='s', edgecolor='0.3', linewidth=0.3, label='Random')
axE.set_xticks(xs); axE.set_xticklabels([str(i+1) for i in xs], fontsize=7)
axE.set_xlabel('held-out worm index', fontsize=10, fontweight='bold')
axE.set_ylabel(r'$I(\mathrm{Basin}_t;\,\mathrm{Basin}_{t+\tau})$  (bits)',
               fontsize=10, fontweight='bold')
axE.set_yscale('log')
axE.tick_params(labelsize=7)
leg_E = axE.legend(prop={'size': 8, 'weight': 'bold'}, frameon=True,
                   loc='center right', bbox_to_anchor=(0.98, 0.5),
                   facecolor='white', edgecolor='0.3', framealpha=1.0)
leg_E.get_frame().set_linewidth(0.8)
axE.text(0, 1, 'E', transform=_letter_trans(axE), fontsize=14,
         fontweight='bold', ha='left', va='bottom')


save(plt.gcf(), 'supp_figure_3')
