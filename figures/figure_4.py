"""Fig. 4 (method) v3 — fly section, panel A = 3D + PR contrast, panel B = 3D + 2D.

Editor note (PRX Life pre-review): the v2 panel B is the central geometric
claim of the paper (arms-and-hub at M=4), but the single 3D Matplotlib
scatter foreshortens the linear-arm structure and crowds the hub. v3 splits
panel B into a 3D view on top and a 2D projection onto the (phi_3, phi_4)
plane below, where the four-arm structure is visually unambiguous.

Panel A's localization claim reads fine in 3D alone (three spike points are
0-dimensional structures), so the corresponding 2D projection that v3
originally placed below A's 3D was redundant.  In its place we promote
S6 panel F (participation-ratio comparison of the fixed- vs
multi-timescale operators) into the main figure, with leave-one-fly-out
SEM error bars rendered as filled circles.  This makes the localized-vs-
collective contrast between A and B both geometric (3D scatters) and
quantitative (PR contrast panel).

Layout (2 columns x 2 rows outer):
  Row 1: A (3D + PR contrast)  |  B (3D + 2D projection)
  Row 2: C (eigenvalues)       |  D (chi_j(t) for a representative fly)
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D


ARM_LABELS_CAP = [f'Arm {j+1}' for j in range(4)]

from _paths import *  # setup() + ARM_PALETTE + *_DATA + save
ARM_LABELS = [f'arm {j+1}' for j in range(4)]

OUT = FLIES_DATA  # data location
STATES = os.path.join(FLIES_DATA, 'states_flies.pkl')

# ---- Caches ----
eigs = np.load(os.path.join(FLIES_DATA, 'fly_eigs_tau2s.npz'))
pf = eigs['phi_fix'][:, :3]; pm = eigs['phi_mt'][:, :3]

z_pcca = np.load(os.path.join(OUT, 'gpcca_flies_M4_tau2s.npz'))
chi = z_pcca['chi']; pi_fly = z_pcca['pi']; assignments = z_pcca['assignments']

with open(STATES, 'rb') as f:
    states_dict = pickle.load(f)
fly_states = [states_dict[i].astype(int) for i in sorted(states_dict)]

# Leave-one-fly-out PR cache (produced by compute_pr_leave_one_out.py)
pr_cache = np.load(os.path.join(OUT, 'pr_leave_one_out.npz'))
pr_fix_full = pr_cache['pr_fix_full']
pr_mt_full  = pr_cache['pr_mt_full']
pr_fix_loo  = pr_cache['pr_fix_loo']
pr_mt_loo   = pr_cache['pr_mt_loo']

# ---- View / axis convention (notebook) ----
ELEV, AZIM = 15, 120
def xyz_of(p):
    """3D plot axis convention: x = phi_3, y = phi_4, z = phi_2."""
    return p[..., 1], p[..., 2], p[..., 0]

def xy_of(p):
    """2D projection: x = phi_3, y = phi_4 (i.e., looking down the phi_2 axis)."""
    return p[..., 1], p[..., 2]

# ---- Fixed-timescale highlighted clusters ----
norms_f = np.linalg.norm(pf, axis=1)
order = np.argsort(norms_f)[::-1]
fixed_spike_clusters = []; seen = []
for cid in order:
    u = pf[cid] / np.linalg.norm(pf[cid])
    if any(abs(u @ v) > 0.9 for v in seen): continue
    seen.append(u); fixed_spike_clusters.append(cid)
    if len(fixed_spike_clusters) >= 3: break
highlight_colors = ['#d7263d', '#1b998b', '#e3a72f']

# ---- Multi-timescale arm geometry (centroid - hub) ----
hub = (pm * pi_fly[:, None]).sum(axis=0) / pi_fly.sum()
arm_dirs = np.zeros((4, 3)); arm_centroids = np.zeros((4, 3))
arm_lengths = np.zeros(4)
for j in range(4):
    mask = chi[:, j] > 0.5
    if mask.sum() < 3: mask = chi[:, j] > 0.3
    pts = pm[mask]; pi_pts = pi_fly[mask]
    centroid = (pts * pi_pts[:, None]).sum(axis=0) / pi_pts.sum()
    arm_centroids[j] = centroid
    d = centroid - hub
    arm_dirs[j] = d / max(np.linalg.norm(d), 1e-12)
    arm_lengths[j] = np.linalg.norm(d)

# ---- Eigenvalue spectrum (panel C) ----
from pipeline import make_transition_matrix
all_states = np.concatenate(fly_states)
TM = make_transition_matrix(all_states, lag=int(z_pcca['tau']))
TM = TM / np.maximum(TM.sum(axis=1, keepdims=True), 1e-12)
evals_full = np.linalg.eigvals(TM)
evals = np.sort(np.abs(evals_full))[::-1]
evals_nt = evals[1:]

# ---- Representative fly chi time series (panel D) ----
rep_fly = 10
rep_chi = chi[fly_states[rep_fly], :]
T_show = 20 * 60 * 100
rep_chi = rep_chi[:T_show]
t_rep = np.arange(len(rep_chi)) / 100 / 60
window = 5 * 100; kernel = np.ones(window) / window
rep_chi_smooth = np.zeros_like(rep_chi)
for j in range(4):
    padded = np.pad(rep_chi[:, j], window//2, mode='edge')
    rep_chi_smooth[:, j] = np.convolve(padded, kernel, mode='valid')[:len(rep_chi)]

# ==================================================================
# Build figure
# ==================================================================
# 2x3 layout:
#   Row 1: A (fixed 3D) | B (multi 3D) | (B') multi 2D projection
#   Row 2: C (eigvals)  | D (PR)       | E (chi)
fig = plt.figure(figsize=(14.0, 8.0))
gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32,
              height_ratios=[1.25, 1.0])

# Helper: draw a single 2D-projection panel for either fixed or multi.
def draw_2d_panel(ax, points, mode):
    """mode in {'fix', 'multi'}.
    For 'fix': scatter all clusters in grey, highlight the 3 spike clusters.
    For 'multi': basin-coloured clusters, hub marker, arm rays.
    """
    if mode == 'fix':
        xx, yy = xy_of(points)
        ax.scatter(xx, yy, c='0.55', s=5, alpha=0.5, edgecolors='none')
        for k, cid in enumerate(fixed_spike_clusters):
            x0, y0 = xy_of(points[cid])
            ax.scatter([x0], [y0], c=highlight_colors[k], s=80,
                       edgecolors='k', linewidths=0.6, zorder=10)
    else:
        fuzzy = chi.max(axis=1) < 0.5
        xx, yy = xy_of(points[fuzzy])
        ax.scatter(xx, yy, c='0.85', s=5, alpha=0.4, edgecolors='none')
        for j in range(4):
            m = (assignments == j) & ~fuzzy
            bx, by = xy_of(points[m])
            ax.scatter(bx, by, c=ARM_PALETTE[j], s=10, alpha=0.85,
                       edgecolors='none', label=ARM_LABELS_CAP[j])
        hx, hy = xy_of(hub)
        for j in range(4):
            ex, ey = xy_of(arm_centroids[j])
            ax.plot([hx, ex], [hy, ey], color=ARM_PALETTE[j],
                    lw=2.8, zorder=10, solid_capstyle='round')
        ax.scatter([hx], [hy], s=140, marker='*', c='red', zorder=20,
                   edgecolors='black', linewidths=0.5)
    ax.set_xlabel(r'$\phi_3$', fontsize=11, fontweight='bold', labelpad=0)
    ax.set_ylabel(r'$\phi_4$', fontsize=11, fontweight='bold', labelpad=0)
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='datalim')


def draw_pr_panel(ax):
    """Participation-ratio contrast: fixed vs multi-timescale operators,
    filled circles with leave-one-fly-out SEM error bars, log y-axis."""
    n_k = len(pr_fix_full)
    k_axis = np.arange(1, n_k + 1)
    n_flies_loo = pr_fix_loo.shape[0]
    sem_fix = pr_fix_loo.std(axis=0, ddof=1) / np.sqrt(n_flies_loo)
    sem_mt  = pr_mt_loo.std(axis=0,  ddof=1) / np.sqrt(n_flies_loo)

    color_fix = '#B22222'   # firebrick — negative control
    color_mt  = '#222222'   # near-black — main operator

    ax.errorbar(k_axis, pr_fix_full, yerr=sem_fix,
                fmt='o', color=color_fix, ms=8, capsize=3.5, lw=1.2,
                markerfacecolor=color_fix, markeredgecolor='black',
                markeredgewidth=0.5, zorder=3)
    ax.errorbar(k_axis, pr_mt_full, yerr=sem_mt,
                fmt='o', color=color_mt, ms=8, capsize=3.5, lw=1.2,
                markerfacecolor=color_mt, markeredgecolor='black',
                markeredgewidth=0.5, zorder=3)

    ax.set_yscale('log')
    ax.set_xticks(k_axis)
    ax.set_xticklabels([f'{k}' for k in k_axis])
    ax.set_xlim(0.5, n_k + 0.5)
    ax.set_xlabel('eigenvector index $k$', fontsize=11, fontweight='bold')
    ax.set_ylabel('participation ratio', fontsize=11, fontweight='bold')
    pr_handles = [
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor=color_mt, markeredgecolor='black',
               markeredgewidth=0.5, markersize=8, label='multi-timescale'),
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor=color_fix, markeredgecolor='black',
               markeredgewidth=0.5, markersize=8, label='fixed-timescale'),
    ]
    pr_leg = ax.legend(handles=pr_handles,
                       prop={'size': 9, 'weight': 'bold'},
                       frameon=True, loc='center right',
                       bbox_to_anchor=(0.98, 0.55),
                       handlelength=0.9, handletextpad=0.4, borderpad=0.4,
                       edgecolor='0.3', facecolor='white', framealpha=1.0)
    pr_leg.get_frame().set_linewidth(0.8)
    ax.tick_params(labelsize=8)

# -- Panel A (top-left): fixed-timescale 3D --
axA = fig.add_subplot(gs[0, 0], projection='3d')
fx, fy, fz = xyz_of(pf)
axA.scatter(fx, fy, fz, c='0.2', s=6, alpha=0.5,
            edgecolors='none', depthshade=False)
for k, cid in enumerate(fixed_spike_clusters):
    p = pf[cid]; px, py, pz = xyz_of(p)
    axA.scatter([px], [py], [pz], c=highlight_colors[k],
                s=70, edgecolors='k', linewidths=0.6, zorder=10)
axA.set_xlabel(r'$\phi_3$', fontsize=11, fontweight='bold', labelpad=-3)
axA.set_ylabel(r'$\phi_4$', fontsize=11, fontweight='bold', labelpad=-3)
axA.set_zlabel(r'$\phi_2$', fontsize=11, fontweight='bold', labelpad=-3)
axA.view_init(elev=ELEV, azim=AZIM)
axA.tick_params(labelsize=6, pad=-3)
axA.text2D(-0.06, 1.04, 'A', transform=axA.transAxes, fontsize=14,
           fontweight='bold', ha='left', va='bottom')

# -- Panel B (top-middle): multi-timescale 3D --
axB3 = fig.add_subplot(gs[0, 1], projection='3d')
fuzzy = chi.max(axis=1) < 0.5
fx, fy, fz = xyz_of(pm[fuzzy])
axB3.scatter(fx, fy, fz, c='0.85', s=4, alpha=0.4,
             edgecolors='none', depthshade=False)
for j in range(4):
    m = (assignments == j) & ~fuzzy
    bx, by, bz = xyz_of(pm[m])
    axB3.scatter(bx, by, bz, c=ARM_PALETTE[j], s=10, alpha=0.85,
                 edgecolors='none', depthshade=False)
hx, hy, hz = xyz_of(hub)
axB3.scatter([hx], [hy], [hz], s=140, marker='*', c='red', zorder=20,
             edgecolors='black', linewidths=0.5)
for j in range(4):
    end = arm_centroids[j]
    ex, ey, ez = xyz_of(end)
    axB3.plot([hx, ex], [hy, ey], [hz, ez], color=ARM_PALETTE[j],
              lw=3.0, zorder=10, solid_capstyle='round')
axB3.set_xlabel(r'$\phi_3$', fontsize=11, fontweight='bold', labelpad=-3)
axB3.set_ylabel(r'$\phi_4$', fontsize=11, fontweight='bold', labelpad=-3)
axB3.set_zlabel(r'$\phi_2$', fontsize=11, fontweight='bold', labelpad=-3)
axB3.view_init(elev=ELEV, azim=AZIM)
axB3.tick_params(labelsize=6, pad=-3)
axB3.text2D(-0.06, 1.04, 'B', transform=axB3.transAxes, fontsize=14,
            fontweight='bold', ha='left', va='bottom')

# -- Panel B' (top-right): multi-timescale 2D projection (companion to B) --
axB2 = fig.add_subplot(gs[0, 2])
draw_2d_panel(axB2, pm, 'multi')
arm_handles = [
    Line2D([0], [0], marker='o', linestyle='none',
           markerfacecolor=ARM_PALETTE[j], markeredgecolor='none',
           markersize=7, label=ARM_LABELS_CAP[j])
    for j in range(4)
]
axB2.legend(handles=arm_handles, prop={'size': 9, 'weight': 'bold'},
            loc='lower right', frameon=False,
            handlelength=1.0, handletextpad=0.4, labelspacing=0.2,
            borderpad=0.2)

# -- Panel C (bottom-left): eigenvalue spectrum --
axC = fig.add_subplot(gs[1, 0])
k_axis = np.arange(2, 12)
axC.semilogy(k_axis, evals_nt[:10], 'o-', color='0.15', ms=5, lw=1.2)
axC.axvline(4, color=ARM_PALETTE[0], lw=1.0, ls='--')
axC.set_xlabel('eigenvalue index $k$', fontsize=11, fontweight='bold')
axC.set_ylabel(r'$|\lambda_k|$', fontsize=11, fontweight='bold')
axC.set_xlim(1.5, 11.5)
axC.text(-0.18, 1.05, 'C', transform=axC.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# -- Panel D (bottom-middle): participation-ratio contrast --
axD = fig.add_subplot(gs[1, 1])
draw_pr_panel(axD)
axD.text(-0.18, 1.05, 'D', transform=axD.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# -- Panel E (bottom-right): representative fly chi(t) --
axE = fig.add_subplot(gs[1, 2])
for j in range(4):
    axE.plot(t_rep, rep_chi_smooth[:, j], color=ARM_PALETTE[j],
             lw=1.0, alpha=0.95, label=ARM_LABELS_CAP[j])
axE.set_xlabel('time (min)', fontsize=11, fontweight='bold')
axE.set_ylabel(r'$\chi_j(t)$', fontsize=11, fontweight='bold')
axE.set_ylim(-0.02, 1.02)
axE.set_xlim(0, 20)
axE.text(-0.18, 1.05, 'E', transform=axE.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')
axE.legend(loc='upper right', prop={'size': 9, 'weight': 'bold'},
           ncol=4, handlelength=1.4, columnspacing=1.0, borderpad=0.2)

save(plt.gcf(), 'figure_4')
