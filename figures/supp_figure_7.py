"""Supp. Fig. S7 (flies): parameter selection, spectral diagnostics, and
basin-count justification — 8 panels (2 rows x 4 cols).

The participation-ratio comparison is shown in main Fig. 4 panel D (with
leave-one-fly-out SEM error bars); the remaining diagnostics are here.

Layout (2 x 4):
  Row 1:
    A. PCA cumulative variance vs PC index (selects 15 PCs)
    B. Cao's E_1(d) saturation (selects d = 12)
    C. Entropy gap Delta-h(N) (selects N = 1000)
    D. Multi-timescale |lambda_k(tau)| vs lag
  Row 2:
    E. Fixed-timescale |lambda_k(tau)| vs lag
    F. tau sweep: crispness, gap |lambda_5|/|lambda_6|, vestigial count vs tau
    G. M sweep at tau = 2 s: basin sizes (sorted) vs M
    H. Held-out CV PI per transition vs M (leave-one-fly-out)

Companion to Kaur, Jain, & Berman (2026).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = FLIES_DATA  # data location
d  = np.load(os.path.join(OUT, 'flies_supp_method_data.npz'))
v2 = np.load(os.path.join(OUT, 'flies_supp_method_v2_data.npz'))
cv = np.load(os.path.join(OUT, 'cv_flies_tau2s.npz'))

# ---- CV stats ----
pi_mat = cv['pi_matrix']
M_cv   = cv['M_values']
mean_pi = np.nanmean(pi_mat, axis=0)
n_valid = np.sum(~np.isnan(pi_mat), axis=0)
sem_pi  = np.nanstd(pi_mat, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))

# Bootstrap CI of mean PI per M (n_boot = 5000)
RNG = np.random.default_rng(0)
n_indiv = pi_mat.shape[0]
ci_lo = np.zeros(len(M_cv)); ci_hi = np.zeros(len(M_cv))
for j, _ in enumerate(M_cv):
    col = pi_mat[:, j]; col = col[~np.isnan(col)]
    if len(col) == 0: continue
    boot = np.array([col[RNG.integers(0, len(col), len(col))].mean()
                     for _ in range(5000)])
    ci_lo[j] = np.quantile(boot, 0.025)
    ci_hi[j] = np.quantile(boot, 0.975)

# ==================================================================
fig = plt.figure(figsize=(17.0, 7.5))
gs = GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.45)

# ------ A: PCA cumulative variance ------
axA = fig.add_subplot(gs[0, 0])
var = d['pca_var']; k = np.arange(1, len(var) + 1)
cum = np.cumsum(var) / var.sum()
axA.plot(k, cum, 'o-', color='0.15', ms=4, lw=1.2)
axA.axvline(15, color=ARM_PALETTE[1], lw=1.0, ls='--')
axA.set_xlabel('PC index', fontsize=11, fontweight='bold')
axA.set_ylabel('cumulative variance explained', fontsize=11, fontweight='bold')
axA.set_ylim(0, 1.05)
axA.text(-0.22, 1.04, 'A', transform=axA.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ------ B: Cao's E_1(d) ------
axB = fig.add_subplot(gs[0, 1])
Ks = d['cao_Ks']; E1 = d['cao_E1']
axB.plot(Ks, E1, 'o-', color='0.15', ms=4, lw=1.2)
axB.axvline(12, color=ARM_PALETTE[1], lw=1.0, ls='--')
axB.set_xlabel(r'embedding dim. $d$', fontsize=11, fontweight='bold')
axB.set_ylabel(r"Cao's $E_1(d)$", fontsize=11, fontweight='bold')
axB.text(-0.22, 1.04, 'B', transform=axB.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ------ C: entropy gap Delta-h vs N ------
axC = fig.add_subplot(gs[0, 2])
Ns = d['entropy_N']; diff = d['entropy_diff']
axC.semilogx(Ns, diff, 'o-', color='0.15', ms=4, lw=1.2)
axC.axvline(1200, color=ARM_PALETTE[1], lw=1.0, ls='--')
axC.set_xlabel(r'cluster count $N$', fontsize=11, fontweight='bold')
axC.set_ylabel(r'entropy gap $\Delta h$ (nats)', fontsize=11, fontweight='bold')
axC.text(-0.22, 1.04, 'C', transform=axC.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ------ D: multi-timescale |lambda_k(tau)| ------
axD = fig.add_subplot(gs[0, 3])
lags_s = d['lags_s']; em = d['eigs_mt']
for k in range(1, 5):
    axD.loglog(lags_s, em[:, k], 'o-', color=ARM_PALETTE[(k - 1) % 4],
               ms=3, lw=1.0, label=rf'$|\lambda_{k+1}|$')
axD.axvline(2.0, color='0.4', lw=0.8, ls=':')
axD.set_xlabel(r'lag $\tau$ (s)', fontsize=11, fontweight='bold')
axD.set_ylabel(r'$|\lambda_k(\tau)|$  (multi-timescale)',
               fontsize=11, fontweight='bold')
axD.legend(prop={'size': 8, 'weight': 'bold'}, ncol=2, loc='lower left')
axD.text(-0.22, 1.04, 'D', transform=axD.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ------ E: fixed-timescale |lambda_k(tau)| ------
axE = fig.add_subplot(gs[1, 0])
ef = d['eigs_ft']
for k in range(1, 5):
    axE.loglog(lags_s, ef[:, k], 's--', color=ARM_PALETTE[(k - 1) % 4],
               ms=3, lw=1.0, label=rf'$|\lambda_{k+1}|$')
axE.set_xlabel(r'lag $\tau$ (s)', fontsize=11, fontweight='bold')
axE.set_ylabel(r'$|\lambda_k(\tau)|$  (fixed-timescale)',
               fontsize=11, fontweight='bold')
axE.legend(prop={'size': 8, 'weight': 'bold'}, ncol=2, loc='lower left')
axE.text(-0.22, 1.04, 'E', transform=axE.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ------ F: tau sweep ------
axF = fig.add_subplot(gs[1, 1])
ax2 = axF.twinx()
taus = v2['tau_taus']
crisp = v2['tau_crisp']; gap = v2['tau_gap_5_6']; vest = v2['tau_vest_cnt']
l1, = axF.semilogx(taus, crisp, 'o-', color='C0', ms=4, lw=1.2,
                    label='crispness (left)')
l2, = axF.semilogx(taus, gap,   's-', color='C3', ms=4, lw=1.2,
                    label=r'$|\lambda_5|/|\lambda_6|$ (left)')
l3, = ax2.semilogx(taus, vest,  '^-', color='0.4',  ms=4, lw=1.2,
                    label='vest count (right)')
ax2.set_yscale('log')
ax2.set_ylabel('vestigial-basin\ncluster count', fontsize=11, fontweight='bold')
axF.set_xlabel(r'lag $\tau$ (s)', fontsize=11, fontweight='bold')
axF.set_ylabel('crispness  /  spectral-gap ratio',
               fontsize=11, fontweight='bold')
axF.axvline(2.0, color='0.4', lw=0.8, ls=':')
# Place legend below the plot so it doesn't overlap the curves
axF.legend(handles=[l1, l2, l3], prop={'size': 8, 'weight': 'bold'},
           loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3,
           frameon=False, handlelength=1.3, columnspacing=1.0)
axF.text(-0.22, 1.04, 'F', transform=axF.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ------ G: M sweep at tau=2s, basin sizes ------
axG = fig.add_subplot(gs[1, 2])
M_vals = v2['M_values']
sizes  = v2['basin_sizes']            # (len(M_vals), max(M_vals))
N_clust = 1000
viridis = plt.cm.viridis(np.linspace(0.1, 0.92, sizes.shape[1]))
for mi, M in enumerate(M_vals):
    bottom = 0.0
    row = sizes[mi]
    for j, v in enumerate(row):
        if not np.isfinite(v) or v == 0: continue
        axG.bar(M, v, bottom=bottom, color=viridis[j],
                edgecolor='w', lw=0.4, width=0.7)
        bottom += v
axG.axhline(5, color='0.4', lw=0.8, ls=':')
axG.set_xticks(M_vals)
axG.set_xlabel('basin count $M$', fontsize=11, fontweight='bold')
axG.set_ylabel('basin size (clusters), sorted',
               fontsize=11, fontweight='bold')
axG.set_ylim(0, N_clust * 1.05)
axG.text(-0.22, 1.04, 'G', transform=axG.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ------ H: held-out CV PI vs M ------
axH = fig.add_subplot(gs[1, 3])
for k in range(n_indiv):
    axH.plot(M_cv, pi_mat[k], color='0.7', lw=0.6, alpha=0.5)
axH.fill_between(M_cv, ci_lo, ci_hi, color='C0', alpha=0.2)
axH.errorbar(M_cv, mean_pi, yerr=sem_pi, color='C0', lw=2.0, marker='o',
             ms=4, capsize=3)
elbow_M = 4
axH.axvline(elbow_M, color=ARM_PALETTE[0], lw=1.0, ls='--')
axH.set_xticks(M_cv)
axH.set_xlabel('M (number of basins)', fontsize=11, fontweight='bold')
axH.set_ylabel('held-out PI per transition\n(bits-equivalent)',
               fontsize=11, fontweight='bold')
axH.text(-0.22, 1.04, 'H', transform=axH.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

save(plt.gcf(), 'supp_figure_7')
