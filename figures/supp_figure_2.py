"""Worm supplemental figure 1: operator diagnostics (parameter selection
and spectral diagnostics).  7 panels in a 2x4 grid.

Layout:
  Row 1 (parameter selection):
    A. PCA cumulative variance vs shuffled noise floor
    B. Cao's E_1(d) saturation at d=7
    C. Entropy gap Delta h vs N (multi + fixed)
    D. Multi-timescale |lambda_k(tau)|
  Row 2 (spectral diagnostics):
    E. Fixed-timescale |lambda_k(tau)|
    F. Predictive MI vs Markov benchmark (basin-level)
    G. Apparent decay rate r_2(tau) -- basin-level non-Markovianity
       (moved from main figure in v5; tangential to validation message)
    (last cell intentionally empty)
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = WORMS_DATA  # data location
fr   = 16
TAU_S = 3.0

# ---- Caches ----
d   = np.load(os.path.join(OUT, 'worms_supp_data.npz'))
ad  = np.load(os.path.join(OUT, 'arm_dynamics_worms_tau3s.npz'),
              allow_pickle=True)

# ==========================================================
fig = plt.figure(figsize=(15.0, 7.5))
gs = GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.40,
              left=0.05, right=0.97, top=0.94, bottom=0.08)

# ---- A: PCA cumulative variance ----
axA = fig.add_subplot(gs[0, 0])
var = d['pca_var']; shuf = d['pca_var_shuf']
n_show = min(20, len(var))
k = np.arange(1, n_show + 1)
cum_data = np.cumsum(var[:n_show]) / var.sum()
cum_shuf = np.cumsum(shuf[:n_show]) / shuf.sum()
axA.plot(k, cum_data, 'o-', color='0.15', ms=4, lw=1.2, label='data')
axA.plot(k, cum_shuf, 's--', color='crimson', ms=3, lw=1.0, label='shuffled')
n_keep = int(d['pca_n_retained']) if 'pca_n_retained' in d.files else 4
axA.axvline(n_keep, color=ARM_PALETTE[1], lw=1.0, ls='--')
axA.set_xlabel('PC index', fontsize=11, fontweight='bold')
axA.set_ylabel('cumulative variance', fontsize=11, fontweight='bold')
axA.set_ylim(0, 1.05)
axA.legend(prop={'size': 9, 'weight': 'bold'}, frameon=False,
           loc='center right')
axA.text(-0.22, 1.04, 'A', transform=axA.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- B: Cao's E_1(d) ----
axB = fig.add_subplot(gs[0, 1])
Ks = d['cao_Ks']; E1 = d['cao_E1']
axB.plot(Ks, E1, 'o-', color='0.15', ms=4, lw=1.2)
axB.axvline(7, color=ARM_PALETTE[1], lw=1.0, ls='--')
axB.set_xlabel(r'embedding dim. $d$', fontsize=11, fontweight='bold')
axB.set_ylabel(r"Cao's $E_1(d)$", fontsize=11, fontweight='bold')
axB.text(-0.22, 1.04, 'B', transform=axB.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- C: entropy gap Delta h vs N ----
axC = fig.add_subplot(gs[0, 2])
Ns_mt = d['entropy_N']; diff_mt = d['entropy_diff']
axC.semilogx(Ns_mt, diff_mt, 'o-', color='0.15', ms=4, lw=1.2,
             label='Multi')
if 'entropy_N_ft' in d.files:
    Ns_ft = d['entropy_N_ft']; diff_ft = d['entropy_diff_ft']
    axC.semilogx(Ns_ft, diff_ft, 's--', color='crimson', ms=3, lw=1.0,
                 label='Fixed')
axC.axvline(250, color=ARM_PALETTE[1], lw=1.0, ls='--')
axC.set_xlabel(r'cluster count $N$', fontsize=11, fontweight='bold')
axC.set_ylabel(r'entropy gap $\Delta h$ (nats)',
               fontsize=11, fontweight='bold')
axC.legend(prop={'size': 9, 'weight': 'bold'}, frameon=False,
           loc='upper left')
axC.text(-0.22, 1.04, 'C', transform=axC.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- D: multi-timescale |lambda_k(tau)| ----
axD = fig.add_subplot(gs[0, 3])
lags_s = d['lags_s']; em = d['eigs_mt']
for kk in range(1, 5):
    axD.loglog(lags_s, em[:, kk], 'o-', color=ARM_PALETTE[(kk - 1) % 4],
               ms=3, lw=1.0, label=rf'$|\lambda_{kk+1}|$')
axD.axvline(TAU_S, color='0.4', lw=0.8, ls=':')
axD.set_xlabel(r'lag $\tau$ (s)', fontsize=11, fontweight='bold')
axD.set_ylabel(r'$|\lambda_k(\tau)|$  (multi-timescale)',
               fontsize=11, fontweight='bold')
axD.legend(prop={'size': 8, 'weight': 'bold'}, ncol=2, loc='lower left')
axD.text(-0.22, 1.04, 'D', transform=axD.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- E: fixed-timescale |lambda_k(tau)| ----
axE = fig.add_subplot(gs[1, 0])
ef = d['eigs_ft']
for kk in range(1, 5):
    axE.loglog(lags_s, ef[:, kk], 's--', color=ARM_PALETTE[(kk - 1) % 4],
               ms=3, lw=1.0, label=rf'$|\lambda_{kk+1}|$')
axE.set_xlabel(r'lag $\tau$ (s)', fontsize=11, fontweight='bold')
axE.set_ylabel(r'$|\lambda_k(\tau)|$  (fixed-timescale)',
               fontsize=11, fontweight='bold')
axE.legend(prop={'size': 8, 'weight': 'bold'}, ncol=2, loc='upper right')
axE.text(-0.22, 1.04, 'E', transform=axE.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- F: predictive MI vs Markov benchmark (basin-level) ----
axF = fig.add_subplot(gs[1, 1])
lags_ad = ad['lags_s'] / fr
mi_emp = ad['mi_emp']; mi_markov = ad['mi_markov']
axF.semilogx(lags_ad, mi_emp,    'o-', color='0.15', ms=3, lw=1.1,
             label='Data')
axF.semilogx(lags_ad, mi_markov, 's--', color='crimson', ms=3, lw=1.0,
             label='Markov')
axF.set_xlabel(r'lag $\tau$ (s)', fontsize=11, fontweight='bold')
axF.set_ylabel(r'$I(\mathrm{Basin}(t);\,\mathrm{Basin}(t+\tau))$  (bits)',
               fontsize=11, fontweight='bold')
leg_F = axF.legend(prop={'size': 9, 'weight': 'bold'}, loc='upper right',
                   frameon=True, facecolor='white', edgecolor='0.3',
                   framealpha=1.0)
leg_F.get_frame().set_linewidth(0.8)
axF.text(-0.22, 1.04, 'F', transform=axF.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- G: apparent decay rate r_2(tau) (was main panel F in v4) ----
axG = fig.add_subplot(gs[1, 2])
lags_s_ad   = ad['lags_s'] / fr
apparent_r  = ad['apparent_r_hz']
r2 = apparent_r[:, 0]
axG.loglog(lags_s_ad, r2, 'o-', color='0.15', ms=4, lw=1.2,
           label=r'$r_2(\tau)$')
axG.axhline(r2[0], color='0.5', lw=0.8, ls='--', label='Markov')
axG.axvline(TAU_S, color=ARM_PALETTE[0], lw=0.8, ls=':')
axG.set_xlabel(r'lag $\tau$ (s)', fontsize=11, fontweight='bold')
axG.set_ylabel(r'apparent decay rate (s$^{-1}$)',
               fontsize=11, fontweight='bold')
axG.legend(prop={'size': 9, 'weight': 'bold'}, frameon=False,
           loc='lower left')
axG.text(-0.22, 1.04, 'G', transform=axG.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# (last cell at gs[1, 3] is intentionally blank)

save(plt.gcf(), 'supp_figure_2')
