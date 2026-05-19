"""Assemble the Lorenz supplementary figure.

After promoting the eigenvalue-vs-lag panel to main Fig 2C, the supplement
contains 4 panels in a 2x2 grid:
  A. Cao's E1 for fixed-timescale (d=8 selection).
  B. Cao's E1 for multi-timescale (d=7 selection).
  C. Mean dwell time vs inverse temperature beta (was D).
  D. Cluster-count selection: Delta_h vs N (multi + fixed) at beta=0.5 (was E).
"""
import os, sys, numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = LORENZ_DATA  # data location
d = np.load(os.path.join(OUT, 'lorenz_supp_data.npz'))

fig = plt.figure(figsize=(7.8, 7.0))
gs = GridSpec(2, 2, figure=fig, wspace=0.45, hspace=0.45)

# ---- A: Cao fixed ----
axA = fig.add_subplot(gs[0, 0])
axA.plot(d['cao_Ks_ft'], d['cao_E1_ft'], 'o-', color='0.15', ms=4, lw=1.1)
axA.axvline(8, color='crimson', lw=1.0, ls='--')
axA.set_xlabel(r'embedding dim. $d$', fontsize=11, fontweight='bold')
axA.set_ylabel(r"Cao's $E_1(d)$ (fixed)", fontsize=11, fontweight='bold')
axA.text(-0.25, 1.04, 'A', transform=axA.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- B: Cao multi ----
axB = fig.add_subplot(gs[0, 1])
axB.plot(d['cao_Ks_mt'], d['cao_E1_mt'], 'o-', color='0.15', ms=4, lw=1.1)
axB.axvline(7, color=ARM_PALETTE[1], lw=1.0, ls='--')
axB.set_xlabel(r'embedding dim. $d$', fontsize=11, fontweight='bold')
axB.set_ylabel(r"Cao's $E_1(d)$ (multi)", fontsize=11, fontweight='bold')
axB.text(-0.25, 1.04, 'B', transform=axB.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- C: dwell time vs beta (was D in old layout) ----
axC = fig.add_subplot(gs[1, 0])
axC.semilogy(d['betas'], d['dwell_mean'], 'o-', color='0.15', ms=4, lw=1.2)
axC.set_xlabel(r'inverse temperature $\beta$', fontsize=11, fontweight='bold')
axC.set_ylabel(r'mean dwell time $\langle \tau \rangle$ (s)',
               fontsize=11, fontweight='bold')
axC.text(-0.25, 1.04, 'C', transform=axC.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

# ---- D: Delta_h vs N (cluster-count selection) (was E in old layout) ----
axD = fig.add_subplot(gs[1, 1])
Ns = d['dh_N']
axD.semilogx(Ns, d['dh_multi'], 'o-', color='0.15', ms=4, lw=1.2,
             label='multi')
axD.semilogx(Ns, d['dh_fixed'], 's--', color='crimson', ms=3.5, lw=1.0,
             label='fixed')
axD.axvline(1300, color='0.4', lw=0.8, ls=':', alpha=0.8)
axD.set_xlabel(r'cluster count $N$', fontsize=11, fontweight='bold')
axD.set_ylabel(r'entropy gap $\Delta h$ (nats/s)',
               fontsize=11, fontweight='bold')
axD.legend(prop={'size': 9, 'weight': 'bold'}, frameon=False,
           loc='upper left')
axD.text(-0.25, 1.04, 'D', transform=axD.transAxes, fontsize=14,
         fontweight='bold', ha='left', va='bottom')

save(plt.gcf(), 'supp_figure_1')
