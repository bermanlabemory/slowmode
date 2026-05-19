"""Standalone panel: eigenvalue magnitude vs lag at beta=0.5, multi vs fixed.

Promoted from supp Fig S1C to main Fig 2C. Output is a single-panel PNG and
PDF that will be hand-placed into Fig 2 in Illustrator.
"""
import os, sys, numpy as np
import matplotlib.pyplot as plt


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = LORENZ_DATA  # data location
d = np.load(os.path.join(OUT, 'lorenz_supp_data.npz'))

fig, ax = plt.subplots(figsize=(3.6, 3.0))
lags_mt = d['lags_s']; em = d['eigs_mt_0p5']
lags_ft = d['lags_s']; ef = d['eigs_ft_0p5']
for kk in range(1, 4):
    col = ARM_PALETTE[(kk - 1) % 4]
    ax.loglog(lags_mt, em[:, kk], 'o-', color=col, ms=3.5, lw=1.0)
    ax.loglog(lags_ft, ef[:, kk], 's--', color=col, ms=3, lw=0.8, alpha=0.55)
ax.plot([], [], 'o-', color='0.15', label='multi-timescale')
ax.plot([], [], 's--', color='0.15', alpha=0.55, label='fixed-timescale')
ax.set_xlabel(r'lag $\tau$ (s)')
ax.set_ylabel(r'$|\lambda_k(\tau)|$')
ax.legend(fontsize=7, frameon=False, loc='lower left')

plt.tight_layout()
save(plt.gcf(), 'figure_2C')
