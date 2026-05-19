"""Fig 5 with panel D replaced by sigma_logtau (Costa directional test, log-normal version).

Same as build_fig5_tau2s.py except the per-fly Costa scatter now uses
sigma_logtau = std(log(dwells per fly per arm)) instead of the power-law
exponent alpha.  This statistic does not require fitting a power law and
is far more sensitive to slow-mode modulation: pooled r flips from -0.22
(alpha) to +0.65 (sigma_logtau).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save
ARM_LABELS = [f'arm {j+1}' for j in range(4)]

OUT = FLIES_DATA  # data location
fr = 100; M = 4

ARM_LABELS_CAP = [f'Arm {j+1}' for j in range(4)]

ARM_TITLES = [
    'Arm 1\nIdle & Slow',
    'Arm 2\nAnterior Movements',
    'Arm 3\nPosterior & Wing Movements',
    'Arm 4\nLocomotion',
]

# ---- Load caches ----
dyn = np.load(os.path.join(OUT, 'arm_dynamics_results_tau2s.npz'),
              allow_pickle=True)
ln = np.load(os.path.join(OUT, 'lognormal_reanalysis.npz'), allow_pickle=True)
bdn_path = os.path.join(OUT, 'behavior_density_chi_tau2s.npz')
have_density = os.path.exists(bdn_path)
if have_density:
    bdn = np.load(bdn_path)

lags_s     = dyn['lags_s'] / fr
apparent_r = dyn['apparent_r_hz']
mi_emp     = dyn['mi_emp']
mi_markov  = dyn['mi_markov']

per_fly_sigma  = ln['sigma_slow']        # (30, 4)
per_fly_logtau = ln['std_logtau']        # (30, 4)

# ---- Build figure ----
fig = plt.figure(figsize=(11.5, 8.0))
gs = GridSpec(2, 3, figure=fig, hspace=0.05, wspace=0.32,
              height_ratios=[0.85, 1.0])

# Panel A: chi-weighted behavior density
gs_A = gs[0, :].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.12)
if have_density:
    cond = bdn['cond']; gx = bdn['x_edges']; gy = bdn['y_edges']
    cond_mean = np.nanmean(cond, axis=0, keepdims=True)
    dev = cond - cond_mean
    vmax = np.nanpercentile(np.abs(dev), 99)
    for j in range(M):
        axA = fig.add_subplot(gs_A[0, j])
        im = axA.pcolormesh(gx, gy, dev[j], cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, shading='auto',
                            rasterized=True)
        axA.set_aspect('equal')
        axA.set_title(ARM_TITLES[j], color=ARM_PALETTE[j],
                      fontsize=12, fontweight='bold', pad=4)
        axA.set_xticks([]); axA.set_yticks([])
        for sp in axA.spines.values(): sp.set_visible(False)
        if j == 0:
            axA.text(-0.02, 1.22, 'A', transform=axA.transAxes,
                     fontsize=15, fontweight='bold', ha='left', va='top')
    axA_cb = fig.add_subplot(gs_A[0, 4])
    cb = fig.colorbar(im, cax=axA_cb)
    cb.set_label(r'$\chi_j$ Enrichment Above Uniform',
                 fontsize=10, fontweight='bold')
    cb.ax.tick_params(labelsize=7)

# Panel B: apparent decay rate
axB = fig.add_subplot(gs[1, 0])
for k in range(3):
    axB.loglog(lags_s, apparent_r[:, k], 'o-', color=ARM_PALETTE[k+1],
               ms=3, lw=1.0, alpha=0.9, label=rf'$r_{k+2}$')
axB.set_xlabel(r'lag $\tau$ (s)', fontsize=11, fontweight='bold')
axB.set_ylabel(r'apparent decay rate $-\log|\lambda_k|/\tau$ (s$^{-1}$)',
               fontsize=11, fontweight='bold')
axB.legend(prop={'size': 9, 'weight': 'bold'}, ncol=2, loc='lower left')
axB.text(-0.18, 1.04, 'B', transform=axB.transAxes,
         fontsize=15, fontweight='bold', ha='left', va='bottom')

# Panel C: predictive MI
axC = fig.add_subplot(gs[1, 1])
mask_c = lags_s >= 1.0
axC.semilogx(lags_s[mask_c], mi_emp[mask_c], 'o-', color='0.15',
             ms=3.5, lw=1.2, label='Data')
axC.semilogx(lags_s[mask_c], mi_markov[mask_c], 's--', color='crimson',
             ms=3, lw=1.0, label='Markov')
axC.set_xlabel(r'lag $\tau$ (s)', fontsize=11, fontweight='bold')
axC.set_ylabel(r'$I(\mathrm{Arm}(t);\,\mathrm{Arm}(t+\tau))$  (bits)',
               fontsize=11, fontweight='bold')
leg_C = axC.legend(prop={'size': 9, 'weight': 'bold'}, loc='upper right',
                   frameon=True, facecolor='white', edgecolor='0.3',
                   framealpha=1.0)
leg_C.get_frame().set_linewidth(0.8)
axC.text(-0.18, 1.04, 'C', transform=axC.transAxes,
         fontsize=15, fontweight='bold', ha='left', va='bottom')

# Panel D: per-fly Costa test, sigma_logtau version
axD = fig.add_subplot(gs[1, 2])
per_arm_stats = []
for j in range(M):
    s = per_fly_sigma[:, j]; y = per_fly_logtau[:, j]
    ok = np.isfinite(s) & np.isfinite(y)
    axD.scatter(s[ok], y[ok], c=ARM_PALETTE[j], s=22, alpha=0.8,
                edgecolors='none', zorder=3)
    if ok.sum() >= 3:
        m_j, b_j = np.polyfit(s[ok], y[ok], 1)
        r_j, p_j = pearsonr(s[ok], y[ok])
        xs_j = np.linspace(s[ok].min(), s[ok].max(), 50)
        axD.plot(xs_j, m_j * xs_j + b_j, color=ARM_PALETTE[j],
                 lw=1.6, alpha=0.95, zorder=4)
        per_arm_stats.append((j, r_j, p_j))
axD.set_xlabel(r'slow-mode fluctuation strength ($\sigma_{\mathrm{slow}}$)',
               fontsize=11, fontweight='bold')
axD.set_ylabel(r'dwell-tail width ($\sigma_{\log\tau}$)',
               fontsize=11, fontweight='bold')
for j in range(M):
    axD.scatter([], [], c=ARM_PALETTE[j], s=30, label=ARM_LABELS_CAP[j])
leg_D = axD.legend(loc='lower right', prop={'size': 9, 'weight': 'bold'},
                   handlelength=0.6, handletextpad=0.4,
                   borderpad=0.4, borderaxespad=0.4, labelspacing=0.3,
                   frameon=True, facecolor='white', edgecolor='0.6',
                   framealpha=1.0)
leg_D.get_frame().set_linewidth(0.8)
axD.text(-0.18, 1.04, 'D', transform=axD.transAxes,
         fontsize=15, fontweight='bold', ha='left', va='bottom')

print('Per-arm sigma_logtau Costa stats:')
for j, r_j, p_j in per_arm_stats:
    print(f'  arm {j+1}: r = {r_j:+.3f}, p = {p_j:.3g}')

save(plt.gcf(), 'figure_5')
