"""Refined lognormal/trunc-PL diagnostic figure (v3).

Updates v2 to incorporate truncated-PL as the primary alternative
functional form (better fit than LN at the pooled level). Four panels:

  A: pooled per-arm CCDFs with PL, LN, and trunc-PL fits overlaid
  B: side-by-side Vuong R histograms (PL vs LN and PL vs trunc-PL per fly)
  C: σ_logτ vs σ_slow per fly per arm (same as v2)
  D: slow-mode shape with GED fit (same as v2)
"""
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.special import gamma as gamma_fn
import powerlaw
import warnings
warnings.filterwarnings('ignore')


ARM_TITLES = [
    'Arm 1 — Idle & Slow',
    'Arm 2 — Anterior Movements',
    'Arm 3 — Posterior & Wing Movements',
    'Arm 4 — Locomotion',
]

from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = FLIES_DATA  # data location
# Core data
d = np.load(os.path.join(OUT, 'lognormal_reanalysis.npz'), allow_pickle=True)
dw = np.load(os.path.join(OUT, 'lognormal_pooled_dwells.npz'))
M = 4

# Per-fly trunc-PL data
tp = np.load(os.path.join(OUT, 'per_fly_truncpl_fits.npz'))

# Pooled trunc-PL fit params
with open(os.path.join(OUT, 'fly_pooled_fits.pkl'), 'rb') as f:
    fly_pool = pickle.load(f)
pool_results = fly_pool['results']

dwells_arm = {a: dw[f'arm{a}'] for a in range(M)}
sigma_slow = d['sigma_slow']
std_logtau = d['std_logtau']
R_pl_ln = d['R_pl_ln']
R_pl_tp = tp['per_fly_R_tp']
ln_mu = d['ln_mu']; ln_sig = d['ln_sig']
ged_eta = d['ged_eta']
slow_dev = [d[f'pooled_slow_dev_arm{a}'] for a in range(M)]

fig = plt.figure(figsize=(11.5, 9.5))
gs = gridspec.GridSpec(2, 2, hspace=0.46, wspace=0.32,
                        left=0.08, right=0.97, top=0.94, bottom=0.07)

# --- A: pooled CCDFs with PL, LN, trunc-PL overlays ---
axA = fig.add_subplot(gs[0, 0])
for a in range(M):
    d_arm = dwells_arm[a]
    sd = np.sort(d_arm)
    ccdf = 1 - np.arange(len(sd)) / len(sd)
    axA.loglog(sd[:-1], ccdf[:-1], color=ARM_PALETTE[a], lw=1.5,
               alpha=0.95)

    # Use pooled fits from fly_pooled_fits.pkl
    r = pool_results[a]
    xmin = r['pl_xmin']
    idx_xmin = np.searchsorted(sd, xmin)
    if idx_xmin >= len(sd):
        continue
    ccdf_at_xmin = 1 - idx_xmin / len(sd)
    tau_fit = np.logspace(np.log10(xmin), np.log10(sd.max()*1.2), 80)

    # PL fit
    ccdf_pl = ccdf_at_xmin * (tau_fit / xmin) ** (-(r['pl_alpha'] - 1))
    axA.loglog(tau_fit, ccdf_pl, color=ARM_PALETTE[a], lw=0.9, ls='--',
               alpha=0.85)

    # LN fit (conditional on t ≥ xmin)
    ln_sf = stats.norm.sf((np.log(tau_fit) - r['ln_mu']) / r['ln_sigma'])
    ln_sf_at_xmin = stats.norm.sf((np.log(xmin) - r['ln_mu']) / r['ln_sigma'])
    if ln_sf_at_xmin > 0:
        ccdf_ln = ccdf_at_xmin * ln_sf / ln_sf_at_xmin
        axA.loglog(tau_fit, ccdf_ln, color=ARM_PALETTE[a], lw=0.9, ls=':',
                   alpha=0.85)

    # Truncated PL fit
    f_tp = tau_fit**(-r['tp_alpha']) * np.exp(-r['tp_lambda'] * tau_fit)
    if f_tp.sum() > 0:
        cdf_tp = np.cumsum(f_tp[:-1] * np.diff(tau_fit))
        if cdf_tp[-1] > 0:
            cdf_tp = cdf_tp / cdf_tp[-1]
            ccdf_tp = ccdf_at_xmin * (1 - cdf_tp)
            axA.loglog(tau_fit[:-1], ccdf_tp, color=ARM_PALETTE[a], lw=1.6,
                       alpha=0.95)

axA.set_xlabel(r'dwell time $\tau$ (s)', fontsize=11, fontweight='bold')
axA.set_ylabel(r'CCDF $P(T \geq \tau)$', fontsize=11, fontweight='bold')
A_handles = [Line2D([0], [0], color=ARM_PALETTE[a], lw=1.8,
                    label=ARM_TITLES[a]) for a in range(M)]
A_handles += [
    Line2D([0], [0], color='0.4', lw=1.0, ls='--', label='power law'),
    Line2D([0], [0], color='0.4', lw=1.0, ls=':', label='log-normal'),
    Line2D([0], [0], color='0.4', lw=1.8, label='truncated power-law'),
]
axA.legend(handles=A_handles, loc='lower left',
           prop={'size': 9, 'weight': 'bold'}, frameon=False)
axA.set_xlim(0.05, 5e3)
axA.set_ylim(5e-5, 1.3)

# --- B: Per-fly Vuong R histograms for both alternatives ---
axB = fig.add_subplot(gs[0, 1])
all_R_ln = R_pl_ln[~np.isnan(R_pl_ln)]
all_R_tp = R_pl_tp[~np.isnan(R_pl_tp)]
bins_R = np.linspace(-15, 5, 30)
axB.hist(all_R_ln, bins=bins_R, color='#888', alpha=0.7,
         label='Power Law vs.\nLog Normal', edgecolor='none')
axB.hist(all_R_tp, bins=bins_R, color='#d55e00', alpha=0.7,
         label='Power Law vs.\nTruncated Power-Law', edgecolor='none')
axB.axvline(0, color='k', lw=1.0)
axB.set_xlabel(r'normalized log-likelihood ratio $R$ (Vuong test)',
               fontsize=11, fontweight='bold')
axB.set_ylabel('# of fly-arm pairs', fontsize=11, fontweight='bold')
axB.legend(loc='upper left', prop={'size': 9, 'weight': 'bold'},
           frameon=False, labelspacing=0.6)

# --- C: σ_logτ vs σ_slow scatter (unchanged) ---
axC = fig.add_subplot(gs[1, 0])
for a in range(M):
    x = sigma_slow[:, a]; y = std_logtau[:, a]
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 5: continue
    axC.scatter(x[valid], y[valid], s=22, color=ARM_PALETTE[a], alpha=0.75,
                edgecolor='none', label=f'Arm {a+1}')
    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    xx = np.linspace(x[valid].min(), x[valid].max(), 50)
    axC.plot(xx, slope*xx + intercept, color=ARM_PALETTE[a], lw=1.2, alpha=0.9)
axC.set_xlabel(r'$\sigma_{\rm slow}$  (windowed std of $\bar\chi_a$, $W=60$ s)',
               fontsize=11, fontweight='bold')
axC.set_ylabel(r'$\sigma_{\log\tau}$  per fly per arm',
               fontsize=11, fontweight='bold')
axC.legend(loc='upper left', prop={'size': 10, 'weight': 'bold'},
           frameon=False)

# --- D: slow-mode shape (unchanged) ---
axD = fig.add_subplot(gs[1, 1])
def ged_pdf(x, eta, scale):
    return eta / (2 * scale * gamma_fn(1./eta)) * np.exp(-(np.abs(x)/scale)**eta)
for a in range(M):
    z = slow_dev[a]
    if len(z) == 0: continue
    z_std = np.std(z)
    eta = ged_eta[a]
    scale = (np.mean(np.abs(z)**eta))**(1./eta)
    bins = np.linspace(z.mean() - 4*z_std, z.mean() + 4*z_std, 80)
    axD.hist(z, bins=bins, density=True, alpha=0.4, color=ARM_PALETTE[a],
             edgecolor='none')
    xx = np.linspace(bins[0], bins[-1], 300)
    axD.plot(xx, ged_pdf(xx - z.mean(), eta, scale),
             color=ARM_PALETTE[a], lw=1.4,
             label=f'Arm {a+1}: $\\eta={eta:.2f}$')
xx = np.linspace(-4, 4, 300)
axD.plot(xx, stats.norm.pdf(xx), color='k', ls=':', lw=1.0, alpha=0.5,
         label='Gaussian ref. ($\\eta=2$)')
axD.set_xlabel(r'logit$(\bar\chi_a)$ deviation (within-basin)',
               fontsize=11, fontweight='bold')
axD.set_ylabel('density', fontsize=11, fontweight='bold')
axD.legend(loc='upper right', prop={'size': 10, 'weight': 'bold'},
           frameon=False)
axD.set_yscale('log')
axD.set_ylim(1e-3, None)

for ax, letter in [(axA, 'A'), (axB, 'B'), (axC, 'C'), (axD, 'D')]:
    ax.text(-0.13, 1.05, letter, transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')

save(fig, 'supp_figure_10')
plt.close(fig)
