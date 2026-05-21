"""Supp. Fig. S5 (worms): residence-time model selection and slow-mode shape.

Three panels:
  A. Pooled per-basin dwell-time CCDFs with power-law, log-normal, and
     truncated-power-law maximum-likelihood fits overlaid.
  B. Per-worm-per-basin Vuong normalized log-likelihood ratios R comparing
     the power law against log-normal and against the truncated power law.
  C. Empirical slow-mode shape: within-basin logit(chi_b) deviation with a
     generalized-error-distribution (GED) fit.

Residences use the canonical Delta=2 s definition
(pipeline.metastable_residences; Methods Sec. "Dwell-time distributions").
The dwell-time fits (A, B) are computed live; the slow-mode shape (C) is a
property of chi and is read from the cached re-analysis.

Companion to Kaur, Jain, & Berman (2026).
"""
import os, sys, pickle, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.special import gamma as gamma_fn
import powerlaw
warnings.filterwarnings('ignore')


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save
from pipeline import metastable_residences, residences_per_individual

OUT = WORMS_DATA  # data location
fr = 16
M = 2
MIN_RES = 30
BASIN_COLORS = ['#D55E00', '#0072B2']  # pirouette=red, run=blue (match S4)
BASIN_NAMES = ['pirouette', 'run']

# Slow-mode shape (GED) is a property of chi; read from the cached re-analysis.
d = np.load(os.path.join(OUT, 'lognormal_reanalysis_worms.npz'),
            allow_pickle=True)
ged_eta = d['ged_eta']
slow_dev = [d[f'pooled_slow_dev_basin{b}'] for b in range(M)]

# Residences (Delta=2 s) from states + memberships.
with open(os.path.join(OUT, 'states_worms.pkl'), 'rb') as f:
    _sd = pickle.load(f)
worm_states = [_sd[i].astype(int) for i in sorted(_sd)]
chi = np.load(os.path.join(OUT, 'gpcca_worms_M2_tau3s.npz'))['chi']
dwells_basin = metastable_residences(worm_states, chi, framerate=fr, delta=2.0)
per_indiv = residences_per_individual(worm_states, chi, framerate=fr, delta=2.0)

# Per-worm-per-basin Vuong ratios (worm-basin pairs with >= MIN_RES residences).
R_pl_ln, R_pl_tp = [], []
for pid in per_indiv:
    for b in range(M):
        dd = pid[b]
        if len(dd) < MIN_RES:
            continue
        fit = powerlaw.Fit(dd, discrete=False, verbose=False)
        r_ln, _ = fit.distribution_compare('power_law', 'lognormal',
                                           normalized_ratio=True)
        r_tp, _ = fit.distribution_compare('power_law', 'truncated_power_law',
                                           normalized_ratio=True)
        R_pl_ln.append(r_ln); R_pl_tp.append(r_tp)
R_pl_ln = np.array(R_pl_ln); R_pl_tp = np.array(R_pl_tp)

# Pooled fits per basin.
pool_fits = []
for b in range(M):
    fit = powerlaw.Fit(dwells_basin[b], discrete=False, verbose=False)
    pool_fits.append(dict(
        pl_alpha=fit.power_law.alpha, pl_xmin=fit.power_law.xmin,
        ln_mu=fit.lognormal.mu, ln_sigma=fit.lognormal.sigma,
        tp_alpha=fit.truncated_power_law.alpha,
        tp_lambda=fit.truncated_power_law.parameter2))

fig = plt.figure(figsize=(14.5, 4.6))
gs = gridspec.GridSpec(1, 3, wspace=0.34,
                       left=0.06, right=0.97, top=0.90, bottom=0.16)

# --- A: pooled CCDFs with three fits ---
axA = fig.add_subplot(gs[0, 0])
for b in range(M):
    sd = np.sort(dwells_basin[b])
    ccdf = 1 - np.arange(len(sd)) / len(sd)
    axA.loglog(sd[:-1], ccdf[:-1], color=BASIN_COLORS[b], lw=1.6, alpha=0.95)
    r = pool_fits[b]; xmin = r['pl_xmin']
    idx_xmin = np.searchsorted(sd, xmin)
    if idx_xmin >= len(sd): continue
    c0 = 1 - idx_xmin / len(sd)
    tau = np.logspace(np.log10(xmin), np.log10(sd.max() * 1.2), 80)
    axA.loglog(tau, c0 * (tau / xmin) ** (-(r['pl_alpha'] - 1)),
               color=BASIN_COLORS[b], lw=0.9, ls='--', alpha=0.85)
    ln_sf = stats.norm.sf((np.log(tau) - r['ln_mu']) / r['ln_sigma'])
    ln0 = stats.norm.sf((np.log(xmin) - r['ln_mu']) / r['ln_sigma'])
    if ln0 > 0:
        axA.loglog(tau, c0 * ln_sf / ln0, color=BASIN_COLORS[b], lw=0.9,
                   ls=':', alpha=0.85)
    ftp = tau ** (-r['tp_alpha']) * np.exp(-r['tp_lambda'] * tau)
    cdf = np.cumsum(ftp[:-1] * np.diff(tau))
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
        axA.loglog(tau[:-1], c0 * (1 - cdf), color=BASIN_COLORS[b], lw=1.6,
                   alpha=0.95)
axA.set_xlabel(r'dwell time $\tau$ (s)', fontsize=11, fontweight='bold')
axA.set_ylabel(r'CCDF $P(T \geq \tau)$', fontsize=11, fontweight='bold')
A_handles = [
    Line2D([0], [0], color=BASIN_COLORS[0], lw=1.8,
           label=f'pirouette basin (n={len(dwells_basin[0])})'),
    Line2D([0], [0], color=BASIN_COLORS[1], lw=1.8,
           label=f'run basin (n={len(dwells_basin[1])})'),
    Line2D([0], [0], color='0.4', lw=1.0, ls='--', label='power law'),
    Line2D([0], [0], color='0.4', lw=1.0, ls=':', label='log-normal'),
    Line2D([0], [0], color='0.4', lw=1.8, label='truncated power-law'),
]
axA.legend(handles=A_handles, loc='lower left', fontsize=9, frameon=False)
axA.set_xlim(0.05, 5e2); axA.set_ylim(5e-4, 1.3)

# --- B: Vuong R distributions ---
axB = fig.add_subplot(gs[0, 1])
bins_R = np.linspace(-5, 3, 16)
axB.hist(R_pl_ln, bins=bins_R, color='#888', alpha=0.7,
         label='Power Law vs.\nLog Normal', edgecolor='none')
axB.hist(R_pl_tp, bins=bins_R, color='#d55e00', alpha=0.7,
         label='Power Law vs.\nTruncated Power-Law', edgecolor='none')
axB.axvline(0, color='k', lw=1.0)
axB.set_xlabel(r'normalized log-likelihood ratio $R$ (Vuong)',
               fontsize=11, fontweight='bold')
axB.set_ylabel('# of worm-basin pairs', fontsize=11, fontweight='bold')
axB.legend(loc='upper left', fontsize=9, frameon=False, labelspacing=0.6)

# --- C: slow-mode shape (GED) ---
axC = fig.add_subplot(gs[0, 2])
def ged_pdf(x, eta, scale):
    return eta / (2 * scale * gamma_fn(1./eta)) * np.exp(-(np.abs(x)/scale)**eta)
for b in range(M):
    z = slow_dev[b]
    if len(z) == 0: continue
    z_std = np.std(z); eta = ged_eta[b]
    scale = (np.mean(np.abs(z)**eta))**(1./eta)
    bins = np.linspace(z.mean() - 4*z_std, z.mean() + 4*z_std, 70)
    axC.hist(z, bins=bins, density=True, alpha=0.4, color=BASIN_COLORS[b],
             edgecolor='none')
    xx = np.linspace(bins[0], bins[-1], 300)
    axC.plot(xx, ged_pdf(xx - z.mean(), eta, scale),
             color=BASIN_COLORS[b], lw=1.6,
             label=f'{BASIN_NAMES[b]}: $\\eta={eta:.2f}$')
xx = np.linspace(-4, 4, 300)
axC.plot(xx, stats.norm.pdf(xx), color='k', ls=':', lw=1.0, alpha=0.5,
         label='Gaussian ref. ($\\eta=2$)')
axC.set_xlabel(r'logit$(\bar\chi_b)$ deviation (within-basin)',
               fontsize=11, fontweight='bold')
axC.set_ylabel('density', fontsize=11, fontweight='bold')
axC.legend(loc='upper right', fontsize=9, frameon=False)
axC.set_yscale('log'); axC.set_ylim(1e-3, None)

for ax, letter in [(axA, 'A'), (axB, 'B'), (axC, 'C')]:
    ax.text(-0.13, 1.05, letter, transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')

save(fig, 'supp_figure_5')
plt.close(fig)
