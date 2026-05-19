"""Refined lognormal/trunc-PL diagnostic figure for worms (v3)."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.special import gamma as gamma_fn
import powerlaw
import warnings
warnings.filterwarnings('ignore')


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = WORMS_DATA  # data location
M = 2
BASIN_COLORS = ['#D55E00', '#0072B2']  # pirouette=red, run=blue (match S4)
BASIN_NAMES = ['pirouette', 'run']

d = np.load(os.path.join(OUT, 'lognormal_reanalysis_worms.npz'),
            allow_pickle=True)
dw = np.load(os.path.join(OUT, 'lognormal_pooled_dwells_worms.npz'))
tp = np.load(os.path.join(OUT, 'per_worm_truncpl_fits.npz'))

dwells_basin = [dw['basin0'], dw['basin1']]
sigma_slow = d['sigma_slow']
std_logtau = d['std_logtau']
R_pl_ln = d['R_pl_ln']
R_pl_tp = tp['per_worm_R_tp']
ged_eta = d['ged_eta']
slow_dev = [d[f'pooled_slow_dev_basin{b}'] for b in range(M)]

# Compute pooled fits per basin
pool_fits = []
for b in range(M):
    fit = powerlaw.Fit(dwells_basin[b], discrete=False, verbose=False)
    pool_fits.append(dict(
        pl_alpha=fit.power_law.alpha, pl_xmin=fit.power_law.xmin,
        ln_mu=fit.lognormal.mu, ln_sigma=fit.lognormal.sigma,
        tp_alpha=fit.truncated_power_law.alpha,
        tp_lambda=fit.truncated_power_law.parameter2))

fig = plt.figure(figsize=(11.5, 9.5))
gs = gridspec.GridSpec(2, 2, hspace=0.46, wspace=0.32,
                        left=0.08, right=0.97, top=0.94, bottom=0.07)

# --- A: pooled CCDFs with three fits ---
axA = fig.add_subplot(gs[0, 0])
for b in range(M):
    d_arr = dwells_basin[b]
    sd = np.sort(d_arr)
    ccdf = 1 - np.arange(len(sd))/len(sd)
    axA.loglog(sd[:-1], ccdf[:-1], color=BASIN_COLORS[b], lw=1.6,
               alpha=0.95)

    r = pool_fits[b]
    xmin = r['pl_xmin']
    idx_xmin = np.searchsorted(sd, xmin)
    if idx_xmin >= len(sd): continue
    ccdf_at_xmin = 1 - idx_xmin/len(sd)
    tau_fit = np.logspace(np.log10(xmin), np.log10(sd.max()*1.2), 80)

    ccdf_pl = ccdf_at_xmin * (tau_fit/xmin)**(-(r['pl_alpha']-1))
    axA.loglog(tau_fit, ccdf_pl, color=BASIN_COLORS[b], lw=0.9, ls='--',
               alpha=0.85)

    ln_sf = stats.norm.sf((np.log(tau_fit) - r['ln_mu'])/r['ln_sigma'])
    ln_sf_at_xmin = stats.norm.sf((np.log(xmin) - r['ln_mu'])/r['ln_sigma'])
    if ln_sf_at_xmin > 0:
        ccdf_ln = ccdf_at_xmin * ln_sf/ln_sf_at_xmin
        axA.loglog(tau_fit, ccdf_ln, color=BASIN_COLORS[b], lw=0.9, ls=':',
                   alpha=0.85)

    f_tp = tau_fit**(-r['tp_alpha']) * np.exp(-r['tp_lambda']*tau_fit)
    if f_tp.sum() > 0:
        cdf_tp = np.cumsum(f_tp[:-1] * np.diff(tau_fit))
        if cdf_tp[-1] > 0:
            cdf_tp = cdf_tp / cdf_tp[-1]
            ccdf_tp = ccdf_at_xmin * (1 - cdf_tp)
            axA.loglog(tau_fit[:-1], ccdf_tp, color=BASIN_COLORS[b], lw=1.6,
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
axA.set_xlim(0.05, 5e2)
axA.set_ylim(5e-4, 1.3)

# --- B: Vuong R distributions ---
axB = fig.add_subplot(gs[0, 1])
all_R_ln = R_pl_ln[~np.isnan(R_pl_ln)]
all_R_tp = R_pl_tp[~np.isnan(R_pl_tp)]
bins_R = np.linspace(-5, 3, 16)
axB.hist(all_R_ln, bins=bins_R, color='#888', alpha=0.7,
         label='Power Law vs.\nLog Normal', edgecolor='none')
axB.hist(all_R_tp, bins=bins_R, color='#d55e00', alpha=0.7,
         label='Power Law vs.\nTruncated Power-Law', edgecolor='none')
axB.axvline(0, color='k', lw=1.0)
axB.set_xlabel(r'normalized log-likelihood ratio $R$ (Vuong)',
               fontsize=11, fontweight='bold')
axB.set_ylabel('# of worm-basin pairs', fontsize=11, fontweight='bold')
axB.legend(loc='upper left', fontsize=9, frameon=False,
           labelspacing=0.6)

# --- C: σ_logτ vs σ_slow ---
axC = fig.add_subplot(gs[1, 0])
for b in range(M):
    x = sigma_slow[:, b]; y = std_logtau[:, b]
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 5: continue
    r_b, p_b = stats.pearsonr(x[valid], y[valid])
    axC.scatter(x[valid], y[valid], s=42, color=BASIN_COLORS[b], alpha=0.85,
                edgecolor='none',
                label=f'{BASIN_NAMES[b]}: $r = {r_b:+.2f}$, $p = {p_b:.2g}$')
    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    xx = np.linspace(x[valid].min(), x[valid].max(), 50)
    axC.plot(xx, slope*xx + intercept, color=BASIN_COLORS[b], lw=1.4, alpha=0.9)
axC.set_xlabel(r'$\sigma_{\rm slow}$  (windowed std of $\bar\chi_b$, $W=60$ s)',
               fontsize=11, fontweight='bold')
axC.set_ylabel(r'$\sigma_{\log\tau}$  per worm per basin',
               fontsize=11, fontweight='bold')

# --- D: slow-mode shape ---
axD = fig.add_subplot(gs[1, 1])
def ged_pdf(x, eta, scale):
    return eta / (2 * scale * gamma_fn(1./eta)) * np.exp(-(np.abs(x)/scale)**eta)
for b in range(M):
    z = slow_dev[b]
    if len(z) == 0: continue
    z_std = np.std(z)
    eta = ged_eta[b]
    scale = (np.mean(np.abs(z)**eta))**(1./eta)
    bins = np.linspace(z.mean() - 4*z_std, z.mean() + 4*z_std, 70)
    axD.hist(z, bins=bins, density=True, alpha=0.4, color=BASIN_COLORS[b],
             edgecolor='none')
    xx = np.linspace(bins[0], bins[-1], 300)
    axD.plot(xx, ged_pdf(xx - z.mean(), eta, scale),
             color=BASIN_COLORS[b], lw=1.6,
             label=f'{BASIN_NAMES[b]}: $\\eta={eta:.2f}$')
xx = np.linspace(-4, 4, 300)
axD.plot(xx, stats.norm.pdf(xx), color='k', ls=':', lw=1.0, alpha=0.5,
         label='Gaussian ref. ($\\eta=2$)')
axD.set_xlabel(r'logit$(\bar\chi_b)$ deviation (within-basin)',
               fontsize=11, fontweight='bold')
axD.set_ylabel('density', fontsize=11, fontweight='bold')
axD.legend(loc='upper right', fontsize=9, frameon=False)
axD.set_yscale('log')
axD.set_ylim(1e-3, None)

for ax, letter in [(axA, 'A'), (axB, 'B'), (axC, 'C'), (axD, 'D')]:
    ax.text(-0.13, 1.05, letter, transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')

save(fig, 'supp_figure_5')
plt.close(fig)
