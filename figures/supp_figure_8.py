"""Fly supp — Costa heavy-tail per-fly test (Fig. S8).

Three panels (after dropping old D = mean sigma_slow(W) and old E = example
scatters at W = 10 / 600 s; D's claim was wrong and E was redundant with C).

  A. Per-fly dwell CCDFs, all four arms, coloured by sigma_slow.
  B. Per-arm Costa scatter (4 sub-panels), regression line + per-arm r in caption.
     Disattenuated per-arm correlations now live in B's caption (relocated from
     old E's caption).
  C. Pooled & per-arm Pearson r(sigma_slow(W), alpha) vs window W, spanning
     the full width of row 3 as a banner panel.
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from scipy.stats import pearsonr
from scipy.special import erf
import powerlaw as plw


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = FLIES_DATA  # data location
STATES = os.path.join(FLIES_DATA, 'states_flies.pkl')
fr = 100; M = 4

ARM_TITLES = [
    'Arm 1 — Idle & Slow',
    'Arm 2 — Anterior Movements',
    'Arm 3 — Posterior & Wing Movements',
    'Arm 4 — Locomotion',
]

# ---- Caches ----
refit = np.load(os.path.join(OUT, 'per_fly_pcca_refit_tau2s.npz'))
per_fly_alpha   = refit['per_fly_alpha']
per_fly_sigma   = refit['per_fly_sigma']

with open(STATES, 'rb') as f:
    states_dict = pickle.load(f)
fly_states = [states_dict[i].astype(int) for i in sorted(states_dict)]
N_flies = len(fly_states)

z_pcca = np.load(os.path.join(OUT, 'gpcca_flies_M4_tau2s.npz'))
assignments = z_pcca['assignments']
fly_arm_seq = [assignments[fs] for fs in fly_states]

W_sweep = np.load(os.path.join(OUT, 'costa_W_sweep_tau2s.npz'))
W_grid_s_sw = W_sweep['W_grid_s']
sigma_W_sw  = W_sweep['sigma_W']
r_pooled    = W_sweep['r_pooled'];  p_pooled = W_sweep['p_pooled']
r_arm       = W_sweep['r_arm'];     p_arm    = W_sweep['p_arm']
mean_sigma  = W_sweep['mean_sigma']

# ==================================================================
fig = plt.figure(figsize=(14.0, 10.0))
gs = GridSpec(3, 4, figure=fig, wspace=0.45, hspace=0.55,
              left=0.06, right=0.97, top=0.95, bottom=0.06,
              height_ratios=[1.0, 1.0, 1.06])

# ---- A (was D): per-fly dwell CCDFs ----
def runs(x, val):
    d = np.diff(np.concatenate([[False], x == val, [False]]).astype(int))
    starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
    return ends - starts

axA = [fig.add_subplot(gs[0, j]) for j in range(M)]
for armj in range(M):
    ax = axA[armj]
    sigma_values = per_fly_sigma[:, armj]
    rng = np.ptp(sigma_values)
    sigma_norm = ((sigma_values - np.nanmin(sigma_values)) / rng) if rng > 0 \
                 else np.full_like(sigma_values, 0.5)
    for fi, s in enumerate(fly_arm_seq):
        rl = runs(s, armj) / fr
        rl = rl[rl >= 0.5]
        if len(rl) < 3: continue
        xs = np.sort(rl); ys = 1 - np.arange(len(xs)) / len(xs)
        col = cm.magma(0.15 + 0.7 *
                       (sigma_norm[fi] if np.isfinite(sigma_norm[fi]) else 0.5))
        ax.loglog(xs, ys, color=col, lw=0.8, alpha=0.6)
    xref = np.logspace(0, 2.5, 20)
    ax.loglog(xref, (xref / xref[0]) ** (-1.0), 'k--', lw=0.9,
              label=r'$\mu=2$ (slope $-1$)')

    # Per-arm pooled lognormal MLE overlay (gray dashed). Computed on the
    # pool across all flies for this arm; normalized to the data CCDF at
    # xmin so the overlay represents the typical bulk shape.
    pooled = []
    for s in fly_arm_seq:
        rl = runs(s, armj) / fr
        pooled.append(rl[rl >= 0.5])
    pooled = np.concatenate(pooled)
    if len(pooled) >= 30:
        fit = plw.Fit(pooled, discrete=False, verbose=False)
        xmin = float(fit.power_law.xmin)
        mu_LN = float(fit.lognormal.mu)
        sigma_LN = float(fit.lognormal.sigma)
        R_LN, p_LN = fit.distribution_compare('power_law', 'lognormal',
                                               normalized_ratio=True)
        frac_above_xmin = float(np.mean(pooled >= xmin))
        x_grid = np.logspace(np.log10(xmin), np.log10(pooled.max()), 200)
        full_ccdf = 0.5 * (1.0 - erf((np.log(x_grid) - mu_LN)
                                      / (sigma_LN * np.sqrt(2.0))))
        overlay = full_ccdf * (frac_above_xmin / full_ccdf[0])
        sig = '' if p_LN >= 0.05 else ('*' if p_LN >= 0.01
                                       else ('**' if p_LN >= 0.001 else '***'))
        ax.loglog(x_grid, overlay, color='0.25', ls='--', lw=1.4,
                  label='Log-Normal Fit')

    ax.set_xlabel('dwell time (s)', fontsize=10, fontweight='bold')
    ax.set_ylabel(r'$P(\mathrm{dwell} \geq t)$' if armj == 0 else '',
                  fontsize=10, fontweight='bold')
    ax.set_title(ARM_TITLES[armj], color=ARM_PALETTE[armj],
                 fontsize=10, fontweight='bold')
    ax.legend(prop={'size': 8, 'weight': 'bold'},
              loc='lower left', frameon=False)
    ax.tick_params(labelsize=7)
    if armj == 0:
        ax.text(-0.36, 1.10, 'A', transform=ax.transAxes,
                fontsize=15, fontweight='bold', ha='left', va='bottom')

# ---- B (was E): per-arm Costa scatter ----
axB = [fig.add_subplot(gs[1, j]) for j in range(M)]
for armj in range(M):
    ax = axB[armj]
    s = per_fly_sigma[:, armj]; a = per_fly_alpha[:, armj]
    ok = np.isfinite(s) & np.isfinite(a)
    ax.scatter(s[ok], a[ok], c=ARM_PALETTE[armj], s=22, alpha=0.85,
               edgecolors='none')
    if ok.sum() >= 4:
        m_, b_ = np.polyfit(s[ok], a[ok], 1)
        xs = np.linspace(s[ok].min()*0.9, s[ok].max()*1.1, 60)
        ax.plot(xs, m_ * xs + b_, color='0.15', lw=1.0)
    ax.set_xlabel(r'$\sigma_{\mathrm{slow}}$ (W = 60 s)',
                  fontsize=10, fontweight='bold')
    ax.set_ylabel(r'$\alpha$' if armj == 0 else '',
                  fontsize=10, fontweight='bold')
    ax.set_title(ARM_TITLES[armj], color=ARM_PALETTE[armj],
                 fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=7)
    if armj == 0:
        ax.text(-0.36, 1.10, 'B', transform=ax.transAxes,
                fontsize=15, fontweight='bold', ha='left', va='bottom')

# ---- C: pooled + per-arm r(W), banner panel spanning row 3 ----
axC = fig.add_subplot(gs[2, :])
for j in range(M):
    sig_mask = p_arm[:, j] < 0.05
    axC.semilogx(W_grid_s_sw, r_arm[:, j], 'o-', color=ARM_PALETTE[j],
                 ms=4, lw=1.2, label=f'Arm {j+1}')
    if sig_mask.any():
        axC.scatter(W_grid_s_sw[sig_mask], r_arm[sig_mask, j],
                    s=80, facecolors='none', edgecolors='black',
                    linewidths=0.7, zorder=10)
axC.semilogx(W_grid_s_sw, r_pooled, 'k--s', ms=3.5, lw=1.4, label='Pooled')
axC.set_xlim(3, W_grid_s_sw.max() * 1.05)
axC.set_xlabel('window size $W$ (s)', fontsize=11, fontweight='bold')
axC.set_ylabel(r'Pearson $r(\sigma_{\mathrm{slow}}(W),\,\alpha)$',
               fontsize=11, fontweight='bold')
leg_C = axC.legend(prop={'size': 10, 'weight': 'bold'}, loc='upper left',
                   ncol=1, handlelength=1.2, labelspacing=0.3,
                   frameon=True, facecolor='white', edgecolor='0.3',
                   framealpha=1.0)
leg_C.get_frame().set_linewidth(0.8)
axC.tick_params(labelsize=8)
axC.text(-0.06, 1.04, 'C', transform=axC.transAxes,
         fontsize=15, fontweight='bold', ha='left', va='bottom')

save(plt.gcf(), 'supp_figure_8')
# Print stats for the caption
print('\n--- For caption of panel B (per-arm Costa) ---')
for j in range(M):
    s = per_fly_sigma[:, j]; a = per_fly_alpha[:, j]
    ok = np.isfinite(s) & np.isfinite(a)
    if ok.sum() >= 4:
        rr, pp = pearsonr(s[ok], a[ok])
        print(f'  {ARM_TITLES[j]}: n={ok.sum()}, r={rr:+.2f}, p={pp:.2e}')
