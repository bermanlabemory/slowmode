"""Paper-quality per-worm dwell-time fit figure (target: supp Fig S4).

For each of the 12 worms in the Stephens-Broekmans dataset, show the
empirical dwell-time CCDF in each of the two G-PCCA basins (pirouette,
run), with three candidate fits overlaid: pure power-law (dashed),
log-normal (dotted), and truncated power-law (solid red). This is the
diagnostic underlying the per-individual α columns in the body text:
the visual story is that fits of any two-parameter family are noisy at
per-worm sample sizes (n_dwells ≈ 50–200), justifying the use of the
fit-free statistic σ_logτ at the per-individual level.
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import powerlaw
import warnings
warnings.filterwarnings('ignore')
from scipy import stats


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = WORMS_DATA  # data location
fr = 16; M = 2
BASIN_COLORS = ['#D55E00', '#0072B2']  # pirouette=red, run=blue
BASIN_NAMES = ['pirouette', 'run']

# --- Load data ---
with open(os.path.join(WORMS_DATA, 'states_worms.pkl'), 'rb') as f:
    states_dict = pickle.load(f)
worm_states = [states_dict[i].astype(int) for i in sorted(states_dict)]
N_worms = len(worm_states)
z = np.load(os.path.join(OUT, 'gpcca_worms_M2_tau3s.npz'))
chi = z['chi']; assign = z['assignments']
worm_arm = [assign[ws] for ws in worm_states]

# Per-worm dwell sequences
def dwells_per_worm(arm_seq, M):
    out = [[] for _ in range(M)]
    cur = arm_seq[0]; n = 1
    for a in arm_seq[1:]:
        if a == cur: n += 1
        else: out[cur].append(n); cur = a; n = 1
    out[cur].append(n)
    return [np.array(o, dtype=float)/fr for o in out]
dwells = [dwells_per_worm(s, M) for s in worm_arm]

# ============================================================
# Compute per-worm fits and Vuong tests for both basins
# ============================================================
fits = {}  # fits[(w, b)] = dict with all fit parameters and R values
for w in range(N_worms):
    for b in range(M):
        d = dwells[w][b]
        if len(d) < 30:
            fits[(w, b)] = None
            continue
        try:
            fit = powerlaw.Fit(d, discrete=False, verbose=False)
            R_ln, p_ln = fit.distribution_compare('power_law', 'lognormal',
                                                    normalized_ratio=True)
            R_tp, p_tp = fit.distribution_compare('power_law',
                                                    'truncated_power_law',
                                                    normalized_ratio=True)
            fits[(w, b)] = dict(
                alpha=fit.power_law.alpha, xmin=fit.power_law.xmin,
                ln_mu=fit.lognormal.mu, ln_sig=fit.lognormal.sigma,
                tp_alpha=fit.truncated_power_law.alpha,
                tp_lam=fit.truncated_power_law.parameter2,
                R_ln=R_ln, R_tp=R_tp, n=len(d))
        except Exception:
            fits[(w, b)] = None

# ============================================================
# Figure: 3x4 grid, one panel per worm
# ============================================================
fig = plt.figure(figsize=(14, 10.5))
gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.30,
              left=0.06, right=0.97, top=0.92, bottom=0.07)

for w in range(N_worms):
    ax = fig.add_subplot(gs[w // 4, w % 4])

    for b in range(M):
        d = dwells[w][b]
        if len(d) < 5:
            continue
        # Empirical CCDF
        sd = np.sort(d)
        ccdf = 1 - np.arange(len(sd))/len(sd)
        ax.loglog(sd[:-1], ccdf[:-1], color=BASIN_COLORS[b], lw=1.0,
                  marker='o', ms=2.0, mfc=BASIN_COLORS[b], mec='none',
                  alpha=0.7)

        f = fits.get((w, b))
        if f is None:
            continue
        alpha, xmin = f['alpha'], f['xmin']
        mu, sig = f['ln_mu'], f['ln_sig']
        tp_alpha, tp_lam = f['tp_alpha'], f['tp_lam']

        # Anchor all fits at CCDF(xmin)
        idx_xmin = np.searchsorted(sd, xmin)
        if idx_xmin >= len(sd):
            continue
        ccdf_at_xmin = 1 - idx_xmin/len(sd)
        tau_fit = np.logspace(np.log10(xmin), np.log10(sd.max()*1.2), 80)

        # PL fit
        ccdf_pl = ccdf_at_xmin * (tau_fit / xmin) ** (-(alpha - 1))
        ax.loglog(tau_fit, ccdf_pl, color=BASIN_COLORS[b], lw=1.0,
                  ls='--', alpha=0.85)

        # LN fit (conditional on t >= xmin)
        ln_sf = stats.norm.sf((np.log(tau_fit) - mu) / sig)
        ln_sf_at_xmin = stats.norm.sf((np.log(xmin) - mu) / sig)
        if ln_sf_at_xmin > 0:
            ccdf_ln = ccdf_at_xmin * ln_sf / ln_sf_at_xmin
            ax.loglog(tau_fit, ccdf_ln, color=BASIN_COLORS[b], lw=0.9,
                      ls=':', alpha=0.85)

        # Truncated PL fit (conditional on t >= xmin)
        f_tp = tau_fit**(-tp_alpha) * np.exp(-tp_lam * tau_fit)
        if np.all(np.isfinite(f_tp)) and f_tp.sum() > 0:
            cdf_tp = np.cumsum(f_tp[:-1] * np.diff(tau_fit))
            if cdf_tp[-1] > 0:
                cdf_tp = cdf_tp / cdf_tp[-1]
                ccdf_tp = ccdf_at_xmin * (1 - cdf_tp)
                ax.loglog(tau_fit[:-1], ccdf_tp, color=BASIN_COLORS[b],
                          lw=1.3, alpha=0.95)

    # Per-worm annotation: PL α / trunc-PL α / n_dwells per basin
    lines = []
    for b in range(M):
        f = fits.get((w, b))
        n_b = len(dwells[w][b])
        if f is None:
            lines.append((f'n={n_b}, fit failed', BASIN_COLORS[b]))
        else:
            lines.append(
                (f'$\\alpha_{{\\rm PL}}$={f["alpha"]:.2f}, '
                 f'$\\alpha_{{\\rm tr}}$={f["tp_alpha"]:.2f}, n={n_b}',
                 BASIN_COLORS[b]))
    y_pos = 0.04
    for txt, color in reversed(lines):
        ax.text(0.04, y_pos, txt, transform=ax.transAxes, fontsize=7.5,
                color=color, fontweight='bold')
        y_pos += 0.07

    ax.set_title(f'Worm {w+1}', fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.set_xlim(0.05, 5e3)
    ax.set_ylim(1e-3, 1.5)
    if w % 4 == 0:
        ax.set_ylabel('CCDF  P(T ≥ τ)', fontsize=11, fontweight='bold')
    if w >= 8:
        ax.set_xlabel('dwell time τ (s)', fontsize=11, fontweight='bold')

# Single shared legend
legend_elems = [
    Line2D([0], [0], color=BASIN_COLORS[0], marker='o', ms=4, lw=1.0,
           mec='none', label='pirouette empirical'),
    Line2D([0], [0], color=BASIN_COLORS[1], marker='o', ms=4, lw=1.0,
           mec='none', label='run empirical'),
    Line2D([0], [0], color='gray', lw=1.0, ls='--',
           label='pure power-law fit'),
    Line2D([0], [0], color='gray', lw=0.9, ls=':',
           label='lognormal fit'),
    Line2D([0], [0], color='gray', lw=1.3,
           label='truncated power-law fit'),
]
fig.legend(handles=legend_elems, loc='upper center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 0.97), frameon=False)

save(fig, 'supp_figure_4')
