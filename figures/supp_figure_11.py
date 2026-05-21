"""Supp. Fig. S11: residence-time cutoffs are robust to the smoothing scale Delta.

The metastable residences are defined from the soft membership smoothed with a
Delta=2 s moving average. If the heavy tail / long exponential cutoff were a
product of that smoothing, the fitted cutoff 1/lambda would scale with Delta.
We sweep Delta in {0 (raw hard argmax), 0.5, 1, 2, 5} s and refit the truncated
power law f(tau) ~ tau^-alpha exp(-lambda tau) (powerlaw pkg, KS-min xmin) per
arm/basin, for both species. The exponential cutoff 1/lambda stays at tens to
hundreds of seconds across Delta -- one to two decades above Delta -- so it is
set by the dynamics, not the smoother. (Delta=0 is the raw sub-second flicker
regime shown for contrast; Delta=2 s, the working value, is marked.)

  A,B: worms -- alpha vs Delta (A), cutoff 1/lambda vs Delta (B)
  C,D: flies -- alpha vs Delta (C), cutoff 1/lambda vs Delta (D)

Companion to Kaur, Jain, & Berman (2026).
"""
import os, sys, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')
import powerlaw
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

DELTAS = [0.0, 0.5, 1.0, 2.0, 5.0]
WORK = 2.0
WORM_COL = ['#D55E00', '#0072B2']   # pirouette=red, run=blue (match S4/S5)

FLY = dict(states=os.path.join(FLIES_DATA, 'states_flies.pkl'),
           npz=os.path.join(FLIES_DATA, 'gpcca_flies_M4_tau2s.npz'),
           fr=100, M=4, arms=['Arm 1', 'Arm 2', 'Arm 3', 'Arm 4'],
           cols=[ARM_PALETTE[i] for i in range(4)])
WORM = dict(states=os.path.join(WORMS_DATA, 'states_worms.pkl'),
            npz=os.path.join(WORMS_DATA, 'gpcca_worms_M2_tau3s.npz'),
            fr=16, M=2, arms=['Pirouette', 'Run'], cols=WORM_COL)


def load_states(path):
    with open(path, 'rb') as f:
        sd = pickle.load(f)
    return [sd[i].astype(int) for i in sorted(sd)]


def label_seq(chi_seq, delta, fr):
    if delta == 0:
        return np.argmax(chi_seq, axis=1)
    w = max(1, int(delta * fr))
    return np.argmax(uniform_filter1d(chi_seq, size=w, axis=0, mode='nearest'),
                     axis=1)


def runs(lab, M, fr):
    out = [[] for _ in range(M)]
    cur = lab[0]; n = 1
    for a in lab[1:]:
        if a == cur:
            n += 1
        else:
            out[cur].append(n); cur = a; n = 1
    out[cur].append(n)
    return [np.array(o, float) / fr for o in out]


def fit_tpl(d):
    if len(d) < 30:
        return np.nan, np.nan, np.nan
    fit = powerlaw.Fit(d, verbose=False)
    R_tp, _ = fit.distribution_compare('power_law', 'truncated_power_law',
                                       normalized_ratio=True)
    tp = fit.truncated_power_law
    inv = (1.0 / tp.parameter2) if (tp.parameter2 and tp.parameter2 > 0) else np.nan
    return tp.parameter1, inv, R_tp


def sweep(cfg):
    seqs = load_states(cfg['states'])
    chi = np.load(cfg['npz'])['chi']
    fr, M = cfg['fr'], cfg['M']
    alpha = np.full((len(DELTAS), M), np.nan)
    invlam = np.full((len(DELTAS), M), np.nan)
    Rtp = np.full((len(DELTAS), M), np.nan)
    print(f'\n=== {cfg["arms"]} : truncated-PL fits vs Delta ===')
    for di, delta in enumerate(DELTAS):
        pool = [[] for _ in range(M)]
        for s in seqs:
            d = runs(label_seq(chi[s], delta, fr), M, fr)
            for a in range(M):
                pool[a].append(d[a])
        pool = [np.concatenate(x) for x in pool]
        for a in range(M):
            alpha[di, a], invlam[di, a], Rtp[di, a] = fit_tpl(pool[a])
        msg = '  '.join(f'{cfg["arms"][a]}: a={alpha[di,a]:.2f},1/l={invlam[di,a]:.0f}s'
                        for a in range(M))
        print(f'  Delta={delta:>4}s | {msg}')
    return alpha, invlam, Rtp


fa, fl, fr_ = sweep(FLY)
wa, wl, wr_ = sweep(WORM)

dd = np.array(DELTAS)
LAB, LEG, LET = 12.5, 9.4, 14
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
specs = [(WORM, wa, wl, 0), (FLY, fa, fl, 1)]   # worms first (mentioned first)
for cfg, A, L, row in specs:
    axa, axl = axes[row, 0], axes[row, 1]
    for a in range(cfg['M']):
        axa.plot(dd, A[:, a], 'o-', color=cfg['cols'][a], lw=1.6, ms=5,
                 label=cfg['arms'][a])
        axl.plot(dd, L[:, a], 'o-', color=cfg['cols'][a], lw=1.6, ms=5,
                 label=cfg['arms'][a])
    for ax in (axa, axl):
        ax.axvline(WORK, color='0.6', ls=':', lw=1.2)
        ax.set_xlabel(r'membership smoothing $\Delta$ (s)', fontsize=LAB,
                      fontweight='bold')
    axa.axhline(2.0, color='0.7', ls='--', lw=0.8)
    axa.set_ylabel('truncated power-law\nexponent ($\\alpha$)', fontsize=LAB,
                   fontweight='bold')
    axl.set_yscale('log')
    axl.set_ylabel(r'exponential cutoff $1/\lambda$ (s)', fontsize=LAB,
                   fontweight='bold')
    axa.set_title(chr(65 + 2 * row), loc='left', fontsize=LET, fontweight='bold')
    axl.set_title(chr(66 + 2 * row), loc='left', fontsize=LET, fontweight='bold')
    axa.legend(prop={'size': LEG, 'weight': 'bold'}, ncol=2, frameon=False)
plt.tight_layout()
save(fig, 'supp_figure_11')
