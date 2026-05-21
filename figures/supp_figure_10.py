"""Supp. Fig. S10: the heavy fly residence-time tails are not an artifact of
the membership smoothing.

We hold the per-cluster soft membership chi and the Delta=2 s smoothing+argmax
residence pipeline FIXED, and replace only the temporal cluster sequence with a
first-order (one-step) Markov chain fit to the lag-1 cluster transitions of the
data. A first-order chain has no memory beyond one frame; exit times from a
metastable set of clusters are asymptotically exponential. Because the surrogate
passes through the identical chi and the identical Delta=2 s smoother, any tail
the smoothing could "manufacture" is present in both curves -- so a heavier data
tail is genuine (super-Markovian) dynamics, and the long cutoff cannot be a
product of the 2 s smoother.

  A-D: four fly basins, data vs one-step-Markov surrogate (CCDF)
  E-F: two worm basins, data vs surrogate
  G:   tail mass (fraction of residences > 30 s), data vs surrogate, all basins

Companion to Kaur, Jain, & Berman (2026).
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

DELTA = 2.0
rng = np.random.default_rng(0)

FLY = dict(states=os.path.join(FLIES_DATA, 'states_flies.pkl'),
           npz=os.path.join(FLIES_DATA, 'gpcca_flies_M4_tau2s.npz'),
           fr=100, M=4,
           arms=['Arm 1 (Idle & Slow)', 'Arm 2 (Anterior)',
                 'Arm 3 (Post. & Wing)', 'Arm 4 (Locomotion)'])
WORM = dict(states=os.path.join(WORMS_DATA, 'states_worms.pkl'),
            npz=os.path.join(WORMS_DATA, 'gpcca_worms_M2_tau3s.npz'),
            fr=16, M=2,
            arms=['Pirouette', 'Run'])
WORM_COL = ['#D55E00', '#0072B2']   # pirouette=red, run=blue (match S4/S5)


def load_states(path):
    with open(path, 'rb') as f:
        sd = pickle.load(f)
    return [sd[i].astype(int) for i in sorted(sd)]


def build_T(seqs, Ncl):
    """Row-stochastic lag-1 cluster transition matrix, pooled over individuals."""
    T = np.zeros((Ncl, Ncl))
    for s in seqs:
        np.add.at(T, (s[:-1], s[1:]), 1.0)
    rs = T.sum(1)
    empty = np.flatnonzero(rs == 0)
    T[empty, empty] = 1.0                      # self-loop for unseen states
    T /= T.sum(1, keepdims=True)
    return T


def simulate(T, length, start, rng):
    """Exact run-length simulation of a first-order Markov chain."""
    Ncl = T.shape[0]
    pdiag = np.diag(T).copy()
    off = T.copy()
    off[np.arange(Ncl), np.arange(Ncl)] = 0.0
    rowsum = off.sum(1)
    safe = rowsum > 0
    offcdf = np.zeros_like(off)
    offcdf[safe] = np.cumsum(off[safe] / rowsum[safe, None], axis=1)
    seq = np.empty(length, dtype=np.int32)
    t = 0
    state = int(start)
    while t < length:
        ps = pdiag[state]
        if ps >= 1.0 or not safe[state]:
            seq[t:] = state
            break
        dwell = int(rng.geometric(1.0 - ps))   # frames in this state, mean 1/(1-ps)
        end = min(t + dwell, length)
        seq[t:end] = state
        t = end
        if t >= length:
            break
        state = int(np.searchsorted(offcdf[state], rng.random()))
        if state >= Ncl:
            state = Ncl - 1
    return seq


def dwells_from_seq(seq, chi, fr, M):
    w = max(1, int(DELTA * fr))
    lab = np.argmax(uniform_filter1d(chi[seq], size=w, axis=0, mode='nearest'),
                    axis=1)
    out = [[] for _ in range(M)]
    cur = lab[0]; n = 1
    for a in lab[1:]:
        if a == cur:
            n += 1
        else:
            out[cur].append(n); cur = a; n = 1
    out[cur].append(n)
    return [np.array(o, float) / fr for o in out]


def run_species(cfg):
    seqs = load_states(cfg['states'])
    chi = np.load(cfg['npz'])['chi']
    Ncl = chi.shape[0]
    fr, M = cfg['fr'], cfg['M']
    T = build_T(seqs, Ncl)
    pi = np.zeros(Ncl)
    for s in seqs:
        np.add.at(pi, s, 1.0)
    pi /= pi.sum()
    data = [[] for _ in range(M)]
    surr = [[] for _ in range(M)]
    for s in seqs:
        d = dwells_from_seq(s, chi, fr, M)
        for a in range(M):
            data[a].append(d[a])
        ss = simulate(T, len(s), rng.choice(Ncl, p=pi), rng)
        sd = dwells_from_seq(ss, chi, fr, M)
        for a in range(M):
            surr[a].append(sd[a])
    data = [np.concatenate(x) for x in data]
    surr = [np.concatenate(x) for x in surr]
    return data, surr


def ccdf(d):
    ds = np.sort(d)
    return ds, 1.0 - np.arange(len(ds)) / len(ds)


def frac_beyond(d, thr):
    return float(np.mean(np.asarray(d) >= thr))


print('Building one-step-Markov surrogate nulls...')
fly_data, fly_surr = run_species(FLY)
worm_data, worm_surr = run_species(WORM)

# ---- figure: row 1 = fly CCDFs, row 2 = worm CCDFs (E,F) + tail mass (G) ----
fig = plt.figure(figsize=(13, 6.4))
gs = fig.add_gridspec(2, 4, hspace=0.48, wspace=0.34)

def _letter(ax, s):
    ax.text(-0.22, 1.06, s, transform=ax.transAxes, fontsize=13,
            fontweight='bold', ha='left', va='bottom')

fly_letters = ['A', 'B', 'C', 'D']
for a in range(FLY['M']):
    ax = fig.add_subplot(gs[0, a])
    xs, ys = ccdf(fly_surr[a]); xd, yd = ccdf(fly_data[a])
    ax.loglog(xs, ys, color='0.55', ls='--', lw=1.8, label='one-step Markov')
    ax.loglog(xd, yd, color=ARM_PALETTE[a], lw=2.0, label='data')
    ax.set_title(FLY['arms'][a], fontsize=9)
    ax.set_xlabel(r'residence $\tau$ (s)', fontsize=9)
    if a == 0:
        ax.set_ylabel(r'$P(T \geq \tau)$  (flies)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower left', frameon=False)
    ax.set_ylim(8e-4, 1.3)
    _letter(ax, fly_letters[a])

worm_letters = ['E', 'F']
for a in range(WORM['M']):
    ax = fig.add_subplot(gs[1, a])
    xs, ys = ccdf(worm_surr[a]); xd, yd = ccdf(worm_data[a])
    ax.loglog(xs, ys, color='0.55', ls='--', lw=1.8, label='one-step Markov')
    ax.loglog(xd, yd, color=WORM_COL[a], lw=2.0, label='data')
    ax.set_title(WORM['arms'][a], fontsize=9)
    ax.set_xlabel(r'residence $\tau$ (s)', fontsize=9)
    if a == 0:
        ax.set_ylabel(r'$P(T \geq \tau)$  (worms)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower left', frameon=False)
    ax.set_ylim(8e-4, 1.3)
    _letter(ax, worm_letters[a])

# ---- G: tail mass (% residences > 30 s), data vs surrogate, all six basins ----
axG = fig.add_subplot(gs[1, 2:4])
basins = FLY['arms'] + WORM['arms']
cols   = [ARM_PALETTE[a] for a in range(FLY['M'])] + WORM_COL
data_all = fly_data + worm_data
surr_all = fly_surr + worm_surr
x = np.arange(len(basins)); bw = 0.38
data_pct = [100 * frac_beyond(d, 30) for d in data_all]
surr_pct = [100 * frac_beyond(s, 30) for s in surr_all]
axG.bar(x - bw/2, data_pct, bw, color=cols, edgecolor='0.2', lw=0.4,
        label='data')
axG.bar(x + bw/2, surr_pct, bw, color='0.7', edgecolor='0.2', lw=0.4,
        label='one-step Markov')
axG.set_xticks(x)
axG.set_xticklabels(['Arm 1', 'Arm 2', 'Arm 3', 'Arm 4', 'Pir.', 'Run'],
                    fontsize=8, rotation=20, ha='right')
axG.set_ylabel(r'% residences $>$ 30 s', fontsize=10, fontweight='bold')
axG.legend(fontsize=8, frameon=False, loc='upper right')
_letter(axG, 'G')

# ---- console summary ----
print('\n=== tail mass (% residences > 30 s): data vs surrogate ===')
for b, d, s in zip(basins, data_pct, surr_pct):
    print(f'  {b:22s} data={d:5.1f}%   surrogate={s:5.1f}%')

save(plt.gcf(), 'supp_figure_10')
