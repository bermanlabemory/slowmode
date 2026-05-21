"""Supp. Fig. S9 (flies): reproducibility / individuality of the slow modes.

Two panels:
  A. Per-fly G-PCCA refit arm-direction cosines vs the pooled arms,
     occupancy-filtered.
  B. Per-fly arm occupancy, stacked.

(A subspace-alignment panel was dropped: at M=4 the chance alignment of two
random 2D subspaces in 4D ambient space is itself substantial, so the test
cannot reliably distinguish the observed alignment from a permutation null.)

Companion to Kaur, Jain, & Berman (2026).
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = FLIES_DATA  # data location
STATES = os.path.join(FLIES_DATA, 'states_flies.pkl')
fr = 100; M = 4

# Behavioral names per arm (match Fig 5 panel A).
BEHAVIOR_LABELS_LEGEND = [
    'Idle & Slow',
    'Anterior Movements',
    'Posterior & Wing Movements',
    'Locomotion',
]

# ---- Caches ----
refit = np.load(os.path.join(OUT, 'per_fly_pcca_refit_tau2s.npz'))
per_fly_arm_cos = refit['pf_arm_cos']

with open(STATES, 'rb') as f:
    states_dict = pickle.load(f)
fly_states = [states_dict[i].astype(int) for i in sorted(states_dict)]
N_flies = len(fly_states)

z_pcca = np.load(os.path.join(OUT, 'gpcca_flies_M4_tau2s.npz'))
assignments = z_pcca['assignments']
fly_arm_seq = [assignments[fs] for fs in fly_states]
occ_per_fly = np.zeros((N_flies, M))
for f, s in enumerate(fly_arm_seq):
    for k in range(M):
        occ_per_fly[f, k] = (s == k).mean()
MIN_OCC = 0.02
per_fly_cos_filt = np.where(occ_per_fly >= MIN_OCC, per_fly_arm_cos, np.nan)

# ==================================================================
fig = plt.figure(figsize=(11.0, 4.0))
gs  = GridSpec(1, 3, figure=fig, wspace=0.45,
               left=0.07, right=0.86, top=0.88, bottom=0.18)

# ---- A: per-fly refit cosines ----
axA = fig.add_subplot(gs[0, 0])
x_off = np.arange(M); jitter_rng = np.random.RandomState(1)
for j in range(M):
    c = per_fly_cos_filt[:, j]; c = c[np.isfinite(c)]
    jitter = jitter_rng.uniform(-0.15, 0.15, len(c))
    axA.scatter(x_off[j] + jitter, c, c=ARM_PALETTE[j], s=20,
                alpha=0.75, edgecolors='none')
    axA.hlines(np.median(c), x_off[j] - 0.3, x_off[j] + 0.3,
               color='k', lw=1.8, zorder=10)
axA.axhline(1, color='0.5', lw=0.6, ls=':')
axA.set_xticks(x_off)
axA.set_xticklabels(BEHAVIOR_LABELS_LEGEND, fontsize=9, fontweight='bold',
                    rotation=30, ha='right')
axA.set_ylabel('cosine sim.\nper-fly arm vs pooled',
               fontsize=11, fontweight='bold')
axA.set_ylim(-0.05, 1.05)
axA.tick_params(labelsize=8)
axA.text(-0.22, 1.04, 'A', transform=axA.transAxes,
         fontsize=15, fontweight='bold', ha='left', va='bottom')

# ---- B: per-fly arm occupancy stacks (spans cols 1-2) ----
axB = fig.add_subplot(gs[0, 1:])
order = np.argsort(occ_per_fly[:, 0])
bottom = np.zeros(N_flies)
for j in range(M):
    axB.bar(np.arange(N_flies), occ_per_fly[order, j],
            bottom=bottom, color=ARM_PALETTE[j], edgecolor='none',
            width=1.0, align='center', label=BEHAVIOR_LABELS_LEGEND[j])
    bottom += occ_per_fly[order, j]
axB.set_xlabel('fly (sorted by Idle & Slow occupancy)',
               fontsize=11, fontweight='bold')
axB.set_ylabel('arm occupancy fraction', fontsize=11, fontweight='bold')
axB.set_xlim(-0.5, N_flies - 0.5); axB.set_ylim(0, 1)
axB.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
           prop={'size': 10, 'weight': 'bold'}, handlelength=1.2)
axB.tick_params(labelsize=8)
axB.text(-0.10, 1.04, 'B', transform=axB.transAxes,
         fontsize=15, fontweight='bold', ha='left', va='bottom')

save(plt.gcf(), 'supp_figure_9')
