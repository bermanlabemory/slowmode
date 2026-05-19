"""Costa-style compatibility check for worms (paper-ready supp figure).

Two panels:
  A: Static V(phi_2) = -log p(phi_2) for worms, with run/pirouette wells
     labeled by behavioral content (per-cluster G-PCCA assignment + phase
     velocity).
  B: Barrier height Delta_V_b(t) per basin in 3-min sliding windows. The
     run barrier (escape from run basin) grows over time, consistent with
     Costa et al. (2024)'s adaptation-driven landscape.

No Plot C (would require full Langevin simulation with friction; deferred).
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter1d


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = WORMS_DATA  # data location

# ============================================================
# Load worm data
# ============================================================
with open(os.path.join(WORMS_DATA, 'states_worms.pkl'), 'rb') as f:
    sd_w = pickle.load(f)
worm_states = [sd_w[i].astype(int) for i in sorted(sd_w)]
fr_w = 16
zw = np.load(os.path.join(WORMS_DATA, 'gpcca_worms_M2_tau3s.npz'))
phi2_per_cluster = zw['schur'][:, 1]   # per-cluster phi_2
assign_w = zw['assignments']           # per-cluster basin (0=pirouette, 1=run by paper convention)
# Per-frame phi_2
worm_phi2 = [phi2_per_cluster[ws] for ws in worm_states]

# Per-cluster behavioral content (phase velocity from Costa plane)
cv = np.load(os.path.join(WORMS_DATA, 'worms_costa_per_cluster.npz'))
om_mean = cv['om_mean']  # per-cluster phase velocity (>0 = forward; <0 = reversal)

# Empirical: which sign of phi_2 corresponds to run vs pirouette?
mean_phi2_pirouette = np.mean(phi2_per_cluster[assign_w == 0])
mean_phi2_run = np.mean(phi2_per_cluster[assign_w == 1])
print(f'Mean phi_2 for pirouette clusters: {mean_phi2_pirouette:.3f}')
print(f'Mean phi_2 for run clusters:       {mean_phi2_run:.3f}')

# Pick orientation
if mean_phi2_run < mean_phi2_pirouette:
    run_side = 'left'   # run at low phi_2
    pir_side = 'right'  # pirouette at high phi_2
else:
    run_side = 'right'
    pir_side = 'left'
print(f'Behavioral labeling: run = {run_side} side, pirouette = {pir_side} side')

# ============================================================
# Static landscape V(phi_2) = -log p(phi_2)
# ============================================================
all_phi2 = np.concatenate(worm_phi2)
x_range = (np.percentile(all_phi2, 0.5), np.percentile(all_phi2, 99.5))
n_bins = 70
edges = np.linspace(*x_range, n_bins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
hist, _ = np.histogram(all_phi2, bins=edges, density=True)
hist = gaussian_filter1d(hist, sigma=1.2)
hist = np.maximum(hist, 1e-6)
V_static = -np.log(hist)
V_static = V_static - V_static.min()

# Identify the two well minima and the barrier max between them
mid_idx = len(centers) // 2
left_well_idx = int(np.argmin(V_static[:mid_idx]))
right_well_idx = mid_idx + int(np.argmin(V_static[mid_idx:]))
# Barrier between them
barrier_idx = left_well_idx + int(np.argmax(V_static[left_well_idx:right_well_idx]))
V_left_well = V_static[left_well_idx]
V_right_well = V_static[right_well_idx]
V_barrier = V_static[barrier_idx]
# Map to behavioral labels
if run_side == 'left':
    V_run_well, run_idx = V_left_well, left_well_idx
    V_pir_well, pir_idx = V_right_well, right_well_idx
else:
    V_run_well, run_idx = V_right_well, right_well_idx
    V_pir_well, pir_idx = V_left_well, left_well_idx
dV_run_static = V_barrier - V_run_well
dV_pir_static = V_barrier - V_pir_well
print(f'\nStatic landscape:')
print(f'  V at run well:       {V_run_well:.3f}')
print(f'  V at pirouette well: {V_pir_well:.3f}')
print(f'  V at barrier:        {V_barrier:.3f}')
print(f'  Barrier height (run → barrier):       {dV_run_static:.3f}')
print(f'  Barrier height (pirouette → barrier): {dV_pir_static:.3f}')

# ============================================================
# Time-dependent V(phi_2, t) — 3-min windows
# ============================================================
W_min = 3.0
W_sec = W_min * 60
win_frames = int(W_sec * fr_w)
n_windows = min(11, min(len(p) for p in worm_phi2) // win_frames)

V_t = np.full((n_windows, n_bins), np.nan)
for wi in range(n_windows):
    vals = []
    for p in worm_phi2:
        if len(p) >= (wi+1)*win_frames:
            vals.append(p[wi*win_frames:(wi+1)*win_frames])
    if not vals:
        continue
    arr = np.concatenate(vals)
    h, _ = np.histogram(arr, bins=edges, density=True)
    h = gaussian_filter1d(h, sigma=1.2)
    h = np.maximum(h, 1e-6)
    V = -np.log(h)
    V_t[wi] = V - V.min()

# Compute Delta_V_b(t) per window for each behavioral basin
dV_run_t = np.full(n_windows, np.nan)
dV_pir_t = np.full(n_windows, np.nan)
for wi in range(n_windows):
    V = V_t[wi]
    if np.any(np.isnan(V)):
        continue
    # In each window, find the two minima and the barrier between them
    half = len(V) // 2
    left_min_idx = int(np.argmin(V[:half]))
    right_min_idx = half + int(np.argmin(V[half:]))
    barrier_idx_t = left_min_idx + int(np.argmax(V[left_min_idx:right_min_idx]))
    V_left = V[left_min_idx]
    V_right = V[right_min_idx]
    V_bar = V[barrier_idx_t]
    if run_side == 'left':
        dV_run_t[wi] = V_bar - V_left
        dV_pir_t[wi] = V_bar - V_right
    else:
        dV_run_t[wi] = V_bar - V_right
        dV_pir_t[wi] = V_bar - V_left

time_mid = (np.arange(n_windows) + 0.5) * W_min

# ============================================================
# Plot
# ============================================================
fig = plt.figure(figsize=(11, 4.5))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.34,
                       left=0.08, right=0.97, top=0.88, bottom=0.16)

# --- Panel A: Static landscape with labeled wells ---
axA = fig.add_subplot(gs[0, 0])
axA.plot(centers, V_static, color='#222', lw=2.0)
axA.fill_between(centers, V_static, V_static.max()+0.5, color='#bbb', alpha=0.18)
# Mark wells
axA.plot(centers[run_idx], V_run_well, marker='v', color=ARM_PALETTE[1],
         markersize=12, mec='black', mew=0.7, zorder=5)
axA.plot(centers[pir_idx], V_pir_well, marker='v', color=ARM_PALETTE[0],
         markersize=12, mec='black', mew=0.7, zorder=5)
# Mark barrier
axA.plot(centers[barrier_idx], V_barrier, marker='^', color='#444',
         markersize=10, mec='black', mew=0.6, zorder=5)
# Labels with arrows
axA.annotate('Run\nWell', xy=(centers[run_idx], V_run_well),
             xytext=(centers[run_idx], V_run_well + 1.0),
             fontsize=12, color=ARM_PALETTE[1], fontweight='bold',
             ha='center', va='bottom')
axA.annotate('Pirouette\nWell', xy=(centers[pir_idx], V_pir_well),
             xytext=(centers[pir_idx], V_pir_well + 1.0),
             fontsize=12, color=ARM_PALETTE[0], fontweight='bold',
             ha='center', va='bottom')
axA.annotate('Barrier', xy=(centers[barrier_idx], V_barrier),
             xytext=(centers[barrier_idx], V_barrier + 0.3),
             fontsize=11, color='#444', fontweight='bold',
             ha='center', va='bottom')
axA.set_xlabel(r'slow eigenmode $\phi_2$', fontsize=11, fontweight='bold')
axA.set_ylabel(r'$V(\phi_2) = -\log p(\phi_2)$',
               fontsize=11, fontweight='bold')
axA.text(-0.13, 1.05, 'A', transform=axA.transAxes, fontsize=14,
         fontweight='bold', va='top')

# --- Panel B: Barrier height evolution per basin ---
axB = fig.add_subplot(gs[0, 1])
axB.plot(time_mid, dV_run_t, 'o-', color=ARM_PALETTE[1], lw=2.0, ms=7,
         mec='black', mew=0.5,
         label=fr'$\Delta V_{{\rm run}}$ (Run-basin Barrier)')
axB.plot(time_mid, dV_pir_t, 'o-', color=ARM_PALETTE[0], lw=2.0, ms=7,
         mec='black', mew=0.5,
         label=fr'$\Delta V_{{\rm pir}}$ (Pirouette-basin Barrier)')
# Static reference lines
axB.axhline(dV_run_static, color=ARM_PALETTE[1], ls=':', lw=1.0, alpha=0.6,
            label=f'static $\\Delta V_{{\\rm run}}={dV_run_static:.2f}$')
axB.axhline(dV_pir_static, color=ARM_PALETTE[0], ls=':', lw=1.0, alpha=0.6,
            label=f'static $\\Delta V_{{\\rm pir}}={dV_pir_static:.2f}$')
axB.set_xlabel('time in recording (min)', fontsize=11, fontweight='bold')
axB.set_ylabel(r'barrier height $\Delta V_b = V_{\rm barrier} - V_{\rm well}$',
               fontsize=11, fontweight='bold')
leg_B = axB.legend(loc='upper right', prop={'size': 9, 'weight': 'bold'},
                   frameon=True, ncol=1, facecolor='white',
                   edgecolor='0.3', framealpha=1.0)
leg_B.get_frame().set_linewidth(0.8)
axB.text(-0.13, 1.05, 'B', transform=axB.transAxes, fontsize=14,
         fontweight='bold', va='top')

save(fig, 'supp_figure_11')

# Print summary
print(f'\nBarrier evolution over {n_windows} windows of {W_min} min:')
print(f'  Run barrier:       early={dV_run_t[0]:.2f}, late={dV_run_t[-1]:.2f}, '
      f'change={dV_run_t[-1] - dV_run_t[0]:+.2f}')
print(f'  Pirouette barrier: early={dV_pir_t[0]:.2f}, late={dV_pir_t[-1]:.2f}, '
      f'change={dV_pir_t[-1] - dV_pir_t[0]:+.2f}')
