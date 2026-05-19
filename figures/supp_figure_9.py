"""Spectral / operator-derived definition of the slow-mode fluctuation
strength σ_slow, replacing the windowed-variance proxy used in the main
analysis.

Definition.  At the working lag τ = 2 s with M = 4, the slowest *intra-
basin* relaxation mode of the multi-timescale operator is the eigenvalue
just past the spectral gap, |λ_5| ≈ 0.60.  Its implied timescale is
t_5 = -τ / log|λ_5| ≈ 3.9 s, corresponding to a cutoff frequency
f_cut = 1/(2π t_5) ≈ 0.041 Hz.  We define
    σ_slow,jk = √( (2/T_j) Σ_{f ≤ f_cut} |X_k^{(j)}(f)|² )
where X_k^{(j)}(f) is the FFT of the soft membership χ_k(t) for fly j
restricted to its recording length T_j, and the sum is taken over
positive frequencies below f_cut. This quantifies the variance of χ_k(t)
contributed only by modes slower than the operator's intra-basin
relaxation rate, which is what the Costa T_s is meant to represent.

Then re-test the per-fly Costa correlation (α vs σ_slow) at the
spectral definition and compare with the W = 60 s descriptive proxy.

Output: costa_spectral_sigma_tau2s.npz
        fig_costa_spectral_sigma.png
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr


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

# ---- Determine spectral cutoff from operator ----
gp = np.load(os.path.join(OUT, 'gpcca_flies_M4_tau2s.npz'))
eigvals_abs = gp['eigvals']                      # [|λ_2|, |λ_3|, |λ_4|, |λ_5|]
tau_s = float(gp['tau']) / fr                    # 2.0
# eigvals[3] is |λ_5| because eigvals starts at |λ_2| (idx 0), so |λ_5| at idx 3
lam5 = float(eigvals_abs[3])
t5 = -tau_s / np.log(lam5)                       # implied timescale (s)
f_cut = 1.0 / (2 * np.pi * t5)                   # cutoff frequency (Hz)
print(f'M = {M}, |λ_5| = {lam5:.3f}, intra-basin relaxation t_5 = {t5:.2f} s, '
      f'spectral cutoff f_cut = {f_cut:.4f} Hz')

# ---- Load per-fly chi sequences ----
with open(STATES, 'rb') as f:
    states_dict = pickle.load(f)
fly_states = [states_dict[i].astype(int) for i in sorted(states_dict)]
N_flies = len(fly_states)
chi = gp['chi']                                  # (N_clusters, M)
fly_chi = [chi[fs, :].astype(float) for fs in fly_states]


def sigma_slow_spectral(chi_t, fs, f_cut):
    """Integrated low-frequency power of χ_k(t) for f ≤ f_cut, returned
    as a standard-deviation-like number per arm."""
    n = chi_t.shape[0]
    if n < 4: return np.full(chi_t.shape[1], np.nan)
    # Detrend per-arm by subtracting the mean (DC removal)
    x = chi_t - chi_t.mean(axis=0, keepdims=True)
    # FFT and one-sided power
    X = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    P = (np.abs(X)**2) / (fs * n)                # PSD normalization
    # Sum over positive frequencies below f_cut (skip DC at index 0)
    mask = (freqs > 0) & (freqs <= f_cut)
    if mask.sum() < 1:
        return np.full(chi_t.shape[1], np.nan)
    var_lf = 2.0 * P[mask].sum(axis=0) * (freqs[1] - freqs[0])  # one-sided × df
    return np.sqrt(np.maximum(var_lf, 0.0))


# ---- Compute σ_slow_spectral per fly per arm ----
sigma_spec = np.full((N_flies, M), np.nan)
for f in range(N_flies):
    sigma_spec[f] = sigma_slow_spectral(fly_chi[f], fr, f_cut)

# ---- Per-fly α from existing cache ----
refit = np.load(os.path.join(OUT, 'per_fly_pcca_refit_tau2s.npz'))
per_fly_alpha = refit['per_fly_alpha']
per_fly_sigma_W60 = refit['per_fly_sigma']        # the W=60s proxy

# ---- Per-arm and pooled Pearson correlations ----
print('\n=== Spectral σ_slow vs windowed σ_slow (W=60s) ===')
print(f'{"arm":>4} {"n":>4} {"r_spec":>9} {"p_spec":>10}  {"r_W60":>9} {"p_W60":>10}')
r_spec_arm = np.zeros(M); p_spec_arm = np.zeros(M)
r_W60_arm  = np.zeros(M); p_W60_arm  = np.zeros(M)
for j in range(M):
    a = per_fly_alpha[:, j]
    s_sp = sigma_spec[:, j]
    s_w  = per_fly_sigma_W60[:, j]
    ok = np.isfinite(a) & np.isfinite(s_sp) & np.isfinite(s_w)
    if ok.sum() >= 4:
        r_spec_arm[j], p_spec_arm[j] = pearsonr(s_sp[ok], a[ok])
        r_W60_arm[j],  p_W60_arm[j]  = pearsonr(s_w[ok], a[ok])
    else:
        r_spec_arm[j] = p_spec_arm[j] = r_W60_arm[j] = p_W60_arm[j] = np.nan
    print(f'{j+1:>4d} {int(ok.sum()):>4d}  {r_spec_arm[j]:>+9.3f}  {p_spec_arm[j]:>10.4f}  '
          f'{r_W60_arm[j]:>+9.3f}  {p_W60_arm[j]:>10.4f}')

# Pooled
all_a = per_fly_alpha.ravel(); all_s_sp = sigma_spec.ravel()
all_s_w = per_fly_sigma_W60.ravel()
ok = np.isfinite(all_a) & np.isfinite(all_s_sp) & np.isfinite(all_s_w)
r_pool_spec, p_pool_spec = pearsonr(all_s_sp[ok], all_a[ok])
r_pool_W60,  p_pool_W60  = pearsonr(all_s_w[ok],  all_a[ok])
print(f'pool {int(ok.sum()):>4d}  {r_pool_spec:>+9.3f}  {p_pool_spec:>10.4f}  '
      f'{r_pool_W60:>+9.3f}  {p_pool_W60:>10.4f}')

# Cross-definition correlation: do the two σ_slow definitions agree?
cross_r, cross_p = pearsonr(all_s_sp[ok], all_s_w[ok])
print(f'\nspectral vs windowed σ_slow agreement: r = {cross_r:+.3f}  '
      f'(p = {cross_p:.2e})')

# ---- Save ----
np.savez(os.path.join(OUT, 'costa_spectral_sigma_tau2s.npz'),
         f_cut=f_cut, t5=t5, lam5=lam5,
         sigma_spec=sigma_spec, per_fly_alpha=per_fly_alpha,
         per_fly_sigma_W60=per_fly_sigma_W60,
         r_spec_arm=r_spec_arm, p_spec_arm=p_spec_arm,
         r_W60_arm=r_W60_arm,  p_W60_arm=p_W60_arm,
         r_pool_spec=r_pool_spec, p_pool_spec=p_pool_spec,
         r_pool_W60=r_pool_W60,  p_pool_W60=p_pool_W60,
         cross_r=cross_r, cross_p=cross_p)

# ---- Plot: 2x3 layout — top row spectral scatters per arm, bottom: comparisons ----
fig = plt.figure(figsize=(15, 9))
gs = GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.45)

# Top row: per-arm scatter under SPECTRAL σ_slow
for j in range(M):
    ax = fig.add_subplot(gs[0, j])
    a = per_fly_alpha[:, j]; s = sigma_spec[:, j]
    ok = np.isfinite(a) & np.isfinite(s)
    ax.scatter(s[ok], a[ok], c=ARM_PALETTE[j], s=22, alpha=0.85,
               edgecolors='none')
    if ok.sum() >= 4:
        m_, b_ = np.polyfit(s[ok], a[ok], 1)
        xs = np.linspace(s[ok].min()*0.9, s[ok].max()*1.1, 50)
        ax.plot(xs, m_ * xs + b_, color='0.15', lw=1.0)
    ax.set_xlabel(r'$\sigma_{\mathrm{slow}}$ (spectral)',
                  fontsize=10, fontweight='bold')
    ax.set_ylabel(r'$\alpha$' if j == 0 else '',
                  fontsize=10, fontweight='bold')
    ax.set_title(ARM_TITLES[j], color=ARM_PALETTE[j], fontsize=10,
                 fontweight='bold')
    ax.tick_params(labelsize=7)
    if j == 0:
        ax.text(-0.30, 1.10, 'A', transform=ax.transAxes, fontsize=15,
                fontweight='bold', ha='left', va='bottom')

# Bottom row: r-comparison bars; cross-definition agreement
ax = fig.add_subplot(gs[1, 0:2])
x = np.arange(M); w = 0.38
ax.bar(x - w/2, r_W60_arm,  width=w, color='0.55', label=r'$\sigma_{\mathrm{slow}}$ (W=60s)')
ax.bar(x + w/2, r_spec_arm, width=w, color='0.15',
       label=r'$\sigma_{\mathrm{slow}}$ (spectral, $f \leq$' f' {f_cut:.3f} Hz)')
ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f'Arm {j+1}' for j in range(M)], fontweight='bold')
ax.set_ylabel(r'Pearson $r(\sigma_{\mathrm{slow}}, \alpha)$',
              fontsize=11, fontweight='bold')
ax.legend(prop={'size': 10, 'weight': 'bold'}, loc='lower left')
ax.text(-0.10, 1.04, 'B', transform=ax.transAxes, fontsize=15,
        fontweight='bold', ha='left', va='bottom')

# Cross-definition scatter
ax = fig.add_subplot(gs[1, 2:4])
for j in range(M):
    s_sp = sigma_spec[:, j]; s_w = per_fly_sigma_W60[:, j]
    ok = np.isfinite(s_sp) & np.isfinite(s_w)
    ax.scatter(s_w[ok], s_sp[ok], c=ARM_PALETTE[j], s=20, alpha=0.85,
               edgecolors='none', label=ARM_TITLES[j])
ax.set_xlabel(r'$\sigma_{\mathrm{slow}}$ (W=60s)',
              fontsize=11, fontweight='bold')
ax.set_ylabel(r'$\sigma_{\mathrm{slow}}$ (spectral)',
              fontsize=11, fontweight='bold')
ax.legend(prop={'size': 9, 'weight': 'bold'}, loc='upper left')
ax.text(-0.10, 1.04, 'C', transform=ax.transAxes, fontsize=15,
        fontweight='bold', ha='left', va='bottom')

save(plt.gcf(), 'supp_figure_9')
print('\nSaved fig_costa_spectral_sigma.{png,pdf} and costa_spectral_sigma_tau2s.npz')
