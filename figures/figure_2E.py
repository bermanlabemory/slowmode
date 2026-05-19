"""Figure 2E: |Pearson r| between phi_2 and the hidden driver h(t)
as a function of mean dwell time, averaged across 5 random seeds of the
kmeans clustering, with SEM error bars.  Multi-timescale and
fixed-timescale pipelines shown side-by-side.
"""
import os, sys, numpy as np
import matplotlib.pyplot as plt


from _paths import *  # setup() + ARM_PALETTE + *_DATA + save

OUT = LORENZ_DATA  # data location
d = np.load(os.path.join(OUT, 'lorenz_corr_seeds.npz'))
betas = d['betas']
dwell = d['dwell_mean']
order = np.argsort(dwell)
dwell = dwell[order]

def stats(arr2d):
    """arr2d shape (n_seeds, n_betas) -> mean & SEM of |arr| per beta."""
    A = np.abs(arr2d)[:, order]
    return A.mean(axis=0), A.std(axis=0, ddof=1) / np.sqrt(A.shape[0])

m_mt, s_mt = stats(d['corr_multi'])      # Pearson, multi
m_ft, s_ft = stats(d['corr_fixed'])      # Pearson, fixed
rm_mt, rs_mt = stats(d['rho_multi'])     # Spearman, multi (for printout)
rm_ft, rs_ft = stats(d['rho_fixed'])     # Spearman, fixed (for printout)
n_seeds = d['corr_multi'].shape[0]

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.errorbar(dwell, m_mt, yerr=s_mt, fmt='o-', color='0.15', ms=5, lw=1.4,
            capsize=3, elinewidth=0.9, label='multi-timescale')
ax.errorbar(dwell, m_ft, yerr=s_ft, fmt='s--', color='crimson',
            ms=4, lw=1.1, capsize=3, elinewidth=0.9, label='fixed-timescale')
ax.set_xscale('log')
ax.set_xlabel(r'mean dwell time $\langle \tau \rangle$ (s)', fontsize=9)
ax.set_ylabel(r'$|r|$ (mean $\pm$ SEM)', fontsize=9)
ax.set_ylim(-0.05, 1.05)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8, frameon=False, loc='center right')
plt.tight_layout()
save(fig, 'figure_2E')
plt.close(fig)

# Numerical summary
print(f'\nN seeds = {n_seeds}\n')
print(' β       <τ> (s)     |r|_multi      |ρ|_multi      |r|_fixed      |ρ|_fixed')
for b, t, m1, s1, m2, s2, m3, s3, m4, s4 in zip(
        betas[order], dwell, m_mt, s_mt, rm_mt, rs_mt, m_ft, s_ft, rm_ft, rs_ft):
    print(f'  {b:.2f}   {t:8.1f}    '
          f'{m1:.3f}±{s1:.3f}   {m2:.3f}±{s2:.3f}   '
          f'{m3:.3f}±{s3:.3f}   {m4:.3f}±{s4:.3f}')
