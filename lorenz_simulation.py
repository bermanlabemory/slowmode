"""Stochastically driven Lorenz simulator.

Generates the dataset analyzed in Fig 2 of Kaur, Jain, and Berman (2026).  The Lorenz `sigma` parameter is logistically modulated by a
slow hidden driver `h(t)`, which is itself sampled by Metropolis dynamics in
a symmetric double-well potential U(h) = h^4 - 8 h^2.  The two minima at
h = +-2 correspond to two metastable wings of the Lorenz attractor; the
inverse temperature beta sets the barrier-crossing rate (and therefore the
mean dwell time per well).

Equations (Methods Sec. "Modified Lorenz system"):

    sigma(t) = sigma_0 [ 1 + gamma / (1 + exp(-h(t))) ]
    dx/dt    = sigma(t) (y - x)
    dy/dt    = (rho - z) x - y
    dz/dt    = x y - beta_L z

with sigma_0 = 8, rho = 28, beta_L = 8/3 (the standard Lorenz parameters)
and gamma = 1.  The driver is sampled at the same rate as the Lorenz
trajectory (frame rate fs = 100 Hz by default).

Usage::

    from lorenz_simulation import simulate

    t, xyz, h = simulate(beta=0.5, T=2000.0, discard=200.0, seed=0)
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def double_well_potential(h, b=1.0, c=8.0):
    """U(h) = b h^4 - c h^2.  Default minima at h = +- sqrt(c/(2b)) = +-2."""
    return b * h ** 4 - c * h ** 2


def metropolis_double_well(N, beta, step=1.0, b=1.0, c=8.0, x_init=None,
                           seed=None):
    """Sample N steps of the double-well driver h(t) by Metropolis dynamics.

    Returns
    -------
    h : (N,) ndarray of floats.
    """
    rng = np.random.default_rng(seed)
    x = x_init if x_init is not None else (rng.random() * 6 - 3)
    E = double_well_potential(x, b, c)
    out = np.empty(N)
    for n in range(N):
        dx = (rng.random() * 2 - 1) * step
        x_new = x + dx
        E_new = double_well_potential(x_new, b, c)
        if rng.random() < np.exp(-beta * (E_new - E)):
            x = x_new; E = E_new
        out[n] = x
    return out


def simulate(beta, T=2000.0, discard=200.0, fs=100.0, gamma=1.0,
             sigma0=8.0, rho=28.0, beta_lorenz=8.0 / 3.0, seed=None,
             x0=(-8.0, -8.0, 27.0), rtol=1e-8, atol=1e-8):
    """Simulate the driven Lorenz system at one beta value.

    Parameters
    ----------
    beta : float
        Inverse temperature for the double-well driver; controls dwell time.
    T : float
        Total simulation time in seconds (incl. discarded transient).
    discard : float
        Initial transient discarded from the returned series (in seconds).
    fs : float
        Sampling frequency in Hz (also the driver sampling rate).
    gamma : float
        Logistic-coupling amplitude for the sigma modulation.

    Returns
    -------
    t : (N,) ndarray
        Time stamps in seconds (after discard).
    xyz : (N, 3) ndarray
        Lorenz trajectory (x, y, z).
    h : (N,) ndarray
        Hidden driver (after discard); its sign labels the metastable basin.
    """
    dt = 1.0 / fs
    t_full = np.arange(0.0, T, dt)
    n_full = len(t_full)
    n_discard = int(discard * fs)

    h_full = metropolis_double_well(n_full, beta=beta, seed=seed)

    def rhs(t, state, sigma0, rho, beta_lorenz, gamma, drive, fs):
        x, y, z = state
        idx = int(np.clip(t * fs, 0, len(drive) - 1))
        dr = gamma / (1.0 + np.exp(-drive[idx]))
        sigma_t = sigma0 * (1.0 + dr)
        return [sigma_t * (y - x),
                (rho - z) * x - y,
                x * y - beta_lorenz * z]

    sol = solve_ivp(rhs, (0.0, T), list(x0), t_eval=t_full,
                    args=(sigma0, rho, beta_lorenz, gamma, h_full, fs),
                    rtol=rtol, atol=atol)
    xyz = sol.y.T
    return (t_full[n_discard:],
            xyz[n_discard:],
            h_full[n_discard:])


def mean_dwell_time(h, fs):
    """Mean dwell time (in seconds) per well for the driver `h(t)`.

    A "dwell" is a contiguous run of the same sign of h.  Useful for
    parameterising Fig 2D as Pearson |r| vs mean dwell time across beta.
    """
    sgn = np.sign(h)
    sgn[sgn == 0] = 1
    transitions = np.where(np.diff(sgn) != 0)[0]
    if transitions.size < 2:
        return np.inf
    durations = np.diff(transitions) / fs
    return float(durations.mean())
