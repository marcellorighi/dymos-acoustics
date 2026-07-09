"""
dryden_timeseries.py
=====================

Time-domain Dryden turbulence generator (MIL-SPEC 1797A / MIL-HDBK-1797),
and utilities for robust/stochastic trajectory optimisation.

This module provides:

1. generate_dryden_time_series()
   Pre-generates a fixed gust time history (wu, wv, ww, p_turb, q_turb)
   before the optimisation. The ODE then interpolates into this fixed array
   at the current time 't', decoupling the gust realisation from the
   trajectory geometry (see module docstring for why this matters).

2. DrydenEnsemble
   Generates and stores N independent Dryden realisations (different seeds),
   for use in Sample Average Approximation (SAA) robust optimisation.

3. compute_trajectory_robustness()
   Given a fixed optimal trajectory and N gust realisations, evaluates the
   distribution of the objective (e.g. PA annoyance) over the ensemble.
   This is the core of the SAA approach to robust optimisation.

----------------------------------------------------------------------------
BACKGROUND: WHY TIME-DOMAIN OVER SPATIAL FROZEN FIELD
----------------------------------------------------------------------------
The spatially-varying DrydenField in drone_ode_6dof.py evaluates gusts at
the drone's current position (x,y,z). This couples the gust experienced to
the trajectory taken, so the optimizer can (spuriously) find "good"
trajectories that avoid high-gust regions of the frozen field -- but those
regions are arbitrary artifacts of the random seed, not physical features.

The time-domain approach pre-generates a gust time history before
optimisation. The same gust sequence is experienced regardless of the
drone's position, so the optimiser truly optimises the trajectory *against*
the disturbance rather than *around* it. The trajectory found is then
optimal for that disturbance realisation.

----------------------------------------------------------------------------
MIL-SPEC 1797A TRANSFER FUNCTIONS
----------------------------------------------------------------------------
The Dryden PSDs are rational functions of frequency, implemented as
continuous-time transfer functions and discretised with scipy.signal.

Longitudinal (u):
    H_u(s) = sigma_u * sqrt(2*L_u/(pi*V)) * 1 / (1 + (L_u/V)*s)

Lateral (v) and vertical (w) -- same form, second-order:
    H_w(s) = sigma_w * sqrt(L_w/(pi*V)) * (1 + sqrt(3)*(L_w/V)*s)
                                         / (1 + (L_w/V)*s)^2

Rotational gusts (p_turb from w, MIL-SPEC 1797A App.):
    H_p(s) = sigma_w/V * sqrt(0.8/V) * (pi/(4*b))^(1/6)
             / (sqrt(L_w/V) * (1 + (4*b/(pi*V))*s))
where b = rotor span = 2 * arm_length.

All transfer functions are evaluated at airspeed V (assumed constant for
the purposes of pre-generating the series; for variable-airspeed trajectories
use a slowly time-varying V estimate, or use the mean airspeed).
"""

import numpy as np
from scipy.signal import lti, cont2discrete
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DrydenParams:
    """
    MIL-SPEC 1797A Dryden turbulence parameters.
    Defaults are for LOW ALTITUDE (<300 m AGL), MODERATE turbulence.
    """
    V: float = 5.0          # [m/s] mean airspeed (used to shape the filter)
    # Turbulence intensities [m/s RMS]
    sigma_u: float = 1.5    # longitudinal
    sigma_v: float = 1.5    # lateral
    sigma_w: float = 0.75   # vertical (typically ~sigma_u/2 at low altitude)
    # Length scales [m] (MIL-SPEC low-altitude values)
    L_u: float = 200.0      # longitudinal
    L_v: float = 200.0      # lateral
    L_w: float = 50.0       # vertical (shorter at low altitude)
    # Rotational gust parameters
    arm_length: float = 0.25  # [m] rotor arm
    # Mean wind profile (log-law)
    v_ref: float = 5.0      # [m/s] mean wind at z_ref
    z_ref: float = 20.0     # [m]   reference height
    alpha: float = 0.15     # [-]   wind shear exponent


def _dryden_filter_u(params: DrydenParams, dt: float):
    """First-order Dryden filter for longitudinal gust u."""
    tau = params.L_u / params.V
    gain = params.sigma_u * np.sqrt(2.0 * params.L_u / (np.pi * params.V))
    # H(s) = gain / (tau*s + 1)
    sys_c = lti([gain], [tau, 1.0])
    sys_d = cont2discrete((sys_c.num, sys_c.den), dt, method='bilinear')
    return sys_d[0].ravel(), sys_d[1].ravel()   # (b, a) coefficients


def _dryden_filter_w(params: DrydenParams, dt: float, component='w'):
    """
    Second-order Dryden filter for lateral (v) or vertical (w) gust.
    H(s) = gain * (1 + sqrt(3)*tau*s) / (tau*s + 1)^2
    """
    if component == 'v':
        L, sigma = params.L_v, params.sigma_v
    else:
        L, sigma = params.L_w, params.sigma_w
    tau  = L / params.V
    gain = sigma * np.sqrt(L / (np.pi * params.V))
    # Numerator: gain * [sqrt(3)*tau, 1]
    # Denominator: [tau^2, 2*tau, 1]
    num_c = gain * np.array([np.sqrt(3.0) * tau, 1.0])
    den_c = np.array([tau**2, 2.0 * tau, 1.0])
    sys_d = cont2discrete((num_c, den_c), dt, method='bilinear')
    return sys_d[0].ravel(), sys_d[1].ravel()


def _dryden_filter_p(params: DrydenParams, dt: float):
    """
    First-order rotational gust filter for roll rate p_turb.
    MIL-SPEC 1797A: driven by vertical gust w, scaled by span b.
    H_p(s) = (sigma_w/V) * sqrt(0.8/V) * (pi/(4b))^(1/6)
              / (sqrt(L_w/V) * (1 + (4b/(pi*V))*s))
    """
    b    = 2.0 * params.arm_length
    tau  = 4.0 * b / (np.pi * params.V)
    gain = ((params.sigma_w / params.V)
            * np.sqrt(0.8 / params.V)
            * (np.pi / (4.0 * b)) ** (1.0 / 6.0)
            / np.sqrt(params.L_w / params.V))
    sys_d = cont2discrete(([gain], [tau, 1.0]), dt, method='bilinear')
    return sys_d[0].ravel(), sys_d[1].ravel()


def _apply_filter(b_coef, a_coef, white_noise):
    """Apply a discrete-time IIR filter using direct-form II."""
    from scipy.signal import lfilter
    return lfilter(b_coef, a_coef, white_noise)


def generate_dryden_time_series(t: np.ndarray,
                                 params: Optional[DrydenParams] = None,
                                 seed: int = 42,
                                 altitude: float = 50.0) -> dict:
    """
    Generate a Dryden turbulence time history as a fixed array, suitable
    for injection into a Dymos ODE via time interpolation.

    Parameters
    ----------
    t : 1D array
        Time vector [s], uniformly spaced (e.g. np.arange(0, T, dt)).
        dt = t[1]-t[0] is inferred automatically.
    params : DrydenParams, optional
        Turbulence parameters. Defaults to moderate low-altitude turbulence.
    seed : int
        Random seed for reproducibility. Use different seeds for ensemble
        generation (see DrydenEnsemble).
    altitude : float
        Representative altitude [m] for intensity scaling. Used to scale
        sigma values via a simple ramp: intensity -> 0 near ground.

    Returns
    -------
    dict with keys:
        't'       : time vector [s], same as input
        'wu'      : longitudinal gust [m/s]
        'wv'      : lateral gust [m/s]
        'ww'      : vertical gust [m/s]
        'p_turb'  : roll rate gust [rad/s]
        'q_turb'  : pitch rate gust [rad/s] (same filter as p, independent)
        'mean_wx' : mean wind profile [m/s] (deterministic, altitude-dependent)

    Usage in Dymos ODE
    ------------------
    Store this dict and interpolate inside compute():

        import numpy as np
        # (pre-generated outside the ODE, passed via ode_init_kwargs)
        wu_t = dryden_ts['wu']
        t_ts = dryden_ts['t']

        # Inside compute():
        wind_x = np.interp(t_current, t_ts, wu_t)

    where t_current is the Dymos phase time input 't_phase' (add it as an
    ODE input: self.add_input('t_phase', shape=(nn,), units=None)).
    """
    if params is None:
        params = DrydenParams()

    t = np.asarray(t, dtype=float).ravel()
    n = len(t)
    dt = float(t[1] - t[0])

    # Altitude-dependent intensity scaling
    # (turbulence is weaker near the ground below ~10 m, stronger above)
    z_scale = float(1.0 - np.exp(-altitude / 10.0))

    rng = np.random.default_rng(seed)
    # Pre-roll: generate extra samples at the start to flush filter transients.
    # Without this, the filter output has near-zero variance at t=0 because
    # the filter state initialises to zero (not the stationary distribution).
    # The warm-up length is set to 5x the longest time constant in the filters.
    tau_max = max(params.L_u, params.L_v, params.L_w) / params.V
    n_warmup = max(int(5.0 * tau_max / dt), 500)
    n_total  = n + n_warmup
    # Scale white noise by 1/sqrt(dt): the MIL-SPEC transfer functions assume
    # continuous-time white noise with PSD=1 (units: [unit^2 / Hz]). Discrete
    # samples from standard_normal have variance=1 per sample, not per Hz.
    # Multiplying by 1/sqrt(dt) converts to the correct spectral density.
    white = rng.standard_normal((5, n_total)) / np.sqrt(dt)

    # Build filters
    b_u, a_u = _dryden_filter_u(params, dt)
    b_w, a_w = _dryden_filter_w(params, dt, 'w')
    b_v, a_v = _dryden_filter_w(params, dt, 'v')
    b_p, a_p = _dryden_filter_p(params, dt)

    wu = z_scale * _apply_filter(b_u, a_u, white[0])[n_warmup:]
    wv = z_scale * _apply_filter(b_v, a_v, white[1])[n_warmup:]
    ww = z_scale * _apply_filter(b_w, a_w, white[2])[n_warmup:]
    p_turb = z_scale * _apply_filter(b_p, a_p, white[3])[n_warmup:]
    q_turb = z_scale * _apply_filter(b_p, a_p, white[4])[n_warmup:]

    # Mean wind (deterministic, log-law profile)
    mean_wx = params.v_ref * (max(altitude, 0.1) / params.z_ref) ** params.alpha

    return {
        't':       t,
        'wu':      wu,
        'wv':      wv,
        'ww':      ww,
        'p_turb':  p_turb,
        'q_turb':  q_turb,
        'mean_wx': np.full(n, mean_wx),
    }


class DrydenEnsemble:
    """
    Generate and store N independent Dryden realisations for use in
    Sample Average Approximation (SAA) robust optimisation.

    Each member of the ensemble is a complete gust time history with a
    different random seed, representing a different possible wind realisation
    the drone might encounter.

    Usage
    -----
    ensemble = DrydenEnsemble(t, N=20, params=DrydenParams())
    # Access member i:
    wu_i = ensemble[i]['wu']
    # Pass to ODE for optimisation:
    for i in range(ensemble.N):
        result_i = run_optimisation(dryden_ts=ensemble[i])
        objectives[i] = result_i['J']
    mean_J = np.mean(objectives)
    """

    def __init__(self, t: np.ndarray, N: int = 20,
                 params: Optional[DrydenParams] = None,
                 base_seed: int = 0):
        self.t = t
        self.N = N
        self.params = params or DrydenParams()
        self.members = [
            generate_dryden_time_series(t, params=self.params,
                                        seed=base_seed + i)
            for i in range(N)
        ]

    def __getitem__(self, i):
        return self.members[i]

    def statistics(self, key: str) -> dict:
        """Return mean and std of a gust component across the ensemble."""
        data = np.array([m[key] for m in self.members])
        return {'mean': data.mean(axis=0),
                'std':  data.std(axis=0),
                'p5':   np.percentile(data, 5, axis=0),
                'p95':  np.percentile(data, 95, axis=0)}


def compute_trajectory_robustness(trajectory: dict,
                                   ensemble: DrydenEnsemble,
                                   objective_func,
                                   verbose: bool = True) -> dict:
    """
    Evaluate the robustness of a fixed optimal trajectory by computing the
    objective over all ensemble members. This is the post-optimisation step
    of the SAA approach: given the trajectory found by the optimizer (for
    one wind realisation), how does it perform across all realisations?

    Parameters
    ----------
    trajectory : dict
        Fixed trajectory from Dymos: keys 't', 'x', 'y', 'z', 'vx', 'vy',
        'vz', 'ax', 'ay', 'az' (world-frame accelerations from ODE output).
    ensemble : DrydenEnsemble
        N independent gust realisations.
    objective_func : callable
        objective_func(trajectory, gust_member) -> float.
        Evaluates your acoustic/annoyance objective for the given trajectory
        and wind realisation. Implement using the existing acoustic pipeline:
        estimate_rotor_rpm() -> estimate_received_spl_fine() ->
        compute_zwicker_indicators_windowed() -> PA.
    verbose : bool

    Returns
    -------
    dict:
        'objectives'   : array of N objective values
        'mean'         : mean objective across ensemble
        'std'          : standard deviation (measure of robustness)
        'p95'          : 95th percentile (worst-case risk measure)
        'cvar_90'      : Conditional Value at Risk at 90% (mean of worst 10%)
        'robust_index' : mean + std  (simple Taguchi-style robustness metric)
    """
    objectives = np.zeros(ensemble.N)
    for i in range(ensemble.N):
        objectives[i] = objective_func(trajectory, ensemble[i])
        if verbose:
            print(f"  Ensemble member {i+1}/{ensemble.N}: J = {objectives[i]:.4f}")

    sorted_obj = np.sort(objectives)
    n90 = max(1, int(0.1 * ensemble.N))
    cvar_90 = float(np.mean(sorted_obj[-n90:]))

    return {
        'objectives':   objectives,
        'mean':         float(np.mean(objectives)),
        'std':          float(np.std(objectives)),
        'p95':          float(np.percentile(objectives, 95)),
        'cvar_90':      cvar_90,
        'robust_index': float(np.mean(objectives) + np.std(objectives)),
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Generate a single realisation and plot ---
    dt   = 1.0 / 500.0
    T    = 10.0
    t    = np.arange(0, T, dt)
    ts   = generate_dryden_time_series(t, seed=42, altitude=50.0)

    print("Dryden time series stats:")
    for key in ['wu', 'wv', 'ww', 'p_turb', 'q_turb']:
        print(f"  {key:8s}: mean={ts[key].mean():.4f}, "
              f"std={ts[key].std():.4f}, "
              f"min={ts[key].min():.4f}, max={ts[key].max():.4f}")

    # --- Generate ensemble and show spread ---
    ensemble = DrydenEnsemble(t, N=20, base_seed=0)
    stats = ensemble.statistics('wu')
    print(f"\nEnsemble wu std range: "
          f"{stats['std'].min():.3f} - {stats['std'].max():.3f} m/s")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, ts['wu'], label='wu (longitudinal)')
    axes[0].plot(t, ts['wv'], label='wv (lateral)', alpha=0.7)
    axes[0].set_ylabel('Gust [m/s]')
    axes[0].legend()

    axes[1].plot(t, ts['ww'], label='ww (vertical)', color='green')
    axes[1].set_ylabel('Vertical gust [m/s]')
    axes[1].legend()

    axes[2].plot(t, ts['p_turb'], label='p_turb (roll)', color='red')
    axes[2].plot(t, ts['q_turb'], label='q_turb (pitch)', color='orange',
                 alpha=0.7)
    axes[2].set_ylabel('Angular gust [rad/s]')
    axes[2].set_xlabel('Time [s]')
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    
    # plt.savefig('/mnt/user-data/outputs/dryden_timeseries_example.png', dpi=120)
    # print("\nPlot saved.")