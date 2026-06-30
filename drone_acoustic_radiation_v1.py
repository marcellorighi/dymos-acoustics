"""
drone_acoustic_radiation.py
=============================

Extend the rotor-RPM pipeline (rotor_rpm_estimation.py) with a simple,
literature-grounded link between RPM and radiated acoustic power, plus a
directivity pattern, to estimate the received sound pressure level (SPL) at
a fixed observer location as the drone flies a trajectory.

----------------------------------------------------------------------------
Modeling chain and assumptions (all exposed as parameters in AcousticParams)
----------------------------------------------------------------------------
1. RPM -> acoustic power (per rotor):
       P_acoustic = P_ref * (RPM / RPM_ref) ** n
   LITERATURE BASIS: rotor/propeller noise power is widely approximated as
   scaling with roughly the 5th-6th power of rotational (or tip) speed in
   the tip-Mach regimes typical of small rotors/drones -- this follows from
   classical dipole loading-noise scaling (Gutin theory) and is a common
   engineering rule of thumb in rotor acoustics (e.g. "doubling RPM adds
   ~15-18 dB"). Default here is n = 5 (~15 dB per doubling); treat this as
   an approximate, tunable exponent, not a precise universal constant --
   actual values depend on blade design, loading, and Mach regime, and
   ideally should be calibrated against measured data for your propeller.

2. Calibration: P_ref (the absolute scale of acoustic power) is NOT
   knowable from RPM scaling alone -- it must be anchored to a real
   measurement. `calibrate_p_ref()` below lets you supply a known
   reference SPL (e.g. from a datasheet, "62 dBA at 1 m, hover") and backs
   out the P_ref that reproduces it, given the RPM, distance, and angle of
   that reference measurement.

3. Directivity: rotor loading noise is approximately dipole-like, radiating
   most strongly in the plane of the rotor disk and least along the thrust
   (rotation) axis:
       D(theta) = 1.5 * sin(theta)^2
   where theta is the polar angle measured from the rotor's thrust axis
   (theta = 90 deg: in-plane, maximum; theta = 0/180 deg: on-axis, zero).
   The factor 1.5 normalizes D so its average over the full sphere is 1,
   i.e. integrating the radiated intensity over all directions reproduces
   exactly P_acoustic (no artificial gain/loss from the directivity shape).
   This is a simplified, single-lobe approximation of real propeller
   directivity (real patterns have additional lobes depending on blade
   number / harmonic order), adequate for first-order, broadband-level
   estimates.

4. Geometry / propagation:
   - The rotor thrust axis is assumed VERTICAL (consistent with the
     no-attitude assumption used in rotor_rpm_estimation.py: the drone is
     treated as if it never tilts).
   - All 4 rotors are treated as co-located at the drone's CG position for
     propagation purposes (valid once observer distance >> rotor spacing).
   - Free-field spherical spreading is assumed: no ground reflection,
     atmospheric absorption, or near-field effects.
   - The 4 rotors are summed as INCOHERENT power sources (powers add, not
     pressures) -- standard engineering simplification for broadband/
     overall levels; it does not capture phase interference between blade
     tones.

5. SPL is in dB re 2e-5 Pa (or equivalently dB re 1e-12 W/m^2 reference
   intensity), via the standard free-field point-source relation:
       SPL(r, theta) = SWL + 10*log10(D(theta)) - 20*log10(r) - 11
   where SWL = 10*log10(P_acoustic_total / 1e-12) is the sound power level
   of the combined 4-rotor source, in dB re 1e-12 W.
"""

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt


@dataclass
class AcousticParams:
    """RPM-to-acoustic-power and directivity assumptions -- see module docstring."""

    rpm_ref: float = 5000.0     # [RPM]  reference rotational speed for the power law
    p_ref: float = 0.01         # [W]    acoustic power of ONE rotor at rpm_ref (CALIBRATE THIS)
    n_exponent: float = 5.0     # [-]    RPM exponent in P ~ RPM^n (literature: ~5-6)
    p_ref_min: float = 1e-9     # [W]    floor to avoid zero/negative power for tiny RPM


def acoustic_power_per_rotor(rpm: np.ndarray, params: AcousticParams) -> np.ndarray:
    """Acoustic power [W] radiated by a single rotor, given its RPM time series."""
    rpm = np.clip(np.asarray(rpm, dtype=float), 0.0, None)
    P = params.p_ref * (rpm / params.rpm_ref) ** params.n_exponent
    return np.clip(P, params.p_ref_min, None)


def directivity_factor(theta_rad: np.ndarray) -> np.ndarray:
    """
    Dipole-like directivity factor D(theta), normalized so its average over
    the full sphere is 1 (sphere-integrated power = P_acoustic, unaffected
    by directivity shape). theta is measured from the rotor thrust axis.
    """
    return 1.5 * np.sin(theta_rad) ** 2


def observer_geometry(x, y, z, observer_xyz, z_up: bool = True):
    """
    Compute, at each time step, the distance r and the polar angle theta
    (from the vertical rotor axis) between the drone position and a fixed
    observer location.

    Parameters
    ----------
    x, y, z : array-like
        Drone position time series [m].
    observer_xyz : tuple of 3 floats
        Fixed observer location (x_obs, y_obs, z_obs) [m].
    z_up : bool
        Must match the convention used upstream (rotor_rpm_estimation.py):
        True for z-up (ENU-like), False for z-down (NED-like). Only affects
        which sign convention "vertical" (the assumed rotor axis) uses --
        the rotor axis is always treated as aligned with the world z-axis.

    Returns
    -------
    r : ndarray
        Distance from drone to observer at each time [m].
    theta : ndarray
        Polar angle from the rotor (vertical) axis to the observer
        direction, in radians, in [0, pi].
    """
    x, y, z = (np.asarray(v, dtype=float).ravel() for v in (x, y, z))
    x_obs, y_obs, z_obs = observer_xyz

    dx = x_obs - x
    dy = y_obs - y
    dz = z_obs - z
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.clip(r, 1e-3, None)  # avoid singularity if observer coincides with drone

    # Angle from the vertical axis: cos(theta) = |dz_world_up| / r.
    # "world up" is +z if z_up else -z; using the same vertical component
    # either way since directivity is symmetric about the rotor plane.
    cos_theta = np.clip(np.abs(dz) / r, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return r, theta


def estimate_received_spl(t, x, y, z,
                           rpm_front, rpm_rear, rpm_right, rpm_left,
                           observer_xyz,
                           params: AcousticParams = None,
                           z_up: bool = True) -> dict:
    """
    Combine 4 rotor RPM time series and drone position into a received SPL
    time series at a fixed observer location.

    Parameters
    ----------
    t, x, y, z : array-like
        Time and drone position time series.
    rpm_front, rpm_rear, rpm_right, rpm_left : array-like
        Per-rotor RPM time series, e.g. from estimate_rotor_rpm().
    observer_xyz : tuple of 3 floats
        Fixed observer location (x_obs, y_obs, z_obs) [m].
    params : AcousticParams, optional
        RPM-to-power and directivity assumptions (see module docstring).
    z_up : bool
        Must match the convention used in rotor_rpm_estimation.py.

    Returns
    -------
    dict with keys:
        't'            : time vector [s]
        'spl_db'       : received overall SPL time series [dB]
        'swl_db'       : combined 4-rotor sound power level [dB re 1e-12 W]
        'r'            : drone-to-observer distance [m]
        'theta_deg'    : polar angle from rotor axis to observer [deg]
        'p_total'      : combined 4-rotor acoustic power [W]
    """
    if params is None:
        params = AcousticParams()

    t = np.asarray(t, dtype=float).ravel()
    rpm_front, rpm_rear, rpm_right, rpm_left = (
        np.asarray(v, dtype=float).ravel() for v in (rpm_front, rpm_rear, rpm_right, rpm_left)
    )

    # --- 1. RPM -> acoustic power, per rotor, summed incoherently ---
    p_total = (
        acoustic_power_per_rotor(rpm_front, params)
        + acoustic_power_per_rotor(rpm_rear, params)
        + acoustic_power_per_rotor(rpm_right, params)
        + acoustic_power_per_rotor(rpm_left, params)
    )
    swl_db = 10.0 * np.log10(p_total / 1e-12)

    # --- 2. geometry: distance and angle from rotor axis to observer ---
    r, theta = observer_geometry(x, y, z, observer_xyz, z_up=z_up)

    # --- 3. directivity-weighted free-field propagation ---
    D = directivity_factor(theta)
    spl_db = swl_db + 10.0 * np.log10(D) - 20.0 * np.log10(r) - 11.0

    return {
        "t": t,
        "spl_db": spl_db,
        "swl_db": swl_db,
        "r": r,
        "theta_deg": np.degrees(theta),
        "p_total": p_total,
    }


def calibrate_p_ref(spl_ref_db: float, rpm_ref_measurement: float, r_ref: float,
                     theta_ref_deg: float, n_rotors_in_measurement: int = 1,
                     n_exponent: float = 5.0) -> float:
    """
    Back out p_ref (acoustic power of ONE rotor at rpm_ref) from a known
    reference measurement, e.g. a datasheet value like "62 dBA at 1 m,
    hover, all 4 rotors at 5000 RPM".

    Parameters
    ----------
    spl_ref_db : float
        Measured/reported SPL at the reference condition [dB].
    rpm_ref_measurement : float
        RPM at which that reference measurement was taken. This also
        becomes AcousticParams.rpm_ref if you use the returned p_ref
        directly with the default rpm_ref.
    r_ref : float
        Distance from drone/rotor to the microphone for that measurement [m].
    theta_ref_deg : float
        Angle from the rotor axis to the microphone for that measurement
        [deg]. If unknown, 90 deg (in-plane) is a common measurement
        convention for propeller noise testing.
    n_rotors_in_measurement : int
        Number of rotors active during the reference measurement (4 for a
        whole-drone hover measurement, 1 for an isolated single-rotor test).
    n_exponent : float
        Same RPM exponent you intend to use in AcousticParams.

    Returns
    -------
    p_ref : float
        Acoustic power [W] of a single rotor at rpm_ref_measurement,
        consistent with the supplied measurement and exponent. Use this as
        AcousticParams.p_ref together with
        AcousticParams.rpm_ref = rpm_ref_measurement.
    """
    theta_ref = np.radians(theta_ref_deg)
    D_ref = directivity_factor(theta_ref)

    # invert SPL = SWL + 10log10(D) - 20log10(r) - 11
    swl_ref_db = spl_ref_db - 10.0 * np.log10(D_ref) + 20.0 * np.log10(r_ref) + 11.0
    p_total_ref = 1e-12 * 10.0 ** (swl_ref_db / 10.0)
    p_ref_single_rotor = p_total_ref / n_rotors_in_measurement
    return p_ref_single_rotor


@dataclass
class FineGridParams:
    """
    Assumptions for upsampling the coarse Dymos time history to a fine time
    grid suitable for acoustic/psychoacoustic post-processing.
    """

    fs: float = 48000.0           # [Hz] fine-grid sample rate.
    # Default matches what the MOSQITO/Zwicker pipeline (zwicker_annoyance.py)
    # expects, so the two stages of the pipeline share one sample rate.
    interp_method: str = "cubic"   # "cubic" (smooth, default) or "linear"
    use_integrated_phase: bool = True
    # True (recommended): rotor phase = 2*pi * integral(RPM(t)/60 dt), exact
    # for time-varying RPM.
    # False: literal phase = 2*pi*(RPM(t)/60)*t, only exact if RPM is
    # constant -- provided for direct comparison/diagnostics if you want it.
    disturbance_amplitude_rad: float = 0.25 # 0.05
    # [rad] small phase-jitter amplitude added to each rotor's phase before
    # taking the sine, representing e.g. turbulence-driven loading/RPM
    # fluctuations not otherwise captured by the trajectory optimization.
    # 0.05 rad (~3 deg) is a deliberately small starting point; increase if
    # you want a "rougher" sounding tone, decrease/zero for a pure tone.
    disturbance_bandwidth_hz: float = 2.0 #20.0
    # [Hz] low-pass cutoff applied to the random phase jitter, so the
    # disturbance only contains slow-ish wander (e.g. up to a few tens of
    # Hz) rather than being smeared as broadband noise to the Nyquist
    # frequency, which would not be physically representative.
    random_seed: int = 42 # None       # set for reproducible disturbance realizations


def _dedupe_time_and_signals(t, signals: dict) -> tuple:
    """
    Dymos timeseries output often has duplicate time values at phase/segment
    boundaries (the last node of one segment and the first node of the next
    share the same time). CubicSpline (and, strictly, np.interp too)
    requires a strictly increasing x-array, so this removes duplicate/
    non-increasing time points, keeping the FIRST occurrence of each time
    value, and applies the same filtering mask to every signal so they all
    stay aligned with the cleaned time vector.

    Parameters
    ----------
    t : array-like
        Coarse time vector, possibly with duplicate/repeated values.
    signals : dict of array-like
        Other coarse signals (position, RPM, ...) sampled at the same time
        points as t.

    Returns
    -------
    t_clean : ndarray
    signals_clean : dict of ndarray, same keys as `signals`
    """
    t = np.asarray(t, dtype=float).ravel()
    # keep first occurrence of each strictly-increasing time value
    keep = np.concatenate(([True], np.diff(t) > 0))
    t_clean = t[keep]
    signals_clean = {
        name: np.asarray(sig, dtype=float).ravel()[keep] for name, sig in signals.items()
    }
    return t_clean, signals_clean


def generate_fine_time_grid(t_coarse, fs: float) -> np.ndarray:
    """Uniform fine time grid spanning the coarse trajectory's duration."""
    t_coarse = np.asarray(t_coarse, dtype=float).ravel()
    n_fine = int(np.floor((t_coarse[-1] - t_coarse[0]) * fs)) + 1
    return t_coarse[0] + np.arange(n_fine) / fs


def interpolate_to_fine_grid(t_coarse, signal_coarse, t_fine, method: str = "cubic") -> np.ndarray:
    """Interpolate one coarse Dymos signal onto the fine time grid."""
    t_coarse = np.asarray(t_coarse, dtype=float).ravel()
    signal_coarse = np.asarray(signal_coarse, dtype=float).ravel()
    if method == "cubic":
        spline = CubicSpline(t_coarse, signal_coarse)
        return spline(t_fine)
    elif method == "linear":
        return np.interp(t_fine, t_coarse, signal_coarse)
    else:
        raise ValueError("interp_method must be 'cubic' or 'linear'")


def _band_limited_noise(n_samples: int, fs: float, bandwidth_hz: float, rng) -> np.ndarray:
    """Gaussian white noise, low-pass filtered to `bandwidth_hz`, unit std after filtering."""
    noise = rng.normal(0.0, 1.0, size=n_samples)
    if bandwidth_hz is None or bandwidth_hz <= 0:
        return noise
    nyq = fs / 2.0
    wn = min(bandwidth_hz / nyq, 0.99)  # keep strictly below Nyquist
    b, a = butter(N=2, Wn=wn, btype="low")
    filtered = filtfilt(b, a, noise)
    std = np.std(filtered)
    return filtered / std if std > 0 else filtered


def generate_rotor_azimuth(rpm_fine: np.ndarray, t_fine: np.ndarray,
                            fine_params: FineGridParams = None, rng=None):
    """
    Generate the azimuth signal psi(t) = sin(phase(t) + disturbance(t)) for
    one rotor on the fine time grid, given its (already interpolated) RPM
    time series.

    Returns
    -------
    psi : ndarray
        sin(phase + disturbance), dimensionless, in [-1, 1].
    phase : ndarray
        The underlying (un-disturbed) phase [rad], returned for diagnostics
        / use in synthesizing blade-passage-frequency tones later
        (e.g. sin(n_blades * phase) for the n-th blade harmonic).
    disturbance : ndarray
        The band-limited phase jitter actually added [rad].
    """
    if fine_params is None:
        fine_params = FineGridParams()
    if rng is None:
        rng = np.random.default_rng(fine_params.random_seed)

    rpm_fine = np.asarray(rpm_fine, dtype=float).ravel()

    if fine_params.use_integrated_phase:
        # Exact for time-varying RPM: phase = 2*pi * integral(f(t) dt),
        # f(t) = RPM(t)/60 [Hz].
        phase = 2.0 * np.pi * cumulative_trapezoid(rpm_fine / 60.0, t_fine, initial=0.0)
    else:
        # Literal formula as originally specified; exact only if RPM is
        # (locally) constant -- see module/function docstrings for why.
        phase = 2.0 * np.pi * (rpm_fine / 60.0) * t_fine

    disturbance = fine_params.disturbance_amplitude_rad * _band_limited_noise(
        n_samples=t_fine.size, fs=fine_params.fs,
        bandwidth_hz=fine_params.disturbance_bandwidth_hz, rng=rng,
    )

    psi = np.sin(phase + disturbance) * acoustic_power_per_rotor(rpm_fine, fine_params)
    return psi, phase, disturbance


def estimate_received_spl_fine(t, x, y, z,
                                rpm_front, rpm_rear, rpm_right, rpm_left,
                                observer_xyz,
                                acoustic_params: AcousticParams = None,
                                fine_params: FineGridParams = None,
                                z_up: bool = True) -> dict:
    """
    Fine-time-grid version of estimate_received_spl(): upsamples the coarse
    Dymos time history to a fine grid (suitable for downstream
    psychoacoustic/time-domain analysis), generates each rotor's azimuth
    signal (with a small disturbance), and recomputes the received SPL
    along the fine grid using the same RPM-to-power and directivity model
    as estimate_received_spl().

    Parameters
    ----------
    t, x, y, z : array-like
        COARSE Dymos time and drone position time series.
    rpm_front, rpm_rear, rpm_right, rpm_left : array-like
        COARSE per-rotor RPM time series, e.g. from estimate_rotor_rpm().
    observer_xyz : tuple of 3 floats
        Fixed observer location (x_obs, y_obs, z_obs) [m].
    acoustic_params : AcousticParams, optional
        RPM-to-power and directivity assumptions (see module docstring).
    fine_params : FineGridParams, optional
        Upsampling rate, interpolation method, and rotor-phase/disturbance
        assumptions (see FineGridParams docstring).
    z_up : bool
        Must match the convention used in rotor_rpm_estimation.py.

    Returns
    -------
    dict with keys:
        't_fine'                       : fine time vector [s]
        'spl_db', 'swl_db', 'r',
        'theta_deg', 'p_total'          : same meaning as in
                                           estimate_received_spl(), but on
                                           the fine grid
        'rpm_front_fine', ... '_left_fine' : interpolated per-rotor RPM [RPM]
        'psi_front', ... '_left'        : per-rotor azimuth signal sin(phase
                                           + disturbance), dimensionless
        'phase_front', ... '_left'      : per-rotor underlying phase [rad]
                                           (un-disturbed); useful for
                                           synthesizing blade-passage tones
                                           later, e.g. sin(n_blades * phase)
    """
    if acoustic_params is None:
        acoustic_params = AcousticParams()
    if fine_params is None:
        fine_params = FineGridParams()

    t = np.asarray(t, dtype=float).ravel()
    rotor_rpm_coarse = {
        "front": rpm_front, "rear": rpm_rear, "right": rpm_right, "left": rpm_left,
    }

    # --- 0. clean up duplicate/non-increasing time points (common at Dymos
    #         phase/segment boundaries) before any interpolation ---
    t, cleaned = _dedupe_time_and_signals(
        t, {"x": x, "y": y, "z": z, **rotor_rpm_coarse}
    )
    x, y, z = cleaned["x"], cleaned["y"], cleaned["z"]
    rotor_rpm_coarse = {name: cleaned[name] for name in rotor_rpm_coarse}

    # --- 1. fine time grid ---
    t_fine = generate_fine_time_grid(t, fine_params.fs)

    # quick sanity check: is fs comfortably above the rotor rotation rate?
    max_rpm = max(np.max(np.asarray(v, dtype=float)) for v in rotor_rpm_coarse.values())
    max_rotation_hz = max_rpm / 60.0
    if fine_params.fs < 10 * max_rotation_hz:
        import warnings
        warnings.warn(
            f"fs={fine_params.fs:.0f} Hz is less than 10x the max rotor "
            f"rotation rate ({max_rotation_hz:.1f} Hz). Consider raising fs "
            "for adequate resolution of rotor harmonics."
        )

    # --- 2. interpolate position and RPM signals onto the fine grid ---
    x_fine = interpolate_to_fine_grid(t, x, t_fine, fine_params.interp_method)
    y_fine = interpolate_to_fine_grid(t, y, t_fine, fine_params.interp_method)
    z_fine = interpolate_to_fine_grid(t, z, t_fine, fine_params.interp_method)

    rpm_fine = {
        name: interpolate_to_fine_grid(t, rpm, t_fine, fine_params.interp_method)
        for name, rpm in rotor_rpm_coarse.items()
    }
    # RPM can't go negative; clip any small interpolation overshoot.
    for name in rpm_fine:
        rpm_fine[name] = np.clip(rpm_fine[name], 0.0, None)

    # --- 3. per-rotor azimuth signal (with disturbance) ---
    rng = np.random.default_rng(fine_params.random_seed)
    psi, phase, disturbance = {}, {}, {}
    for name in rpm_fine:
        psi[name], phase[name], disturbance[name] = generate_rotor_azimuth(
            rpm_fine[name], t_fine, fine_params, rng=rng
        )

    # --- 4. sound power / SPL, same model as estimate_received_spl(), on t_fine ---
    p_total = sum(acoustic_power_per_rotor(rpm_fine[name], acoustic_params) for name in rpm_fine)
    swl_db = 10.0 * np.log10(p_total / 1e-12)

    r, theta = observer_geometry(x_fine, y_fine, z_fine, observer_xyz, z_up=z_up)
    D = directivity_factor(theta)
    spl_db = swl_db + 10.0 * np.log10(D) - 20.0 * np.log10(r) - 11.0

    result = {
        "t_fine": t_fine,
        "spl_db": spl_db,
        "swl_db": swl_db,
        "r": r,
        "theta_deg": np.degrees(theta),
        "p_total": p_total,
    }
    for name in rpm_fine:
        result[f"rpm_{name}_fine"] = rpm_fine[name]
        result[f"psi_{name}"] = psi[name]
        result[f"phase_{name}"] = phase[name]
    return result



    # --- Minimal usage example, continuing from the rotor RPM estimation step ---
    from rotor_rpm_estimation import estimate_rotor_rpm, DroneParams

    n = 500
    t = np.linspace(0, 10, n)
    vx = 5.0 + 2.0 * t / t[-1]
    vy = np.zeros(n)
    vz = 0.5 * np.sin(0.3 * t)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    az = np.gradient(vz, t)
    x = np.cumsum(vx) * (t[1] - t[0])
    y = np.cumsum(vy) * (t[1] - t[0])
    z = 50.0 + np.cumsum(vz) * (t[1] - t[0])  # start at 50 m altitude
    wx = 3.0 * np.ones(n)
    wy = np.zeros(n)
    wz = np.zeros(n)

    rpm_result = estimate_rotor_rpm(t, x, y, z, vx, vy, vz, ax, ay, az, wx, wy, wz)

    # Calibrate p_ref against a plausible datasheet-style reference:
    # "62 dB at 1 m, in-plane (90 deg), all 4 rotors at 5000 RPM hover"
    p_ref = calibrate_p_ref(
        spl_ref_db=62.0,
        rpm_ref_measurement=5000.0,
        r_ref=1.0,
        theta_ref_deg=90.0,
        n_rotors_in_measurement=4,
        n_exponent=5.0,
    )
    acoustic_params = AcousticParams(rpm_ref=5000.0, p_ref=p_ref, n_exponent=5.0)

    # Fixed ground observer, 50 m horizontally from the trajectory's start, at ground level
    observer_xyz = (0.0, 50.0, 0.0)

    spl_result = estimate_received_spl(
        t, x, y, z,
        rpm_result["rpm_front"], rpm_result["rpm_rear"],
        rpm_result["rpm_right"], rpm_result["rpm_left"],
        observer_xyz, params=acoustic_params,
    )

    print("Received SPL at observer (synthetic example, coarse grid):")
    print(f"  SPL range      : {spl_result['spl_db'].min():.1f} - {spl_result['spl_db'].max():.1f} dB")
    print(f"  distance range : {spl_result['r'].min():.1f} - {spl_result['r'].max():.1f} m")
    print(f"  theta range     : {spl_result['theta_deg'].min():.1f} - {spl_result['theta_deg'].max():.1f} deg")

    # --- Fine-time-grid version, with rotor azimuth + disturbance ---
    fine_params = FineGridParams(fs=48000.0, disturbance_amplitude_rad=0.05)
    spl_fine = estimate_received_spl_fine(
        t, x, y, z,
        rpm_result["rpm_front"], rpm_result["rpm_rear"],
        rpm_result["rpm_right"], rpm_result["rpm_left"],
        observer_xyz, acoustic_params=acoustic_params, fine_params=fine_params,
    )

    print("\nReceived SPL at observer (fine grid, 48 kHz):")
    print(f"  n samples      : {spl_fine['t_fine'].size}")
    print(f"  t_fine range   : {spl_fine['t_fine'][0]:.3f} - {spl_fine['t_fine'][-1]:.3f} s")
    print(f"  SPL range      : {spl_fine['spl_db'].min():.1f} - {spl_fine['spl_db'].max():.1f} dB")
    print(f"  psi_front range: {spl_fine['psi_front'].min():.2f} - {spl_fine['psi_front'].max():.2f}")