"""
zwicker_annoyance.py
=====================

Compute the classic Zwicker psychoacoustic indicators

    - Loudness            N   [sone]   (ISO 532-1, "Zwicker method")
    - Sharpness           S   [acum]   (DIN 45692)
    - Roughness           R   [asper]  (Daniel & Weber, 1997)
    - Fluctuation strength F  [vacil]
    - Psychoacoustic Annoyance PA      (Zwicker & Fastl combination of the above)

from a calibrated time-domain acoustic pressure signal.

These indicators are all derived from the time signal via the Bark-scale
("specific loudness") spectrum, i.e. they require both a time-domain input
and an internal frequency/Bark-domain analysis. They are not approximations
in the sense of being guesses - they follow standardized signal-processing
models - but they ARE only as accurate as those models (e.g. ISO 532-1 is
validated for broadband, free- or diffuse-field sounds; results for very
unusual signals should be interpreted with care).

This script relies on MOSQITO (Modular Sound Quality Integrated Toolbox),
an open-source, validated implementation of these standards:
    https://github.com/Eomys/MoSQITo

Install:
    pip install mosqito --break-system-packages

Notes
-----
- The input signal must be a calibrated mono pressure signal in Pascals
  (not just normalized -1..1 audio). If you only have a normalized signal,
  you need a calibration factor (Pa per unit) from your measurement chain,
  otherwise loudness/sharpness/roughness values will be physically wrong
  (they are level-dependent), even though the script will still run.
- MOSQITO's *_st (stationary) functions assume the signal's statistics do
  not change much over its duration. For signals with significant level or
  spectral variation over time, use the *_tv (time-varying) functions
  instead (see compute_zwicker_indicators(..., stationary=False)).
"""

import numpy as np


def _first(out):
    """
    Defensively extract the primary array/value from a MOSQITO function's
    return, regardless of how many extra outputs (spectra, axes, etc.) that
    particular version of the library tacks on.
    """
    if isinstance(out, tuple):
        return out[0]
    return out


def compute_zwicker_indicators(signal: np.ndarray, fs: int, stationary: bool = True,
                                use_fs_approximation: bool = False) -> dict:
    """
    Compute Zwicker-based psychoacoustic indicators and the Zwicker/Fastl
    Psychoacoustic Annoyance (PA) index for a mono time-domain signal.

    Parameters
    ----------
    signal : 1D numpy array
        Calibrated mono acoustic pressure signal, in Pascals.
    fs : int
        Sampling frequency, in Hz.
    stationary : bool, default True
        If True, use the *_st (stationary) MOSQITO routines and return single
        scalar values for N, S, R, F.
        If False, use the *_tv (time-varying) routines and additionally return
        the N5 percentile of loudness (exceeded 5% of the signal duration),
        which is the value Zwicker's PA formula is meant to use for
        non-stationary sounds, plus the full time series for inspection/plots.
    use_fs_approximation : bool, default False
        If True and MOSQITO doesn't provide fluctuation strength (true as of
        v1.2.1), fall back to a low-accuracy literature-based approximation
        (see approximate_fluctuation_strength()) instead of returning None/
        treating F as 0 in the PA formula.

    Returns
    -------
    dict
        {
          'loudness_sone'        : float,
          'sharpness_acum'       : float,
          'roughness_asper'      : float,
          'fluctuation_vacil'    : float,
          'annoyance_PA'         : float,
          # present only if stationary=False:
          'loudness_N5_sone'     : float,
          'time_series'          : {...}   # raw arrays for plotting
        }
    """
    from mosqito.sq_metrics import (
        loudness_zwst,
        sharpness_din_st,
        roughness_dw,
    )

    signal = np.asarray(signal, dtype=float)

    if stationary:
        # --- Loudness (ISO 532-1, Zwicker, stationary) ---
        N = _first(loudness_zwst(signal, fs))
        N_val = float(np.mean(N)) if np.ndim(N) > 0 else float(N)

        # --- Sharpness (DIN 45692, stationary) ---
        S_val = float(np.mean(_first(sharpness_din_st(signal, fs, weighting="din"))))

        # --- Roughness (Daniel & Weber) ---
        # roughness_dw operates on a time-varying basis internally; we average
        # its output to get a single representative value for a stationary sound.
        R = _first(roughness_dw(signal, fs))
        R_val = float(np.mean(R))

        # --- Fluctuation strength ---
        # MOSQITO versions differ in where this lives; try the common locations.
        F_val = _compute_fluctuation_strength(signal, fs, use_approximation=use_fs_approximation)

        result = {
            "loudness_sone": N_val,
            "sharpness_acum": S_val,
            "roughness_asper": R_val,
            "fluctuation_vacil": F_val,
        }
        result["annoyance_PA"] = zwicker_annoyance(N_val, S_val, R_val, F_val)
        return result

    else:
        from mosqito.sq_metrics import loudness_zwtv, sharpness_din_tv

        N_out = loudness_zwtv(signal, fs)
        N = _first(N_out)
        N5 = float(np.percentile(N, 95))  # exceeded 5% of the time

        S = _first(sharpness_din_tv(signal, fs, weighting="din"))
        S_val = float(np.mean(S))

        R = _first(roughness_dw(signal, fs))
        R_val = float(np.mean(R))

        F_val = _compute_fluctuation_strength(signal, fs, use_approximation=use_fs_approximation)

        result = {
            "loudness_sone": float(np.mean(N)),
            "loudness_N5_sone": N5,
            "sharpness_acum": S_val,
            "roughness_asper": R_val,
            "fluctuation_vacil": F_val,
            "time_series": {
                "N": N,
                "S": S,
                "R": R,
            },
        }
        # Zwicker's PA formula is normally evaluated with N5 for time-varying sounds
        result["annoyance_PA"] = zwicker_annoyance(N5, S_val, R_val, F_val)
        return result


def approximate_fluctuation_strength(signal: np.ndarray, fs: int,
                                      fmod_range: tuple = (0.5, 20.0)) -> float:
    """
    Low-accuracy fluctuation strength approximation, for use when MOSQITO
    does not provide a fluctuation_strength implementation (as in v1.2.1).

    Uses the classical Zwicker & Fastl closed-form result for a single
    sinusoidally amplitude-modulated tone:

        FS  ~=  0.008 * dL_dB / (fmod/4 + 4/fmod)      [vacil]

    where dL_dB is the modulation depth in dB and fmod is the modulation
    frequency in Hz. This formula peaks at fmod = 4 Hz, matching the known
    psychoacoustic maximum sensitivity to slow loudness fluctuations, and
    falls off for faster (-> roughness territory) or slower modulation.

    To apply it to an arbitrary signal (not just a clean AM tone), this
    function:
      1. Extracts the signal's envelope via the analytic signal (Hilbert
         transform).
      2. Looks at the envelope's power spectrum, restricted to
         `fmod_range` (default 0.5-20 Hz, i.e. excluding both DC/very slow
         trends and the roughness regime above ~20 Hz), and finds the
         dominant modulation frequency fmod.
      3. Estimates the modulation depth dL_dB from the envelope's
         peak-to-trough range, converted to dB.
      4. Plugs both into the formula above.

    CAVEATS (please read before using the result quantitatively):
    - This is only exact for a single sinusoidal AM tone. Real signals
      (e.g. a multi-rotor pressure signal with several simultaneous,
      time-varying modulation frequencies) will only be approximated, and
      can be substantially off if multiple modulation components are
      similarly strong.
    - It is a single time-averaged estimate, not a true psychoacoustic
      model with masking, critical bands, etc. (unlike loudness/sharpness/
      roughness, which ARE full standardized models in MOSQITO).
    - Treat the result as indicative (right order of magnitude, right
      qualitative trend if you compare across trajectories/scenarios), not
      as a substitute for a validated fluctuation-strength implementation.

    Returns
    -------
    FS_vacil : float
    """
    from scipy.signal import hilbert, welch

    signal = np.asarray(signal, dtype=float).ravel()
    envelope = np.abs(hilbert(signal))

    # avoid log(0) for near-silent stretches
    envelope_db = 20.0 * np.log10(np.clip(envelope, 1e-12, None))

    # dominant modulation frequency, from the envelope's own power spectrum
    f_welch, pxx = welch(envelope - np.mean(envelope), fs=fs, nperseg=min(len(envelope), fs * 2))
    band = (f_welch >= fmod_range[0]) & (f_welch <= fmod_range[1])
    if not np.any(band) or np.all(pxx[band] == 0):
        return 0.0
    fmod = float(f_welch[band][np.argmax(pxx[band])])
    fmod = max(fmod, 1e-3)  # guard against divide-by-zero in the formula below

    # modulation depth from envelope dB range (5th-95th percentile, robust to outliers)
    dL_dB = float(np.percentile(envelope_db, 95) - np.percentile(envelope_db, 5))

    FS = 0.008 * dL_dB / (fmod / 4.0 + 4.0 / fmod)
    return max(FS, 0.0)


def _compute_fluctuation_strength(signal: np.ndarray, fs: int, use_approximation: bool = False):
    """
    Try the available MOSQITO fluctuation-strength implementation.

    NOTE: as of MOSQITO 1.2.1 (confirmed via `dir(mosqito.sq_metrics)`),
    fluctuation strength is NOT implemented in this library at all -- there
    is no fluctuation_strength* function in sq_metrics. This helper still
    probes for it (in case you're on a newer/dev version that adds it
    later), but on 1.2.1 it will always fall through to the warning below
    and return None.
    """
    import importlib
    import warnings

    candidates = [
        ("mosqito.sq_metrics", "fluctuation_strength"),
        ("mosqito.sq_metrics", "fluctuation_strength_din"),
    ]
    for module_name, func_name in candidates:
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
        except (ImportError, AttributeError):
            continue
        F = _first(func(signal, fs))
        return float(np.mean(F))

    if use_approximation:
        return approximate_fluctuation_strength(signal, fs)

    warnings.warn(
        "Fluctuation strength is not implemented in your installed MOSQITO "
        "version (confirmed absent in mosqito.sq_metrics for v1.2.1). "
        "Continuing without it; PA below uses F=0, which slightly "
        "underestimates annoyance for sounds with strong slow modulation "
        "(e.g. sirens). Check https://github.com/Eomys/MoSQITo for newer "
        "releases, or pass use_approximation=True to "
        "compute_zwicker_indicators() for a low-accuracy literature-based "
        "estimate (see approximate_fluctuation_strength())."
    )
    return None


def zwicker_annoyance(N: float, S: float, R: float, F) -> float:
    """
    Zwicker & Fastl Psychoacoustic Annoyance (PA), combining loudness,
    sharpness, roughness and fluctuation strength into one index.

        wS  = (S - 1.75) * 0.25 * log10(N + 10),   if S > 1.75 acum, else 0
        wFR = 2.18 / N**0.4 * (0.4*F + 0.6*R)
        PA  = N * (1 + sqrt(wS**2 + wFR**2))

    N should be in sone (use N5 for time-varying sounds), S in acum,
    R in asper, F in vacil. If F is None (e.g. not available in your
    installed MOSQITO version), it is treated as 0 -- PA will then slightly
    underestimate annoyance for sounds with strong slow modulation.

    Reference: Zwicker, E. and Fastl, H., "Psychoacoustics: Facts and Models",
    Springer, and the PA formulation as used e.g. by Widmann (1992).
    """
    if F is None:
        F = 0.0
    N = max(N, 1e-6)  # avoid log/pow domain errors for (near-)silent signals
    wS = (S - 1.75) * 0.25 * np.log10(N + 10) if S > 1.75 else 0.0
    wFR = (2.18 / N**0.4) * (0.4 * F + 0.6 * R)
    PA = N * (1 + np.sqrt(wS**2 + wFR**2))
    return float(PA)


def compute_zwicker_indicators_windowed(signal: np.ndarray, fs: int,
                                         window_s: float = 2.0, hop_s: float = 0.5,
                                         stationary: bool = True,
                                         use_fs_approximation: bool = False) -> dict:
    """
    Compute Zwicker indicators on a sliding window over a long signal,
    producing a "psychoacoustic spectrogram": each indicator's value as a
    function of time, instead of one number for the whole signal.

    This is the natural way to analyze a long (e.g. ~100 s flyover) signal
    whose level/spectrum/modulation content changes substantially over
    time -- a single stationary call would blur all of that into one
    number, while windowing reveals e.g. how annoyance peaks as the drone
    passes overhead.

    Parameters
    ----------
    signal : 1D array-like
        Calibrated pressure signal [Pa], e.g. spl_fine['p_signal'].
    fs : int
        Sample rate [Hz].
    window_s : float, default 2.0
        Window length [s]. MOSQITO's stationary loudness/sharpness models
        (ISO 532-1, DIN 45692) were validated on signals on the order of a
        fraction of a second to a few seconds; 1-4 s is a reasonable
        starting point. Too short and loudness/sharpness become noisy
        (insufficient low-frequency resolution); too long and you wash out
        the time variation you're trying to see.
    hop_s : float, default 0.5
        Step between successive window starts [s]. hop_s < window_s gives
        overlapping windows (smoother time series); hop_s = window_s gives
        non-overlapping windows.
    stationary : bool, default True
        Whether each individual window is itself treated as stationary
        (almost always appropriate once windows are a few seconds or
        shorter) or time-varying internally. Stationary is faster and
        usually sufficient -- the windowing itself is what captures the
        signal's overall non-stationarity.
    use_fs_approximation : bool, default False
        See compute_zwicker_indicators().

    Returns
    -------
    dict with keys:
        't_center'           : window center times [s]
        'loudness_sone'      : array, one value per window
        'sharpness_acum'     : array
        'roughness_asper'    : array
        'fluctuation_vacil'  : array (NaN where unavailable/None)
        'annoyance_PA'       : array
    """
    signal = np.asarray(signal, dtype=float).ravel()
    n = signal.size
    win_n = int(round(window_s * fs))
    hop_n = int(round(hop_s * fs))
    if win_n < 1 or win_n > n:
        raise ValueError(
            f"window_s={window_s} s ({win_n} samples) doesn't fit the signal "
            f"({n} samples at fs={fs} Hz)."
        )

    starts = np.arange(0, n - win_n + 1, hop_n)
    t_center = (starts + win_n / 2.0) / fs

    keys = ["loudness_sone", "sharpness_acum", "roughness_asper",
            "fluctuation_vacil", "annoyance_PA"]
    out = {k: np.full(len(starts), np.nan) for k in keys}

    for i, s0 in enumerate(starts):
        segment = signal[s0:s0 + win_n]
        ind = compute_zwicker_indicators(
            segment, fs, stationary=stationary, use_fs_approximation=use_fs_approximation
        )
        for k in keys:
            v = ind.get(k, None)
            if v is not None:
                out[k][i] = v

    out["t_center"] = t_center
    return out


if __name__ == "__main__":
    # --- Minimal usage example with a synthetic test signal ---
    # Replace this with your own calibrated measurement, e.g.:
    #   import soundfile as sf
    #   signal, fs = sf.read("my_recording.wav")
    #   signal = signal * calibration_factor_pa_per_unit

    fs = 48000  # MOSQITO's ISO 532-1 implementation expects 48 kHz
    duration = 1.0
    t = np.arange(0, duration, 1 / fs)

    # A simple amplitude-modulated tone as a illustrative test case
    # (modulated tones are exactly the kind of signal roughness/fluctuation
    # strength are designed to pick up on).
    carrier = np.sin(2 * np.pi * 1000 * t)
    modulator = 1 + 0.5 * np.sin(2 * np.pi * 70 * t)  # 70 Hz -> roughness range
    test_signal = 0.5 * carrier * modulator  # amplitude in Pa (illustrative only)

    indicators = compute_zwicker_indicators(test_signal, fs, stationary=True)

    print("Zwicker psychoacoustic indicators:")
    for key, value in indicators.items():
        if value is None:
            print(f"  {key:20s}: n/a")
        else:
            print(f"  {key:20s}: {value:.4f}")