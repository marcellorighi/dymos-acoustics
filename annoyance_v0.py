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


def compute_zwicker_indicators(signal: np.ndarray, fs: int, stationary: bool = True) -> dict:
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
        F_val = _compute_fluctuation_strength(signal, fs)

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

        F_val = _compute_fluctuation_strength(signal, fs)

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


def _compute_fluctuation_strength(signal: np.ndarray, fs: int):
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

    warnings.warn(
        "Fluctuation strength is not implemented in your installed MOSQITO "
        "version (confirmed absent in mosqito.sq_metrics for v1.2.1). "
        "Continuing without it; PA below uses F=0, which slightly "
        "underestimates annoyance for sounds with strong slow modulation "
        "(e.g. sirens). Check https://github.com/Eomys/MoSQITo for newer "
        "releases, or implement fluctuation strength separately if you "
        "need it (e.g. Zwicker & Fastl 1990, or Sottek's model)."
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
    modulator = 1 + 0.75 * np.sin(2 * np.pi * 70 * t)  +  0.25 * np.sin(2 * np.pi * 70.1 * t) # 70 Hz -> roughness range
    test_signal = 0.5 * carrier * modulator  # amplitude in Pa (illustrative only)

    indicators = compute_zwicker_indicators(test_signal, fs, stationary=True)

    print("Zwicker psychoacoustic indicators:")
    for key, value in indicators.items():
        if value is None:
            print(f"  {key:20s}: n/a")
        else:
            print(f"  {key:20s}: {value:.4f}")
            
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Subplot 1: Input AoA
    ax[0].plot(t, carrier, label="Carrier")
    ax[0].plot(t, modulator, label="Modulator")
    ax[0].plot(t, test_signal, label="Test Signal")
    ax[0].set_xlabel(r'$\t\,[s]$')
    ax[0].set_ylabel(r'$W\, [Pa]$')
    ax[0].legend()
    ax[0].grid(True)

    plt.tight_layout()
    plt.show()
    