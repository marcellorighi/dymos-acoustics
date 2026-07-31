#!/usr/bin/env python3
# """
# Analisi armonica robusta dei WAV NEAPTIDE – DJI Matrice 300 RTK.

# Versione "safe":
# - evita pandas;
# - evita scipy.signal;
# - evita numpy.polyfit;
# - estrae i picchi usando intervalli contigui;
# - salva CSV con il modulo standard csv;
# - stampa un checkpoint dopo ogni passaggio;
# - conserva i risultati parziali anche se una fase successiva fallisce.
# """

# from __future__ import annotations

# import os

# # Impostare i limiti PRIMA di importare NumPy/Matplotlib.
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# import argparse
# import csv
# import math
# import re
# import statistics
# import sys
# import traceback
from pathlib import Path
# from typing import Any

# # import matplotlib

# # matplotlib.use("Agg")

# import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt 

# from scipy import signal as scipy_signal


# P0_UPA = 20.0

DEFAULT_INPUT_DIR = Path(
    r"U:\GitHub\DRACONIAN\Acoustic Data"
    r"\SegmentedAudio_Matrice\SegmentedAudio_Matrice"
)

#    / "DJIMatrice300RTK" 92.5 Hz
#    / "DJIMavic2Enterprise" 188 Hz
#    / "HolybroS500" 190. Hz 
#    / "TarotX6_B1" 148 Hz 
#    / "TarotX6_B2" 166 Hz 


DEFAULT_INPUT_DIR = (
    Path.home()
    / "Documents"
    / "zhaw"
    / "BAZL"
    / "DRACONIAN" 
    / "Dymos"
    / "Acoustics"
    / "Neaptide_data"
    / "DJIMavic2Enterprise" # 188 Hz
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "Documents"
    / "zhaw"
    / "BAZL"
    / "DRACONIAN" 
    / "Dymos"
    / "Acoustics"
    / "Neaptide_data"
    / "DJIMavic2Enterprise" # 188 Hz
)


# def log(message: str) -> None:
#     print(message, flush=True)


# def microphone_number(path: Path) -> int:
#     match = re.search(r"_(\d+)\.wav$", path.name, flags=re.IGNORECASE)
#     return int(match.group(1)) if match else -1


def read_neaptide_wav(
    path: Path,
    scale_factor: float = 20.0,
) -> tuple[int, np.ndarray]:
    fs, raw = wavfile.read(path)

    if raw.ndim == 2:
        raw_float = raw.astype(np.float64).mean(axis=1)
    else:
        raw_float = raw.astype(np.float64)

    pressure_uPa = raw_float / scale_factor
    pressure_uPa -= float(np.mean(pressure_uPa))

    if not bool(np.all(np.isfinite(pressure_uPa))):
        raise ValueError(f"Valori non finiti nel file {path}")

    return int(fs), pressure_uPa


# def overall_spl_db(pressure_uPa: np.ndarray) -> float:
#     mean_square = float(np.mean(pressure_uPa * pressure_uPa))
#     if mean_square <= 0.0:
#         return float("-inf")

#     rms = math.sqrt(mean_square)
#     return 20.0 * math.log10(rms / P0_UPA)

# import numpy as np

def welch_psd_numpy(
    signal: np.ndarray,
    fs: int,
    nperseg: int,
    overlap_fraction: float = 0.75,
    pref: float = 20e-6,
    psd_floor: float = 1e-30,
) -> dict:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Il segnale deve essere monodimensionale.")

    nperseg = min(max(256, int(nperseg)), len(x))
    noverlap = min(
        max(0, int(round(overlap_fraction * nperseg))),
        nperseg - 1,
    )
    step = nperseg - noverlap
    nfft = nperseg

    # 1. Hanning window energy normalization
    window = np.hanning(nperseg)
    window_power = float(np.sum(window**2))

    starts = range(0, len(x) - nperseg + 1, step)
    psd_sum = np.zeros(nfft // 2 + 1, dtype=np.float64)
    segment_count = 0

    for start in starts:
        segment = np.array(x[start : start + nperseg], dtype=np.float64, copy=True)
        segment -= float(np.mean(segment))
        segment *= window

        spectrum = np.fft.rfft(segment, n=nfft)
        psd = (np.abs(spectrum) ** 2) / (fs * window_power)

        if nfft % 2 == 0:
            psd[1:-1] *= 2.0
        else:
            psd[1:] *= 2.0

        psd_sum += psd
        segment_count += 1

    if segment_count == 0:
        raise RuntimeError("No segments available for Welch.")

    mean_psd = psd_sum / float(segment_count)
    frequency = np.fft.rfftfreq(nfft, d=1.0 / fs)
    df = fs / nfft  # Bin width [Hz]

    # --- NARROWBAND METRICS ---
    psd_db_hz = 10 * np.log10(np.maximum(mean_psd, psd_floor) / (pref**2))
    spl_bin = 10 * np.log10(np.maximum(mean_psd * df, psd_floor) / (pref**2))

    p_meansq = np.sum(mean_psd) * df
    overall_level_db = 10 * np.log10(np.maximum(p_meansq, psd_floor) / (pref**2))

    # --- 🟢 NEW: 1/3-OCTAVE BAND AGGREGATION ---
    f_center_third, spl_third = _compute_third_octave_bands(
        frequency, mean_psd, df, fs, pref=pref, psd_floor=psd_floor
    )

    return {
        "freq": frequency,
        "df": df,
        "psd": mean_psd,
        "psd_db_hz": psd_db_hz,
        "spl_bin": spl_bin,
        "overall_level_db": overall_level_db,
        # 1/3-Octave outputs
        "freq_third_oct": f_center_third,
        "spl_third_oct": spl_third,
    }


def _compute_third_octave_bands(
    frequency: np.ndarray,
    mean_psd: np.ndarray,
    df: float,
    fs: float,
    pref: float = 20e-6,
    psd_floor: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Groups PSD narrowband bins into standard preferred 1/3-octave center frequencies (ISO 266 / ANSI S1.6).
    """
    # Standard nominal center frequencies (10 Hz to 20 kHz)
    standard_centers = np.array([
        10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 
        400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 
        6300, 8000, 10000, 12500, 16000, 20000
    ], dtype=float)

    # Filter center frequencies below Nyquist (fs / 2)
    valid_mask = standard_centers <= (fs / 2.0)
    f_centers = standard_centers[valid_mask]

    # Exact band edge factors for 1/3-octave: 2^(1/6)
    factor = 2.0 ** (1.0 / 6.0)
    
    spl_third = []
    valid_centers = []

    for fc in f_centers:
        f_lower = fc / factor
        f_upper = fc * factor

        # Find FFT bins that fall within this 1/3-octave frequency band
        bin_mask = (frequency >= f_lower) & (frequency < f_upper)

        if np.any(bin_mask):
            # Sum total acoustic power contained inside this band
            p_band = np.sum(mean_psd[bin_mask]) * df
            spl_band = 10 * np.log10(np.maximum(p_band, psd_floor) / (pref**2))
            
            spl_third.append(spl_band)
            valid_centers.append(fc)

    return np.array(valid_centers), np.array(spl_third)

# def stft_psd_numpy(
#     signal: np.ndarray,
#     fs: int,
#     nperseg: int = 4096,
#     overlap_fraction: float = 0.75,
#     nfft: int = 16384,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     x = np.asarray(signal, dtype=np.float64)
#     nperseg = min(max(256, int(nperseg)), len(x))
#     nfft = max(int(nfft), nperseg)

#     noverlap = min(
#         max(0, int(round(overlap_fraction * nperseg))),
#         nperseg - 1,
#     )
#     step = nperseg - noverlap

#     window = np.hanning(nperseg)
#     window_power = float(np.sum(window * window))

#     starts = list(range(0, len(x) - nperseg + 1, step))
#     if not starts:
#         starts = [0]

#     matrix = np.empty((nfft // 2 + 1, len(starts)), dtype=np.float64)
#     times = np.empty(len(starts), dtype=np.float64)

#     for column, start in enumerate(starts):
#         segment = np.array(x[start : start + nperseg], dtype=np.float64, copy=True)

#         if len(segment) < nperseg:
#             padded = np.zeros(nperseg, dtype=np.float64)
#             padded[: len(segment)] = segment
#             segment = padded

#         segment -= float(np.mean(segment))
#         segment *= window

#         spectrum = np.fft.rfft(segment, n=nfft)
#         psd = (
#             spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
#         ) / (fs * window_power)

#         if nfft % 2 == 0:
#             psd[1:-1] *= 2.0
#         else:
#             psd[1:] *= 2.0

#         matrix[:, column] = psd
#         times[column] = (start + nperseg / 2.0) / fs

#     frequency = np.fft.rfftfreq(nfft, d=1.0 / fs)
#     return frequency, times, matrix


# def psd_level_db(psd: np.ndarray) -> np.ndarray:
#     safe_psd = np.maximum(np.asarray(psd, dtype=np.float64), 1e-30)
#     return 10.0 * np.log10(safe_psd / (P0_UPA * P0_UPA))


# def estimate_comb_frequency(
#     frequency: np.ndarray,
#     psd: np.ndarray,
#     search_min_hz: float = 80.0,
#     search_max_hz: float = 110.0,
#     max_order: int = 8,
# ) -> float:
#     best_candidate = search_min_hz
#     best_score = float("-inf")

#     candidate = search_min_hz
#     while candidate <= search_max_hz + 1e-12:
#         score = 0.0

#         for order in range(1, max_order + 1):
#             expected = order * candidate
#             half_width = max(2.5, 0.015 * expected)

#             left = int(np.searchsorted(frequency, expected - half_width, side="left"))
#             right = int(np.searchsorted(frequency, expected + half_width, side="right"))

#             if right > left:
#                 local_max = float(np.max(psd[left:right]))
#                 score += local_max / math.sqrt(order)

#         if score > best_score:
#             best_score = score
#             best_candidate = candidate

#         candidate += 0.05

#     return float(best_candidate)


# def safe_median(values: np.ndarray) -> float:
#     if values.size == 0:
#         return float("nan")

#     # statistics.median evita il percorso interno di np.median.
#     return float(statistics.median(values.astype(float).tolist()))


# def extract_harmonics_safe(
#     frequency: np.ndarray,
#     psd: np.ndarray,
#     base_frequency_hz: float,
#     max_order: int,
#     microphone: int | str,
#     manoeuvre: str,
# ) -> list[dict[str, Any]]:
#     """
#     Estrae un massimo in ciascuna banda armonica.

#     Restituisce una lista di dizionari e non crea DataFrame.
#     """
#     levels = psd_level_db(psd)
#     rows: list[dict[str, Any]] = []

#     for order in range(1, max_order + 1):
#         log(
#             f"[{manoeuvre}] armoniche – mic {microphone}, "
#             f"ordine {order}/{max_order}"
#         )

#         expected = float(order * base_frequency_hz)
#         half_width = float(max(5.0, 0.045 * expected))

#         left = int(
#             np.searchsorted(
#                 frequency,
#                 expected - half_width,
#                 side="left",
#             )
#         )
#         right = int(
#             np.searchsorted(
#                 frequency,
#                 expected + half_width,
#                 side="right",
#             )
#         )

#         if right <= left:
#             continue

#         local_levels = levels[left:right]
#         local_peak_offset = int(np.argmax(local_levels))
#         peak_index = left + local_peak_offset

#         measured = float(frequency[peak_index])
#         peak_level = float(levels[peak_index])

#         broad_half_width = 1.8 * half_width
#         broad_left = int(
#             np.searchsorted(
#                 frequency,
#                 expected - broad_half_width,
#                 side="left",
#             )
#         )
#         broad_right = int(
#             np.searchsorted(
#                 frequency,
#                 expected + broad_half_width,
#                 side="right",
#             )
#         )

#         exclusion_width = float(max(3.0, 0.012 * expected))
#         exclusion_left = int(
#             np.searchsorted(
#                 frequency,
#                 measured - exclusion_width,
#                 side="left",
#             )
#         )
#         exclusion_right = int(
#             np.searchsorted(
#                 frequency,
#                 measured + exclusion_width,
#                 side="right",
#             )
#         )

#         floor_parts: list[np.ndarray] = []

#         if exclusion_left > broad_left:
#             floor_parts.append(levels[broad_left:exclusion_left])

#         if broad_right > exclusion_right:
#             floor_parts.append(levels[exclusion_right:broad_right])

#         if floor_parts:
#             floor_values = np.concatenate(floor_parts)
#             local_floor = safe_median(floor_values)
#         else:
#             local_floor = float("nan")

#         prominence = (
#             peak_level - local_floor
#             if math.isfinite(local_floor)
#             else float("nan")
#         )

#         rows.append(
#             {
#                 "manovra": manoeuvre,
#                 "microfono": microphone,
#                 "ordine": order,
#                 "frequenza_base_Hz": base_frequency_hz,
#                 "frequenza_attesa_Hz": expected,
#                 "frequenza_picco_Hz": measured,
#                 "errore_Hz": measured - expected,
#                 "livello_picco_dB_re_20uPa2_Hz": peak_level,
#                 "fondo_locale_dB_re_20uPa2_Hz": local_floor,
#                 "prominenza_tonale_dB": prominence,
#             }
#         )

#     return rows


# def fit_harmonic_decay_safe(
#     rows: list[dict[str, Any]],
#     minimum_prominence_db: float = 6.0,
# ) -> tuple[float, float, float, int]:
#     valid = [
#         row
#         for row in rows
#         if math.isfinite(float(row["livello_picco_dB_re_20uPa2_Hz"]))
#         and math.isfinite(float(row["prominenza_tonale_dB"]))
#         and float(row["prominenza_tonale_dB"]) >= minimum_prominence_db
#     ]

#     if len(valid) < 3:
#         valid = [
#             row
#             for row in rows
#             if math.isfinite(float(row["livello_picco_dB_re_20uPa2_Hz"]))
#         ]

#     if len(valid) < 2:
#         return float("nan"), float("nan"), float("nan"), len(valid)

#     x = [float(row["ordine"]) for row in valid]
#     y = [float(row["livello_picco_dB_re_20uPa2_Hz"]) for row in valid]

#     x_mean = sum(x) / len(x)
#     y_mean = sum(y) / len(y)

#     denominator = sum((value - x_mean) ** 2 for value in x)

#     if denominator <= 0.0:
#         return float("nan"), float("nan"), float("nan"), len(valid)

#     numerator = sum(
#         (x_value - x_mean) * (y_value - y_mean)
#         for x_value, y_value in zip(x, y)
#     )

#     slope = numerator / denominator
#     intercept = y_mean - slope * x_mean

#     predictions = [intercept + slope * value for value in x]
#     residual_sum = sum(
#         (observed - predicted) ** 2
#         for observed, predicted in zip(y, predictions)
#     )
#     total_sum = sum((observed - y_mean) ** 2 for observed in y)

#     r_squared = (
#         1.0 - residual_sum / total_sum
#         if total_sum > 0.0
#         else float("nan")
#     )

#     return slope, intercept, r_squared, len(valid)


# def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
#     if not rows:
#         path.write_text("", encoding="utf-8")
#         return

#     fieldnames = list(rows[0].keys())

#     with path.open("w", newline="", encoding="utf-8-sig") as handle:
#         writer = csv.DictWriter(handle, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)


# def write_spectrum_csv(
#     path: Path,
#     frequency: np.ndarray,
#     mean_psd: np.ndarray,
# ) -> None:
#     levels = psd_level_db(mean_psd)

#     with path.open("w", newline="", encoding="utf-8-sig") as handle:
#         writer = csv.writer(handle)
#         writer.writerow(
#             [
#                 "frequenza_Hz",
#                 "PSD_media_uPa2_Hz",
#                 "livello_PSD_medio_dB_re_20uPa2_Hz",
#             ]
#         )

#         for frequency_value, psd_value, level_value in zip(
#             frequency,
#             mean_psd,
#             levels,
#         ):
#             writer.writerow(
#                 [
#                     float(frequency_value),
#                     float(psd_value),
#                     float(level_value),
#                 ]
#             )


# def save_spectrum_plot(
#     frequency: np.ndarray,
#     individual_psd: list[np.ndarray],
#     labels: list[str],
#     mean_psd: np.ndarray,
#     base_frequency_hz: float,
#     max_order: int,
#     title: str,
#     output_path: Path,
#     max_frequency_hz: float = 1500.0,
# ) -> None:
#     left = int(np.searchsorted(frequency, 20.0, side="left"))
#     right = int(
#         np.searchsorted(
#             frequency,
#             max_frequency_hz,
#             side="right",
#         )
#     )

#     plt.figure(figsize=(11, 6))

#     for spectrum, label in zip(individual_psd, labels):
#         plt.plot(
#             frequency[left:right],
#             psd_level_db(spectrum[left:right]),
#             linewidth=0.7,
#             alpha=0.42,
#             label=label,
#         )

#     plt.plot(
#         frequency[left:right],
#         psd_level_db(mean_psd[left:right]),
#         linewidth=1.8,
#         label="Media lineare dei microfoni",
#     )

#     for order in range(1, max_order + 1):
#         harmonic = order * base_frequency_hz
#         if harmonic > max_frequency_hz:
#             break
#         plt.axvline(harmonic, linewidth=0.7, alpha=0.28)

#     plt.xlabel("Frequenza [Hz]")
#     plt.ylabel("PSD [dB re 20 µPa²/Hz]")
#     plt.title(title)
#     plt.grid(True, alpha=0.25)
#     plt.legend(ncol=2, fontsize=8)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=180)
#     plt.close()


# def save_decay_plot(
#     rows: list[dict[str, Any]],
#     slope: float,
#     intercept: float,
#     r_squared: float,
#     title: str,
#     output_path: Path,
# ) -> None:
#     orders = [float(row["ordine"]) for row in rows]
#     levels = [
#         float(row["livello_picco_dB_re_20uPa2_Hz"])
#         for row in rows
#     ]

#     plt.figure(figsize=(8, 5.5))
#     plt.scatter(orders, levels, label="Picchi misurati")

#     if (
#         orders
#         and math.isfinite(slope)
#         and math.isfinite(intercept)
#     ):
#         x_min = min(orders)
#         x_max = max(orders)
#         x_line = np.linspace(x_min, x_max, 100)
#         y_line = intercept + slope * x_line

#         plt.plot(
#             x_line,
#             y_line,
#             label=(
#                 f"Fit: {slope:.2f} dB/ordine, "
#                 f"R²={r_squared:.3f}"
#             ),
#         )

#     plt.xlabel("Ordine armonico")
#     plt.ylabel("Livello del picco [dB re 20 µPa²/Hz]")
#     plt.title(title)
#     plt.grid(True, alpha=0.25)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=180)
#     plt.close()


# def save_spectrogram(
#     pressure_uPa: np.ndarray,
#     fs: int,
#     title: str,
#     output_path: Path,
#     max_frequency_hz: float = 1500.0,
# ) -> None:
#     frequency, time, psd_matrix = stft_psd_numpy(
#         pressure_uPa,
#         fs,
#         nperseg=4096,
#         overlap_fraction=0.75,
#         nfft=16384,
#     )

#     left = int(np.searchsorted(frequency, 20.0, side="left"))
#     right = int(
#         np.searchsorted(
#             frequency,
#             max_frequency_hz,
#             side="right",
#         )
#     )

#     levels = psd_level_db(psd_matrix[left:right, :])

#     vmax = float(np.percentile(levels, 99.5))
#     vmin = vmax - 55.0

#     plt.figure(figsize=(11, 6))
#     mesh = plt.pcolormesh(
#         time,
#         frequency[left:right],
#         levels,
#         shading="auto",
#         vmin=vmin,
#         vmax=vmax,
#     )
#     plt.colorbar(mesh, label="PSD [dB re 20 µPa²/Hz]")
#     plt.xlabel("Tempo [s]")
#     plt.ylabel("Frequenza [Hz]")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=180)
#     plt.close()


# def analyse_group(
#     files: list[Path],
#     output_dir: Path,
#     label: str,
#     scale_factor: float,
#     requested_base_frequency: float | None,
#     max_order: int,
# ) -> dict[str, Any] | None:
#     if not files:
#         log(f"[{label}] Nessun file trovato.")
#         return None

#     log(f"[{label}] Funzione avviata con {len(files)} file.")

#     files = sorted(files, key=microphone_number)
#     spectra: list[np.ndarray] = []
#     labels: list[str] = []
#     file_rows: list[dict[str, Any]] = []
#     loaded_signals: list[tuple[Path, int, np.ndarray]] = []
#     frequency_reference: np.ndarray | None = None

#     nperseg = 65536 if label.lower() == "hover" else 16384

#     for index, path in enumerate(files, start=1):
#         log(f"[{label}] {index}/{len(files)} – lettura {path.name}")

#         fs, pressure_uPa = read_neaptide_wav(
#             path,
#             scale_factor=scale_factor,
#         )

#         duration = len(pressure_uPa) / fs
#         spl = overall_spl_db(pressure_uPa)

#         log(
#             f"[{label}] fs={fs} Hz, durata={duration:.3f} s, "
#             f"SPL={spl:.2f} dB. Calcolo Welch..."
#         )

#         frequency, psd = welch_psd_numpy(
#             pressure_uPa,
#             fs,
#             nperseg=nperseg,
#             overlap_fraction=0.75,
#         )

#         if frequency_reference is None:
#             frequency_reference = frequency
#         elif not np.array_equal(frequency_reference, frequency):
#             psd = np.interp(frequency_reference, frequency, psd)

#         spectra.append(psd)
#         mic = microphone_number(path)
#         labels.append(f"Mic {mic}")
#         loaded_signals.append((path, fs, pressure_uPa))

#         file_rows.append(
#             {
#                 "manovra": label,
#                 "microfono": mic,
#                 "file": path.name,
#                 "sample_rate_Hz": fs,
#                 "durata_s": duration,
#                 "SPL_RMS_dB_re_20uPa": spl,
#             }
#         )

#         log(f"[{label}] Welch completata per {path.name}.")

#     if frequency_reference is None or not spectra:
#         raise RuntimeError(f"[{label}] Nessuno spettro disponibile.")

#     # Media incrementale, senza vstack.
#     mean_psd = np.zeros_like(spectra[0])

#     for spectrum in spectra:
#         mean_psd += spectrum

#     mean_psd /= float(len(spectra))

#     if requested_base_frequency is None:
#         base_frequency_hz = estimate_comb_frequency(
#             frequency_reference,
#             mean_psd,
#             search_min_hz=80.0,
#             search_max_hz=110.0,
#             max_order=max_order,
#         )
#         log(
#             f"[{label}] Frequenza acustica di base stimata: "
#             f"{base_frequency_hz:.2f} Hz"
#         )
#     else:
#         base_frequency_hz = float(requested_base_frequency)
#         log(
#             f"[{label}] Frequenza acustica di base imposta: "
#             f"{base_frequency_hz:.2f} Hz"
#         )

#     # Salva subito le informazioni dei file.
#     write_rows_csv(
#         output_dir / f"{label.lower()}_file_info.csv",
#         file_rows,
#     )
#     log(f"[{label}] Salvato file_info.csv")

#     log(f"[{label}] Estrazione armoniche della media...")
#     average_rows = extract_harmonics_safe(
#         frequency_reference,
#         mean_psd,
#         base_frequency_hz,
#         max_order,
#         microphone="media",
#         manoeuvre=label,
#     )

#     write_rows_csv(
#         output_dir / f"{label.lower()}_harmonics_average.csv",
#         average_rows,
#     )
#     log(f"[{label}] Salvate armoniche medie.")

#     all_rows = list(average_rows)

#     for index, (path, spectrum) in enumerate(
#         zip(files, spectra),
#         start=1,
#     ):
#         mic = microphone_number(path)
#         log(
#             f"[{label}] Estrazione armoniche microfono "
#             f"{mic} ({index}/{len(files)})..."
#         )

#         microphone_rows = extract_harmonics_safe(
#             frequency_reference,
#             spectrum,
#             base_frequency_hz,
#             max_order,
#             microphone=mic,
#             manoeuvre=label,
#         )
#         all_rows.extend(microphone_rows)

#         # Checkpoint aggiornato a ogni microfono.
#         write_rows_csv(
#             output_dir
#             / f"{label.lower()}_harmonics_all_microphones.csv",
#             all_rows,
#         )

#     log(f"[{label}] Tutte le armoniche sono state estratte.")

#     slope, intercept, r_squared, fit_points = (
#         fit_harmonic_decay_safe(average_rows)
#     )

#     log(
#         f"[{label}] Fit completato: "
#         f"{slope:.3f} dB/ordine, R²={r_squared:.3f}"
#     )

#     write_spectrum_csv(
#         output_dir / f"{label.lower()}_mean_spectrum.csv",
#         frequency_reference,
#         mean_psd,
#     )
#     log(f"[{label}] Salvato spettro medio CSV.")

#     summary = {
#         "manovra": label,
#         "numero_file": len(files),
#         "frequenza_base_acustica_Hz": base_frequency_hz,
#         "pendenza_decadimento_dB_per_ordine": slope,
#         "intercetta_fit_dB": intercept,
#         "R2_fit_lineare_in_dB": r_squared,
#         "numero_punti_fit": fit_points,
#         "SPL_medio_microfoni_dB_re_20uPa": (
#             sum(
#                 float(row["SPL_RMS_dB_re_20uPa"])
#                 for row in file_rows
#             )
#             / len(file_rows)
#         ),
#     }

#     write_rows_csv(
#         output_dir / f"{label.lower()}_summary.csv",
#         [summary],
#     )
#     log(f"[{label}] Salvato summary.csv.")

#     log(f"[{label}] Creazione dello spettro medio PNG...")
#     save_spectrum_plot(
#         frequency_reference,
#         spectra,
#         labels,
#         mean_psd,
#         base_frequency_hz,
#         max_order,
#         title=(
#             f"DJI Matrice 300 RTK – {label} – "
#             f"spettro medio Welch"
#         ),
#         output_path=(
#             output_dir / f"{label.lower()}_mean_spectrum.png"
#         ),
#     )
#     log(f"[{label}] Salvato spettro medio PNG.")

#     log(f"[{label}] Creazione del grafico di decadimento...")
#     save_decay_plot(
#         average_rows,
#         slope,
#         intercept,
#         r_squared,
#         title=(
#             f"DJI Matrice 300 RTK – {label} – "
#             f"decadimento armonico"
#         ),
#         output_path=(
#             output_dir / f"{label.lower()}_harmonic_decay.png"
#         ),
#     )
#     log(f"[{label}] Salvato grafico di decadimento.")

#     if label.lower() == "lateral":
#         for index, (path, fs, pressure_uPa) in enumerate(
#             loaded_signals,
#             start=1,
#         ):
#             mic = microphone_number(path)

#             log(
#                 f"[{label}] Spettrogramma microfono {mic} "
#                 f"({index}/{len(loaded_signals)})..."
#             )

#             save_spectrogram(
#                 pressure_uPa,
#                 fs,
#                 title=(
#                     f"DJI Matrice 300 RTK – Lateral – "
#                     f"microfono {mic}"
#                 ),
#                 output_path=(
#                     output_dir
#                     / f"lateral_mic{mic}_spectrogram.png"
#                 ),
#             )

#     log(f"[{label}] Analisi completata.")
#     return summary

def plot_all_welch(directory, nperseg=None):

    wav_files = sorted(directory.glob("*.wav"))

    plt.figure(figsize=(10,6))

    for wav in wav_files:

        fs, signal = read_neaptide_wav(wav)

        print(f"Duration = {len(signal)/fs:.1f} s")

        duration = len(signal) / fs

        t_start = 1.0 #duration/2 - 1
        t_end   = 9.0 # duration/2 + 1

        signal = signal[int(t_start*fs):int(t_end*fs)]

        if nperseg is None:
            nperseg = fs

        f, psd = welch_psd_numpy(signal, fs, nperseg)

        spl = 10*np.log10(psd / 20.0**2)

        plt.plot(f, spl, label=wav.stem)
        plt.xscale("log")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Level [dB re 20 µPa²/Hz]")
    plt.grid(True)
    plt.legend()

    plt.show()

    # fig, axes = plt.subplots(
    #     len(wav_files),
    #     1,
    #     figsize=(12, 2.8*len(wav_files)),
    #     sharex=True,
    #     constrained_layout=True,
    # )

    fig, axes = plt.subplots(
        1,
        len(wav_files),
        figsize=(3*len(wav_files), 5),
        sharey=True,
    )

    # Handle the case of only one file
    if len(wav_files) == 1:
        axes = [axes]

    for ax, wav in zip(axes, wav_files):

        fs, signal = read_neaptide_wav(wav)

        print(f"{wav.name}: Duration = {len(signal)/fs:.1f} s")

        t_start = 0.0
        t_end = 10.0

        signal = signal[int(t_start*fs):int(t_end*fs)]

        f, t, Sxx = scipy_signal.spectrogram(
            signal,
            fs,
            window="hann",
            nperseg=4096,
            noverlap=3072,
            scaling="density",
        )

        # im = ax.pcolormesh(
        #     t,
        #     f,
        #     10*np.log10(np.maximum(Sxx, 1e-30)),
        #     shading="gouraud",
        #     vmin=-80,
        #     vmax=-20,
        # )
        im = ax.pcolormesh(
            t,
            f,
            10*np.log10(np.maximum(Sxx, 1e-30)),
            shading="gouraud",
            vmin=+20,
            vmax=+100,
        )

        ax.set_yscale("log")
        ax.set_ylim(20, 20000)

        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(wav.stem)

    axes[-1].set_xlabel("Time [s]")

    fig.colorbar(im, ax=axes, label="PSD [dB/Hz]")

    # plt.tight_layout()
    plt.show()

def compute_spl_from_wav(wav, fmin=100, fmax=20000):

    fs, pressure = read_neaptide_wav(wav)

    # Optional: remove transient part
    t_start = 1.0
    t_end = 9.0

    pressure = pressure[
        int(t_start*fs):
        int(t_end*fs)
    ]

    freq, psd = welch_psd_numpy(
        pressure,
        fs,
        nperseg=fs
    )

    # integrate PSD over frequency band
    mask = (freq >= fmin) & (freq <= fmax)

    p_rms_squared = np.trapezoid(
        psd[mask],
        freq[mask]
    )

    pref = 20.0 # µPa

    SPL = 10*np.log10(
        p_rms_squared / pref**2
    )

    return SPL    

def harmonic_directivity(X, *params):
    theta, freq = X
    # params packed as: A, a1,b1,a2,b2,a3,b3  -- one full set per harmonic
    params = np.array(params).reshape(n_harm, 7)

    idx = np.array([harmonic_index[f] for f in freq])
    A, a1, b1, a2, b2, a3, b3 = params[idx].T

    return (
        A
        + a1*np.cos(theta) + b1*np.sin(theta)
        + a2*np.cos(2*theta) + b2*np.sin(2*theta)
        + a3*np.cos(3*theta) + b3*np.sin(3*theta)
    )

# def harmonic_directivity(X, A,
#                           a10, a11,
#                           a20, a21,
#                           a30, a31):

#     theta, freq = X

#     x = np.log(freq / f_ref)

#     a1 = a10 + a11*x
#     a2 = a20 + a21*x
#     a3 = a30 + a31*x

#     return (
#         A
#         + a1*np.cos(theta)
#         + a2*np.cos(2*theta)
#         + a3*np.cos(3*theta)
#     )

# def harmonic_directivity(theta, A, a1, a2, a3):

#     return (
#         A
#         + a1*np.cos(theta)
#         + a2*np.cos(2*theta)
#         + a3*np.cos(3*theta)
#     )

def integrate_band(freq, psd, f_low, f_high):

    mask = (
        (freq >= f_low)
        &
        (freq <= f_high)
    )

    power = np.trapz(
        psd[mask],
        freq[mask]
    )

    return 10*np.log10(
        power / 20.0**2
    )



def plot_dashboard(pa, title):
    active_keys = [k for k in acoustic_metadata.keys() if k in pa]
    num_plots = len(active_keys)

    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 2.2 * num_plots), sharex=True)
    if num_plots == 1:
        axs = [axs]

    for idx, key in enumerate(active_keys):
        signal = np.asarray(pa[key], dtype=float)
        t_acoustic = pa["t_center"]
        is_valid = ~np.isnan(signal)
        meta = acoustic_metadata[key]
        if np.any(is_valid):
            axs[idx].plot(t_acoustic[is_valid], signal[is_valid],
                          color=meta['color'], linewidth=2, label=meta['label'])
        else:
            axs[idx].text(0.5, 0.5, 'Metric Data Unavailable',
                          transform=axs[idx].transAxes, ha='center', va='center',
                          color='gray', fontstyle='italic')
        axs[idx].set_ylabel(meta['label'], fontsize=10, fontweight='bold')
        axs[idx].grid(True, linestyle=':', alpha=0.5)

    axs[-1].set_xlabel('Time [seconds]', fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.show()

import numpy as np

def extract_harmonic_peaks(
    welch_result: dict,
    f0_approx: float,
    num_harmonics: int = 5,
    search_margin_pct: float = 0.05,  # +/- 5% search window around k * f0
    integration_bins: int = 1         # Bins on each side of peak to sum (1 = 3 bins total)
) -> dict:
    """
    Extracts the peak Sound Pressure Levels (SPL) for the first M harmonics of f0.
    
    Parameters:
    -----------
    welch_result : dict
        Output from welch_psd_numpy containing 'freq', 'psd', 'df', 'pref'.
    f0_approx : float
        Approximate fundamental frequency (e.g., Blade Passage Frequency in Hz).
    num_harmonics : int
        Number of harmonics to extract (1 to M).
    search_margin_pct : float
        Relative window width to account for f0 uncertainty (e.g., 0.05 = +/-5%).
    integration_bins : int
        Number of neighboring bins to sum energy around the peak to account for leakage.
        
    Returns:
    --------
    dict containing harmonic orders, exact frequencies, peak SPLs, and relative dB levels.
    """
    freq = welch_result["freq"]
    psd = welch_result["psd"]
    df = welch_result["df"]
    pref = welch_result.get("pref", 20e-6)
    
    harmonics_info = []
    
    for k in range(1, num_harmonics + 1):
        target_f = k * f0_approx
        
        # 1. Define search window limits [Hz]
        f_min = target_f * (1.0 - search_margin_pct)
        f_max = target_f * (1.0 + search_margin_pct)
        
        # Mask frequency range
        window_mask = (freq >= f_min) & (freq <= f_max)
        
        if not np.any(window_mask):
            continue
            
        # Indices corresponding to search window
        window_indices = np.where(window_mask)[0]
        
        # 2. Find exact peak index inside window
        peak_in_window = np.argmax(psd[window_indices])
        exact_peak_idx = window_indices[peak_in_window]
        exact_freq = freq[exact_peak_idx]
        
        # 3. Integrate power over neighboring bins (to handle spectral leakage)
        idx_start = max(0, exact_peak_idx - integration_bins)
        idx_end = min(len(psd), exact_peak_idx + integration_bins + 1)
        
        integrated_power = np.sum(psd[idx_start:idx_end]) * df
        
        # Convert integrated harmonic power to SPL [dB]
        spl_peak_db = 10.0 * np.log10(np.maximum(integrated_power, 1e-30) / (pref**2))
        
        harmonics_info.append({
            "harmonic_order": k,
            "target_freq_hz": target_f,
            "exact_freq_hz": exact_freq,
            "spl_db": spl_peak_db,
            "peak_index": exact_peak_idx
        })
        
    # Extract arrays
    orders = np.array([h["harmonic_order"] for h in harmonics_info])
    exact_freqs = np.array([h["exact_freq_hz"] for h in harmonics_info])
    spl_levels = np.array([h["spl_db"] for h in harmonics_info])
    
    # 4. Compute relative SPL levels with respect to the fundamental (k=1)
    if len(spl_levels) > 0:
        f0_spl = spl_levels[0]
        relative_spl_db = spl_levels - f0_spl  # dB relative to 1st harmonic (f0 sits at 0 dB)
    else:
        relative_spl_db = np.array([])
        
    return {
        "harmonic_orders": orders,
        "exact_freqs": exact_freqs,
        "spl_db": spl_levels,
        "relative_spl_db": relative_spl_db,  # Relative to f0 peak [dB]
        "raw_details": harmonics_info
    }

Dr = 7.755 
d = 2.585
Hr = 7.755
hs = 0.375
hi = 3.878

# ----------------------------------------------------
# Per-signal dashboard (reusing your plotting logic)
# ----------------------------------------------------
acoustic_metadata = {
    'loudness_sone':     {'label': 'Loudness [Sone]',      'color': '#2980B9'},
    'sharpness_acum':    {'label': 'Sharpness [Acum]',     'color': '#8E44AD'},
    'roughness_asper':   {'label': 'Roughness [Asper]',    'color': '#16A085'},
    'fluctuation_vacil': {'label': 'Fluctuation [Vacil]',  'color': '#F39C12'},
    'annoyance_PA':      {'label': 'Total Annoyance [PA]', 'color': '#D35400'}
}

x_coord = np.array([
    3*d,
    2*d,
    d,
    0.,
    0.,
    0.,
    0.,
])
   
y_coord = np.array([
    0.,
    0.,
    0.,
    0.,
    hi,
    Hr,
    Hr + hs,
])

x_drone = 3*d
y_drone = Hr 

theta_rad = np.arctan2(- x_coord + x_drone, - y_coord + y_drone)
theta_deg = np.degrees(theta_rad)

print(theta_deg)

# plot_all_welch(DEFAULT_INPUT_DIR)

wav_files = sorted(DEFAULT_INPUT_DIR.glob("*.wav"))

RPM = 5000 
n_blades = 1
BPF = 92.5 #RPM/60*n_blades
f_ref = BPF * 1.0 
# harmonics = [
#     BPF,
#     2*BPF,
#     3*BPF,
# ]

# Define your reference frequencies - e.g. multiples of BPF
# harmonics = BPF * [1, 2, 3, 4, 5]          # multiples of the base BPF you want (5th, ...)
harmonic_numbers = list(range(1, 17))  # 1st through 10th
harmonics = BPF * np.array(harmonic_numbers)

n_harm = len(harmonics)
harmonic_index = {h: i for i, h in enumerate(harmonics)}

BPF_M_list = [BPF * m for m in harmonics]   # e.g. [475, ...]
n_f_ref = len(BPF_M_list)

half_width = 12.5  # Hz, the +/- 5 you had

from zwicker_annoyance_v3 import compute_zwicker_indicators_windowed

spectra = {}
calibration_factor = 5.e-8
num_harmonics = 15
search_margin_pct = 0.06
f0_guess = 92.0
harmonic_numbers = list(range(1, num_harmonics+1)) 

all_microphone_harmonics = {}

for wav in wav_files:

    fs, pressure = read_neaptide_wav(wav)

    print(f"{wav.name}: fs={fs} Hz, duration={len(pressure)/fs:.2f} s")

    # Optional: select steady-state part
    t_start = 0.5
    t_end = 9.5

    pressure = calibration_factor * pressure[
        int(t_start*fs):
        int(t_end*fs)
    ]

    # Welch PSD
    nperseg = fs  # 1 second segments

    result = welch_psd_numpy(
        pressure,
        fs,
        nperseg
    )

        # "freq": frequency,
        # "df": df,
        # "psd": mean_psd,
        # "psd_db_hz": psd_db_hz,
        # "spl_bin": spl_bin,
        # "overall_level_db": overall_level_db,
        # # 1/3-Octave outputs
        # "freq_third_oct": f_center_third,
        # "spl_third_oct": spl_third,

    freq, df, psd, psd_db_hz, spl_bin, overall_level_db, freq_third_oct, spl_third_oct = result["freq"], result["df"], result["psd"], result["psd_db_hz"], result["spl_bin"], result["overall_level_db"], result["freq_third_oct"], result["spl_third_oct"]

    print(f"Sampling frequency (fs): {fs} Hz")
    print(f"Segment size (nperseg): {nperseg}")
    print(f"Bin width (df): {result['df']} Hz")

    spectra[wav.stem] = {
        "fs": fs,
        "freq": freq,
        "df": df,
        "psd": psd,               # Linear PSD [Pa^2 / Hz]
        "psd_db_hz": psd_db_hz,         # Spectral density level [dB/Hz]
        "spl_bin": spl_bin,             # Narrowband bin SPL [dB]
        "overall_level_db": overall_level_db,
        "freq_third_oct": freq_third_oct, 
        "spl_third_oct": spl_third_oct,
    }

    # --- Psychoacoustic (Zwicker) indicators, computed on the same
    #     steady-state time-domain segment used for the PSD above ---
    # pa = compute_zwicker_indicators_windowed(
    #     pressure, fs,
    #     window_s=1.0, hop_s=0.25,
    #     stationary=True,
    #     use_fs_approximation=True,
    # )

    # plot_dashboard(pa, title=wav.stem)
 
    mic_id = wav.stem 
        
    # 3. Extract Harmonic Peaks for this specific microphone
    harmonics = extract_harmonic_peaks(
        welch_result=result,
        f0_approx=f0_guess,
        num_harmonics=num_harmonics,
        search_margin_pct=search_margin_pct,
        integration_bins=1
    )
    
    # 4. Save to master dictionary
    all_microphone_harmonics[mic_id] = {
        "wav_path": str(wav),
        "fs": fs,
        "overall_spl_db": result["overall_level_db"],
        "harmonic_orders": harmonics["harmonic_orders"].tolist(), # Convert to list for easy export
        "exact_freqs": harmonics["exact_freqs"].tolist(),
        "spl_db": harmonics["spl_db"].tolist(),
        "relative_spl_db": harmonics["relative_spl_db"].tolist()
    }

    # # 2. Extract first 6 harmonics for an estimated BPF of 185 Hz
    # f0_guess = 92.0  # Approx fundamental frequency (Hz)
    # harmonics = extract_harmonic_peaks(
    #     welch_result=result,
    #     f0_approx=f0_guess,
    #     num_harmonics=15,
    #     search_margin_pct=0.06, # Window of +/- 6%
    #     integration_bins=1      # Integrates 3 bins (peak +/- 1)
    # )

    # 3. Print the results table
    print(f"{'Harmonic':<10} | {'Freq (Hz)':<10} | {'SPL (dB)':<10} | {'Rel to f0 (dB)':<15}")
    print("-" * 55)
    for k, f_hz, spl, rel_spl in zip(
        harmonics["harmonic_orders"], 
        harmonics["exact_freqs"], 
        harmonics["spl_db"], 
        harmonics["relative_spl_db"]
    ):
        print(f"{k:<10} | {f_hz:<10.1f} | {spl:<10.2f} | {rel_spl:<15.2f}")



plt.figure(figsize=(10,5))

for name, data in spectra.items():

    plt.plot(
        data["freq"],
        data["psd_db_hz"],
        # data["freq_third_oct"],
        # data["spl_third_oct"],
        label=name
    )

    # plt.plot(
    #     data["freq"],
    #     data["spl_bin"],"o",
    #     # data["freq_third_oct"],
    #     # data["spl_third_oct"],
    #     label=name
    # )

plt.xscale("log")
plt.xlim(20,20000)

plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD level [dB re 20 µPa²/Hz]")
plt.ylabel("SPL level [dB]")

plt.grid(True, which="both")
plt.legend()

plt.show()

# save data 
import pandas as pd
rows = []

for wav, theta in zip(wav_files, theta_rad):
    mic_id = wav.stem
    data = all_microphone_harmonics[mic_id]
    
    for k, freq, spl, rel_spl in zip(
        data["harmonic_orders"],
        data["exact_freqs"],
        data["spl_db"],
        data["relative_spl_db"]
    ):
        rows.append({
            "mic_id": mic_id,
            "theta_rad": theta,  # <--- ADD THIS LINE HERE
            "overall_spl_db": data["overall_spl_db"],
            "harmonic_order": k,
            "exact_freq_hz": freq,
            "spl_db": spl,
            "relative_spl_db": rel_spl
        })

df_harmonics = pd.DataFrame(rows)

# Save to CSV
df_harmonics.to_csv("drone_microphone_harmonics.csv", index=False)
print("Saved harmonics to drone_microphone_harmonics.csv")

# Interpolation 

from scipy.optimize import curve_fit

def directivity_single_freq(theta, A, a1, b1, a2, b2):
    """2nd-order Fourier series directivity model."""
    return (A + a1 * np.cos(theta) 
              + b1 * np.sin(theta) 
              + a2 * np.cos(2 * theta) 
              + b2 * np.sin(2 * theta))

# -------------------------------------------------------------
# 1. Map your mic IDs to theta_rad (if not already in df)
# -------------------------------------------------------------
# Example mapping (replace this with your actual theta_rad array):
# mic_to_theta = dict(zip(unique_mic_ids, theta_rad))
# df_harmonics["theta_rad"] = df_harmonics["mic_id"].map(mic_to_theta)

# Dictionary to store fitted parameters: {harmonic_order: [A, a1, b1, a2, b2]}
coeffs_per_harmonic = {}

# -------------------------------------------------------------
# 2. Fit directivity model per harmonic order
# -------------------------------------------------------------
# Unique harmonic orders (1, 2, ..., M)
unique_harmonics = sorted(df_harmonics["harmonic_order"].unique())

for h in unique_harmonics:
    # Filter DataFrame for the current harmonic order
    df_h = df_harmonics[df_harmonics["harmonic_order"] == h].dropna(subset=["spl_db", "theta_rad"])
    
    # Need at least 5 points to fit a 5-parameter model (A, a1, b1, a2, b2)
    if len(df_h) < 5:
        print(f"Warning: Harmonic {h} only has {len(df_h)} data points. Skipping curve fit.")
        continue
    
    theta_data = df_h["theta_rad"].values
    spl_data = df_h["spl_db"].values
    
    # Initial guess for p0:
    # A_0 = mean SPL for this harmonic, rest set to 0
    p0 = [np.mean(spl_data), 0.0, 0.0, 0.0, 0.0]
    
    try:
        popt_h, pcov_h = curve_fit(
            directivity_single_freq,
            theta_data,
            spl_data,
            p0=p0
        )
        
        coeffs_per_harmonic[h] = popt_h
        print(f"Harmonic {h:2d} | Fit successful: A={popt_h[0]:.2f} dB, a1={popt_h[1]:.2f}, b1={popt_h[2]:.2f}")

    except RuntimeError:
        print(f"Error: Fit failed to converge for Harmonic {h}.")

theta_dense = np.linspace(0, 0.6 * np.pi, 360)

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})

# Plot the first 3 harmonics as an example
for h in [1, 2, 3, 4, 5, 6, 7, 8]:
    if h not in coeffs_per_harmonic:
        continue
        
    popt = coeffs_per_harmonic[h]
    
    # 1. Filter original measured points for this harmonic
    df_h = df_harmonics[df_harmonics["harmonic_order"] == h]
    
    # 2. Evaluate continuous fit
    spl_fit = directivity_single_freq(theta_dense, *popt)
    
    # 3. Plot measured markers
    lines = ax.plot(df_h["theta_rad"], df_h["spl_db"], 'o', label=f'H{h} Data')
    color = lines[0].get_color()
    
    # 4. Plot smooth fitted curve (matching color)
    ax.plot(theta_dense, spl_fit, '-', color=color, linewidth=2, label=f'H{h} Fit (A={popt[0]:.1f} dB)')

ax.set_theta_zero_location("N")  # 0 deg at top
ax.set_theta_direction(-1)       # Clockwise
ax.set_title("Directivity Fits per Harmonic Order", pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.grid(True)
plt.show()

from scipy.interpolate import PchipInterpolator

# 1. Sort the extracted harmonic orders (1, 2, 3, ...)
harmonics_sorted = np.array(sorted(coeffs_per_harmonic.keys()))  # e.g., array([1, 2, 3, 4, 5, 6])

# 2. Extract matrix of fitted coefficients (shape: [num_harmonics, 5])
coeffs_array = np.array([coeffs_per_harmonic[h] for h in harmonics_sorted])

# 3. Create continuous PchipInterpolator per coefficient as a function of HARMONIC NUMBER
coeff_names = ["A", "a1", "b1", "a2", "b2"]

smooth_coeffs_h = {
    name: PchipInterpolator(harmonics_sorted, coeffs_array[:, k])
    for k, name in enumerate(coeff_names)
}

# 4. Continuous Directivity Model as a function of (theta, harmonic_number)
def harmonic_directivity_smooth_h(theta, h_num):
    """
    Evaluates directivity at angle theta (rad) for harmonic order h_num (can be float or array).
    """
    A  = smooth_coeffs_h["A"](h_num)
    a1 = smooth_coeffs_h["a1"](h_num)
    b1 = smooth_coeffs_h["b1"](h_num)
    a2 = smooth_coeffs_h["a2"](h_num)
    b2 = smooth_coeffs_h["b2"](h_num)
    
    return (A + a1 * np.cos(theta) + b1 * np.sin(theta)
              + a2 * np.cos(2 * theta) + b2 * np.sin(2 * theta))


# Define dense grid of theta (0 to 2pi) and continuous harmonic numbers (1 to max_h)
theta_grid = np.linspace(0, 0.6 * np.pi, 360)
h_grid = np.linspace(harmonics_sorted.min(), harmonics_sorted.max(), 200)

# Create 2D meshgrid
THETA, H = np.meshgrid(theta_grid, h_grid)

# Compute smooth SPL surface across (Harmonic Number vs Theta)
SPL_surface = harmonic_directivity_smooth_h(THETA, H)

# --- 2D Polar/Cartesian Contour Plot ---
plt.figure(figsize=(10, 6))
cp = plt.contourf(H, np.degrees(THETA), SPL_surface, levels=50, cmap="viridis")
plt.colorbar(cp, label="SPL [dB]")
plt.xlabel("Harmonic Number ($h$)")
plt.ylabel("Angle $\\theta$ [degrees]")
plt.title("Smooth Directivity Surface Across Harmonic Numbers")
plt.grid(True, alpha=0.3)
plt.show()

# save coeffs 
output_path = DEFAULT_OUTPUT_DIR / "directivity_model_pchip.npz"

np.savez(
    output_path,
    harmonic_numbers=harmonic_numbers,
    coeffs=coeffs_array,
    coeff_names=np.array(coeff_names)
)

print(f"Saved directivity model to {output_path}")