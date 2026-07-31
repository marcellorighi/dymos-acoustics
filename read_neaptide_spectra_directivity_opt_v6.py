#!/usr/bin/env python3
"""
Analisi armonica robusta dei WAV NEAPTIDE – DJI Matrice 300 RTK.

Versione "safe":
- evita pandas;
- evita scipy.signal;
- evita numpy.polyfit;
- estrae i picchi usando intervalli contigui;
- salva CSV con il modulo standard csv;
- stampa un checkpoint dopo ogni passaggio;
- conserva i risultati parziali anche se una fase successiva fallisce.
"""

from __future__ import annotations

import os

# Impostare i limiti PRIMA di importare NumPy/Matplotlib.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
import math
import re
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any

# import matplotlib

# matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal as scipy_signal


P0_UPA = 20.0

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
    / "DJIMatrice300RTK" 
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
    / "DJIMatrice300RTK" 
)


def log(message: str) -> None:
    print(message, flush=True)


def microphone_number(path: Path) -> int:
    match = re.search(r"_(\d+)\.wav$", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


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


def overall_spl_db(pressure_uPa: np.ndarray) -> float:
    mean_square = float(np.mean(pressure_uPa * pressure_uPa))
    if mean_square <= 0.0:
        return float("-inf")

    rms = math.sqrt(mean_square)
    return 20.0 * math.log10(rms / P0_UPA)

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

    window = np.hanning(nperseg)
    window_power = float(np.sum(window * window))

    starts = range(0, len(x) - nperseg + 1, step)
    psd_sum = np.zeros(nfft // 2 + 1, dtype=np.float64)
    segment_count = 0

    for start in starts:
        segment = np.array(x[start : start + nperseg], dtype=np.float64, copy=True)
        segment -= float(np.mean(segment))
        segment *= window

        spectrum = np.fft.rfft(segment, n=nfft)
        real_part = spectrum.real
        imag_part = spectrum.imag
        psd = (real_part * real_part + imag_part * imag_part) / (fs * window_power)

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

    # --- PSD level, dB re pref^2/Hz (per-bin spectral density level,
    #     NOT a true SPL -- see note below) ---
    spl = 10 * np.log10(np.maximum(mean_psd, psd_floor) / pref**2)

    # --- Overall broadband sound level, dB re pref, from integrating
    #     the PSD across the full frequency range analyzed ---
    p_meansq = np.trapezoid(mean_psd, frequency)
    overall_level_db = 10 * np.log10(max(p_meansq, psd_floor) / pref**2)

    return {
        "freq": frequency,
        "psd": mean_psd,
        "spl": spl,
        "overall_level_db": overall_level_db,
    }

# def welch_psd_numpy(
#     signal: np.ndarray,
#     fs: int,
#     nperseg: int,
#     overlap_fraction: float = 0.75,
# ) -> tuple[np.ndarray, np.ndarray]:
#     x = np.asarray(signal, dtype=np.float64)

#     if x.ndim != 1:
#         raise ValueError("Il segnale deve essere monodimensionale.")

#     nperseg = min(max(256, int(nperseg)), len(x))
#     noverlap = min(
#         max(0, int(round(overlap_fraction * nperseg))),
#         nperseg - 1,
#     )
#     step = nperseg - noverlap
#     nfft = nperseg

#     window = np.hanning(nperseg)
#     window_power = float(np.sum(window * window))

#     starts = range(0, len(x) - nperseg + 1, step)
#     psd_sum = np.zeros(nfft // 2 + 1, dtype=np.float64)
#     segment_count = 0

#     for start in starts:
#         segment = np.array(x[start : start + nperseg], dtype=np.float64, copy=True)
#         segment -= float(np.mean(segment))
#         segment *= window

#         spectrum = np.fft.rfft(segment, n=nfft)
#         real_part = spectrum.real
#         imag_part = spectrum.imag
#         psd = (real_part * real_part + imag_part * imag_part) / (
#             fs * window_power
#         )

#         if nfft % 2 == 0:
#             psd[1:-1] *= 2.0
#         else:
#             psd[1:] *= 2.0

#         psd_sum += psd
#         segment_count += 1

#     if segment_count == 0:
#         raise RuntimeError("Nessun segmento disponibile per Welch.")

#     mean_psd = psd_sum / float(segment_count)
#     frequency = np.fft.rfftfreq(nfft, d=1.0 / fs)

#     return frequency, mean_psd


def stft_psd_numpy(
    signal: np.ndarray,
    fs: int,
    nperseg: int = 4096,
    overlap_fraction: float = 0.75,
    nfft: int = 16384,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=np.float64)
    nperseg = min(max(256, int(nperseg)), len(x))
    nfft = max(int(nfft), nperseg)

    noverlap = min(
        max(0, int(round(overlap_fraction * nperseg))),
        nperseg - 1,
    )
    step = nperseg - noverlap

    window = np.hanning(nperseg)
    window_power = float(np.sum(window * window))

    starts = list(range(0, len(x) - nperseg + 1, step))
    if not starts:
        starts = [0]

    matrix = np.empty((nfft // 2 + 1, len(starts)), dtype=np.float64)
    times = np.empty(len(starts), dtype=np.float64)

    for column, start in enumerate(starts):
        segment = np.array(x[start : start + nperseg], dtype=np.float64, copy=True)

        if len(segment) < nperseg:
            padded = np.zeros(nperseg, dtype=np.float64)
            padded[: len(segment)] = segment
            segment = padded

        segment -= float(np.mean(segment))
        segment *= window

        spectrum = np.fft.rfft(segment, n=nfft)
        psd = (
            spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        ) / (fs * window_power)

        if nfft % 2 == 0:
            psd[1:-1] *= 2.0
        else:
            psd[1:] *= 2.0

        matrix[:, column] = psd
        times[column] = (start + nperseg / 2.0) / fs

    frequency = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return frequency, times, matrix


def psd_level_db(psd: np.ndarray) -> np.ndarray:
    safe_psd = np.maximum(np.asarray(psd, dtype=np.float64), 1e-30)
    return 10.0 * np.log10(safe_psd / (P0_UPA * P0_UPA))


def estimate_comb_frequency(
    frequency: np.ndarray,
    psd: np.ndarray,
    search_min_hz: float = 80.0,
    search_max_hz: float = 110.0,
    max_order: int = 8,
) -> float:
    best_candidate = search_min_hz
    best_score = float("-inf")

    candidate = search_min_hz
    while candidate <= search_max_hz + 1e-12:
        score = 0.0

        for order in range(1, max_order + 1):
            expected = order * candidate
            half_width = max(2.5, 0.015 * expected)

            left = int(np.searchsorted(frequency, expected - half_width, side="left"))
            right = int(np.searchsorted(frequency, expected + half_width, side="right"))

            if right > left:
                local_max = float(np.max(psd[left:right]))
                score += local_max / math.sqrt(order)

        if score > best_score:
            best_score = score
            best_candidate = candidate

        candidate += 0.05

    return float(best_candidate)


def safe_median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")

    # statistics.median evita il percorso interno di np.median.
    return float(statistics.median(values.astype(float).tolist()))


def extract_harmonics_safe(
    frequency: np.ndarray,
    psd: np.ndarray,
    base_frequency_hz: float,
    max_order: int,
    microphone: int | str,
    manoeuvre: str,
) -> list[dict[str, Any]]:
    """
    Estrae un massimo in ciascuna banda armonica.

    Restituisce una lista di dizionari e non crea DataFrame.
    """
    levels = psd_level_db(psd)
    rows: list[dict[str, Any]] = []

    for order in range(1, max_order + 1):
        log(
            f"[{manoeuvre}] armoniche – mic {microphone}, "
            f"ordine {order}/{max_order}"
        )

        expected = float(order * base_frequency_hz)
        half_width = float(max(5.0, 0.045 * expected))

        left = int(
            np.searchsorted(
                frequency,
                expected - half_width,
                side="left",
            )
        )
        right = int(
            np.searchsorted(
                frequency,
                expected + half_width,
                side="right",
            )
        )

        if right <= left:
            continue

        local_levels = levels[left:right]
        local_peak_offset = int(np.argmax(local_levels))
        peak_index = left + local_peak_offset

        measured = float(frequency[peak_index])
        peak_level = float(levels[peak_index])

        broad_half_width = 1.8 * half_width
        broad_left = int(
            np.searchsorted(
                frequency,
                expected - broad_half_width,
                side="left",
            )
        )
        broad_right = int(
            np.searchsorted(
                frequency,
                expected + broad_half_width,
                side="right",
            )
        )

        exclusion_width = float(max(3.0, 0.012 * expected))
        exclusion_left = int(
            np.searchsorted(
                frequency,
                measured - exclusion_width,
                side="left",
            )
        )
        exclusion_right = int(
            np.searchsorted(
                frequency,
                measured + exclusion_width,
                side="right",
            )
        )

        floor_parts: list[np.ndarray] = []

        if exclusion_left > broad_left:
            floor_parts.append(levels[broad_left:exclusion_left])

        if broad_right > exclusion_right:
            floor_parts.append(levels[exclusion_right:broad_right])

        if floor_parts:
            floor_values = np.concatenate(floor_parts)
            local_floor = safe_median(floor_values)
        else:
            local_floor = float("nan")

        prominence = (
            peak_level - local_floor
            if math.isfinite(local_floor)
            else float("nan")
        )

        rows.append(
            {
                "manovra": manoeuvre,
                "microfono": microphone,
                "ordine": order,
                "frequenza_base_Hz": base_frequency_hz,
                "frequenza_attesa_Hz": expected,
                "frequenza_picco_Hz": measured,
                "errore_Hz": measured - expected,
                "livello_picco_dB_re_20uPa2_Hz": peak_level,
                "fondo_locale_dB_re_20uPa2_Hz": local_floor,
                "prominenza_tonale_dB": prominence,
            }
        )

    return rows


def fit_harmonic_decay_safe(
    rows: list[dict[str, Any]],
    minimum_prominence_db: float = 6.0,
) -> tuple[float, float, float, int]:
    valid = [
        row
        for row in rows
        if math.isfinite(float(row["livello_picco_dB_re_20uPa2_Hz"]))
        and math.isfinite(float(row["prominenza_tonale_dB"]))
        and float(row["prominenza_tonale_dB"]) >= minimum_prominence_db
    ]

    if len(valid) < 3:
        valid = [
            row
            for row in rows
            if math.isfinite(float(row["livello_picco_dB_re_20uPa2_Hz"]))
        ]

    if len(valid) < 2:
        return float("nan"), float("nan"), float("nan"), len(valid)

    x = [float(row["ordine"]) for row in valid]
    y = [float(row["livello_picco_dB_re_20uPa2_Hz"]) for row in valid]

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    denominator = sum((value - x_mean) ** 2 for value in x)

    if denominator <= 0.0:
        return float("nan"), float("nan"), float("nan"), len(valid)

    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    )

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    predictions = [intercept + slope * value for value in x]
    residual_sum = sum(
        (observed - predicted) ** 2
        for observed, predicted in zip(y, predictions)
    )
    total_sum = sum((observed - y_mean) ** 2 for observed in y)

    r_squared = (
        1.0 - residual_sum / total_sum
        if total_sum > 0.0
        else float("nan")
    )

    return slope, intercept, r_squared, len(valid)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_spectrum_csv(
    path: Path,
    frequency: np.ndarray,
    mean_psd: np.ndarray,
) -> None:
    levels = psd_level_db(mean_psd)

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequenza_Hz",
                "PSD_media_uPa2_Hz",
                "livello_PSD_medio_dB_re_20uPa2_Hz",
            ]
        )

        for frequency_value, psd_value, level_value in zip(
            frequency,
            mean_psd,
            levels,
        ):
            writer.writerow(
                [
                    float(frequency_value),
                    float(psd_value),
                    float(level_value),
                ]
            )


def save_spectrum_plot(
    frequency: np.ndarray,
    individual_psd: list[np.ndarray],
    labels: list[str],
    mean_psd: np.ndarray,
    base_frequency_hz: float,
    max_order: int,
    title: str,
    output_path: Path,
    max_frequency_hz: float = 1500.0,
) -> None:
    left = int(np.searchsorted(frequency, 20.0, side="left"))
    right = int(
        np.searchsorted(
            frequency,
            max_frequency_hz,
            side="right",
        )
    )

    plt.figure(figsize=(11, 6))

    for spectrum, label in zip(individual_psd, labels):
        plt.plot(
            frequency[left:right],
            psd_level_db(spectrum[left:right]),
            linewidth=0.7,
            alpha=0.42,
            label=label,
        )

    plt.plot(
        frequency[left:right],
        psd_level_db(mean_psd[left:right]),
        linewidth=1.8,
        label="Media lineare dei microfoni",
    )

    for order in range(1, max_order + 1):
        harmonic = order * base_frequency_hz
        if harmonic > max_frequency_hz:
            break
        plt.axvline(harmonic, linewidth=0.7, alpha=0.28)

    plt.xlabel("Frequenza [Hz]")
    plt.ylabel("PSD [dB re 20 µPa²/Hz]")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_decay_plot(
    rows: list[dict[str, Any]],
    slope: float,
    intercept: float,
    r_squared: float,
    title: str,
    output_path: Path,
) -> None:
    orders = [float(row["ordine"]) for row in rows]
    levels = [
        float(row["livello_picco_dB_re_20uPa2_Hz"])
        for row in rows
    ]

    plt.figure(figsize=(8, 5.5))
    plt.scatter(orders, levels, label="Picchi misurati")

    if (
        orders
        and math.isfinite(slope)
        and math.isfinite(intercept)
    ):
        x_min = min(orders)
        x_max = max(orders)
        x_line = np.linspace(x_min, x_max, 100)
        y_line = intercept + slope * x_line

        plt.plot(
            x_line,
            y_line,
            label=(
                f"Fit: {slope:.2f} dB/ordine, "
                f"R²={r_squared:.3f}"
            ),
        )

    plt.xlabel("Ordine armonico")
    plt.ylabel("Livello del picco [dB re 20 µPa²/Hz]")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_spectrogram(
    pressure_uPa: np.ndarray,
    fs: int,
    title: str,
    output_path: Path,
    max_frequency_hz: float = 1500.0,
) -> None:
    frequency, time, psd_matrix = stft_psd_numpy(
        pressure_uPa,
        fs,
        nperseg=4096,
        overlap_fraction=0.75,
        nfft=16384,
    )

    left = int(np.searchsorted(frequency, 20.0, side="left"))
    right = int(
        np.searchsorted(
            frequency,
            max_frequency_hz,
            side="right",
        )
    )

    levels = psd_level_db(psd_matrix[left:right, :])

    vmax = float(np.percentile(levels, 99.5))
    vmin = vmax - 55.0

    plt.figure(figsize=(11, 6))
    mesh = plt.pcolormesh(
        time,
        frequency[left:right],
        levels,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(mesh, label="PSD [dB re 20 µPa²/Hz]")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Frequenza [Hz]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def analyse_group(
    files: list[Path],
    output_dir: Path,
    label: str,
    scale_factor: float,
    requested_base_frequency: float | None,
    max_order: int,
) -> dict[str, Any] | None:
    if not files:
        log(f"[{label}] Nessun file trovato.")
        return None

    log(f"[{label}] Funzione avviata con {len(files)} file.")

    files = sorted(files, key=microphone_number)
    spectra: list[np.ndarray] = []
    labels: list[str] = []
    file_rows: list[dict[str, Any]] = []
    loaded_signals: list[tuple[Path, int, np.ndarray]] = []
    frequency_reference: np.ndarray | None = None

    nperseg = 65536 if label.lower() == "hover" else 16384

    for index, path in enumerate(files, start=1):
        log(f"[{label}] {index}/{len(files)} – lettura {path.name}")

        fs, pressure_uPa = read_neaptide_wav(
            path,
            scale_factor=scale_factor,
        )

        duration = len(pressure_uPa) / fs
        spl = overall_spl_db(pressure_uPa)

        log(
            f"[{label}] fs={fs} Hz, durata={duration:.3f} s, "
            f"SPL={spl:.2f} dB. Calcolo Welch..."
        )

        frequency, psd = welch_psd_numpy(
            pressure_uPa,
            fs,
            nperseg=nperseg,
            overlap_fraction=0.75,
        )

        if frequency_reference is None:
            frequency_reference = frequency
        elif not np.array_equal(frequency_reference, frequency):
            psd = np.interp(frequency_reference, frequency, psd)

        spectra.append(psd)
        mic = microphone_number(path)
        labels.append(f"Mic {mic}")
        loaded_signals.append((path, fs, pressure_uPa))

        file_rows.append(
            {
                "manovra": label,
                "microfono": mic,
                "file": path.name,
                "sample_rate_Hz": fs,
                "durata_s": duration,
                "SPL_RMS_dB_re_20uPa": spl,
            }
        )

        log(f"[{label}] Welch completata per {path.name}.")

    if frequency_reference is None or not spectra:
        raise RuntimeError(f"[{label}] Nessuno spettro disponibile.")

    # Media incrementale, senza vstack.
    mean_psd = np.zeros_like(spectra[0])

    for spectrum in spectra:
        mean_psd += spectrum

    mean_psd /= float(len(spectra))

    if requested_base_frequency is None:
        base_frequency_hz = estimate_comb_frequency(
            frequency_reference,
            mean_psd,
            search_min_hz=80.0,
            search_max_hz=110.0,
            max_order=max_order,
        )
        log(
            f"[{label}] Frequenza acustica di base stimata: "
            f"{base_frequency_hz:.2f} Hz"
        )
    else:
        base_frequency_hz = float(requested_base_frequency)
        log(
            f"[{label}] Frequenza acustica di base imposta: "
            f"{base_frequency_hz:.2f} Hz"
        )

    # Salva subito le informazioni dei file.
    write_rows_csv(
        output_dir / f"{label.lower()}_file_info.csv",
        file_rows,
    )
    log(f"[{label}] Salvato file_info.csv")

    log(f"[{label}] Estrazione armoniche della media...")
    average_rows = extract_harmonics_safe(
        frequency_reference,
        mean_psd,
        base_frequency_hz,
        max_order,
        microphone="media",
        manoeuvre=label,
    )

    write_rows_csv(
        output_dir / f"{label.lower()}_harmonics_average.csv",
        average_rows,
    )
    log(f"[{label}] Salvate armoniche medie.")

    all_rows = list(average_rows)

    for index, (path, spectrum) in enumerate(
        zip(files, spectra),
        start=1,
    ):
        mic = microphone_number(path)
        log(
            f"[{label}] Estrazione armoniche microfono "
            f"{mic} ({index}/{len(files)})..."
        )

        microphone_rows = extract_harmonics_safe(
            frequency_reference,
            spectrum,
            base_frequency_hz,
            max_order,
            microphone=mic,
            manoeuvre=label,
        )
        all_rows.extend(microphone_rows)

        # Checkpoint aggiornato a ogni microfono.
        write_rows_csv(
            output_dir
            / f"{label.lower()}_harmonics_all_microphones.csv",
            all_rows,
        )

    log(f"[{label}] Tutte le armoniche sono state estratte.")

    slope, intercept, r_squared, fit_points = (
        fit_harmonic_decay_safe(average_rows)
    )

    log(
        f"[{label}] Fit completato: "
        f"{slope:.3f} dB/ordine, R²={r_squared:.3f}"
    )

    write_spectrum_csv(
        output_dir / f"{label.lower()}_mean_spectrum.csv",
        frequency_reference,
        mean_psd,
    )
    log(f"[{label}] Salvato spettro medio CSV.")

    summary = {
        "manovra": label,
        "numero_file": len(files),
        "frequenza_base_acustica_Hz": base_frequency_hz,
        "pendenza_decadimento_dB_per_ordine": slope,
        "intercetta_fit_dB": intercept,
        "R2_fit_lineare_in_dB": r_squared,
        "numero_punti_fit": fit_points,
        "SPL_medio_microfoni_dB_re_20uPa": (
            sum(
                float(row["SPL_RMS_dB_re_20uPa"])
                for row in file_rows
            )
            / len(file_rows)
        ),
    }

    write_rows_csv(
        output_dir / f"{label.lower()}_summary.csv",
        [summary],
    )
    log(f"[{label}] Salvato summary.csv.")

    log(f"[{label}] Creazione dello spettro medio PNG...")
    save_spectrum_plot(
        frequency_reference,
        spectra,
        labels,
        mean_psd,
        base_frequency_hz,
        max_order,
        title=(
            f"DJI Matrice 300 RTK – {label} – "
            f"spettro medio Welch"
        ),
        output_path=(
            output_dir / f"{label.lower()}_mean_spectrum.png"
        ),
    )
    log(f"[{label}] Salvato spettro medio PNG.")

    log(f"[{label}] Creazione del grafico di decadimento...")
    save_decay_plot(
        average_rows,
        slope,
        intercept,
        r_squared,
        title=(
            f"DJI Matrice 300 RTK – {label} – "
            f"decadimento armonico"
        ),
        output_path=(
            output_dir / f"{label.lower()}_harmonic_decay.png"
        ),
    )
    log(f"[{label}] Salvato grafico di decadimento.")

    if label.lower() == "lateral":
        for index, (path, fs, pressure_uPa) in enumerate(
            loaded_signals,
            start=1,
        ):
            mic = microphone_number(path)

            log(
                f"[{label}] Spettrogramma microfono {mic} "
                f"({index}/{len(loaded_signals)})..."
            )

            save_spectrogram(
                pressure_uPa,
                fs,
                title=(
                    f"DJI Matrice 300 RTK – Lateral – "
                    f"microfono {mic}"
                ),
                output_path=(
                    output_dir
                    / f"lateral_mic{mic}_spectrogram.png"
                ),
            )

    log(f"[{label}] Analisi completata.")
    return summary

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

    freq, psd, spl, overall_db = result["freq"], result["psd"], result["spl"], result["overall_level_db"]

    # Convert PSD to SPL spectrum
    pref = 20.0  # µPa

    # spl = 10*np.log10(
    #     np.maximum(psd, 1e-30) / pref**2
    # )

    spectra[wav.stem] = {
        "fs": fs,
        "freq": freq,
        "psd": psd,
        "spl": spl,
        "overall_db": overall_db,
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

plt.figure(figsize=(10,5))

for name, data in spectra.items():

    plt.plot(
        data["freq"],
        data["psd"],
        label=name
    )

plt.xscale("log")
plt.xlim(20,20000)

plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD level [dB re 20 µPa²/Hz]")
plt.ylabel("SPL level [dB]")

plt.grid(True, which="both")
plt.legend()

plt.show()


# find peaks --------------------------------------------------------------------------
from scipy.signal import find_peaks

for name, data in spectra.items():
    freq = data["freq"]
    spl = data["spl"]

    peaks, props = find_peaks(spl, prominence=3)  # tune prominence to your noise floor
    peak_freqs = freq[peaks]

    # Check which peaks are near your harmonics of interest
    for h_num in [1, 2, 3, 4, 5, 8]:
        target = BPF * h_num
        nearby = peak_freqs[np.abs(peak_freqs - target) < 5]
        if len(nearby) > 0:
            print(f"{name}: peak near {h_num}th harmonic ({target:.0f} Hz) at {nearby}")
# -------------------------------------------------------------------------------------

BPF_SPL = []

for name, data in spectra.items():

    freq = data["freq"]

    for BPF_M in harmonics:

        mask = (
            (freq >= BPF_M-half_width) &
            (freq <= BPF_M+half_width)
        )

        # level = #10*np.log10(
        level =  np.trapezoid(
                data["psd"][mask],
                freq[mask]
            ) / pref**2
         #)

        BPF_SPL.append(level)

# print(BPF_SPL)

theta_stretched = np.repeat(theta_rad, n_f_ref)   # [t1,t1,t1,...,t2,t2,t2,...]
harmonics_stretched = np.tile(harmonics, len(theta_rad))

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, PchipInterpolator

# Stage 1: independent per-harmonic fit (as before)
coeffs_per_harmonic = {}
def directivity_single_freq(theta, A, a1, b1, a2, b2):
    return (A + a1*np.cos(theta) + b1*np.sin(theta)
              + a2*np.cos(2*theta) + b2*np.sin(2*theta))

for h in harmonics:
    mask = np.isclose(harmonics_stretched, h)
    popt_h, _ = curve_fit(
        directivity_single_freq,
        theta_stretched[mask], np.array(BPF_SPL)[mask],
        p0=[70,0,0,0,0]
    )
    coeffs_per_harmonic[h] = popt_h

# Reshape into arrays: coeffs_array[i, j] = j-th coefficient at i-th harmonic
freqs_sorted = np.array(sorted(coeffs_per_harmonic.keys()))
coeffs_array = np.array([coeffs_per_harmonic[h] for h in freqs_sorted])  # (n_f_ref, 7)

# Stage 2: smooth interpolant per coefficient, as a function of frequency
coeff_names = ["A","a1","b1","a2","b2"]
smooth_coeffs = {
    name: PchipInterpolator(freqs_sorted, coeffs_array[:, k])
    for k, name in enumerate(coeff_names)
}

# Final smooth, continuous-in-frequency model:
def harmonic_directivity_smooth(theta, freq):
    A  = smooth_coeffs["A"](freq)
    a1 = smooth_coeffs["a1"](freq); b1 = smooth_coeffs["b1"](freq)
    a2 = smooth_coeffs["a2"](freq); b2 = smooth_coeffs["b2"](freq)
    return (A + a1*np.cos(theta) + b1*np.sin(theta)
              + a2*np.cos(2*theta) + b2*np.sin(2*theta))










# from scipy.interpolate import UnivariateSpline

# # ---- Stage 1: independent per-harmonic fit ----
# def directivity_single_freq(theta, A, a1, b1, a2, b2):
#     return (A + a1*np.cos(theta) + b1*np.sin(theta)
#               + a2*np.cos(2*theta) + b2*np.sin(2*theta))

# coeffs_per_harmonic = {}
# errs_per_harmonic = {}

# for h in harmonics:
#     mask = np.isclose(harmonics_stretched, h)
#     popt_h, pcov_h = curve_fit(
#         directivity_single_freq,
#         theta_stretched[mask], np.array(BPF_SPL)[mask],
#         p0=[70,0,0,0,0]
#     )
#     coeffs_per_harmonic[h] = popt_h
#     errs_per_harmonic[h] = np.sqrt(np.diag(pcov_h))

# freqs_sorted = np.array(sorted(coeffs_per_harmonic.keys()))
# coeffs_array = np.array([coeffs_per_harmonic[h] for h in freqs_sorted])
# coeffs_err_array = np.array([errs_per_harmonic[h] for h in freqs_sorted])

# # ---- Stage 2: smoothing spline per coefficient ----
# coeff_names = ["A","a1","b1","a2","b2"]
# smooth_coeffs = {}

# for k, name in enumerate(coeff_names):
#     y = coeffs_array[:, k]
#     w = 1.0 / np.clip(coeffs_err_array[:, k], 1e-6, None)
#     s_val = 5 * len(w)   # <-- tune this; see previous message re: s=None trap
#     spline = UnivariateSpline(freqs_sorted, y, w=w, k=5, s=s_val)
#     smooth_coeffs[name] = spline

# def harmonic_directivity_smooth(theta, freq):
#     A  = smooth_coeffs["A"](freq)
#     a1 = smooth_coeffs["a1"](freq); b1 = smooth_coeffs["b1"](freq)
#     a2 = smooth_coeffs["a2"](freq); b2 = smooth_coeffs["b2"](freq)
#     return (A + a1*np.cos(theta) + b1*np.sin(theta)
#               + a2*np.cos(2*theta) + b2*np.sin(2*theta))

BPF_SPL_fit = harmonic_directivity_smooth(theta_stretched, harmonics_stretched)
residuals = np.array(BPF_SPL) - BPF_SPL_fit

test_freq = np.array([200.0])  # pick a frequency inside your measured range
for name in coeff_names:
    val = smooth_coeffs[name](test_freq)
    print(name, val)

plt.figure(figsize=(8,6))
theta_plot = np.linspace(0, 0.55*np.pi, 200)

for h in harmonics:
    # measured points at this harmonic
    mask = np.isclose(harmonics_stretched, h)
    plt.scatter(theta_stretched[mask], np.array(BPF_SPL)[mask],
                label=f"{h:.0f} Hz (measured)")

    # fitted curve at this harmonic, over a fine theta grid
    freq_plot = np.full_like(theta_plot, h)
    spl_plot = harmonic_directivity_smooth(theta_plot, freq_plot)
    plt.plot(theta_plot, spl_plot, '--')

# for h in harmonics:
#     mask = np.isclose(harmonics_stretched, h)
#     plt.scatter(theta_stretched[mask], np.array(BPF_SPL)[mask], label=f"{h:.0f} Hz (measured)")

#     freq_plot = np.full_like(theta_deg, h)
#     spl_plot = harmonic_directivity_smooth(theta_plot, freq_plot)   # <-- changed
#     plt.plot(theta_plot, spl_plot, '--')

plt.xlabel("theta [rad]")
plt.ylabel("SPL [dB]")
plt.legend()
plt.grid(True)
plt.title("Directivity fit per harmonic")
plt.show()

theta_plot = np.linspace(0, 0.55*np.pi, 200)  # matches your existing range

import math

n_harm = len(harmonics)
n_cols = 3  # adjust to taste, e.g. 2 for larger individual panels, 3 for more compact
n_rows = math.ceil(n_harm / n_cols)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(5*n_cols, 5*n_rows),
    subplot_kw={"projection": "polar"}
)
axes = np.array(axes).reshape(-1)  # flatten in case of a 2D grid

for ax, h in zip(axes, harmonics):
    mask = np.isclose(harmonics_stretched, h)

    ax.scatter(theta_stretched[mask], np.array(BPF_SPL)[mask],
               color="C0", label="measured", zorder=3, s=50)

    freq_plot = np.full_like(theta_plot, h)
    spl_plot = harmonic_directivity_smooth(theta_plot, freq_plot)
    ax.plot(theta_plot, spl_plot, '--', color="C1", label="fit", linewidth=2)

    ax.set_title(f"{h:.0f} Hz", fontsize=13)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

# hide any unused subplot axes if n_harm doesn't evenly fill the grid
for ax in axes[n_harm:]:
    ax.axis("off")

axes[0].legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

unique_thetas = np.unique(theta_rad)
n_theta = len(unique_thetas)
n_cols = 3
n_rows = math.ceil(n_theta / n_cols)

freq_fine = np.linspace(min(harmonics), max(harmonics), 200)  # smooth fit curve

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharey=True)
axes = np.array(axes).reshape(-1)

for ax, th in zip(axes, unique_thetas):
    mask = np.isclose(theta_stretched, th)

    ax.scatter(harmonics_stretched[mask], np.array(BPF_SPL)[mask],
               color="C0", s=60, zorder=3, label="measured")

    theta_fixed_fine = np.full_like(freq_fine, th)
    spl_fine = harmonic_directivity_smooth(theta_fixed_fine, freq_fine)
    ax.plot(freq_fine, spl_fine, '--', color="C1", label="fit")

    ax.set_title(f"theta = {th:.2f} rad")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("SPL [dB]")
    ax.grid(True)

for ax in axes[n_theta:]:
    ax.axis("off")

axes[0].legend()
plt.tight_layout()
plt.show()

# save coeffs 
harmonic_numbers = freqs_sorted / BPF   # e.g. [1., 2., 3., 4., 5.]

output_path = DEFAULT_OUTPUT_DIR / "directivity_model_pchip.npz"

np.savez(
    output_path,
    harmonic_numbers=harmonic_numbers,
    coeffs=coeffs_array,
    coeff_names=np.array(coeff_names)
)

print(f"Saved directivity model to {output_path}")