import numpy as np
import matplotlib.pyplot as plt
from zwicker_annoyance_v3 import compute_zwicker_indicators_windowed

fs = 48000
duration = 2.5  # seconds
t = np.arange(0, duration, 1 / fs)
pref = 20e-6  # Pa, standard acoustic reference


def zwicker_annoyance_with_relative_importance(N: float, S: float, R: float, F):
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
    if any(np.isnan(v) for v in (N, S, R, F)):
        # Explicit NaN propagation. Without this check, `S > 1.75` below
        # would silently evaluate to False whenever S is NaN (NaN
        # comparisons are always False in Python/NumPy), routing execution
        # to the `else 0.0` branch and producing a falsely "valid" PA value
        # that ignores sharpness entirely, instead of correctly returning
        # NaN. This was a real, silent bug in earlier versions of this
        # function -- guard against it explicitly rather than relying on
        # comparison semantics.
        return float("nan")
    N = max(N, 1e-6)  # avoid log/pow domain errors for (near-)silent signals
    wS = (S - 1.75) * 0.25 * np.log10(N + 10) if S > 1.75 else 0.0
    wFR = (2.18 / N**0.4) * (0.4 * F + 0.6 * R)
    PA = N * (1 + np.sqrt(wS**2 + wFR**2))
    print(wS,wFR)
    return float(PA)


def set_spl(x, target_db, pref=pref):
    """Scale a signal to a target overall SPL [dB re 20 uPa]."""
    rms = np.sqrt(np.mean(x**2))
    if rms < 1e-20:
        return x
    target_rms = pref * 10**(target_db / 20)
    return x * (target_rms / rms)


def am_tone(fc, fm, m, dur_t, phase_c=0.0, phase_m=0.0):
    """Amplitude-modulated tone: carrier fc, modulation rate fm, depth m in [0,1]."""
    return (1 + m * np.cos(2 * np.pi * fm * dur_t + phase_m)) * np.sin(2 * np.pi * fc * dur_t + phase_c)


def harmonic_complex(f0, n_harm, dur_t, decay_exponent=1.5):
    """Harmonic tone stack: f0, 2*f0, ..., n_harm*f0, with 1/n^decay amplitude falloff."""
    sig = np.zeros_like(dur_t)
    for n in range(1, n_harm + 1):
        sig += (1.0 / n**decay_exponent) * np.sin(2 * np.pi * n * f0 * dur_t)
    return sig


target_spl = 70.0  # dB, representative drone-at-distance level; adjust to taste

signals = {}

signals["tone_100Hz"] = set_spl(
    np.sin(2 * np.pi * 100 * t), target_spl
)

signals["harmonics_BPF100"] = set_spl(
    harmonic_complex(100, n_harm=5, dur_t=t), target_spl
)

signals["tone1kHz_AM100Hz_roughness"] = set_spl(
    am_tone(fc=1000, fm=100, m=0.9, dur_t=t), target_spl
)

signals["tone1kHz_AM4Hz_fluctuation"] = set_spl(
    am_tone(fc=1000, fm=4, m=0.9, dur_t=t), target_spl
)

rng = np.random.default_rng(0)
white_noise = rng.standard_normal(t.size)

for snr_db in [20, 10, 0]:
    harmonics = harmonic_complex(100, n_harm=5, dur_t=t)
    harmonics = set_spl(harmonics, target_spl)
    noise = set_spl(white_noise, target_spl - snr_db)  # noise SPL below tone by snr_db
    signals[f"harmonics_BPF100_noise_SNR{snr_db}dB"] = harmonics + noise

signals["white_noise_only"] = set_spl(white_noise, target_spl)


# ----------------------------------------------------
# Compute PA indicators for each test signal
# ----------------------------------------------------
results = {}
for name, sig in signals.items():
    print(f"Computing indicators for: {name}")
    results[name] = compute_zwicker_indicators_windowed(
        sig, fs,
        window_s=1.0, hop_s=0.25,   # shorter window than default, given 2.5 s signals
        stationary = True,
        use_fs_approximation = True,
    )

# from timbral_models import timbral_loudness

# for name, sig in signals.items():
#     loud_mosqito = np.nanmean(results[name]['loudness_sone'])   # from your existing pipeline
#     loud_timbral = timbral_loudness(sig, fs=fs)
#     print(f"{name:35s}  MOSQITO: {loud_mosqito:6.2f} sone   timbral_models: {loud_timbral:6.2f} sone")


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


for name, pa in results.items():
    plot_dashboard(pa, title=name)


# ----------------------------------------------------
# Summary comparison: mean indicator value per test signal
# ----------------------------------------------------
fig, axs = plt.subplots(len(acoustic_metadata), 1, figsize=(9, 2.0 * len(acoustic_metadata)), sharex=True)

names = list(results.keys())
for idx, (key, meta) in enumerate(acoustic_metadata.items()):
    means = [np.nanmean(results[n][key]) for n in names]
    axs[idx].bar(names, means, color=meta['color'])
    axs[idx].set_ylabel(meta['label'], fontsize=9)
    axs[idx].tick_params(axis='x', rotation=45, labelsize=8)
    axs[idx].grid(True, axis='y', linestyle=':', alpha=0.5)

fig.suptitle("Mean psychoacoustic indicators across test signals", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(names)

print(means)