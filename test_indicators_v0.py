import numpy as np
from scipy.io import wavfile
from scipy.signal import welch, butter, lfilter

class AnnoyanceAnalyzer:
    """
    A simplified psychoacoustic analyzer implementing metrics based on 
    Zwicker's models for acoustic annoyance.
    """
    def __init__(self, sample_rate, signal):
        self.fs = sample_rate
        # Ensure signal is mono and normalized
        if len(signal.shape) > 1:
            self.signal = np.mean(signal, axis=1)
        else:
            self.signal = signal
        
        # Reference values for psychoacoustics
        self.p_ref = 2e-5  # 20 microPascals (threshold of hearing)

    def _get_spl(self):
        """Calculates Sound Pressure Level."""
        rms = np.sqrt(np.mean(self.signal**2))
        return 20 * np.log10(rms / self.p_ref) if rms > 0 else 0

    def calculate_loudness_simple(self):
        """
        A simplified approximation of Zwicker Loudness (Sones).
        Standard: 1 Sone is defined as a 1kHz tone at 40dB SPL.
        Formula: N = 2^((Lp - 40)/10)
        """
        spl = self._get_spl()
        # Simplified power law for loudness
        loudness_sones = 2**((spl - 40) / 10)
        return max(0, loudness_sones)

    def calculate_sharpness(self):
        """
        Calculates Sharpness (Acum) using the von Bismarck approach.
        High-frequency content increases annoyance.
        """
        freqs, psd = welch(self.signal, self.fs, nperseg=1024)
        
        # Weighting function: increases significantly above 3kHz
        weighting = np.ones_like(freqs)
        mask = freqs > 3000
        weighting[mask] = 1 + 0.00012 * (freqs[mask] - 3000)
        
        # Sharpness is the weighted centroid of the spectrum
        numerator = np.sum(psd * weighting * freqs)
        denominator = np.sum(psd)
        
        # Normalized so 1 Acum is a 1kHz bandwidth noise at 60dB
        return (numerator / denominator) / 1000 if denominator > 0 else 0

    def calculate_roughness_simple(self):
        """
        Simplified Roughness (Asper).
        Roughness is caused by rapid amplitude modulations (15-300 Hz).
        This implementation looks at the modulation envelope.
        """
        # 1. Extract Envelope (Hilbert-like approximation via rectification/low-pass)
        env = np.abs(self.signal)
        b, a = butter(4, 400/(self.fs/2), btype='low')
        env_filtered = lfilter(b, a, env)
        
        # 2. Analyze modulation frequency in the 70Hz range (peak roughness)
        freqs, mod_psd = welch(env_filtered - np.mean(env_filtered), self.fs, nperseg=2048)
        
        # Focus on the 15-300Hz modulation band
        rough_band = (freqs >= 15) & (freqs <= 300)
        roughness_val = np.sum(mod_psd[rough_band]) * 10e4 # Scaled approximation
        return roughness_val

    def get_annoyance_index(self):
        """
        Zwicker's Psychoacoustic Annoyance (PA) formula:
        PA = N_13 * (1 + sqrt(w_S^2 + w_FR^2))
        """
        N = self.calculate_loudness_simple()
        S = self.calculate_sharpness()
        R = self.calculate_roughness_simple()
        
        # Weighting factors (simplified)
        w_S = (S - 1.75) * 0.25 if S > 1.75 else 0
        w_R = R * 0.3
        
        pa = N * (1 + np.sqrt(w_S**2 + w_R**2))
        return {
            "Loudness (Sone)": round(N, 2),
            "Sharpness (Acum)": round(S, 2),
            "Roughness (Asper)": round(R, 3),
            "Annoyance Index": round(pa, 2)
        }

# Example Usage:
# fs, data = wavfile.read('engine_sound.wav')
# analyzer = AnnoyanceAnalyzer(fs, data)
# print(analyzer.get_annoyance_index())

if __name__ == "__main__":
    # Generate a synthetic "annoying" signal: 1kHz tone with high-freq hiss and modulation
    fs = 44100
    t = np.linspace(0, 1, fs)
    # Carrier + Modulation (Roughness) + Hiss (Sharpness)
    signal = (1 + 0.5 * np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t) 
    signal += 0.2 * np.random.normal(0, 1, len(t)) # Add high freq noise
    
    analyzer = AnnoyanceAnalyzer(fs, signal)
    results = analyzer.get_annoyance_index()
    
    print("Simplified Zwicker Annoyance Analysis")
    print("====================================")
    for k, v in results.items():
        print(f"{k}: {v}")
