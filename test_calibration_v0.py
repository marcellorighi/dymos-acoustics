import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

pref = 20e-6  # Pa
target_db = 94.0

p_rms_target = pref * 10**(target_db / 20)
print(f"Target RMS pressure: {p_rms_target:.5f} Pa")

filename = "/Users/marcello/Downloads/Calibration.wav" 
fs, data = wavfile.read(filename)

calib_raw = data.astype(np.float64)

raw_rms = np.sqrt(np.mean(calib_raw**2))
raw_peak = np.max(np.abs(calib_raw))

print(f"Raw RMS: {raw_rms:.3e}")
print(f"Raw peak: {raw_peak:.3e}")

print(f"Sample rate: {fs} Hz")
print(f"Shape: {data.shape}, dtype: {data.dtype}")

calibration_factor = p_rms_target / raw_rms
print(f"Calibration factor: {calibration_factor:.6e} Pa/count")

t = np.arange(len(data)) / fs

plt.figure(figsize=(10, 4))
plt.plot(t, data, linewidth=0.8)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")   # or "Pressure [Pa]" if data is already calibrated
plt.title("Waveform")
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
