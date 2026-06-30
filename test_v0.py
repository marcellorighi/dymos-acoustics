import numpy as np
from rotor_rpm_estimation import estimate_rotor_rpm
from drone_acoustic_radiation_v1 import (
    calibrate_p_ref, AcousticParams, FineGridParams,
    estimate_received_spl_fine,
)

# 1. Coarse RPM estimate from your Dymos output (as before)
rpm_result = estimate_rotor_rpm(t, x, y, z, vx, vy, vz, ax, ay, az, wx, wy, wz)

# 2. Calibrate the RPM-to-power model against a reference measurement
p_ref = calibrate_p_ref(
    spl_ref_db=62.0, rpm_ref_measurement=5000.0,
    r_ref=1.0, theta_ref_deg=90.0,
    n_rotors_in_measurement=4, n_exponent=5.0,
)
acoustic_params = AcousticParams(rpm_ref=5000.0, p_ref=p_ref, n_exponent=5.0)

# 3. Fine-grid settings: sample rate, interpolation, disturbance
fine_params = FineGridParams(
    fs=48000.0,                     # matches the Zwicker/MOSQITO pipeline
    interp_method="cubic",
    use_integrated_phase=True,      # physically correct phase tracking
    disturbance_amplitude_rad=0.05, # small phase jitter (~3 deg)
    disturbance_bandwidth_hz=20.0,
    random_seed=42,                 # set for reproducibility
)

observer_xyz = (0.0, 50.0, 0.0)

# 4. The call
spl_fine = estimate_received_spl_fine(
    t, x, y, z,
    rpm_result["rpm_front"], rpm_result["rpm_rear"],
    rpm_result["rpm_right"], rpm_result["rpm_left"],
    observer_xyz,
    acoustic_params=acoustic_params,
    fine_params=fine_params,
)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(spl_fine["t_fine"], spl_fine["spl_db"])
axes[0].set_ylabel("Received SPL [dB]")

# zoom into a short window to see individual rotor cycles
mask = spl_fine["t_fine"] < 0.05
axes[1].plot(spl_fine["t_fine"][mask], spl_fine["psi_front"][mask], label="front")
axes[1].set_ylabel(r"$\psi$ (azimuth)")
axes[1].set_xlabel("Time [s]")
axes[1].legend()
plt.tight_layout()
plt.show()

