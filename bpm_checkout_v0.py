import numpy as np
import matplotlib.pyplot as plt

def compute_St_peak(U_rel, chord, nu=1.5e-5):
    """
    BPM peak Strouhal number St_1, RP-1218 Eq. B6.
    nu: kinematic viscosity of air [m^2/s], ~1.5e-5 at 20 deg C
    """
    Re_c = U_rel * chord / nu
    if Re_c < 2.47e5:
        St_1 = 0.18
    elif Re_c < 8.0e5:
        St_1 = 0.001756 * Re_c**0.3931
    else:
        St_1 = 0.28
    return St_1

def bpm_spectral_shape_A(St, St_peak):
    St    = np.maximum(np.asarray(St, dtype=float), 1e-10)
    a_abs = np.abs(np.log10(St / St_peak))

    # Clip sqrt argument to avoid RuntimeWarning from np.where eager evaluation.
    # np.where computes ALL branches for ALL elements before selecting, so the
    # sqrt is evaluated even where a_abs > 0.204 (where it would go negative).
    # The clip has no effect on the final selected values since those elements
    # are covered by the other branches anyway.
    sqrt_arg = np.clip(67.552 - 886.788 * a_abs**2, 0.0, None)

    A = np.where(
        a_abs <= 0.204,
        np.sqrt(sqrt_arg) - 8.219,
        np.where(
            a_abs <= 0.244,
            -32.665 * a_abs + 3.981,
            -142.795 * a_abs**3 + 103.656 * a_abs**2
            - 57.757 * a_abs + 6.006
        )
    )
    return A

def compute_segment_broadband_spl_narrowband(
    freqs_narrow,   # Frequency vector [Hz]
    df_narrow,      # Bin width [Hz]
    chord,          # Section chord [m]
    span,           # Radial segment length dr [m]
    U_rel,          # Relative inflow velocity [m/s]
    delta_star_p,   # Pressure side displacement thickness [m]
    delta_star_s,   # Suction side displacement thickness [m]
    r_obs,          # Distance to observer [m]
    c0=343.0,       # Speed of sound [m/s]
    tuning_factors=None
):
    if tuning_factors is None:
        tuning_factors = {'C_amp': 1.0, 'C_delta': 1.0, 'K1_shift': 0.0}

    p_ref = 2e-5
    nu=1.5e-5
    M = U_rel / c0
    
    delta_p = delta_star_p * tuning_factors['C_delta']
    delta_s = delta_star_s * tuning_factors['C_delta']
    
    # Local Strouhal numbers
    St_p = (freqs_narrow * delta_p) / U_rel
    St_s = (freqs_narrow * delta_s) / U_rel
    
    # Peak Strouhal number in standard BPM model (~0.1)
    Re_c = U_rel * chord / nu   # chord Reynolds number, nu=1.5e-5 m^2/s
    if Re_c < 2.47e5:
        St_1 = 0.18
    elif Re_c < 8.0e5:
        St_1 = 0.001756 * Re_c**0.3931
    else:
        St_1 = 0.28
    
    # 1. Base acoustic scaling (M^5 law)
    base_factor_p = (tuning_factors['C_amp'] * delta_p * (M**5) * span) / (r_obs**2)
    base_factor_s = (tuning_factors['C_amp'] * delta_s * (M**5) * span) / (r_obs**2)
    
    # 2. Evaluate correct spectral shapes
    St_peak = compute_St_peak(U_rel, chord)

    # Strouhal numbers at each frequency (arrays)
    St_p = (freqs_narrow * delta_p) / U_rel
    St_s = (freqs_narrow * delta_s) / U_rel

    # Spectral shapes using corrected A function
    A_p = bpm_spectral_shape_A(St_p, St_peak)
    A_s = bpm_spectral_shape_A(St_s, St_peak)
    
    # K1 = 0.0 + tuning_factors['K1_shift']
    Re_c = U_rel * chord / 1.5e-5
    if Re_c < 2.47e5:
        K1 = -4.31 * np.log10(Re_c) + 156.3
    elif Re_c < 8.0e5:
        K1 = -9.0 * np.log10(Re_c) + 181.6
    else:
        K1 = 128.5
    
    # 3. Nominal 1/3-octave SPL level
    spl_p_13 = 10 * np.log10(np.maximum(base_factor_p, 1e-18) / (p_ref**2)) + A_p + K1
    spl_s_13 = 10 * np.log10(np.maximum(base_factor_s, 1e-18) / (p_ref**2)) + A_s + K1
    
    # Incoherent sum of pressure and suction side
    spl_total_13 = 10 * np.log10(10**(spl_p_13 / 10.0) + 10**(spl_s_13 / 10.0))
    
    # 4. Bandwidth Conversion: 1/3-octave SPL -> PSD [dB/Hz] -> Narrow-band SPL
    # bw_13 = freqs_narrow * 0.23156
    # spl_psd = spl_total_13 - 10 * np.log10(bw_13)
    # spl_narrow = spl_psd + 10 * np.log10(df_narrow)

    bw_13 = freqs_narrow * 0.23156      # 1/3-octave bandwidth
    spl_psd = spl_total_13 - 10*np.log10(bw_13)   # 1/3-oct -> PSD
    spl_narrow = spl_psd + 10*np.log10(df_narrow)  # PSD -> narrowband
    
    # Return acoustic pressure squared [Pa^2]
    return (p_ref**2) * (10**(spl_narrow / 10.0))

def compute_high_res_broadband_spectrum(
    blade_sections, 
    f_min=100.0, 
    f_max=12000.0, 
    df=5.0, 
    r_obs=1.0, 
    num_blades=2, 
    tuning_factors=None
):
    freqs_narrow = np.arange(f_min, f_max + df, df)
    p_ref = 2e-5
    total_p_sq = np.zeros_like(freqs_narrow, dtype=float)
    
    for section in blade_sections:
        p_sq_sec = compute_segment_broadband_spl_narrowband(
            freqs_narrow=freqs_narrow,
            df_narrow=df,
            chord=section['chord'],
            span=section['dr'],
            U_rel=section['U_rel'],
            delta_star_p=section['delta_p'],
            delta_star_s=section['delta_s'],
            r_obs=r_obs,
            tuning_factors=tuning_factors
        )
        total_p_sq += p_sq_sec * num_blades
        
    total_spl_narrow = 10 * np.log10(np.maximum(total_p_sq, 1e-18) / (p_ref**2))
    return freqs_narrow, total_spl_narrow

def calibrated_delta_star(chord, U_rel, f_peak_measured,
                           St_peak=0.20, side='pressure'):
    delta_star = St_peak * U_rel / f_peak_measured
    if side == 'suction':
        delta_star *= 1.3
    return delta_star

# For your test sections, targeting f_peak = 1500 Hz
f_peak_target = 5000.0   # Hz -- midpoint of your observed 1-3 kHz range

test_sections = []
for chord, dr, U_rel in [(0.025, 0.02, 45.0),
                          (0.020, 0.02, 65.0),
                          (0.015, 0.02, 85.0)]:
    test_sections.append({
        'chord':   chord,
        'dr':      dr,
        'U_rel':   U_rel,
        'delta_p': calibrated_delta_star(chord, U_rel, f_peak_target, side='pressure'),
        'delta_s': calibrated_delta_star(chord, U_rel, f_peak_target, side='suction'),
    })
    
def drone_delta_star_tripped(chord, U_rel, nu=1.5e-5, side='pressure'):
    """
    Assume artificially tripped (fully turbulent) BL from leading edge.
    Uses RP-1218 Eq. B8 for tripped condition.
    Valid assumption if blade has rough surface, serrations, or operates
    in turbulent inflow (which drone rotors typically do).
    """
    Re_c = U_rel * chord / nu
    # Tripped turbulent BL: RP-1218 Eq B8
    # delta*_p / c = 10^(3.411 - 1.5397*log10(Re_c) + 0.1059*log10(Re_c)^2)
    log_Re = np.log10(Re_c)
    delta_p_over_c = 10**(3.411 - 1.5397*log_Re + 0.1059*log_Re**2)
    delta_p = delta_p_over_c * chord

    if side == 'suction':
        # RP-1218 Eq B9 for suction side (tripped)
        delta_s_over_c = 10**(3.0187 - 1.5397*log_Re + 0.1059*log_Re**2)
        return delta_s_over_c * chord

    return delta_p

def drone_delta_star(chord, U_rel, nu=1.5e-5, side='pressure'):
    """
    Displacement thickness estimate suitable for small drone rotors
    (Re ~ 5e4 - 3e5), accounting for the fact that the boundary layer
    is laminar/transitional, not fully turbulent as BPM assumes.

    For laminar BL the displacement thickness at the trailing edge is
    estimated from the Blasius solution but with a separation/transition
    correction factor that accounts for the adverse pressure gradient
    on a real cambered blade, which thickens the BL significantly
    beyond the flat-plate Blasius value.
    """
    Re_c = U_rel * chord / nu

    if Re_c < 3e5:
        # Blasius laminar base
        delta_blasius = 1.72 * chord / np.sqrt(Re_c)
        # Adverse pressure gradient correction for a cambered blade:
        # real delta* is typically 3-8x the flat-plate Blasius value
        # due to flow deceleration on the pressure side and risk of
        # laminar separation on the suction side.
        # Use a factor of 5 as a central estimate (tune against data).
        apc = 5.0
        delta_star = delta_blasius * apc
    else:
        # Turbulent (original BPM range)
        delta_star = 0.0375 * chord / Re_c**0.2

    if side == 'suction':
        delta_star *= 1.5   # suction side thicker due to adverse gradient

    return delta_star

def flat_plate_delta_star(chord, U_rel, nu=1.5e-5, side='pressure'):
    """
    Flat-plate zero-pressure-gradient displacement thickness at x=chord.
    BPM uses different correlations for pressure and suction sides.
    This is the simplest estimate -- replace with XFOIL output if available.
    """
    Re_c = U_rel * chord / nu

    if Re_c < 3e5:
        # Laminar (Blasius): delta* = 1.72 * x / sqrt(Re_x)
        delta_star = 1.72 * chord / np.sqrt(Re_c)
    else:
        # Turbulent: delta* ~ 0.0375 * chord / Re_c^0.2  (1/7 power law)
        delta_star = 0.0375 * chord / Re_c**0.2

    # Suction side is typically 10-30% thicker than pressure side
    if side == 'suction':
        delta_star *= 1.2

    return delta_star

nu = 1.5e-5
f_peak_target = 3500.0   # Hz

test_sections = []
for chord, dr, U_rel in [(0.025, 0.02, 45.0),
                          (0.020, 0.02, 65.0),
                          (0.015, 0.02, 85.0)]:
    test_sections.append({
        'chord':   chord,
        'dr':      dr,
        'U_rel':   U_rel,
        'delta_p': calibrated_delta_star(chord, U_rel,
                       f_peak_measured=f_peak_target, side='pressure'),
        'delta_s': calibrated_delta_star(chord, U_rel,
                       f_peak_measured=f_peak_target, side='suction'),
    })

# 2. Compute spectrum
freqs, spl = compute_high_res_broadband_spectrum(
    blade_sections=test_sections,
    f_min=100, f_max=32000, df=5.0, r_obs=2.0, num_blades=2
)

# 3. Plot
plt.figure(figsize=(8, 4))
plt.semilogx(freqs, spl, label="Corrected BPM Broadband Noise")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Narrowband SPL [dB / 5 Hz]")
plt.grid(True, which="both", ls=":")
plt.title("Rotor Broadband Noise Spectrum")
plt.legend()
plt.show()