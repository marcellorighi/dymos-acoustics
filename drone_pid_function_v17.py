import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from dryden_timeseries import generate_dryden_time_series, DrydenParams
from rotor_rpm_estimation import estimate_rotor_rpm
from drone_acoustic_radiation_v3 import (
        calibrate_p_ref, AcousticParams, FineGridParams,
        estimate_received_spl_fine,
    )
from zwicker_annoyance_v2 import compute_zwicker_indicators_windowed
import pandas as pd
from scipy.stats import qmc  # Requires scipy >= 1.7.0 for Latin Hypercube
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol as sobol_analyzer
from tqdm import tqdm


# # --- 1. Global System Parameters ---
# MASS = 6.5              # kg
# I_xx = 0.012            # kg*m^2
# I_yy = 0.012            # kg*m^2
# I_zz = 0.020            # kg*m^2
# g = 9.81                # m/s^2
# arm_length = 0.25       # m 
# t_ref = MASS * g /4     # N 
# rpm_ref = 5000.         # RPM 

def compute_acoustic_spectrum_debug(spl_fine):
    """
    Computes the single-sided frequency spectrum (Narrowband SPL) 
    of the combined acoustic pressure signal at the observer.
    
    Parameters
    ----------
    spl_fine : dict
        The output dictionary from estimate_received_spl_fine()
        containing 't_fine' and 'p_signal'.
        
    Returns
    -------
    dict with keys:
        'frequencies' : 1D array of frequency bins [Hz]
        'spl_frequency': 1D array of Sound Pressure Levels [dB SPL] per bin
        'p_rms_fft'    : Raw RMS pressure amplitudes per frequency bin [Pa]
    """
    t_fine = np.asarray(spl_fine["t_fine"])
    p_signal = np.asarray(spl_fine["p_signal"])
    
    # 1. Automatically calculate the sampling frequency (fs)
    n_samples = len(t_fine)
    duration = t_fine[-1] - t_fine[0]
    fs = n_samples / duration  # [Hz]
    
    # 2. Apply a Hann window to prevent spectral leakage
    window = np.hanning(n_samples)
    windowed_signal = p_signal * window
    
    # Amplitude correction factor for applying the Hann window
    window_loss_factor = np.sqrt(8.0 / 3.0) 
    
    # 3. Compute the Real Fast Fourier Transform (RFFT)
    # Using rfft because physical pressure is a real-valued signal
    p_fft = np.fft.rfft(windowed_signal)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0/fs)
    
    # 4. Convert to RMS pressure per frequency bin
    # Scale by 2/N for single-sided spectrum, and adjust for windowing loss
    p_amplitude = (2.0 / n_samples) * np.abs(p_fft) * window_loss_factor
    p_rms = p_amplitude / np.sqrt(2.0)  # Convert peak to RMS
    
    # 5. Convert physical pressure (Pa) to Sound Pressure Level (dB SPL)
    p_ref = 2e-5  # Auditory reference threshold [Pa]
    # Use np.maximum to prevent log10(0.0) errors
    spl_frequency = 20.0 * np.log10(np.maximum(p_rms, 1e-10) / p_ref)
    
    return {
        "frequencies": frequencies,
        "spl_frequency": spl_frequency,
        "p_rms_fft": p_rms,
        "fs": fs
    }

def plot_acoustic_spectrum(spectrum_data, max_freq_display=1000.0):
    """
    Plots the narrowband SPL spectrum up to a user-defined frequency limit.
    """
    freqs = spectrum_data["frequencies"]
    spl_db = spectrum_data["spl_frequency"]
    
    # Filter arrays for cleaner plotting (we don't need to see up to fs/2)
    mask = freqs <= max_freq_display
    
    plt.figure(figsize=(10, 4))
    plt.plot(freqs[mask], spl_db[mask], color='#2980B9', linewidth=1.5, label='Synthetic Signal')
    
    plt.title('Narrowband Acoustic Frequency Spectrum at Observer (Debug Mode)', fontsize=12, fontweight='bold')
    plt.xlabel('Frequency [Hz]', fontsize=10)
    plt.ylabel('Sound Pressure Level [dB SPL]', fontsize=10)
    plt.grid(True, which="both", linestyle=':', alpha=0.5)
    plt.xlim(0, max_freq_display)
    
    # Find and label the absolute loudest peak (likely the dominant BPF)
    peak_idx = np.argmax(spl_db[mask])
    peak_freq = freqs[mask][peak_idx]
    peak_val = spl_db[mask][peak_idx]
    
    plt.annotate(f'Dominant Tone: {peak_freq:.1f} Hz ({peak_val:.1f} dB)',
                 xy=(peak_freq, peak_val),
                 xytext=(peak_freq + (max_freq_display * 0.05), peak_val - 5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
    
    plt.tight_layout()
    plt.show()

def compute_rotor_rpm_from_controls(time_steps, history_controls, arm_length, 
                                    t_ref, rpm_ref, plant_params, rpm_min=1000.0, rpm_max=8000.0) -> dict:
    """
    Directly computes individual rotor RPMs from the physical control forces and moments
    generated during the simulation flight.

    Parameters
    ----------
    time_steps : array-like (N,)
        Simulation time array.
    history_controls : array-like (N, 6)
        Logged controls matrix from simulation where columns are: [Fx, Fy, Fz, Mx, My, Mz].
    arm_length : float
        The physical distance from the drone's center of mass to any rotor center [m].
    t_ref : float
        Reference thrust for *one single rotor* at a measured calibration state [N].
    rpm_ref : float
        Reference RPM for *one single rotor* corresponding to t_ref [RPM].
    rpm_min, rpm_max : float, optional
        Physical saturation limit constraints of the brushless motors.

    Returns
    -------
    dict containing synchronized time-series arrays for individual rotor RPMs.
    """
    # Extract the active control forces and moments from the logged matrix history
    F_z = np.asarray(history_controls[:, 2], dtype=float)  # Collective Vertical Force [N]
    M_x = np.asarray(history_controls[:, 3], dtype=float)  # Roll Moment [N*m]
    M_y = np.asarray(history_controls[:, 4], dtype=float)  # Pitch Moment [N*m]
    
    # 1. Map Body Forces/Moments into Individual Rotor Thrusts (+ configuration)
    # Average thrust allocated to each motor from the collective throttle vertical force
    thrust_avg = F_z / 4.0

    # R_rotor = plat_params['R_rotor']
    # R_ref = 0.22 
    
    # Differential thrust allocations mapping moments through the lever arm length L
    # F_front = thrust_avg + M_y / (2.0 * arm_length)
    # F_rear  = thrust_avg - M_y / (2.0 * arm_length)
    # F_right = thrust_avg + M_x / (2.0 * arm_length)
    # F_left  = thrust_avg - M_x / (2.0 * arm_length)

    F_front = thrust_avg + M_y / (1.0 * arm_length)
    F_rear  = thrust_avg - M_y / (1.0 * arm_length)
    F_right = thrust_avg + M_x / (1.0 * arm_length)
    F_left  = thrust_avg - M_x / (1.0 * arm_length)
    
    # Prevent negative values physically (rotors cannot pull backward)
    F_front = np.clip(F_front, 0.0, None)
    F_rear  = np.clip(F_rear,  0.0, None)
    F_right = np.clip(F_right, 0.0, None)
    F_left  = np.clip(F_left,  0.0, None)
    
    # 2. Derive the quadratic Thrust-to-RPM scaling factor (kT) from your calibration pair
    # F = kT * RPM^2  =>  kT = T_ref / (RPM_ref^2)
    kT = t_ref / (rpm_ref ** 2)
    
    # 3. Calculate RPM values and apply physical motor limits
    def thrust_to_rpm(F):
        rpm = np.sqrt(np.abs(F) / kT) # * (R_ref / R_rotor)**1.5 
        return np.clip(rpm, rpm_min, rpm_max)

    rpm_front = thrust_to_rpm(F_front)
    rpm_rear  = thrust_to_rpm(F_rear)
    rpm_right = thrust_to_rpm(F_right)
    rpm_left  = thrust_to_rpm(F_left)
    rpm_avg   = thrust_to_rpm(thrust_avg)

    return {
        "t": np.asarray(time_steps, dtype=float),
        "rpm_front": rpm_front,
        "rpm_rear": rpm_rear,
        "rpm_right": rpm_right,
        "rpm_left": rpm_left,
        "rpm_avg": rpm_avg,
        "thrust_total": F_z,
        "thrust_avg": thrust_avg
    }

def drone_derivatives(t, state, controls):
    """
    Calculates the right-hand side (derivatives) of the 6-DoF equations of motion.
    """
    # Unpack state
    pos = state[0:3]
    vel = state[3:6]
    euler = state[6:9]
    omega = state[9:12]
    
    phi, theta, psi = euler
    p, q, r = omega
    
    # Unpack pre-computed controls (frozen for the current timestep)
    F_x, F_y, F_z, M_x, M_y, M_z = controls
    
    # --- Translational Dynamics (Inertial Frame) ---
    # Gravity acts along the negative Z axis
    gravity_force = np.array([0.0, 0.0, -MASS * g])
    
    # Rotation matrix from Body Frame to Inertial Frame (Z-Y-X sequence)
    R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    R_body_to_inertial = R_z @ R_y @ R_x
    
    # Total thrust vector in body frame (F_z is the primary collective thrust)
    # Including optional direct body forces F_x, F_y if your specific drone configuration has them
    total_force_body = np.array([F_x, F_y, F_z])
    total_force_inertial = R_body_to_inertial @ total_force_body + gravity_force
    
    accel = total_force_inertial / MASS
    
    # --- Rotational Kinematics (Euler Angle Derivatives) ---
    # Transforms body angular velocities (p,q,r) to earth-fixed Euler rate changes
    d_euler = np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ]) @ omega

    # --- Rotational Dynamics (Euler's Rigid Body Equations in Body Frame) ---
    dp = (M_x - (I_zz - I_yy) * q * r) / I_xx
    dq = (M_y - (I_xx - I_zz) * p * r) / I_yy
    dr = (M_z - (I_yy - I_xx) * p * q) / I_zz
    d_omega = np.array([dp, dq, dr])
    
    # Pack array derivatives
    dstatedt = np.zeros(12)
    dstatedt[0:3]  = vel      # dpos/dt
    dstatedt[3:6]  = accel    # dvel/dt
    dstatedt[6:9]  = d_euler  # deuler/dt
    dstatedt[9:12] = d_omega  # domega/dt
    
    return dstatedt


# --- 2. Step PID Control Updates (Evaluated Once per Step) ---
class DronePIDController:
    def __init__(self, kp_vel=0.15, ki_vel=0.05, kp_att=2.5, kd_att=0.4, kp_alt=15.0, kd_alt=10.0, ki_alt=15.0, MASS=1.5, g=9.81):
        # Assign custom gains passed from the optimizer/simulation function
        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.kp_alt = kp_alt
        self.kd_alt = kd_alt
        self.ki_alt = ki_alt
        self.MASS = MASS
        self.g = g

        # --- Actuator Bandwidth Filter Setup ---
        motor_bandwidth_hz = 10.0  # Adjust based on real drone specs (10-30Hz typical)
        self.tau = 1.0 / (2.0 * np.pi * motor_bandwidth_hz)
        
        # State tracking (Integrators)
        self.integral_vx = 0.0
        self.integral_vy = 0.0
        self.integral_vz = 0.0
        self.dt = 0.01

        self.F_z_act = MASS * g  # Initialize hovering at equilibrium
        self.M_x_act = 0.0
        self.M_y_act = 0.0
        self.M_z_act = 0.0

    def compute_controls_velocity_tracking(self, state, target_velocity, target_yaw):
        vx, vy, vz = state[3:6]
        phi, theta, psi = state[6:9]
        p, q, r = state[9:12]
        
        # 1. Calculate raw tracking errors (Run ONCE)
        error_vx = target_velocity[0] - vx
        error_vy = target_velocity[1] - vy
        error_vz = target_velocity[2] - vz
        
        # 2. Update the integrators (Run ONCE)
        self.integral_vx += error_vx * self.dt
        self.integral_vy += error_vy * self.dt
        self.integral_vz += error_vz * self.dt
        
        # Anti-Windup Clamp: Protect the integrators from expanding to infinity
        self.integral_vx = np.clip(self.integral_vx, -2.0, 2.0)
        self.integral_vy = np.clip(self.integral_vy, -2.0, 2.0)
        self.integral_vz = np.clip(self.integral_vz, -5.0, 5.0)
        
        # --- TESTED CALIBRATION COEFFS ---
        # Change these multipliers between 1.0 and -1.0 to match your physics engine
        SIGN_X = +1.0  
        SIGN_Y = -1.0  
        
        # Compute the explicit attitude commands
        theta_cmd = SIGN_X * (self.kp_vel * error_vx + self.ki_vel * self.integral_vx)
        phi_cmd   = SIGN_Y * (self.kp_vel * error_vy + self.ki_vel * self.integral_vy)

        # 3. Inner Loop & Altitude Controls
        # Note: F_z acts as a pure P-controller on Z-velocity tracking

        # --- 1. Compute your explicit commanded values (Raw Ideal Outputs) ---
        F_z_cmd = self.kp_alt * (target_velocity[2] - vz) + self.ki_alt * self.integral_vz + (self.MASS * self.g)
        M_x_cmd = self.kp_att * (phi_cmd - phi) + self.kd_att * (0.0 - p)
        M_y_cmd = self.kp_att * (theta_cmd - theta) + self.kd_att * (0.0 - q)
        M_z_cmd = self.kp_att * (target_yaw - psi) + self.kd_att * (0.0 - r)

        alpha = self.dt / (self.tau + self.dt)
        
        # F_z = self.kp_alt * (target_velocity[2] - vz) + self.ki_alt * self.integral_vz + (MASS * g)
        # M_x = self.kp_att * (phi_cmd - phi) + self.kd_att * (0.0 - p)
        # M_y = self.kp_att * (theta_cmd - theta) + self.kd_att * (0.0 - q)
        # M_z = self.kp_att * (target_yaw - psi) + self.kd_att * (0.0 - r)

        self.F_z_act += alpha * (F_z_cmd - self.F_z_act)
        self.M_x_act += alpha * (M_x_cmd - self.M_x_act)
        self.M_y_act += alpha * (M_y_cmd - self.M_y_act)
        self.M_z_act += alpha * (M_z_cmd - self.M_z_act)
        
        controls = np.array([0.0, 0.0, self.F_z_act, self.M_x_act, self.M_y_act, self.M_z_act])
        return controls, theta_cmd, phi_cmd

def drone_derivatives_with_turb(t, state, controls, dryden_ts, aero_params=None):
    """
    6-DoF Drone Equations of Motion with injected Dryden Turbulence.
    """
    # Unpack drone params 
    I_xx = aero_params['Ixx']
    I_yy = aero_params['Iyy']
    I_zz = aero_params['Izz']
    L_arm = aero_params['L_arm']
    MASS = aero_params['MASS']
    g = aero_params['g']

    # Unpack state
    pos = state[0:3]
    vel = state[3:6]  # Inertial ground velocity [vx, vy, vz]
    euler = state[6:9]
    omega = state[9:12] # Actual body angular rates [p, q, r]
    
    phi, theta, psi = euler
    
    # 1. TIME INTERPOLATION OF THE DRYDEN GUSTS
    # Extract gusts at the current continuous time 't'
    t_ts = dryden_ts['t']
    wu = np.interp(t, t_ts, dryden_ts['wu'])  # Longitudinal gust (Body X)
    wv = np.interp(t, t_ts, dryden_ts['wv'])  # Lateral gust (Body Y)
    ww = np.interp(t, t_ts, dryden_ts['ww'])  # Vertical gust (Body Z)
    
    p_turb = np.interp(t, t_ts, dryden_ts['p_turb']) # Roll rate gust
    q_turb = np.interp(t, t_ts, dryden_ts['q_turb']) # Pitch rate gust
    
    # Unpack pre-computed controls
    F_x, F_y, F_z, M_x, M_y, M_z = controls
    
    # 2. TRANSLATIONAL DYNAMICS WITH WIND
    gravity_force = np.array([0.0, 0.0, -MASS * g])
    
    R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    R_body_to_inertial = R_z @ R_y @ R_x
    
    # Total controlled forces expressed in the body frame
    vel_inertial = state[3:6]
    vel_body = R_body_to_inertial.T @ vel_inertial  # [u, v, w] of drone
    
    # Subtract wind gust vectors to find true airspeed components
    u_air = vel_body[0] - wu
    v_air = vel_body[1] - wv
    w_air = vel_body[2] - ww

    vmag_air = np.sqrt(u_air**2 + v_air**2 + w_air**2)
    
    # Multipliers scaled by a drag constant and the drone's current arm size!
    # Pushes drag up if the drone or its structural arms expand
    # L_arm = 0.30 # TEMP
    drag_coeff_planar   = 0.1 * (1.0 + L_arm) 
    drag_coeff_vertical = 0.25 * (1.0 + L_arm)
    
    # Standard quadratic drag force: F = -C * V * |V|
    # F_drag_x = -drag_coeff_planar * u_air * np.abs(u_air)
    # F_drag_y = -drag_coeff_planar * v_air * np.abs(v_air)
    # F_drag_z = -drag_coeff_vertical * w_air * np.abs(w_air)
    
    F_drag_x = -drag_coeff_planar * u_air * vmag_air
    F_drag_y = -drag_coeff_planar * v_air * vmag_air
    F_drag_z = -drag_coeff_vertical * w_air * vmag_air

    delta_F_body = np.array([F_drag_x, F_drag_y, F_drag_z])
    
    # Add directly into total_force_body!
    total_force_body = np.array([F_x, F_y, F_z]) + delta_F_body

    # total_force_body = np.array([F_x, F_y, F_z])
    total_force_inertial = R_body_to_inertial @ total_force_body + gravity_force
    accel = total_force_inertial / MASS
    
    # 3. ROTATIONAL KINEMATICS (Euler Rate Changes)
    # Important: The kinematic rotation of the airframe is driven by its TOTAL angular velocity 
    # relative to the air mass.
    omega_total = omega + np.array([p_turb, q_turb, 0.0])
    
    d_euler = np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ]) @ omega_total

    # 4. ROTATIONAL DYNAMICS (Euler's Equations)
    # The control moments fight against the drone's actual state rates
    p, q, r = omega
    dp = (M_x - (I_zz - I_yy) * q * r) / I_xx
    dq = (M_y - (I_xx - I_zz) * p * r) / I_yy
    dr = (M_z - (I_yy - I_xx) * p * q) / I_zz
    d_omega = np.array([dp, dq, dr])
    
    dstatedt = np.zeros(12)
    dstatedt[0:3]  = vel
    dstatedt[3:6]  = accel
    dstatedt[6:9]  = d_euler
    dstatedt[9:12] = d_omega
    
    return dstatedt

# --- 3. Explicit Runge-Kutta 4th Order Integrator ---
def rk4_step_with_turb(t, state, dt, controls, dryden_ts, plant_params):
    """
    Advances the state vector by a timestep dt using explicit RK4 scheme,
    keeping the control variables frozen across sub-steps.
    """
    k1 = drone_derivatives_with_turb(t,           state,           controls, dryden_ts, aero_params=plant_params)
    k2 = drone_derivatives_with_turb(t + 0.5*dt,  state + 0.5*dt*k1, controls, dryden_ts, aero_params=plant_params)
    k3 = drone_derivatives_with_turb(t + 0.5*dt,  state + 0.5*dt*k2, controls, dryden_ts, aero_params=plant_params)
    k4 = drone_derivatives_with_turb(t + dt,      state + dt*k3,    controls, dryden_ts, aero_params=plant_params)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def run_drone_acoustic_simulation(gains_vector, dryden_ts, velocity_schedule, time_steps, dt, plant_params, initial_state=None):
    """
    Runs a full 6-DoF drone simulation with specified PID gains under turbulence
    and returns the resulting sound pressure level (SPL) metrics.
    
    Parameters:
    -----------
    gains_vector : list or np.array
        [kp_vel, ki_vel, kp_att, kd_att, kp_alt, kd_alt]
    """
    
    # 1. Unpack gains and initialize the controller
    kp_vel, ki_vel, kp_att, kd_att, kp_alt, kd_alt = gains_vector
    controller = DronePIDController(
        kp_vel=kp_vel, ki_vel=ki_vel, 
        kp_att=kp_att, kd_att=kd_att, 
        kp_alt=kp_alt, kd_alt=kd_alt,
        MASS = plant_params.get('MASS',1.5),
        g=plant_params.get('g',9.81)
    )
    
    # 2. Reset / Pre-allocate matrices
    # state = np.zeros(12)
    # state[2] = 0.0  # Start at 5m hover altitude???
    target_yaw = 0.0

    if initial_state is None:
        # Default IC: Drone starting steady at 50m altitude
        state = np.zeros(12)
        state[2] = 50.0  # z0 = 50.0 meters
    else:
        state = np.asarray(initial_state, copy=True, dtype=float)
     

    history_state = np.zeros((len(time_steps), 12))
    history_accel = np.zeros((len(time_steps), 3))
    history_gusts = np.zeros((len(time_steps), 5))  # wu, wv, ww, p_turb, q_turb
    history_controls = np.zeros((len(time_steps), 6))

    # 3. Core Simulation Loop
    for idx, t in enumerate(time_steps):
        target_velocity = velocity_schedule[0][1]
        for switch_time, commanded_velocity in velocity_schedule:
            if t >= switch_time:
                target_velocity = commanded_velocity
            else:
                break

        current_controls, theta_cmd, phi_cmd = controller.compute_controls_velocity_tracking(
            state, target_velocity, target_yaw
        )
        
        dstatedt = drone_derivatives_with_turb(t, state, current_controls, dryden_ts,plant_params)
        history_accel[idx, :] = dstatedt[3:6]

        state = rk4_step_with_turb(t, state, dt, current_controls, dryden_ts,plant_params)
        history_state[idx, :] = state
        history_controls[idx, :] = current_controls

        t_ts = dryden_ts['t']
        history_gusts[idx, 0] = np.interp(t, t_ts, dryden_ts['wu'])
        history_gusts[idx, 1] = np.interp(t, t_ts, dryden_ts['wv'])
        history_gusts[idx, 2] = np.interp(t, t_ts, dryden_ts['ww'])
        history_gusts[idx, 3] = np.interp(t, t_ts, dryden_ts['p_turb'])
        history_gusts[idx, 4] = np.interp(t, t_ts, dryden_ts['q_turb'])

    rpm_result = compute_rotor_rpm_from_controls(time_steps, history_controls, plant_params['L_arm'], 
                                    plant_params['t_ref'], plant_params['rpm_ref'], plant_params, rpm_min=1000.0, rpm_max=8000.0)
    
    # Model Calibration
    p_ref = calibrate_p_ref(
        spl_ref_db=92.0, rpm_ref_measurement=5000.0, # 72 db 
        r_ref=5.0, theta_ref_deg=90.0,
        n_rotors_in_measurement=4, n_exponent=5.0,
    )
    acoustic_params = AcousticParams(rpm_ref=5000.0, p_ref=p_ref, n_exponent=5.0)

    fine_params = FineGridParams(
        fs=48000.0, interp_method="cubic", use_integrated_phase=True,
        disturbance_amplitude_rad=0.05, disturbance_bandwidth_hz=20.0, random_seed=42,
    )

    observer_xyz = (5.0, 0.0, 0.0)

    # Compute Received SPL
    spl_fine = estimate_received_spl_fine(
        time_steps, history_state[:, 0], history_state[:, 1], history_state[:, 2],
        rpm_result["rpm_front"], rpm_result["rpm_rear"],
        rpm_result["rpm_right"], rpm_result["rpm_left"],
        observer_xyz, acoustic_params=acoustic_params, fine_params=fine_params,
    )
    
    pa = compute_zwicker_indicators_windowed(
        spl_fine["p_signal"], fs=fine_params.fs,
        window_s=2.0, hop_s=0.5,
        use_fs_approximation=True,
    )
    # Return metrics you want to evaluate or optimize (e.g. mean SPL or max SPL)
    return spl_fine, pa, rpm_result, history_state, history_controls, history_accel, history_gusts

def generate_dynamic_trajectory(
    vx_max: float, vy_max: float, vz_max: float, t_climb: float, 
    dt: float = 0.01, t_total: float = 4.0, mode: str = 'climb_cruise'
):
    """
    Generates reference trajectories and waypoints for different flight profiles:
    - 'climb_cruise': 3D diagonal climb up to t_climb, then level cruise.
    - 'hover': Stationary hover at [0, 0, 0] velocity.
    - 'fly_by': Pure constant velocity planar flight along Vx (Vy=0, Vz=0).
    """
    time_steps = np.arange(0.0, t_total + dt/2, dt)
    
    # Initialize references to zero
    v_x_reference_array = np.zeros_like(time_steps, dtype=float)
    v_y_reference_array = np.zeros_like(time_steps, dtype=float)
    v_z_reference_array = np.zeros_like(time_steps, dtype=float)
    
    if mode == 'hover':
        # All reference velocities remain 0.0
        # print("mode = hover")
        velocity_schedule_waypoints = [
            (0.0,     np.array([0.0, 0.0, 0.0])),
            (t_total, np.array([0.0, 0.0, 0.0]))
        ]
        
    elif mode == 'fly_by':
        # print("mode = fly by")
        # Constant velocity along X axis, no climbing, no lateral drift
        v_x_reference_array[:] = vx_max
        velocity_schedule_waypoints = [
            (0.0,     np.array([float(vx_max), 0.0, 0.0])),
            (t_total, np.array([float(vx_max), 0.0, 0.0]))
        ]
        
    elif mode == 'climb_cruise':
        # print("mode = climb + cruise")    
        # Your original 3D climb-to-cruise transition
        v_x_reference_array[:] = vx_max
        v_y_reference_array[:] = vy_max
        
        for i, t in enumerate(time_steps):
            if t < t_climb:
                v_z_reference_array[i] = vz_max
            else:
                v_z_reference_array[i] = 0.0
                
        velocity_schedule_waypoints = [
            (0.0,     np.array([float(vx_max), float(vy_max), float(vz_max)])),  
            (t_climb, np.array([float(vx_max), float(vy_max), 0.0]))            
        ]
    else:
        raise ValueError(f"Unknown trajectory mode: {mode}")
        
    return velocity_schedule_waypoints, time_steps, v_x_reference_array, v_y_reference_array, v_z_reference_array


def update_drone_geometry(R_rotor: float, L_arm: float, baseline_params: dict = None):
    """
    Translates raw geometric choices from the co-design optimizer into 
    physically consistent mass, inertia, and aerodynamic scaling factors.
    """
    # Start with a baseline config dictionary if provided, or define defaults
    p = baseline_params.copy() if baseline_params else {}
    
    # 1. Assign the raw geometric values
    p['R_rotor'] = R_rotor
    p['L_arm'] = L_arm
    
    # 2. Update Drone Mass Properties (Approximation)
    # Assume a fixed central chassis mass, but arm mass scales with length
    m_body = 1.2  # kg (central electronics, battery, camera)
    m_per_meter_arm = 0.4  # kg/m for carbon fiber tubes
    m_motor_prop = 0.15  # kg per motor+prop assembly
    
    total_mass = m_body + 4 * (L_arm * m_per_meter_arm + m_motor_prop)
    p['MASS'] = total_mass
    p['g'] = 9.81 
    p['t_ref'] = p['MASS']  * p['g'] /4     # N 
    p['rpm_ref'] = 5000.
    
    # 3. Update Moments of Inertia (Solid Mechanics Matrix)
    # Using parallel axis theorem approximations for a quadcopter layout
    p['Ixx'] = 0.5 * m_body * (0.15**2) + 2 * m_motor_prop * (L_arm**2)
    p['Iyy'] = 0.5 * m_body * (0.15**2) + 2 * m_motor_prop * (L_arm**2)
    p['Izz'] = p['Ixx'] + p['Iyy']  # Perpendicular axis theorem approximation
    
    # 4. Scale Aerodynamic Coefficients Based on Propeller Scaling Laws
    # Thrust coefficient scales roughly with R^4, Torque with R^5
    R_ref = 0.127  # Reference radius (e.g., standard 5-inch prop = 0.127m)
    kt_ref = 2.9e-5
    kq_ref = 1.1e-6
    
    p['kt'] = kt_ref * (R_rotor / R_ref)**4
    p['kq'] = kq_ref * (R_rotor / R_ref)**5
    
    return p

def check_tracking_performance(hist_state, v_z_ref, max_allowable_error=6.0):
    """
    Checks tracking by directly comparing actual velocity against the reference vector.
    """
    # Assuming actual vertical velocity is in column 3 of your state history
    actual_velocity = hist_state[:, 5] 
    
    min_len = min(len(actual_velocity), len(v_z_ref))
    error_vector = actual_velocity[:min_len] - v_z_ref[:min_len]
    
    max_error = np.max(np.abs(error_vector))
    return max_error > max_allowable_error

def calculate_optimization_objective(hist_state, v_x_ref, v_y_ref, v_z_ref, pa, rpm_results, time_steps, mode='climb_cruise'):
    """
    Computes a multi-objective codesign cost function including:
      1. Acoustics (Annoyance / Loudness)
      2. Non-dimensional tracking accuracy (ITAE)
      3. Transit speed performance (Slowness penalty)
      4. Energy/Maneuver Efficiency (Specific Energy Consumption)
    """
    # --- 1. Acoustic Cost ---
    if 'annoyance_PA' in pa and len(pa['annoyance_PA']) > 0:
        acoustic_cost = np.nanmean(pa['annoyance_PA'])
    else:
        acoustic_cost = 99.0

    # --- 2. Initialize Penalties ---
    itae_penalty = 0.0
    slowness_penalty = 0.0
    efficiency_cost = 0.0
    
    vx_act, vy_act, vz_act = hist_state[:, 3], hist_state[:, 4], hist_state[:, 5]
    n_steps = min(len(vx_act), len(v_x_ref))
    dt = time_steps[1] - time_steps[0]

    # --- 3. Compute Electrical Power Profile ---
    # Approximate power: P ~ constant * sum(RPM^3)
    # Replace 1e-9 with your actual motor power scaling coefficient (K_p) if known
    motor_power_coefficient = 1.2e-9 
    power_profile = (
        rpm_results['rpm_front']**3 +
        rpm_results['rpm_rear']**3 +
        rpm_results['rpm_left']**3 +
        rpm_results['rpm_right']**3
    ) * motor_power_coefficient  # Watts
    # rpm_avg = 0.25 * (rpm_results['rpm_front'] + rpm_results['rpm_rear'] + rpm_results['rpm_left'] + rpm_results['rpm_right'])
    # power_profile = np.sum(rpm_avg**3, axis=1) * motor_power_coefficient  # Watts
    total_energy_joules = np.trapezoid(power_profile[:n_steps], time_steps[:n_steps])

    # --- 4. Mode-Dependent Calculations ---
    if mode in ['climb_cruise', 'fly_by']:
        # A. Non-Dimensional ITAE (Tracking Accuracy)
        ref_magnitudes = np.sqrt(v_x_ref**2 + v_y_ref**2 + v_z_ref**2)
        v_char = max(np.max(ref_magnitudes), 1.0)
        
        error_magnitude = np.sqrt(
            (vx_act[:n_steps] - v_x_ref[:n_steps])**2 +
            (vy_act[:n_steps] - v_y_ref[:n_steps])**2 +
            (vz_act[:n_steps] - v_z_ref[:n_steps])**2
        )
        nondim_errors = error_magnitude / v_char
        
        # Calculate ITAE (Integral of Time-weighted Absolute Error)
        time_weights = time_steps[:n_steps]
        itae_penalty = np.trapezoid(time_weights * nondim_errors, time_steps[:n_steps])
        
        # B. Slowness Penalty (Incentivize Speed)
        actual_speed_profile = np.sqrt(vx_act**2 + vy_act**2 + vz_act**2)
        v_avg_actual = np.mean(actual_speed_profile[:n_steps])
        v_design_target = 12.0  # Your target mission speed threshold (m/s)
        
        # Penalizes the design heavily if actual average velocity drops below threshold
        slowness_penalty = max(0.0, (v_design_target / (v_avg_actual + 1e-3)) - 1.0)
        
        # C. Specific Energy Consumption (Efficiency)
        distance_traveled = np.trapezoid(actual_speed_profile[:n_steps], time_steps[:n_steps])
        efficiency_cost = total_energy_joules / max(distance_traveled, 0.1)  # J/m

    elif mode == 'hover':
        # Hover tracking error (how steady it stays at 0 velocity)
        error_magnitude = np.sqrt(vx_act[:n_steps]**2 + vy_act[:n_steps]**2 + vz_act[:n_steps]**2)
        itae_penalty = np.trapezoid(time_steps[:n_steps] * error_magnitude, time_steps[:n_steps])
        
        # For hover, efficiency is simply average hover power (J/s)
        efficiency_cost = total_energy_joules / time_steps[-1]  # Average Watts

    # --- 5. Weighted Objective Synthesis ---
    w_acoustic = 1.0
    w_itae = 2.0
    w_slowness = 10.0      # Active only if drone is too slow in cruise
    w_efficiency = 0.005   # Scale based on magnitude of Joules/meter or Watts

    total_objective = (
        (w_acoustic * acoustic_cost) +
        (w_itae * itae_penalty) +
        (w_slowness * slowness_penalty) +
        (w_efficiency * efficiency_cost)
    )

    return total_objective, acoustic_cost, itae_penalty, slowness_penalty, efficiency_cost

def compute_combined_objective(hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa, vx_max, vz_max, alpha=0.4, beta=0.2, p=2):
    """
    Computes the 3-way multi-objective optimization cost:
    Acoustic Annoyance vs. Tracking Accuracy (ITAE) vs. Speed Aggressiveness.
    """
    # 1. Calculate Annoyance Dose
    acoustic_time_grid = np.linspace(0, time_steps[-1], len(pa["annoyance_PA"]))
    annoyance_dose = np.trapezoid(pa["annoyance_PA"], x=acoustic_time_grid)
    
    # 2. Calculate 3D Velocity ITAE (Safe length matching)
    min_len = min(len(hist_state), len(time_steps))
    t_weighted = time_steps[:min_len]
    
    err_x = np.abs(v_x_ref[:min_len] - hist_state[:min_len, 3])
    err_y = np.abs(v_y_ref[:min_len] - hist_state[:min_len, 4])
    err_z = np.abs(v_z_ref[:min_len] - hist_state[:min_len, 5])
    
    itae_value = np.trapezoid(t_weighted * (err_x + err_y + err_z), x=t_weighted)
    
    # 3. Calculate Speed Penalty (Direct Linear Inverse Barrier to coax SNOPT)

    vx_ref = 10.
    vz_ref = 5. 
    epsilon = 0.1
    speed_penalty = 1.0 / ((vx_max/vx_ref)**p + (vz_max/vz_ref)**p + epsilon)
    
    # 4. Scaling Factors (keeps metrics balanced in a similar order of magnitude)
    pa_scale    = 0.025   
    itae_scale  = 0.2   
    speed_scale = 10.   
    
    # 5. Formulate Scaled Combined Cost
    acoustic_part = (1.0 - alpha - beta) * (annoyance_dose * pa_scale)
    itae_part     = alpha * (itae_value * itae_scale)
    speed_part    = beta * (speed_penalty * speed_scale)
    
    combined_cost = acoustic_part + itae_part + speed_penalty
    
    return float(combined_cost), annoyance_dose, itae_value, speed_penalty 

def evaluate_drone_codesign(X, initial_state, dt, t_end, debug=False, mode='climb_cruise'):
    """
    The standardized callable interface for the optimizer.
    """
    # 1. UNPACK THE DESIGN VECTOR
    R_rotor   = float(X[0])
    L_arm     = float(X[1])
    vz_max     = float(X[2])   
    t_climb   = float(X[3])
    vx_max    = float(X[4])   
    vy_max    = float(X[5])   
    gains     = [float(val) for val in X[6:12]]

    # 2. GENERATE ENVIRONMENTAL WIND NOISE & TARGET TRAJECTORY
    t_dryden = np.arange(0, t_end + dt, dt)
    dryden_ts = generate_dryden_time_series(
        t_dryden,
        params=DrydenParams(
            V=5.0, sigma_u=1.5, sigma_v=1.5, sigma_w=0.75,
            L_u=200.0, L_v=200.0, L_w=50.0,
            arm_length=L_arm,  
            v_ref=5.0, z_ref=20.0,
        ),
        seed=42,
        altitude=50.0,
    )

    waypoints, time_steps, v_x_ref, v_y_ref, v_z_ref = generate_dynamic_trajectory(
        vx_max, vy_max, vz_max, t_climb, dt=dt, t_total=t_end, mode=mode
    )
    
    drone_params = update_drone_geometry(R_rotor, L_arm)

    if debug:
        # --- BYPASS MODE (Runs raw without try/except net for easier bug hunting) ---
        outputs = run_drone_acoustic_simulation(
            gains, dryden_ts, waypoints, time_steps, dt=dt, plant_params=drone_params, initial_state=initial_state
        )
        spl_fine, pa, rpm_results, hist_state, _, _, _ = outputs
        
        # Call separated objective engine (Weights matching optimizer config: alpha=0.4, beta=0.2)
        # cost, noise, itae, speed_penalty = compute_combined_objective(
        #     hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa, vx_max, vz_max, alpha=0.4, beta=0.2, p=1
        # )

        total_objective, acoustic_cost, itae_penalty, slowness_penalty, efficiency_cost = calculate_optimization_objective(hist_state, v_x_ref, v_y_ref, v_z_ref, pa, rpm_results, time_steps, mode='climb_cruise')
        
        result_dict = {"objective": total_objective, "feasible": True}
        return result_dict, hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa, rpm_results, spl_fine
    
    else:
        # --- OPTIMIZER MODE ---
        try:
            outputs = run_drone_acoustic_simulation(
                gains, dryden_ts, waypoints, time_steps, dt=dt, plant_params=drone_params, initial_state=initial_state
            )
            spl_fine, pa, rpm_result, hist_state, _, _, _ = outputs
            # spl_fine, pa, rpm_result, history_state, history_controls, history_accel, history_gusts
            # Enforce the tracking safety constraint during optimization
            if check_tracking_performance(hist_state, v_z_ref):
                return {"objective": 99.0, "feasible": False}
                
            # Call separated objective engine
            # cost, noise, itae, speed_penalty = compute_combined_objective(
            #     hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa, vx_max, vz_max, alpha=0.4, beta=0.2, p=1
            # )

            total_objective, acoustic_cost, itae_penalty, slowness_penalty, efficiency_cost = calculate_optimization_objective(hist_state, v_x_ref, v_y_ref, v_z_ref, pa, rpm_result, time_steps, mode='climb_cruise')

            print(f"Cost: {total_objective:.4f} (Noise: {acoustic_cost:.2f}, ITAE: {itae_penalty:.2f}, Speed: {slowness_penalty:.2f}), Efficiency: {efficiency_cost:.2f}))")
            # print(f"Cost: {cost:.4f} (Noise: {noise:.2f}, ITAE: {itae:.2f}, Speed: {speed_penalty:.2f}))")
            return {"objective": total_objective, "feasible": True}
            
        except Exception as e:
            import traceback
            print(f"\n💥 CRASH DETECTED: {e}")
            traceback.print_exc() 
            return {"objective": 99.0, "feasible": False}
    
def plot_codesign_debug_results(hist_state, time_steps, v_z_ref):
    """
    Generates diagnostic plots to verify the physical soundness 
    of the drone's trajectory tracking and dynamic behavior.
    """
    # Assuming standard state column mapping:
    # Column 2: Position Z (Altitude)
    # Column 3: Velocity Z (Climb Rate)
    actual_alt = hist_state[:, 2]
    actual_v_z = hist_state[:, 5]
    
    # Create a clean, professional 2-panel figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- Top Panel: Velocity Tracking Profile ---
    ax1.plot(time_steps, v_z_ref, 'r--', linewidth=2, label='Target Vz (Reference)')
    ax1.plot(time_steps[:len(actual_v_z)], actual_v_z, 'b-', linewidth=2, label='Actual Vz (Drone)')
    ax1.set_ylabel('Vertical Velocity [m/s]', fontsize=11)
    ax1.set_title('Co-Design Simulation Physics Verification', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # --- Bottom Panel: Altitude / Flight Path Profile ---
    ax2.plot(time_steps[:len(actual_alt)], actual_alt, 'g-', linewidth=2, label='Actual Altitude (Z)')
    ax2.set_xlabel('Time [seconds]', fontsize=11)
    ax2.set_ylabel('Altitude [meters]', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()

def plot_comprehensive_diagnostics(hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa, 
                                   gains=None, gain_bounds=None, rpm_history=None):
    """
    Generates four independent diagnostic dashboards:
    1. Flight Dynamics Panel (Kinematics & State Trajectories)
    2. Psychoacoustic Panel (All calculated sound metrics)
    3. Actuator Effort Panel (Rotor RPMs over time)
    4. Control Gain Space (Optimized gains vs. Upper/Lower search bounds)
    """
    # Slice limits for safe length matching
    actual_x   = hist_state[:, 0]
    actual_y   = hist_state[:, 1]
    actual_alt = hist_state[:, 2]
    actual_vx  = hist_state[:, 3]
    actual_vy  = hist_state[:, 4]
    actual_vz  = hist_state[:, 5]
    
    # ----------------------------------------------------
    # DASHBOARD 1: FLIGHT DYNAMICS
    # ----------------------------------------------------
    fig1, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time_steps, v_z_ref, 'r--', linewidth=2, label='Target Vz')
    axs[0].plot(time_steps[:len(actual_vz)], actual_vz, 'b-', linewidth=2, label='Actual Vz')
    axs[0].set_ylabel('Velocity Z [m/s]')
    axs[0].set_title('Flight Dynamics Panel', fontsize=12, fontweight='bold')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='upper right')
    
    axs[1].plot(time_steps, v_x_ref, 'm--', linewidth=1.5, alpha=0.8, label='Target Vx')
    axs[1].plot(time_steps[:len(actual_vx)], actual_vx, 'm-', label='Actual Vx')
    axs[1].plot(time_steps[:len(actual_vy)], actual_vy, 'c-', label='Actual Vy')
    axs[1].set_ylabel('Planar Speed [m/s]')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(loc='upper right')
    
    axs[2].plot(time_steps[:len(actual_x)], actual_x, 'b-', linewidth=2, label='Position (X)')
    axs[2].plot(time_steps[:len(actual_y)], actual_y, 'r-', linewidth=2, label='Position (Y)')
    axs[2].plot(time_steps[:len(actual_alt)], actual_alt, 'g-', linewidth=2, label='Altitude (Z)')
    axs[2].set_xlabel('Time [seconds]')
    axs[2].set_ylabel('Position / Distance [meters]') 
    axs[2].grid(True, linestyle=':', alpha=0.6)
    axs[2].legend(loc='lower right')
    plt.tight_layout()

    # ----------------------------------------------------
    # DASHBOARD 2: FULL PSYCHOACOUSTIC PROFILE
    # ----------------------------------------------------
    acoustic_metadata = {
        'loudness_sone':     {'label': 'Loudness [Sone]',        'color': '#2980B9'},
        'sharpness_acum':    {'label': 'Sharpness [Acum]',       'color': '#8E44AD'},
        'roughness_asper':   {'label': 'Roughness [Asper]',      'color': '#16A085'},
        'fluctuation_vacil': {'label': 'Fluctuation [Vacil]',    'color': '#F39C12'},
        'annoyance_PA':      {'label': 'Total Annoyance [PA]',   'color': '#D35400'}
    }
    active_keys = [k for k in acoustic_metadata.keys() if k in pa]
    num_plots = len(active_keys)
    
    fig2, axs2 = plt.subplots(num_plots, 1, figsize=(10, 2.2 * num_plots), sharex=True)
    if num_plots == 1: axs2 = [axs2]
        
    for idx, key in enumerate(active_keys):
        signal = np.asarray(pa[key], dtype=float)
        t_acoustic = np.linspace(0, time_steps[-1], len(signal))
        is_valid = ~np.isnan(signal)
        meta = acoustic_metadata[key]
        if np.any(is_valid):
            axs2[idx].plot(t_acoustic[is_valid], signal[is_valid], color=meta['color'], linewidth=2, label=meta['label'])
        else:
            axs2[idx].text(0.5, 0.5, 'Metric Data Unavailable', transform=axs2[idx].transAxes, ha='center', va='center', color='gray', fontstyle='italic')
        axs2[idx].set_ylabel(meta['label'], fontsize=10, fontweight='bold')
        axs2[idx].grid(True, linestyle=':', alpha=0.5)
    axs2[-1].set_xlabel('Time [seconds]', fontsize=11)
    fig2.suptitle('Complete Psychoacoustic Sound Footprint', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    # ----------------------------------------------------
    # NEW DASHBOARD 3: ROTOR ACTUATOR EFFORT (RPMs)
    # ----------------------------------------------------
    if rpm_history is not None:
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4))
        num_rotors = rpm_history.shape[1] if len(rpm_history.shape) > 1 else 1
        
        t_rpm = time_steps[:len(rpm_history)]
        if num_rotors > 1:
            for r in range(num_rotors):
                ax3.plot(t_rpm, rpm_history[:, r], linewidth=1.5, alpha=0.8, label=f'Rotor {r+1}')
        else:
            ax3.plot(t_rpm, rpm_history, color='#2C3E50', linewidth=2, label='Rotor System')
            
        ax3.set_title('Actuator Motor Commands', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time [seconds]')
        ax3.set_ylabel('Rotor Velocity [RPM]')
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(loc='upper right')
        plt.tight_layout()

    # ----------------------------------------------------
    # NEW DASHBOARD 4: CONTROL GAINS BOUNDARY CONSTRAINTS
    # ----------------------------------------------------
    if gains is not None and gain_bounds is not None:
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 5))
        gain_labels = ['Kp_vel', 'Ki_vel', 'Kp_att', 'Kd_att', 'Kp_alt', 'Kd_alt']
        
        # Unpack lower and upper boundary vectors for plotting bounds windows
        low_bounds = [b[0] for b in gain_bounds]
        upp_bounds = [b[1] for b in gain_bounds]
        x_indices = np.arange(len(gains))
        
        # Plot boundary indicators using high-visibility markers
        ax4.scatter(x_indices, low_bounds, color='#C0392B', marker='_', s=300, linewidth=3, label='Lower Limit')
        ax4.scatter(x_indices, upp_bounds, color='#27AE60', marker='_', s=300, linewidth=3, label='Upper Limit')
        
        # Plot optimized design points as prominent filled dots
        ax4.scatter(x_indices, gains, color='#2980B9', marker='o', s=100, zorder=3, label='Optimized Value')
        
        # Draw dotted connectors between boundaries so it looks like a clean variance chart
        for i in range(len(gains)):
            ax4.vlines(i, low_bounds[i], upp_bounds[i], colors='gray', linestyles=':', alpha=0.7)
            # Add a visual flag if a gain is hugging a boundary edge closely
            if np.isclose(gains[i], low_bounds[i], rtol=1e-2) or np.isclose(gains[i], upp_bounds[i], rtol=1e-2):
                ax4.text(i, gains[i], ' [BOUNDED]', color='orange', fontsize=9, fontweight='bold', va='bottom', ha='center')

        ax4.set_title('Optimized Controller Gains vs. Design Search Space Bounds', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_indices)
        ax4.set_xticklabels(gain_labels, fontweight='bold')
        ax4.set_ylabel('Gain Numeric Value')
        ax4.set_yscale('log')  # Uses log scale since Kp_alt (20.0) vs Ki_vel (0.01) have massive step changes
        ax4.grid(True, which="both", linestyle=':', alpha=0.4)
        ax4.legend(loc='upper right')
        plt.tight_layout()
        
    plt.show()

if __name__ == "__main__":

    dt = 0.002 
    t_end = 12.01
    vz_max = 5.0 
    vx_max = 5.00 
    vy_max = 0. 
    t_climb = 3.0

    # Define a test drone configuration (middle of the bounds)
    test_X = [0.22, 0.25,  vz_max, t_climb, vx_max, vy_max, 0.15, 0.04, 3.0, 0.5, 15.0, 7.0]
    
    # # --- ADD THIS TEMPORARY PRINT BLOCK TO YOUR MAIN ---
    # outputs_traj = generate_dynamic_trajectory(
    #     vx_max, vy_max, vx_max, t_climb, dt=dt, t_total=t_end,
    #     mode='hover'
    # )

    # waypoints, time_steps, v_x_ref, v_y_ref, v_z_ref = outputs_traj

    # Set drone starting position to 10m up, sitting perfectly still
    my_init_state = np.zeros(12)
    my_init_state[2] = 7.76 # 10 meters altitude

    # print("--- TRAJECTORY ARRAY DIAGNOSIS ---")
    # print(f"Shape of time_steps:        {time_steps.shape}")
    # # print(f"Shape of velocity_schedule: {velocity_schedule.shape}")
    # print(f"Expected dt:                {dt}")
    # print(f"Calculated array length:    {len(time_steps)}")
    # print("---------------------------------")

    print("Testing the co-design callable wrapper...")
    # test_result, hist_state, time_steps, v_x_ref, v_z_ref, pa = evaluate_drone_codesign(test_X, dt=dt, t_end=t_end, debug = True)

    test_result, hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa, rpm_results, spl_fine = evaluate_drone_codesign(
    test_X, initial_state=my_init_state, dt=dt, t_end=t_end, debug=True, mode='climb_cruise'
    )
       
    # 3. Execute your new Frequency Domain Debugger!
    spectrum = compute_acoustic_spectrum_debug(spl_fine)
    
    # 4. Render the spectrum graph (zoomed to 800 Hz to see key harmonics)
    plot_acoustic_spectrum(spectrum, max_freq_display=800.0)

    print("\n--- Evaluation Test Results ---")
    print(f"Objective (Annoyance Dose): {test_result['objective']}")
    print(f"Is Configuration Feasible?: {test_result['feasible']}")

    # if test_result['feasible'] and hist_state is not None:
    #     plot_codesign_debug_results(hist_state, time_steps, v_z_ref)
    
    rpm_history = np.column_stack((
    rpm_results["rpm_front"],
    rpm_results["rpm_left"],
    rpm_results["rpm_right"],
    rpm_results["rpm_rear"]
    ))

    if test_result['feasible'] and hist_state is not None and pa is not None:
        plot_comprehensive_diagnostics(hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa, gains = None, gain_bounds=None, rpm_history=rpm_history)
