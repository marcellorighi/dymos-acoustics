import numpy as np

# --- 1. Global System Parameters ---
MASS = 1.5              # kg
I_xx = 0.012            # kg*m^2
I_yy = 0.012            # kg*m^2
I_zz = 0.020            # kg*m^2
g = 9.81                # m/s^2
arm_length = 0.25       # m 
t_ref = MASS * g /4     # N 
rpm_ref = 5000.         # RPM 

# State Vector Index Map:
# x[0:3]  = Position (x, y, z) in Inertial Frame
# x[3:6]  = Velocity (vx, vy, vz) in Inertial Frame
# x[6:9]  = Euler Angles (phi, theta, psi) -> Roll, Pitch, Yaw
# x[9:12] = Angular Velocities (p, q, r) in Body Frame

import numpy as np

def compute_rotor_rpm_from_controls(time_steps, history_controls, arm_length, 
                                    t_ref, rpm_ref, rpm_min=1000.0, rpm_max=8000.0) -> dict:
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
    
    # Differential thrust allocations mapping moments through the lever arm length L
    F_front = thrust_avg + M_y / (2.0 * arm_length)
    F_rear  = thrust_avg - M_y / (2.0 * arm_length)
    F_right = thrust_avg + M_x / (2.0 * arm_length)
    F_left  = thrust_avg - M_x / (2.0 * arm_length)
    
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
        rpm = np.sqrt(np.abs(F) / kT)
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
    def __init__(self, kp_vel=0.15, ki_vel=0.05, kp_att=2.5, kd_att=0.4, kp_alt=15.0, kd_alt=10.0):
        # Assign custom gains passed from the optimizer/simulation function
        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.kp_alt = kp_alt
        self.kd_alt = kd_alt
        
        # State tracking (Integrators)
        self.integral_vx = 0.0
        self.integral_vy = 0.0
        self.dt = 0.01

    def compute_controls_velocity_tracking(self, state, target_velocity, target_yaw):
        vx, vy, vz = state[3:6]
        phi, theta, psi = state[6:9]
        p, q, r = state[9:12]
        
        # Calculate raw tracking errors
        error_vx = target_velocity[0] - vx
        error_vy = target_velocity[1] - vy
        
        # Update the integrators (accumulate error over time)
        self.integral_vx += error_vx * self.dt
        self.integral_vy += error_vy * self.dt
        
        # Anti-Windup Clamp: Protect the integrators from expanding to infinity
        self.integral_vx = np.clip(self.integral_vx, -2.0, 2.0)
        self.integral_vy = np.clip(self.integral_vy, -2.0, 2.0)
        
        # --- PI Control Laws ---
        # Pitch combines current error + accumulated historical error
        # theta_cmd = -1.0 * (self.kp_vel * error_vx + self.ki_vel * self.integral_vx)
        
        # # Roll combines current error + accumulated historical error
        # phi_cmd   = +1.0 * (self.kp_vel * error_vy + self.ki_vel * self.integral_vy)
        
        # # Safety limits on maximum body tilt angles
        # theta_cmd = np.clip(theta_cmd, -0.4, 0.4)
        # phi_cmd   = np.clip(phi_cmd, -0.4, 0.4)
        
        # 1. Calculate raw tracking errors
        error_vx = target_velocity[0] - vx
        error_vy = target_velocity[1] - vy
        
        # 2. Accumulate historical errors
        self.integral_vx += error_vx * self.dt
        self.integral_vy += error_vy * self.dt
        
        # Anti-Windup Clamps
        self.integral_vx = np.clip(self.integral_vx, -2.0, 2.0)
        self.integral_vy = np.clip(self.integral_vy, -2.0, 2.0)
        
        # --- TESTED CALIBRATION COEFFS ---
        # Change these multipliers between 1.0 and -1.0 to match your physics engine
        SIGN_X = +1.0  
        SIGN_Y = -1.0  
        
        # Compute the explicit commands
        theta_cmd = SIGN_X * (self.kp_vel * error_vx + self.ki_vel * self.integral_vx)
        phi_cmd   = SIGN_Y * (self.kp_vel * error_vy + self.ki_vel * self.integral_vy)

        # Altitude and Inner Loops remain the same
        F_z = self.kp_alt * (target_velocity[2] - vz) + (MASS * g)
        M_x = self.kp_att * (phi_cmd - phi) + self.kd_att * (0.0 - p)
        M_y = self.kp_att * (theta_cmd - theta) + self.kd_att * (0.0 - q)
        M_z = self.kp_att * (target_yaw - psi) + self.kd_att * (0.0 - r)
        
        controls = np.array([0.0, 0.0, F_z, M_x, M_y, M_z])
        return controls, theta_cmd, phi_cmd

def drone_derivatives_with_turb(t, state, controls, dryden_ts):
    """
    6-DoF Drone Equations of Motion with injected Dryden Turbulence.
    """
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
    total_force_body = np.array([F_x, F_y, F_z])
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
def rk4_step_with_turb(t, state, dt, controls, dryden_ts):
    """
    Advances the state vector by a timestep dt using explicit RK4 scheme,
    keeping the control variables frozen across sub-steps.
    """
    k1 = drone_derivatives_with_turb(t,           state,           controls, dryden_ts)
    k2 = drone_derivatives_with_turb(t + 0.5*dt,  state + 0.5*dt*k1, controls, dryden_ts)
    k3 = drone_derivatives_with_turb(t + 0.5*dt,  state + 0.5*dt*k2, controls, dryden_ts)
    k4 = drone_derivatives_with_turb(t + dt,      state + dt*k3,    controls, dryden_ts)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def run_drone_acoustic_simulation(gains_vector, dryden_ts, velocity_schedule, time_steps, dt):
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
        kp_alt=kp_alt, kd_alt=kd_alt
    )
    
    # 2. Reset / Pre-allocate matrices
    state = np.zeros(12)
    state[2] = 5.0  # Start at 5m hover altitude
    target_yaw = 0.0
    
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
        
        dstatedt = drone_derivatives_with_turb(t, state, current_controls, dryden_ts)
        history_accel[idx, :] = dstatedt[3:6]

        state = rk4_step_with_turb(t, state, dt, current_controls, dryden_ts)
        history_state[idx, :] = state
        history_controls[idx, :] = current_controls

        t_ts = dryden_ts['t']
        history_gusts[idx, 0] = np.interp(t, t_ts, dryden_ts['wu'])
        history_gusts[idx, 1] = np.interp(t, t_ts, dryden_ts['wv'])
        history_gusts[idx, 2] = np.interp(t, t_ts, dryden_ts['ww'])
        history_gusts[idx, 3] = np.interp(t, t_ts, dryden_ts['p_turb'])
        history_gusts[idx, 4] = np.interp(t, t_ts, dryden_ts['q_turb'])

    # 4. Acoustic Pipeline Processing
    # Estimate Coarse RPMs
    # rpm_result = estimate_rotor_rpm(
    #     t=time_steps, x=history_state[:, 0], y=history_state[:, 1], z=history_state[:, 2],
    #     vx=history_state[:, 3], vy=history_state[:, 4], vz=history_state[:, 5],
    #     ax=history_accel[:, 0], ay=history_accel[:, 1], az=history_accel[:, 2],
    #     wx=history_gusts[:, 0], wy=history_gusts[:, 1], wz=history_gusts[:, 2]
    # )

    rpm_result = compute_rotor_rpm_from_controls(time_steps, history_controls, arm_length, 
                                    t_ref, rpm_ref, rpm_min=1000.0, rpm_max=8000.0)
    
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

    observer_xyz = (25.0, 2.0, 0.0)

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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from dryden_timeseries import generate_dryden_time_series, DrydenParams
from rotor_rpm_estimation import estimate_rotor_rpm
from drone_acoustic_radiation_v2 import (
        calibrate_p_ref, AcousticParams, FineGridParams,
        estimate_received_spl_fine,
    )
from zwicker_annoyance_v2 import compute_zwicker_indicators_windowed
import pandas as pd
from scipy.stats import qmc  # Requires scipy >= 1.7.0 for Latin Hypercube
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol as sobol_analyzer
from tqdm import tqdm

if __name__ == "__main__":
    # Define time domain and environment setup parameters
    dt = 0.01  
    t_end = 2.0
    time_steps = np.arange(0, t_end, dt)
    
    t_dryden = np.arange(0, t_end + dt, dt)

    dryden_ts = generate_dryden_time_series(
        t_dryden,
        params=DrydenParams(
            V=5.0,          # mean airspeed [m/s] -- your best estimate
            sigma_u=1.5,    # longitudinal intensity [m/s]
            sigma_v=1.5,    # lateral
            sigma_w=0.75,   # vertical
            L_u=200.0,      # length scales [m]
            L_v=200.0,
            L_w=50.0,
            arm_length=0.25,
            v_ref=5.0,
            z_ref=20.0,
        ),
        seed=42,            # fix for reproducibility; vary for ensemble runs
        altitude=50.0,      # representative flight altitude
        )

    # Setup your velocity schedule and dryden_ts here...
    velocity_schedule = [
    (0.0,  np.array([0.0, 0.0, 1.0])),   # 0 to 2s: Climb straight up at 2 m/s
    (2.0,  np.array([3.0, 0.0, 0.0])),   # 2 to 5s: Accelerate forward to 3 m/s, hold altitude
    (5.0,  np.array([5.0, 1.0, 0.0])),   # 5 to 8s: Push forward to 5 m/s, drift right at 1 m/s
    (8.0,  np.array([5.0, 0.0, -0.5])),  # 8 to 12s: Keep forward speed, begin slow descent
    (12.0, np.array([2.0, -2.0, 0.0])),  # 12 to 15s: Slow down X, fly left at 2 m/s
    (15.0, np.array([0.0, 0.0, 0.0])),   # 15 to 18s: Brake completely and hover in place
    (18.0, np.array([-3.0, 0.0, 1.0])),  # 18 to 22s: Fly backward at 3 m/s while climbing
    (22.0, np.array([0.0, 4.0, 0.0])),   # 22 to 25s: Fly hard right at 4 m/s
    (25.0, np.array([1.0, 1.0, -1.0])),  # 25 to 28s: Slow diagonal descent
    (28.0, np.array([0.0, 0.0, 0.0]))    # 28s+: Final hover profile
    ] 
    
    # 1. Define the exact parameter structure for SALib
    custom_gains_matrix = [
        [0.05, 0.01, 2.5, 0.4, 15.0, 10.0],  # Configuration 1: Soft Tuning Baseline
        [0.15, 0.04, 3.0, 0.5, 18.0, 12.0],  # Configuration 2: Moderate Balanced Tracking
        [0.30, 0.10, 4.5, 0.8, 22.0, 14.0],  # Configuration 3: Aggressive High-Stiffness Loop
        [0.02, 0.005, 1.5, 0.2, 8.0,  5.0 ]   # Configuration 4: Ultra-Dampened Smooth Profile
    ]

    # Explicit names to easily keep track of which run is which in your plots
    config_names = [
        "Soft_Baseline",
        "Moderate_Balanced",
        "Aggressive_Stiff",
        "Ultra_Dampened"
    ]

    compiled_results = []

    print(f"Beginning evaluation of {len(custom_gains_matrix)} fixed gain profiles...")

    # 2. Iterate using standard Python list unpackings
    for idx, full_gains in enumerate(tqdm(custom_gains_matrix, desc="Evaluating Fixed Profiles")):
        run_name = config_names[idx]
        
        try:
            # Execute the simulation using the precise list profile
            outputs = run_drone_acoustic_simulation(
                full_gains, dryden_ts, velocity_schedule, time_steps, dt
            )
            spl_fine, pa, _, hist_state, _, _, _ = outputs
            
            if np.any(np.isnan(hist_state)) or np.any(np.isinf(hist_state)):
                raise ValueError("Simulation diverged internally.")
                
            # Extract time signal and calculate the cumulative dose
            annoyance_time_signal = np.asarray(pa["annoyance_PA"], dtype=float)
            pa_dose = np.trapezoid(annoyance_time_signal, x=time_steps)
            max_annoyance = float(np.nanmax(annoyance_time_signal))
            
        except Exception as e:
            print(f"\n  -> Profile '{run_name}' encountered an error! Filling with penalties.")
            pa_dose = 99.0
            max_annoyance = 99.0
            annoyance_time_signal = np.full_like(time_steps, 99.0)

        # 3. Save into a clean Pandas structure
        compiled_results.append({
            'Profile_Name': run_name,
            'Kp_Velocity': full_gains[0],
            'Ki_Velocity': full_gains[1],
            'Kp_Attitude': full_gains[2],
            'Kd_Attitude': full_gains[3],
            'Kp_Altitude': full_gains[4],
            'Kd_Altitude': full_gains[5],
            'PA_Dose': pa_dose,
            'Max_Annoyance': max_annoyance,
            'Full_Annoyance_Array': annoyance_time_signal
        })

    df_fixed = pd.DataFrame(compiled_results)
    print("\nEvaluation Complete.")
    print(df_fixed[['Profile_Name', 'PA_Dose', 'Max_Annoyance']])



