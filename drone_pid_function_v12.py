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


# --- 1. Global System Parameters ---
MASS = 1.5              # kg
I_xx = 0.012            # kg*m^2
I_yy = 0.012            # kg*m^2
I_zz = 0.020            # kg*m^2
g = 9.81                # m/s^2
arm_length = 0.25       # m 
t_ref = MASS * g /4     # N 
rpm_ref = 5000.         # RPM 

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

def run_drone_acoustic_simulation(gains_vector, dryden_ts, velocity_schedule, time_steps, dt, plant_params):
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

    observer_xyz = (20.0, 0.0, 0.0)

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

def generate_dynamic_trajectory(vx_max: float, vy_max: float, v_max: float, t_climb: float, dt: float = 0.01, t_total: float = 4.0):
    time_steps = np.arange(0.0, t_total + dt/2, dt)
    
    # 1. WAYPOINTS: Fully 3D Climb + Lateral Translation, then transition to pure level cruise
    velocity_schedule_waypoints = [
        # Phase 1: Full 3D Diagonal Climb (moving along X, Y, and Z simultaneously)
        (0.0,     np.array([float(vx_max), float(vy_max), float(v_max)])),  
        # Phase 2: Level Planar Cruise (Z velocity drops to 0, holding X and Y speeds)
        (t_climb, np.array([float(vx_max), float(vy_max), 0.0]))            
    ]
    
    # 2. MATCHING REFERENCE ARRAYS FOR ALL THREE AXES
    v_x_reference_array = np.zeros_like(time_steps, dtype=float)
    v_y_reference_array = np.zeros_like(time_steps, dtype=float)
    v_z_reference_array = np.zeros_like(time_steps, dtype=float)
    
    for i, t in enumerate(time_steps):
        # Planar velocities are maintained across the entire flight profile
        v_x_reference_array[i] = vx_max  
        v_y_reference_array[i] = vy_max  
        
        # Vertical velocity cuts off cleanly at t_climb
        if t < t_climb:
            v_z_reference_array[i] = v_max
        else:
            v_z_reference_array[i] = 0.0
            
    # Return all three reference arrays for the physics loop and plotting tools
    return velocity_schedule_waypoints, time_steps, v_x_reference_array, v_y_reference_array, v_z_reference_array

# def generate_dynamic_trajectory(v_max: float, t_climb: float, dt: float = 0.01, t_total: float = 4.0):
#     time_steps = np.arange(0.0, t_total + dt/2, dt)
    
#     # 1. The waypoint list format (What the SIMULATOR core needs)
#     velocity_schedule_waypoints = [
#         (0.0,     np.array([0.0, 0.0, float(v_max)])),
#         (t_climb, np.array([0.0, 0.0, 0.0]))
#     ]
    
#     # 2. A clean flat array of the Z-velocities over time (What the ERROR CHECKER needs)
#     v_z_reference_array = np.zeros_like(time_steps, dtype=float)
#     for i, t in enumerate(time_steps):
#         if t < t_climb:
#             v_z_reference_array[i] = (v_max / 2.0) * (1.0 - np.cos(np.pi * t / t_climb))
#         else:
#             v_z_reference_array[i] = v_max
            
#     return velocity_schedule_waypoints, time_steps, v_z_reference_array

# def generate_dynamic_trajectory(v_max: float, t_climb: float, dt: float = 0.01, t_total: float = 4.0):
#     """
#     Generates a smooth, time-varying target velocity profile for the drone using
#     a cosine-accelerated transition ramp.
    
#     Parameters:
#     -----------
#     v_max : float
#         The maximum target tracking velocity reached during the maneuver [m/s].
#     t_climb : float
#         The duration of the active acceleration/climb phase [s]. Must be <= t_total.
#     dt : float, optional
#         The simulation time step [s]. Default is 0.01.
#     t_total : float, optional
#         The total flight duration for the evaluation window [s]. Default is 4.0.
        
#     Returns:
#     --------
#     velocity_schedule : np.ndarray
#         Array of target velocities corresponding to each step in time_steps.
#     time_steps : np.ndarray
#         The uniform time grid of the flight simulation.
#     """
#     # 1. Establish the clean master time grid
#     time_steps = np.arange(0.0, t_total + dt/2, dt)
    
#     # Initialize an array of zeros matching the time grid size
#     velocity_schedule = np.zeros_like(time_steps, dtype=float)
    
#     # 2. Build the smooth trajectory profile across time
#     for i, t in enumerate(time_steps):
#         if t < t_climb:
#             # Cosine acceleration ramp: smoothly transitions from 0.0 to v_max
#             # Acceleration starts at 0, peaks in the middle, and tapers to 0 at t_climb.
#             velocity_schedule[i] = (v_max / 2.0) * (1.0 - np.cos(np.pi * t / t_climb))
#         else:
#             # Steady-state phase: hold the maximum velocity for the rest of the run
#             velocity_schedule[i] = v_max
            
#     return velocity_schedule, time_steps

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
    p['mass'] = total_mass
    
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

def evaluate_drone_codesign(X, dt, t_end, debug=False):
    """
    The standardized callable interface for the optimizer with integrated multi-objective ITAE.
    """
    # 1. UNPACK THE DESIGN VECTOR
    R_rotor   = float(X[0])
    L_arm     = float(X[1])
    v_max     = float(X[2])   
    t_climb   = float(X[3])
    vx_max    = float(X[4])   
    vy_max    = float(X[5])   
    gains     = [float(val) for val in X[6:12]]

    # 2. GENERATE TIME GRIDS & ENVIRONMENTAL WIND NOISE
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

    # 3. GENERATE TARGET TRAJECTORY
    outputs_traj = generate_dynamic_trajectory(
        vx_max, vy_max, v_max, t_climb, dt=dt, t_total=t_end
    )
    waypoints, time_steps, v_x_ref, v_y_ref, v_z_ref = outputs_traj
    
    drone_params = update_drone_geometry(R_rotor, L_arm)

    # --- MULTI-OBJECTIVE WEIGHT BALANCING CONFIGURATION ---
    # alpha = 0.0 -> Minimizes pure acoustic noise (lazy tracking)
    # alpha = 1.0 -> Minimizes pure tracking error (very loud/aggressive motors)
    alpha = 0.4  
    
    # Normalization scaling factor (keeps ITAE magnitude visually balanced with Sone/PA units)
    itae_scale = 1.0  
    pa_scale = 0.1 

    if debug:
        # --- BYPASS MODE ---
        outputs = run_drone_acoustic_simulation(
            gains, dryden_ts, waypoints, time_steps, dt=dt, plant_params=drone_params
        )
        spl_fine, pa, _, hist_state, _, _, _ = outputs
        
        # if check_tracking_performance(hist_state, v_z_ref):
        #     print("🛑 DEBUG INFO: Simulation ran, but failed tracking performance!")
        #     return {"objective": 99.0, "feasible": False}, None, None, None, None, None
            
        # A. Calculate Annoyance Dose
        acoustic_time_grid = np.linspace(0, time_steps[-1], len(pa["annoyance_PA"]))
        annoyance_dose = np.trapezoid(pa["annoyance_PA"], x=acoustic_time_grid)
        
        # B. Calculate 3D Velocity ITAE (Safe length matching)
        min_len = min(len(hist_state), len(time_steps))
        t_weighted = time_steps[:min_len]
        
        err_x = np.abs(v_x_ref[:min_len] - hist_state[:min_len, 3])
        err_y = np.abs(v_y_ref[:min_len] - hist_state[:min_len, 4])
        err_z = np.abs(v_z_ref[:min_len] - hist_state[:min_len, 5])
        
        itae_value = np.trapezoid(t_weighted * (err_x + err_y + err_z), x=t_weighted)
        
        # C. Formulate Scaled Combined Cost
        combined_cost = (1.0 - alpha) * annoyance_dose * pa_scale + alpha * (itae_value * itae_scale)
        
        result_dict = {"objective": float(combined_cost), "feasible": True}
        # Explicitly passing v_y_ref down the line now for debugging dashboards
        return result_dict, hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa
    
    else:
        # --- OPTIMIZER MODE ---
        try:
            outputs = run_drone_acoustic_simulation(
                gains, dryden_ts, waypoints, time_steps, dt=dt, plant_params=drone_params
            )
            spl_fine, pa, _, hist_state, _, _, _ = outputs
            
            if check_tracking_performance(hist_state, v_z_ref):
                return {"objective": 99.0, "feasible": False}
                
            # A. Calculate Annoyance Dose
            acoustic_time_grid = np.linspace(0, time_steps[-1], len(pa["annoyance_PA"]))
            annoyance_dose = np.trapezoid(pa["annoyance_PA"], x=acoustic_time_grid)
            
            # B. Calculate 3D Velocity ITAE
            min_len = min(len(hist_state), len(time_steps))
            t_weighted = time_steps[:min_len]
            
            err_x = np.abs(v_x_ref[:min_len] - hist_state[:min_len, 3])
            err_y = np.abs(v_y_ref[:min_len] - hist_state[:min_len, 4])
            err_z = np.abs(v_z_ref[:min_len] - hist_state[:min_len, 5])
            
            itae_value = np.trapezoid(t_weighted * (err_x + err_y + err_z), x=t_weighted)
            
            # C. Formulate Scaled Combined Cost
            combined_cost = (1.0 - alpha) * annoyance_dose * pa_scale + alpha * (itae_value * itae_scale)
            
            print(f"Cost: {combined_cost:.4f} (Noise: {annoyance_dose:.2f}, ITAE: {itae_value:.2f})")
            return {"objective": float(combined_cost), "feasible": True}
            
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

def plot_comprehensive_diagnostics(hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa):
    """
    Generates two independent diagnostic dashboards:
    1. Flight Dynamics Panel (Kinematics & State Trajectories)
    2. Psychoacoustic Panel (All calculated sound metrics)
    """
    # ----------------------------------------------------
    # DASHBOARD 1: FLIGHT DYNAMICS
    # ----------------------------------------------------
    actual_x   = hist_state[:, 0]
    actual_y   = hist_state[:, 1]
    actual_alt = hist_state[:, 2]
    actual_vx  = hist_state[:, 3]
    actual_vy  = hist_state[:, 4]
    actual_vz  = hist_state[:, 5]
    
    fig1, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Panel A: Vertical Velocity Tracking
    axs[0].plot(time_steps, v_z_ref, 'r--', linewidth=2, label='Target Vz')
    axs[0].plot(time_steps[:len(actual_vz)], actual_vz, 'b-', linewidth=2, label='Actual Vz')
    axs[0].set_ylabel('Velocity Z [m/s]')
    axs[0].set_title('Flight Dynamics Panel', fontsize=12, fontweight='bold')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='upper right')
    
    # Panel B: Lateral Velocities Tracking (FIXED: Added Target Vx Line)
    axs[1].plot(time_steps, v_x_ref, 'm--', linewidth=1.5, alpha=0.8, label='Target Vx')
    axs[1].plot(time_steps[:len(actual_vx)], actual_vx, 'm-', label='Actual Vx')
    axs[1].plot(time_steps[:len(actual_vy)], actual_vy, 'c-', label='Actual Vy')
    axs[1].set_ylabel('Planar Speed [m/s]')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(loc='upper right')
    
    # Panel C: Total 3D Spatial Trajectory (FIXED: Updated Y-label from Altitude to Position)
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
    
    if num_plots == 1:
        axs2 = [axs2]
        
    for idx, key in enumerate(active_keys):
        signal = np.asarray(pa[key], dtype=float)
        t_acoustic = np.linspace(0, time_steps[-1], len(signal))
        is_valid = ~np.isnan(signal)
        
        meta = acoustic_metadata[key]
        if np.any(is_valid):
            axs2[idx].plot(t_acoustic[is_valid], signal[is_valid], color=meta['color'], linewidth=2, label=meta['label'])
        else:
            axs2[idx].text(0.5, 0.5, 'Metric Data Unavailable', transform=axs2[idx].transAxes, 
                           ha='center', va='center', color='gray', fontstyle='italic')
            
        axs2[idx].set_ylabel(meta['label'], fontsize=10, fontweight='bold')
        axs2[idx].grid(True, linestyle=':', alpha=0.5)
        
    axs2[-1].set_xlabel('Time [seconds]', fontsize=11)
    fig2.suptitle('Complete Psychoacoustic Sound Footprint', fontsize=14, fontweight='bold', y=0.99)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    dt = 0.01 
    t_end = 4.01
    vz_max = 5.0 
    vx_max = 5.00 
    vy_max = 0. 
    t_climb = 3.0

    # Define a test drone configuration (middle of the bounds)
    test_X = [0.22, 0.25,  vz_max, t_climb, vx_max, vy_max, 0.15, 0.04, 3.0, 0.5, 15.0, 7.0]
    
    # --- ADD THIS TEMPORARY PRINT BLOCK TO YOUR MAIN ---
    outputs_traj = generate_dynamic_trajectory(
        vx_max, vy_max, vx_max, t_climb, dt=dt, t_total=t_end
    )
    waypoints, time_steps, v_x_ref, v_y_ref, v_z_ref = outputs_traj
    # velocity_schedule, time_steps, v_x_reference_array, v_z_reference_array = generate_dynamic_trajectory(vx_max, test_X[2], test_X[3], dt=dt, t_total=t_end)

    print("--- TRAJECTORY ARRAY DIAGNOSIS ---")
    print(f"Shape of time_steps:        {time_steps.shape}")
    # print(f"Shape of velocity_schedule: {velocity_schedule.shape}")
    print(f"Expected dt:                {dt}")
    print(f"Calculated array length:    {len(time_steps)}")
    print("---------------------------------")

    print("Testing the co-design callable wrapper...")
    # test_result, hist_state, time_steps, v_x_ref, v_z_ref, pa = evaluate_drone_codesign(test_X, dt=dt, t_end=t_end, debug = True)

    test_result, hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa = evaluate_drone_codesign(
    test_X, dt=dt, t_end=t_end, debug=True
    )
    
    print("\n--- Evaluation Test Results ---")
    print(f"Objective (Annoyance Dose): {test_result['objective']}")
    print(f"Is Configuration Feasible?: {test_result['feasible']}")

    # if test_result['feasible'] and hist_state is not None:
    #     plot_codesign_debug_results(hist_state, time_steps, v_z_ref)
    
    if test_result['feasible'] and hist_state is not None and pa is not None:
        plot_comprehensive_diagnostics(hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa)

