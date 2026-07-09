import numpy as np

# --- 1. Global System Parameters ---
MASS = 1.5              # kg
I_xx = 0.012            # kg*m^2
I_yy = 0.012            # kg*m^2
I_zz = 0.020            # kg*m^2
g = 9.81                # m/s^2

# State Vector Index Map:
# x[0:3]  = Position (x, y, z) in Inertial Frame
# x[3:6]  = Velocity (vx, vy, vz) in Inertial Frame
# x[6:9]  = Euler Angles (phi, theta, psi) -> Roll, Pitch, Yaw
# x[9:12] = Angular Velocities (p, q, r) in Body Frame

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
    def __init__(self):
        # Existing attitude and altitude gains
        self.kp_alt, self.kd_alt = 15.0, 10.0
        self.kp_att, self.kd_att = 2.5, 0.4
        
        # New Velocity PI gains
        self.kp_vel = 0.15
        self.ki_vel = 0.05   # Integral gain to eliminate wind drift
        
        # Error accumulators (Integrators)
        self.integral_vx = 0.0
        self.integral_vy = 0.0
        self.dt = 0.01       # Match your simulation time step

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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Modified Simulation Loop to Save History ---
if __name__ == "__main__":
    dt = 0.01  
    t_end = 45.0
    time_steps = np.arange(0, t_end, dt)
    
    # Initialize the drone at rest, but let's start it at an altitude of 5 meters 
    # so it doesn't immediately crash into the ground while trying to fly forward.
    state = np.zeros(12) 
    state[2] = 5.0  # Set initial Z position to 5m
    
    controller = DronePIDController()
    
    # Define your constant velocity profile: [Vx, Vy, Vz] in m/s
    # e.g., Cruise forward at 2 m/s, hold lateral position (0 m/s), and maintain altitude (0 m/s)
    target_velocity = np.array([5.0, 0.0, 0.0])
    target_yaw = 0.0  # Keep the drone facing forward (0 rad)
    
    # Pre-allocate historical data matrices
    history_state = np.zeros((len(time_steps), 12))
    history_gusts = np.zeros((len(time_steps), 5)) 
    history_controls = np.zeros((len(time_steps), 6)) # Arr
    history_accel = np.zeros((len(time_steps), 3))

    # generating turbulent velocties 
    from dryden_timeseries import generate_dryden_time_series, DrydenParams

    #T_max = T          # seconds, matches your duration_bounds upper limit
    #dt_dryden = 0.05       # [s] 20 Hz is more than enough for Dymos GL nodes
                # (your ~54 nodes over ~30s gives ~1.8 Hz resolution)
                # no need to pre-generate at 500 Hz -- that's only
                # needed for the acoustic post-processing step

    # t_dryden = t_ref #np.arange(0, T_max + dt_dryden, dt_dryden)
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
    
    velocity_schedule = [
    (0.0,  np.array([0.0, 0.0, 2.0])),   # 0 to 2s: Climb straight up at 2 m/s
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

    print("Running simulation with Dryden turbulence active...")
    for idx, t in enumerate(time_steps):
        
        target_velocity = velocity_schedule[0][1]
        
        # Scan through the schedule to find the active command for the current time 't'
        for switch_time, commanded_velocity in velocity_schedule:
            if t >= switch_time:
                target_velocity = commanded_velocity
            else:
                # Since the schedule is sorted by time, we can stop checking 
                # as soon as we hit a switch_time in the future
                break

        # Note: If your controller uses a velocity tracker, it should read 
        # the GROUND speed (state[3:6]) to keep moving at a uniform inertial target, 
        # while the turbulence buffets its orientation and attitude.
        current_controls, theta_cmd, phi_cmd = controller.compute_controls_velocity_tracking(
            state, target_velocity, target_yaw
        )
        
        dstatedt = drone_derivatives_with_turb(t, state, current_controls, dryden_ts)
        history_accel[idx, :] = dstatedt[3:6]

        # Advance physics using the turbulence-aware RK4 step
        state = rk4_step_with_turb(t, state, dt, current_controls, dryden_ts)
        history_state[idx, :] = state
        history_controls[idx, :] = current_controls

        t_ts = dryden_ts['t']
        history_gusts[idx, 0] = np.interp(t, t_ts, dryden_ts['wu'])
        history_gusts[idx, 1] = np.interp(t, t_ts, dryden_ts['wv'])
        history_gusts[idx, 2] = np.interp(t, t_ts, dryden_ts['ww'])
        history_gusts[idx, 3] = np.interp(t, t_ts, dryden_ts['p_turb'])
        history_gusts[idx, 4] = np.interp(t, t_ts, dryden_ts['q_turb'])

        if idx % 50 == 0:
            print(f"t={t:.2f}s | Vx={state[3]:.2f} | Cmd_Theta={theta_cmd:.4f} | Actual_Theta={state[7]:.4f}")

    # --- 2. Professional Plotting Configuration ---
    # Use a clean, modern aesthetic profile
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(3, 1, figsize=(11, 14), sharex=True)
    fig.suptitle('6-DoF Quadcopter Closed-Loop Flight Telemetry', fontsize=16, fontweight='bold', y=0.96)

    # Palette selections (Professional mute-tones)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Subplot 1: Inertial Positions (x, y, z)
    axs[0].plot(time_steps, history_state[:, 0], label='Position X', color=colors[0], linewidth=2)
    axs[0].plot(time_steps, history_state[:, 1], label='Position Y', color=colors[1], linewidth=2)
    axs[0].plot(time_steps, history_state[:, 2], label='Altitude Z', color=colors[2], linewidth=2)
    # Add dashed target setpoint lines
    # axs[0].axhline(y=target_position[0], color=colors[0], linestyle='--', alpha=0.6, label='Target X')
    # axs[0].axhline(y=target_position[1], color=colors[1], linestyle='--', alpha=0.6, label='Target Y')
    # axs[0].axhline(y=target_position[2], color=colors[2], linestyle='--', alpha=0.6, label='Target Z')
    axs[0].set_ylabel('Position / Altitude [m]', fontsize=11, fontweight='bold')
    axs[0].set_title('Inertial Coordinates (Translation)', fontsize=12, loc='left', pad=6)
    axs[0].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    # Subplot 2: Translational Linear Velocities (vx, vy, vz)
    axs[1].plot(time_steps, history_state[:, 3], label='Velocity $V_x$', color=colors[0], linewidth=1.8)
    axs[1].plot(time_steps, history_state[:, 4], label='Velocity $V_y$', color=colors[1], linewidth=1.8)
    axs[1].plot(time_steps, history_state[:, 5], label='Velocity $V_z$', color=colors[2], linewidth=1.8)
    axs[1].set_ylabel('Linear Velocity [m/s]', fontsize=11, fontweight='bold')
    axs[1].set_title('Inertial Frame Velocities', fontsize=12, loc='left', pad=6)
    axs[1].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    # Subplot 3: Attitude Euler Angles (convert radians to degrees for readability)
    axs[2].plot(time_steps, np.degrees(history_state[:, 6]), label='Roll ($\phi$)', color=colors[0], linewidth=1.8)
    axs[2].plot(time_steps, np.degrees(history_state[:, 7]), label='Pitch ($\\theta$)', color=colors[1], linewidth=1.8)
    axs[2].plot(time_steps, np.degrees(history_state[:, 8]), label='Yaw ($\psi$)', color=colors[2], linewidth=1.8)
    axs[2].axhline(y=np.degrees(target_yaw), color=colors[2], linestyle='--', alpha=0.6, label='Target Yaw')
    axs[2].set_ylabel('Orientation Angle [$^\circ$]', fontsize=11, fontweight='bold')
    axs[2].set_xlabel('Simulation Time [seconds]', fontsize=12, fontweight='bold')
    axs[2].set_title('Attitude Configurations (Rotational)', fontsize=12, loc='left', pad=6)
    axs[2].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    # Clean layout alignments to eliminate overlapping texts
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle('Injected Dryden Turbulence Disturbance Telemetry', fontsize=14, fontweight='bold')

    # Subplot 1: Linear Velocity Gusts
    axs[0].plot(time_steps, history_gusts[:, 0], label='Longitudinal Gust ($w_u$)', color='#1f77b4', alpha=0.85)
    axs[0].plot(time_steps, history_gusts[:, 1], label='Lateral Gust ($w_v$)', color='#ff7f0e', alpha=0.85)
    axs[0].plot(time_steps, history_gusts[:, 2], label='Vertical Gust ($w_w$)', color='#2ca02c', alpha=0.85)
    axs[0].set_ylabel('Velocity Disturbance [m/s]', fontsize=11, fontweight='bold')
    axs[0].set_title('Linear Air Mass Fluctuations', fontsize=11, loc='left')
    axs[0].legend(loc='upper right', frameon=True, facecolor='white')

    # Subplot 2: Rotational Angular Gusts
    axs[1].plot(time_steps, history_gusts[:, 3], label='Roll Rate Gust ($p_{turb}$)', color='#d62728', alpha=0.85)
    axs[1].plot(time_steps, history_gusts[:, 4], label='Pitch Rate Gust ($q_{turb}$)', color='#9467bd', alpha=0.85)
    axs[1].set_ylabel('Angular Rate Disturbance [rad/s]', fontsize=11, fontweight='bold')
    axs[1].set_xlabel('Simulation Time [seconds]', fontsize=11, fontweight='bold')
    axs[1].set_title('Rotational Kinematic Disturbances', fontsize=11, loc='left')
    axs[1].legend(loc='upper right', frameon=True, facecolor='white')

    plt.tight_layout()
    plt.show()

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle('Control System Response to Dryden Turbulence', fontsize=14, fontweight='bold', y=0.96)

    # Subplot 1: Thrust (Force) Allocation
    # Note: Fx and Fy are included for completion but will register as 0.0
    axs[0].plot(time_steps, history_controls[:, 2], label='Collective Thrust ($F_z$)', color='#2ca02c', linewidth=2)
    # Add a baseline marker for the hover thrust requirement (Mass * g)
    axs[0].axhline(y=1.5 * 9.81, color='black', linestyle=':', alpha=0.5, label='Nominal Hover Thrust')
    axs[0].set_ylabel('Control Forces [N]', fontsize=11, fontweight='bold')
    axs[0].set_title('Vertical Force Command Actuation', fontsize=11, loc='left')
    axs[0].legend(loc='upper right', frameon=True, facecolor='white')

    # Subplot 2: Control Moments (Attitude Corrections)
    axs[1].plot(time_steps, history_controls[:, 3], label='Roll Moment ($M_x$)', color='#d62728', linewidth=1.5)
    axs[1].plot(time_steps, history_controls[:, 4], label='Pitch Moment ($M_y$)', color='#9467bd', linewidth=1.5)
    axs[1].plot(time_steps, history_controls[:, 5], label='Yaw Moment ($M_z$)', color='#bcbd22', linewidth=1.5)
    axs[1].set_ylabel('Control Moments [N·m]', fontsize=11, fontweight='bold')
    axs[1].set_xlabel('Simulation Time [seconds]', fontsize=11, fontweight='bold')
    axs[1].set_title('Attitude Stabilization Moments', fontsize=11, loc='left')
    axs[1].legend(loc='upper right', frameon=True, facecolor='white')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    from rotor_rpm_estimation import estimate_rotor_rpm
    from drone_acoustic_radiation_v2 import (
        calibrate_p_ref, AcousticParams, FineGridParams,
        estimate_received_spl_fine,
    )

    # 1. Coarse RPM estimate from your Dymos output (as before)
    rpm_result = estimate_rotor_rpm(
        t=time_steps,
        x=history_state[:, 0],
        y=history_state[:, 1],
        z=history_state[:, 2],
        vx=history_state[:, 3],
        vy=history_state[:, 4],
        vz=history_state[:, 5],
        ax=history_accel[:, 0],
        ay=history_accel[:, 1],
        az=history_accel[:, 2],
        wx=history_gusts[:, 0],
        wy=history_gusts[:, 1],
        wz=history_gusts[:, 2]
    )
    
    # 2. Calibrate the RPM-to-power model against a reference measurement
    p_ref = calibrate_p_ref(
        spl_ref_db=72.0, rpm_ref_measurement=5000.0,
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

    observer_xyz = (25.0, 2.0, 0.0)

    # 4. The call
    spl_fine = estimate_received_spl_fine(
        time_steps, history_state[:, 0],
        history_state[:, 1],
        history_state[:, 2],
        rpm_result["rpm_front"], rpm_result["rpm_rear"],
        rpm_result["rpm_right"], rpm_result["rpm_left"],
        observer_xyz,
        acoustic_params=acoustic_params,
        fine_params=fine_params,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(spl_fine["t_fine"], spl_fine["p_signal"])
    axes[0].set_ylabel("Received SPL [dB]")

    # zoom into a short window to see individual rotor cycles
    # mask = spl_fine["t_fine"] < 0.05

    for name in ["front", "rear", "right", "left"]:
        axes[1].plot(spl_fine["t_fine"], spl_fine[f"p_rotor_{name}"], label=name)
    axes[1].legend()
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Pressure [Pa]")

    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print(spl_fine["p_total"].shape)

    from zwicker_annoyance_v2 import compute_zwicker_indicators_windowed


    #---------
    print("p_signal stats: min={:.6f}, max={:.6f}, std={:.6f} Pa".format(
        spl_fine["p_signal"].min(), spl_fine["p_signal"].max(), spl_fine["p_signal"].std()))
    print("spl_db range:", spl_fine["spl_db"].min(), "-", spl_fine["spl_db"].max(), "dB")

    # Check the actual loudness values, bypassing the floor guard entirely
    result_raw = compute_zwicker_indicators_windowed(
        spl_fine["p_signal"], fs=fine_params.fs,
        window_s=2.0, hop_s=0.5,
        use_fs_approximation=True,
        loudness_floor_sone=0.0,   # disable the guard to see real N values
    )
    print("loudness_sone stats:", np.nanmin(result_raw["loudness_sone"]),
        np.nanmedian(result_raw["loudness_sone"]), np.nanmax(result_raw["loudness_sone"]))
    #---------
    result = compute_zwicker_indicators_windowed(
        spl_fine["p_signal"], fs=fine_params.fs,
        window_s=2.0, hop_s=0.5,
        use_fs_approximation=True,
    )

    # diagnostics go HERE, on the windowed result (arrays), not inside the library file
    print("NaN sharpness windows:", np.isnan(result["sharpness_acum"]).sum(), "/", len(result["sharpness_acum"]))
    print("NaN PA windows:", np.isnan(result["annoyance_PA"]).sum())

    fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    for ax, key in zip(axes, ["loudness_sone", "sharpness_acum", "roughness_asper", "fluctuation_vacil", "annoyance_PA"]):
        ax.plot(result["t_center"], result[key])
        ax.set_ylabel(key)
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the continuous flight path
    # We use a distinct color and line thickness to clearly show the flight track
    x_pos = history_state[:, 0]
    y_pos = history_state[:, 1]
    z_pos = history_state[:, 2]

    flight_line = ax.plot(x_pos, y_pos, z_pos, 
                        label='Drone Flight Path', 
                        color='#1f77b4', 
                        linewidth=2.5, 
                        zorder=3)

    # 4. Mark the Start and End points uniquely
    ax.scatter(x_pos[0], y_pos[0], z_pos[0], 
            color='green', s=100, marker='^', label='Takeoff Point', zorder=5)
    ax.scatter(x_pos[-1], y_pos[-1], z_pos[-1], 
            color='red', s=100, marker='v', label='Landing/Final Position', zorder=5)

    # 5. Add a 2D "Shadow" or projection on the floor (Optional but highly recommended)
    # This significantly improves the human eye's perception of 3D depth!
    ax.plot(x_pos, y_pos, np.zeros_like(z_pos), 
            color='gray', linestyle='--', alpha=0.5, label='Ground Track Projection', zorder=2)

    # 6. Customize axes labels and titles
    ax.set_title('3D Quadcopter Trajectory Flight Profile\n(10-Stage Velocity Schedule under Dryden Turbulence)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Inertial X Position [meters]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Inertial Y Position [meters]', fontsize=11, fontweight='bold')
    ax.set_zlabel('Altitude Z [meters]', fontsize=11, fontweight='bold')

    # Set an equal aspect ratio profile so the geometry isn't distorted
    # (This ensures 1 meter on X looks exactly the same as 1 meter on Z)
    ax.set_box_aspect([np.ptp(x_pos), np.ptp(y_pos), np.ptp(z_pos) + 2.0]) 

    # Adjust the initial viewing angle for the best perspective (elevation, azimuth)
    ax.view_init(elev=25, azim=-135)

    # Place the legend cleanly
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.show()