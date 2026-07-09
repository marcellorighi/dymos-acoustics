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
        # Extremely simplified gain initialization (Must be tuned based on your system behavior)
        self.kp_pos, self.kd_pos = 0.4, 0.2
        self.kp_alt, self.kd_alt = 15.0, 10.0
        self.kp_att, self.kd_att = 2.5, 0.4

    def compute_controls_velocity_tracking(self, state, target_velocity, target_yaw):
        """
        Directly tracks a target velocity vector [vx_cmd, vy_cmd, vz_cmd]
        """
        # Extract current velocities and orientations
        vx, vy, vz = state[3:6]
        phi, theta, psi = state[6:9]
        p, q, r = state[9:12]
        
        # Velocity error PID gains (typically different from position gains)
        kp_vel_xy = 0.5
        kp_vel_z = 12.0
        kd_vel_z = 8.0
        
        # 1. Direct Velocity-to-Attitude Mapping
        # Error = Target Velocity - Current Velocity
        # If target_vx > current_vx, we need negative pitch to lean forward
        theta_cmd = -(kp_vel_xy * (target_velocity[0] - vx))
        # If target_vy > current_vy, we need positive roll to lean right
        phi_cmd = (kp_vel_xy * (target_velocity[1] - vy))
        
        theta_cmd = np.clip(theta_cmd, -0.5, 0.5)
        phi_cmd = np.clip(phi_cmd, -0.5, 0.5)
        
        # 2. Vertical Velocity Loop to Thrust
        F_z = kp_vel_z * (target_velocity[2] - vz) + (MASS * g)
        
        # 3. Inner Attitude Loop stays exactly the same
        M_x = self.kp_att * (phi_cmd - phi) + self.kd_att * (0.0 - p)
        M_y = self.kp_att * (theta_cmd - theta) + self.kd_att * (0.0 - q)
        M_z = self.kp_att * (target_yaw - psi) + self.kd_att * (0.0 - r)
        
        return np.array([0.0, 0.0, F_z, M_x, M_y, M_z])

    def compute_controls(self, state, target_pos, target_yaw):
        # Extract states
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        phi, theta, psi = state[6:9]
        p, q, r = state[9:12]
        
        # 1. Outer Loop: Position to Commanded Attitude
        # To go positive X, the drone must pitch down (-theta)
        theta_cmd = -(self.kp_pos * (target_pos[0] - x) + self.kd_pos * (0.0 - vx))
        # To go positive Y, the drone must roll right (+phi)
        phi_cmd = (self.kp_pos * (target_pos[1] - y) + self.kd_pos * (0.0 - vy))
        
        # Clip command angles to prevent inversion instabilities (e.g. max 30 degrees)
        theta_cmd = np.clip(theta_cmd, -0.5, 0.5)
        phi_cmd = np.clip(phi_cmd, -0.5, 0.5)
        
        # 2. Inner Loop: Core Flight Controller Execution
        # Altitude to Vertical Thrust Force Fz (Adding feed-forward gravity offset)
        F_z = self.kp_alt * (target_pos[2] - z) + self.kd_alt * (0.0 - vz) + (MASS * g)
        
        # Attitude Errors to Control Moments
        M_x = self.kp_att * (phi_cmd - phi) + self.kd_att * (0.0 - p)
        M_y = self.kp_att * (theta_cmd - theta) + self.kd_att * (0.0 - q)
        M_z = self.kp_att * (target_yaw - psi) + self.kd_att * (0.0 - r)
        
        # Assuming no structural auxiliary direct sideways thrust engines
        F_x, F_y = 0.0, 0.0 
        
        return np.array([F_x, F_y, F_z, M_x, M_y, M_z])


# --- 3. Explicit Runge-Kutta 4th Order Integrator ---
def rk4_step(t, state, dt, controls):
    """
    Advances the state vector by a timestep dt using explicit RK4 scheme,
    keeping the control variables frozen across sub-steps.
    """
    k1 = drone_derivatives(t,           state,           controls)
    k2 = drone_derivatives(t + 0.5*dt,  state + 0.5*dt*k1, controls)
    k3 = drone_derivatives(t + 0.5*dt,  state + 0.5*dt*k2, controls)
    k4 = drone_derivatives(t + dt,      state + dt*k3,    controls)
    
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return next_state

import matplotlib.pyplot as plt

# --- 1. Modified Simulation Loop to Save History ---
if __name__ == "__main__":
    dt = 0.01  
    t_end = 155.0
    time_steps = np.arange(0, t_end, dt)
    
    state = np.zeros(12) 
    controller = DronePIDController()
    
    # --- Configuration Parameters ---
    MAX_SPEED = 10.0          # Hard limit velocity: 10 m/s
    ACCEPTANCE_RADIUS = 0.3  # meters
    DECELERATION_DIST = 2.0  # Distance at which the drone begins braking smoothly

    # Define your waypoint track
    waypoints = [
        np.array([0.0,  0.0,  5.0]),
        np.array([15.0, 0.0,  5.0]), # Made distances larger to test 10 m/s
        np.array([15.0, 15.0, 7.0]),
        np.array([0.0,  0.0,  2.0])
    ]
    current_wp_idx = 0

    # acceptance_radius = 5.25 # meters
    
    # target_position = np.array([2.0, -1.0, 5.0])
    target_yaw = 0.785  # 45 degrees
    
    # Pre-allocate historical data matrices
    history_state = np.zeros((len(time_steps), 12))

    # 2. Inside your simulation loop:
   # --- Inside Your Main Simulation Loop ---
    for idx, t in enumerate(time_steps):
        current_pos = state[0:3]
        target_wp = waypoints[current_wp_idx]
        current_wp_idx_changed = False
        
        # 1. Compute vector pointing to the active waypoint
        vector_to_wp = target_wp - current_pos
        distance_to_wp = np.linalg.norm(vector_to_wp)
        
        # 2. Check if the waypoint is reached
        if distance_to_wp < ACCEPTANCE_RADIUS:
            if current_wp_idx < len(waypoints) - 1:
                current_wp_idx += 1
                current_wp_idx_changed = True
                print(f"[{t:.2f}s] Waypoint {current_wp_idx-1} Cleared! Routing to WP {current_wp_idx}: {waypoints[current_wp_idx]}")
                target_wp = waypoints[current_wp_idx]
                vector_to_wp = target_wp - current_pos
                distance_to_wp = np.linalg.norm(vector_to_wp)
                # Reset your PID integrator states on waypoint changes!
                # controller.reset_integrators()
            else:
                # If it's the final waypoint, hold position smoothly
                vector_to_wp = target_wp - current_pos
                distance_to_wp = np.linalg.norm(vector_to_wp)

        if current_wp_idx_changed:
            print(f"  -> DEBUG Vector to new WP: {vector_to_wp}")

        # 3. KINEMATIC GUIDANCE LAW (Speed Limiter & Braking)
        if distance_to_wp > 0.001:
            direction_unit_vector = vector_to_wp / distance_to_wp
            
            # If far away, travel at MAX_SPEED. If close, scale down speed proportionally (Braking profile)
            if distance_to_wp > DECELERATION_DIST:
                target_speed = MAX_SPEED
            else:
                # Linear ramp-down to 0 speed as distance approaches 0
                target_speed = MAX_SPEED * (distance_to_wp / DECELERATION_DIST)
                
            # Calculate the target velocity vector
            target_velocity = direction_unit_vector * target_speed
        else:
            target_velocity = np.zeros(3)

        # 4. Pass the calculated velocity vector to your Velocity-Tracking Controller
        current_controls = controller.compute_controls_velocity_tracking(state, target_velocity, target_yaw)
        
        # Propagate state via integration step
        state = rk4_step(t, state, dt, current_controls)

        # Save current state to history
        history_state[idx, :] = state

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