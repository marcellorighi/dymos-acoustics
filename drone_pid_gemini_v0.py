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


# --- 4. Main Simulation Loop Execution ---
if __name__ == "__main__":
    # Initialization
    dt = 0.01  # 100 Hz control loop and integration resolution
    t_end = 4.0
    time_steps = np.arange(0, t_end, dt)
    
    # State initialized at rest at origin
    state = np.zeros(12) 
    
    controller = DronePIDController()
    
    # Setpoints: Go to coordinate (2, -1, 5) and rotate yaw to 45 degrees (0.785 rad)
    target_position = np.array([2.0, -1.0, 5.0])
    target_yaw = 0.785
    
    print(f"Starting tracking simulation loop...")
    for t in time_steps:
        # Evaluate controllers exactly ONCE per full step
        current_controls = controller.compute_controls(state, target_position, target_yaw)
        
        # Propagate system via RK4 explicit step holding controls constant
        state = rk4_step(t, state, dt, current_controls)
        
        if round(t, 2) % 1.0 == 0:
            print(f"Time: {t:.1f}s | Pos: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}] | Alt Control Fz: {current_controls[2]:.1f} N")