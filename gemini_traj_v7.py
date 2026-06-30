import openmdao.api as om
import dymos as dm
import numpy as np
import random

class DrydenField:
    def __init__(self, n_waves=20, seed=42):
        np.random.seed(seed)
        # Define a range of spatial frequencies (rad/m)
        # Dryden peaks usually between 0.01 and 0.5 rad/m
        self.k = np.logspace(-2, 0, n_waves) 
        
        # Random phases for each wave in 3D
        self.phi_x = np.random.uniform(0, 2*np.pi, n_waves)
        self.phi_y = np.random.uniform(0, 2*np.pi, n_waves)
        self.phi_z = np.random.uniform(0, 2*np.pi, n_waves)
        
        # Directions for the wave vectors
        self.dirs = np.random.randn(n_waves, 3)
        self.dirs /= np.linalg.norm(self.dirs, axis=1)[:, None]

    def get_gust(self, x, y, z, sigma=1.5):
        """
        Computes a smooth, differentiable gust at (x, y, z).
        This is compatible with Dymos because it uses pure NumPy math.
        """
        u_gust = np.zeros_like(x)
        
        # Apply the Dryden-like summation
        # Local turbulence intensity often scales with sqrt(z)
        intensity = sigma * (1.0 - np.exp(-z/5.0)) 
        
        for i in range(len(self.k)):
            # Spatial projection: k_vec dot position_vec
            proj = self.k[i] * (self.dirs[i,0]*x + self.dirs[i,1]*y + self.dirs[i,2]*z)
            u_gust += np.sin(proj + self.phi_x[i])
            
        return u_gust * (intensity / np.sqrt(len(self.k)))

# Initialize the field globally so it's "Frozen"
dryden_field = DrydenField()

class DroneODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('avoid_points', types=list)
        self.options.declare('penalty_strength', types=float, default=100.0)
        self.options.declare('v_ref', default=5.0)  # Wind speed at ref height
        self.options.declare('z_ref', default=20.0) # Reference height

    def setup(self):
        nn = self.options['num_nodes']
        # Inputs: States & Controls
        for var in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']:
            self.add_input(var, shape=(nn,), units=None)

        # Outputs: Rates & Instantaneous Penalty
        for var in ['x_dot', 'y_dot', 'z_dot', 'vx_dot', 'vy_dot', 'vz_dot', 'inst_penalty', 'acc_mag2', 'energy', 'wind_x', 'wind_y', 'wind_z']:
            self.add_output(var, shape=(nn,), units=None)

        # Use finite difference for partials to keep the script simple
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        # Get the spatially varying Dryden gust
        # (Assuming your implementation provides wx, wy, wz)
        outputs['wind_x'] = dryden_field.get_gust(x, y, z, sigma=1.0)
        outputs['wind_y'] = dryden_field.get_gust(y, x, z, sigma=0.5)
        outputs['wind_z'] = dryden_field.get_gust(y, x, z, sigma=0.25)

        # Let's assume Mean Wind from earlier + this new Gust
        v_ref, z_ref = 5.0, 20.0
        mean_wx = v_ref * (np.maximum(z, 0.1) / z_ref)**0.15

        # Kinematics
        # 2. Kinematics: Ground Velocity = Air Velocity + Wind
        # Here, vx, vy, vz are treated as the drone's velocity relative to AIR
        outputs['x_dot'] = inputs['vx'] + mean_wx + outputs['wind_x']
        outputs['y_dot'] = inputs['vy'] + outputs['wind_y']
        outputs['z_dot'] = inputs['vz'] + outputs['wind_z']

        # 3. Dynamics: Accelerations change the Air Velocity
        outputs['vx_dot'] = inputs['ax']
        outputs['vy_dot'] = inputs['ay']
        outputs['vz_dot'] = inputs['az']

        # outputs['x_dot'], outputs['y_dot'], outputs['z_dot'] = inputs['vx'], inputs['vy'], inputs['vz']
        # outputs['vx_dot'], outputs['vy_dot'], outputs['vz_dot'] = inputs['ax'], inputs['ay'], inputs['az']

        # Penalty calculation: 1 / dist^2
        avoid_points = self.options['avoid_points']
        penalty_strength = self.options['penalty_strength']
        total_penalty = np.zeros(self.options['num_nodes'])

        for (px, py, pz) in avoid_points:
            dist2 = (inputs['x'] - px)**2 + (inputs['y'] - py)**2 + (inputs['z'] - pz)**2
            # Avoid division by zero with a small epsilon
            total_penalty += penalty_strength / (dist2 + 1e-4)
        
        outputs['inst_penalty'] = total_penalty
        outputs['acc_mag2'] = inputs['ax']**2 + inputs['ay']**2 + inputs['az']**2
        #v_mag3 = inputs['vx']**3 + inputs['vy']**3 + inputs['vz']**3
        eps = 1e-6 
        v_mag2 = inputs['vx']**2 + inputs['vy']**2 + inputs['vz']**2
        outputs['energy'] = np.power(v_mag2 + eps, 1.5)
        #outputs['energy'] = inputs['vx']**2 + inputs['vy']**2 + inputs['vz']**2

# --- Setup Problem ---
p = om.Problem()
p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
p.driver.opt_settings['print_level'] = 5
# p.driver.opt_settings['delta'] = 1e-1
p.driver.opt_settings['max_iter'] = 20

# Generate Obstacles
avoid_points = [(random.uniform(150, 450), random.uniform(150, 450), 0) for _ in range(80)]

traj = dm.Trajectory()
phase = dm.Phase(ode_class=DroneODE, 
                 ode_init_kwargs={'avoid_points': avoid_points},
                 transcription=dm.GaussLobatto(num_segments=24, order=3))
p.model.add_subsystem('traj', traj)
traj.add_phase('phase0', phase)

# Time, States, and Controls
phase.set_time_options(fix_initial=True, fix_duration=False, duration_bounds=(5, 200))

# States: x, y, z are fixed at start (0,0,0) and end (500,500,10)
#for s in ['x', 'y', 'z']:
for s in ['x', 'y']:
    phase.add_state(s, fix_initial=True, fix_final=True, ref=500.0, rate_source=f'{s}_dot')

phase.add_state('z', fix_initial=True, lower = 0, upper = 50., ref=5.0, fix_final=True, rate_source=f'z_dot')

# Velocities: Fixed at 0 at start/end
for v in ['vx', 'vy', 'vz']:
    phase.add_state(v, fix_initial=True, fix_final=True, rate_source=f'{v}_dot')

phase.add_state('acc_integral', rate_source='acc_mag2', lower = 0, ref=5.0, fix_initial=True)

phase.add_state('energy_spent', 
                rate_source='energy', 
                fix_initial=True, 
                fix_final=False,
                lower = 0, 
                ref=500.0, 
                units=None)

# Accelerations as Controls
for a in ['ax', 'ay', 'az']:
    phase.add_control(a, lower=-8.0, upper=8.0, rate_continuity=True, 
                  rate2_continuity=False)

# INTEGRATE PENALTY: This creates the "Accumulated Reward" automatically
phase.add_state('total_penalty', rate_source='inst_penalty', ref=5000.0, fix_initial=True, fix_final=False)



phase.add_timeseries_output('wind_x')
phase.add_timeseries_output('wind_y')
phase.add_timeseries_output('wind_z')



# Objective: Minimize Time + Penalty_Integral
class ObjectiveComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('time', units=None)
        self.add_input('penalty', units=None)
        self.add_input('acc_integral', units=None)
        self.add_input('energy_final', units=None) # New input
        self.add_output('J')
        self.declare_partials('*', '*', method='fd')
    def compute(self, inputs, outputs):
        # outputs['J'] = inputs['time'] + 20.0 * inputs['penalty']
        outputs['J'] = 0.003 * ( inputs['time'] + 6.0 * inputs['penalty'] + 2.0 * inputs['acc_integral'] + 0.001 * inputs['energy_final'] )
        # outputs['J'] = 0.003 * ( inputs['time'] + 6.0 * inputs['penalty'] + 5.0 * inputs['acc_integral'] + 0.001 * inputs['energy_final'] )
        # outputs['J'] = inputs['time'] + 4.0 * inputs['penalty'] + 0.02 * inputs['energy_final']

p.model.add_subsystem('obj_comp', ObjectiveComp())
p.model.connect('traj.phase0.timeseries.time', 'obj_comp.time', src_indices=[-1])
p.model.connect('traj.phase0.timeseries.total_penalty', 'obj_comp.penalty', src_indices=[-1])
p.model.connect('traj.phase0.timeseries.acc_integral', 'obj_comp.acc_integral', src_indices=[-1])
p.model.connect('traj.phase0.timeseries.energy_spent', 
                'obj_comp.energy_final', src_indices=[-1])
p.model.add_objective('obj_comp.J')

p.setup()

# --- Initial Guesses ---
p.set_val('traj.phase0.t_duration', 30.0)
p.set_val('traj.phase0.states:x', phase.interp('x', [0, 500.]))
p.set_val('traj.phase0.states:y', phase.interp('y', [0, 450.]))
p.set_val('traj.phase0.states:z', phase.interp('z', [0, 10]))
p.set_val('traj.phase0.states:total_penalty', 0.0)
p.set_val('traj.phase0.states:energy_spent', phase.interp('energy_spent', [0, 100]))
p.set_val('traj.phase0.states:acc_integral', phase.interp('energy_spent', [0, 100]))

p.run_driver()

# Extracting the values from the objective component
final_time = p.get_val('obj_comp.time')[0]
final_penalty = p.get_val('obj_comp.penalty')[0]
final_energy = p.get_val('obj_comp.energy_final')[0]
acc_integral = p.get_val('obj_comp.acc_integral')[0]
final_total_J = p.get_val('obj_comp.J')[0]

print(f"\n{'='*30}")
print(f"OPTIMIZATION RESULTS")
print(f"{'='*30}")
print(f"Final Time:         {final_time:.4f} s")
print(f"Obstacle Penalty:   {final_penalty:.4f}")
print(f"Energy Expenditure: {final_energy:.4f}")
print(f"Acceleration integ: {acc_integral:.4f}")
print(f"Total Objective J:  {final_total_J:.4f}")
print(f"{'='*30}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Extract Data ---
t = p.get_val('traj.phase0.timeseries.time')
x = p.get_val('traj.phase0.timeseries.x')
y = p.get_val('traj.phase0.timeseries.y')
z = p.get_val('traj.phase0.timeseries.z')

vx = p.get_val('traj.phase0.timeseries.vx')
vy = p.get_val('traj.phase0.timeseries.vy')
vz = p.get_val('traj.phase0.timeseries.vz')

ax = p.get_val('traj.phase0.timeseries.ax')
ay = p.get_val('traj.phase0.timeseries.ay')
az = p.get_val('traj.phase0.timeseries.az')

# Extract the wind components calculated by the ODE
wx = p.get_val('traj.phase0.timeseries.wind_x')
wy = p.get_val('traj.phase0.timeseries.wind_y')
wz = p.get_val('traj.phase0.timeseries.wind_z')

# --- 2. Plotting ---
fig = plt.figure(figsize=(15, 10))

# Plot 1: 3D Trajectory
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(x, y, z, 'b-', label='Path')
ax1.scatter(x[0], y[0], z[0], color='g', label='Start')
ax1.scatter(x[-1], y[-1], z[-1], color='r', label='End')

# Plot Obstacles (as simple points for clarity)
obs_x, obs_y, obs_z = zip(*avoid_points)
ax1.scatter(obs_x, obs_y, obs_z, color='k', marker='x', alpha=0.5, label='Obstacles')

ax1.set_title("3D Drone Trajectory")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.legend()

# Plot 2: Velocity Components
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t, vx, label='Vx')
ax2.plot(t, vy, label='Vy')
ax2.plot(t, vz, label='Vz')
ax2.set_title("Velocity Components")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("m/s")
ax2.legend()
ax2.grid(True)

# Plot 3: Acceleration (Controls)
ax3 = fig.add_subplot(2, 2, 3)
ax3.step(t, ax, where='post', label='ax')
ax3.step(t, ay, where='post', label='ay')
ax3.step(t, az, where='post', label='az')
ax3.set_title("Acceleration (Controls)")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("m/s²")
ax3.legend()
ax3.grid(True)

# Plot 4: Altitude (Z) over time
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(t, z, color='purple')
ax4.set_title("Altitude Profile")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Z (m)")
ax4.grid(True)

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot A: The Wind Profile (Physics check)
# This shows how wind changes with height (Z)
ax1.plot(wx, z, 'b-', label='Wind X (Shear)')
ax1.set_xlabel('Wind Speed (m/s)')
ax1.set_ylabel('Altitude Z (m)')
ax1.set_title('Wind Profile vs. Altitude')
ax1.grid(True)
ax1.legend()

# Plot B: Wind Experienced Over Time
# This shows the "Timeline" of the disturbances
ax2.plot(t, wx, label='Wind X')
ax2.plot(t, wy, label='Wind Y')
ax2.plot(t, wz, label='Wind Z')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Wind Speed (m/s)')
ax2.set_title('Wind Experienced during Flight')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

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
# mask = spl_fine["t_fine"] < 0.05
#axes[1].plot(spl_fine["t_fine"], spl_fine["p_total"], 'k.-', label="p_total")
axes[1].plot(spl_fine["t_fine"], spl_fine["psi_front"], 'k-', label="p_total")
#axes[1].plot(spl_fine["t_fine"][mask], spl_fine["psi_front"][mask], label="front")
axes[1].set_ylabel(r"$\psi$ (azimuth)")
axes[1].set_xlabel("Time [s]")
axes[1].legend()
plt.tight_layout()
plt.show()

print(spl_fine["p_total"].shape)







#from rotor_rpm_estimation import estimate_rotor_rpm, DroneParams
#from drone_acoustic_radiation_v0 import calibrate_p_ref, AcousticParams, estimate_received_spl

##n = 500
##t = np.linspace(0, 10, n)
##vx = 5.0 + 2.0 * t / t[-1]
##vy = np.zeros(n)
##vz = 0.5 * np.sin(0.3 * t)
##ax = np.gradient(vx, t)
##ay = np.gradient(vy, t)
##az = np.gradient(vz, t)
##x = np.cumsum(vx) * (t[1] - t[0])
##y = np.cumsum(vy) * (t[1] - t[0])
##z = 50.0 + np.cumsum(vz) * (t[1] - t[0])  # start at 50 m altitude
##wx = 3.0 * np.ones(n)
##wy = np.zeros(n)
##wz = np.zeros(n)

#rpm_result = estimate_rotor_rpm(t, x, y, z, vx, vy, vz, ax, ay, az, wx, wy, wz)

## Calibrate p_ref against a plausible datasheet-style reference:
## "62 dB at 1 m, in-plane (90 deg), all 4 rotors at 5000 RPM hover"
#p_ref = calibrate_p_ref(
    #spl_ref_db=62.0,
    #rpm_ref_measurement=5000.0,
    #r_ref=1.0,
    #theta_ref_deg=90.0,
    #n_rotors_in_measurement=4,
    #n_exponent=5.0,
#)
#acoustic_params = AcousticParams(rpm_ref=5000.0, p_ref=p_ref, n_exponent=5.0)

## Fixed ground observer, 50 m horizontally from the trajectory's start, at ground level
#observer_xyz = (0.0, 50.0, 0.0)

#spl_result = estimate_received_spl(
    #t, x, y, z,
    #rpm_result["rpm_front"], rpm_result["rpm_rear"],
    #rpm_result["rpm_right"], rpm_result["rpm_left"],
    #observer_xyz, params=acoustic_params,
#)

#print("Received SPL at observer (synthetic example):")
#print(f"  SPL range      : {spl_result['spl_db'].min():.1f} - {spl_result['spl_db'].max():.1f} dB")
#print(f"  distance range : {spl_result['r'].min():.1f} - {spl_result['r'].max():.1f} m")
#print(f"  theta range     : {spl_result['theta_deg'].min():.1f} - {spl_result['theta_deg'].max():.1f} deg")

#import matplotlib.pyplot as plt

#fig, ax = plt.subplots(3, 1, figsize=(10, 8))

#ax[0].plot(t, spl_result['spl_db'], label="SPL")
#ax[0].plot(t, spl_result['swl_db'], label="SWL")
#ax[0].plot(t, spl_result['p_total'], label="P_TOTAL")
#ax[0].set_xlabel(r'$\t\,[s]$')
#ax[0].set_ylabel(r'$W\, [Pa]$')
#ax[0].legend()
#ax[0].grid(True)

#ax[1].plot(t, spl_result['theta_deg'], label="THETA")
#ax[1].set_xlabel(r'$\t\,[s]$')
#ax[1].set_ylabel(r'$\theta\, [Pa]$')
#ax[1].legend()
#ax[1].grid(True)

#ax[2].plot(t, rpm_result["rpm_front"], label="FRONT")
#ax[2].plot(t, rpm_result["rpm_rear"], label="REAR")
#ax[2].plot(t, rpm_result["rpm_right"], label="RIGHT")
#ax[2].plot(t, rpm_result["rpm_left"], label="LEFT")
#ax[2].set_xlabel(r'$\t\,[s]$')
#ax[2].set_ylabel(r'$\Omega\, [RPM]$')
#ax[2].legend()
#ax[2].grid(True)

#plt.tight_layout()
#plt.show()
