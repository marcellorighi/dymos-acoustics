import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 1. SALIB SENSITIVITY IMPORTS ---
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol as sobol_analyzer

# --- 2. YOUR DRONE PROJECT IMPORTS ---
# Replace 'drone_pid_function_v16' with your actual filename if it changes!
from drone_pid_function_v18 import (
    ParametricDronePlant, 
    DronePIDController, 
    run_drone_acoustic_simulation,
    generate_dryden_time_series,
    generate_dynamic_trajectory,
    DrydenParams
)
from drone_acoustic_radiation_v3 import (
        calibrate_p_ref, AcousticParams, FineGridParams,
        estimate_received_spl_fine,
    )
from zwicker_annoyance_v2 import compute_zwicker_indicators_windowed

# --- 3. GLOBAL CONFIGURATION & ENVIRONMENT SETUP ---
dt = 0.005
t_end = 8.0
my_init_state = np.zeros(12)
my_init_state[2] = 0.0  # Initial altitude

# Pre-generate environmental conditions so all variants face the exact same wind fields
t_dryden = np.arange(0, t_end + dt, dt)
dryden_ts = generate_dryden_time_series(
    t_dryden,
    params=DrydenParams(
        V=5.0, sigma_u=1.5, sigma_v=1.5, sigma_w=0.75,
        L_u=200.0, L_v=200.0, L_w=50.0,
        arm_length=0.45,  # Nominal baseline arm length for wind correlation width
        v_ref=5.0, z_ref=20.0,
    ),
    seed=42,
    altitude=50.0,
)

# Generate tracking trajectory profile coordinates
# Locked to a fixed test nominal profile so tracking error calculations match perfectly
waypoints, time_steps, v_x_ref, v_y_ref, v_z_ref = generate_dynamic_trajectory(
    vx_max=6.0, vy_max=0.0, vz_max=4.0, t_climb=2.5, dt=dt, t_total=t_end, mode='climb_cruise'
)

# --- 4. DEFINE THE SOBOL EXPERIMENT PROBLEM SPACE ---
# Sequence of columns generated in the parameter sampling array
problem = {
    'num_vars': 8,
    'names': [
        'R_rotor', 'L_arm', 'motor_bandwidth_hz', 
        'Kp_vel', 'Ki_vel', 'Kp_att', 'Kp_alt', 'Kd_alt'
    ],
    'bounds': [
        [0.10, 0.32],   # R_rotor [m]
        [0.35, 0.60],   # L_arm [m]
        [10.0, 30.0],   # motor_bandwidth_hz [Hz]
        [0.05, 0.35],   # Kp_vel
        [0.01, 0.14],   # Ki_vel
        [2.0,  6.0],    # Kp_att
        [10.0, 35.0],   # Kp_alt
        [4.0,  12.0]    # Kd_alt
    ]
}

# --- 5. DATA VERIFICATION QUALITY CONTROLLER ---
def check_sobol_data_quality(y_dict, penalty):
    print("\n--- 📊 DATA INTEGRITY AUDIT ---")
    for metric, values in y_dict.items():
        failures = np.sum(values == penalty)
        pct = (failures / len(values)) * 100.0
        print(f" • [{metric:17s}]: Failed/Diverged Runs = {failures:4d} / {len(values)} ({pct:.1f}%)")
    print("-" * 35)


# --- 6. EXECUTION BLOCK ---
if __name__ == "__main__":
    # Generate the Saltelli sample matrix configs
    # N is the base sample size (Must be a power of 2). 
    # Total evaluations generated = N * (2 * num_vars + 2)
    N = 2  # Start at 64/128 for rapid testing; scale up to 256+ for smooth interactions (S2)
    param_values = sobol_sampler.sample(problem, N, calc_second_order=True)
    num_simulations = param_values.shape[0]

    print(f"🚀 Initializing Sensitivity Solver Matrix Engine...")
    print(f"   ↳ Base Sample Size (N): {N}")
    print(f"   ↳ Total Flight Evaluators Scheduled: {num_simulations}\n")

    # Metrics layout definitions
    metrics_to_analyze = {
        "annoyance_PA": "Psychoacoustic Annoyance",
        "loudness_sone": "Loudness",
        "sharpness_acum": "Sharpness",
        "roughness_asper": "Roughness",
        "fluctuation_vacil": "Fluctuation Strength"
    }
    
    Y_metrics = {metric: np.zeros(num_simulations) for metric in metrics_to_analyze.keys()}
    PENALTY_VALUE = 99.0

    # Main evaluation trajectory tracking loop
    for i, row in enumerate(tqdm(param_values, desc="Simulating Trajectories")):
        try:
            # Unpack parameters matching problem array mapping indices
            r_rotor       = float(row[0])
            l_arm         = float(row[1])
            motor_bw      = float(row[2])
            v_kp          = float(row[3])
            v_ki          = float(row[4])
            att_kp        = float(row[5])
            alt_kp        = float(row[6])
            alt_kd        = float(row[7])

            # Initialize Unified Plant Configuration
            opt_phys_vars = {'motor_bandwidth_hz': motor_bw}
            plant = ParametricDronePlant(arm_length=l_arm, rotor_radius=r_rotor, design_vars=opt_phys_vars)
            drone_params = plant.to_dict()

            # Initialize Tracking Controller Architecture
            controller = DronePIDController(
                kp_vel=v_kp, ki_vel=v_ki,
                kp_att=att_kp, kd_att=0.4,
                kp_alt=alt_kp, kd_alt=alt_kd, ki_alt=15.0,
                mass=drone_params['mass'], 
                g=drone_params['g'],
                motor_bandwidth_hz=drone_params['motor_bandwidth_hz']
            )

            # Fire off simulation
            outputs = run_drone_acoustic_simulation(
                gains=None, # Passed inside controller object parameter now
                dryden_ts=dryden_ts, 
                waypoints=waypoints, 
                time_steps=time_steps, 
                dt=dt, 
                plant_params=drone_params, 
                initial_state=my_init_state
            )
            # Adjust unpacking depending on whether your function returns 7 items
            spl_fine, pa, _, hist_state, _, _, _ = outputs
            
            # Catch mathematical divergence limits
            if np.any(np.isnan(hist_state)) or np.any(np.isinf(hist_state)):
                raise ValueError("Flight path tracking diverged (Unstable Dynamics)")
                
            # Process and integrate time-domain signals into metric values
            for metric in metrics_to_analyze.keys():
                time_signal = np.asarray(pa[metric], dtype=float)
                dose_value = np.trapezoid(time_signal, x=time_steps)
                
                if np.isnan(dose_value) or np.isnan(time_signal).any():
                    raise ValueError("Acoustic matrix generated NaN elements")
                    
                Y_metrics[metric][i] = float(dose_value)
                
        except Exception as e:
            # Apply uniform penalty tracking mask if the controller layout crashes or flips
            for metric in metrics_to_analyze.keys():
                Y_metrics[metric][i] = PENALTY_VALUE

    # Verify tracking stats before computing decompositions
    check_sobol_data_quality(Y_metrics, PENALTY_VALUE)

    # --- 7. DECOMPOSE SENSITIVITY SCORES VIA SALIB ---
    print("\n📊 Computing variance decompositions...")
    
    for metric, display_name in metrics_to_analyze.items():
        print(f"\n" + f" Sobol Sensitivity Analysis: {display_name} ".center(68, "="))
        
        # Analyze the output array variance allocations
        Si = sobol_analyzer.analyze(problem, Y_metrics[metric], calc_second_order=True)
        
        # Print First-Order and Total-Order Tables
        print(f"\n--- MAIN EFFECTS (First-Order S1 & Total-Order ST) ---")
        print(f"{'Design Parameter':22s} | {'S1 ± Conf':<16} | {'ST ± Conf':<16}")
        print("-" * 62)
        for idx, name in enumerate(problem['names']):
            s1_str = f"{Si['S1'][idx]:.3f} ± {Si['S1_conf'][idx]:.3f}"
            st_str = f"{Si['ST'][idx]:.3f} ± {Si['ST_conf'][idx]:.3f}"
            print(f"{name:22s} | {s1_str:<16} | {st_str:<16}")
            
        # Print Interaction Pair Configurations Matrix
        print(f"\n--- SECOND-ORDER INTERACTIONS (S2 Pairs) ---")
        print(f"{'Parameter Pair':42s} | {'S2 ± Conf':<16}")
        print("-" * 65)
        
        num_vars = problem['num_vars']
        for idx1 in range(num_vars):
            for idx2 in range(idx1 + 1, num_vars):
                name1 = problem['names'][idx1]
                name2 = problem['names'][idx2]
                pair_label = f"{name1} <-> {name2}"
                
                s2_val = Si['S2'][idx1, idx2]
                s2_conf = Si['S2_conf'][idx1, idx2]
                
                if np.isnan(s2_val):
                    s2_str = "NaN (Low Samples)"
                else:
                    s2_str = f"{s2_val:.3f} ± {s2_conf:.3f}"
                    
                print(f"{pair_label:42s} | {s2_str:<16}")
                
    print("\n🎉 Sobol Sensitivity Multi-Indicator Sweep Finished Successfully!")