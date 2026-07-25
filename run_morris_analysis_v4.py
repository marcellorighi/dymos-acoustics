import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. SALIB SENSITIVITY IMPORTS ---
# --- 1. SALIB SENSITIVITY IMPORTS (Updated for Morris) ---
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris as morris_analyzer
from joblib import Parallel, delayed

# --- 2. YOUR DRONE PROJECT IMPORTS ---
# Replace 'drone_pid_function_v16' with your actual filename if it changes!
from drone_pid_function_v18 import (
    evaluate_drone_codesign, plot_comprehensive_diagnostics,
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
from zwicker_annoyance_v3 import compute_zwicker_indicators_windowed


def check_sobol_data_quality(Y_dict, penalty_value):
    print("\n" + "="*50)
    print("      DATA QUALITY & VALIDATION REPORT")
    print("="*50)
    
    for metric_key, y_data in Y_dict.items():
        num_total = len(y_data)
        num_nans = np.isnan(y_data).sum()
        num_infs = np.isinf(y_data).sum()
        
        # Check how many times the penalty value was hit
        # Using np.isclose in case of minor floating point deviations
        num_penalties = np.sum(np.isclose(y_data, penalty_value))
        num_valid = num_total - num_nans - num_infs - num_penalties
        
        # Calculate unique values and variance
        unique_vals = np.unique(y_data)
        variance = np.var(y_data)
        
        print(f"\n[Metric: {metric_key}]")
        print(f"  -> Total Samples:        {num_total}")
        print(f"  -> Valid, Unique Runs:   {num_valid}")
        print(f"  -> Explicit NaN Values:  {num_nans}")
        print(f"  -> Explicit Inf Values:  {num_infs}")
        print(f"  -> Penalized Runs ({penalty_value}): {num_penalties} ({(num_penalties/num_total)*100:.1f}%)")
        print(f"  -> Total Data Variance:  {variance:.6e}")
        
        # Detailed Diagnosis
        if num_penalties == num_total:
            print("  🛑 CRITICAL ERROR: 100% of your runs failed or were penalized!")
            print("     This is why Sobol returns NaNs. Check your try/except block for hidden code errors.")
        elif variance == 0:
            print("  ⚠️ WARNING: Variance is exactly 0.0. All data points have identical values.")
            print(f"     Unique values found: {unique_vals}")
        elif len(unique_vals) < 5:
            print(f"  ⚠️ WARNING: Extremely low data diversity. Unique values: {unique_vals}")
        else:
            print("  ✅ Data looks mathematically valid for sensitivity analysis.")
            
    print("="*50 + "\n")


# 1. DEFINE YOUR CONSTANT FLIGHT CONFIGURATION (Must match your __main__ test case)
dt = 0.002 
t_end = 12.01
vz_max = 5.0 
vx_max = 5.00 
vy_max = 0.0 
t_climb = 3.0

# Set drone starting position (10m up, sitting perfectly still)
my_init_state = np.zeros(12)
my_init_state[2] = 7.76 

# --- 🟢 STEP 1: DEFINE GEOMETRY AS FIXED CONSTANTS ---
FIXED_R_ROTOR = 0.25  # Replace with your nominal/optimal rotor radius
FIXED_L_ARM   = 0.45  # Replace with your nominal/optimal arm length

# --- 🟢 STEP 2: DEFINE THE CONTROLLER-ONLY PROBLEM SPACE (7 Variables) ---
# problem = {
#     'num_vars': 7,
#     'names': [
#         'g1', 'g2', 'g3', 'g4', 'g5', 'g6',  # The 6 flight control gains
#         'motor_bandwidth_hz'                 # The dynamic hardware parameter
#     ],
#     'bounds': [
#         [0.05, 0.35],   # Gain 1
#         [0.01, 0.14],   # Gain 2
#         [1.0,  5.0],    # Gain 3
#         [0.1,  1.0],    # Gain 4
#         [5.0,  20.0],   # Gain 5
#         [2.0,  10.0],   # Gain 6
#         [5.0,  40.0]    # motor_bandwidth_hz range sweep [Hz]
#     ]
# }

# --- 🟢 STEP 1: DEFINE THE FULL 14-VARIABLE SPACE ---
problem = {
    'num_vars': 15,
    'names': [
        'R_rotor', 'L_arm',           # 0, 1
        'vz_max', 't_climb',           # 2, 3
        'vx_max', 'vy_max',           # 4, 5
        'Kp_vel', 'Ki_vel',           # 6, 7
        'Kp_att', 'Kd_att',           # 8, 9
        'Kp_alt', 'Kd_alt',           # 10, 11
        'motor_bandwidth_hz',         # 12
        'parameter2', 'parameter3'    # 13, 14
    ],
    # 🟢 ADD THIS LINE: Explicitly match every variable name to an individual group
    'groups': [
        'R_rotor', 'L_arm',           
        'vz_max', 't_climb',           
        'vx_max', 'vy_max',           
        'Kp_vel', 'Ki_vel',           
        'Kp_att', 'Kd_att',           
        'Kp_alt', 'Kd_alt',           
        'motor_bandwidth_hz',         
        'parameter2', 'parameter3'    
    ],
    'bounds': [
        (0.10, 0.32),   # 0: R_rotor [m]
        (0.35, 0.60),   # 1: L_arm [m]
        (2.0,  10.0),   # 2: vz_max [m/s]
        (1.0,  5.0),    # 3: t_climb [s]
        (3.0,  12.0),   # 4: vx_max [m/s]
        (0.0,  1e-6),   # 5: vy_max [m/s] (Using the tiny non-zero bound trick)
        (0.05, 0.35),   # 6: Kp_vel
        (0.01, 0.14),   # 7: Ki_vel
        (2.0,  6.0),    # 8: Kp_att
        (0.3,  0.9),    # 9: Kd_att
        (10.0, 35.0),   # 10: Kp_alt
        (4.0,  12.0),   # 11: Kd_alt
        (10.0, 30.0),   # 12: motor_bandwidth_hz [Hz]
        (0.001, 0.05),  # 13: parameter2
        (0.5,   2.0)    # 14: parameter3
    ]
}

def evaluate_single_morris_config(i, row, my_init_state, dt, t_end):
    """
    Worker function executed in parallel across CPU cores.
    Processes a complete 14-element vector row directly from SALib.
    """
    local_results = {metric: PENALTY_VALUE for metric in metrics_to_analyze.keys()}
    
    try:
        # 🟢 STEP 2: CONVERT ROW DIRECTLY TO LIST 
        # Because problem['num_vars'] == 14, row contains all 14 elements in order.
        X_full = [float(val) for val in row]
        
        # Unpack velocity vector components directly from the row positions for internal logic if needed
        vz_max, t_climb = X_full[2], X_full[3]
        vx_max, vy_max = X_full[4], X_full[5]

        # Evaluate the drone configuration
        outputs = evaluate_drone_codesign(
            X_full, initial_state=my_init_state, dt=dt, t_end=t_end, debug=True, mode='climb_cruise'
        )
        
        _, hist_state, time_steps, _, _, _, pa, _, _ = outputs
        
        # Stability validation
        if np.any(np.isnan(hist_state)) or np.any(np.isinf(hist_state)):
            raise ValueError("Flight path tracking diverged.")
            
        acoustic_time = np.asarray(pa.get('time', pa.get('t', time_steps)), dtype=float)
            
        for metric in metrics_to_analyze.keys():
            time_signal = np.asarray(pa[metric], dtype=float)
            
            if len(time_signal) == len(acoustic_time):
                dose_value = np.trapezoid(time_signal, x=acoustic_time)
            elif len(time_signal) == len(time_steps):
                dose_value = np.trapezoid(time_signal, x=time_steps)
            else:
                dose_value = np.trapezoid(time_signal) 
            
            if np.isnan(dose_value):
                raise ValueError("Acoustic output contained NaN elements")
                
            local_results[metric] = float(dose_value)
            
    except Exception as e:
        print(f"\n⚠️ [Worker Core] Run #{i} hit instability: {type(e).__name__}: {e}")
        
    return local_results


def plot_morris_screening(morris_indices, metric_name="Acoustic Metric"):
    """
    Creates a standard mu-star vs sigma scatter plot for Morris screening results.
    
    Parameters:
    - morris_indices: The dictionary returned by morris_analyzer.analyze()
    - metric_name: String label for the plot title (e.g., "Psychoacoustic Annoyance")
    """
    names = morris_indices['names']
    mu_star = morris_indices['mu_star']
    sigma = morris_indices['sigma']
    
    plt.figure(figsize=(8, 6), dpi=100)
    
    # 1. Plot the parameters as distinct points
    plt.scatter(mu_star, sigma, color='navy', s=100, zorder=3, edgecolors='black')
    
    # 2. Add text labels next to each dot with a slight offset
    for i, txt in enumerate(names):
        plt.annotate(
            txt, 
            (mu_star[i], sigma[i]), 
            textcoords="offset points", 
            xytext=(8, 5), 
            ha='left', 
            fontsize=10, 
            weight='bold'
        )
    
    # 3. Draw the V-shaped boundary lines to define quadrants
    max_val = max(max(mu_star), max(sigma)) * 1.1
    x_line = np.linspace(0, max_val, 100)
    
    # Diagonal line: Above this line means interactions dominate; below means linear effects dominate
    plt.plot(x_line, x_line, color='gray', linestyle='--', alpha=0.7, label=r'$\sigma = \mu^*$')
    
    # 4. Label the behavioral regions/quadrants
    plt.text(max_val * 0.1, max_val * 0.8, 'Highly Non-Linear /\nStrong Interactions', 
             color='darkred', fontsize=9, fontstyle='italic', alpha=0.7)
    plt.text(max_val * 0.6, max_val * 0.2, 'Linear /\nMonotonic Effects', 
             color='darkgreen', fontsize=9, fontstyle='italic', alpha=0.7)
    
    # 5. Aesthetics & Labels
    plt.title(f"Morris Sensitivity Screening: {metric_name}", fontsize=13, pad=15, weight='bold')
    plt.xlabel(r"Total Influence ($\mu^*$)", fontsize=11, labelpad=8)
    plt.ylabel(r"Non-Linearity & Interactions ($\sigma$)", fontsize=11, labelpad=8)
    
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid(True, linestyle=':', alpha=0.5, zorder=0)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save a high-res version for a paper/thesis
    filename = f"morris_screening_{metric_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"📊 Morris plot saved successfully as: {filename}")
    plt.show()

if __name__ == "__main__":
    # --- 🟢 STEP 3: CONFIGURE MORRIS SAMPLER CONFIGURATIONS ---
    # num_levels must be an even number (usually 4) to ensure grid symmetry
    num_grid_levels = 4 
    num_trajectories = 2  # r parameter
    
    # Generate the sample matrix
    param_values = morris_sampler.sample(
        problem, 
        N=num_trajectories, 
        num_levels=num_grid_levels
    )
    num_simulations = param_values.shape[0]
    
    # Setup infrastructure variables
    my_init_state = np.zeros(12)
    dt, t_end = 0.005, 8.0
    PENALTY_VALUE = 99.0
    
    metrics_to_analyze = {
        "annoyance_PA": "Psychoacoustic Annoyance",
        "loudness_sone": "Loudness",
        "sharpness_acum": "Sharpness",
        "roughness_asper": "Roughness",
        "fluctuation_vacil": "Fluctuation Strength"
    }
    
    Y_metrics = {metric: np.zeros(num_simulations) for metric in metrics_to_analyze.keys()}

    print(f"🚀 Launching Parallel Morris Screening Across {num_simulations} Configurations...")

    # Execute computation graph over parallel processing pool
    NUM_CORES = 8
    results = Parallel(n_jobs=NUM_CORES)(
        delayed(evaluate_single_morris_config)(
            i, row, my_init_state, dt, t_end
        )
        for i, row in enumerate(tqdm(param_values, desc="Screening Grid Space"))
    )

    # Map output collections back into the primary metric collection tables
    for i, local_res in enumerate(results):
        for metric in metrics_to_analyze.keys():
            Y_metrics[metric][i] = local_res[metric]

    # --- 🟢 STEP 4: ANALYZE AND PLOT ---
    # 🟢 1. Initialize the dictionary with the correct name
    morris_results = {}

    for metric, display_name in metrics_to_analyze.items():
        print(f"\n==========================================")
        print(f" Morris Sensitivity Screening: {display_name}")
        print(f"==========================================")
        
        # Run the Morris analyzer
        Si = morris_analyzer.analyze(
            problem, 
            param_values, 
            Y_metrics[metric], 
            num_levels=num_grid_levels
        )
        
        # 🟢 2. Store it using 'morris_results' (NOT sobol_results)
        morris_results[metric] = Si  
        
        # Print Morris Sensitivity Matrix Metrics to terminal
        print(f"{'Variable Name':22s} | {'mu_star':<10} | {'sigma':<10}")
        print("-" * 50)
        for idx, name in enumerate(problem['names']):
            print(f"{name:22s} | {Si['mu_star'][idx]:.4f} | {Si['sigma'][idx]:.4f}")

    # for metric, display_name in metrics_to_analyze.items():
    #     print(f"\n==========================================")
    #     print(f" Morris Results: {display_name}")
    #     print(f"==========================================")
        
    #     # Calculate Elementary Effects
    #     Si = morris_analyzer.analyze(
    #         problem, 
    #         param_values, 
    #         Y_metrics[metric], 
    #         num_levels=num_grid_levels
    #     )
        
    #     # Output values to console terminal log
    #     print(f"{'Variable Name':22s} | {'mu_star':<10} | {'sigma':<10}")
    #     print("-" * 50)
    #     for idx, name in enumerate(problem['names']):
    #         print(f"{name:22s} | {Si['mu_star'][idx]:.4f} | {Si['sigma'][idx]:.4f}")
            
        # Call your visual scatter plot engine
        plot_morris_screening(Si, metric_name=display_name)

    import csv
    import json

    # Define the output filenames
    csv_filename = "morris_screening_results.csv"
    json_filename = "morris_raw_data.json"

    print(f"\n💾 Saving sensitivity metrics to text files...")

    # --- WRITE TO HUMAN-READABLE CSV ---
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Create the column header row
        header = ["Parameter"]
        for metric in metrics_to_analyze.keys():
            header.extend([f"{metric}_mu_star", f"{metric}_sigma"])
        writer.writerow(header)
        
        # Write the calculated values row by row for each parameter
        for idx, name in enumerate(problem['names']):
            row = [name]
            for metric in metrics_to_analyze.keys():
                mu_star_val = morris_results[metric]['mu_star'][idx]
                sigma_val = morris_results[metric]['sigma'][idx]
                row.extend([mu_star_val, sigma_val])
            writer.writerow(row)

    print(f"✅ CSV Summary saved to: {csv_filename}")

    # --- WRITE TO RAW JSON BACKUP ---
    # Convert internal NumPy objects to clean standard lists for serialization
    json_export_data = {}
    for metric, data in morris_results.items():
        json_export_data[metric] = {
            "names": list(data['names']),
            "mu": list(data['mu']),
            "mu_star": list(data['mu_star']),
            "sigma": list(data['sigma'])
        }

    with open(json_filename, 'w') as json_file:
        json.dump(json_export_data, json_file, indent=4)

    print(f"✅ Raw structural data saved to: {json_filename}")


# if __name__ == "__main__":
#     # --- MORRIS CONFIGURATION PARAMETERS ---
#     # N is the number of trajectories. 15-20 is highly standard and accurate.
#     # num_levels is the grid resolution (must be an even integer, usually 4 or 6).
#     N_trajectories = 20  
#     num_grid_levels = 4  
    
#     metrics_to_analyze = {
#         "annoyance_PA": "Psychoacoustic Annoyance",
#         "loudness_sone": "Loudness",
#         "sharpness_acum": "Sharpness",
#         "roughness_asper": "Roughness",
#         "fluctuation_vacil": "Fluctuation Strength"
#     }
#     PENALTY_VALUE = 99.0

#     # 🟢 Generate Morris trajectories
#     # Total Runs = N * (D + 1) -> 20 * (7 + 1) = 160 total simulations!
#     param_values = morris_sampler.sample(
#         problem, 
#         N=N_trajectories, 
#         num_levels=num_grid_levels
#     )
#     num_simulations = param_values.shape[0]
    
#     # Pre-allocate output matrices for SALib
#     Y_metrics = {metric: np.zeros(num_simulations) for metric in metrics_to_analyze.keys()}

#     print(f"🚀 Running Morris Sensitivity Screening...")
#     print(f"Total Simulations Scheduled: {num_simulations} (Across {N_trajectories} trajectories)")

#     # Fire off Parallel execution pool
#     NUM_CORES = 8  # Set to $SLURM_CPUS_PER_TASK if running on the cluster
#     results = Parallel(n_jobs=NUM_CORES)(
#         delayed(evaluate_single_sobol_config)(  # The worker logic remains identical
#             i, row, my_init_state, dt, t_end, vz_max, vx_max, vy_max, t_climb
#         )
#         for i, row in enumerate(tqdm(param_values, desc="Simulating Trajectories (Parallel)"))
#     )

#     # Reassemble parallel worker outputs
#     for i, local_res in enumerate(results):
#         for metric in metrics_to_analyze.keys():
#             Y_metrics[metric][i] = local_res[metric]

#     print("\n🎉 All parallel simulations complete! Analyzing elementary effects...")

#     # Run the quality check once to make sure the simulation outputs vary properly
#     check_sobol_data_quality(Y_metrics, PENALTY_VALUE)

#     morris_results = {}

#     for metric, display_name in metrics_to_analyze.items():
#         print(f"\n==========================================")
#         print(f" Morris Sensitivity Screening: {display_name}")
#         print(f"==========================================")
        
#         # 🟢 Run the Morris analyzer
#         Si = morris_analyzer.analyze(
#             problem, 
#             param_values, 
#             Y_metrics[metric], 
#             num_levels=num_grid_levels
#         )
#         morris_results[metric] = Si  
        
#         # Print Morris Sensitivity Matrix Metrics
#         print(f"\n{'Gain Parameter':20s} | {'mu_star (Total Impact)':<22} | {'sigma (Interactions)':<20}")
#         print("-" * 70)
#         for idx, name in enumerate(problem['names']):
#             # mu_star evaluates overall influence (similar to ST in Sobol)
#             # sigma evaluates non-linearities and/or interaction effects
#             mu_star_str = f"{Si['mu_star'][idx]:.4f}"
#             sigma_str = f"{Si['sigma'][idx]:.4f}"
#             print(f"{name:20s} | {mu_star_str:<22} | {sigma_str:<20}")

#     for metric, display_name in metrics_to_analyze.items():
#             # ... (Your current morris_analyzer.analyze code sits here) ...
#             Si = morris_analyzer.analyze(problem, param_values, Y_metrics[metric], num_levels=num_grid_levels)
#             morris_results[metric] = Si  
            
#             # 🟢 Add the plotting call right here
#             plot_morris_screening(Si, metric_name=display_name)