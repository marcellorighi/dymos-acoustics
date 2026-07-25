import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 1. SALIB SENSITIVITY IMPORTS ---
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol as sobol_analyzer
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
from zwicker_annoyance_v2 import compute_zwicker_indicators_windowed


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

# 2. DEFINING THE SALIB PROBLEM (8 Variables)
problem = {
    'num_vars': 8,
    'names': [
        'R_rotor', 'L_arm', 
        'g1', 'g2', 'g3', 'g4', 'g5', 'g6'  # The 6 position/attitude gains
    ],
    'bounds': [
        [0.10, 0.32],   # R_rotor [m]
        [0.35, 0.60],   # L_arm [m]
        [0.05, 0.35],   # Gain 1
        [0.01, 0.14],   # Gain 2
        [1.0,  5.0],    # Gain 3
        [0.1,  1.0],    # Gain 4
        [5.0,  20.0],   # Gain 5
        [2.0,  10.0]    # Gain 6
    ]
}

def evaluate_single_sobol_config(i, row, my_init_state, dt, t_end, vz_max, vx_max, vy_max, t_climb):
    """
    Worker function executed in parallel on separate CPU cores.
    Returns a dictionary of calculated doses for this specific run index.
    """
    # Initialize a local dictionary for results
    local_results = {metric: PENALTY_VALUE for metric in metrics_to_analyze.keys()}
    
    try:
        # Unpack the 8 variables from SALib row
        r_rotor = float(row[0])
        l_arm   = float(row[1])
        gains   = [float(g) for g in row[2:8]]
        
        motor_bandwidth_hz = 20.0 
        ki_vel = 0.04

        # Reconstruct full 14-element vector X
        X_full = [
            r_rotor, l_arm,              
            vz_max, t_climb,             
            vx_max, vy_max,              
            gains[0], gains[1], gains[2], 
            gains[3], gains[4], gains[5], 
            motor_bandwidth_hz,          
            ki_vel                       
        ]

        # Evaluate the drone configuration
        outputs = evaluate_drone_codesign(
            X_full, initial_state=my_init_state, dt=dt, t_end=t_end, debug=True, mode='climb_cruise'
        )
        
        _, hist_state, time_steps, _, _, _, pa, _, _ = outputs
        
        # Stability check
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
        # Avoid flooding the terminal by printing errors on separate threads compactly
        print(f"\n⚠️ [Worker Core] Run #{i} hit instability: {type(e).__name__}: {e}")
        
    return local_results

if __name__ == "__main__":
    N = 256  # Set low for rapid pipeline validation, scale up to 128+ later
    param_values = sobol_sampler.sample(problem, N, calc_second_order=True)
    num_simulations = param_values.shape[0]

    metrics_to_analyze = {
        "annoyance_PA": "Psychoacoustic Annoyance",
        "loudness_sone": "Loudness",
        "sharpness_acum": "Sharpness",
        "roughness_asper": "Roughness",
        "fluctuation_vacil": "Fluctuation Strength"
    }
    
    Y_metrics = {metric: np.zeros(num_simulations) for metric in metrics_to_analyze.keys()}
    PENALTY_VALUE = 99.0

    print(f"🚀 Running Sobol Analysis via evaluate_drone_codesign wrapper...")
    print(f"Scheduled Evaluations: {num_simulations}\n")

    param_values = sobol_sampler.sample(problem, N, calc_second_order=True)
    num_simulations = param_values.shape[0]
    
    # Pre-allocate output matrices for SALib
    Y_metrics = {metric: np.zeros(num_simulations) for metric in metrics_to_analyze.keys()}

    print(f"🚀 Launching Parallel Sobol Analysis Across All CPU Cores...")
    print(f"Total Evaluations Scheduled: {num_simulations}\n")

    # Fire off Parallel execution pool
    NUM_CORES = 8
    # n_jobs=-1 automatically spins up a worker process for every logical CPU thread available
    results = Parallel(n_jobs=NUM_CORES)(
        delayed(evaluate_single_sobol_config)(
            i, row, my_init_state, dt, t_end, vz_max, vx_max, vy_max, t_climb
        )
        for i, row in enumerate(tqdm(param_values, desc="Simulating Trajectories (Parallel)"))
    )

    # Reassemble parallel worker outputs back into SALib's target arrays
    for i, local_res in enumerate(results):
        for metric in metrics_to_analyze.keys():
            Y_metrics[metric][i] = local_res[metric]

    print("\n🎉 All parallel simulations complete! Proceeding to Sobol analyzer...")

    sobol_results = {}

    for metric, display_name in metrics_to_analyze.items():
        print(f"\n==========================================")
        print(f" Sobol Sensitivity Analysis: {display_name}")
        print(f"==========================================")
        
        # Run the analyzer (ensure calc_second_order=True is set)
        # --- CALL THE CHECKER HERE ---
        # Put this right before you loop over sobol_analyzer.analyze()
        check_sobol_data_quality(Y_metrics, PENALTY_VALUE)

        Si = sobol_analyzer.analyze(problem, Y_metrics[metric], calc_second_order=True)
        sobol_results[metric] = Si  
        
        # 1. Print First-Order (S1) and Total-Order (ST) with Confidence Intervals
        print(f"\n--- MAIN EFFECTS (S1 & ST) ---")
        print(f"{'Gain Parameter':20s} | {'S1 ± Conf':<16} | {'ST ± Conf':<16}")
        print("-" * 60)
        for idx, name in enumerate(problem['names']):
            s1_str = f"{Si['S1'][idx]:.3f} ± {Si['S1_conf'][idx]:.3f}"
            st_str = f"{Si['ST'][idx]:.3f} ± {Si['ST_conf'][idx]:.3f}"
            print(f"{name:20s} | {s1_str:<16} | {st_str:<16}")
            
        # 2. Print Second-Order (S2) Interactions with Confidence Intervals
        print(f"\n--- SECOND-ORDER INTERACTIONS (S2 Matrix Pairs) ---")
        print(f"{'Parameter Pair':42s} | {'S2 ± Conf':<16}")
        print("-" * 65)
        
        num_vars = problem['num_vars']
        # Loop over all unique pairs (avoiding duplicates and self-interactions)
        for idx1 in range(num_vars):
            for idx2 in range(idx1 + 1, num_vars):
                name1 = problem['names'][idx1]
                name2 = problem['names'][idx2]
                pair_label = f"{name1} <-> {name2}"
                
                # Extract values from the S2 and S2_conf matrices
                s2_val = Si['S2'][idx1, idx2]
                s2_conf = Si['S2_conf'][idx1, idx2]
                
                # If your sample size N is low, some S2 arrays might contain NaNs or masked elements. 
                # We filter or handle them gracefully here.
                if np.isnan(s2_val):
                    s2_str = "NaN (Low Samples)"
                else:
                    s2_str = f"{s2_val:.3f} ± {s2_conf:.3f}"
                    
                print(f"{pair_label:42s} | {s2_str:<16}")
