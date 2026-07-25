import numpy as np
from scipy.optimize import differential_evolution
# Make sure to update your import string if your filename changes from v16!
from drone_pid_function_v18 import evaluate_drone_codesign, plot_comprehensive_diagnostics

# 1. Define your rigid boundaries (Must match the exact 14-element sequence)
bounds = [
    # --- Structural / Geometry (0-1) ---
    (0.10, 0.32),   # 0: R_rotor [m]
    (0.35, 0.60),   # 1: L_arm [m]
    
    # --- Trajectory Profile Limits (2-5) ---
    (2.0,  10.0),   # 2: vz_max [m/s]
    (1.0,  5.0),    # 3: t_climb [s]
    (3.0,  12.0),   # 4: vx_max [m/s]
    (0.0,  0.0),    # 5: vy_max [m/s] (locked to 0 for now)
    
    # --- Controller Gains (6-11) ---
    (0.05, 0.35),   # 6: Kp_vel
    (0.01, 0.14),   # 7: Ki_vel (Your consolidated velocity integrator)
    (2.0,  6.0),    # 8: Kp_att
    (0.3,  0.9),    # 9: Kd_att
    (10.0, 35.0),   # 10: Kp_alt
    (4.0,  12.0),   # 11: Kd_alt
    
    # --- New Plant Hardware / Actuator DV (12) ---
    (10.0, 30.0),   # 12: motor_bandwidth_hz [Hz] (actuator bandwidth lag limits)
    
    # --- New Generics / Software DV (13-14) ---
    (0.001, 0.05),  # 13: parameter2 (e.g., control tuning parameter)
    (0.5,   2.0)    # 14: parameter3 (e.g., secondary weight parameter)
]

# Set drone starting position (sitting perfectly at ground level)
my_init_state = np.zeros(12)
my_init_state[2] = 0.0 

# 2. Define the exact objective function adapter that SciPy expects
def scipy_objective_adapter(X):
    """
    SciPy solvers expect the callable to return ONLY a single float number 
    representing the objective value. We must strip away the dictionary wrappers.
    """
    # Force debug=False so it runs fast and uses the safety try/except block
    result = evaluate_drone_codesign(
        X, initial_state=my_init_state, dt=0.005, t_end=8.0, debug=False, mode='climb_cruise'
    )
    # Return the raw performance score to the optimizer
    return result["objective"]


# 3. Execute the Optimization Loop
if __name__ == "__main__":
    print(f"🚀 Initializing Global Co-Design Optimization Loop with {len(bounds)} design variables...")
    
    # Run the genetic solver
    res = differential_evolution(
        scipy_objective_adapter, 
        bounds=bounds, 
        maxiter=12,          # Number of generations
        popsize=5,           # Population multiplier
        disp=True,           # Print progress in the console

        # --- HIGH SPEED MULTI-CORE PROCESSING ---
        workers=-1,          # Parallel processing: Uses ALL available CPU cores
        updating='deferred', # Updates population at the end of generations (required for workers=-1)
        polish=False         # Turns off local Nelder-Mead optimization refinement at the end
    )
    
    print("\n🎉 Optimization Complete!")
    print(f"Optimal Design Vector X*: {res.x}")
    print(f"Minimized Annoyance Dose:  {res.fun:.4f}")
    
    # ----------------------------------------------------
    # POST-OPTIMIZATION VISUALIZATION & DIAGNOSTICS
    # ----------------------------------------------------
    print("\n📊 Extracting physics telemetry for the optimal design...")
    
    # Run the wrapper ONE final time with debug=True to extract full simulation logs
    _, hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa, rpm_results, spl_fine = evaluate_drone_codesign(
        res.x, initial_state=my_init_state, dt=0.005, t_end=8.0, debug=True, mode='climb_cruise'
    )

    # Cleanly slice out ONLY the core 6 gains (indices 6 to 12) for the plotting script
    gains = res.x[6:12]
    gain_bounds = bounds[6:12] 

    # Bundle the dictionary outputs into a 2D matrix matching legacy shapes
    rpm_history = np.column_stack((
        rpm_results["rpm_front"],
        rpm_results["rpm_left"],
        rpm_results["rpm_right"],
        rpm_results["rpm_rear"]
    ))

    # Send data directly into your diagnostic dashboard
    if hist_state is not None and pa is not None:
        plot_comprehensive_diagnostics(
            hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, 
            pa, gains, gain_bounds, rpm_history
        )
        