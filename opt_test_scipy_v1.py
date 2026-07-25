import numpy as np
from scipy.optimize import differential_evolution
from drone_pid_function_v16 import evaluate_drone_codesign
from drone_pid_function_v16 import plot_comprehensive_diagnostics

# 1. Define your rigid boundaries (Must match the sequence your wrapper expects!)
# Sequence: [R_rotor, L_arm, v_max, t_climb, Kp_vel, Ki_vel, Kp_att, Kd_att, Kp_alt, Kd_alt]
bounds = [
    (0.10, 0.32),   # R_rotor [m]
    (0.35, 0.60),   # L_arm [m]
    (2.0,  10.0),   # v_max [m/s]
    (1.0,  5.0),    # t_climb [s]
    (3.0,  12.0),    # vx_max bounds <--- NEW!
    (0.0,  0.0),    # vy_max bounds (locked to 0 for now) <--- NEW!
    (0.05, 0.35),   # Kp_vel
    (0.01, 0.14),   # Ki_vel
    (2.0,  6.0),    # Kp_att
    (0.3,  0.9),    # Kd_att
    (10.0, 35.0),   # Kp_alt
    (4.0,  12.0)    # Kd_alt
]

my_init_state = np.zeros(12)
my_init_state[2] = 0.0 # 10 meters altitude

# 2. Define the exact objective function adapter that SciPy expects
def scipy_objective_adapter(X):
    """
    SciPy solvers expect the callable to return ONLY a single float number 
    representing the objective value. We must strip away the dictionary wrappers.
    """
    # Force debug=False so it runs fast and uses the safety try/except block
    result = evaluate_drone_codesign(X, initial_state=my_init_state, dt=0.005, t_end=8.0, debug=False, mode='climb_cruise')
    
    # Return the raw performance score to the optimizer
    return result["objective"]

from scipy.optimize import NonlinearConstraint

# 3. Execute the Optimization Loop
if __name__ == "__main__":
    print("🚀 Initializing Global Co-Design Optimization Loop...")
    
    # Run the genetic solver
    res = differential_evolution(
        scipy_objective_adapter, 
        bounds=bounds, 
        maxiter=12,       # Number of generations
        popsize=5,        # Population multiplier
        disp=True,         # Print progress in the console

        # --- HIGH SPEED DEBUGGING SETTINGS ---
        workers=-1,          # 🚀 Parallel processing: Uses ALL available CPU cores
        updating='deferred', # Required for parallel processing (updates population at end of generation)
        polish=False         # 🛑 Turns off the local optimizer refinement loop at the end
    )
    
    print("\n🎉 Optimization Complete!")
    print(f"Optimal Design Vector X*: {res.x}")
    print(f"Minimized Annoyance Dose:  {res.fun:.4f}")

    print(f"Minimized Annoyance Dose: {res.fun:.4f}")
    
    # ----------------------------------------------------
    # POST-OPTIMIZATION VISUALIZATION (SciPy)
    # ----------------------------------------------------
    print("\n📊 Extracting physics telemetry for the optimal design...")
    
    # Run the wrapper ONE final time with debug=True using the optimal vector
    _, hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa, rpm_results, spl_fine = evaluate_drone_codesign(
        res.x, initial_state=my_init_state, dt=0.005, t_end=8.0, debug=True, mode='climb_cruise'
    )

    # Call your comprehensive plotting dashboard function
    gains = res.x[6:]

    gain_bounds = bounds[6:]  # or bounds[6:12]

    rpm_history = np.column_stack((
    rpm_results["rpm_front"],
    rpm_results["rpm_left"],
    rpm_results["rpm_right"],
    rpm_results["rpm_rear"]
    ))

    if hist_state is not None and pa is not None:
        plot_comprehensive_diagnostics(hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa, gains, gain_bounds, rpm_history)
