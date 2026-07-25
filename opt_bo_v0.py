import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from drone_pid_function_v18 import evaluate_drone_codesign, plot_comprehensive_diagnostics

# 1. Define your rigid boundaries (Must match the exact 14-element sequence)
# 1. Provide an ordered list of string names matching your 15 variables
variable_names = [
    'R_rotor', 'L_arm', 
    'vz_max', 't_climb', 'vx_max', 'vy_max', 
    'Kp_vel', 'Ki_vel', 'Kp_att', 'Kd_att', 'Kp_alt', 'Kd_alt', 
    'motor_bandwidth_hz', 
    'parameter2', 'parameter3'
]

# your existing bounds array...
bounds = [
    (0.10, 0.32), (0.35, 0.60),
    (2.0,  10.0), (1.0,  5.0), (3.0,  12.0), (0.0,  0.001),
    (0.05, 0.35), (0.01, 0.14), (2.0,  6.0),  (0.3,  0.9), (10.0, 35.0), (4.0,  12.0),
    (10.0, 30.0),
    (0.001, 0.05), (0.5,   2.0)
]

# 2. 🟢 CONVERT AUTOMATICALLY INTO SKOPT SPACE
# Zip matches each boundary pair with its string name instantly
space = [
    Real(low, high, name=name) 
    for (low, high), name in zip(bounds, variable_names)
]

# Set drone starting position (sitting perfectly at ground level)
my_init_state = np.zeros(12)
my_init_state[2] = 0.0 

# 2. Define the exact objective function adapter that SciPy expects
def objective(X):
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

# 3. Run the optimizer (n_calls is tiny compared to DE!)
res = gp_minimize(objective, space, n_calls=300, random_state=42)

print(f"Best score achieved: {res.fun}")
print(f"Best 14-variable configuration: {res.x}")

