import numpy as np
from pyoptsparse import Optimization, OPT

def objective_function(xdict):
    """
    Objective function for the optimization.
    xdict is a dictionary containing the design variables.
    """
    x = xdict["vars"]
    
    # Example objective: Rosenbrock function or your specific physics model
    obj = (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2
    
    funcs = {}
    funcs["obj"] = obj
    
    # Example constraint: x[0] + x[1] >= 1.0
    funcs["con"] = x[0] + x[1]
    
    fail = False
    return funcs, fail

def run_optimization():
    # 1. Initialize the Optimization object
    opt_prob = Optimization("Nonlinear_System_Optimization", objective_function)

    # 2. Add design variables
    # Lower bound (lb), Upper bound (ub), Initial value (value)
    opt_prob.addVarGroup(
        "vars", 
        nVars=2, 
        varType="c", 
        value=np.array([0.5, 0.5]), 
        lower=np.array([-2.0, -2.0]), 
        upper=np.array([2.0, 2.0])
    )

    # 3. Add objective
    opt_prob.addObj("obj")

    # 4. Add constraints
    # lower=1.0 means 'con' >= 1.0
    opt_prob.addConGroup("con", 1, lower=1.0, upper=None)

    # 5. Setup IPOPT solver through pyoptsparse
    # Note: Ensure IPOPT is compiled and available in your environment
    opt_options = {
        "max_iter": 500,
        "tol": 1e-6,
        "print_level": 5,        # 0 (silent) to 12 (very verbose)
        "limited_memory_update_type": "bfgs", # Useful if exact Hessians aren't provided
        "nlp_scaling_method": "gradient-based",
        "derivative_test": "none" # Set to 'first-order' to check gradients
    }

    # Initialize the solver
    opt_solver = OPT("ipopt", options=opt_options)

    # 6. Run optimization
    # If your objective_function does not provide sensitivities, 
    # pyoptsparse can use finite differences (sens='fd')
    print("\n--- Starting IPOPT Optimization via pyoptsparse ---\n")
    sol = opt_solver(opt_prob, sens="fd") 

    # 7. Print results
    print("\n--- Optimization Results ---")
    print(sol)
    
    # Access specific values
    final_vars = sol.xStar
    print(f"Optimal Variables: {final_vars}")

if __name__ == "__main__":
    try:
        run_optimization()
    except Exception as e:
        print(f"Error running optimization: {e}")
        print("Check if pyoptsparse and IPOPT are correctly installed.")
