import openmdao.api as om
import numpy as np

# ---------------------------------------------------------
# 1. EXTERNAL FUNCTIONS (The "Black Box" or Legacy Code)
# ---------------------------------------------------------

def objective_function(w, t):
    """Returns output 'a' = w * t"""
    return w * t

def objective_function_partials(w, t):
    """Returns derivatives of 'a' with respect to w and t"""
    da_dw = t
    da_dt = w
    return da_dw, da_dt

def support_function(w, t):
    """Returns output 'b' = w + t"""
    return w + t

def support_function_partials(w, t):
    """Returns derivatives of 'b' with respect to w and t"""
    db_dw = 1.0
    db_dt = 1.0
    return db_dw, db_dt

# ---------------------------------------------------------
# 2. OPENMDAO COMPONENT WRAPPER
# ---------------------------------------------------------

class ExternalEvalComponent(om.ExplicitComponent):
    """
    A component that wraps external functions for calculation.
    """
    def setup(self):
        # Inputs
        self.add_input('w', val=1.0)
        self.add_input('t', val=1.0)

        # Outputs
        self.add_output('a', val=1.0)
        self.add_output('b', val=1.0)

        # Partials
        # For 'a', we will use the external analytic partials
        self.declare_partials('a', ['w', 't'])
        
        # For 'b', let's demonstrate OpenMDAO's Complex Step method
        # instead of using the external partials function.
        self.declare_partials('b', ['w', 't'], method='cs')

    def compute(self, inputs, outputs):
        w = inputs['w']
        t = inputs['t']

        # Call external functions
        outputs['a'] = objective_function(w, t)
        outputs['b'] = support_function(w, t)

    def compute_partials(self, inputs, partials):
        """Only needed for partials NOT using 'method=cs' or 'fd'"""
        w = inputs['w']
        t = inputs['t']
        
        da_dw, da_dt = objective_function_partials(w, t)
        
        partials['a', 'w'] = da_dw
        partials['a', 't'] = da_dt

# ---------------------------------------------------------
# 3. OPTIMIZATION SETUP
# ---------------------------------------------------------

if __name__ == "__main__":
    prob = om.Problem()
    model = prob.model

    # Add the component
    model.add_subsystem('comp', ExternalEvalComponent(), promotes=['*'])

    # Using pyOptSparse with IPOPT
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'IPOPT'

    # IPOPT specific settings
    prob.driver.opt_settings['max_iter'] = 600
    prob.driver.opt_settings['print_level'] = 5
    prob.driver.opt_settings['tol'] = 1e-7

    # Setup Optimizer (Scipy SLSQP)
    # prob.driver = om.ScipyOptimizeDriver()
    # prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.options['disp'] = True

    # Design Variables
    model.add_design_var('w', lower=0.5, upper=10.0)
    model.add_design_var('t', lower=0.5, upper=10.0)

    # Objective: Minimize 'a'
    model.add_objective('a')

    # Constraint: 'b' must be at least 5.0
    model.add_constraint('b', lower=5.0)

    # Run
    prob.setup()
    prob.run_driver()

    # Results
    print("\n--- Optimization Results ---")
    print(f"Optimal w: {prob.get_val('w')[0]:.4f}")
    print(f"Optimal t: {prob.get_val('t')[0]:.4f}")
    print(f"Objective a (w*t): {prob.get_val('a')[0]:.4f}")
    print(f"Constraint b (w+t): {prob.get_val('b')[0]:.4f}")
