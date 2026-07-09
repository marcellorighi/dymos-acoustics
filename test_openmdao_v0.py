import openmdao.api as om
import numpy as np
# Import your fixed function
from drone_pid_function_v10 import evaluate_drone_codesign

class DroneCoDesignComponent(om.ExplicitComponent):
    """
    Wraps your custom simulation into an OpenMDAO-compatible component.
    """
    def setup(self):
        # Define inputs matching your design vector X elements
        self.add_input('R_rotor', val=0.22)
        self.add_input('L_arm',   val=0.40)
        self.add_input('v_max',   val=5.0)
        self.add_input('t_climb', val=3.0)
        self.add_input('gains',   val=np.array([0.15, 0.04, 3.0, 0.5, 15.0, 7.0]))
        
        # Define outputs for the objective and constraints
        self.add_output('annoyance_dose', val=99.0)
        
        # Tell OpenMDAO to approximate gradients using Finite Difference
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        # 1. Pack the inputs back into the exact vector shape X your function expects
        X = [
            inputs['R_rotor'][0],
            inputs['L_arm'][0],
            inputs['v_max'][0],
            inputs['t_climb'][0],
            *inputs['gains']  # Unpacks the 6 gain values
        ]
        
        # 2. Call your simulation function (always use debug=False for speed)
        result = evaluate_drone_codesign(X, dt=0.01, t_end=4.0, debug=False)
        
        # 3. Assign the objective score
        outputs['annoyance_dose'] = result['objective']

# -----------------------------------------------------------------
#   SET UP THE OPTIMIZATION PROBLEM LOOP
# -----------------------------------------------------------------
if __name__ == "__main__":
    prob = om.Problem()
    model = prob.model
    
    # Add your component to the workflow architecture
    model.add_subsystem('drone_sim', DroneCoDesignComponent(), promotes=['*'])
    
    # Configure a gradient-based driver (SLSQP comes bundled with SciPy)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 50
    prob.driver.options['disp'] = True
    
    # Define design variable boundaries explicitly for the driver
    model.add_design_var('R_rotor', lower=0.10, upper=0.35)
    model.add_design_var('L_arm',   lower=0.20, upper=0.60)
    model.add_design_var('v_max',   lower=2.0,  upper=10.0)
    model.add_design_var('t_climb', lower=1.0,  upper=5.0)
    
    # Define the core objective OpenMDAO must minimize
    model.add_objective('annoyance_dose')
    
    # Initialize variables and execute
    prob.setup()
    prob.run_driver()
    
    # Print out the optimized results
    print("\n🎯 OpenMDAO Gradient-Based Optimization Complete!")
    print(f"Optimized Rotor Radius: {prob.get_val('R_rotor')[0]:.4f} m")
    print(f"Optimized Arm Length:   {prob.get_val('L_arm')[0]:.4f} m")
    print(f"Minimized Annoyance:     {prob.get_val('annoyance_dose')[0]:.4f}")

