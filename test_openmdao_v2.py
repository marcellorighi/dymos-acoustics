import openmdao.api as om
import numpy as np
from drone_pid_function_v11 import evaluate_drone_codesign
from drone_pid_function_v11 import plot_comprehensive_diagnostics
class DroneCoDesignComponent(om.ExplicitComponent):
    """
    Wraps your custom simulation into an OpenMDAO-compatible component,
    tracking both physical attributes and individual control gains.
    """
    def setup(self):
        # Physical Design Variables
        self.add_input('R_rotor', val=0.22)
        self.add_input('L_arm',   val=0.40)
        self.add_input('v_max',   val=5.0)
        self.add_input('t_climb', val=3.0)
        self.add_input('vx_max', val=2.0)
        self.add_input('vy_max', val=0.0)
        
        # Controller Gain Variables (Separated for explicit boundary tracking)
        self.add_input('Kp_vel',  val=0.15)
        self.add_input('Ki_vel',  val=0.04)
        self.add_input('Kp_att',  val=3.0)
        self.add_input('Kd_att',  val=0.5)
        self.add_input('Kp_alt',  val=15.0)
        self.add_input('Kd_alt',  val=7.0)
        
        # Optimization Target
        self.add_output('annoyance_dose', val=99.0)
        
        # Centralized finite-differencing step sizing (crucial for noisy PIDs)
        self.declare_partials(of='*', wrt='*', method='fd', step=1e-4)

    def compute(self, inputs, outputs):
        # Reconstruct the 12-dimensional design vector X for your function
        X = [
            inputs['R_rotor'][0],
            inputs['L_arm'][0],
            inputs['v_max'][0],
            inputs['t_climb'][0],
            inputs['vx_max'][0],
            inputs['vy_max'][0],
            inputs['Kp_vel'][0],
            inputs['Ki_vel'][0],
            inputs['Kp_att'][0],
            inputs['Kd_att'][0],
            inputs['Kp_alt'][0],
            inputs['Kd_alt'][0]
        ]
        
        # Execute tracking-safe evaluation wrapper
        result = evaluate_drone_codesign(X, dt=0.01, t_end=5.0, debug=False)
        outputs['annoyance_dose'] = result['objective']

# -----------------------------------------------------------------
#   SNOPT OPTIMIZATION EXECUTION LOOP
# -----------------------------------------------------------------
if __name__ == "__main__":
    prob = om.Problem()
    model = prob.model
    
    model.add_subsystem('drone_sim', DroneCoDesignComponent(), promotes=['*'])
    
    # 1. Switch Driver to pyOptSparse (Interface required to call SNOPT)
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    
    # 2. Configure Professional SNOPT Performance Options
    snopt_options = prob.driver.opt_settings
    snopt_options['Major iterations limit'] = 20
    snopt_options['Minor iterations limit'] = 500
    snopt_options['iSumm'] = 6 
    snopt_options['Major feasibility tolerance'] = 1e-5  # Constraint relaxation threshold
    snopt_options['Major optimality tolerance'] = 1e-4   # Gradient absolute threshold
    snopt_options['Summary file'] = 'snopt_summary.out'   # Real-time streaming log
    snopt_options['Print file'] = 'snopt_print.out'
    
    # 3. Add Physical Design Variables with Bounds
    model.add_design_var('R_rotor', lower=0.10, upper=0.35)
    model.add_design_var('L_arm',   lower=0.20, upper=0.60)
    model.add_design_var('v_max',   lower=2.0,  upper=10.0)
    model.add_design_var('t_climb', lower=1.0,  upper=5.0)
    model.add_design_var('vx_max', lower=0.0, upper=8.0)
    model.add_design_var('vy_max', lower=0.0, upper=0.0) # Locked to 0 for now
    
    # 4. Add Control Gain Design Variables with Custom Bounds
    model.add_design_var('Kp_vel',  lower=0.05, upper=0.25)
    model.add_design_var('Ki_vel',  lower=0.01, upper=0.08)
    model.add_design_var('Kp_att',  lower=2.0,  upper=4.0)
    model.add_design_var('Kd_att',  lower=0.3,  upper=0.8)
    model.add_design_var('Kp_alt',  lower=10.0, upper=20.0)
    model.add_design_var('Kd_alt',  lower=4.0,  upper=10.0)
    
    # 5. Define Objective
    model.add_objective('annoyance_dose')
    
    # 6. Execute MDO Architecture
    prob.setup()

    prob.set_val('R_rotor', 0.19)   # Starting at 0.25 meters
    prob.set_val('L_arm',   0.27)   # Starting at 0.45 meters
    prob.set_val('v_max',   5.69)    # Starting at 6.0 m/s
    prob.set_val('t_climb', 1.89)    #  
    
    # For control gains, you pass them individually matching your design variables
    prob.set_val('Kp_vel',  0.187)
    prob.set_val('Ki_vel',  0.0173)
    prob.set_val('Kp_att',  2.73)
    prob.set_val('Kd_att',  0.35)
    prob.set_val('Kp_alt', 14.17)
    prob.set_val('Kd_alt',  7.59)

    prob.run_driver()
    
    # Display Optimized Array results
    print("\n🎯 SNOPT Multidisciplinary Co-Design Complete!")
    print(f"Optimized Rotor Radius: {prob.get_val('R_rotor')[0]:.4f} m")
    print(f"Optimized Arm Length:   {prob.get_val('L_arm')[0]:.4f} m")
    print(f"Optimized Kp Velocity:  {prob.get_val('Kp_vel')[0]:.4f}")
    print(f"Minimized Annoyance:     {prob.get_val('annoyance_dose')[0]:.4f}")

    print(f"Minimized Annoyance: {prob.get_val('annoyance_dose')[0]:.4f}")
    
    # ----------------------------------------------------
    # POST-OPTIMIZATION VISUALIZATION (OpenMDAO)
    # ----------------------------------------------------
    print("\n📊 Extracting physics telemetry for the optimal OpenMDAO design...")
    
    # Reconstruct the optimal vector X* from OpenMDAO's database components
    X_opt = [
        prob.get_val('R_rotor')[0],
        prob.get_val('L_arm')[0],
        prob.get_val('v_max')[0],
        prob.get_val('t_climb')[0],
        prob.get_val('Kp_vel')[0],
        prob.get_val('Ki_vel')[0],
        prob.get_val('Kp_att')[0],
        prob.get_val('Kd_att')[0],
        prob.get_val('Kp_alt')[0],
        prob.get_val('Kd_alt')[0]
    ]
    
    # Run the wrapper one final time with debug=True using the optimal vector
    _, hist_state, time_steps, v_z_ref, pa = evaluate_drone_codesign(
        X_opt, dt=0.01, t_end=4.0, debug=True
    )
    
    # Call your comprehensive plotting dashboard function
    if hist_state is not None and pa is not None:
        plot_comprehensive_diagnostics(hist_state, time_steps, v_z_ref, pa)

