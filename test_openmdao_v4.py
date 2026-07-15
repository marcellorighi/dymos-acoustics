import openmdao.api as om
import numpy as np
from drone_pid_function_v13 import evaluate_drone_codesign
from drone_pid_function_v13 import plot_comprehensive_diagnostics
class DroneCoDesignComponent(om.ExplicitComponent):
    """
    Wraps your updated 12-dimensional simulation into an OpenMDAO component.
    """
    def setup(self):
        # Physical Attributes
        self.add_input('R_rotor', val=0.19)
        self.add_input('L_arm',   val=0.27)
        self.add_input('v_max',   val=5.69)
        self.add_input('t_climb', val=1.89)
        self.add_input('vx_max',  val=2.0)
        self.add_input('vy_max',  val=0.0)
        
        # Controller Gains
        self.add_input('Kp_vel',  val=0.187)
        self.add_input('Ki_vel',  val=0.0173)
        self.add_input('Kp_att',  val=2.73)
        self.add_input('Kd_att',  val=0.35)
        self.add_input('Kp_alt',  val=14.17)
        self.add_input('Kd_alt',  val=7.59)
        
        # FIXED: Output is now the combined optimization metric
        self.add_output('combined_cost', val=99.0)
        
        # Centralized finite-differencing step sizing
        self.declare_partials(of='*', wrt='*', method='fd', step=1e-4)

    def compute(self, inputs, outputs):
        # Reconstruct the true 12-dimensional design vector X matching indices
        X = [
            inputs['R_rotor'][0], inputs['L_arm'][0], inputs['v_max'][0], inputs['t_climb'][0],
            inputs['vx_max'][0],  inputs['vy_max'][0],
            inputs['Kp_vel'][0],  inputs['Ki_vel'][0],  inputs['Kp_att'][0],
            inputs['Kd_att'][0],  inputs['Kp_alt'][0],  inputs['Kd_alt'][0]
        ]
        
        # Execute your updated multi-objective function wrapper
        result = evaluate_drone_codesign(X, dt=0.01, t_end=4.0, debug=False)
        
        # Pass the scalar objective tracking combo out to OpenMDAO
        outputs['combined_cost'] = result['objective']

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
    snopt_options['Major feasibility tolerance'] = 1e-5  
    snopt_options['Major optimality tolerance'] = 1e-4   
    snopt_options['Summary file'] = 'snopt_summary.out'   
    snopt_options['Print file'] = 'snopt_print.out'
    
    # 3. Add Physical Design Variables with Bounds
    model.add_design_var('R_rotor', lower=0.10, upper=0.35)
    model.add_design_var('L_arm',   lower=0.20, upper=0.60)
    model.add_design_var('v_max',   lower=2.0,  upper=10.0)
    model.add_design_var('t_climb', lower=1.0,  upper=5.0)
    model.add_design_var('vx_max',  lower=0.0,  upper=8.0)
    model.add_design_var('vy_max',  lower=0.0,  upper=0.0) # Locked to 0 for now
    
    # 4. Add Control Gain Design Variables with Custom Bounds
    model.add_design_var('Kp_vel',  lower=0.05, upper=0.25)
    model.add_design_var('Ki_vel',  lower=0.01, upper=0.08)
    model.add_design_var('Kp_att',  lower=2.0,  upper=4.0)
    model.add_design_var('Kd_att',  lower=0.3,  upper=0.8)
    model.add_design_var('Kp_alt',  lower=10.0, upper=20.0)
    model.add_design_var('Kd_alt',  lower=4.0,  upper=10.0)
    
    # 5. Define Objective (FIXED: points to unified scalar cost now)
    model.add_objective('combined_cost')
    
    # 6. Initialize Variables & Execute Architecture
    prob.setup()

    # Define all baseline parameters right here before running the driver
    prob.set_val('R_rotor', 0.14)   
    prob.set_val('L_arm',   0.36)   
    prob.set_val('v_max',   5.85)    
    prob.set_val('t_climb', 4.7)    
    prob.set_val('vx_max',  6.0)     # Initial forward speed suggestion (e.g. 3m/s)
    prob.set_val('vy_max',  0.0)     
    
    prob.set_val('Kp_vel',  0.187)
    prob.set_val('Ki_vel',  0.019)
    prob.set_val('Kp_att',  3.2)
    prob.set_val('Kd_att',  0.7)
    prob.set_val('Kp_alt', 17.0)
    prob.set_val('Kd_alt',  6.7)

    prob.run_driver()
    
    # Display Optimized Array results
    print("\n🎯 SNOPT Multidisciplinary Co-Design Complete!")
    print(f"Optimized Rotor Radius: {prob.get_val('R_rotor')[0]:.4f} m")
    print(f"Optimized Arm Length:   {prob.get_val('L_arm')[0]:.4f} m")
    print(f"Optimized Target Vx:    {prob.get_val('vx_max')[0]:.4f} m/s")
    print(f"Optimized Kp Velocity:  {prob.get_val('Kp_vel')[0]:.4f}")
    print(f"Minimized Combined Cost: {prob.get_val('combined_cost')[0]:.4f}")
    
    # ----------------------------------------------------
    # POST-OPTIMIZATION VISUALIZATION (OpenMDAO)
    # ----------------------------------------------------
    print("\n📊 Extracting physics telemetry for the optimal OpenMDAO design...")
    
    # FIXED: Reconstruct the true full 12-element vector X* from OpenMDAO's database components
    X_opt = [
        prob.get_val('R_rotor')[0],
        prob.get_val('L_arm')[0],
        prob.get_val('v_max')[0],
        prob.get_val('t_climb')[0],
        prob.get_val('vx_max')[0],
        prob.get_val('vy_max')[0],
        prob.get_val('Kp_vel')[0],
        prob.get_val('Ki_vel')[0],
        prob.get_val('Kp_att')[0],
        prob.get_val('Kd_att')[0],
        prob.get_val('Kp_alt')[0],
        prob.get_val('Kd_alt')[0]
    ]
    
    # Run the wrapper one final time with debug=True using the optimal vector
    res_dict, hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa = evaluate_drone_codesign(
        X_opt, dt=0.01, t_end=12.0, debug=True
    )
    
    # Call your comprehensive plotting dashboard function (passing all required 3D trajectories)
    if hist_state is not None and pa is not None:
        plot_comprehensive_diagnostics(hist_state, time_steps, v_x_ref, v_y_ref, v_z_ref, pa)

