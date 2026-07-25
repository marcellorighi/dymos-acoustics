"""
OpenMDAO Lightweight Design Demo: Stiffened Plate Optimization
================================================================
A simple OpenMDAO model for a thin rectangular flat plate with four sides
simply supported, subjected to uniform normal pressure.

Plate Component computes:
  1. Static stress (sigma) from bending under uniform pressure
  2. Critical buckling stress (local and overall)
  3. Total weight (plate + stiffeners)

Optimization:
  Design Variables: L, W, t, E, n_S, A_S, I_S
  Objective: Minimize total weight
  Constraints: sigma <= sigma_cr_local, sigma <= sigma_cr_overall

Usage:
    python plate_optimization_openmdao.py
"""

import openmdao.api as om
import numpy as np


class Plate(om.ExplicitComponent):
    """
    Component modeling a thin rectangular flat plate with longitudinal stiffeners.
    Four sides simply supported, uniform normal pressure.
    """

    def setup(self):
        # --- Inputs ---
        # Geometry
        self.add_input('L', val=1.0, units='m')           # Plate length
        self.add_input('W', val=0.5, units='m')           # Plate width
        self.add_input('t', val=0.01, units='m')         # Plate thickness
        # Material
        self.add_input('E', val=70e9, units='Pa')         # Young's modulus (default: Al)
        # Stiffener configuration
        self.add_input('n_S', val=3.0)                     # Number of longitudinal stiffeners
        self.add_input('A_S', val=1e-4, units='m**2')     # Stiffener cross-sectional area
        self.add_input('I_S', val=1e-8, units='m**4')     # Stiffener moment of inertia

        # --- Constants (could be promoted to inputs if needed) ---
        self.rho = 2700.0        # Material density [kg/m^3] (aluminum default)
        self.p = 1e5             # Uniform pressure [Pa] (default: 1 bar)
        self.nu = 0.33           # Poisson's ratio (aluminum)

        # --- Outputs ---
        self.add_output('sigma', val=0.0, units='Pa')            # Static bending stress
        self.add_output('sigma_cr_local', val=0.0, units='Pa')   # Local buckling stress
        self.add_output('sigma_cr_overall', val=0.0, units='Pa') # Overall buckling stress
        self.add_output('weight', val=0.0, units='kg')          # Total weight

    def setup_partials(self):
        # Finite difference for simplicity in this demo
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        L = inputs['L']
        W = inputs['W']
        t = inputs['t']
        E = inputs['E']
        n_S = inputs['n_S']
        A_S = inputs['A_S']
        I_S = inputs['I_S']

        rho = self.rho
        p = self.p
        nu = self.nu

        # --- Panel width between stiffeners ---
        # n_S stiffeners divide width into (n_S - 1) panels
        n_S_int = max(2.0, n_S)  # Ensure at least 2 stiffeners for panels
        b = W / (n_S_int - 1.0)  # Width of each sub-panel

        # ============================================================
        # 1. STATIC STRESS (bending under uniform pressure)
        # ============================================================
        # For a simply supported rectangular plate under uniform pressure p:
        # Max bending stress at center. Using Timoshenko/Roark approach:
        # sigma = 6 * k_moment * p * b^2 / t^2
        # where k_moment depends on aspect ratio L/W.
        aspect = L / W
        if aspect < 1.0:
            aspect = 1.0 / aspect
        # Interpolated moment coefficient: square plate -> 0.0479, long strip -> 0.03125
        k_moment = 0.03125 + (0.0479 - 0.03125) * np.exp(-0.8 * (aspect - 1.0))
        sigma = 6.0 * k_moment * p * (b ** 2) / (t ** 2)
        outputs['sigma'] = sigma

        # ============================================================
        # 2. LOCAL BUCKLING STRESS (skin between stiffeners)
        # ============================================================
        # For a simply supported rectangular plate:
        # sigma_cr = k * (pi^2 * E) / (12 * (1 - nu^2)) * (t/b)^2
        # k = 4.0 for simply supported, compression-like (conservative)
        k_local = 4.0
        sigma_cr_local = k_local * (np.pi ** 2) * E / (12.0 * (1.0 - nu ** 2)) * (t / b) ** 2
        outputs['sigma_cr_local'] = sigma_cr_local

        # ============================================================
        # 3. OVERALL BUCKLING STRESS (stiffened panel as wide column)
        # ============================================================
        # Effective width concept: plate contributes over width of ~ 30t each side
        b_eff = min(b, 30.0 * t)

        # Effective area per stiffener bay
        A_eff_per_bay = b_eff * t + A_S

        # Effective moment of inertia (parallel axis theorem for stiffener)
        e = t / 2.0 + np.sqrt(A_S) / 2.0  # centroid offset estimate
        I_eff_per_bay = I_S + A_S * e ** 2 + b_eff * t ** 3 / 12.0

        # Overall buckling as wide column (pinned ends, length L):
        sigma_cr_overall = (np.pi ** 2) * E * I_eff_per_bay / (A_eff_per_bay * L ** 2)
        outputs['sigma_cr_overall'] = sigma_cr_overall

        # ============================================================
        # 4. TOTAL WEIGHT
        # ============================================================
        weight_plate = rho * L * W * t
        weight_stiffeners = rho * n_S * A_S * L
        outputs['weight'] = weight_plate + weight_stiffeners


class PlateOptGroup(om.Group):
    """
    OpenMDAO Group containing the Plate component and constraint margins.
    """

    def setup(self):
        # Add the plate component with promoted variables
        self.add_subsystem('plate', Plate(), promotes=['*'])

        # Add constraint margins as explicit outputs
        # margin_local = sigma_cr_local - sigma  (must be >= 0)
        # margin_overall = sigma_cr_overall - sigma  (must be >= 0)
        self.add_subsystem('constraints', om.ExecComp([
            'margin_local = sigma_cr_local - sigma',
            'margin_overall = sigma_cr_overall - sigma',
        ],
            sigma={'units': 'Pa'},
            sigma_cr_local={'units': 'Pa'},
            sigma_cr_overall={'units': 'Pa'},
            margin_local={'units': 'Pa'},
            margin_overall={'units': 'Pa'},
        ), promotes=['*'])


# =============================================================================
# MAIN: Build Problem, Setup, Run
# =============================================================================

if __name__ == '__main__':

    # --- Build the optimization problem ---
    prob = om.Problem()
    prob.model = PlateOptGroup()

    # --- Driver: ScipyOptimizeDriver with SLSQP ---
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 300
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = True

    # --- Design Variables ---
    # NOTE: These are called on prob.model, not on prob directly
    prob.model.add_design_var('L', lower=2.5, upper=3.0)
    prob.model.add_design_var('W', lower=0.3, upper=1.5)
    prob.model.add_design_var('t', lower=0.01, upper=0.05)
    prob.model.add_design_var('E', lower=10e9, upper=210e9)
    prob.model.add_design_var('n_S', lower=2.0, upper=10.0)
    prob.model.add_design_var('A_S', lower=1e-5, upper=5e-4)
    prob.model.add_design_var('I_S', lower=1e-9, upper=1e-6)

    # --- Objective ---
    prob.model.add_objective('weight')

    # --- Constraints ---
    prob.model.add_constraint('margin_local', lower=0.0)
    prob.model.add_constraint('margin_overall', lower=0.0)

    # --- Setup the problem ---
    prob.setup()

    # --- Set initial values ---
    prob.set_val('L', 1.0)
    prob.set_val('W', 0.5)
    prob.set_val('t', 0.005)
    prob.set_val('E', 70e9)      # Aluminum
    prob.set_val('n_S', 3.0)
    prob.set_val('A_S', 1e-4)
    prob.set_val('I_S', 1e-8)

    # --- Run model once to check initial state ---
    prob.run_model()

    print("=" * 60)
    print("INITIAL DESIGN")
    print("=" * 60)
    print(f"L = {prob.get_val('L')[0]:.4f} m")
    print(f"W = {prob.get_val('W')[0]:.4f} m")
    print(f"t = {prob.get_val('t')[0]:.6f} m")
    print(f"E = {prob.get_val('E')[0]/1e9:.1f} GPa")
    print(f"n_S = {prob.get_val('n_S')[0]:.1f}")
    print(f"A_S = {prob.get_val('A_S')[0]:.2e} m^2")
    print(f"I_S = {prob.get_val('I_S')[0]:.2e} m^4")
    print("-" * 60)
    print(f"Static stress:       {prob.get_val('sigma')[0]/1e6:.3f} MPa")
    print(f"Critical local:      {prob.get_val('sigma_cr_local')[0]/1e6:.3f} MPa")
    print(f"Critical overall:    {prob.get_val('sigma_cr_overall')[0]/1e6:.3f} MPa")
    print(f"Weight:              {prob.get_val('weight')[0]:.4f} kg")
    print(f"Margin local:        {prob.get_val('margin_local')[0]/1e6:.3f} MPa")
    print(f"Margin overall:      {prob.get_val('margin_overall')[0]/1e6:.3f} MPa")
    print()

    # --- Run optimization ---
    print("=" * 60)
    print("RUNNING OPTIMIZATION")
    print("=" * 60)
    prob.run_driver()

    print()
    print("=" * 60)
    print("OPTIMIZED DESIGN")
    print("=" * 60)
    print(f"L = {prob.get_val('L')[0]:.4f} m")
    print(f"W = {prob.get_val('W')[0]:.4f} m")
    print(f"t = {prob.get_val('t')[0]:.6f} m")
    print(f"E = {prob.get_val('E')[0]/1e9:.1f} GPa")
    print(f"n_S = {prob.get_val('n_S')[0]:.1f}")
    print(f"A_S = {prob.get_val('A_S')[0]:.2e} m^2")
    print(f"I_S = {prob.get_val('I_S')[0]:.2e} m^4")
    print("-" * 60)
    print(f"Static stress:       {prob.get_val('sigma')[0]/1e6:.3f} MPa")
    print(f"Critical local:      {prob.get_val('sigma_cr_local')[0]/1e6:.3f} MPa")
    print(f"Critical overall:    {prob.get_val('sigma_cr_overall')[0]/1e6:.3f} MPa")
    print(f"Weight:              {prob.get_val('weight')[0]:.4f} kg")
    print(f"Margin local:        {prob.get_val('margin_local')[0]/1e6:.3f} MPa")
    print(f"Margin overall:      {prob.get_val('margin_overall')[0]/1e6:.3f} MPa")
    print()

    # Note: n_S should be rounded to integer for actual manufacturing
    n_S_opt = prob.get_val('n_S')[0]
    print(f"Note: n_S = {n_S_opt:.2f} should be rounded to integer: {int(round(n_S_opt))}")

