import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


m   = 1.5
g   = 9.80665
Iyy = 0.012


# ---------------------------------------------------------------
# TEST B: attitude only (theta), no position, no altitude
# Just: theta_ddot = My/Iyy, My = Kp*(theta_ref - theta) + Kd*(0 - q)
# Expected: theta tracks theta_ref=0.2 rad from theta0=0
# ---------------------------------------------------------------
def att_only(t, state):
    theta, q = state
    theta_ref = 0.2   # 11.5 degrees
    My = 0.13*(theta_ref - theta) + 0.07*(0 - q)   # physics-derived gains
    q_dot = My / Iyy
    theta_dot = q   # simplified: no phi, no psi coupling
    return [theta_dot, q_dot]

sol = solve_ivp(att_only, (0,5), [0.0, 0.0], max_step=0.005)
plt.figure(); plt.plot(sol.t, np.degrees(sol.y[0])); plt.axhline(11.5, ls='--')
plt.title('TEST B: attitude only'); plt.ylabel('theta [deg]'); plt.xlabel('t [s]')
plt.show()
print(f"TEST B: theta final={np.degrees(sol.y[0,-1]):.2f} deg "
      f"(expect ~11.5), q final={sol.y[1,-1]:.4f} (expect ~0)")
