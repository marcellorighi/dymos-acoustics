import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

m   = 1.5
g   = 9.80665
Iyy = 0.012

# ---------------------------------------------------------------
# TEST A: altitude only, no attitude, no position
# Just: z_ddot = Fz/m - g, Fz = m*g + Kp*(z_ref - z) + Kd*(0 - vz)
# Expected: z tracks z_ref=20 from z0=10 with no oscillation
# ---------------------------------------------------------------
def alt_only(t, state):
    z, vz = state
    z_ref = 20.0
    Fz = m*g + 8.0*(z_ref - z) + 3.0*(0 - vz)   # Kp=8, Kd=3
    az = Fz/m - g
    return [vz, az]

sol = solve_ivp(alt_only, (0,10), [10.0, 0.0], max_step=0.01)
plt.figure(); plt.plot(sol.t, sol.y[0]); plt.axhline(20, ls='--')
plt.title('TEST A: altitude only'); plt.ylabel('z [m]'); plt.xlabel('t [s]')
plt.show()
print(f"TEST A: z final={sol.y[0,-1]:.2f} (expect ~20), "
      f"vz final={sol.y[1,-1]:.4f} (expect ~0)")
