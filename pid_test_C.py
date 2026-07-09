import numpy as np

# Quick unit test -- run this standalone
m=1.5; g=9.80665
phi=0; theta=0; psi=0; Fz=m*g; Fx=0; Fy=0
cp,sp = np.cos(phi),np.sin(phi)
ct,st = np.cos(theta),np.sin(theta)
cy,sy = np.cos(psi),np.sin(psi)
az = (1/m)*((-st)*Fx + (ct*sp)*Fy + (ct*cp)*Fz) - g
print(f"az at hover = {az:.6f}  (must be exactly 0.0)")

