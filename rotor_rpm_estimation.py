"""
rotor_rpm_estimation.py
========================

Estimate four individual rotor RPM time series for a quadrotor ("+"
configuration: one motor each at front / right / rear / left) from the
translational kinematic data produced by a Dymos trajectory optimization,
in the ABSENCE of attitude (roll/pitch/yaw) information.

----------------------------------------------------------------------------
IMPORTANT - READ BEFORE TRUSTING THE NUMBERS
----------------------------------------------------------------------------
A real quadrotor generates horizontal acceleration by tilting its thrust
vector; recovering that tilt history rigorously requires attitude data
(angles and/or angular rates), which this Dymos trajectory does not provide.
This script therefore uses a set of EXPLICIT, FLAGGED ASSUMPTIONS to turn
(t, x, y, z, vx, vy, vz, ax, ay, az, wx, wy, wz) into a plausible RPM
estimate. It is meant for a first-order noise-source estimate (e.g. feeding
the Zwicker psychoacoustic annoyance pipeline), NOT as a substitute for an
actual flight-dynamics/control simulation. Every assumption is exposed as a
named parameter in `DroneParams` so you can calibrate or replace it.

Modeling chain
--------------
1. COLLECTIVE (average) thrust, shared equally by all 4 rotors, balances:
     - weight + vertical acceleration:           m * (g + az)
     - aerodynamic drag from relative airspeed:  0.5 * rho * Cd_A * |v_rel|^2
   where v_rel = (vx - wx, vy - wy, vz - wz).
   ASSUMPTION: no other vertical aerodynamic forces (e.g. wing lift) are
   modeled; this is a pure multi-rotor thrust budget.
   NOTE on linearity: thrust is linear in az but quadratic in |v_rel|
   (standard aerodynamic drag form), and RPM ~ sqrt(thrust) because of the
   rotor law below -- so average RPM is NOT simply linear in az or |v_rel|,
   even though the underlying thrust terms are individually linear/quadratic
   in those quantities. A `drag_model="linear"` option is provided if you'd
   rather keep a strictly linear-in-|v_rel| thrust budget instead.

2. DIFFERENTIAL thrust (front-rear, right-left) is the part that truly
   needs attitude and is approximated here. ASSUMPTION: ax, ay are treated
   as if they corresponded directly to an angular acceleration about the
   relevant body axis, scaled by the rotor arm length L (this preserves
   units: [m/s^2] / [m] = [rad/s^2]). This angular acceleration is then
   converted into a required differential thrust via the assumed moment of
   inertia and arm length:
       dF_pitch (front - rear)  ~=  Iyy * (ax / L) / L
       dF_roll  (right - left)  ~=  Ixx * (ay / L) / L
   This is a dimensionally consistent stand-in, not a rigorous tilt-dynamics
   result. Treat its absolute scale with caution; its main value is
   producing a differential RPM signal that grows with maneuvering
   intensity. If you obtain attitude data later, replace this block with a
   proper control-allocation calculation from the real pitch/roll moments.

3. ROTOR LAW: thrust and RPM are related by F = kT * RPM^2 (a standard
   simplification for fixed-pitch rotors), with kT calibrated so that, at
   the user-supplied hover RPM, each rotor produces mass*g/4 of thrust:
       kT = (m * g / 4) / rpm_hover^2

4. Per-rotor thrust (assuming front/rear independently control pitch and
   left/right independently control roll, i.e. a "+" layout):
       F_front = F_avg + dF_pitch / 2
       F_rear  = F_avg - dF_pitch / 2
       F_right = F_avg + dF_roll  / 2
       F_left  = F_avg - dF_roll  / 2
   Negative thrust (which a normal, non-reversible propeller cannot produce)
   is clipped to zero.

5. Yaw control (reaction-torque balancing) is NOT modeled since no
   yaw/heading information was requested or available.

Coordinate convention
----------------------
Default assumes a right-handed, Z-up (ENU-like) world frame, with x = North/
forward, y = East/right, z = up, and az measured the same way (so hover
means az = 0, climbing means az > 0). If your Dymos model instead uses a
Z-down (NED) convention, set `z_up=False` in DroneParams and the sign of the
gravity term will be flipped accordingly.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class DroneParams:
    """All assumptions / calibration constants live here -- adjust to your drone."""

    mass: float = 1.5            # [kg]            total drone mass
    arm_length: float = 0.25      # [m]             distance from CG to each rotor
    I_xx: float = 0.012           # [kg*m^2]        roll moment of inertia
    I_yy: float = 0.012           # [kg*m^2]        pitch moment of inertia
    rpm_hover: float = 5000.0     # [RPM]           steady-hover RPM, per rotor (calibration point)
    g: float = 9.80665            # [m/s^2]         gravitational acceleration
    rho: float = 1.225            # [kg/m^3]        air density
    Cd_A: float = 0.02            # [m^2]           lumped drag coefficient * reference area
    drag_model: str = "quadratic"  # "quadratic" (~|v_rel|^2) or "linear" (~|v_rel|)
    z_up: bool = True             # True: z-up (ENU-like). False: z-down (NED-like)
    rpm_min: float = 0.0          # [RPM]           floor applied to all rotor RPMs
    rpm_max: float = None         # [RPM]           optional ceiling (None = no cap)


def estimate_rotor_rpm(t, x, y, z, vx, vy, vz, ax, ay, az, wx, wy, wz,
                        params: DroneParams = None) -> dict:
    """
    Estimate front/rear/left/right rotor RPM time series from Dymos
    trajectory optimization output.

    Parameters
    ----------
    t, x, y, z, vx, vy, vz, ax, ay, az, wx, wy, wz : array-like
        Time, position, velocity, acceleration and wind-velocity time
        histories, as extracted from Dymos (any shape that flattens to 1D,
        e.g. the (n, 1) column vectors p.get_val(...) typically returns).
        Units are assumed SI (s, m, m/s, m/s^2).
    params : DroneParams, optional
        Physical assumptions / calibration constants. Defaults are used if
        not provided -- see the DroneParams docstring/fields, and the module
        docstring above for the full modeling chain and its caveats.

    Returns
    -------
    dict with keys:
        't'                 : time vector [s]
        'rpm_front'          : front rotor RPM [RPM]
        'rpm_rear'           : rear rotor RPM [RPM]
        'rpm_right'          : right rotor RPM [RPM]
        'rpm_left'           : left rotor RPM [RPM]
        'rpm_avg'            : collective (average) RPM [RPM]
        # diagnostics, useful for sanity-checking the model:
        'thrust_total'       : total commanded thrust [N]
        'thrust_avg'         : per-rotor average thrust [N]
        'v_rel'              : relative airspeed magnitude [m/s]
        'dF_pitch'           : front-rear differential thrust [N]
        'dF_roll'            : right-left differential thrust [N]
    """
    if params is None:
        params = DroneParams()

    # --- flatten all inputs to 1D (Dymos timeseries are often (n, 1)) ---
    t, x, y, z, vx, vy, vz, ax, ay, az, wx, wy, wz = [
        np.asarray(v, dtype=float).ravel()
        for v in (t, x, y, z, vx, vy, vz, ax, ay, az, wx, wy, wz)
    ]

    m = params.mass
    L = params.arm_length
    g = params.g if params.z_up else -params.g

    # --- 1. relative airspeed (drone velocity relative to the wind) ---
    vrel_x = vx - wx
    vrel_y = vy - wy
    vrel_z = vz - wz
    v_rel = np.sqrt(vrel_x**2 + vrel_y**2 + vrel_z**2)

    if params.drag_model == "quadratic":
        drag_force = 0.5 * params.rho * params.Cd_A * v_rel**2
    elif params.drag_model == "linear":
        drag_force = params.rho * params.Cd_A * v_rel
    else:
        raise ValueError("drag_model must be 'quadratic' or 'linear'")

    # --- 2. total / average thrust (collective control) ---
    thrust_total = m * (g + az) + drag_force
    thrust_total = np.clip(thrust_total, 0.0, None)  # thrust can't be negative
    thrust_avg = thrust_total / 4.0

    # --- 3. differential thrust from ax, ay (heuristic - see module docstring) ---
    dF_pitch = params.I_yy * (ax / L) / L   # front - rear
    dF_roll = params.I_xx * (ay / L) / L    # right - left

    # --- 4. per-rotor thrust, "+" configuration ---
    F_front = np.clip(thrust_avg + dF_pitch / 2.0, 0.0, None)
    F_rear = np.clip(thrust_avg - dF_pitch / 2.0, 0.0, None)
    F_right = np.clip(thrust_avg + dF_roll / 2.0, 0.0, None)
    F_left = np.clip(thrust_avg - dF_roll / 2.0, 0.0, None)

    # --- 5. thrust -> RPM via calibrated rotor law F = kT * RPM^2 ---
    kT = (m * params.g / 4.0) / params.rpm_hover**2

    def to_rpm(F):
        rpm = np.sqrt(F / kT)
        rpm = np.clip(rpm, params.rpm_min, params.rpm_max)
        return rpm

    rpm_front = to_rpm(F_front)
    rpm_rear = to_rpm(F_rear)
    rpm_right = to_rpm(F_right)
    rpm_left = to_rpm(F_left)
    rpm_avg = to_rpm(thrust_avg)

    return {
        "t": t,
        "rpm_front": rpm_front,
        "rpm_rear": rpm_rear,
        "rpm_right": rpm_right,
        "rpm_left": rpm_left,
        "rpm_avg": rpm_avg,
        "thrust_total": thrust_total,
        "thrust_avg": thrust_avg,
        "v_rel": v_rel,
        "dF_pitch": dF_pitch,
        "dF_roll": dF_roll,
    }


if __name__ == "__main__":
    # --- Minimal usage example with synthetic data, to sanity-check the function ---
    # Replace this block with your actual Dymos extraction, e.g.:
    #
    #   t  = p.get_val('traj.phase0.timeseries.time')
    #   x  = p.get_val('traj.phase0.timeseries.x')
    #   ...
    #   wz = p.get_val('traj.phase0.timeseries.wind_z')
    #   result = estimate_rotor_rpm(t, x, y, z, vx, vy, vz, ax, ay, az, wx, wy, wz)

    n = 500
    t = np.linspace(0, 10, n)

    # Fake a climb-and-accelerate maneuver: increasing forward speed, mild climb,
    # a short pitch-up acceleration pulse, and a steady crosswind.
    vx = 5.0 + 2.0 * t / t[-1]
    vy = np.zeros(n)
    vz = 0.5 * np.sin(0.3 * t)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    az = np.gradient(vz, t)
    x = np.cumsum(vx) * (t[1] - t[0])
    y = np.cumsum(vy) * (t[1] - t[0])
    z = np.cumsum(vz) * (t[1] - t[0])
    wx = 3.0 * np.ones(n)   # 3 m/s steady headwind
    wy = np.zeros(n)
    wz = np.zeros(n)

    result = estimate_rotor_rpm(t, x, y, z, vx, vy, vz, ax, ay, az, wx, wy, wz)

    print("Rotor RPM estimate (synthetic example):")
    print(f"  t range        : {result['t'][0]:.1f} - {result['t'][-1]:.1f} s")
    print(f"  rpm_avg  range : {result['rpm_avg'].min():.0f} - {result['rpm_avg'].max():.0f} RPM")
    print(f"  rpm_front range: {result['rpm_front'].min():.0f} - {result['rpm_front'].max():.0f} RPM")
    print(f"  rpm_rear  range: {result['rpm_rear'].min():.0f} - {result['rpm_rear'].max():.0f} RPM")
    print(f"  rpm_right range: {result['rpm_right'].min():.0f} - {result['rpm_right'].max():.0f} RPM")
    print(f"  rpm_left  range: {result['rpm_left'].min():.0f} - {result['rpm_left'].max():.0f} RPM")