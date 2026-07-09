"""
simulate_pid.py
================

High-resolution forward simulation of the 6-DOF drone with a cascaded PID
controller, replacing the user-supplied control_func in
simulate_attitude_response(). Intended for:

  1. Acoustic post-processing: the trajectory output feeds directly into
     estimate_rotor_rpm() -> estimate_received_spl_fine() ->
     compute_zwicker_indicators_windowed()

  2. Control gain optimisation: the PID gains in CascadedPIDParams are the
     parameters to optimise (e.g. minimise PA annoyance over an ensemble of
     Dryden realisations, as discussed in dryden_timeseries.py)

----------------------------------------------------------------------------
CASCADED PID STRUCTURE
----------------------------------------------------------------------------

                   z_ref ──► [Altitude PID] ──────────────────► Fz
                                  |
              x_ref, y_ref ──► [Position PID] ──► phi_cmd, theta_cmd
                                                        |
                                        phi, theta ──► [Attitude PID] ──► Mx, My
                                                        |
                                        psi_ref ──► [Yaw PID] ──────────► Mz

Outer loop (position/altitude) runs at full integration rate but its
output is low-pass filtered to avoid exciting high-frequency attitude
dynamics with noisy position errors -- this is standard in real flight
controllers.

Inner loop (attitude) runs at the same rate but responds faster
(higher gains, shorter time constants).

----------------------------------------------------------------------------
REFERENCE TRAJECTORY
----------------------------------------------------------------------------
The reference (x_ref, y_ref, z_ref, psi_ref)(t) comes from the slow Dymos
optimisation output. It is passed in as a set of time arrays and interpolated
at each integration step. The PID controller then tracks this reference
while rejecting Dryden disturbances.

The reference velocity (vx_ref, vy_ref) is also used in the position PID
as a feed-forward term, which significantly improves tracking performance
and reduces overshoot compared to pure error feedback.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# PID gain dataclass
# ---------------------------------------------------------------------------
@dataclass
class PIDGains:
    """Proportional, integral, derivative gains for a single PID channel."""
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    # Anti-windup: clamp the integral term to [-windup, +windup]
    windup: float = 10.0
    # Output saturation: clamp the PID output to [-sat, +sat]
    # Set to None to disable
    sat: Optional[float] = None


@dataclass
class CascadedPIDParams:
    """
    All PID gains and physical parameters for the cascaded controller.
    Tune these to balance tracking performance vs control effort
    (and by extension, acoustic annoyance).

    Design notes
    ------------
    - Altitude (Fz): slow outer loop, Kp ~ m*g / dz_max where dz_max is the
      maximum acceptable altitude error. For m=1.5 kg, dz_max=2 m -> Kp~7.
    - Position (phi_cmd, theta_cmd): output is a desired angle [rad], so
      Kp has units [rad / m]. Kp ~ 0.1-0.3 rad/m is typical for slow drones.
    - Attitude (Mx, My): output is a moment [N*m], Kp ~ Ixx/tau^2 where
      tau is desired attitude settling time. For Ixx=0.012, tau=0.5s -> Kp~0.05.
    - Yaw (Mz): similar to attitude but typically slower.

    Acoustic implications
    ---------------------
    - High Kp/Kd in attitude PIDs -> large Mx, My -> large differential RPM
      -> higher roughness and sharpness (see drone_ode_6dof.py proxies)
    - High Ki in altitude PID -> integral windup during gusts -> RPM
      fluctuations -> higher fluctuation strength
    - These are the gains the acoustic surrogate model should optimise.
    """
    # Physical parameters (must match DroneODE6DOF)
    mass: float = 1.5         # [kg]
    Ixx:  float = 0.012       # [kg*m^2]
    Iyy:  float = 0.012       # [kg*m^2]
    Izz:  float = 0.020       # [kg*m^2]
    g:    float = 9.80665     # [m/s^2]
    arm_length: float = 0.25  # [m]

    # Attitude limits (prevents unrealistic commands from position PID)
    max_tilt_rad: float = np.radians(30)  # [rad] max phi or theta command

    # Low-pass filter time constant for position PID output [s]
    # Filters phi_cmd, theta_cmd to avoid exciting fast attitude modes
    tau_cmd_filter: float = 0.2   # [s]

    # --- PID gains ---
    # Altitude: z_error -> Fz [N]
    alt:  PIDGains = field(default_factory=lambda: PIDGains(
        Kp=1.17, Ki=0.008, Kd=0.75, windup=5.0,
        sat=None))   # saturation applied via Fz bounds below

    # Position x: x_error -> theta_cmd [rad] (pitch forward)
    pos_x: PIDGains = field(default_factory=lambda: PIDGains(
        Kp=0.15, Ki=0.01, Kd=0.3, windup=0.3,
        sat=np.radians(25)))

    # Position y: y_error -> phi_cmd [rad] (roll right)
    pos_y: PIDGains = field(default_factory=lambda: PIDGains(
        Kp=0.15, Ki=0.01, Kd=0.3, windup=0.3,
        sat=np.radians(25)))

    # Attitude phi: phi_error -> Mx [N*m]
    att_phi: PIDGains = field(default_factory=lambda: PIDGains(
        Kp=0.15, Ki=0.0, Kd=0.7, windup=0.2,
        sat=0.75))

    # Attitude theta: theta_error -> My [N*m]
    att_theta: PIDGains = field(default_factory=lambda: PIDGains(
        Kp=0.15, Ki=0.00, Kd=1.7, windup=0.2,
        sat=0.95))

    # Yaw psi: psi_error -> Mz [N*m]
    yaw: PIDGains = field(default_factory=lambda: PIDGains(
        Kp=0.3, Ki=0.01, Kd=0.1, windup=0.2,
        sat=0.1))

    # Force bounds [N]
    Fz_min: float = -0.5 * 1.5 * 9.80665   # minimum thrust (motors always on)
    Fz_max: float = 2.5 * 1.5 * 9.80665   # maximum thrust

    # Lateral force bounds (small -- quadrotors generate little lateral force)
    Fxy_max: float = 4.0   # [N]


# ---------------------------------------------------------------------------
# PID state (integrator + previous error for derivative)
# ---------------------------------------------------------------------------
class _PIDState:
    """Mutable state for one PID channel."""
    def __init__(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0   # for output filter

    def update(self, error: float, dt: float, gains: PIDGains) -> float:
        # Derivative on error (not on measurement, to avoid derivative kick)
        derivative = (error - self.prev_error) / max(dt, 1e-6)
        # Integral with anti-windup clamping
        self.integral = np.clip(
            self.integral + error * dt,
            -gains.windup, gains.windup)
        output = (gains.Kp * error
                  + gains.Ki * self.integral
                  + gains.Kd * derivative)
        if gains.sat is not None:
            output = np.clip(output, -gains.sat, gains.sat)
        self.prev_error = error
        return float(output)


# ---------------------------------------------------------------------------
# Reference trajectory interpolator
# ---------------------------------------------------------------------------
class ReferenceTrajectory:
    """
    Wraps the Dymos optimisation output as a callable reference for the PID
    controller. Interpolates (x_ref, y_ref, z_ref, psi_ref, vx_ref, vy_ref)
    at any time t using np.interp (linear, clamps at boundaries).

    Parameters
    ----------
    t_ref : array [s]
        Time vector from Dymos timeseries (coarse GL nodes).
    x_ref, y_ref, z_ref : arrays [m]
        Reference position from Dymos.
    psi_ref : array [rad], optional
        Reference yaw. If None, yaw is held constant at psi0.
    vx_ref, vy_ref : arrays [m/s], optional
        Reference velocities for feed-forward in position PID.
        If None, estimated by finite-differencing (x_ref, y_ref).
    """
    def __init__(self, t_ref, x_ref, y_ref, z_ref,
                 psi_ref=None, vx_ref=None, vy_ref=None):
        self.t   = np.asarray(t_ref).ravel()
        self.x   = np.asarray(x_ref).ravel()
        self.y   = np.asarray(y_ref).ravel()
        self.z   = np.asarray(z_ref).ravel()
        self.psi = (np.asarray(psi_ref).ravel() if psi_ref is not None
                    else np.zeros_like(self.t))
        if vx_ref is not None:
            self.vx = np.asarray(vx_ref).ravel()
            self.vy = np.asarray(vy_ref).ravel()
        else:
            # Estimate from finite differences of the reference position
            self.vx = np.gradient(self.x, self.t)
            self.vy = np.gradient(self.y, self.t)

    def at(self, t: float) -> dict:
        """Interpolate all reference quantities at time t."""
        return {
            'x':   float(np.interp(t, self.t, self.x)),
            'y':   float(np.interp(t, self.t, self.y)),
            'z':   float(np.interp(t, self.t, self.z)),
            'psi': float(np.interp(t, self.t, self.psi)),
            'vx':  float(np.interp(t, self.t, self.vx)),
            'vy':  float(np.interp(t, self.t, self.vy)),
        }


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------
def simulate_with_pid(t_span: tuple,
                       dt: float,
                       state0: list,
                       reference: ReferenceTrajectory,
                       params: CascadedPIDParams = None,
                       dryden_ts: dict = None,
                       drone_params: dict = None) -> dict:
    """
    High-resolution forward simulation of the 6-DOF drone with a cascaded
    PID controller. Replaces simulate_attitude_response() when no external
    control_func is available.

    Parameters
    ----------
    t_span : tuple (t0, t1) [s]
    dt : float
        Integration time step [s]. 1/500 recommended for 5-20 Hz dynamics.
    state0 : list of 12 floats
        [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    reference : ReferenceTrajectory
        Reference trajectory from Dymos optimisation output. Use
        ReferenceTrajectory(t, x, y, z) to construct from Dymos arrays.
    params : CascadedPIDParams, optional
        PID gains and physical parameters. Defaults used if None.
    dryden_ts : dict, optional
        Pre-generated Dryden time series from generate_dryden_time_series().
        If None, no wind disturbance is applied (useful for gain tuning
        without turbulence first, then re-run with turbulence).
    drone_params : dict, optional
        Physical parameter overrides (same keys as CascadedPIDParams fields).

    Returns
    -------
    dict with keys:
        't'                   : time vector [s]
        'x','y','z'           : position [m]
        'vx','vy','vz'        : velocity [m/s]
        'phi','theta','psi'   : Euler angles [rad]
        'p','q','r'           : angular rates [rad/s]
        'ax','ay','az'        : world-frame accelerations [m/s^2]
        'Fx','Fy','Fz'        : body-frame forces [N]
        'Mx','My','Mz'        : body-frame moments [N*m]
        'wind_x','wind_y','wind_z' : translational wind [m/s]
        'x_ref','y_ref','z_ref'    : reference trajectory at each t
        'phi_cmd','theta_cmd'      : attitude commands [rad]
        'pid_log'             : dict of PID internal signals for diagnostics

    Feed 'ax','ay','az','wind_x/y/z' into estimate_rotor_rpm() as before.
    """
    if params is None:
        params = CascadedPIDParams()

    m   = params.mass
    g   = params.g
    Ixx = params.Ixx
    Iyy = params.Iyy
    Izz = params.Izz

    # Initialise PID states
    pid = {
        'alt':       _PIDState(),
        'pos_x':     _PIDState(),
        'pos_y':     _PIDState(),
        'att_phi':   _PIDState(),
        'att_theta': _PIDState(),
        'yaw':       _PIDState(),
    }

    # Low-pass filter state for attitude commands
    phi_cmd_filt   = 0.0
    theta_cmd_filt = 0.0

    # Storage for diagnostics
    log = {k: [] for k in ['Fz','Fx','Fy','Mx','My','Mz',
                            'phi_cmd','theta_cmd',
                            'alt_err','pos_x_err','pos_y_err',
                            'att_phi_err','att_theta_err','yaw_err']}

    def _wind_at(t, x_, y_, z_):
        """Get wind at current time/position from pre-generated series or zero."""
        if dryden_ts is not None:
            t_ref = dryden_ts['t']
            wx = float(np.interp(t, t_ref, dryden_ts['wu']))
            wy = float(np.interp(t, t_ref, dryden_ts['wv']))
            wz = float(np.interp(t, t_ref, dryden_ts['ww']))
            pt = float(np.interp(t, t_ref, dryden_ts['p_turb']))
            qt = float(np.interp(t, t_ref, dryden_ts['q_turb']))
            mwx = float(np.interp(t, t_ref, dryden_ts['mean_wx']))
        else:
            wx = wy = wz = pt = qt = 0.0
            v_ref = getattr(params, 'v_ref', 5.0)
            z_ref = getattr(params, 'z_ref', 20.0)
            mwx = v_ref * (max(z_, 0.1) / z_ref) ** 0.15
        return wx, wy, wz, pt, qt, mwx

    def _control(t, state):
        """Cascaded PID controller: state -> (Fx, Fy, Fz, Mx, My, Mz)."""
        nonlocal phi_cmd_filt, theta_cmd_filt

        x_, y_, z_ = state[0], state[1], state[2]
        vx_, vy_, vz_ = state[3], state[4], state[5]
        phi_, theta_, psi_ = state[6], state[7], state[8]
        p_, q_, r_ = state[9], state[10], state[11]

        ref = reference.at(t)

        # ---- Altitude PID: z_error -> Fz --------------------------------
        z_err = ref['z'] - z_
        # temp 
        # Fz_raw = m * g + pid['alt'].update(z_err, dt, params.alt)

        cos_tilt = np.cos(phi_) * np.cos(theta_)
        cos_tilt = max(cos_tilt, 0.5)   # prevent division by zero at extreme tilt
        Fz_raw = (m * g + pid['alt'].update(z_err, dt, params.alt)) / cos_tilt

        Fz = float(np.clip(Fz_raw, params.Fz_min, params.Fz_max))

        # ---- Position PID: position error -> desired pitch/roll ----------
        # Errors in world frame, with velocity feed-forward
        x_err = (ref['x'] - x_) + 0.3 * (ref['vx'] - vx_)
        y_err = (ref['y'] - y_) + 0.3 * (ref['vy'] - vy_)

        # Output is desired angle: positive x_err -> pitch forward (theta > 0)
        #                          positive y_err -> roll right  (phi < 0)
        theta_cmd_raw =  pid['pos_x'].update(x_err, dt, params.pos_x)
        phi_cmd_raw   = -pid['pos_y'].update(y_err, dt, params.pos_y)

        # Clamp to max tilt
        theta_cmd_raw = float(np.clip(theta_cmd_raw,
                                       -params.max_tilt_rad, params.max_tilt_rad))
        phi_cmd_raw   = float(np.clip(phi_cmd_raw,
                                       -params.max_tilt_rad, params.max_tilt_rad))

        # Low-pass filter the attitude commands (avoids exciting fast modes)
        alpha = dt / (params.tau_cmd_filter + dt)
        phi_cmd_filt   = (1 - alpha) * phi_cmd_filt   + alpha * phi_cmd_raw
        theta_cmd_filt = (1 - alpha) * theta_cmd_filt + alpha * theta_cmd_raw

        # ---- Attitude PID: angle error -> Mx, My -------------------------
        phi_err   = phi_cmd_filt   - phi_
        theta_err = theta_cmd_filt - theta_

        Mx = pid['att_phi'].update(phi_err,   dt, params.att_phi)
        My = pid['att_theta'].update(theta_err, dt, params.att_theta)

        # ---- Yaw PID: psi_error -> Mz ------------------------------------
        # Wrap psi error to [-pi, pi]
        psi_err = ref['psi'] - psi_
        psi_err = float(np.arctan2(np.sin(psi_err), np.cos(psi_err)))
        Mz = pid['yaw'].update(psi_err, dt, params.yaw)

        # ---- Lateral forces (small for quadrotor) ------------------------
        # Generate small Fx, Fy proportional to velocity error to help
        # damp translational oscillations; a real quadrotor generates these
        # purely through tilt, so these are small feed-forward terms only.
        Fx = float(np.clip(0.1 * (ref['vx'] - vx_), -params.Fxy_max, params.Fxy_max))
        Fy = float(np.clip(0.1 * (ref['vy'] - vy_), -params.Fxy_max, params.Fxy_max))

        # Log internal signals
        for k, v in [('Fz', Fz), ('Fx', Fx), ('Fy', Fy),
                      ('Mx', Mx), ('My', My), ('Mz', Mz),
                      ('phi_cmd', phi_cmd_filt), ('theta_cmd', theta_cmd_filt),
                      ('alt_err', z_err), ('pos_x_err', x_err),
                      ('pos_y_err', y_err), ('att_phi_err', phi_err),
                      ('att_theta_err', theta_err), ('yaw_err', psi_err)]:
            log[k].append(v)

        return Fx, Fy, Fz, Mx, My, Mz

    def _deriv(t, state):
        x_, y_, z_ = state[0], state[1], state[2]
        vx_, vy_, vz_ = state[3], state[4], state[5]
        phi_, theta_, psi_ = state[6], state[7], state[8]
        p_, q_, r_ = state[9], state[10], state[11]

        Fx_, Fy_, Fz_, Mx_, My_, Mz_ = _control(t, state)
        wx, wy, wz, p_turb, q_turb, mean_wx = _wind_at(t, x_, y_, z_)

        # Translational kinematics
        x_dot  = vx_ + mean_wx + wx
        y_dot  = vy_ + wy
        z_dot_ = vz_ + wz

        # World-frame accelerations
        cp, sp = np.cos(phi_), np.sin(phi_)
        ct, st = np.cos(theta_), np.sin(theta_)
        cy, sy = np.cos(psi_), np.sin(psi_)
        ax_ = (1/m)*((cy*ct)*Fx_+(cy*st*sp-sy*cp)*Fy_+(cy*st*cp+sy*sp)*Fz_)
        ay_ = (1/m)*((sy*ct)*Fx_+(sy*st*sp+cy*cp)*Fy_+(sy*st*cp-cy*sp)*Fz_)
        az_ = (1/m)*((-st)*Fx_  +(ct*sp)*Fy_         +(ct*cp)*Fz_) - g

        # Attitude kinematics
        cos_th = np.cos(theta_)
        cos_th_s = cos_th if abs(cos_th) > 1e-3 \
                   else np.sign(cos_th + 1e-12) * 1e-3
        tan_th = np.sin(theta_) / cos_th_s
        phi_dot_   = p_ + (q_*sp + r_*cp) * tan_th
        theta_dot_ = q_*cp - r_*sp
        psi_dot_   = (q_*sp + r_*cp) / cos_th_s

        # Attitude dynamics (moments enter directly)
        p_dot_ = (Iyy-Izz)/Ixx * q_*r_ + Mx_/Ixx + p_turb
        q_dot_ = (Izz-Ixx)/Iyy * p_*r_ + My_/Iyy + q_turb
        r_dot_ = (Ixx-Iyy)/Izz * p_*q_ + Mz_/Izz

        return [x_dot, y_dot, z_dot_,
                ax_, ay_, az_,
                phi_dot_, theta_dot_, psi_dot_,
                p_dot_, q_dot_, r_dot_]

    # --- Integrate --------------------------------------------------------
    sol = solve_ivp(_deriv, t_span, state0,
                    method='RK45', max_step=dt,
                    rtol=1e-6, atol=1e-8)

    t_sol = sol.t
    n_steps = len(t_sol)
    names = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
    result = {'t': t_sol}
    for i, name in enumerate(names):
        result[name] = sol.y[i]

    # Recompute controls at all solution time points
    ctrls = np.array([_control(ti, sol.y[:, i]) for i, ti in enumerate(t_sol)])
    result['Fx'] = ctrls[:, 0]; result['Fy'] = ctrls[:, 1]
    result['Fz'] = ctrls[:, 2]; result['Mx'] = ctrls[:, 3]
    result['My'] = ctrls[:, 4]; result['Mz'] = ctrls[:, 5]

    # World-frame accelerations
    cp = np.cos(sol.y[6]); sp = np.sin(sol.y[6])
    ct = np.cos(sol.y[7]); st = np.sin(sol.y[7])
    cy = np.cos(sol.y[8]); sy = np.sin(sol.y[8])
    Fx_ = ctrls[:,0]; Fy_ = ctrls[:,1]; Fz_ = ctrls[:,2]
    result['ax'] = (1/m)*((cy*ct)*Fx_+(cy*st*sp-sy*cp)*Fy_+(cy*st*cp+sy*sp)*Fz_)
    result['ay'] = (1/m)*((sy*ct)*Fx_+(sy*st*sp+cy*cp)*Fy_+(sy*st*cp-cy*sp)*Fz_)
    result['az'] = (1/m)*((-st)*Fx_+(ct*sp)*Fy_+(ct*cp)*Fz_) - g

    # Wind at solution points
    winds = [_wind_at(ti, sol.y[0,i], sol.y[1,i], sol.y[2,i])
             for i, ti in enumerate(t_sol)]
    result['wind_x']  = np.array([w[0] for w in winds])
    result['wind_y']  = np.array([w[1] for w in winds])
    result['wind_z']  = np.array([w[2] for w in winds])

    # Reference at solution points
    refs = [reference.at(ti) for ti in t_sol]
    result['x_ref']   = np.array([r['x']   for r in refs])
    result['y_ref']   = np.array([r['y']   for r in refs])
    result['z_ref']   = np.array([r['z']   for r in refs])
    result['phi_cmd'] = np.array(log['phi_cmd'][:n_steps])
    result['theta_cmd'] = np.array(log['theta_cmd'][:n_steps])

    # PID diagnostic log (truncate to solution length)
    result['pid_log'] = {k: np.array(v[:n_steps]) for k, v in log.items()}

    return result


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dryden_timeseries import generate_dryden_time_series, DrydenParams

    # --- Synthetic reference trajectory (replace with Dymos output) ---
    T = 45.0
    t_ref = np.linspace(0, T, 100)
    # Simple straight-line climb and forward flight
    x_ref   = 5.0 * t_ref               # 5 m/s forward
    y_ref   = np.zeros_like(t_ref)
    z_ref   = 10.0 + 2.0 * t_ref        # climb from 10 to 30 m
    psi_ref = np.zeros_like(t_ref)      # heading north

    reference = ReferenceTrajectory(t_ref, x_ref, y_ref, z_ref, psi_ref)

    # --- Initial state: hover at (0,0,10), level ---
    state0 = [0, 0, 10,   # x, y, z
              0, 0, 0,    # vx, vy, vz
              0, 0, 0,    # phi, theta, psi
              0, 0, 0]    # p, q, r

    # --- PID params (default gains) ---
    params = CascadedPIDParams()

    T_max = T          # seconds, matches your duration_bounds upper limit
    dt_dryden = 0.05       # [s] 20 Hz is more than enough for Dymos GL nodes
                # (your ~54 nodes over ~30s gives ~1.8 Hz resolution)
                # no need to pre-generate at 500 Hz -- that's only
                # needed for the acoustic post-processing step

    t_dryden = t_ref #np.arange(0, T_max + dt_dryden, dt_dryden)

    dryden_ts = generate_dryden_time_series(
        t_dryden,
        params=DrydenParams(
            V=5.0,          # mean airspeed [m/s] -- your best estimate
            sigma_u=1.5,    # longitudinal intensity [m/s]
            sigma_v=1.5,    # lateral
            sigma_w=0.75,   # vertical
            L_u=200.0,      # length scales [m]
            L_v=200.0,
            L_w=50.0,
            arm_length=0.25,
            v_ref=5.0,
            z_ref=20.0,
        ),
        seed=42,            # fix for reproducibility; vary for ensemble runs
        altitude=50.0,      # representative flight altitude
    )


    # --- Run without turbulence first (to check gain tuning) ---
    print("Running simulation with turbulence...")
    res = simulate_with_pid(
        t_span=(0, T), dt=2.e-2, #dt=1/500,
        state0=state0,
        reference=reference,
        params=params,
        dryden_ts=None, #dryden_ts,   # turbulence
    )
    print(f"  Steps: {len(res['t'])}")
    print(f"  z range: {res['z'].min():.1f} - {res['z'].max():.1f} m "
          f"(ref: {z_ref.min():.1f} - {z_ref.max():.1f})")
    print(f"  phi range: {np.degrees(res['phi']).min():.1f} - "
          f"{np.degrees(res['phi']).max():.1f} deg")
    print(f"  theta range: {np.degrees(res['theta']).min():.1f} - "
          f"{np.degrees(res['theta']).max():.1f} deg")
    print(f"  Fz range: {res['Fz'].min():.2f} - {res['Fz'].max():.2f} N")
    print(f"  Mx range: {res['Mx'].min():.4f} - {res['Mx'].max():.4f} N*m")

    # --- Quick diagnostic plot ---
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    t = res['t']

    axes[0,0].plot(t, res['z'], label='actual'); axes[0,0].plot(t, res['z_ref'], '--', label='ref')
    axes[0,0].set_ylabel('z [m]'); axes[0,0].legend()

    axes[0,1].plot(t, res['x'], label='x'); axes[0,1].plot(t, res['x_ref'], '--')
    axes[0,1].plot(t, res['y'], label='y'); axes[0,1].plot(t, res['y_ref'], '--')
    axes[0,1].set_ylabel('x,y [m]'); axes[0,1].legend()

    axes[1,0].plot(t, np.degrees(res['phi']), label='phi')
    axes[1,0].plot(t, np.degrees(res['phi_cmd']), '--', label='phi_cmd')
    axes[1,0].set_ylabel('phi [deg]'); axes[1,0].legend()

    axes[1,1].plot(t, np.degrees(res['theta']), label='theta')
    axes[1,1].plot(t, np.degrees(res['theta_cmd']), '--', label='theta_cmd')
    axes[1,1].set_ylabel('theta [deg]'); axes[1,1].legend()

    axes[2,0].plot(t, res['Fz'])
    axes[2,0].set_ylabel('Fz [N]'); axes[2,0].set_xlabel('t [s]')

    axes[2,1].plot(t, res['Mx'], label='Mx'); axes[2,1].plot(t, res['My'], label='My')
    axes[2,1].set_ylabel('Mx,My [N*m]'); axes[2,1].set_xlabel('t [s]'); axes[2,1].legend()

    plt.tight_layout()
    # plt.savefig('/mnt/user-data/outputs/pid_sim_example.png', dpi=120)
    # print("Plot saved.")
    plt.show()

    fig, axes = plt.subplots(figsize=(6, 8))
    axes.plot(t,res['wind_x'],label="wind_x")
    axes.plot(t,res['wind_y'],label="wind_y")
    axes.plot(t,res['wind_z'],label="wind_z")
    axes.set_xlabel('t [s]')
    plt.legend() 
    plt.show() 


    # --- Acoustic post-processing ---
    from rotor_rpm_estimation import estimate_rotor_rpm
    from drone_acoustic_radiation_v2 import (
        calibrate_p_ref, AcousticParams, FineGridParams,
        estimate_received_spl_fine,
    )
    from zwicker_annoyance_v3 import (
        compute_zwicker_indicators_windowed,
        fill_nan_gaps,
        prepare_surrogate_targets,
    )

    # 1. Calibrate RPM-to-power model
    p_ref = calibrate_p_ref(
        spl_ref_db=72.0,
        rpm_ref_measurement=5000.0,
        r_ref=1.0,
        theta_ref_deg=90.0,
        n_rotors_in_measurement=4,
        n_exponent=5.0,
    )
    acoustic_params = AcousticParams(rpm_ref=5000.0, p_ref=p_ref, n_exponent=5.0)

    # 2. Fine-grid settings
    fine_params = FineGridParams(
        fs=48000.0,
        interp_method="cubic",
        use_integrated_phase=True,
        disturbance_amplitude_rad=0.05,
        disturbance_bandwidth_hz=20.0,
        random_seed=42,
    )

    # 3. Observer location
    observer_xyz = (10.0, 0.0, 0.0)

    # 4. RPM estimation -- note res['t'] and res['wind_x/y/z']
    rpm_result = estimate_rotor_rpm(
        res['t'],
        res['x'], res['y'], res['z'],
        res['vx'], res['vy'], res['vz'],
        res['ax'], res['ay'], res['az'],
        res['wind_x'], res['wind_y'], res['wind_z'],
    )

    # 5. Fine-grid SPL
    spl_fine = estimate_received_spl_fine(
        res['t'], res['x'], res['y'], res['z'],
        rpm_result["rpm_front"], rpm_result["rpm_rear"],
        rpm_result["rpm_right"], rpm_result["rpm_left"],
        observer_xyz,
        acoustic_params=acoustic_params,
        fine_params=fine_params,
    )

    # 6. Windowed Zwicker indicators
    result = compute_zwicker_indicators_windowed(
        spl_fine["p_signal"],
        fs=fine_params.fs,
        window_s=2.0,
        hop_s=0.5,
        use_fs_approximation=True,
        r=spl_fine["r"],
        max_distance_m=300.0,
        loudness_floor_sone=0.01,
    )

    # 7. Diagnostics
    print("NaN sharpness windows:",
        np.isnan(result["sharpness_acum"]).sum(), "/",
        len(result["sharpness_acum"]))
    print("NaN PA windows:", np.isnan(result["annoyance_PA"]).sum())
    print("SPL range: {:.1f} - {:.1f} dB".format(
        spl_fine["spl_db"].min(), spl_fine["spl_db"].max()))
    print("Distance range: {:.1f} - {:.1f} m".format(
        spl_fine["r"].min(), spl_fine["r"].max()))

    # 8. Plot (skip fluctuation_vacil if all NaN)
    keys_to_plot = ["loudness_sone", "sharpness_acum",
                    "roughness_asper", "annoyance_PA"]
    if not np.all(np.isnan(result["fluctuation_vacil"])):
        keys_to_plot.insert(3, "fluctuation_vacil")

    fig, axes = plt.subplots(len(keys_to_plot), 1,
                            figsize=(12, 2.5 * len(keys_to_plot)),
                            sharex=True)
    for ax, key in zip(axes, keys_to_plot):
        ax.plot(result["t_center"], result[key])
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

    # 9. Surrogate model targets (NaN -> 0 for inaudible windows)
    surrogate_data = prepare_surrogate_targets(result)
    print("Inaudible windows (PA=0):",
        surrogate_data["was_nan"]["annoyance_PA"].sum(),
        "/", len(result["t_center"]))






    
