"""Lorenz system simulation module.

Implements functions for solving the Lorenz system ODEs and handling
the receiver system for chaos synchronization.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def lorenz_odes(t, state, a=10.0, b=8/3, r=28.0):
    """Define the Lorenz system ODEs.
    
    Args:
        t (float): Time point (not used, but required by solve_ivp).
        state (array-like): State vector [u, v, w].
        a (float, optional): First Lorenz parameter. Defaults to 10.0.
        b (float, optional): Second Lorenz parameter. Defaults to 8/3.
        r (float, optional): Third Lorenz parameter. Defaults to 28.0.
    
    Returns:
        list: Derivatives [du/dt, dv/dt, dw/dt].
    """
    u, v, w = state
    du_dt = a * (v - u)
    dv_dt = r * u - v - u * w
    dw_dt = u * v - b * w
    return [du_dt, dv_dt, dw_dt]

def solve_lorenz(ics, t_span, dt, a=10.0, b=8/3, r=28.0):
    """Solve the Lorenz system using solve_ivp.
    
    Args:
        ics (array-like): Initial conditions [u0, v0, w0].
        t_span (tuple): Time span for integration (t_start, t_end).
        dt (float): Time step for output points.
        a (float, optional): First Lorenz parameter. Defaults to 10.0.
        b (float, optional): Second Lorenz parameter. Defaults to 8/3.
        r (float, optional): Third Lorenz parameter. Defaults to 28.0.
    
    Returns:
        tuple: (t, sol) where t is time array and sol is solution array (3, len(t)).
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        lorenz_odes,
        t_span,
        ics,
        args=(a, b, r),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-6
    )
    return sol.t, sol.y

def lorenz_receiver_odes(t, state_r, me_signal_interp, a=10.0, b=8/3, r=28.0):
    """Define the receiver system ODEs for synchronization.
    
    Args:
        t (float): Time point.
        state_r (array-like): Receiver state vector [ur, vr, wr].
        me_signal_interp (callable): Interpolated me(t) signal function.
        a (float, optional): First Lorenz parameter. Defaults to 10.0.
        b (float, optional): Second Lorenz parameter. Defaults to 8/3.
        r (float, optional): Third Lorenz parameter. Defaults to 28.0.
    
    Returns:
        list: Derivatives [dur/dt, dvr/dt, dwr/dt].
    """
    ur, vr, wr = state_r
    me = me_signal_interp(t)
    
    # Receiver system with me(t) driving signal (Eq. 7)
    dur_dt = a * (vr - ur)
    dvr_dt = r * me - vr - me * wr
    dwr_dt = me * vr - b * wr
    
    return [dur_dt, dvr_dt, dwr_dt]

def solve_lorenz_receiver(ics_r, t_eval, me_signal, t_signal, a=10.0, b=8/3, r=28.0):
    """Solve the receiver system ODEs.
    
    Args:
        ics_r (array-like): Receiver initial conditions [ur0, vr0, wr0].
        t_eval (array-like): Time points for evaluation.
        me_signal (array-like): Encrypted signal me(t).
        t_signal (array-like): Time points for me_signal.
        a (float, optional): First Lorenz parameter. Defaults to 10.0.
        b (float, optional): Second Lorenz parameter. Defaults to 8/3.
        r (float, optional): Third Lorenz parameter. Defaults to 28.0.
    
    Returns:
        tuple: (t, sol_r) where t is time array and sol_r is receiver solution array.
    """
    # Create interpolation function for me(t)
    me_signal_interp = interp1d(
        t_signal,
        me_signal,
        kind='cubic',
        bounds_error=False,
        fill_value=(me_signal[0], me_signal[-1])
    )
    
    # Solve receiver system
    sol = solve_ivp(
        lorenz_receiver_odes,
        (t_eval[0], t_eval[-1]),
        ics_r,
        args=(me_signal_interp, a, b, r),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-6
    )
    
    return sol.t, sol.y