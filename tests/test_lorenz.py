"""Tests for the Lorenz system simulation module."""

import numpy as np
import pytest
from scipy.interpolate import interp1d
from chaos_encryption.lorenz import (
    lorenz_odes,
    solve_lorenz,
    lorenz_receiver_odes,
    solve_lorenz_receiver
)

def test_lorenz_odes_output_shape():
    """Test lorenz_odes output type and shape."""
    t = 0.0
    state = [1.0, 1.0, 1.0]
    derivatives = lorenz_odes(t, state)
    
    assert isinstance(derivatives, list)
    assert len(derivatives) == 3
    assert all(isinstance(d, float) for d in derivatives)

def test_lorenz_odes_known_state():
    """Test lorenz_odes calculation for a simple state."""
    t = 0.0
    state = [1.0, 1.0, 1.0]
    a, b, r = 10.0, 8/3, 28.0
    
    derivatives = lorenz_odes(t, state, a, b, r)
    
    # For state=[1,1,1]:
    # du/dt = a(v-u) = 0
    # dv/dt = ru - v - uw = 28 - 1 - 1 = 26
    # dw/dt = uv - bw = 1 - 8/3
    expected = [0.0, 26.0, 1.0 - 8/3]
    np.testing.assert_allclose(derivatives, expected)

def test_solve_lorenz_output():
    """Test solve_lorenz output shapes and types."""
    ics = [5.0, 5.0, 5.0]
    t_span = (0, 1)
    dt = 0.1
    
    t, sol = solve_lorenz(ics, t_span, dt)
    
    assert isinstance(t, np.ndarray)
    assert isinstance(sol, np.ndarray)
    assert t.shape[0] == sol.shape[1]  # Number of time points
    assert sol.shape[0] == 3  # Three variables (u,v,w)

def test_solve_lorenz_initial_conditions():
    """Test if solve_lorenz solution matches initial conditions."""
    ics = [5.0, 5.0, 5.0]
    t_span = (0, 1)
    dt = 0.1
    
    t, sol = solve_lorenz(ics, t_span, dt)
    
    # Check if solution at t=0 matches initial conditions
    np.testing.assert_allclose(sol[:, 0], ics)

def test_solve_lorenz_time_points():
    """Test if solve_lorenz time points match expected steps."""
    t_span = (0, 1)
    dt = 0.1
    ics = [5.0, 5.0, 5.0]
    
    t, _ = solve_lorenz(ics, t_span, dt)
    
    expected_t = np.arange(t_span[0], t_span[1], dt)
    np.testing.assert_allclose(t, expected_t)

def test_lorenz_receiver_odes_output():
    """Test lorenz_receiver_odes output type and shape."""
    t = 0.0
    state_r = [1.0, 1.0, 1.0]
    # Create a simple constant me(t) function
    me_signal_interp = lambda t: 1.0
    
    derivatives = lorenz_receiver_odes(t, state_r, me_signal_interp)
    
    assert isinstance(derivatives, list)
    assert len(derivatives) == 3
    assert all(isinstance(d, float) for d in derivatives)

def test_solve_lorenz_receiver_sync():
    """Test if receiver synchronizes with sender when m(t)=0."""
    # Generate sender solution
    ics = [5.0, 5.0, 5.0]
    t_span = (0, 20)  # Longer time for synchronization
    dt = 0.01
    t, sol = solve_lorenz(ics, t_span, dt)
    
    # Use u component as me(t) with no message (m(t)=0)
    me_signal = sol[0]  # u component
    t_signal = t
    
    # Different initial conditions for receiver
    ics_r = [25.0, 6.0, 50.0]
    
    # Solve receiver system
    t_r, sol_r = solve_lorenz_receiver(ics_r, t, me_signal, t_signal)
    
    # After transient (t > 10), receiver should synchronize
    sync_idx = t > 10
    sync_error = np.max(np.abs(sol_r[:, sync_idx] - sol[:, sync_idx]))
    
    assert sync_error < 1e-2  # Should be small after synchronization