"""Tests for Perfect Chaotic Encryption (PCE) functionality."""

import numpy as np
import pytest
from chaos_encryption.lorenz import PCEKey, solve_lorenz

def test_pce_key_signal_reproduction():
    """Test if identical PCE keys produce exactly the same chaotic signals."""
    # Create a PCE key with specific parameters
    key = PCEKey(
        ics=[1.0, 1.0, 1.0],
        a=10.0,
        b=8/3,
        r=28.0,
        method='RK45',
        rtol=1e-6,
        atol=1e-6
    )
    
    # Generate two signals with the same key
    t_span = (0, 1)
    dt = 0.01
    t1, sol1 = solve_lorenz(t_span, dt, key=key)
    t2, sol2 = solve_lorenz(t_span, dt, key=key)
    
    # Signals should be exactly identical
    np.testing.assert_allclose(sol1, sol2, rtol=1e-12)

def test_pce_key_different_methods():
    """Test if different solver methods produce sufficiently close results."""
    base_key = PCEKey(
        ics=[1.0, 1.0, 1.0],
        method='RK45'
    )
    
    # Create another key with different solver method
    alt_key = PCEKey(
        ics=[1.0, 1.0, 1.0],
        method='DOP853'
    )
    
    t_span = (0, 1)
    dt = 0.01
    
    # Generate signals with different methods
    t1, sol1 = solve_lorenz(t_span, dt, key=base_key)
    t2, sol2 = solve_lorenz(t_span, dt, key=alt_key)
    
    # For short time spans, different methods should still give similar results
    # Use a more relaxed tolerance since different methods can accumulate differences
    np.testing.assert_allclose(sol1, sol2, rtol=1e-3, atol=1e-3)

def test_pce_key_tolerance_sensitivity():
    """Test sensitivity to solver tolerance settings."""
    base_key = PCEKey(
        ics=[1.0, 1.0, 1.0],
        rtol=1e-6,
        atol=1e-6
    )
    
    # Create another key with different tolerances
    precise_key = PCEKey(
        ics=[1.0, 1.0, 1.0],
        rtol=1e-12,
        atol=1e-12
    )
    
    t_span = (0, 1)
    dt = 0.01
    
    # Generate signals with different tolerances
    t1, sol1 = solve_lorenz(t_span, dt, key=base_key)
    t2, sol2 = solve_lorenz(t_span, dt, key=precise_key)
    
    # Solutions should be close for short time spans
    # Use relaxed tolerances since chaotic systems are sensitive to solver settings
    np.testing.assert_allclose(sol1, sol2, rtol=1e-3, atol=1e-3)

def test_pce_key_validation():
    """Test PCEKey validation and error handling."""
    # Test invalid initial conditions
    with pytest.raises(ValueError, match="must be a 3D vector"):
        PCEKey(ics=[1.0, 1.0])  # Wrong dimension
    
    # Test invalid solver method
    with pytest.raises(ValueError, match="Invalid solver method"):
        PCEKey(ics=[1.0, 1.0, 1.0], method='INVALID')
    
    # Test invalid tolerance values
    with pytest.raises(ValueError, match="must be positive"):
        PCEKey(ics=[1.0, 1.0, 1.0], rtol=-1e-6)  # Negative tolerance
    
    # Test invalid Lorenz parameters
    with pytest.raises(ValueError, match="must be positive"):
        PCEKey(ics=[1.0, 1.0, 1.0], a=-10.0)  # Negative parameter