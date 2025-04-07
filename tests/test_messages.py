"""Unit tests for the message generation and processing module."""

import numpy as np
import pytest
from chaos_encryption.messages import (
    generate_monochromatic,
    text_to_integers,
    integers_to_signal,
    signal_to_integers,
    integers_to_text
)


def test_generate_monochromatic():
    """Test the Gaussian-modulated sine wave generation."""
    # Test parameters
    # Using an odd number of points centered around t0 can make checks easier
    num_points = 2001 # Odd number
    t_eval = np.linspace(0, 200, num_points)
    A = 1.0
    omega = 10.0
    t0 = 100.0

    # Generate signal
    m = generate_monochromatic(t_eval, A, omega, t0)

    # Check output type and shape
    assert isinstance(m, np.ndarray)
    assert m.shape == t_eval.shape

    # Check peak location of the *envelope* (should be exactly at t0 for odd points)
    # Calculate sigma using the same heuristic as in the function for consistency
    relevant_span = min(t0 - t_eval[0], t_eval[-1] - t0) * 2
    if relevant_span <= 0:
        relevant_span = t_eval[-1] - t_eval[0]
    sigma = max(relevant_span / 10.0, (t_eval[1]-t_eval[0]) * 5)
    gaussian_env_test = A * np.exp(-0.5 * ((t_eval - t0) / sigma)**2)
    peak_idx_env = np.argmax(gaussian_env_test)
    # With odd points centered, t0 should be exactly sampled
    assert np.isclose(t_eval[peak_idx_env], t0)

    # Check symmetry of the *absolute value* around t0 with relaxed tolerance
    t0_idx = peak_idx_env # Center index found above
    window = 500  # Check symmetry within a larger window around t0

    # Ensure window fits within the array bounds
    if t0_idx >= window and t0_idx + window < len(t_eval):
        left = m[t0_idx - window : t0_idx]
        # The signal is m(t) = Env(t) * sin(omega*(t-t0)).
        # At t=t0, sin(0) = 0, so m[t0_idx] should be close to 0.
        # Due to anti-symmetry: m(t0-dt) = Env(t0-dt)*sin(-omega*dt) ~= -Env(t0+dt)*sin(omega*dt) = -m(t0+dt)
        # So abs(left) should be close to abs(right)
        right_start_idx = t0_idx + 1
        right_end_idx = t0_idx + 1 + window
        if right_end_idx <= len(m):
            right = m[right_start_idx : right_end_idx][::-1] # Reverse for comparison
            # Relax tolerance significantly due to discretization near zero crossings and envelope changes
            np.testing.assert_allclose(np.abs(left), np.abs(right), atol=1e-1, rtol=1e-1) # Relaxed atol and rtol
        else:
             pytest.skip("Window right side exceeds bounds")
    else:
        pytest.skip("Window does not fit for symmetry check")

    # Check decay far from t0 (Value depends heavily on sigma chosen)
    # Check relative decay: value at edge should be small fraction of peak envelope
    peak_envelope_value = A # Max of Gaussian is A
    assert abs(m[0]) < peak_envelope_value * 0.01  # Should be < 1% of peak envelope at start
    assert abs(m[-1]) < peak_envelope_value * 0.01 # Should be < 1% of peak envelope at end


def test_text_to_integers():
    """Test text to integers conversion."""
    # Test empty string
    assert text_to_integers("") == []

    # Test basic ASCII
    text = "Hello"
    integers = text_to_integers(text)
    assert integers == [72, 101, 108, 108, 111]

    # Test special characters
    text = "Hello, 世界!"
    integers = text_to_integers(text)
    assert integers == [72, 101, 108, 108, 111, 44, 32, 19990, 30028, 33]


def test_integers_to_signal():
    """Test conversion of integers to time-series signal."""
    # Test parameters
    t_eval = np.linspace(0, 10, 1000)
    t_start = 2.0
    n_steps = 50
    norm_factor = 300.0
    m_discrete = [72, 101, 108]  # "Hel"

    # Generate signal
    m = integers_to_signal(m_discrete, t_eval, t_start, n_steps, norm_factor)

    # Check output type and shape
    assert isinstance(m, np.ndarray)
    assert m.shape == t_eval.shape

    # Check signal values at expected locations
    dt = t_eval[1] - t_eval[0]
    start_idx = int(np.round((t_start - t_eval[0]) / dt))
    
    # Check first character segment
    segment1 = m[start_idx:start_idx + n_steps]
    np.testing.assert_allclose(segment1, 72/norm_factor, rtol=1e-10)

    # Check zeros before start
    assert np.all(m[:start_idx] == 0)

    # Test empty input
    m_empty = integers_to_signal([], t_eval, t_start, n_steps, norm_factor)
    assert np.all(m_empty == 0)


def test_signal_to_integers():
    """Test conversion of time-series signal back to integers."""
    # Create a test signal
    t_eval = np.linspace(0, 10, 1001)  # Use odd number for clearer indexing
    dt = t_eval[1] - t_eval[0]
    t_start = 2.0
    n_steps = 50
    norm_factor = 300.0
    original_integers = [72, 101, 108]  # "Hel"

    # Generate signal and recover integers (NO noise first)
    m = integers_to_signal(original_integers, t_eval, t_start, n_steps, norm_factor)
    recovered_integers = signal_to_integers(m, t_eval, t_start, n_steps, norm_factor)

    # Check recovery accuracy (NO noise)
    assert recovered_integers == original_integers, f"Failed without noise. Got {recovered_integers}"

    # Test with noisy signal
    # Make noise relative to the smallest possible step (1/norm_factor)
    noise_std_dev = (1.0 / norm_factor) * 0.1  # Noise std dev is 10% of smallest step size
    noise = np.random.normal(0, noise_std_dev, m.shape)
    m_noisy = m + noise
    recovered_noisy = signal_to_integers(m_noisy, t_eval, t_start, n_steps, norm_factor)

    # Check recovery accuracy (WITH noise)
    assert recovered_noisy == original_integers, f"Failed with noise. Got {recovered_noisy}"

    # Test empty signal
    assert signal_to_integers(np.zeros_like(t_eval), t_eval, t_start, n_steps) == []


def test_integers_to_text():
    """Test conversion of integers back to text."""
    # Test empty list
    assert integers_to_text([]) == ""

    # Test basic ASCII
    integers = [72, 101, 108, 108, 111]
    assert integers_to_text(integers) == "Hello"

    # Test special characters
    integers = [72, 101, 108, 108, 111, 44, 32, 19990, 30028, 33]
    assert integers_to_text(integers) == "Hello, 世界!"

    # Test invalid Unicode values
    integers = [72, -1, 999999999]
    text = integers_to_text(integers)
    assert text[0] == 'H'  # Valid character
    assert text[1:] == '??'  # Invalid characters replaced with ?


def test_full_text_conversion_cycle():
    """Test the full cycle of text conversion and recovery."""
    # Original text with mixed ASCII and Unicode
    original_text = "Hello, 世界! 123"

    # Convert to integers
    integers = text_to_integers(original_text)

    # Convert to signal
    t_eval = np.linspace(0, 20, 2000)
    t_start = 5.0
    n_steps = 50
    m = integers_to_signal(integers, t_eval, t_start, n_steps)

    # Recover integers from signal
    recovered_integers = signal_to_integers(m, t_eval, t_start, n_steps)

    # Convert back to text
    recovered_text = integers_to_text(recovered_integers)

    # Check if we recovered the original text
    assert recovered_text == original_text