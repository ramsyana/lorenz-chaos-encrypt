import numpy as np
import pytest
from chaos_encryption.hacking import high_pass_filter, hack_pce_signal
from chaos_encryption.pce import pce_add_encrypt, pce_convolve_encrypt
from chaos_encryption.lorenz import solve_lorenz
from chaos_encryption.messages import generate_monochromatic

@pytest.fixture
def time_params():
    t_span = [0, 10]  # Shorter time span for testing
    dt = 0.01
    t = np.arange(t_span[0], t_span[1], dt)
    return t, dt

@pytest.fixture
def lorenz_params():
    return dict(a=10, b=8/3, r=28)

@pytest.fixture
def message_params_low_freq():
    return dict(A=1, omega=10, t0=5)  # ~1.59 Hz, below cutoff

@pytest.fixture
def message_params_high_freq():
    return dict(A=1, omega=300, t0=5)  # ~47.7 Hz, above cutoff

def test_high_pass_filter_sine_waves(time_params):
    t, dt = time_params
    # Create a signal with both low and high frequency components
    low_freq = 1  # 1 Hz
    high_freq = 20  # 20 Hz
    signal = np.sin(2*np.pi*low_freq*t) + 0.5*np.sin(2*np.pi*high_freq*t)
    
    # Test with cutoff between low and high frequencies
    cutoff_freq = 10  # 10 Hz
    filtered = high_pass_filter(signal, dt, cutoff_freq)
    
    # The filtered signal should mostly contain the high frequency component
    # Calculate power at test frequencies using FFT
    n = len(t)
    freqs = np.fft.fftfreq(n, dt)
    signal_fft = np.fft.fft(filtered)
    
    # Check power at low frequency is significantly reduced
    low_freq_power = np.abs(signal_fft[np.abs(freqs - low_freq).argmin()])
    high_freq_power = np.abs(signal_fft[np.abs(freqs - high_freq).argmin()])
    
    assert low_freq_power < 0.1  # Low frequency should be mostly filtered out
    assert high_freq_power > 0.2  # High frequency should remain strong

def test_high_pass_filter_dc_removal(time_params):
    t, dt = time_params
    # Create a signal with DC offset
    signal = 2.0 + np.sin(2*np.pi*5*t)
    filtered = high_pass_filter(signal, dt, cutoff_freq=1)
    
    # DC component should be removed
    assert np.abs(np.mean(filtered)) < 0.1

def test_hack_pce_signal_add_high_freq(time_params, lorenz_params, message_params_high_freq):
    t, dt = time_params
    
    # Generate chaotic carrier
    ics = [5.0, 5.0, 5.0]
    t_sol, sol = solve_lorenz(ics, [t[0], t[-1]], dt, **lorenz_params)
    u_signal = sol[0]
    
    # Generate high-frequency message (above cutoff)
    m_signal = generate_monochromatic(t_sol, **message_params_high_freq)
    
    # Encrypt using PCE addition
    me_signal = pce_add_encrypt(u_signal, m_signal)
    
    # Try to hack the signal
    cutoff_freq = 35  # As suggested in the paper
    mh_signal = hack_pce_signal(me_signal, dt, cutoff_freq)
    
    # For high-frequency message (above cutoff), expect strong correlation
    # demonstrating vulnerability of PCE-Addition to frequency analysis
    correlation = np.corrcoef(m_signal, mh_signal)[0,1]
    assert 0.5 < np.abs(correlation) < 1.0  # Strong correlation expected

def test_hack_pce_signal_add_low_freq(time_params, lorenz_params, message_params_low_freq):
    t, dt = time_params
    
    # Generate chaotic carrier
    ics = [5.0, 5.0, 5.0]
    t_sol, sol = solve_lorenz(ics, [t[0], t[-1]], dt, **lorenz_params)
    u_signal = sol[0]
    
    # Generate low-frequency message (below cutoff)
    m_signal = generate_monochromatic(t_sol, **message_params_low_freq)
    
    # Encrypt using PCE addition
    me_signal = pce_add_encrypt(u_signal, m_signal)
    
    # Try to hack the signal
    cutoff_freq = 35  # As suggested in the paper
    mh_signal = hack_pce_signal(me_signal, dt, cutoff_freq)
    
    # For low-frequency message (below cutoff), expect near-zero correlation
    # as the high-pass filter removes the message
    correlation = np.corrcoef(m_signal, mh_signal)[0,1]
    assert np.abs(correlation) < 0.1  # Near-zero correlation expected

def test_hack_pce_signal_convolve(time_params, lorenz_params, message_params_high_freq):
    t, dt = time_params
    
    # Generate chaotic carrier
    ics = [5.0, 5.0, 5.0]
    t_sol, sol = solve_lorenz(ics, [t[0], t[-1]], dt, **lorenz_params)
    u_signal = sol[0]
    
    # Generate message using the same time array as the chaotic signal
    m_signal = generate_monochromatic(t_sol, **message_params_high_freq)
    
    # Encrypt using PCE convolution
    me_signal = pce_convolve_encrypt(u_signal, m_signal)
    
    # Try to hack the signal
    cutoff_freq = 35  # As suggested in the paper
    mh_signal = hack_pce_signal(me_signal, dt, cutoff_freq)
    
    # For convolution, we need to check both correlation and signal magnitude
    # The convolution operation affects both the frequency content and amplitude
    correlation = np.corrcoef(m_signal, mh_signal)[0,1]
    
    # Calculate relative magnitude between hacked and original signals
    m_magnitude = np.sqrt(np.mean(np.square(m_signal)))
    mh_magnitude = np.sqrt(np.mean(np.square(mh_signal)))

    # Avoid division by zero if original message is zero (edge case)
    if m_magnitude < 1e-10:
        if mh_magnitude < 1e-10:
            magnitude_ratio = 1.0  # Both zero, ratio is effectively 1
        else:
            magnitude_ratio = np.inf  # Original zero, hacked non-zero
    else:
        magnitude_ratio = mh_magnitude / m_magnitude

    if not np.isnan(correlation):
        # Check that the recovery is not perfect
        # Either the correlation is not extremely high (e.g., < 0.99)
        # OR the magnitude ratio is noticeably different from 1 (e.g., outside [0.9, 1.1])
        assert (np.abs(correlation) < 0.99) or not (0.9 < magnitude_ratio < 1.1), \
            f"PCE convolution hack yielded unexpectedly high correlation ({correlation:.3f}) " \
            f"and near-original magnitude (ratio {magnitude_ratio:.3f}). " \
            f"Expected significant distortion."
    else:
        # If correlation is NaN (likely due to zero variance), hacked signal should be effectively zero
        assert mh_magnitude < 1e-9, "Correlation is NaN, but hacked signal magnitude is not near zero"

def test_hack_pce_signal_edge_cases(time_params):
    t, dt = time_params
    
    # Test with zero signal
    zero_signal = np.zeros_like(t)
    result = hack_pce_signal(zero_signal, dt, 10)
    assert np.allclose(result, 0)
    
    # Test with constant signal
    const_signal = np.ones_like(t)
    result = hack_pce_signal(const_signal, dt, 10)
    assert np.all(np.abs(result) < 0.1)  # Should remove DC component