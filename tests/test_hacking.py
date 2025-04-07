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
def message_params():
    return dict(A=1, omega=10, t0=5)

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

def test_hack_pce_signal_add(time_params, lorenz_params, message_params):
    t, dt = time_params
    
    # Generate chaotic carrier
    ics = [5.0, 5.0, 5.0]
    t_sol, sol = solve_lorenz(ics, [t[0], t[-1]], dt, **lorenz_params)
    u_signal = sol[0]
    
    # Generate message
    m_signal = generate_monochromatic(t, **message_params)
    
    # Encrypt using PCE addition
    me_signal = pce_add_encrypt(u_signal, m_signal)
    
    # Try to hack the signal
    cutoff_freq = 35  # As suggested in the paper
    mh_signal = hack_pce_signal(me_signal, dt, cutoff_freq)
    
    # The hacked signal should have some correlation with the original message
    # but not be perfect (as noted in the paper)
    correlation = np.corrcoef(m_signal, mh_signal)[0,1]
    assert 0.1 < np.abs(correlation) < 0.9  # Some correlation but not perfect

def test_hack_pce_signal_convolve(time_params, lorenz_params, message_params):
    t, dt = time_params
    
    # Generate chaotic carrier
    ics = [5.0, 5.0, 5.0]
    t_sol, sol = solve_lorenz(ics, [t[0], t[-1]], dt, **lorenz_params)
    u_signal = sol[0]
    
    # Generate message
    m_signal = generate_monochromatic(t, **message_params)
    
    # Encrypt using PCE convolution
    me_signal = pce_convolve_encrypt(u_signal, m_signal)
    
    # Try to hack the signal
    cutoff_freq = 35  # As suggested in the paper
    mh_signal = hack_pce_signal(me_signal, dt, cutoff_freq)
    
    # The correlation should be lower than with PCE addition
    # as convolution should be more resistant to this attack
    correlation = np.corrcoef(m_signal, mh_signal)[0,1]
    assert np.abs(correlation) < 0.5  # Lower correlation expected

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