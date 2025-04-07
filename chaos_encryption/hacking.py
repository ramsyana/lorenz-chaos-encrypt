import numpy as np
from scipy.fft import fft, ifft, fftfreq

def high_pass_filter(signal, dt, cutoff_freq):
    """Apply a high-pass filter to remove components below cutoff frequency.
    
    Args:
        signal (np.array): Input signal to filter
        dt (float): Time step
        cutoff_freq (float): Cutoff frequency in Hz
        
    Returns:
        np.array: Filtered signal with low frequencies removed
    """
    n = len(signal)
    freqs = fftfreq(n, dt)
    fft_signal = fft(signal)
    
    # Create frequency mask
    mask = np.abs(freqs) > cutoff_freq
    
    # Apply mask and inverse transform
    filtered_fft = fft_signal * mask
    filtered_signal = ifft(filtered_fft)
    
    return np.real(filtered_signal)

def hack_sync_signal(signal, dt, w_co):
    """Attempt to hack synchronization-encrypted signal using frequency analysis.
    
    Args:
        signal (np.array): Encrypted signal
        dt (float): Time step
        w_co (float): Cutoff frequency
        
    Returns:
        np.array: Attempted recovery of original message
    """
    return high_pass_filter(signal, dt, w_co)

def hack_pce_signal(signal, dt, w_co):
    """Attempt to hack PCE-encrypted signal using frequency analysis.
    
    Args:
        signal (np.array): Encrypted signal
        dt (float): Time step
        w_co (float): Cutoff frequency
        
    Returns:
        np.array: Attempted recovery of original message
    """
    return high_pass_filter(signal, dt, w_co)