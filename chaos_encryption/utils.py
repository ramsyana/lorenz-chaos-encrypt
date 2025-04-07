import numpy as np
from scipy.fft import fft, fftfreq

def calculate_l2_norm(signal1, signal2):
    """Calculate L2 norm (Euclidean distance) between two signals.
    
    Args:
        signal1 (np.array): First signal
        signal2 (np.array): Second signal
        
    Returns:
        float: L2 norm between the signals
    """
    return np.sqrt(np.mean((signal1 - signal2) ** 2))

def calculate_fft(signal, dt):
    """Calculate the FFT of a signal.
    
    Args:
        signal (np.array): Input signal
        dt (float): Time step
        
    Returns:
        tuple: (frequencies, fft_values)
    """
    n = len(signal)
    freqs = fftfreq(n, dt)
    fft_values = fft(signal)
    
    # Sort by frequency
    idx = np.argsort(freqs)
    return freqs[idx], fft_values[idx]

def calculate_fidelity_text(original_text, recovered_text):
    """Calculate the fidelity (character match percentage) between original and recovered text.
    
    Args:
        original_text (str): Original text message
        recovered_text (str): Recovered text message
        
    Returns:
        float: Percentage of matching characters
    """
    if len(original_text) != len(recovered_text):
        return 0.0
    
    matches = sum(1 for a, b in zip(original_text, recovered_text) if a == b)
    return (matches / len(original_text)) * 100