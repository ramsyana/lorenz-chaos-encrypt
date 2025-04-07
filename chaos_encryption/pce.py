"""PCE (Perfect Chaotic Encryption) implementation.

This module implements PCE using both addition and convolution methods.
The addition method is similar to synchronization but requires exact chaotic signal.
"""

import numpy as np

def pce_add_encrypt(u_signal: np.ndarray, m_signal: np.ndarray) -> np.ndarray:
    """Encrypt message using PCE addition method.
    
    Args:
        u_signal: Sender's chaotic signal (u component)
        m_signal: Message signal to encrypt
        
    Returns:
        Encrypted signal me(t) = u(t) + m(t)
    """
    if u_signal.shape != m_signal.shape:
        raise ValueError("Chaotic and message signals must have same shape")
    
    return u_signal + m_signal

def pce_add_decrypt(me_signal: np.ndarray, u_signal: np.ndarray) -> np.ndarray:
    """Decrypt message using PCE addition method.
    
    Args:
        me_signal: Received encrypted signal
        u_signal: Sender's chaotic signal (exact same as used in encryption)
        
    Returns:
        Recovered message signal mr(t) = me(t) - u(t)
    """
    if me_signal.shape != u_signal.shape:
        raise ValueError("Encrypted and chaotic signals must have same shape")
        
    return me_signal - u_signal

def pce_convolve_encrypt(u_signal: np.ndarray, m_signal: np.ndarray) -> np.ndarray:
    """Encrypt message using PCE convolution method in frequency domain.
    
    Args:
        u_signal: Sender's chaotic signal (u component)
        m_signal: Message signal to encrypt
        
    Returns:
        Encrypted signal me(t) = u(t) * m(t) (convolution)
    """
    if u_signal.shape != m_signal.shape:
        raise ValueError("Chaotic and message signals must have same shape")
    
    # Compute FFTs
    F_u = np.fft.fft(u_signal)
    F_m = np.fft.fft(m_signal)
    
    # Multiply in frequency domain (equivalent to convolution in time domain)
    F_me = F_u * F_m
    
    # Inverse FFT and take real part (result should be real since inputs are real)
    me_signal = np.fft.ifft(F_me)
    return np.real(me_signal)

def pce_convolve_decrypt(me_signal: np.ndarray, u_signal: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """Decrypt message using PCE convolution method in frequency domain.
    
    Args:
        me_signal: Received encrypted signal
        u_signal: Sender's chaotic signal (exact same as used in encryption)
        epsilon: Small value for regularization to avoid division by zero.
                 Should be very small, close to machine epsilon perhaps.
        
    Returns:
        Recovered message signal mr(t) through deconvolution
    """
    if me_signal.shape != u_signal.shape:
        raise ValueError("Encrypted and chaotic signals must have same shape")
    
    # Compute FFTs
    F_me = np.fft.fft(me_signal)
    F_u = np.fft.fft(u_signal)
    
    # --- Simpler Regularization ---
    # Avoid division by zero or very small numbers in F_u.
    # Add epsilon directly, or use a threshold. Adding epsilon is simpler.
    # A very small epsilon should have minimal impact unless F_u is pathologically small.
    F_u_reg = F_u + epsilon # Direct addition
    
    F_mr = F_me / F_u_reg # Use the regularized denominator
    
    # Inverse FFT and ensure real output
    mr_signal = np.fft.ifft(F_mr)
    # Check if imaginary part is truly small before discarding
    if np.max(np.abs(np.imag(mr_signal))) > 1e-9 * np.max(np.abs(np.real(mr_signal))):
         print("Warning: Significant imaginary part in ifft result during decryption.")
    return np.real(mr_signal)