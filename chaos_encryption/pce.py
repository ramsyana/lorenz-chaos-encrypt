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