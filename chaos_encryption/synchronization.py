import numpy as np

def sync_encrypt(u_signal, m_signal):
    """Encrypts by adding the message to the chaotic signal (u component).

    Args:
        u_signal (numpy.ndarray): Emitter's chaotic signal (u component).
        m_signal (numpy.ndarray): Message signal to encrypt.

    Returns:
        numpy.ndarray: Encrypted signal me_signal = u_signal + m_signal.
    """
    if u_signal.shape != m_signal.shape:
        raise ValueError("u_signal and m_signal must have the same shape")
    
    return u_signal + m_signal

def sync_decrypt(me_signal, ur_signal):
    """Decrypts by subtracting the receiver's synchronized signal (ur component).

    Args:
        me_signal (numpy.ndarray): Received encrypted signal.
        ur_signal (numpy.ndarray): Receiver's synchronized signal (ur component).

    Returns:
        numpy.ndarray: Recovered message signal mr_signal = me_signal - ur_signal.
    """
    if me_signal.shape != ur_signal.shape:
        raise ValueError("me_signal and ur_signal must have the same shape")
    
    return me_signal - ur_signal