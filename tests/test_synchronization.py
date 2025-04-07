import numpy as np
import pytest
from chaos_encryption.synchronization import sync_encrypt, sync_decrypt

def test_sync_encrypt_shape_validation():
    """Test that sync_encrypt raises ValueError for mismatched shapes."""
    u_signal = np.array([1.0, 2.0, 3.0])
    m_signal = np.array([1.0, 2.0])
    
    with pytest.raises(ValueError):
        sync_encrypt(u_signal, m_signal)

def test_sync_decrypt_shape_validation():
    """Test that sync_decrypt raises ValueError for mismatched shapes."""
    me_signal = np.array([1.0, 2.0, 3.0])
    ur_signal = np.array([1.0, 2.0])
    
    with pytest.raises(ValueError):
        sync_decrypt(me_signal, ur_signal)

def test_sync_encrypt_element_wise_addition():
    """Test that sync_encrypt performs correct element-wise addition."""
    u_signal = np.array([1.0, 2.0, 3.0])
    m_signal = np.array([0.1, 0.2, 0.3])
    expected = np.array([1.1, 2.2, 3.3])
    
    me_signal = sync_encrypt(u_signal, m_signal)
    np.testing.assert_allclose(me_signal, expected)

def test_sync_decrypt_element_wise_subtraction():
    """Test that sync_decrypt performs correct element-wise subtraction."""
    me_signal = np.array([1.1, 2.2, 3.3])
    ur_signal = np.array([1.0, 2.0, 3.0])
    expected = np.array([0.1, 0.2, 0.3])
    
    mr_signal = sync_decrypt(me_signal, ur_signal)
    np.testing.assert_allclose(mr_signal, expected)

def test_sync_encryption_decryption_cycle():
    """Test the complete encryption-decryption cycle recovers the original message."""
    u_signal = np.array([1.0, 2.0, 3.0])
    m_signal = np.array([0.1, 0.2, 0.3])
    
    # Encrypt the message
    me_signal = sync_encrypt(u_signal, m_signal)
    
    # Decrypt using the same chaotic signal (perfect synchronization case)
    mr_signal = sync_decrypt(me_signal, u_signal)
    
    # Verify recovered message matches original
    np.testing.assert_allclose(mr_signal, m_signal)