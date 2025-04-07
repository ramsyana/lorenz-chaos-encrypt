"""Unit tests for PCE (Perfect Chaotic Encryption) implementation."""

import numpy as np
import pytest
from chaos_encryption.pce import pce_add_encrypt, pce_add_decrypt

def test_pce_add_shape_validation():
    """Test shape validation in PCE addition functions."""
    u = np.array([1.0, 2.0, 3.0])
    m = np.array([0.1, 0.2])
    
    with pytest.raises(ValueError):
        pce_add_encrypt(u, m)
    
    with pytest.raises(ValueError):
        pce_add_decrypt(u, m)

def test_pce_add_encryption():
    """Test basic PCE addition encryption."""
    u = np.array([1.0, 2.0, 3.0])
    m = np.array([0.1, 0.2, 0.3])
    
    me = pce_add_encrypt(u, m)
    expected = np.array([1.1, 2.2, 3.3])
    np.testing.assert_allclose(me, expected)

def test_pce_add_decryption():
    """Test basic PCE addition decryption."""
    me = np.array([1.1, 2.2, 3.3])
    u = np.array([1.0, 2.0, 3.0])
    
    mr = pce_add_decrypt(me, u)
    expected = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(mr, expected)

def test_pce_add_encryption_decryption_cycle():
    """Test full PCE addition encryption-decryption cycle."""
    # Original signals
    u = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    m = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    
    # Encrypt
    me = pce_add_encrypt(u, m)
    
    # Decrypt
    mr = pce_add_decrypt(me, u)
    
    # Verify recovered message matches original
    np.testing.assert_allclose(mr, m)