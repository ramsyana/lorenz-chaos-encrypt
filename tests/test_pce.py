"""Unit tests for PCE (Perfect Chaotic Encryption) implementation."""

import numpy as np
import pytest
from chaos_encryption.pce import (pce_add_encrypt, pce_add_decrypt,
                                 pce_convolve_encrypt, pce_convolve_decrypt)

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

def test_pce_convolve_shape_validation():
    """Test shape validation in PCE convolution functions."""
    u = np.array([1.0, 2.0, 3.0])
    m = np.array([0.1, 0.2])
    
    with pytest.raises(ValueError):
        pce_convolve_encrypt(u, m)
    
    with pytest.raises(ValueError):
        pce_convolve_decrypt(u, m)

def test_pce_convolve_encryption():
    """Test basic PCE convolution encryption."""
    # Test with simple known convolution
    # Convolution of [1, 2] and [3, 4] (padded)
    # [1, 2, 0] * [3, 4, 0] -> should approximate [3, 10, 8] (linear convolution result)
    u = np.array([1.0, 2.0, 0.0])
    m = np.array([3.0, 4.0, 0.0])
    expected_conv = np.convolve(u, m)[:len(u)]  # Linear convolution result truncated

    me = pce_convolve_encrypt(u, m)
    # FFT convolution is circular, result depends on padding
    # For this simple case, it should match the linear convolution start
    np.testing.assert_allclose(me, expected_conv, atol=1e-8)

def test_pce_convolve_decryption():
    """Test basic PCE convolution decryption."""
    # Let u be simple, e.g., a scaled delta
    u = np.array([2.0, 0.0, 0.0])  # Scaled Delta
    # me should be result of convolving u and expected m
    expected_m = np.array([0.1, 0.2, 0.3])
    # Calculate manually: Convolution with [2,0,0] just scales
    me = expected_m * 2.0

    # Decrypt using default small epsilon
    mr = pce_convolve_decrypt(me, u)
    # Relax tolerance for FFT/IFFT cycle
    np.testing.assert_allclose(mr, expected_m, atol=1e-8)

def test_pce_convolve_encryption_decryption_cycle():
    """Test full PCE convolution encryption-decryption cycle."""
    # Original signals - use longer random signals
    np.random.seed(42)  # for reproducibility
    u = np.random.rand(128) - 0.5
    m = (np.random.rand(128) - 0.5) * 0.1  # smaller amplitude message

    # Encrypt
    me = pce_convolve_encrypt(u, m)

    # Decrypt using default small epsilon
    mr = pce_convolve_decrypt(me, u)

    # Verify recovered message matches original
    # Relax tolerance significantly for full FFT cycle
    np.testing.assert_allclose(mr, m, atol=1e-8)  # Absolute tolerance is better here

def test_pce_convolve_numerical_stability():
    """Test PCE convolution with small signal values in u."""
    # Use a signal u that has frequency components close to zero
    np.random.seed(43)
    N = 128
    # Create a u signal that's mostly low frequency, likely small high freq FFT components
    t = np.linspace(0, 10, N)
    u = np.sin(t * 0.5) + 0.1 * (np.random.rand(N) - 0.5)  # Low freq + small noise
    m = (np.random.rand(N) - 0.5) * 0.1

    # Encrypt
    me = pce_convolve_encrypt(u, m)

    # Decrypt with different epsilon values
    # Epsilon should be very small, related to machine precision
    mr_tiny_eps = pce_convolve_decrypt(me, u, epsilon=1e-15)
    mr_small_eps = pce_convolve_decrypt(me, u, epsilon=1e-10)

    # Both should recover the message with reasonable accuracy
    # The accuracy depends more on how close F_u gets to zero than epsilon itself,
    # as long as epsilon prevents actual division by zero.
    np.testing.assert_allclose(mr_tiny_eps, m, atol=1e-7)  # Increased tolerance
    np.testing.assert_allclose(mr_small_eps, m, atol=1e-7)  # Increased tolerance

    # Verify that the results are very close to each other
    np.testing.assert_allclose(mr_tiny_eps, mr_small_eps, atol=1e-9)