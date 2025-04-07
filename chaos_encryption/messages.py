"""Message generation and processing module for chaos-based encryption.

This module provides functions for generating and processing different types of message signals
used in chaos-based encryption, including Gaussian-modulated sine waves and text conversion.
"""

import numpy as np
from typing import List, Union


def generate_monochromatic(t_eval: np.ndarray, A: float, omega: float, t0: float) -> np.ndarray:
    """Generate a Gaussian-modulated sine wave message signal.

    Args:
        t_eval: Time array for signal evaluation.
        A: Amplitude of the signal.
        omega: Angular frequency of the carrier sine wave.
        t0: Center time of the Gaussian envelope.

    Returns:
        Message signal array m(t) = A * exp(-(t-t0)^2) * sin(omega*t).
    """
    # Calculate appropriate sigma based on time array span
    time_span = t_eval[-1] - t_eval[0]
    sigma = time_span / 40.0  # Narrower Gaussian for better localization
    
    # Generate Gaussian envelope with proper scaling
    gaussian_env = A * np.exp(-((t_eval - t0) / sigma)**2)
    
    # Generate carrier with phase aligned to envelope peak
    carrier = np.sin(omega * (t_eval - t0))
    
    return gaussian_env * carrier


def text_to_integers(text: str) -> List[int]:
    """Convert a text string to a list of Unicode integer values.

    Args:
        text: Input text string.

    Returns:
        List of Unicode integer values corresponding to each character.
    """
    if not text:  # Handle empty string case
        return []
    return [ord(char) for char in text]


def integers_to_signal(
    m_discrete: List[int],
    t_eval: np.ndarray,
    t_start: float,
    n_steps: int,
    norm_factor: float = 300.0
) -> np.ndarray:
    """Convert a list of integers into a time-series signal.

    Args:
        m_discrete: List of integer values to convert.
        t_eval: Time array for signal evaluation.
        t_start: Start time for the message in the signal.
        n_steps: Number of time steps per character.
        norm_factor: Normalization factor for the signal amplitude (default: 300.0).

    Returns:
        Time-series signal array where each integer is represented by a constant
        segment of n_steps duration with amplitude = integer_value/norm_factor.
    """
    if not m_discrete:  # Handle empty list case
        return np.zeros_like(t_eval)

    # Initialize output signal
    m_signal = np.zeros_like(t_eval)
    dt = t_eval[1] - t_eval[0]

    # Calculate start index with precise timing
    start_idx = int(np.round((t_start - t_eval[0]) / dt))
    if start_idx < 0 or start_idx >= len(t_eval):
        return m_signal

    # Generate signal segments with exact values
    for i, value in enumerate(m_discrete):
        seg_start = start_idx + i * n_steps
        seg_end = seg_start + n_steps
        
        if seg_end > len(t_eval):
            break
            
        # Set exact segment value
        m_signal[seg_start:seg_end] = value / norm_factor

    return m_signal


def signal_to_integers(
    m_recovered: np.ndarray,
    t_eval: np.ndarray,
    t_start: float,
    n_steps: int,
    norm_factor: float = 300.0,
    threshold: float = 0.1  # Threshold for valid segments as fraction of norm_factor
) -> List[int]:
    """Convert a recovered time-series signal back to integers.

    Args:
        m_recovered: Recovered message signal array.
        t_eval: Time array corresponding to the signal.
        t_start: Original start time of the message.
        n_steps: Number of time steps per character.
        norm_factor: Original normalization factor (default: 300.0).
        threshold: Minimum amplitude threshold as fraction of norm_factor (default: 0.1).

    Returns:
        List of recovered integer values.
    """
    dt = t_eval[1] - t_eval[0]
    start_idx = int(np.round((t_start - t_eval[0]) / dt))
    
    if start_idx < 0 or start_idx >= len(t_eval):
        return []

    # Extract signal segments with exact timing
    m_segments = []
    current_idx = start_idx
    min_valid_amplitude = threshold * norm_factor

    while current_idx + n_steps <= len(m_recovered):
        # Extract complete segment
        segment = m_recovered[current_idx:current_idx + n_steps]
        
        # Convert segment to integer values with rounding
        segment_values = np.round(segment * norm_factor)
        # Use mode of rounded values for more robust noise handling
        unique_values, counts = np.unique(segment_values, return_counts=True)
        mode_idx = np.argmax(counts)
        segment_value = int(unique_values[mode_idx])
        
        # Only accept segments with sufficient amplitude
        if abs(segment_value) >= min_valid_amplitude:
            m_segments.append(segment_value)
        elif len(m_segments) > 0:
            # Stop after finding all valid segments
            break
            
        current_idx += n_steps

    return m_segments


def integers_to_text(m_rec_discrete: List[int]) -> str:
    """Convert a list of integers back to text.

    Args:
        m_rec_discrete: List of recovered integer values.

    Returns:
        Recovered text string.
    """
    if not m_rec_discrete:  # Handle empty list case
        return ""

    # Convert integers to characters with error handling
    text_chars = []
    for value in m_rec_discrete:
        try:
            text_chars.append(chr(value))
        except (ValueError, OverflowError):
            # Replace invalid Unicode values with a placeholder
            text_chars.append('?')
    
    return ''.join(text_chars)