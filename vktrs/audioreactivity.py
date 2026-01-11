"""
Audioreactive signal processing for VKTRS.

Extract and transform audio signals to drive animation parameters.
Supports stem-based reactivity (vocals, drums, bass) and frequency-specific effects.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import List, Callable, Optional


# ============================================================================
# Signal Extraction
# ============================================================================

def create_signal(audio_data: np.ndarray,
                 sr: int = 22050,
                 hop_length: int = 512) -> np.ndarray:
    """
    Create base signal from audio data.

    Args:
        audio_data: Audio waveform
        sr: Sample rate
        hop_length: Hop length for frame-based features

    Returns:
        RMS energy envelope
    """
    import librosa
    rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
    return rms


# ============================================================================
# Signal Transformations
# ============================================================================

def bandpass_filter(signal_data: np.ndarray,
                   low_freq: float,
                   high_freq: float,
                   sr: int = 22050,
                   order: int = 5) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to isolate frequency range.

    Args:
        signal_data: Input signal
        low_freq: Low cutoff frequency (Hz)
        high_freq: High cutoff frequency (Hz)
        sr: Sample rate
        order: Filter order

    Returns:
        Filtered signal
    """
    nyquist = sr / 2
    low = low_freq / nyquist
    high = high_freq / nyquist

    b, a = scipy_signal.butter(order, [low, high], btype='band')
    filtered = scipy_signal.filtfilt(b, a, signal_data)
    return filtered


def envelope_follower(signal_data: np.ndarray,
                     attack: float = 0.01,
                     release: float = 0.1,
                     sr: int = 22050) -> np.ndarray:
    """
    Extract amplitude envelope from signal.

    Args:
        signal_data: Input signal
        attack: Attack time in seconds
        release: Release time in seconds
        sr: Sample rate

    Returns:
        Envelope signal
    """
    envelope = np.zeros_like(signal_data)
    envelope[0] = abs(signal_data[0])

    attack_coef = np.exp(-1.0 / (attack * sr))
    release_coef = np.exp(-1.0 / (release * sr))

    for i in range(1, len(signal_data)):
        curr_abs = abs(signal_data[i])
        if curr_abs > envelope[i-1]:
            # Attack
            envelope[i] = attack_coef * envelope[i-1] + (1 - attack_coef) * curr_abs
        else:
            # Release
            envelope[i] = release_coef * envelope[i-1] + (1 - release_coef) * curr_abs

    return envelope


def normalize_signal(signal_data: np.ndarray,
                    min_val: float = 0.0,
                    max_val: float = 1.0) -> np.ndarray:
    """
    Normalize signal to range [min_val, max_val].

    Args:
        signal_data: Input signal
        min_val: Minimum output value
        max_val: Maximum output value

    Returns:
        Normalized signal
    """
    sig_min = np.min(signal_data)
    sig_max = np.max(signal_data)

    if sig_max - sig_min < 1e-10:
        return np.full_like(signal_data, (min_val + max_val) / 2)

    normalized = (signal_data - sig_min) / (sig_max - sig_min)
    return normalized * (max_val - min_val) + min_val


def smooth_signal(signal_data: np.ndarray,
                 window_size: int = 5) -> np.ndarray:
    """
    Smooth signal using moving average.

    Args:
        signal_data: Input signal
        window_size: Window size for averaging

    Returns:
        Smoothed signal
    """
    if window_size < 2:
        return signal_data

    window = np.ones(window_size) / window_size
    smoothed = np.convolve(signal_data, window, mode='same')
    return smoothed


def detect_peaks(signal_data: np.ndarray,
                threshold: float = 0.5,
                min_distance: int = 10) -> np.ndarray:
    """
    Detect peaks in signal.

    Args:
        signal_data: Input signal
        threshold: Minimum peak height (relative)
        min_distance: Minimum distance between peaks

    Returns:
        Binary signal (1 at peaks, 0 elsewhere)
    """
    from scipy.signal import find_peaks

    # Normalize for threshold
    normalized = normalize_signal(signal_data)

    peaks, _ = find_peaks(normalized, height=threshold, distance=min_distance)

    peak_signal = np.zeros_like(signal_data)
    peak_signal[peaks] = 1.0

    return peak_signal


# ============================================================================
# Signal Pipeline
# ============================================================================

def apply_transformations(signal_data: np.ndarray,
                         transformations: List[str],
                         sr: int = 22050) -> np.ndarray:
    """
    Apply chain of transformations to signal.

    Args:
        signal_data: Input signal
        transformations: List of transformation specs
            - 'bandpass:low:high' - Bandpass filter
            - 'envelope' - Envelope follower
            - 'normalize' - Normalize to 0-1
            - 'smooth:window' - Moving average
            - 'peaks:threshold' - Peak detection

        sr: Sample rate

    Returns:
        Transformed signal

    Example:
        >>> transforms = ['bandpass:60:250', 'envelope', 'normalize']
        >>> bass_signal = apply_transformations(audio, transforms)
    """
    result = signal_data.copy()

    for transform in transformations:
        parts = transform.split(':')
        transform_name = parts[0]

        if transform_name == 'bandpass':
            low_freq = float(parts[1])
            high_freq = float(parts[2])
            result = bandpass_filter(result, low_freq, high_freq, sr)

        elif transform_name == 'envelope':
            result = envelope_follower(result, sr=sr)

        elif transform_name == 'normalize':
            result = normalize_signal(result)

        elif transform_name == 'smooth':
            window_size = int(parts[1]) if len(parts) > 1 else 5
            result = smooth_signal(result, window_size)

        elif transform_name == 'peaks':
            threshold = float(parts[1]) if len(parts) > 1 else 0.5
            result = detect_peaks(result, threshold)

        else:
            print(f"⚠️  Unknown transformation: {transform_name}")

    return result


# ============================================================================
# Parameter Mapping
# ============================================================================

def map_signal_to_parameter(signal_data: np.ndarray,
                           param_name: str,
                           base_value: float,
                           amplitude: float,
                           offset: float = 0.0) -> np.ndarray:
    """
    Map signal to animation parameter values.

    Args:
        signal_data: Input signal (normalized 0-1)
        param_name: Parameter name (for reference)
        base_value: Base parameter value
        amplitude: Modulation amplitude
        offset: Signal offset

    Returns:
        Parameter values over time

    Example:
        >>> # Map bass signal to zoom
        >>> zoom_values = map_signal_to_parameter(
        ...     bass_signal, 'zoom', base_value=1.0, amplitude=0.2
        ... )
    """
    modulated = base_value + amplitude * (signal_data + offset)
    return modulated


def create_audioreactive_curve(signal_data: np.ndarray,
                              timestamps: np.ndarray,
                              param_name: str,
                              **mapping_kwargs) -> dict:
    """
    Create keyframed curve from audioreactive signal.

    Args:
        signal_data: Input signal
        timestamps: Time values for each signal sample
        param_name: Parameter name
        **mapping_kwargs: Arguments for map_signal_to_parameter

    Returns:
        Dict with 'times' and 'values' for keyframed curve
    """
    values = map_signal_to_parameter(signal_data, param_name, **mapping_kwargs)

    return {
        'param': param_name,
        'times': timestamps.tolist(),
        'values': values.tolist()
    }
