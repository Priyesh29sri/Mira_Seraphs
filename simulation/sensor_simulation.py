"""
Sensor Simulation

Generates synthetic IMU and audio signals from vehicle dynamics
"""

import numpy as np
from scipy import signal
import soundfile as sf
from typing import Tuple, Optional


def simulate_imu_signal(
    acceleration: np.ndarray,
    time: np.ndarray,
    sensor_location: int = 4,  # Node index (4 = body)
    noise_level: float = 0.1,
    sampling_rate: int = 1000,
) -> Tuple[np.ndarray, int]:
    """
    Simulate IMU accelerometer signal
    
    Args:
        acceleration: (n_steps, 6) acceleration array from vehicle model
        time: Time vector
        sensor_location: Which node has the IMU sensor
        noise_level: Measurement noise std dev (m/s²)
        sampling_rate: Desired sampling rate (Hz)
    
    Returns:
        imu_signal: (n_samples, 3) array - [ax, ay, az] in m/s²
        sampling_rate: Actual sampling rate
    """
    # Extract acceleration at sensor location (vertical = z-axis)
    az = acceleration[:, sensor_location]
    
    # For simplicity, assume only vertical vibration (ax=0, ay=0, az=vibration)
    # In a full 3D model, we'd have lateral accelerations too
    ax = np.zeros_like(az)
    ay = np.zeros_like(az)
    
    # Resample to desired sampling rate
    current_dt = time[1] - time[0]
    current_rate = 1.0 / current_dt
    
    if current_rate != sampling_rate:
        n_samples = int(len(time) * sampling_rate / current_rate)
        az = signal.resample(az, n_samples)
        ax = signal.resample(ax, n_samples)
        ay = signal.resample(ay, n_samples)
    
    # Add measurement noise
    az += np.random.normal(0, noise_level, len(az))
    ax += np.random.normal(0, noise_level, len(ax))
    ay += np.random.normal(0, noise_level, len(ay))
    
    # Combine into 3-axis signal
    imu_signal = np.column_stack([ax, ay, az])
    
    return imu_signal, sampling_rate


def simulate_audio_signal(
    acceleration: np.ndarray,
    time: np.ndarray,
    microphone_location: int = 5,  # Node index (5 = engine)
    noise_level: float = 0.01,
    sampling_rate: int = 44100,
) -> Tuple[np.ndarray, int]:
    """
    Simulate microphone audio signal
    
    Audio is modeled as a bandpassed version of acceleration
    (simulating structure-borne noise transmission)
    
    Args:
        acceleration: (n_steps, 6) acceleration array
        time: Time vector
        microphone_location: Which node contributes most to audio
        noise_level: Background noise level
        sampling_rate: Audio sampling rate (Hz)
    
    Returns:
        audio_signal: (n_samples,) mono audio
        sampling_rate: Actual sampling rate
    """
    # Extract acceleration at microphone location
    a = acceleration[:, microphone_location]
    
    # Resample to audio sampling rate
    current_dt = time[1] - time[0]
    current_rate = 1.0 / current_dt
    n_samples = int(len(time) * sampling_rate / current_rate)
    audio = signal.resample(a, n_samples)
    
    # Bandpass filter to audio range (20 Hz - 10 kHz)
    # Design Butterworth bandpass filter
    nyquist = sampling_rate / 2
    low_freq = 20.0 / nyquist
    high_freq = 10000.0 / nyquist
    b, a_coeff = signal.butter(4, [low_freq, high_freq], btype='band')
    audio = signal.filtfilt(b, a_coeff, audio)
    
    # Normalize to reasonable amplitude (audio typically in [-1, 1] range)
    if np.max(np.abs(audio)) > 0:
        audio = audio / (np.max(np.abs(audio)) * 2)  # Scale to [-0.5, 0.5]
    
    # Add background noise
    audio += np.random.normal(0, noise_level, len(audio))
    
    # Clip to prevent saturation
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio, sampling_rate


def save_imu_to_csv(
    imu_signal: np.ndarray,
    time: np.ndarray,
    filepath: str,
):
    """
    Save IMU data to CSV
    
    Args:
        imu_signal: (n_samples, 3) array
        time: Time vector (or None to generate)
        filepath: Output CSV path
    """
    import pandas as pd
    
    if time is None:
        time = np.arange(len(imu_signal)) / 1000.0  # Assume 1kHz
    
    df = pd.DataFrame({
        'time_sec': time[:len(imu_signal)],
        'ax_m_s2': imu_signal[:, 0],
        'ay_m_s2': imu_signal[:, 1],
        'az_m_s2': imu_signal[:, 2],
    })
    
    df.to_csv(filepath, index=False)


def save_audio_to_wav(
    audio_signal: np.ndarray,
    sampling_rate: int,
    filepath: str,
):
    """
    Save audio to WAV file
    
    Args:
        audio_signal: (n_samples,) mono audio
        sampling_rate: Sampling rate (Hz)
        filepath: Output WAV path
    """
    sf.write(filepath, audio_signal, sampling_rate)


def add_sensor_drift(
    signal: np.ndarray,
    drift_rate: float = 0.01,
) -> np.ndarray:
    """
    Add slow drift to sensor signal (realistic sensor behavior)
    
    Args:
        signal: Input signal
        drift_rate: Drift rate (units per sample)
    
    Returns:
        Signal with drift added
    """
    drift = np.cumsum(np.random.randn(len(signal)) * drift_rate)
    return signal + drift


def add_quantization_noise(
    signal: np.ndarray,
    bits: int = 16,
    full_scale: float = 10.0,
) -> np.ndarray:
    """
    Add quantization noise (ADC simulation)
    
    Args:
        signal: Input signal
        bits: ADC resolution (bits)
        full_scale: Full-scale range
    
    Returns:
        Quantized signal
    """
    levels = 2 ** bits
    step_size = (2 * full_scale) / levels
    
    quantized = np.round(signal / step_size) * step_size
    return quantized
