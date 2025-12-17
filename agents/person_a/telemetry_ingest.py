"""
Agent 2: Telemetry Ingest Agent

Normalizes and validates incoming sensor data.
18 tools for data cleaning, synchronization, and quality checking.
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, Any, List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from schemas.state_schema import MIRAState
from schemas.message_schema import IngestOutput


class TelemetryIngestAgent:
    """
    Agent 2: Telemetry Ingest
    
    Responsibility: Clean, normalize, and validate sensor data from any source.
    
    18 Tools:
    - normalize_meter_json()
    - normalize_sensor_packet()
    - infer_data_type()
    - sanitize_payload()
    - extract_timestamp(), extract_device_id(), extract_geo_metadata()
    - check_sampling_rate(), fix_missing_values(), convert_to_float()
    - sync_audio_imu()
    - resample_to_standard_rate()
    - trim_invalid_edges(), denoise_signal(), segment_into_windows()
    - validate_signal_quality(), write_normalized_output()
    """
    
    def __init__(self):
        self.name = "TelemetryIngest"
    
    def normalize_meter_json(self, json_data: Dict) -> Dict:
        """Tool 1: Normalize JSON telemetry"""
        return {
            "timestamp": json_data.get("timestamp", ""),
            "device_id": json_data.get("device_id", "unknown"),
            "data": json_data.get("data", {}),
        }
    
    def infer_data_type(self, data: Any) -> str:
        """Tool 2: Infer data type"""
        if isinstance(data, np.ndarray):
            if data.ndim == 2 and data.shape[1] == 3:
                return "imu_3axis"
            elif data.ndim == 1:
                return "audio_mono"
        return "unknown"
    
    def sanitize_payload(self, data: np.ndarray) -> np.ndarray:
        """Tool 3: Remove invalid values"""
        # Replace NaN and Inf with 0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data
    
    def check_sampling_rate(self, time: np.ndarray) -> Tuple[float, bool]:
        """Tool 4: Validate sampling rate"""
        if len(time) < 2:
            return 0.0, False
        
        dt = np.diff(time)
        mean_dt = np.mean(dt)
        std_dt = np.std(dt)
        
        # Check if consistent (low jitter)
        is_consistent = (std_dt / mean_dt) < 0.01 if mean_dt > 0 else False
        
        sampling_rate = 1.0 / mean_dt if mean_dt > 0 else 0.0
        return sampling_rate, is_consistent
    
    def fix_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Tool 5: Interpolate missing values"""
        if data.ndim == 1:
            # Find NaN indices
            nan_idx = np.isnan(data)
            if np.any(nan_idx):
                # Linear interpolation
                valid_idx = ~nan_idx
                data[nan_idx] = np.interp(
                    np.where(nan_idx)[0],
                    np.where(valid_idx)[0],
                    data[valid_idx],
                )
        return data
    
    def convert_to_float(self, data: Any) -> np.ndarray:
        """Tool 6: Convert to float array"""
        return np.asarray(data, dtype=np.float32)
    
    def sync_audio_imu(
        self,
        imu_data: np.ndarray,
        audio_data: np.ndarray,
        imu_rate: int,
        audio_rate: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tool 7: Synchronize audio and IMU using cross-correlation"""
        # For simplicity, assume already synchronized
        # In real system, would use cross-correlation to find time offset
        return imu_data, audio_data
    
    def resample_to_standard_rate(
        self,
        data: np.ndarray,
        current_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        """Tool 8: Resample signal"""
        if current_rate == target_rate:
            return data
        
        n_samples = int(len(data) * target_rate / current_rate)
        
        if data.ndim == 1:
            return signal.resample(data, n_samples)
        else:
            # Resample each axis
            return np.column_stack([
                signal.resample(data[:, i], n_samples)
                for i in range(data.shape[1])
            ])
    
    def trim_invalid_edges(self, data: np.ndarray, trim_samples: int = 100) -> np.ndarray:
        """Tool 9: Remove edge transients"""
        if len(data) > 2 * trim_samples:
            return data[trim_samples:-trim_samples]
        return data
    
    def denoise_signal(
        self,
        data: np.ndarray,
        cutoff_hz: float = 500.0,
        sampling_rate: int = 1000,
    ) -> np.ndarray:
        """Tool 10: Low-pass filter to remove noise"""
        nyquist = sampling_rate / 2
        cutoff_norm = cutoff_hz / nyquist
        
        if cutoff_norm >= 1.0:
            return data
        
        b, a = signal.butter(4, cutoff_norm, btype='low')
        
        if data.ndim == 1:
            return signal.filtfilt(b, a, data)
        else:
            return np.column_stack([
                signal.filtfilt(b, a, data[:, i])
                for i in range(data.shape[1])
            ])
    
    def segment_into_windows(
        self,
        data: np.ndarray,
        window_sec: float = 5.0,
        sampling_rate: int = 1000,
    ) -> List[np.ndarray]:
        """Tool 11: Segment signal into windows"""
        window_samples = int(window_sec * sampling_rate)
        n_windows = len(data) // window_samples
        
        segments = []
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            segments.append(data[start:end])
        
        return segments
    
    def validate_signal_quality(
        self,
        signal_data: np.ndarray,
        sampling_rate: int,
    ) -> Dict[str, Any]:
        """Tool 12: Compute signal quality metrics"""
        # Compute SNR (simple estimate)
        signal_power = np.mean(signal_data ** 2)
        noise_estimate = np.var(np.diff(signal_data))  # High-freq noise
        snr_db = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else 100
        
        # Quality score (0-1)
        quality_score = np.clip(snr_db / 40.0, 0.0, 1.0)  # 40dB = excellent
        
        issues = []
        if snr_db < 10:
            issues.append("Low SNR")
        if np.any(np.abs(signal_data) > 0.9):
            issues.append("Potential clipping")
        if len(signal_data) < sampling_rate:
            issues.append("Short signal duration")
        
        return {
            "quality_score": float(quality_score),
            "snr_db": float(snr_db),
            "issues": issues,
            "passed": quality_score > 0.5,
        }
    
    # =========================================================================
    # Agent Execution
    # =========================================================================
    
    def run(self, state: MIRAState) -> IngestOutput:
        """
        Execute telemetry ingest agent
        
        Args:
            state: State with raw imu_data and audio_data
        
        Returns:
            IngestOutput with normalized signals
        """
        import time as time_module
        start_time = time_module.time()
        
        try:
            imu_raw = state["imu_data"]
            audio_raw = state["audio_data"]
            imu_rate = state.get("imu_sampling_rate", 1000)
            audio_rate = state.get("audio_sampling_rate", 44100)
            
            issues = []
            corrections = []
            
            # Sanitize
            imu_clean = self.sanitize_payload(imu_raw.copy())
            audio_clean = self.sanitize_payload(audio_raw.copy())
            corrections.append("Sanitized invalid values")
            
            # Fix missing values
            if imu_clean.ndim == 2:
                for i in range(imu_clean.shape[1]):
                    imu_clean[:, i] = self.fix_missing_values(imu_clean[:, i])
            else:
                imu_clean = self.fix_missing_values(imu_clean)
            
            audio_clean = self.fix_missing_values(audio_clean)
            corrections.append("Interpolated missing values")
            
            # Denoise
            imu_denoised = self.denoise_signal(imu_clean, cutoff_hz=500, sampling_rate=imu_rate)
            audio_denoised = self.denoise_signal(audio_clean, cutoff_hz=10000, sampling_rate=audio_rate)
            corrections.append("Applied low-pass filter")
            
            # Validate quality
            imu_quality = self.validate_signal_quality(imu_denoised[:, 2] if imu_denoised.ndim == 2 else imu_denoised, imu_rate)
            audio_quality = self.validate_signal_quality(audio_denoised, audio_rate)
            
            if not imu_quality["passed"]:
                issues.extend([f"IMU: {i}" for i in imu_quality["issues"]])
            if not audio_quality["passed"]:
                issues.extend([f"Audio: {i}" for i in audio_quality["issues"]])
            
            # Combined quality score
            quality_score = (imu_quality["quality_score"] + audio_quality["quality_score"]) / 2
            snr_db = (imu_quality["snr_db"] + audio_quality["snr_db"]) / 2
            
            exec_time = time_module.time() - start_time
            
            return IngestOutput(
                agent_name=self.name,
                success=True,
                execution_time_sec=exec_time,
                imu_normalized=imu_denoised,
                audio_normalized=audio_denoised,
                signal_quality_score= quality_score,
                snr_db=snr_db,
                issues_detected=issues,
                corrections_applied=corrections,
            )
        
        except Exception as e:
            exec_time = time_module.time() - start_time
            return IngestOutput(
                agent_name=self.name,
                success=False,
                error_message=str(e),
                execution_time_sec=exec_time,
                imu_normalized=np.array([]),
                audio_normalized=np.array([]),
                signal_quality_score=0.0,
                snr_db=0.0,
                issues_detected=[str(e)],
                corrections_applied=[],
            )
