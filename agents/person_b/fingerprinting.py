"""
Agent 5: Fingerprinting Agent

Extract signal features using spectrograms and embeddings.
11 tools - MEMORY OPTIMIZED
"""
import numpy as np
from scipy import signal
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from schemas.state_schema import MIRAState
from schemas.message_schema import FingerprintOutput


class FingerprintingAgent:
    """Agent 5: Fingerprinting - Extract features from signals"""
    
    def __init__(self):
        self.name = "Fingerprinting"
    
    def compute_stft(self, signal_data: np.ndarray, fs: int = 1000) -> Dict:
        """Tool 1: Compute STFT"""
        f, t, Zxx = signal.stft(signal_data, fs=fs, nperseg=256)
        return {"frequencies": f, "times": t, "stft": Zxx}
    
    def compute_mel_spectrogram(self, signal_data: np.ndarray, fs: int = 1000) -> np.ndarray:
        """Tool 2: Compute mel-spectrogram (simplified)"""
        f, t, Zxx = signal.stft(signal_data, fs=fs, nperseg=256)
        S = np.abs(Zxx)
        
        # Resize to standard size (128x128)
        from scipy.ndimage import zoom
        target_shape = (128, 128)
        zoom_factors = (target_shape[0] / S.shape[0], target_shape[1] / S.shape[1])
        S_resized = zoom(S, zoom_factors, order=1)
        
        return S_resized
    
    def normalize_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """Tool 3: Normalize to dB scale"""
        spec_db = 20 * np.log10(spec + 1e-10)
        # Normalize to [0, 1]
        spec_db = (spec_db - np.min(spec_db)) / (np.max(spec_db) - np.min(spec_db) + 1e-10)
        return spec_db
    
    def extract_peak_frequencies(self, spec: np.ndarray, freqs: np.ndarray, n_peaks: int = 5) -> list:
        """Tool 4: Find peak frequencies"""
        # Sum over time
        power_spectrum = np.mean(spec, axis=1)
        peak_indices = np.argsort(power_spectrum)[-n_peaks:][::-1]
        return [float(freqs[i]) for i in peak_indices if i < len(freqs)]
    
    def flatten_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """Tool 5: Flatten to vector"""
        return spec.flatten()
    
    def apply_pca_embedding(self, features: np.ndarray, n_components: int = 64) -> np.ndarray:
        """Tool 6: PCA dimensionality reduction"""
        from sklearn.decomposition import PCA
        
        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Limit components to avoid memory issues
        n_comp = min(n_components, features.shape[1], 64)
        pca = PCA(n_components=n_comp)
        
        if features.shape[0] == 1:
            # Single sample - can't fit PCA, return truncated
            return features[0, :n_comp]
        
        embedded = pca.fit_transform(features)
        return embedded[0] if embedded.shape[0] == 1 else embedded
    
    def apply_umap_embedding(self, features: np.ndarray, n_components: int = 32) -> np.ndarray:
        """Tool 7: UMAP (simplified - just use PCA for single sample)"""
        # For single sample, UMAP doesn't work, use PCA
        return self.apply_pca_embedding(features, n_components)
    
    def combine_imu_audio_features(self, imu_features: np.ndarray, audio_features: np.ndarray) -> np.ndarray:
        """Tool 8: Concatenate features"""
        # Take first N features from each to limit size
        imu_trunc = imu_features[:16] if len(imu_features) > 16 else imu_features
        audio_trunc = audio_features[:16] if len(audio_features) > 16 else audio_features
        return np.concatenate([imu_trunc, audio_trunc])
    
    def save_fingerprint_vector(self, vector: np.ndarray, path: str) -> None:
        """Tool 9: Save fingerprint"""
        np.save(path, vector)
    
    def run(self, state: MIRAState) -> FingerprintOutput:
        """Execute fingerprinting agent"""
        import time as time_module
        start_time = time_module.time()
        
        try:
            # Get IMU signal
            imu = state.get("imu_normalized", state.get("imu_data"))
            if imu.ndim == 2:
                imu = imu[:, 2]  # Z-axis only
            
            # Compute spectrogram
            spec = self.compute_mel_spectrogram(imu, fs=1000)
            spec_norm = self.normalize_spectrogram(spec)
            
            # Extract features
            flat_features = self.flatten_spectrogram(spec_norm)
            
            # Reduce dimensionality (memory efficient)
            fingerprint = self.apply_pca_embedding(flat_features, n_components=32)
            
            # Ensure fingerprint is 32D
            if len(fingerprint) > 32:
                fingerprint = fingerprint[:32]
            elif len(fingerprint) < 32:
                fingerprint = np.pad(fingerprint, (0, 32 - len(fingerprint)))
            
            # Extract peak frequencies
            stft_result = self.compute_stft(imu, fs=1000)
            peak_freqs = self.extract_peak_frequencies(
                np.abs(stft_result["stft"]),
                stft_result["frequencies"],
                n_peaks=4
            )
            
            # Harmonic ratios (simplified)
            if len(peak_freqs) >= 2:
                harmonic_ratios = [peak_freqs[i] / peak_freqs[0] for i in range(1, min(4, len(peak_freqs)))]
            else:
                harmonic_ratios = [1.0]
            
            exec_time = time_module.time() - start_time
            
            return FingerprintOutput(
                agent_name=self.name,
                success=True,
                execution_time_sec=exec_time,
                spectrogram=spec_norm,
                fingerprint_vector=fingerprint,
                peak_frequencies=peak_freqs,
                harmonic_ratios=harmonic_ratios,
                embedding_method="PCA",
            )
        
        except Exception as e:
            exec_time = time_module.time() - start_time
            return FingerprintOutput(
                agent_name=self.name,
                success=False,
                error_message=str(e),
                execution_time_sec=exec_time,
                spectrogram=np.zeros((128, 128)),
                fingerprint_vector=np.zeros(32),
                peak_frequencies=[],
                harmonic_ratios=[],
                embedding_method="failed",
            )
