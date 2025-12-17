"""
Tool Output Schemas

Defines Pydantic models for all 80+ tool outputs.
Each tool returns a structured dict matching these schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Any, Dict
import numpy as np


# =============================================================================
# Physics Tools
# =============================================================================

class MassMatrixOutput(BaseModel):
    """Output from build_mass_matrix()"""
    matrix: Any  # np.ndarray, shape (n_dof, n_dof)
    n_dof: int
    total_mass_kg: float
    is_symmetric: bool
    is_positive_definite: bool
    
    class Config:
        arbitrary_types_allowed = True


class StiffnessMatrixOutput(BaseModel):
    """Output from build_stiffness_matrix()"""
    matrix: Any  # np.ndarray
    n_dof: int
    condition_number: float
    eigenvalues: Any  # np.ndarray
    
    class Config:
        arbitrary_types_allowed = True


class ODEIntegrationOutput(BaseModel):
    """Output from integrate_system_ode()"""
    displacement: Any  # np.ndarray, shape (n_steps, n_dof)
    velocity: Any  # np.ndarray
    acceleration: Any  # np.ndarray
    time: Any  # np.ndarray
    n_steps: int
    duration_sec: float
    
    class Config:
        arbitrary_types_allowed = True


class FaultLocationOutput(BaseModel):
    """Output from estimate_fault_coordinates()"""
    x_meters: float = Field(..., ge=-2.0, le=2.0)
    y_meters: float = Field(..., ge=-1.0, le=1.0)
    z_meters: float = Field(..., ge=-0.5, le=0.5)
    uncertainty_radius_cm: float = Field(..., gt=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    method: str  # "L1_inverse", "L2_inverse", "modal_energy"
    computation_time_sec: float


class HeatmapOutput(BaseModel):
    """Output from generate_fault_heatmap()"""
    heatmap: Any  # np.ndarray, shape (128, 128)
    max_energy_location: Tuple[int, int]
    max_energy_value: float
    normalization_method: str
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Signal Processing Tools
# =============================================================================

class FFTOutput(BaseModel):
    """Output from compute_fft_spectrum()"""
    frequencies: Any  # np.ndarray
    amplitudes: Any  # np.ndarray
    phases: Any  # np.ndarray
    sampling_rate: int
    n_fft: int
    
    class Config:
        arbitrary_types_allowed = True


class STFTOutput(BaseModel):
    """Output from compute_stft()"""
    spectrogram: Any  # np.ndarray, shape (n_freq, n_time)
    frequencies: Any  # np.ndarray
    times: Any  # np.ndarray
    window_length: int
    hop_length: int
    
    class Config:
        arbitrary_types_allowed = True


class MelSpectrogramOutput(BaseModel):
    """Output from compute_mel_spectrogram()"""
    mel_spec: Any  # np.ndarray
    mel_frequencies: Any  # np.ndarray
    n_mels: int
    fmin: float
    fmax: float
    
    class Config:
        arbitrary_types_allowed = True


class PeakFrequenciesOutput(BaseModel):
    """Output from extract_peak_frequencies()"""
    peak_frequencies: List[float]
    peak_amplitudes: List[float]
    peak_indices: List[int]
    n_peaks: int
    detection_threshold: float


# =============================================================================
# Machine Learning Tools
# =============================================================================

class PCAEmbeddingOutput(BaseModel):
    """Output from apply_pca_embedding()"""
    embedded: Any  # np.ndarray
    explained_variance: List[float]
    n_components: int
    total_explained_variance: float
    
    class Config:
        arbitrary_types_allowed = True


class UMAPEmbeddingOutput(BaseModel):
    """Output from apply_umap_embedding()"""
    embedded: Any  # np.ndarray
    n_neighbors: int
    min_dist: float
    metric: str
    
    class Config:
        arbitrary_types_allowed = True


class KNNMatchOutput(BaseModel):
    """Output from knn_match()"""
    neighbor_indices: List[int]
    neighbor_distances: List[float]
    neighbor_run_ids: List[str]
    k: int
    distance_metric: str


class ClusteringOutput(BaseModel):
    """Output from cluster_fingerprints()"""
    cluster_labels: Any  # np.ndarray
    n_clusters: int
    cluster_sizes: Dict[int, int]
    silhouette_score: float
    algorithm: str  # "DBSCAN", "KMeans"
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Causal Inference Tools
# =============================================================================

class CorrelationOutput(BaseModel):
    """Output from compute_feature_correlations()"""
    correlation_matrix: Any  # np.ndarray
    feature_names: List[str]
    top_correlations: List[Dict[str, Any]]  # [{"feature1": str, "feature2": str, "corr": float}]
    
    class Config:
        arbitrary_types_allowed = True


class TreatmentEffectOutput(BaseModel):
    """Output from estimate_treatment_effect()"""
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str  # "propensity_score", "matching"


class BayesianCausalOutput(BaseModel):
    """Output from bayesian_cause_posterior()"""
    cause_probabilities: Dict[str, float]  # {"imbalance": 0.78, "loose_mount": 0.15, ...}
    prior: Dict[str, float]
    likelihood: Dict[str, float]
    evidence: float
    top_cause: str
    top_cause_probability: float


# =============================================================================
# Experiment Design Tools
# =============================================================================

class UncertaintyCheckOutput(BaseModel):
    """Output from check_uncertainty_threshold()"""
    current_uncertainty_bits: float
    threshold_bits: float
    exceeds_threshold: bool
    uncertainty_sources: List[str]


class ExperimentDesignOutput(BaseModel):
    """Output from design_new_speed_profile()"""
    new_speed_kmh: float
    new_load_kg: Optional[float] = None
    duration_sec: float
    predicted_info_gain_bits: float
    rationale: str


# =============================================================================
# Scheduling Tools
# =============================================================================

class RepairUrgencyOutput(BaseModel):
    """Output from compute_repair_urgency()"""
    urgency: int = Field(..., ge=1, le=10)
    factors: Dict[str, float]  # {"severity": 0.6, "safety_risk": 0.8, ...}
    recommendation: str  # "immediate", "soon", "routine"


class WorkshopSelectionOutput(BaseModel):
    """Output from select_workshop_type()"""
    workshop_type: str
    services_required: List[str]
    estimated_cost_range: str
    estimated_time_hours: float


# =============================================================================
# Explanation Tools
# =============================================================================

class FaultSummaryOutput(BaseModel):
    """Output from summarize_fault_location()"""
    location_description: str  # "front-right wheel assembly"
    coordinates: str  # "(x=0.45m, y=-0.12m)"
    confidence_description: str  # "high confidence (92%)"


class CausalSummaryOutput(BaseModel):
    """Output from summarize_causal_reasoning()"""
    root_cause_description: str
    probability_description: str
    supporting_evidence: List[str]


class HumanReportOutput(BaseModel):
    """Output from generate_human_readable_report()"""
    report_text: str
    sections: Dict[str, str]  # {"summary": "...", "evidence": "...", "recommendation": "..."}
    word_count: int


# =============================================================================
# Visualization Tools
# =============================================================================

class VisualizationOutput(BaseModel):
    """Output from visualization tools"""
    image_path: str
    image_size: Tuple[int, int]
    format: str  # "png", "jpg"
    success: bool


# =============================================================================
# Data Management Tools
# =============================================================================

class DataSaveOutput(BaseModel):
    """Output from data saving tools"""
    file_path: str
    file_size_bytes: int
    format: str
    success: bool


class ManifestOutput(BaseModel):
    """Output from index_dataset_manifest()"""
    manifest_path: str
    total_runs: int
    fault_type_counts: Dict[str, int]
    updated: bool


# =============================================================================
# Validation Output
# =============================================================================

class SignalQualityOutput(BaseModel):
    """Output from validate_signal_quality()"""
    quality_score: float = Field(..., ge=0.0, le=1.0)
    snr_db: float
    issues: List[str]
    passed: bool
    metrics: Dict[str, float]
