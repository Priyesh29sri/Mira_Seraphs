"""
LangGraph State Schema

Defines the shared state structure passed between all 13 agents.
This is the single source of truth for the diagnostic pipeline.
"""

from typing import TypedDict, Optional, List, Dict, Any
import numpy as np
from datetime import datetime


class MIRAState(TypedDict, total=False):
    """
    Complete state for MIRA-Wave diagnostic pipeline.
    
    All agents read from and write to this shared state.
    TypedDict with total=False allows for partial state updates.
    """
    
    # ========================================================================
    # Run Identification & Metadata
    # ========================================================================
    run_id: str  # Unique identifier (e.g., "run_042", UUID)
    timestamp: datetime  # When the run was created
    
    # ========================================================================
    # Input Parameters (Simulation or Real Data)
    # ========================================================================
    fault_type: str  # "imbalance", "loose_mount", "bearing_wear"
    severity: float  # 0.0 (no fault) to 1.0 (critical)
    speed_kmh: float  # Vehicle speed in km/h
    load_kg: float  # Vehicle load in kg
    
    # ========================================================================
    # Raw Sensor Data (from Simulation Engine or Telemetry Ingest)
    # ========================================================================
    imu_data: np.ndarray  # Shape: (n_samples, 3) - [ax, ay, az] in m/s²
    audio_data: np.ndarray  # Shape: (n_samples,) - microphone signal
    imu_sampling_rate: int  # Typically 1000 Hz
    audio_sampling_rate: int  # Typically 44100 Hz
    
    # Metadata from sensors
    metadata: Dict[str, Any]  # device_id, geo_location, etc.
    
    # ========================================================================
    # Processed Signals (from Telemetry Ingest Agent)
    # ========================================================================
    imu_normalized: Optional[np.ndarray]  # Cleaned, resampled IMU
    audio_normalized: Optional[np.ndarray]  # Cleaned, resampled audio
    signal_quality_score: Optional[float]  # 0-1, based on SNR
    
    # ========================================================================
    # Physics Analysis Results (from Wavefield Physics Agent)
    # ========================================================================
    heatmap: Optional[np.ndarray]  # Shape: (128, 128) - 2D energy map
    fault_location: Optional[Dict[str, float]]  # {x_m, y_m, z_m, uncertainty_cm}
    modal_energies: Optional[np.ndarray]  # Energy in each mode
    dominant_frequency_hz: Optional[float]  # Peak frequency
    spectral_peaks: Optional[List[Dict[str, float]]]  # [{freq, amplitude}]
    
    # ========================================================================
    # Fingerprinting Results (from Fingerprinting Agent)
    # ========================================================================
    spectrogram: Optional[np.ndarray]  # Shape: (128, 128) - time-frequency
    fingerprint_vector: Optional[np.ndarray]  # Shape: (32,) - embedded features
    peak_frequencies: Optional[List[float]]  # Extracted peaks
    harmonic_ratios: Optional[List[float]]  # Harmonic structure
    
    # ========================================================================
    # Fleet Analysis (from Fleet Matching Agent)
    # ========================================================================
    fleet_matches: Optional[List[Dict[str, Any]]]  # Similar runs from database
    cluster_id: Optional[int]  # Which cluster this run belongs to
    similarity_scores: Optional[List[float]]  # Cosine similarity to matches
    cluster_centroid: Optional[np.ndarray]  # Center of cluster
    
    # ========================================================================
    # Causal Inference (from Causal Inference Agent)
    # ========================================================================
    causal_results: Optional[Dict[str, Any]]  # Root cause analysis
    # Structure: {
    #   "causes": [{"type": str, "probability": float, "confidence": float}],
    #   "correlations": {...},
    #   "treatment_effect": float
    # }
    
    # ========================================================================
    # Active Experimentation (from Active Experiment Agent)
    # ========================================================================
    experiment_plan: Optional[Dict[str, Any]]  # Suggested next test
    # Structure: {
    #   "instruction": str,
    #   "predicted_info_gain": float,
    #   "new_speed_kmh": float,
    #   "new_load_kg": float
    # }
    uncertainty_score: Optional[float]  # Current diagnostic uncertainty (entropy)
    
    # ========================================================================
    # Repair Scheduling (from Scheduler Agent)
    # ========================================================================
    repair_schedule: Optional[Dict[str, Any]]  # Repair plan
    # Structure: {
    #   "urgency": int (1-10),
    #   "workshop_type": str,
    #   "estimated_cost": str,
    #   "estimated_time": str,
    #   "priority": str
    # }
    
    # ========================================================================
    # Natural Language Explanation (from Explanation Agent)
    # ========================================================================
    explanation_text: Optional[str]  # Full human-readable report
    summary: Optional[str]  # One-line summary
    
    # ========================================================================
    # Orchestration & Debugging
    # ========================================================================
    messages: List[Dict[str, str]]  # Agent conversation history
    # Structure: [{"agent": str, "message": str, "timestamp": str}]
    
    current_agent: Optional[str]  # Which agent is currently executing
    workflow_stage: Optional[str]  # "simulation", "analysis", "reasoning", etc.
    errors: Optional[List[str]]  # Any errors encountered
    
    # ========================================================================
    # File Paths (for outputs)
    # ========================================================================
    output_dir: Optional[str]  # Where to save results
    heatmap_path: Optional[str]  # Path to saved heatmap image
    spectrogram_path: Optional[str]  # Path to saved spectrogram
    report_path: Optional[str]  # Path to saved text report
    
    # ========================================================================
    # Execution Metadata
    # ========================================================================
    total_execution_time_sec: Optional[float]  # End-to-end runtime
    agent_timings: Optional[Dict[str, float]]  # Time per agent


# Type hints for numpy arrays (for better IDE support)
NDArray = np.ndarray


def create_initial_state(
    run_id: str,
    fault_type: str,
    severity: float,
    speed_kmh: float,
    load_kg: float = 500.0,
) -> MIRAState:
    """
    Create initial state for a diagnostic run.
    
    Args:
        run_id: Unique run identifier
        fault_type: One of ["imbalance", "loose_mount", "bearing_wear"]
        severity: Fault severity (0.0 to 1.0)
        speed_kmh: Vehicle speed
        load_kg: Vehicle load
    
    Returns:
        Initialized MIRAState with input parameters
    """
    return MIRAState(
        run_id=run_id,
        timestamp=datetime.now(),
        fault_type=fault_type,
        severity=severity,
        speed_kmh=speed_kmh,
        load_kg=load_kg,
        messages=[],
        errors=[],
        workflow_stage="initialized",
    )
