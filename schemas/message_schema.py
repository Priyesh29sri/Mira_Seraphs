"""
Inter-Agent Message Schemas

Defines structured messages passed between agents.
Each agent outputs a specific message type.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime


class AgentMessage(BaseModel):
    """Base class for all agent messages"""
    agent_name: str = Field(..., description="Name of the agent that produced this message")
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = Field(True, description="Whether the agent executed successfully")
    error_message: Optional[str] = Field(None, description="Error details if success=False")
    execution_time_sec: float = Field(..., description="Agent execution time")
    
    class Config:
        arbitrary_types_allowed = True  # Allow numpy arrays


# =============================================================================
# PERSON A: Simulation & Ingest
# =============================================================================

class SimulationOutput(AgentMessage):
    """Output from Simulation Engine Agent"""
    agent_name: str = "SimulationEngine"
    
    imu_data: Any  # np.ndarray, shape (n_samples, 3)
    audio_data: Any  # np.ndarray, shape (n_samples,)
    imu_sampling_rate: int = 1000
    audio_sampling_rate: int = 44100
    simulation_duration_sec: float
    
    # Ground truth (known in simulation)
    fault_type: str
    severity: float
    speed_kmh: float
    load_kg: float
    true_fault_location: Dict[str, float]  # {x_m, y_m, z_m}
    

class IngestOutput(AgentMessage):
    """Output from Telemetry Ingest Agent"""
    agent_name: str = "TelemetryIngest"
    
    imu_normalized: Any  # np.ndarray
    audio_normalized: Any  # np.ndarray
    signal_quality_score: float = Field(..., ge=0.0, le=1.0)
    snr_db: float
    issues_detected: List[str] = Field(default_factory=list)
    corrections_applied: List[str] = Field(default_factory=list)


class DataManagerOutput(AgentMessage):
    """Output from Data Manager Agent"""
    agent_name: str = "DataManager"
    
    run_folder: str
    files_saved: List[str]
    manifest_updated: bool


# =============================================================================
# PERSON B: Physics & Fingerprinting
# =============================================================================

class PhysicsOutput(AgentMessage):
    """Output from Wavefield Physics Agent"""
    agent_name: str = "WavefieldPhysics"
    
    heatmap: Any  # np.ndarray, shape (128, 128)
    fault_location: Dict[str, float]  # {x_m, y_m, z_m, uncertainty_cm, confidence}
    modal_energies: Any  # np.ndarray
    dominant_frequency_hz: float
    spectral_peaks: List[Dict[str, float]]  # [{freq_hz, amplitude_db}]
    localization_method: str  # "L1_inverse" or "L2_inverse"


class FingerprintOutput(AgentMessage):
    """Output from Fingerprinting Agent"""
    agent_name: str = "Fingerprinting"
    
    spectrogram: Any  # np.ndarray, shape (128, 128)
    fingerprint_vector: Any  # np.ndarray, shape (32,)
    peak_frequencies: List[float]
    harmonic_ratios: List[float]
    embedding_method: str  # "PCA+UMAP"


class HeatmapVisualizationOutput(AgentMessage):
    """Output from Heatmap Visualization Agent"""
    agent_name: str = "HeatmapVisualization"
    
    heatmap_image_path: str
    vehicle_diagram_used: str
    colormap: str = "jet"


# =============================================================================
# PERSON C: AI Reasoning
# =============================================================================

class FleetMatchOutput(AgentMessage):
    """Output from Fleet Matching Agent"""
    agent_name: str = "FleetMatching"
    
    matches: List[Dict[str, Any]]  # [{"run_id": str, "similarity": float, "fault_type": str}]
    cluster_id: int
    cluster_size: int
    similarity_scores: List[float]
    centroid_vector: Any  # np.ndarray


class CausalOutput(AgentMessage):
    """Output from Causal Inference Agent"""
    agent_name: str = "CausalInference"
    
    causes: List[Dict[str, Any]] = Field(default_factory=list)  # List of cause dicts
    top_cause: str = ""
    top_cause_probability: float = 0.0
    confidence_interval: Dict[str, float] = Field(default_factory=dict)
    correlations: Dict[str, float] = Field(default_factory=dict)
    treatment_effect: Optional[float] = None


class ExperimentOutput(AgentMessage):
    """Output from Active Experiment Agent"""
    agent_name: str = "ActiveExperiment"
    
    experiment_suggested: bool
    instruction: Optional[str] = None  # "Increase speed to 60 km/h"
    predicted_info_gain_bits: Optional[float] = None
    new_speed_kmh: Optional[float] = None
    new_load_kg: Optional[float] = None
    uncertainty_current: float
    uncertainty_threshold: float = 1.5  # bits


class ScheduleOutput(AgentMessage):
    """Output from Scheduler Agent"""
    agent_name: str = "Scheduler"
    
    urgency: int = Field(..., ge=1, le=10)
    workshop_type: str
    estimated_cost: str
    estimated_time: str
    priority: str  # "low", "medium", "high", "critical"


class ExplanationOutput(AgentMessage):
    """Output from Explanation Agent"""
    agent_name: str = "Explanation"
    
    full_report: str
    summary: str
    fault_location_description: str
    root_cause_description: str
    recommendation: str


# =============================================================================
# PERSON D: Orchestration
# =============================================================================

class OrchestratorOutput(AgentMessage):
    """Output from Master Orchestrator Agent"""
    agent_name: str = "MasterOrchestrator"
    
    workflow_completed: bool
    agents_executed: List[str]
    total_execution_time_sec: float
    agent_timings: Dict[str, float]
    final_state_summary: Dict[str, Any]


class UIAgentOutput(AgentMessage):
    """Output from UI Agent"""
    agent_name: str = "UIAgent"
    
    visualizations_generated: List[str]
    report_path: str
    dashboard_url: Optional[str] = None


# =============================================================================
# Utility Functions
# =============================================================================

def create_error_message(agent_name: str, error: Exception) -> Dict[str, Any]:
    """Create standardized error message"""
    return {
        "agent_name": agent_name,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error_message": str(error),
        "execution_time_sec": 0.0,
    }
