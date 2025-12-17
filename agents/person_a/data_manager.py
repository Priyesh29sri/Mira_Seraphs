"""
Agent 3: Data Manager Agent

Manages files, folders, and dataset organization.
6 tools for data persistence and manifest management.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from schemas.state_schema import MIRAState
from schemas.message_schema import DataManagerOutput


class DataManagerAgent:
    """
    Agent 3: Data Manager
    
    Responsibility: Organize diagnostic data into structured folders with manifest.
    
    6 Tools:
    - create_run_folder()
    - save_imu_csv()
    - save_audio_wav()
    - save_meta_json()
    - generate_run_id()
    - index_dataset_manifest()
    """
    
    def __init__(self, base_dir: str = "data/diagnostic_runs"):
        self.name = "DataManager"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_run_id(self, prefix: str = "diag") -> str:
        """Tool 1: Generate unique run ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"
    
    def create_run_folder(self, run_id: str) -> Path:
        """Tool 2: Create folder for run"""
        run_path = self.base_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        return run_path
    
    def save_imu_csv(
        self,
        imu_data,  # np.ndarray
        run_path: Path,
        filename: str = "imu_processed.csv",
    ) -> str:
        """Tool 3: Save IMU data to CSV"""
        import pandas as pd
        import numpy as np
        
        filepath = run_path / filename
        
        # Create time vector
        n_samples = len(imu_data)
        time = np.arange(n_samples) / 1000.0  # Assume 1kHz
        
        # Create dataframe
        if imu_data.ndim == 2:
            df = pd.DataFrame({
                "time_sec": time,
                "ax_m_s2": imu_data[:, 0],
                "ay_m_s2": imu_data[:, 1],
                "az_m_s2": imu_data[:, 2],
            })
        else:
            df = pd.DataFrame({
                "time_sec": time,
                "az_m_s2": imu_data,
            })
        
        df.to_csv(filepath, index=False)
        return str(filepath)
    
    def save_audio_wav(
        self,
        audio_data,  # np.ndarray
        run_path: Path,
        sampling_rate: int = 44100,
        filename: str = "audio_processed.wav",
    ) -> str:
        """Tool 4: Save audio to WAV"""
        import soundfile as sf
        
        filepath = run_path / filename
        sf.write(str(filepath), audio_data, sampling_rate)
        return str(filepath)
    
    def save_meta_json(
        self,
        metadata: Dict[str, Any],
        run_path: Path,
        filename: str = "results.json",
    ) -> str:
        """Tool 5: Save metadata JSON"""
        filepath = run_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)  # default=str handles datetime
        
        return str(filepath)
    
    def index_dataset_manifest(
        self,
        run_metadata: Dict[str, Any],
        manifest_path: str = None,
    ) -> Dict[str, Any]:
        """Tool 6: Update dataset manifest"""
        if manifest_path is None:
            manifest_path = self.base_dir / "manifest.json"
        else:
            manifest_path = Path(manifest_path)
        
        # Load existing manifest
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {
                "dataset_name": "MIRA-Wave Diagnostic Runs",
                "created": datetime.now().isoformat(),
                "runs": [],
            }
        
        # Add new run
        manifest["runs"].append(run_metadata)
        manifest["total_runs"] = len(manifest["runs"])
        manifest["last_updated"] = datetime.now().isoformat()
        
        # Save manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        return {
            "manifest_path": str(manifest_path),
            "total_runs": manifest["total_runs"],
            "updated": True,
        }
    
    # =========================================================================
    # Agent Execution
    # =========================================================================
    
    def run(self, state: MIRAState) -> DataManagerOutput:
        """
        Execute data manager agent
        
        Args:
            state: Complete state with all diagnostic results
        
        Returns:
            DataManagerOutput with saved file paths
        """
        import time as time_module
        start_time = time_module.time()
        
        try:
            # Generate run ID if not present
            run_id = state.get("run_id")
            if run_id is None:
                run_id = self.generate_run_id()
            
            # Create run folder
            run_folder = self.create_run_folder(run_id)
            
            files_saved = []
            
            # Save IMU if available
            if state.get("imu_normalized") is not None:
                imu_path = self.save_imu_csv(state["imu_normalized"], run_folder)
                files_saved.append(imu_path)
            
            # Save audio if available
            if state.get("audio_normalized") is not None:
                audio_path = self.save_audio_wav(
                    state["audio_normalized"],
                    run_folder,
                    sampling_rate=state.get("audio_sampling_rate", 44100),
                )
                files_saved.append(audio_path)
            
            # Compile results metadata
            results_meta = {
                "run_id": run_id,
                "timestamp": state.get("timestamp", datetime.now()).isoformat(),
                "input_parameters": {
                    "fault_type": state.get("fault_type"),
                    "severity": state.get("severity"),
                    "speed_kmh": state.get("speed_kmh"),
                    "load_kg": state.get("load_kg"),
                },
                "fault_location": state.get("fault_location"),
                "causal_results": state.get("causal_results"),
                "repair_schedule": state.get("repair_schedule"),
                "explanation_summary": state.get("summary"),
            }
            
            # Save results
            results_path = self.save_meta_json(results_meta, run_folder)
            files_saved.append(results_path)
            
            # Update manifest
            manifest_result = self.index_dataset_manifest(results_meta)
            
            exec_time = time_module.time() - start_time
            
            return DataManagerOutput(
                agent_name=self.name,
                success=True,
                execution_time_sec=exec_time,
                run_folder=str(run_folder),
                files_saved=files_saved,
                manifest_updated=manifest_result["updated"],
            )
        
        except Exception as e:
            exec_time = time_module.time() - start_time
            return DataManagerOutput(
                agent_name=self.name,
                success=False,
                error_message=str(e),
                execution_time_sec=exec_time,
                run_folder="",
                files_saved=[],
                manifest_updated=False,
            )
