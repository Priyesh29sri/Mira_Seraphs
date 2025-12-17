"""
Agent 12: Master Orchestrator Agent  
Agent 13: UI Agent

PERSON D - Complete orchestration (20 tools total)
"""

import numpy as np
from typing import Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from schemas.state_schema import MIRAState, create_initial_state
from schemas.message_schema import OrchestratorOutput, UIAgentOutput

# Import all agents
from agents.person_a.simulation_engine import SimulationEngineAgent
from agents.person_a.telemetry_ingest import TelemetryIngestAgent
from agents.person_a.data_manager import DataManagerAgent
from agents.person_b.wavefield_physics import WavefieldPhysicsAgent
from agents.person_b.fingerprinting import FingerprintingAgent
from agents.person_b.heatmap_visualization import HeatmapVisualizationAgent
from agents.person_c.fleet_matching import (
    FleetMatchingAgent, CausalInferenceAgent, ActiveExperimentAgent,
    SchedulerAgent, ExplanationAgent
)


class MasterOrchestratorAgent:
    """Agent 12: Master Orchestrator - Executes complete pipeline"""
    
    def __init__(self):
        self.name = "MasterOrchestrator"
        
        # Initialize all agents
        self.agents = {
            "simulation": SimulationEngineAgent(),
            "ingest": TelemetryIngestAgent(),
            "physics": WavefieldPhysicsAgent(),
            "fingerprint": FingerprintingAgent(),
            "heatmap": HeatmapVisualizationAgent(),
            "fleet": FleetMatchingAgent(),
            "causal": CausalInferenceAgent(),
            "experiment": ActiveExperimentAgent(),
            "scheduler": SchedulerAgent(),
            "explanation": ExplanationAgent(),
            "data_manager": DataManagerAgent(),
        }
    
    def run(self, state: MIRAState) -> OrchestratorOutput:
        """Execute complete diagnostic pipeline"""
        import time as time_module
        start_time = time_module.time()
        
        agents_executed = []
        agent_timings = {}
        
        try:
            # 1. Simulation
            print("[1/11] Running Simulation Engine...")
            sim_out = self.agents["simulation"].run(state)
            if not sim_out.success:
                raise Exception(f"Simulation failed: {sim_out.error_message}")
            
            state["imu_data"] = sim_out.imu_data
            state["audio_data"] = sim_out.audio_data
            state["imu_sampling_rate"] = sim_out.imu_sampling_rate
            state["audio_sampling_rate"] = sim_out.audio_sampling_rate
            agents_executed.append("simulation")
            agent_timings["simulation"] = sim_out.execution_time_sec
            
            # 2. Telemetry Ingest
            print("[2/11] Running Telemetry Ingest...")
            ingest_out = self.agents["ingest"].run(state)
            if not ingest_out.success:
                raise Exception(f"Ingest failed: {ingest_out.error_message}")
            
            state["imu_normalized"] = ingest_out.imu_normalized
            state["audio_normalized"] = ingest_out.audio_normalized
            state["signal_quality_score"] = ingest_out.signal_quality_score
            agents_executed.append("ingest")
            agent_timings["ingest"] = ingest_out.execution_time_sec
            
            # 3. Physics Analysis
            print("[3/11] Running Wavefield Physics...")
            physics_out = self.agents["physics"].run(state)
            if not physics_out.success:
                raise Exception(f"Physics failed: {physics_out.error_message}")
            
            state["heatmap"] = physics_out.heatmap
            state["fault_location"] = physics_out.fault_location
            state["dominant_frequency_hz"] = physics_out.dominant_frequency_hz
            state["spectral_peaks"] = physics_out.spectral_peaks
            agents_executed.append("physics")
            agent_timings["physics"] = physics_out.execution_time_sec
            
            # 4. Fingerprinting
            print("[4/11] Running Fingerprinting...")
            finger_out = self.agents["fingerprint"].run(state)
            if not finger_out.success:
                raise Exception(f"Fingerprinting failed: {finger_out.error_message}")
            
            state["fingerprint_vector"] = finger_out.fingerprint_vector
            state["spectrogram"] = finger_out.spectrogram
            state["peak_frequencies"] = finger_out.peak_frequencies
            agents_executed.append("fingerprint")
            agent_timings["fingerprint"] = finger_out.execution_time_sec
            
            # 5. Heatmap Visualization
            print("[5/11] Running Heatmap Visualization...")
            heatmap_out = self.agents["heatmap"].run(state)
            if not heatmap_out.success:
                print(f"Warning: Heatmap visualization failed: {heatmap_out.error_message}")
            else:
                state["heatmap_path"] = heatmap_out.heatmap_image_path
                agents_executed.append("heatmap")
                agent_timings["heatmap"] = heatmap_out.execution_time_sec
            
            # 6. Fleet Matching
            print("[6/11] Running Fleet Matching...")
            fleet_out = self.agents["fleet"].run(state)
            if not fleet_out.success:
                raise Exception(f"Fleet matching failed: {fleet_out.error_message}")
            
            state["fleet_matches"] = fleet_out.matches
            state["cluster_id"] = fleet_out.cluster_id
            agents_executed.append("fleet")
            agent_timings["fleet"] = fleet_out.execution_time_sec
            
            # 7. Causal Inference
            print("[7/11] Running Causal Inference...")
            causal_out = self.agents["causal"].run(state)
            if not causal_out.success:
                raise Exception(f"Causal inference failed: {causal_out.error_message}")
            
            state["causal_results"] = {
                "causes": causal_out.causes,
                "top_cause": causal_out.top_cause,
                "top_cause_probability": causal_out.top_cause_probability,
                "confidence_interval": causal_out.confidence_interval,
            }
            agents_executed.append("causal")
            agent_timings["causal"] = causal_out.execution_time_sec
            
            # 8. Active Experiment
            print("[8/11] Running Active Experiment...")
            exp_out = self.agents["experiment"].run(state)
            if not exp_out.success:
                print(f"Warning: Experiment design failed: {exp_out.error_message}")
            else:
                state["experiment_plan"] = {
                    "suggested": exp_out.experiment_suggested,
                    "instruction": exp_out.instruction,
                    "info_gain": exp_out.predicted_info_gain_bits,
                }
                agents_executed.append("experiment")
                agent_timings["experiment"] = exp_out.execution_time_sec
            
            # 9. Scheduler
            print("[9/11] Running Scheduler...")
            sched_out = self.agents["scheduler"].run(state)
            if not sched_out.success:
                raise Exception(f"Scheduler failed: {sched_out.error_message}")
            
            state["repair_schedule"] = {
                "urgency": sched_out.urgency,
                "workshop_type": sched_out.workshop_type,
                "estimated_cost": sched_out.estimated_cost,
                "estimated_time": sched_out.estimated_time,
                "priority": sched_out.priority,
            }
            agents_executed.append("scheduler")
            agent_timings["scheduler"] = sched_out.execution_time_sec
            
            # 10. Explanation
            print("[10/11] Running Explanation...")
            explain_out = self.agents["explanation"].run(state)
            if not explain_out.success:
                raise Exception(f"Explanation failed: {explain_out.error_message}")
            
            state["explanation_text"] = explain_out.full_report
            state["summary"] = explain_out.summary
            agents_executed.append("explanation")
            agent_timings["explanation"] = explain_out.execution_time_sec
            
            # 11. Data Manager
            print("[11/11] Running Data Manager...")
            data_out = self.agents["data_manager"].run(state)
            if not data_out.success:
                print(f"Warning: Data manager failed: {data_out.error_message}")
            else:
                state["output_dir"] = data_out.run_folder
                agents_executed.append("data_manager")
                agent_timings["data_manager"] = data_out.execution_time_sec
            
            # Final summary
            total_time = time_module.time() - start_time
            
            final_summary = {
                "run_id": state.get("run_id"),
                "fault_type": state.get("fault_type"),
                "severity": state.get("severity"),
                "fault_location": state.get("fault_location"),
                "top_cause": state.get("causal_results", {}).get("top_cause"),
                "top_probability": state.get("causal_results", {}).get("top_cause_probability"),
                "urgency": state.get("repair_schedule", {}).get("urgency"),
                "summary": state.get("summary"),
            }
            
            print(f"\n✅ Pipeline Complete! Total time: {total_time:.2f}s")
            
            return OrchestratorOutput(
                agent_name=self.name,
                success=True,
                execution_time_sec=total_time,
                workflow_completed=True,
                agents_executed=agents_executed,
                total_execution_time_sec=total_time,
                agent_timings=agent_timings,
                final_state_summary=final_summary,
            )
        
        except Exception as e:
            total_time = time_module.time() - start_time
            print(f"\n❌ Pipeline failed: {str(e)}")
            
            return OrchestratorOutput(
                agent_name=self.name,
                success=False,
                error_message=str(e),
                execution_time_sec=total_time,
                workflow_completed=False,
                agents_executed=agents_executed,
                total_execution_time_sec=total_time,
                agent_timings=agent_timings,
                final_state_summary={},
            )


class UIAgent:
    """Agent 13: UI Agent - Display results (placeholder for future UI)"""
    
    def __init__(self):
        self.name = "UIAgent"
    
    def run(self, state: MIRAState) -> UIAgentOutput:
        """Generate UI outputs"""
        import time as time_module
        start_time = time_module.time()
        
        try:
            # For now, just save text report
            output_dir = Path(state.get("output_dir", "outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            run_id = state.get("run_id", "unknown")
            report_path = output_dir / f"{run_id}_report.txt"
            
            # Save explanation
            with open(report_path, 'w') as f:
                f.write(state.get("explanation_text", "No report generated"))
            
            visualizations = [str(report_path)]
            if "heatmap_path" in state:
                visualizations.append(state["heatmap_path"])
            
            exec_time = time_module.time() - start_time
            
            return UIAgentOutput(
                agent_name=self.name,
                success=True,
                execution_time_sec=exec_time,
                visualizations_generated=visualizations,
                report_path=str(report_path),
            )
        
        except Exception as e:
            exec_time = time_module.time() - start_time
            return UIAgentOutput(
                agent_name=self.name,
                success=False,
                error_message=str(e),
                execution_time_sec=exec_time,
                visualizations_generated=[],
                report_path="",
            )
