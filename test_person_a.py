"""
Quick Test Script - PERSON A Pipeline

Tests the simulation → ingest → data management pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from schemas.state_schema import create_initial_state
from agents.person_a.simulation_engine import SimulationEngineAgent
from agents.person_a.telemetry_ingest import TelemetryIngestAgent
from agents.person_a.data_manager import DataManagerAgent


def test_person_a_pipeline():
    """Test complete PERSON A pipeline"""
    
    print("=" * 70)
    print("MIRA-Wave PERSON A Pipeline Test")
    print("=" * 70)
    
    # Create initial state
    print("\n[1] Creating initial state...")
    state = create_initial_state(
        run_id="test_demo_001",
        fault_type="imbalance",
        severity=0.6,
        speed_kmh=40.0,
        load_kg=500.0,
    )
    print(f"   Run ID: {state['run_id']}")
    print(f"   Fault: {state['fault_type']} (severity={state['severity']})")
    print(f"   Speed: {state['speed_kmh']} km/h")
    
    # Agent 1: Generate data
    print("\n[2] Running Simulation Engine Agent...")
    sim_agent = SimulationEngineAgent()
    sim_output = sim_agent.run(state)
    
    if sim_output.success:
        print(f"   ✅ Generated {len(sim_output.imu_data)} IMU samples")
        print(f"   ✅ Generated {len(sim_output.audio_data)} audio samples")
        print(f"   ✅ Execution time: {sim_output.execution_time_sec:.2f}s")
        print(f"   ✅ Ground truth location: ({sim_output.true_fault_location['x_m']:.2f}, "
              f"{sim_output.true_fault_location['y_m']:.2f}) m")
    else:
        print(f"   ❌ Error: {sim_output.error_message}")
        return
    
    # Update state
    state["imu_data"] = sim_output.imu_data
    state["audio_data"] = sim_output.audio_data
    state["imu_sampling_rate"] = sim_output.imu_sampling_rate
    state["audio_sampling_rate"] = sim_output.audio_sampling_rate
    
    # Agent 2: Clean data
    print("\n[3] Running Telemetry Ingest Agent...")
    ingest_agent = TelemetryIngestAgent()
    ingest_output = ingest_agent.run(state)
    
    if ingest_output.success:
        print(f"   ✅ Signal quality score: {ingest_output.signal_quality_score:.3f}")
        print(f"   ✅ SNR: {ingest_output.snr_db:.1f} dB")
        print(f"   ✅ Corrections applied: {len(ingest_output.corrections_applied)}")
        if ingest_output.issues_detected:
            print(f"   ⚠️  Issues: {', '.join(ingest_output.issues_detected)}")
        print(f"   ✅ Execution time: {ingest_output.execution_time_sec:.2f}s")
    else:
        print(f"   ❌ Error: {ingest_output.error_message}")
        return
    
    # Update state
    state["imu_normalized"] = ingest_output.imu_normalized
    state["audio_normalized"] = ingest_output.audio_normalized
    state["signal_quality_score"] = ingest_output.signal_quality_score
    
    # Agent 3: Save results
    print("\n[4] Running Data Manager Agent...")
    data_agent = DataManagerAgent(base_dir="data/test_runs")
    data_output = data_agent.run(state)
    
    if data_output.success:
        print(f"   ✅ Created run folder: {data_output.run_folder}")
        print(f"   ✅ Saved {len(data_output.files_saved)} files:")
        for file_path in data_output.files_saved:
            print(f"      - {Path(file_path).name}")
        print(f"   ✅ Manifest updated: {data_output.manifest_updated}")
        print(f"   ✅ Execution time: {data_output.execution_time_sec:.2f}s")
    else:
        print(f"   ❌ Error: {data_output.error_message}")
        return
    
    # Summary
    print("\n" + "=" * 70)
    print("PERSON A Pipeline Complete! ✅")
    print("=" * 70)
    print(f"\nTotal execution time: {sim_output.execution_time_sec + ingest_output.execution_time_sec + data_output.execution_time_sec:.2f}s")
    print(f"Output location: {data_output.run_folder}")
    print("\nNext steps:")
    print("  - Implement PERSON B agents (physics analysis)")
    print("  - Generate fault heatmap")
    print("  - Extract fingerprints")


if __name__ == "__main__":
    test_person_a_pipeline()
