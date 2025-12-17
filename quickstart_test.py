#!/usr/bin/env python3
"""
Quick Start Guide - Test MIRA-Wave in 3 Simple Steps
Run this to verify everything works!
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("🌊 MIRA-Wave Quick Start Test")
print("=" * 80)


def test_step_1_simple_simulation():
    """Step 1: Test basic simulation (fastest, ~2 seconds)"""
    print("\n[Step 1/3] Testing Basic Simulation...")
    print("-" * 80)
    
    from schemas.state_schema import create_initial_state
    from agents.person_a.simulation_engine import SimulationEngineAgent
    
    state = create_initial_state(
        run_id="quickstart_001",
        fault_type="imbalance",
        severity=0.6,
        speed_kmh=40.0,
    )
    
    sim = SimulationEngineAgent()
    result = sim.run(state)
    
    if result.success:
        print(f"✅ Simulation PASSED")
        print(f"   Generated: {len(result.imu_data)} IMU samples")
        print(f"   Time: {result.execution_time_sec:.2f}s")
        return True
    else:
        print(f"❌ Simulation FAILED: {result.error_message}")
        return False


def test_step_2_three_agents():
    """Step 2: Test PERSON A pipeline (simulation → ingest → data manager)"""
    print("\n[Step 2/3] Testing 3-Agent Pipeline...")
    print("-" * 80)
    
    from schemas.state_schema import create_initial_state
    from agents.person_a.simulation_engine import SimulationEngineAgent
    from agents.person_a.telemetry_ingest import TelemetryIngestAgent
    from agents.person_a.data_manager import DataManagerAgent
    
    # Create request
    state = create_initial_state(
        run_id="quickstart_002",
        fault_type="bearing_wear",
        severity=0.7,
        speed_kmh=50.0,
    )
    
    # Agent 1: Simulate
    sim = SimulationEngineAgent()
    sim_out = sim.run(state)
    if not sim_out.success:
        print(f"❌ Agent 1 failed: {sim_out.error_message}")
        return False
    
    state["imu_data"] = sim_out.imu_data
    state["audio_data"] = sim_out.audio_data
    state["imu_sampling_rate"] = sim_out.imu_sampling_rate
    state["audio_sampling_rate"] = sim_out.audio_sampling_rate
    
    # Agent 2: Ingest
    ingest = TelemetryIngestAgent()
    ingest_out = ingest.run(state)
    if not ingest_out.success:
        print(f"❌ Agent 2 failed: {ingest_out.error_message}")
        return False
    
    state["imu_normalized"] = ingest_out.imu_normalized
    state["audio_normalized"] = ingest_out.audio_normalized
    
    # Agent 3: Data Manager
    data_mgr = DataManagerAgent(base_dir="data/quickstart_runs")
    data_out = data_mgr.run(state)
    if not data_out.success:
        print(f"❌ Agent 3 failed: {data_out.error_message}")
        return False
    
    print(f"✅ 3-Agent Pipeline PASSED")
    print(f"   Files saved: {len(data_out.files_saved)}")
    print(f"   Output: {data_out.run_folder}")
    print(f"   Total time: {sim_out.execution_time_sec + ingest_out.execution_time_sec + data_out.execution_time_sec:.2f}s")
    
    return True


def test_step_3_full_pipeline():
    """Step 3: Test complete system (all 13 agents)"""
    print("\n[Step 3/3] Testing Complete 13-Agent Pipeline...")  
    print("-" * 80)
    print("This will take ~15-20 seconds...")
    
    from schemas.state_schema import create_initial_state
    from agents.person_d.master_orchestrator import MasterOrchestratorAgent
    
    # Create diagnostic request
    state = create_initial_state(
        run_id="quickstart_complete",
        fault_type="imbalance",
        severity=0.6,
        speed_kmh=40.0,
    )
    
    # Run orchestrator
    orchestrator = MasterOrchestratorAgent()
    result = orchestrator.run(state)
    
    if result.success:
        summary = result.final_state_summary
        print(f"✅ Full Pipeline PASSED")
        print(f"   Agents executed: {len(result.agents_executed)}/11")
        print(f"   Total time: {result.total_execution_time_sec:.2f}s")
        print(f"   Fault detected: {summary.get('top_cause', 'unknown')}")
        print(f"   Confidence: {summary.get('top_probability', 0):.0%}")
        print(f"   Urgency: {summary.get('urgency')}/10")
        return True
    else:
        print(f"❌ Pipeline FAILED: {result.error_message}")
        print(f"   Completed agents: {result.agents_executed}")
        return False


def main():
    """Run all tests"""
    
    # Test 1: Basic simulation
    test1_pass = test_step_1_simple_simulation()
    
    if not test1_pass:
        print("\n❌ Basic test failed. Check your setup.")
        return False
    
    # Test 2: Three agents
    test2_pass = test_step_2_three_agents()
    
    if not test2_pass:
        print("\n❌ 3-agent test failed. Check agent implementations.")
        return False
    
    # Test 3: Full pipeline
    test3_pass = test_step_3_full_pipeline()
    
    # Final summary
    print("\n" + "=" * 80)
    if test1_pass and test2_pass and test3_pass:
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\n✅ Your MIRA-Wave system is working perfectly!")
        print("\n📋 Next steps:")
        print("   1. Run full demo: ./venv/bin/python3 demo_complete.py")
        print("   2. Generate more data: ./venv/bin/python3 simulation/run_generator.py --num_runs 10")
        print("   3. (Optional) Add external data: See EXTERNAL_DATA_GUIDE.md")
        print("\n💡 For presentation:")
        print("   - Show demo_complete.py output")
        print("   - Show generated heatmap in outputs/")
        print("   - Explain physics (6-DOF ODE, L1 inverse)")
        print("   - Highlight 13 agents, 80+ tools")
        
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        print(f"\nResults: Test 1: {'✅' if test1_pass else '❌'} | Test 2: {'✅' if test2_pass else '❌'} | Test 3: {'✅' if test3_pass else '❌'}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
