"""
Complete End-to-End Demo - MIRA-Wave Full Pipeline

Runs all 13 agents in sequence to demonstrate complete fault diagnosis.
OPTIMIZED FOR 8GB RAM M1 MacBook
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from schemas.state_schema import create_initial_state
from agents.person_d.master_orchestrator import MasterOrchestratorAgent, UIAgent


def run_complete_demo():
    """Run complete MIRA-Wave diagnostic pipeline"""
    
    print("=" * 80)
    print("MIRA-Wave Complete Diagnostic Pipeline Demo")
    print("13 Agents | 80+ Tools | Physics-Based Fault Diagnosis")
    print("=" * 80)
    
    # Create initial state
    print("\n📋 Creating diagnostic request...")
    state = create_initial_state(
        run_id="demo_complete_001",
        fault_type="imbalance",
        severity=0.6,
        speed_kmh=40.0,
        load_kg=500.0,
    )
    
    print(f"   Run ID: {state['run_id']}")
    print(f"   Fault: {state['fault_type']} (severity={state['severity']})")
    print(f"   Speed: {state['speed_kmh']} km/h")
    
    # Run orchestrator
    print(f"\n🚀 Launching Master Orchestrator (11 agents)...")
    print("-" * 80)
    
    orchestrator = MasterOrchestratorAgent()
    result = orchestrator.run(state)
    
    print("-" * 80)
    
    if result.success:
        print(f"\n✅ Pipeline Complete!")
        print(f"\n📊 Execution Summary:")
        print(f"   Total time: {result.total_execution_time_sec:.2f} seconds")
        print(f"   Agents executed: {len(result.agents_executed)}")
        print(f"\n⏱️  Agent Timings:")
        for agent, time in result.agent_timings.items():
            print(f"   {agent:20s}: {time:6.2f}s")
        
        print(f"\n🎯 Diagnostic Results:")
        summary = result.final_state_summary
        print(f"   Fault detected: {summary.get('top_cause', 'unknown')}")
        print(f"   Confidence: {summary.get('top_probability', 0):.1%}")
        print(f"   Location: ({summary.get('fault_location', {}).get('x_meters', 0):.2f}, "
              f"{summary.get('fault_location', {}).get('y_meters', 0):.2f}) m")
        print(f"   Urgency: {summary.get('urgency', 0)}/10")
        
        # Run UI agent to generate final outputs
        print(f"\n📄 Generating final outputs...")
        ui_agent = UIAgent()
        ui_result = ui_agent.run(state)
        
        if ui_result.success:
            print(f"   Report saved: {ui_result.report_path}")
            print(f"   Visualizations: {len(ui_result.visualizations_generated)} files")
        
        print(f"\n📝 Summary:")
        print(f"   {summary.get('summary', 'No summary available')}")
        
        print("\n" + "=" * 80)
        print("✨ MIRA-Wave Demo Complete!")
        print("=" * 80)
        
        print("\n💡 Next Steps:")
        print("   - View detailed report in outputs/")
        print("   - Check heatmap visualization")
        print("   - Generate more runs with different faults")
        print("   - Build Streamlit UI for interactive exploration")
        
        return result
    
    else:
        print(f"\n❌ Pipeline Failed")
        print(f"   Error: {result.error_message}")
        print(f"   Agents completed: {len(result.agents_executed)}")
        
        return None


if __name__ == "__main__":
    result = run_complete_demo()
    
    if result:
        print(f"\n✅ SUCCESS - All systems operational")
        sys.exit(0)
    else:
        print(f"\n❌ FAILURE - Check errors above")
        sys.exit(1)
