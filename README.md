# 🌊 MIRA-Wave

> **Multi-Agent Resonance Intelligence & Vibration Analysis Engine**  
> Physics-driven automotive fault diagnosis using LangGraph multi-agent system

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)

---

## 🎯 What is MIRA-Wave?

MIRA-Wave is a revolutionary **agentic AI system** that diagnoses vehicle faults by combining:
- 🧮 **Physics simulation** (mass-spring-damper ODEs, modal analysis)
- 🤖 **13 specialized agents** (80+ micro-tools orchestrated via LangGraph)
- 🧠 **Causal AI reasoning** (fleet-level intelligence, active experimentation)
- 📊 **Explainable diagnostics** (natural language reports + visualizations)

**Unlike traditional ML**, MIRA-Wave doesn't just *detect* faults—it **localizes** them to specific coordinates, **explains** root causes using physics, and **self-experiments** to reduce uncertainty.

**All without any hardware.** Everything runs in simulation.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🎯 Fault Localization** | Maps faults to 2D vehicle coordinates with uncertainty quantification |
| **🔬 Physics-Based** | Uses modal analysis, Green's functions, L1/L2 inverse solvers |
| **🤖 13 Agents, 80+ Tools** | Modular multi-agent architecture via LangGraph |
| **🧠 Causal Reasoning** | Bayesian inference, treatment effects, confidence intervals |
| **🧪 Active Learning** | System proposes experiments to reduce diagnostic uncertainty |
| **📚 Fleet Intelligence** | Learns from simulated fleet via fingerprint matching |
| **📝 Explainable AI** | Natural language reports citing specific evidence |
| **♾️ Unlimited Data** | Auto-generates synthetic datasets with perfect ground truth |

---

## 🏗️ Architecture

### 13 Agents Organized into 4 Personas

```
PERSON A: Simulation & Ingest          PERSON B: Physics & Fingerprinting
├─ Agent 1: Simulation Engine (12)     ├─ Agent 4: Wavefield Physics (14)
├─ Agent 2: Telemetry Ingest (18)      ├─ Agent 5: Fingerprinting (11)
└─ Agent 3: Data Manager (6)           └─ Agent 6: Heatmap Visualization (5)

PERSON C: AI Reasoning                 PERSON D: Orchestration
├─ Agent 7: Fleet Matching (10)        ├─ Agent 12: Master Orchestrator (10)
├─ Agent 8: Causal Inference (8)       └─ Agent 13: UI Agent (10)
├─ Agent 9: Active Experiment (5)
├─ Agent 10: Scheduler (3)
└─ Agent 11: Explanation (5)
```

**Total**: 13 agents, 80+ tools, orchestrated via LangGraph StateGraph

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google Gemini API key ([Get it free](https://ai.google.dev/))

### Installation

```bash
# Clone repository
git clone https://github.com/priyesh/mira-wave.git
cd mira-wave

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env and add your GOOGLE_API_KEY
```

### Generate Simulated Data

```bash
# Generate 100 runs (3 fault types × varying severity/speed)
python simulation/run_generator.py --num_runs 100 --output data/simulated_runs/
```

Expected output:
```
Generating run 001/100: imbalance, severity=0.4, speed=40 km/h... ✓
Generating run 002/100: loose_mount, severity=0.6, speed=50 km/h... ✓
...
Generated 100 runs in data/simulated_runs/
```

### Run Diagnostic Pipeline

```bash
# Diagnose a single run
python examples/single_run_demo.py --run_id run_042

# Or specify fault parameters
python examples/single_run_demo.py --fault imbalance --severity 0.6 --speed 40
```

**Output**:
```
🔬 MIRA-Wave Diagnostic Pipeline
================================

[1/9] Simulation Engine: Generating fault data... ✓
[2/9] Telemetry Ingest: Normalizing signals... ✓
[3/9] Wavefield Physics: Localizing fault... ✓
      📍 Location: (x=0.45m, y=-0.12m) ± 3cm
[4/9] Fingerprinting: Extracting features... ✓
[5/9] Fleet Matching: Finding similar cases... ✓
      🔗 Matched 12 similar runs (cluster ID: 3)
[6/9] Causal Inference: Determining root cause... ✓
      📊 Wheel imbalance: 78% confidence
[7/9] Active Experiment: Designing test... ✓
      🧪 Suggest: "Increase speed to 60 km/h"
[8/9] Scheduler: Planning repair... ✓
      🔧 Wheel balancing, urgency: 7/10
[9/9] Explanation: Generating report... ✓

📄 Report saved to: outputs/run_042_report.txt
📊 Heatmap saved to: outputs/run_042_heatmap.png
🎵 Spectrogram saved to: outputs/run_042_spectrogram.png
```

---

## 📊 Example Output

### Input
```json
{
  "fault_type": "imbalance",
  "severity": 0.6,
  "speed_kmh": 40,
  "load_kg": 500
}
```

### Generated Outputs

**1. Fault Heatmap**  
![Heatmap Example](docs/images/heatmap_example.png)

**2. Fingerprint Spectrogram**  
![Spectrogram Example](docs/images/spectrogram_example.png)

**3. Natural Language Report**
```
The vehicle exhibits a wheel imbalance fault with severity 0.6 
located at the front-right wheel assembly (x=0.45m, y=-0.12m).

Physical Evidence:
- Dominant frequency: 13.3 Hz (1× rotation order at 40 km/h)
- Vibration amplitude: 2.4 m/s² RMS (3× baseline)
- Spectral peak: 35 dB at 13.3 Hz

Fleet Comparison:
This pattern matches 87% of known wheel imbalance cases from 
the fleet database (N=45 similar vehicles).

Root Cause Analysis:
Bayesian causal inference assigns 78% confidence to wheel imbalance
based on feature correlation (0.83) and cluster membership.

Recommended Action:
Wheel balancing at alignment shop.
Estimated cost: $50-$80
Estimated time: 1 hour
Urgency: 7/10

Active Experiment Suggestion:
To confirm diagnosis, run test: "Increase speed to 60 km/h for 30s"
Expected information gain: 2.3 bits
```

**4. Structured JSON Output**
```json
{
  "fault_location": {"x": 0.45, "y": -0.12, "z": 0.05, "uncertainty_cm": 3.0},
  "causal_results": {
    "causes": [
      {"type": "wheel_imbalance", "probability": 0.78, "confidence": 0.92},
      {"type": "tire_wear", "probability": 0.15, "confidence": 0.65}
    ]
  },
  "repair_schedule": {
    "urgency": 7,
    "workshop_type": "wheel_balancing",
    "estimated_cost": "$50-$80"
  }
}
```

---

## 🧪 Physics Behind the System

### Vehicle Model

6-DOF mass-spring-damper system:
$$M\ddot{x} + C\dot{x} + Kx = F_{external} + F_{fault}$$

**Components**:
- 4 wheels (unsprung mass)
- 1 vehicle body (sprung mass)
- 1 engine (mounted to body)

### Fault Types Simulated

| Fault | Physics Mechanism | Frequency Signature |
|-------|------------------|---------------------|
| **Imbalance** | Centrifugal force $F = me\omega^2$ | 1× rotation frequency |
| **Loose Mount** | Bilinear stiffness + backlash | Harmonics (1×, 2×, 3×) + impacts |
| **Bearing Wear** | Hertzian contact + spall impacts | BPFO, BPFI, BSF frequencies |

### Localization Method

1. **Modal Decomposition**: $x(t) = \sum \phi_i q_i(t)$
2. **Inverse Problem**: Solve $G \mathbf{f} = \mathbf{x}$ for fault source $\mathbf{f}$
3. **L1 Regularization**: Promote sparse localization
4. **2D Heatmap**: Energy distribution across vehicle

---

## 📂 Project Structure

```
mira-wave/
├── agents/                  # 13 agent implementations
│   ├── person_a/           # Simulation & ingest
│   ├── person_b/           # Physics & fingerprinting
│   ├── person_c/           # AI reasoning
│   └── person_d/           # Orchestration
├── langgraph/              # LangGraph state & graph
├── simulation/             # Physics simulation
├── tools/                  # Shared utilities
├── schemas/                # Pydantic models
├── data/                   # Generated datasets
├── docs/                   # Documentation
├── tests/                  # Unit & integration tests
├── examples/               # Demo scripts
├── sill.md                 # Anti-hallucination rules
└── requirements.txt        # Dependencies
```

---

## 🧑‍💻 Usage Examples

### Generate Custom Fault Scenario

```python
from simulation.vehicle_model import VehicleModel
from simulation.fault_models import ImbalanceFault

# Create vehicle
vehicle = VehicleModel()

# Add imbalance fault
fault = ImbalanceFault(severity=0.6, location=(0.45, -0.12))
vehicle.add_fault(fault)

# Simulate
imu_data, audio_data = vehicle.simulate(duration=10.0, speed_kmh=40)
```

### Run Physics Analysis

```python
from agents.person_b.wavefield_physics import WavefieldPhysicsAgent

agent = WavefieldPhysicsAgent()
result = agent.analyze(imu_data)

print(f"Fault location: ({result['x_meters']:.2f}, {result['y_meters']:.2f})")
print(f"Confidence: {result['confidence']:.1%}")
```

### Use LangGraph Pipeline

```python
from langgraph.executor import run_diagnostic_pipeline

state = {
    "fault_type": "imbalance",
    "severity": 0.6,
    "speed_kmh": 40
}

final_state = run_diagnostic_pipeline(state)
print(final_state["explanation_text"])
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_simulation.py -v

# Run with coverage
pytest tests/ --cov=agents --cov=langgraph --cov=simulation
```

---

## 📚 Documentation

- **[Architecture](docs/ARCHITECTURE.md)**: System design & agent interaction
- **[Data Flow](docs/DATA_FLOW.md)**: Step-by-step data transformation
- **[Physics](docs/PHYSICS.md)**: Mathematical details & algorithms
- **[API](docs/API.md)**: Tool & agent API reference
- **[Demo](docs/DEMO.md)**: Example runs & visualizations

---

## 🎓 Real-World Application Path

### Current: Simulation (Phase 1)
- ✅ Fully synthetic data
- ✅ Perfect ground truth
- ✅ Unlimited scenarios

### Next: Validation (Phase 2)
- Compare with NASA/CWRU bearing datasets
- Tune physics models against real measurements

### Future: Real Vehicles (Phase 3)
1. **Data Acquisition**: OBD-II + IMU sensors
2. **CAN Bus Integration**: Parse vehicle messages
3. **Model Calibration**: Vehicle-specific parameters
4. **Fleet Deployment**: Cloud-based orchestration
5. **Edge Computing**: Run on vehicle ECUs

---

## 🏆 Why MIRA-Wave is Novel

| MIRA-Wave | Traditional ML | OEM Diagnostics |
|-----------|----------------|-----------------|
| ✅ Localizes to coordinates | ❌ Class only | ⚠️ Sensor-specific |
| ✅ Physics-based explanations | ❌ Black box | ⚠️ Rule-based |
| ✅ Self-generated data | ❌ Large labeled sets | ✅ Fleet data |
| ✅ Active experimentation | ❌ Static predictions | ⚠️ Manual tests |
| ✅ Open-source | ⚠️ Research code | 🔒 Proprietary |

---

## 📈 Roadmap

### v0.1 (Current)
- [x] 13 agents implemented
- [x] Physics simulation working
- [x] LangGraph orchestration
- [x] Batch data generation

### v0.2 (Next)
- [ ] UI dashboard (Streamlit)
- [ ] Real-time monitoring
- [ ] External dataset integration
- [ ] Performance optimization

### v1.0 (Future)
- [ ] 3D visualization
- [ ] More fault types (cracks, gears, belts)
- [ ] Reinforcement learning for experiments
- [ ] Federated learning

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for Contribution**:
- Additional fault models (gear mesh, belt slip, etc.)
- Integration with real vehicle datasets
- UI/UX improvements
- Performance optimizations

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Project Lead**: Priyesh Srivastava  
**GitHub**: [github.com/priyesh/mira-wave](https://github.com/priyesh/mira-wave)  
**Email**: priyesh@example.com

---

## 🙏 Acknowledgments

- **LangGraph** team for multi-agent framework
- **NASA** for bearing fault databases
- **CWRU** for vibration datasets
- Physics community for modal analysis methods

---

## 📊 Citation

If you use MIRA-Wave in your research, please cite:

```bibtex
@software{mira_wave_2025,
  author = {Srivastava, Priyesh},
  title = {MIRA-Wave: Multi-Agent Resonance Intelligence for Vehicle Diagnostics},
  year = {2025},
  url = {https://github.com/priyesh/mira-wave}
}
```

---

**Built with ❤️ using LangGraph, physics, and agentic AI**

*Welcome to the future of automotive diagnostics.* 🌊
