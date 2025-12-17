# 🤖 LLM Integration & External Data Guide

## 📍 Where LLMs are Used (Currently Minimal)

### Current State: **95% Deterministic, 5% LLM-Ready**

The system is built to support LLMs but currently uses **deterministic algorithms** for most tasks:

| Agent | Current Implementation | LLM Usage | Status |
|-------|----------------------|-----------|---------|
| **1-6** (Simulation, Physics) | Pure physics/math | ❌ None | Deterministic |
| **7-10** (Fleet, Causal, Experiment, Scheduler) | sklearn algorithms | ❌ None | Rule-based |
| **11** (Explanation) | Template-based | ✅ **CAN use LLM** | **Ready for Gemini** |

### Where to Add Gemini LLM (Optional Enhancement)

**Agent 11 (Explanation)** is the ONLY agent that would benefit from LLM:

```python
# Current: agents/person_c/fleet_matching.py (line ~400)
def generate_human_readable_report(self, state: MIRAState) -> str:
    # Template-based (no LLM)
    report = f"""MIRA-Wave Diagnostic Report
    ...
    """
    return report

# LLM-Enhanced Version (OPTIONAL):
def generate_human_readable_report_with_llm(self, state: MIRAState) -> str:
    import google.generativeai as genai
    
    # Configure Gemini
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
    
    # Build prompt from state
    prompt = f"""Generate a diagnostic report for:
    Fault: {state['fault_type']}
    Severity: {state['severity']}
    Location: {state['fault_location']}
    Root Cause: {state['causal_results']['top_cause']} ({state['causal_results']['top_cause_probability']:.0%})
    
    Make it professional, clear, and actionable."""
    
    # Generate with Gemini
    response = model.generate_content(prompt)
    return response.text
```

**Why LLMs aren't critical here**: Physics, math, and algorithms give more accurate results than LLMs for technical analysis. LLMs help with natural language explanations only.

---

## 📊 External Data Integration Guide

### Where to Put External Datasets

```
/Users/priyeshsrivastava/MIRA/mira-wave/
└── data/
    ├── simulated_runs/          # ✅ Auto-generated (working now)
    ├── external/                # 👈 PUT EXTERNAL DATA HERE
    │   ├── nasa_bearing/        # NASA bearing dataset
    │   ├── cwru_bearing/        # CWRU bearing dataset
    │   └── kaggle_auto/         # Kaggle automotive data
    └── fleet_database/          # Aggregated fingerprints
```

---

## 🔍 Recommended External Datasets

### 1. NASA Bearing Dataset ⭐ (Best for Validation)

**Download**:
```bash
cd /Users/priyeshsrivastava/MIRA/mira-wave/data/external
mkdir nasa_bearing && cd nasa_bearing

# Download from NASA
wget https://ti.arc.nasa.gov/c/6/
# Or manual: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
```

**What it contains**:
- Real bearing vibration data
- Known fault progression (healthy → failure)
- 20 kHz sampling rate
- 4 bearings with different fault modes

**How to use**:
```python
# Create adapter: tools/external_data_loaders.py
import pandas as pd

def load_nasa_bearing_run(file_path: str) -> np.ndarray:
    """Load NASA bearing data and convert to our format"""
    # NASA format: CSV with columns [timestamp, bearing1, bearing2, ...]
    df = pd.read_csv(file_path, header=None)
    
    # Extract bearing channel (usually column 4)
    vibration = df.iloc[:, 4].values
    
    # Downsample from 20kHz to 1kHz (our standard)
    from scipy.signal import resample
    target_samples = len(vibration) // 20
    resampled = resample(vibration, target_samples)
    
    return resampled

# Use in your pipeline
from tools.external_data_loaders import load_nasa_bearing_run

# Instead of simulation:
# imu_data = sim_agent.run(state).imu_data
# Use real data:
imu_data = load_nasa_bearing_run("data/external/nasa_bearing/2003.11.25.10.47.32")
```

### 2. CWRU Bearing Dataset ⭐⭐ (Most Popular)

**Download**:
```bash
cd data/external
mkdir cwru_bearing && cd cwru_bearing

# Download manually from: https://engineering.case.edu/bearingdatacenter
```

**What it contains**:
- Motor bearing data at different loads/speeds
- Ball, inner race, outer race faults
- Multiple fault sizes (0.007", 0.014", 0.021")
- 12 kHz and 48 kHz sampling rates

**How to use**:
```python
from scipy.io import loadmat

def load_cwru_bearing(mat_file: str) -> np.ndarray:
    """Load CWRU .mat file"""
    data = loadmat(mat_file)
    
    # CWRU uses different key names, find the vibration data
    for key in data.keys():
        if 'DE_time' in key:  # Drive End accelerometer
            vibration = data[key].flatten()
            break
    
    # Resample to 1kHz
    from scipy.signal import resample
    target_samples = len(vibration) // 12  # From 12kHz to 1kHz
    return resample(vibration, target_samples)
```

### 3. Kaggle Automotive Datasets (Less Structured)

**Search Kaggle for**:
- "automotive vibration"
- "vehicle fault detection"
- "car sensor data"

**Good options**:
```bash
# If you find a dataset, download via Kaggle CLI:
pip install kaggle

# Configure API key (get from kaggle.com/account)
mkdir ~/.kaggle
echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d <dataset-name> -p data/external/kaggle_auto/
```

---

## 🔧 Simple Integration Script

Create `tools/external_data_loaders.py`:

```python
"""
External Data Loaders - Converts external datasets to MIRA format
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
from pathlib import Path


def load_nasa_bearing(file_path: str, target_rate: int = 1000) -> dict:
    """
    Load NASA bearing dataset
    
    Args:
        file_path: Path to NASA CSV file
        target_rate: Target sampling rate (Hz)
    
    Returns:
        dict with 'imu_data', 'sampling_rate', 'metadata'
    """
    df = pd.read_csv(file_path, header=None)
    vibration = df.iloc[:, 4].values  # Bearing channel
    
    # Downsample from 20kHz to target
    target_samples = int(len(vibration) * target_rate / 20000)
    resampled = resample(vibration, target_samples)
    
    # Convert to 3-axis format (z-axis only)
    imu_3axis = np.zeros((len(resampled), 3))
    imu_3axis[:, 2] = resampled  # Z-axis
    
    return {
        "imu_data": imu_3axis,
        "sampling_rate": target_rate,
        "metadata": {
            "source": "NASA Bearing Dataset",
            "original_rate": 20000,
            "file": file_path,
        }
    }


def load_cwru_bearing(mat_file: str, target_rate: int = 1000) -> dict:
    """
    Load CWRU bearing .mat file
    
    Args:
        mat_file: Path to .mat file
        target_rate: Target sampling rate
    
    Returns:
        dict with 'imu_data', 'sampling_rate', 'metadata'
    """
    data = loadmat(mat_file)
    
    # Find vibration data key
    vibration = None
    for key in data.keys():
        if 'DE_time' in key:
            vibration = data[key].flatten()
            break
    
    if vibration is None:
        raise ValueError(f"Could not find vibration data in {mat_file}")
    
    # Downsample from 12kHz to target
    target_samples = int(len(vibration) * target_rate / 12000)
    resampled = resample(vibration, target_samples)
    
    # Convert to 3-axis
    imu_3axis = np.zeros((len(resampled), 3))
    imu_3axis[:, 2] = resampled
    
    return {
        "imu_data": imu_3axis,
        "sampling_rate": target_rate,
        "metadata": {
            "source": "CWRU Bearing Dataset",
            "original_rate": 12000,
            "file": mat_file,
        }
    }


def load_any_csv(csv_file: str, column: int = 0, target_rate: int = 1000) -> dict:
    """
    Generic CSV loader for Kaggle/other datasets
    
    Args:
        csv_file: Path to CSV
        column: Which column contains vibration data
        target_rate: Target sampling rate
    
    Returns:
        dict with 'imu_data', 'sampling_rate', 'metadata'
    """
    df = pd.read_csv(csv_file)
    
    if isinstance(column, str):
        vibration = df[column].values
    else:
        vibration = df.iloc[:, column].values
    
    # If too many samples, downsample to target rate
    if len(vibration) > 100000:
        vibration = resample(vibration, 100000)
    
    # Convert to 3-axis
    imu_3axis = np.zeros((len(vibration), 3))
    imu_3axis[:, 2] = vibration
    
    return {
        "imu_data": imu_3axis,
        "sampling_rate": target_rate,
        "metadata": {
            "source": "CSV File",
            "file": csv_file,
        }
    }
```

---

## 🎯 How to Use External Data (Step-by-Step)

### Option 1: Replace Simulation with Real Data

```python
# demo_with_real_data.py
from tools.external_data_loaders import load_nasa_bearing
from schemas.state_schema import create_initial_state
from agents.person_a.telemetry_ingest import TelemetryIngestAgent
from agents.person_b.wavefield_physics import WavefieldPhysicsAgent
# ... import other agents

# Load real data instead of simulating
real_data = load_nasa_bearing("data/external/nasa_bearing/2003.11.25.10.47.32")

# Create state with real data
state = create_initial_state(
    run_id="real_nasa_001",
    fault_type="bearing_wear",  # We know it's bearing fault
    severity=0.8,  # Estimate from NASA documentation
    speed_kmh=0,  # Unknown for NASA data
)

# Inject real IMU data
state["imu_data"] = real_data["imu_data"]
state["imu_sampling_rate"] = real_data["sampling_rate"]

# Skip simulation agent, go straight to ingest
ingest_agent = TelemetryIngestAgent()
ingest_out = ingest_agent.run(state)

# Continue with physics analysis
physics_agent = WavefieldPhysicsAgent()
physics_out = physics_agent.run(state)

print(f"Fault detected at: ({physics_out.fault_location['x_meters']:.2f}, {physics_out.fault_location['y_meters']:.2f})")
```

### Option 2: Validate Simulation Against Real Data

```python
# validation_script.py
from tools.external_data_loaders import load_cwru_bearing
from simulation import VehicleModel, create_fault
from agents.person_b.fingerprinting import FingerprintingAgent

# Generate simulated bearing fault
vehicle = VehicleModel()
fault = create_fault("bearing_wear", severity=0.6)
vehicle.add_fault_force(fault.get_force_function())
time, _, accel = vehicle.simulate(duration=10.0)

# Load real bearing data
real_data = load_cwru_bearing("data/external/cwru_bearing/107.mat")

# Extract fingerprints from both
finger_agent = FingerprintingAgent()

# Simulated
state_sim = {"imu_normalized": accel[:, 4]}  # Body acceleration
fp_sim = finger_agent.run(state_sim).fingerprint_vector

# Real
state_real = {"imu_normalized": real_data["imu_data"][:, 2]}
fp_real = finger_agent.run(state_real).fingerprint_vector

# Compare
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([fp_sim], [fp_real])[0][0]

print(f"Simulation vs Real data similarity: {similarity:.2%}")
print(f"✅ Validation {'PASSED' if similarity > 0.7 else 'FAILED'}")
```

---

## 🎓 Simplified Recommendation

**For your demo/hackathon**:

1. **Start with simulated data** (already working) ✅
2. **If judges ask about real data**, show:
   ```bash
   # Download one CWRU file manually
   # Place in: data/external/cwru_bearing/107.mat
   
   # Run validation
   python validation_script.py
   ```
3. **No need for LLM** - physics is more accurate than GPT for this task

**Why this approach**:
- ✅ Simulated data gives perfect ground truth
- ✅ Can generate unlimited test cases
- ✅ Real data is just validation, not essential
- ✅ Focus demo on physics + multi-agent architecture (more impressive!)

---

## 📦 Quick Setup for External Data (5 minutes)

```bash
cd /Users/priyeshsrivastava/MIRA/mira-wave

# Create external data folder
mkdir -p data/external/{nasa_bearing,cwru_bearing,kaggle_auto}

# Download CWRU (manual)
# Go to: https://engineering.case.edu/bearingdatacenter
# Download "107.mat" (Inner Race Fault, 0.007 inch)
# Save to: data/external/cwru_bearing/107.mat

# Create loader script
./venv/bin/python3 -c "
from scipy.io import loadmat
import numpy as np

# Test load
data = loadmat('data/external/cwru_bearing/107.mat')
print('✅ CWRU data loaded successfully')
print(f'Keys: {list(data.keys())}')
"
```

Done! You now have real data integrated. 🎉

---

## Bottom Line

**LLMs**: Not needed (optional for explanation only)  
**External Data**: Nice-to-have validation, not essential  
**Your System**: Already impressive with simulated data!

Focus on: Physics accuracy + Multi-agent orchestration + End-to-end demo
