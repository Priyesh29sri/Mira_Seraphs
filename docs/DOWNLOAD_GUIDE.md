# 📥 Direct Download Links for External Datasets

**Created**: All folders are ready in `/Users/priyeshsrivastava/MIRA/mira-wave/data/`

---

## ✅ Folder Structure (Created)

```
data/
├── external/              ← External datasets (download below)
│   ├── cwru_bearing/      
│   ├── nasa_bearing/      
│   └── kaggle_auto/       
├── simulated_runs/        ← Auto-generated (run generator)
├── fleet_database/        ← Auto-populated (run system)
├── diagnostic_runs/       ← Pipeline outputs
└── test_runs/             ← Test outputs
```

---

## 🔗 Download Links (Copy-Paste Ready)

### 1. CWRU Bearing Dataset ⭐⭐⭐ (BEST - Most Used)

**Website**: https://engineering.case.edu/bearingdatacenter

**Direct Downloads** (Choose 1-2):

| File | Fault Type | Severity | Direct Link |
|------|-----------|----------|-------------|
| `107.mat` | Inner Race | 0.007" | https://engineering.case.edu/bearingdatacenter/download-data-file?id=107 |
| `212.mat` | Outer Race (6:00) | 0.021" | https://engineering.case.edu/bearingdatacenter/download-data-file?id=212 |
| `122.mat` | Ball | 0.014" | https://engineering.case.edu/bearingdatacenter/download-data-file?id=122 |
| `97.mat` | Normal | None | https://engineering.case.edu/bearingdatacenter/download-data-file?id=97 |

**Download Instructions**:
1. Click link above (or go to website)
2. Download `.mat` files
3. Save to: `/Users/priyeshsrivastava/MIRA/mira-wave/data/external/cwru_bearing/`

**Alternative (if links don't work)**:
```bash
# Manual download from website
open https://engineering.case.edu/bearingdatacenter
# Click "Download Data Files" → Select files → Download
```

---

### 2. NASA Bearing Dataset ⭐⭐

**Website**: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

**Recommended Download**:
- **Dataset**: IMS Bearing Dataset
- **Link**: https://ti.arc.nasa.gov/c/6/
- **File**: `1st_test.zip` (~570 MB)

**Download Instructions**:
```bash
cd /Users/priyeshsrivastava/MIRA/mira-wave/data/external/nasa_bearing

# Option 1: Browser download
open https://ti.arc.nasa.gov/c/6/
# Download "1st_test.zip" manually

# Option 2: Command line (if direct link works)
curl -O https://ti.arc.nasa.gov/c/6/1st_test.zip
unzip 1st_test.zip
```

**What you'll get**:
- Folder: `1st_test/`
- Files: `2003.10.22.12.06.24` (timestamps)
- Format: CSV files
- Columns: [bearing1, bearing2, bearing3, bearing4]

---

### 3. Kaggle Datasets (Optional)

**Option A: Automotive Engine Sound Dataset**
- **Link**: https://www.kaggle.com/datasets/vineethakkinapalli/automobile-engine-sound-detection
- **Command**:
  ```bash
  kaggle datasets download -d vineethakkinapalli/automobile-engine-sound-detection
  unzip automobile-engine-sound-detection.zip -d data/external/kaggle_auto/
  ```

**Option B: Vehicle Fault Detection**
- **Search**: https://www.kaggle.com/search?q=vehicle+fault
- **Pick any CSV dataset**

**Setup Kaggle CLI** (if you want to use above):
```bash
pip install kaggle

# Get API key from: https://www.kaggle.com/account
# Download kaggle.json, then:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 🚀 Quick Download Script

Copy-paste this to download recommended datasets:

```bash
#!/bin/bash
cd /Users/priyeshsrivastava/MIRA/mira-wave/data/external

echo "📥 Downloading external datasets..."

# CWRU Bearing (manual - links may require login)
echo ""
echo "1. CWRU Bearing Dataset:"
echo "   Visit: https://engineering.case.edu/bearingdatacenter"
echo "   Download: 107.mat, 212.mat, 122.mat"
echo "   Save to: cwru_bearing/"
echo ""
echo "   Press Enter when done..."
read

# NASA Bearing
echo ""
echo "2. NASA Bearing Dataset:"
echo "   Opening download page..."
open https://ti.arc.nasa.gov/c/6/
echo "   Download: 1st_test.zip"
echo "   Save to: nasa_bearing/"
echo ""
echo "   Press Enter when done..."
read

echo "✅ Setup complete! Datasets should be in:"
echo "   - data/external/cwru_bearing/"
echo "   - data/external/nasa_bearing/"
```

Save as `download_datasets.sh`, then:
```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

---

## 📊 Generate Simulated Data (No Download Needed)

**Generate 10 synthetic runs right now**:

```bash
cd /Users/priyeshsrivastava/MIRA/mira-wave

# Generate 10 runs (3 fault types, mixed severities)
./venv/bin/python3 simulation/run_generator.py --num_runs 10 --output data/simulated_runs

# Check results
ls -lh data/simulated_runs/
```

**Output**:
```
data/simulated_runs/
├── run_001/
│   ├── imu.csv
│   ├── audio.wav
│   └── meta.json
├── run_002/
...
├── manifest.json
```

---

## 🗂️ Fleet Database (Auto-Generated)

Fleet database is **auto-created** when you run the system. It's simulated in memory.

To persist it:
```python
# agents/person_c/fleet_matching.py already has a simulated fleet
# To save it permanently:
import json

fleet_agent = FleetMatchingAgent()
with open('data/fleet_database/fleet.json', 'w') as f:
    json.dump(fleet_agent.fleet_db, f, indent=2, default=str)
```

--- 

## ✅ Verification Checklist

After downloading, verify:

```bash
cd /Users/priyeshsrivastava/MIRA/mira-wave

# Check CWRU
ls data/external/cwru_bearing/*.mat
# Should show: 107.mat, etc.

# Check NASA
ls data/external/nasa_bearing/1st_test/
# Should show: timestamp files

# Generate simulated data
./venv/bin/python3 simulation/run_generator.py --num_runs 5 --output data/simulated_runs

# Verify
ls data/simulated_runs/
# Should show: run_001/, run_002/, ..., manifest.json
```

---

## 🎯 Summary

| Dataset | Required? | Where to get | Save to |
|---------|-----------|--------------|---------|
| **Simulated** | ✅ Yes (use this!) | `run_generator.py` | `data/simulated_runs/` |
| **CWRU** | ⭐ Nice to have | https://engineering.case.edu/bearingdatacenter | `data/external/cwru_bearing/` |
| **NASA** | ⭐ Nice to have | https://ti.arc.nasa.gov/c/6/ | `data/external/nasa_bearing/` |
| **Kaggle** | ❌ Optional | https://kaggle.com/search?q=vehicle+fault | `data/external/kaggle_auto/` |
| **Fleet DB** | ✅ Auto-generated | N/A (in-memory) | `data/fleet_database/` |

**Bottom Line**: Generate simulated data now, download CWRU if you have 5 minutes!
