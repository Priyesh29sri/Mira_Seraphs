#!/bin/bash
# Automated Dataset Downloader for MIRA-Wave
# Downloads NASA, attempts CWRU (may require manual click), generates simulated data

set -e  # Exit on error

cd "$(dirname "$0")"
BASE_DIR="/Users/priyeshsrivastava/MIRA/mira-wave"

echo "🌊 MIRA-Wave Dataset Downloader"
echo "=" * 70

# 1. NASA Bearing Dataset
echo ""
echo "[1/3] Downloading NASA Bearing Dataset..."
cd "$BASE_DIR/data/external/nasa_bearing"

if [ -f "1st_test.zip" ]; then
    echo "✅ NASA dataset already downloaded"
else
    echo "📥 Downloading 1st_test.zip (~280KB)..."
    curl -L -o "1st_test.zip" "https://ti.arc.nasa.gov/c/6/"
    
    if [ -f "1st_test.zip" ]; then
        echo "📦 Extracting..."
        unzip -q 1st_test.zip
        echo "✅ NASA dataset downloaded and extracted"
        echo "   Files: $(ls -1 1st_test | wc -l) bearing data files"
    else
echo "❌ Download failed. Try manual download:"
        echo "   https://ti.arc.nasa.gov/c/6/"
    fi
fi

# 2. CWRU Bearing Dataset (requires manual download)
echo ""
echo "[2/3] CWRU Bearing Dataset..."
cd "$BASE_DIR/data/external/cwru_bearing"

if ls *.mat 1> /dev/null 2>&1; then
    echo "✅ CWRU files already present: $(ls -1 *.mat | wc -l) files"
else
    echo "⚠️  CWRU requires manual download"
    echo ""
    echo "   I'm opening the download page in your browser..."
    echo "   Please download these files:"
    echo "   - 107.mat (Inner Race Fault 0.007\")"
    echo "   - 212.mat (Outer Race Fault 0.021\")"
    echo "   - 122.mat (Ball Fault 0.014\")"
    echo "   - 97.mat (Normal Baseline)"
    echo ""
    echo "   Save them to: $BASE_DIR/data/external/cwru_bearing/"
    echo ""
    
    # Open browser to download page
    open "https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data"
    
    echo "   Press Enter when downloads are complete..."
    read
    
    if ls *.mat 1> /dev/null 2>&1; then
        echo "✅ Found $(ls -1 *.mat | wc -l) .mat files"
    else
        echo "⚠️  No .mat files found. You can download them later."
    fi
fi

# 3. Generate Simulated Data
echo ""
echo "[3/3] Generating Simulated Data..."
cd "$BASE_DIR"

if [ -d "data/simulated_runs" ] && [ "$(ls -A data/simulated_runs)" ]; then
    echo "✅ Simulated data already exists"
    echo "   Runs: $(ls -1d data/simulated_runs/run_* 2>/dev/null | wc -l)"
else
    echo "📊 Generating 10 synthetic fault runs..."
    ./venv/bin/python3 simulation/run_generator.py --num_runs 10 --output data/simulated_runs
    
    if [ -d "data/simulated_runs/run_001" ]; then
        echo "✅ Generated $(ls -1d data/simulated_runs/run_* | wc -l) runs"
    else
        echo "❌ Simulation failed. Check Python environment."
    fi
fi

# Summary
echo ""
echo "=" * 70
echo "📊 Dataset Summary"
echo "=" * 70

echo ""
echo "NASA Bearing:"
if [ -d "data/external/nasa_bearing/1st_test" ]; then
    nasa_count=$(ls -1 data/external/nasa_bearing/1st_test | wc -l)
    echo "  ✅ $nasa_count files in data/external/nasa_bearing/1st_test/"
else
    echo "  ❌ Not downloaded"
fi

echo ""
echo "CWRU Bearing:"
cwru_count=$(ls -1 data/external/cwru_bearing/*.mat 2>/dev/null | wc -l)
if [ $cwru_count -gt 0 ]; then
    echo "  ✅ $cwru_count .mat files in data/external/cwru_bearing/"
    ls -1 data/external/cwru_bearing/*.mat | sed 's/^/     - /'
else
    echo "  ⚠️  Manual download required"
    echo "     Visit: https://engineering.case.edu/bearingdatacenter"
fi

echo ""
echo "Simulated Data:"
sim_count=$(ls -1d data/simulated_runs/run_* 2>/dev/null | wc -l)
if [ $sim_count -gt 0 ]; then
    echo "  ✅ $sim_count runs in data/simulated_runs/"
else
    echo "  ❌ No simulated data"
fi

echo ""
echo "=" * 70
echo "✅ Setup Complete!"
echo ""
echo "Next steps:"
echo "  1. Test system: ./venv/bin/python3 quickstart_test.py"
echo "  2. Run full demo: ./venv/bin/python3 demo_complete.py"
echo "  3. View results in: outputs/"
