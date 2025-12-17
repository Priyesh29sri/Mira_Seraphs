"""
Batch Run Generator

Generates a dataset of simulated vehicle fault runs.
Creates combinations of:
- Fault types (imbalance, loose_mount, bearing_wear)
- Severities (0.2, 0.4, 0.6, 0.8, 1.0)
- Speeds (20, 30, 40, 50, 60 km/h)
"""

import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from simulation.vehicle_model import VehicleModel, VehicleParameters
from simulation.fault_models import create_fault
from simulation.sensor_simulation import (
    simulate_imu_signal,
    simulate_audio_signal,
    save_imu_to_csv,
    save_audio_to_wav,
)


def generate_single_run(
    run_id: str,
    fault_type: str,
    severity: float,
    speed_kmh: float,
    load_kg: float,
    output_dir: str,
    duration: float = 10.0,
    verbose: bool = False,
) -> dict:
    """
    Generate a single diagnostic run
    
    Args:
        run_id: Unique run identifier (e.g., "run_001")
        fault_type: "imbalance", "loose_mount", or "bearing_wear"
        severity: 0.0 to 1.0
        speed_kmh: Vehicle speed
        load_kg: Vehicle load
        output_dir: Where to save files
        duration: Simulation duration (seconds)
        verbose: Print progress
    
    Returns:
        Metadata dict
    """
    if verbose:
        print(f"Generating {run_id}: {fault_type}, severity={severity:.1f}, speed={speed_kmh} km/h")
    
    # Create output directory
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create vehicle model
    vehicle = VehicleModel()
    vehicle.set_speed_profile(speed_kmh, ramp_time=2.0)
    
    # Create and add fault WITH RANDOMIZATION
    fault_params = {}
    
    # Randomize wheel position (0=FL, 1=FR, 2=RL, 3=RR)
    wheel_index = np.random.randint(0, 4)
    
    if fault_type == "imbalance":
        fault_params = {
            "wheel_index": wheel_index,
        }
    elif fault_type == "bearing_wear":
        # Randomize bearing defect type
        defect_types = ["inner_race", "outer_race", "ball"]
        fault_params = {
            "wheel_index": wheel_index,
            "defect_type": np.random.choice(defect_types),
        }
    elif fault_type == "loose_mount":
        # Loose mount can be at engine (4) or any wheel
        mount_location = np.random.choice([0, 1, 2, 3, 4])
        fault_params = {
            "location_index": mount_location,
        }
    
    fault = create_fault(fault_type, severity, **fault_params)
    vehicle.add_fault_force(fault.get_force_function())
    
    # Simulate vehicle dynamics
    time, displacement, acceleration = vehicle.simulate(duration=duration, dt=0.001)
    
    # Generate sensor signals WITH VARIED NOISE
    # Add more noise variation to create distinct audio
    noise_variation = np.random.uniform(0.03, 0.08)
    
    imu_data, imu_rate = simulate_imu_signal(
        acceleration,
        time,
        sensor_location=4,  # Body
        noise_level=noise_variation,
        sampling_rate=1000,
    )
    
    audio_data, audio_rate = simulate_audio_signal(
        acceleration,
        time,
        microphone_location=5,  # Engine
        noise_level=noise_variation * 0.5,  # Less noise in audio
        sampling_rate=44100,
    )
    
    # Add fault-specific audio characteristics
    if fault_type == "bearing_wear":
        # Bearing creates high-frequency clicks
        t_audio = np.arange(len(audio_data)) / audio_rate
        clicks = np.sin(2 * np.pi * 800 * t_audio) * severity * 0.1
        audio_data = audio_data + clicks
    elif fault_type == "loose_mount":
        # Loose mount creates impact sounds (lower frequency)
        t_audio = np.arange(len(audio_data)) / audio_rate
        impacts = np.sin(2 * np.pi * 120 * t_audio) * severity * 0.2
        audio_data = audio_data + impacts
    
    # Save sensor data
    imu_path = run_dir / "imu.csv"
    audio_path = run_dir / "audio.wav"
    
    imu_time = np.arange(len(imu_data)) / imu_rate
    save_imu_to_csv(imu_data, imu_time, str(imu_path))
    save_audio_to_wav(audio_data, audio_rate, str(audio_path))
    
    # Get fault location for ground truth
    positions = vehicle.get_wheel_positions()
    
    if fault_type in ["imbalance", "bearing_wear"]:
        fault_location_key = ["front_left", "front_right", "rear_left", "rear_right"][wheel_index]
    else:
        # Loose mount location
        mount_loc = fault_params.get("location_index", 4)
        if mount_loc == 4:
            fault_location_key = "engine"
        else:
            fault_location_key = ["front_left", "front_right", "rear_left", "rear_right"][mount_loc]
    
    fault_x, fault_y = positions[fault_location_key]
    
    # Create metadata
    metadata = {
        "run_id": run_id,
        "timestamp": "2025-12-07T01:30:00+05:30",
        "fault_type": fault_type,
        "severity": float(severity),
        "speed_kmh": float(speed_kmh),
        "load_kg": float(load_kg),
        "duration_sec": float(duration),
        "imu_sampling_rate": imu_rate,
        "audio_sampling_rate": audio_rate,
        "n_imu_samples": len(imu_data),
        "n_audio_samples": len(audio_data),
        "fault_parameters": fault_params,
        "ground_truth": {
            "fault_location_x_m": float(fault_x),
            "fault_location_y_m": float(fault_y),
            "fault_location_z_m": 0.0,
            "fault_location_name": fault_location_key,
            "fault_metadata": fault.get_metadata(),
        },
        "files": {
            "imu": "imu.csv",
            "audio": "audio.wav",
            "metadata": "meta.json",
        },
    }
    
    # Save metadata
    meta_path = run_dir / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def generate_dataset(
    output_dir: str,
    num_runs: int = 100,
    fault_types: list = None,
    severities: list = None,
    speeds: list = None,
) -> dict:
    """
    Generate a complete dataset
    
    Args:
        output_dir: Where to save all runs
        num_runs: Total number of runs to generate
        fault_types: List of fault types (default: all 3)
        severities: List of severities (default: [0.2, 0.4, 0.6, 0.8])
        speeds: List of speeds in km/h (default: [20, 30, 40, 50, 60])
    
    Returns:
        Manifest dict with dataset statistics
    """
    # Default parameters
    if fault_types is None:
        fault_types = ["imbalance", "loose_mount", "bearing_wear"]
    if severities is None:
        severities = [0.2, 0.4, 0.6, 0.8]
    if speeds is None:
        speeds = [20, 30, 40, 50, 60]
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate parameter combinations
    import itertools
    combinations = list(itertools.product(fault_types, severities, speeds))
    
    # Limit to num_runs
    if len(combinations) > num_runs:
        # Sample evenly across fault types
        runs_per_fault = num_runs // len(fault_types)
        selected = []
        for ft in fault_types:
            ft_combos = [c for c in combinations if c[0] == ft]
            selected.extend(np.random.choice(len(ft_combos), runs_per_fault, replace=False))
        combinations = [combinations[i] for i in selected[:num_runs]]
    
    # Generate runs
    manifest = {
        "dataset_name": "MIRA-Wave Simulated Faults",
        "num_runs": num_runs,
        "fault_types": fault_types,
        "runs": [],
    }
    
    for i, (fault_type, severity, speed) in enumerate(tqdm(combinations, desc="Generating runs")):
        run_id = f"run_{i+1:03d}"
        load_kg = 500.0  # Fixed load for now
        
        metadata = generate_single_run(
            run_id=run_id,
            fault_type=fault_type,
            severity=severity,
            speed_kmh=speed,
            load_kg=load_kg,
            output_dir=output_dir,
            duration=10.0,
            verbose=False,
        )
        
        manifest["runs"].append(metadata)
    
    # Save manifest
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Generated {num_runs} runs in {output_dir}")
    print(f"📊 Fault type distribution:")
    for ft in fault_types:
        count = sum(1 for r in manifest["runs"] if r["fault_type"] == ft)
        print(f"   {ft}: {count} runs")
    
    return manifest


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MIRA-Wave simulated dataset")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of runs to generate")
    parser.add_argument("--output", type=str, default="data/simulated_runs", help="Output directory")
    
    args = parser.parse_args()
    
    generate_dataset(
        output_dir=args.output,
        num_runs=args.num_runs,
    )
