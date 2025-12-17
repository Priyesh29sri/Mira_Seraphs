"""
Example JSON templates for MIRA-Wave data structures
"""

# Example: Meta.json (simulation output metadata)
SIMULATION_META_EXAMPLE = {
    "run_id": "run_042",
    "timestamp": "2025-12-07T00:30:00+05:30",
    "fault_type": "imbalance",
    "severity": 0.6,
    "speed_kmh": 40.0,
    "load_kg": 500.0,
    "duration_sec": 10.0,
    "imu_sampling_rate": 1000,
    "audio_sampling_rate": 44100,
    "n_imu_samples": 10000,
    "n_audio_samples": 441000,
    "ground_truth": {
        "fault_location_x_m": 0.45,
        "fault_location_y_m": -0.12,
        "fault_location_z_m": 0.0,
        "fault_metadata": {
            "fault_type": "imbalance",
            "severity": 0.6,
            "wheel_index": 1,
            "mass_eccentricity_kg_m": 0.009,
            "phase_rad": 0.0
        }
    },
    "files": {
        "imu": "imu.csv",
        "audio": "audio.wav",
        "metadata": "meta.json"
    }
}


# Example: Physics analysis output
PHYSICS_OUTPUT_EXAMPLE = {
    "fault_location": {
        "x_meters": 0.47,
        "y_meters": -0.10,
        "z_meters": 0.05,
        "uncertainty_radius_cm": 3.2,
        "confidence": 0.92,
        "method": "L1_inverse"
    },
    "dominant_frequency_hz": 13.3,
    "spectral_peaks": [
        {"freq_hz": 13.3, "amplitude_db": 35.2},
        {"freq_hz": 26.6, "amplitude_db": 18.5},
        {"freq_hz": 39.9, "amplitude_db": 12.1}
    ],
    "modal_energies": [0.65, 0.20, 0.10, 0.03, 0.02, 0.00]
}


# Example: Fingerprint output
FINGERPRINT_OUTPUT_EXAMPLE = {
    "fingerprint_vector": [0.12, -0.34, 0.56, -0.78, 0.23, -0.45, 0.67, -0.89,
                           0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77, -0.88,
                           0.01, -0.12, 0.23, -0.34, 0.45, -0.56, 0.67, -0.78,
                           0.09, -0.18, 0.27, -0.36, 0.45, -0.54, 0.63, -0.72],
    "peak_frequencies": [13.3, 26.6, 39.9, 53.2],
    "harmonic_ratios": [1.0, 0.52, 0.35, 0.18],
    "embedding_method": "PCA+UMAP"
}


# Example: Fleet matching output
FLEET_MATCH_OUTPUT_EXAMPLE = {
    "matches": [
        {"run_id": "run_015", "similarity": 0.94, "fault_type": "imbalance", "severity": 0.6},
        {"run_id": "run_027", "similarity": 0.89, "fault_type": "imbalance", "severity": 0.6},
        {"run_id": "run_061", "similarity": 0.85, "fault_type": "imbalance", "severity": 0.8},
    ],
    "cluster_id": 3,
    "cluster_size": 18,
    "similarity_scores": [0.94, 0.89, 0.85, 0.82, 0.79]
}


# Example: Causal inference output
CAUSAL_OUTPUT_EXAMPLE = {
    "causes": [
        {"type": "wheel_imbalance", "probability": 0.78, "confidence": 0.92},
        {"type": "tire_wear", "probability": 0.15, "confidence": 0.65},
        {"type": "suspension_looseness", "probability": 0.07, "confidence": 0.43}
    ],
    "top_cause": "wheel_imbalance",
    "top_cause_probability": 0.78,
    "confidence_interval": {"lower": 0.72, "upper": 0.84},
    "correlations": {
        "severity_vs_amplitude": 0.89,
        "speed_vs_frequency": 0.96
    },
    "treatment_effect": 0.42
}


# Example: Experiment plan output
EXPERIMENT_OUTPUT_EXAMPLE = {
    "experiment_suggested": True,
    "instruction": "Increase speed to 60 km/h for 30 seconds to confirm diagnosis",
    "predicted_info_gain_bits": 2.3,
    "new_speed_kmh": 60.0,
    "new_load_kg": 500.0,
    "uncertainty_current": 1.8,
    "uncertainty_threshold": 1.5,
    "rationale": "Higher speed will amplify centrifugal forces, confirming imbalance vs. bearing fault"
}


# Example: Repair schedule output
SCHEDULE_OUTPUT_EXAMPLE = {
    "urgency": 7,
    "workshop_type": "wheel_balancing",
    "estimated_cost": "$50-$80",
    "estimated_time": "1 hour",
    "priority": "medium-high",
    "recommendation": "Schedule within 1 week. Safe to drive but vibration may worsen."
}


# Example: Explanation output
EXPLANATION_OUTPUT_EXAMPLE = {
    "summary": "Wheel imbalance detected at front-right wheel with 78% confidence",
    "full_report": """
The vehicle exhibits a wheel imbalance fault with severity 0.6 located at the
front-right wheel assembly (x=0.45m, y=-0.12m).

Physical Evidence:
- Dominant frequency: 13.3 Hz (1× rotation order at 40 km/h)
- Vibration amplitude: 2.4 m/s² RMS (3× baseline)
- Spectral peak: 35 dB at 13.3 Hz

Fleet Comparison:
This pattern matches 87% of known wheel imbalance cases from the fleet database
(N=45 similar vehicles).

Root Cause Analysis:
Bayesian causal inference assigns 78% confidence to wheel imbalance based on:
- Feature correlation: 0.83
- Treatment effect: 0.42
- Cluster membership: cluster 3 (imbalance pattern)

Recommended Action:
Wheel balancing at alignment shop.
Estimated cost: $50-$80
Estimated time: 1 hour
Urgency: 7/10

Active Experiment Suggestion:
To confirm diagnosis, run test: "Increase speed to 60 km/h for 30s"
Expected information gain: 2.3 bits
"""
}


# Example: Complete final state
FINAL_STATE_EXAMPLE = {
    "run_id": "run_042",
    "fault_type": "imbalance",
    "severity": 0.6,
    "speed_kmh": 40.0,
    "fault_location": PHYSICS_OUTPUT_EXAMPLE["fault_location"],
    "fingerprint_vector": FINGERPRINT_OUTPUT_EXAMPLE["fingerprint_vector"],
    "fleet_matches": FLEET_MATCH_OUTPUT_EXAMPLE["matches"],
    "causal_results": CAUSAL_OUTPUT_EXAMPLE,
    "experiment_plan": EXPERIMENT_OUTPUT_EXAMPLE,
    "repair_schedule": SCHEDULE_OUTPUT_EXAMPLE,
    "explanation_text": EXPLANATION_OUTPUT_EXAMPLE["full_report"],
    "workflow_stage": "completed",
    "total_execution_time_sec": 12.5
}
