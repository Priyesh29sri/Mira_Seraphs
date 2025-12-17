"""
Agent 1: Simulation Engine Agent

Generates synthetic vehicle vibration data with faults.
12 tools for physics simulation, fault injection, and sensor generation.
"""

import numpy as np
from typing import Dict, Any, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from simulation import (
    VehicleModel,
    VehicleParameters,
    create_fault,
    simulate_imu_signal,
    simulate_audio_signal,
    rotation_frequency_hz,
)
from schemas.state_schema import MIRAState
from schemas.message_schema import SimulationOutput


class SimulationEngineAgent:
    """
    Agent 1: Simulation Engine
    
    Responsibility: Generate realistic vehicle vibration data with injected faults.
    
    12 Tools:
    - build_mass_matrix()
    - build_stiffness_matrix()
    - build_damping_matrix()
    - integrate_system_ode()
    - apply_fault_forces()
    - apply_speed_profile()
    - inject_imbalance_force()
    - inject_loose_mount()
    - inject_bearing_noise()
    - vary_severity_curve()
    - simulate_imu_signal()
    - simulate_audio_signal()
    """
    
    def __init__(self):
        self.name = "SimulationEngine"
        self.vehicle = None
        
    # =========================================================================
    # Tool 1-3: Build System Matrices
    # =========================================================================
    
    def build_mass_matrix(self, params: VehicleParameters = None) -> Dict[str, Any]:
        """Tool 1: Build mass matrix M"""
        if params is None:
            params = VehicleParameters()
        
        vehicle = VehicleModel(params)
        M = vehicle.M
        
        return {
            "matrix": M,
            "n_dof": M.shape[0],
            "total_mass_kg": params.total_mass(),
            "is_symmetric": np.allclose(M, M.T),
            "is_positive_definite": np.all(np.linalg.eigvals(M) > 0),
        }
    
    def build_stiffness_matrix(self, params: VehicleParameters = None) -> Dict[str, Any]:
        """Tool 2: Build stiffness matrix K"""
        if params is None:
            params = VehicleParameters()
        
        vehicle = VehicleModel(params)
        K = vehicle.K
        eigenvalues = np.linalg.eigvals(K)
        
        return {
            "matrix": K,
            "n_dof": K.shape[0],
            "condition_number": np.linalg.cond(K),
            "eigenvalues": eigenvalues,
        }
    
    def build_damping_matrix(self, params: VehicleParameters = None) -> Dict[str, Any]:
        """Tool 3: Build damping matrix C"""
        if params is None:
            params = VehicleParameters()
        
        vehicle = VehicleModel(params)
        C = vehicle.C
        
        return {
            "matrix": C,
            "n_dof": C.shape[0],
        }
    
    # =========================================================================
    # Tool 4-6: Simulation Control
    # =========================================================================
    
    def integrate_system_ode(
        self,
        duration: float = 10.0,
        dt: float = 0.001,
    ) -> Dict[str, Any]:
        """Tool 4: Integrate system ODEs"""
        if self.vehicle is None:
            raise ValueError("Vehicle not initialized. Call apply_speed_profile first.")
        
        time, displacement, acceleration = self.vehicle.simulate(duration=duration, dt=dt)
        
        # Compute velocity by integrating acceleration
        velocity = np.cumsum(acceleration, axis=0) * dt
        
        return {
            "displacement": displacement,
            "velocity": velocity,
            "acceleration": acceleration,
            "time": time,
            "n_steps": len(time),
            "duration_sec": duration,
        }
    
    def apply_fault_forces(self, fault_force_func) -> None:
        """Tool 5: Apply fault force function"""
        if self.vehicle is None:
            self.vehicle = VehicleModel()
        
        self.vehicle.add_fault_force(fault_force_func)
    
    def apply_speed_profile(self, speed_kmh: float, ramp_time: float = 2.0) -> None:
        """Tool 6: Set vehicle speed profile"""
        if self.vehicle is None:
            self.vehicle = VehicleModel()
        
        self.vehicle.set_speed_profile(speed_kmh, ramp_time)
    
    # =========================================================================
    # Tool 7-9: Fault Injection
    # =========================================================================
    
    def inject_imbalance_force(
        self,
        severity: float,
        wheel_index: int = 1,
    ) -> Dict[str, Any]:
        """Tool 7: Inject wheel imbalance fault"""
        fault = create_fault("imbalance", severity, wheel_index=wheel_index)
        self.apply_fault_forces(fault.get_force_function())
        
        return fault.get_metadata()
    
    def inject_loose_mount(self, severity: float, location: str = "engine") -> Dict[str, Any]:
        """Tool 8: Inject loose mount fault"""
        fault = create_fault("loose_mount", severity, location=location)
        self.apply_fault_forces(fault.get_force_function())
        
        return fault.get_metadata()
    
    def inject_bearing_noise(
        self,
        severity: float,
        wheel_index: int = 1,
        defect_type: str = "outer_race",
    ) -> Dict[str, Any]:
        """Tool 9: Inject bearing fault"""
        fault = create_fault(
            "bearing_wear",
            severity,
            wheel_index=wheel_index,
            defect_type=defect_type,
        )
        self.apply_fault_forces(fault.get_force_function())
        
        return fault.get_metadata()
    
    # =========================================================================
    # Tool 10: Severity Variation
    # =========================================================================
    
    def vary_severity_curve(self, base_severity: float, variation: float = 0.1) -> float:
        """Tool 10: Add random variation to severity"""
        return np.clip(base_severity + np.random.normal(0, variation), 0.0, 1.0)
    
    # =========================================================================
    # Tool 11-12: Sensor Simulation
    # =========================================================================
    
    def simulate_imu_signal_tool(
        self,
        acceleration: np.ndarray,
        time: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Tool 11: Generate IMU signal from acceleration"""
        return simulate_imu_signal(
            acceleration,
            time,
            sensor_location=4,  # Body
            noise_level=0.05,
            sampling_rate=1000,
        )
    
    def simulate_audio_signal_tool(
        self,
        acceleration: np.ndarray,
        time: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Tool 12: Generate audio signal"""
        return simulate_audio_signal(
            acceleration,
            time,
            microphone_location=5,  # Engine
            noise_level=0.01,
            sampling_rate=44100,
        )
    
    # =========================================================================
    # Agent Execution
    # =========================================================================
    
    def run(self, state: MIRAState) -> SimulationOutput:
        """
        Execute simulation agent
        
        Args:
            state: Current MIRA state with fault_type, severity, speed_kmh
        
        Returns:
            SimulationOutput with generated sensor data
        """
        import time as time_module
        start_time = time_module.time()
        
        try:
            # Initialize vehicle
            self.vehicle = VehicleModel()
            
            # Apply speed profile
            self.apply_speed_profile(state["speed_kmh"], ramp_time=2.0)
            
            # Inject fault
            fault_type = state["fault_type"]
            severity = state["severity"]
            
            if fault_type == "imbalance":
                fault_meta = self.inject_imbalance_force(severity, wheel_index=1)
            elif fault_type == "loose_mount":
                fault_meta = self.inject_loose_mount(severity, location="engine")
            elif fault_type == "bearing_wear":
                fault_meta = self.inject_bearing_noise(severity, wheel_index=1)
            else:
                raise ValueError(f"Unknown fault type: {fault_type}")
            
            # Simulate dynamics
            result = self.integrate_system_ode(duration=10.0, dt=0.001)
            
            # Generate sensor signals
            imu_data, imu_rate = self.simulate_imu_signal_tool(
                result["acceleration"],
                result["time"],
            )
            audio_data, audio_rate = self.simulate_audio_signal_tool(
                result["acceleration"],
                result["time"],
            )
            
            # Get fault location (ground truth)
            positions = self.vehicle.get_wheel_positions()
            if fault_type in ["imbalance", "bearing_wear"]:
                location_key = "front_right"  # wheel_index=1
            else:
                location_key = "engine"
            
            fault_x, fault_y = positions[location_key]
            
            exec_time = time_module.time() - start_time
            
            return SimulationOutput(
                agent_name=self.name,
                success=True,
                execution_time_sec=exec_time,
                imu_data=imu_data,
                audio_data=audio_data,
                imu_sampling_rate=imu_rate,
                audio_sampling_rate=audio_rate,
                simulation_duration_sec=10.0,
                fault_type=fault_type,
                severity=severity,
                speed_kmh=state["speed_kmh"],
                load_kg=state["load_kg"],
                true_fault_location={
                    "x_m": float(fault_x),
                    "y_m": float(fault_y),
                    "z_m": 0.0,
                },
            )
        
        except Exception as e:
            exec_time = time_module.time() - start_time
            return SimulationOutput(
                agent_name=self.name,
                success=False,
                error_message=str(e),
                execution_time_sec=exec_time,
                imu_data=np.array([]),
                audio_data=np.array([]),
                imu_sampling_rate=1000,
                audio_sampling_rate=44100,
                simulation_duration_sec=0.0,
                fault_type=state["fault_type"],
                severity=state["severity"],
                speed_kmh=state["speed_kmh"],
                load_kg=state["load_kg"],
                true_fault_location={"x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
            )
