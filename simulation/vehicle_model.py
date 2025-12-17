"""
Vehicle Physics Model

Implements a 6-DOF mass-spring-damper system for vehicle vibration simulation.

System equation: M*x'' + C*x' + K*x = F_external + F_fault

Components:
- 4 wheels (nodes 0-3)
- 1 vehicle body (node 4)
- 1 engine (node 5)
"""

import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class VehicleParameters:
    """Physical parameters for the vehicle model"""
    # Masses (kg)
    wheel_mass: float = 20.0  # Each wheel
    body_mass: float = 1000.0  # Vehicle body
    engine_mass: float = 150.0  # Engine
    
    # Geometry (meters)
    wheelbase: float = 2.5  # Front-rear wheel distance
    track_width: float = 1.5  # Left-right wheel distance
    wheel_radius: float = 0.3  # Wheel radius
    
    # Stiffness (N/m)
    tire_stiffness: float = 200000.0  # Tire vertical stiffness
    suspension_stiffness: float = 20000.0  # Suspension spring
    engine_mount_stiffness: float = 50000.0  # Engine mount
    
    # Damping (N·s/m)
    tire_damping: float = 500.0
    suspension_damping: float = 1500.0
    engine_mount_damping: float = 800.0
    
    def total_mass(self) -> float:
        """Calculate total vehicle mass"""
        return 4 * self.wheel_mass + self.body_mass + self.engine_mass


class VehicleModel:
    """
    6-DOF vehicle vibration model
    
    Degrees of freedom (vertical displacements):
    0: Front-left wheel
    1: Front-right wheel
    2: Rear-left wheel
    3: Rear-right wheel
    4: Vehicle body (center of mass)
    5: Engine
    """
    
    def __init__(self, params: Optional[VehicleParameters] = None):
        self.params = params if params is not None else VehicleParameters()
        self.n_dof = 6
        
        # Build system matrices
        self.M = self._build_mass_matrix()
        self.K = self._build_stiffness_matrix()
        self.C = self._build_damping_matrix()
        
        # Fault force function (set by add_fault)
        self.fault_force_func = None
        
        # Speed profile function
        self.speed_func = lambda t: 0.0
        
    def _build_mass_matrix(self) -> np.ndarray:
        """
        Build mass matrix M (diagonal for point masses)
        
        Returns:
            6x6 mass matrix
        """
        masses = [
            self.params.wheel_mass,  # FL wheel
            self.params.wheel_mass,  # FR wheel
            self.params.wheel_mass,  # RL wheel
            self.params.wheel_mass,  # RR wheel
            self.params.body_mass,   # Body
            self.params.engine_mass, # Engine
        ]
        return np.diag(masses)
    
    def _build_stiffness_matrix(self) -> np.ndarray:
        """
        Build stiffness matrix K
        
        Connections:
        - Each wheel to ground (tire stiffness)
        - Each wheel to body (suspension)
        - Engine to body (mount)
        
        Returns:
            6x6 stiffness matrix
        """
        K = np.zeros((6, 6))
        k_tire = self.params.tire_stiffness
        k_susp = self.params.suspension_stiffness
        k_mount = self.params.engine_mount_stiffness
        
        # Tire stiffnesses (wheel to ground, which is fixed at z=0)
        for i in range(4):  # 4 wheels
            K[i, i] += k_tire
        
        # Suspension stiffnesses (wheel to body)
        for i in range(4):
            K[i, i] += k_susp  # Diagonal term
            K[i, 4] -= k_susp  # Coupling to body
            K[4, i] -= k_susp  # Coupling from body
            K[4, 4] += k_susp  # Body diagonal
        
        # Engine mount (engine to body)
        K[5, 5] += k_mount  # Engine diagonal
        K[5, 4] -= k_mount  # Coupling to body
        K[4, 5] -= k_mount  # Coupling from body
        K[4, 4] += k_mount  # Body diagonal
        
        return K
    
    def _build_damping_matrix(self) -> np.ndarray:
        """
        Build damping matrix C (proportional damping)
        
        Same structure as stiffness matrix but with damping coefficients
        
        Returns:
            6x6 damping matrix
        """
        C = np.zeros((6, 6))
        c_tire = self.params.tire_damping
        c_susp = self.params.suspension_damping
        c_mount = self.params.engine_mount_damping
        
        # Tire damping
        for i in range(4):
            C[i, i] += c_tire
        
        # Suspension damping
        for i in range(4):
            C[i, i] += c_susp
            C[i, 4] -= c_susp
            C[4, i] -= c_susp
            C[4, 4] += c_susp
        
        # Engine mount damping
        C[5, 5] += c_mount
        C[5, 4] -= c_mount
        C[4, 5] -= c_mount
        C[4, 4] += c_mount
        
        return C
    
    def compute_eigenmodes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute natural frequencies and mode shapes
        
        Solves generalized eigenvalue problem: K*phi = omega^2*M*phi
        
        Returns:
            natural_frequencies: Array of natural frequencies (Hz)
            mode_shapes: Matrix of mode shapes (columns are modes)
        """
        # Solve eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(self.M, self.K))
        
        # Sort by frequency
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Convert to natural frequencies (Hz)
        natural_frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
        
        return natural_frequencies, eigenvectors
    
    def set_speed_profile(self, speed_kmh: float, ramp_time: float = 2.0):
        """
        Set vehicle speed profile (ramps up then constant)
        
        Args:
            speed_kmh: Target speed in km/h
            ramp_time: Time to reach target speed (seconds)
        """
        speed_mps = speed_kmh / 3.6  # Convert to m/s
        
        def speed_func(t):
            if t < ramp_time:
                return speed_mps * (t / ramp_time)  # Linear ramp
            else:
                return speed_mps
        
        self.speed_func = speed_func
    
    def add_fault_force(self, fault_force_func):
        """
        Add fault-induced forces
        
        Args:
            fault_force_func: Function f(t, x, v, omega) -> F
                t: time
                x: displacement vector
                v: velocity vector
                omega: angular velocity (rad/s)
                Returns: 6-element force vector
        """
        self.fault_force_func = fault_force_func
    
    def _system_ode(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        ODE system for scipy.integrate.odeint
        
        state = [x0, x1, ..., x5, v0, v1, ..., v5]  (12 elements)
        
        Returns:
            dstate/dt = [v0, v1, ..., v5, a0, a1, ..., a5]
        """
        # Extract displacement and velocity
        x = state[:6]  # displacement
        v = state[6:]  # velocity
        
        # Compute angular velocity from speed
        speed_mps = self.speed_func(t)
        omega = speed_mps / self.params.wheel_radius  # rad/s
        
        # Compute forces from springs and dampers
        F_spring = -self.K @ x
        F_damper = -self.C @ v
        
        # Add fault forces if present
        F_fault = np.zeros(6)
        if self.fault_force_func is not None:
            F_fault = self.fault_force_func(t, x, v, omega)
        
        # Total force
        F_total = F_spring + F_damper + F_fault
        
        # Acceleration: a = M^{-1} * F
        a = np.linalg.solve(self.M, F_total)
        
        # Return derivative: [v, a]
        return np.concatenate([v, a])
    
    def simulate(
        self,
        duration: float,
        dt: float = 0.001,
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate vehicle dynamics
        
        Args:
            duration: Simulation duration (seconds)
            dt: Time step (seconds)
            initial_state: Initial [x, v] state (12 elements)
        
        Returns:
            time: Time vector
            displacement: (n_steps, 6) array
            velocity: (n_steps, 6) array
        """
        # Time vector
        time = np.arange(0, duration, dt)
        
        # Initial conditions
        if initial_state is None:
            initial_state = np.zeros(12)  # Start at rest
        
        # Integrate ODE
        state_history = odeint(self._system_ode, initial_state, time)
        
        # Extract displacement and velocity
        displacement = state_history[:, :6]
        velocity = state_history[:, 6:]
        
        # Compute acceleration by differentiating velocity
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1]) / dt
        
        return time, displacement, acceleration
    
    def get_wheel_positions(self) -> Dict[str, Tuple[float, float]]:
        """
        Get (x, y) positions of wheels in vehicle frame
        
        Returns:
            Dict mapping wheel name to (x, y) position
        """
        wb = self.params.wheelbase
        tw = self.params.track_width
        
        return {
            "front_left": (wb / 2, tw / 2),
            "front_right": (wb / 2, -tw / 2),
            "rear_left": (-wb / 2, tw / 2),
            "rear_right": (-wb / 2, -tw / 2),
            "body": (0.0, 0.0),
            "engine": (0.5, 0.0),  # Slightly forward of center
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system"""
        natural_freq, _ = self.compute_eigenmodes()
        
        return {
            "n_dof": self.n_dof,
            "total_mass_kg": self.params.total_mass(),
            "natural_frequencies_hz": natural_freq.tolist(),
            "stiffness_matrix_condition": np.linalg.cond(self.K),
            "mass_matrix_diagonal": np.diag(self.M).tolist(),
        }


# =============================================================================
# Utility Functions
# =============================================================================

def rotation_frequency_hz(speed_kmh: float, wheel_radius_m: float = 0.3) -> float:
    """
    Calculate wheel rotation frequency
    
    Args:
        speed_kmh: Vehicle speed in km/h
        wheel_radius_m: Wheel radius in meters
    
    Returns:
        Rotation frequency in Hz
    """
    speed_mps = speed_kmh / 3.6
    omega_rad_s = speed_mps / wheel_radius_m
    freq_hz = omega_rad_s / (2 * np.pi)
    return freq_hz
