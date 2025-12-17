"""
Fault Models

Implements physics-based fault force functions for:
1. Imbalance (centrifugal force)
2. Loose Mount (bilinear stiffness)
3. Bearing Wear (impact pulses)
"""

import numpy as np
from typing import Callable, Dict, Any


class ImbalanceFault:
    """
    Wheel imbalance fault
    
    Physics: Centrifugal force F = m*e*omega^2*cos(omega*t + phi)
    where m*e is mass * eccentricity
    """
    
    def __init__(
        self,
        severity: float,
        wheel_index: int = 1,  # Front-right wheel (0=FL, 1=FR, 2=RL, 3=RR)
        phase: float = 0.0,
    ):
        """
        Args:
            severity: 0.0 to 1.0 (maps to mass*eccentricity product)
            wheel_index: Which wheel has imbalance (0-3)
            phase: Initial phase angle (radians)
        """
        assert 0.0 <= severity <= 1.0, "Severity must be in [0, 1]"
        assert 0 <= wheel_index <= 3, "Wheel index must be 0-3"
        
        self.severity = severity
        self.wheel_index = wheel_index
        self.phase = phase
        
        # Map severity to mass*eccentricity (kg·m)
        # Severity 1.0 = 50g at 30cm = 0.015 kg·m
        self.mass_eccentricity = severity * 0.015  # kg·m
    
    def get_force_function(self) -> Callable:
        """
        Returns force function f(t, x, v, omega) -> F
        """
        def force_func(t, x, v, omega):
            F = np.zeros(6)
            
            # Centrifugal force magnitude
            F_mag = self.mass_eccentricity * (omega ** 2)
            
            # Apply to specific wheel (vertical direction)
            F[self.wheel_index] = F_mag * np.cos(omega * t + self.phase)
            
            return F
        
        return force_func
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return fault metadata"""
        return {
            "fault_type": "imbalance",
            "severity": self.severity,
            "wheel_index": self.wheel_index,
            "mass_eccentricity_kg_m": self.mass_eccentricity,
            "phase_rad": self.phase,
        }


class LooseMountFault:
    """
    Loose mount fault (e.g., engine mount with backlash)
    
    Physics: Bilinear stiffness - different stiffness in compression vs tension
    Creates harmonic distortion and impacts
    """
    
    def __init__(
        self,
        severity: float,
        backlash_gap_mm: float = 2.0,
        location: str = "engine",  # "engine" or "suspension_{index}"
    ):
        """
        Args:
            severity: 0.0 to 1.0 (reduction in stiffness)
            backlash_gap_mm: Gap size in millimeters
            location: Where the loose mount is
        """
        assert 0.0 <= severity <= 1.0, "Severity must be in [0, 1]"
        
        self.severity = severity
        self.backlash_gap = backlash_gap_mm / 1000.0  # Convert to meters
        self.location = location
        
        # Stiffness reduction factor
        self.stiffness_factor = 1.0 - 0.9 * severity  # Max 90% reduction
        
        # Contact stiffness (when hitting backlash limit)
        self.contact_stiffness = 5e5  # N/m (very stiff contact)
    
    def get_force_function(self) -> Callable:
        """
        Returns force function with bilinear stiffness
        """
        def force_func(t, x, v, omega):
            F = np.zeros(6)
            
            if self.location == "engine":
                # Engine mount (between nodes 4 and 5)
                relative_disp = x[5] - x[4]  # Engine - body
                relative_vel = v[5] - v[4]
                
                # Bilinear spring force
                if abs(relative_disp) < self.backlash_gap:
                    # Inside gap: reduced stiffness
                    F_spring = -50000 * self.stiffness_factor * relative_disp
                else:
                    # Hit limit: contact stiffness + impact
                    gap_excess = abs(relative_disp) - self.backlash_gap
                    F_spring = -self.contact_stiffness * gap_excess * np.sign(relative_disp)
                    
                    # Add impact damping
                    F_spring -= 1000 * relative_vel
                
                # Apply equal and opposite forces
                F[5] += F_spring
                F[4] -= F_spring
            
            return F
        
        return force_func
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return fault metadata"""
        return {
            "fault_type": "loose_mount",
            "severity": self.severity,
            "backlash_gap_mm": self.backlash_gap * 1000,
            "stiffness_factor": self.stiffness_factor,
            "location": self.location,
        }


class BearingFault:
    """
    Bearing fault (rolling element defect)
    
    Physics: Periodic impacts when rolling elements hit spall/defect
    Creates impulses at bearing characteristic frequencies (BPFO, BPFI, etc.)
    """
    
    def __init__(
        self,
        severity: float,
        wheel_index: int = 1,
        defect_type: str = "outer_race",  # "outer_race", "inner_race", "ball"
        n_elements: int = 9,  # Number of rolling elements
        contact_angle_deg: float = 0.0,
    ):
        """
        Args:
            severity: 0.0 to 1.0 (impact magnitude)
            wheel_index: Which wheel has bearing fault
            defect_type: Type of defect
            n_elements: Number of rolling elements in bearing
            contact_angle_deg: Contact angle (degrees)
        """
        assert 0.0 <= severity <= 1.0, "Severity must be in [0, 1]"
        assert 0 <= wheel_index <= 3, "Wheel index must be 0-3"
        
        self.severity = severity
        self.wheel_index = wheel_index
        self.defect_type = defect_type
        self.n_elements = n_elements
        self.contact_angle = np.radians(contact_angle_deg)
        
        # Bearing geometry (typical values)
        self.d_D_ratio = 0.25  # ball diameter / pitch diameter
        
        # Impact magnitude (kg·m/s²)
        self.impact_magnitude = severity * 500.0  # Up to 500 N
        
        # Compute characteristic frequency multiplier
        self._compute_frequency_multiplier()
    
    def _compute_frequency_multiplier(self):
        """
        Compute bearing characteristic frequency as multiple of shaft frequency
        
        BPFO = (n/2) * (1 - (d/D)*cos(alpha))
        BPFI = (n/2) * (1 + (d/D)*cos(alpha))
        BSF = (D/2d) * (1 - (d/D)^2*cos^2(alpha))
        """
        n = self.n_elements
        d_D = self.d_D_ratio
        cos_a = np.cos(self.contact_angle)
        
        if self.defect_type == "outer_race":
            # Ball Pass Frequency Outer race
            self.freq_multiplier = (n / 2) * (1 - d_D * cos_a)
        elif self.defect_type == "inner_race":
            # Ball Pass Frequency Inner race
            self.freq_multiplier = (n / 2) * (1 + d_D * cos_a)
        elif self.defect_type == "ball":
            # Ball Spin Frequency
            self.freq_multiplier = (1 / (2 * d_D)) * (1 - (d_D * cos_a) ** 2)
        else:
            self.freq_multiplier = 1.0  # Default to 1× rotation
    
    def get_force_function(self) -> Callable:
        """
        Returns force function with periodic impacts
        """
        def force_func(t, x, v, omega):
            F = np.zeros(6)
            
            # Characteristic frequency
            f_char = (omega / (2 * np.pi)) * self.freq_multiplier  # Hz
            omega_char = 2 * np.pi * f_char  # rad/s
            
            # Impact pulse (Gaussian envelope)
            pulse_width = 0.001  # 1ms pulse duration
            phase = omega_char * t
            
            # Create pulse train (positive pulses when sin crosses zero positively)
            pulse_strength = np.exp(-((phase % (2 * np.pi)) - np.pi) ** 2 / (2 * pulse_width ** 2))
            
            # Impact force
            F_impact = self.impact_magnitude * pulse_strength
            
            # Apply to wheel
            F[self.wheel_index] = F_impact
            
            return F
        
        return force_func
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return fault metadata"""
        return {
            "fault_type": "bearing_wear",
            "severity": self.severity,
            "wheel_index": self.wheel_index,
            "defect_type": self.defect_type,
            "n_elements": self.n_elements,
            "frequency_multiplier": self.freq_multiplier,
            "impact_magnitude_N": self.impact_magnitude,
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_fault(
    fault_type: str,
    severity: float,
    **kwargs
) -> object:
    """
    Factory function to create fault objects
    
    Args:
        fault_type: "imbalance", "loose_mount", or "bearing_wear"
        severity: 0.0 to 1.0
        **kwargs: Additional parameters for specific fault types
    
    Returns:
        Fault object (ImbalanceFault, LooseMountFault, or BearingFault)
    
    Example:
        fault = create_fault("imbalance", severity=0.6, wheel_index=1)
    """
    if fault_type == "imbalance":
        return ImbalanceFault(severity, **kwargs)
    elif fault_type == "loose_mount":
        return LooseMountFault(severity, **kwargs)
    elif fault_type == "bearing_wear":
        return BearingFault(severity, **kwargs)
    else:
        raise ValueError(f"Unknown fault type: {fault_type}")
