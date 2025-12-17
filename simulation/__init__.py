"""Simulation package"""

from simulation.vehicle_model import VehicleModel, VehicleParameters, rotation_frequency_hz
from simulation.fault_models import ImbalanceFault, LooseMountFault, BearingFault, create_fault
from simulation.sensor_simulation import (
    simulate_imu_signal,
    simulate_audio_signal,
    save_imu_to_csv,
    save_audio_to_wav,
)


__all__ = [
    "VehicleModel",
    "VehicleParameters",
    "rotation_frequency_hz",
    "ImbalanceFault",
    "LooseMountFault",
    "BearingFault",
    "create_fault",
    "simulate_imu_signal",
    "simulate_audio_signal",
    "save_imu_to_csv",
    "save_audio_to_wav",
]
