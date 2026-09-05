"""
Configuration file for the ML-Based Deviation-Aware Adaptive Assembly
Parameter Recommendation System.

This file locks the Version 2 prototype variables and ranges.
"""

PROFILE_LENGTH_MM = 1000.0
N_POINTS = 100

DEVIATION_RANGES = {
    "dev_offset": (-2.0, 2.0),       # mm
    "dev_tilt": (-1.0, 1.0),         # degree
    "dev_bend": (-1.5, 1.5),         # mm
    "dev_waviness": (0.0, 0.8),      # mm
    "dev_noise": (0.0, 0.15),        # mm
}

DECISION_RANGES = {
    "z_adj": (-2.5, 2.5),            # mm
    "theta_adj": (-1.2, 1.2),        # degree
    "locator_offset": (-1.0, 1.0),   # mm
}

QUALITY_WEIGHTS = {
    "mean_gap": 0.30,
    "max_gap": 0.30,
    "parallelism_error": 0.30,
    "rms_deviation": 0.10,
}

TOLERANCE_LIMITS = {
    "mean_gap": 0.50,
    "max_gap": 1.00,
    "parallelism_error": 0.80,
}

LOCATOR_POSITION = 0.50
LOCATOR_SIGMA = 0.18

DEFAULT_DATASET_SIZE = 3000
RANDOM_STATE = 42