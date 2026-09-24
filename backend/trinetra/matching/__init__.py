"""Correspondence search and robust model fitting."""
import numpy as np


def empty_matches():
    """No correspondences: (points_a, points_b, scores)."""
    return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
