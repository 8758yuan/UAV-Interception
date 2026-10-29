"""Test-only rotation helpers copied from the original so3_controller.py."""

import math
from typing import Sequence

import numpy as np


def rodrigues(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    """Convert a unit rotation axis and angle to an SO(3) matrix."""
    unit_axis = _unit3(axis, 'axis')
    if not math.isfinite(angle_rad):
        raise ValueError('angle_rad must be finite')
    axis_skew = skew(unit_axis)
    return (
        np.eye(3)
        + axis_skew * math.sin(angle_rad)
        + axis_skew @ axis_skew * (1.0 - math.cos(angle_rad))
    )


def skew(vector: Sequence[float]) -> np.ndarray:
    """Return the cross-product matrix of a three-vector."""
    x, y, z = _vector3(vector, 'vector')
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _vector3(values: Sequence[float], name: str) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be a finite three-vector')
    return value


def _unit3(values: Sequence[float], name: str) -> np.ndarray:
    value = _vector3(values, name)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError(f'{name} must be nonzero')
    return value / norm
