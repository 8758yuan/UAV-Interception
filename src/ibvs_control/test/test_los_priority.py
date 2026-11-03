"""Formula and optimality checks for the LOS-priority rate projection."""

from dataclasses import replace
import math

import numpy as np
import pytest

from ibvs_control.so3_controller import (
    OuterLoopConfig,
    compute_inner_loop,
    compute_outer_loop,
    los_convergence_constraint,
    prioritize_los_rate,
    rodrigues,
)


@pytest.mark.parametrize('omega2,g,h,expected,active', (
    ((0.2, 0.3, 0), (1, 0, 0), 0.1, (0.2, 0.3, 0), False),
    ((2, 0, 0), (1, 0, 0), 0.5, (1, 0, 0), True),
    ((-0.4, 0.3, 0), (1, 0, 0), 0.6, (0.6, 0.3, 0), False),
    ((-2, 3, 0), (1, 0, 0), 0.6, (0.6, 0.8, 0), True),
    ((-3, 2, 0), (1, 0, 0), -0.6, (-0.6, 0.8, 0), True),
    ((-2, 3, 0), (1, 0, 0), -2, (-2/math.sqrt(13), 3/math.sqrt(13), 0), True),
    ((-2, 3, 0), (1, 0, 0), 2, (1, 0, 0), True),
    ((-2, 0, 0), (1, 0, 0), 1, (1, 0, 0), True),
    ((-2, 0, 0), (1, 0, 0), 0.6, (0.6, 0, 0), False),
    ((0, 2, 0), (0, 0, 0), 1, (0, 1, 0), True),
    ((0, 0.2, 0), (0, 0, 0), -1, (0, 0.2, 0), False),
    ((0, 0.2, 0), (0, 0, 0), 0, (0, 0.2, 0), False),
    ((-2, 3, 0), (1e-14, 0, 0), 6e-15, (0.6, 0.8, 0), True),
))
@pytest.mark.parametrize('angle', (0.0, 0.7))
def test_projection_branches_and_frame_equivariance(
    omega2, g, h, expected, active, angle,
):
    rotation = rodrigues((1, 2, -3), angle)
    rate, saturated = prioritize_los_rate(
        rotation @ omega2, rotation @ g, h, 1.0,
    )
    assert rate == pytest.approx(rotation @ expected, abs=2e-14)
    assert saturated == active
    slack = max(0.0, h - np.linalg.norm(g))
    assert (rotation @ g) @ rate >= h - slack - 1e-14
    assert np.linalg.norm(rate) <= 1.0 + 1e-14


def test_random_projections_satisfy_lexicographic_optimality():
    rng = np.random.default_rng(42)
    for _ in range(200):
        g = rng.normal(size=3)
        omega2 = 3 * rng.normal(size=3)
        limit = rng.uniform(0.1, 3)
        h = rng.uniform(-1.5, 1.5) * limit * np.linalg.norm(g)
        rate, _ = prioritize_los_rate(omega2, g, h, limit)
        slack = max(0.0, h - limit * np.linalg.norm(g))
        bound = h - slack
        assert g @ rate >= bound - 1e-12
        assert np.linalg.norm(rate) <= limit + 1e-12
        # KKT stationarity and nonnegative multipliers prove optimality.
        if h < limit * np.linalg.norm(g):
            normals = []
            if abs(g @ rate - bound) < 1e-10:
                normals.append(-g)
            if abs(np.linalg.norm(rate) - limit) < 1e-10:
                normals.append(rate)
            if normals:
                matrix = np.column_stack(normals)
                multipliers = np.linalg.lstsq(
                    matrix, omega2 - rate, rcond=None,
                )[0]
                assert np.min(multipliers) >= -1e-10
                assert matrix @ multipliers == pytest.approx(
                    omega2 - rate, abs=1e-10,
                )
            else:
                assert rate == pytest.approx(omega2)
        else:
            assert rate == pytest.approx(limit * g / np.linalg.norm(g))


def test_constraint_matches_formula_and_los_error_derivative():
    rotation = rodrigues((1, -2, 3), 0.4)
    position = np.array((-8., -0.6, -0.3))
    velocity = np.array((0.2, -0.4, 0.1))
    los = -position / np.linalg.norm(position)
    designed_b = np.array((1., 0., 0.))
    designed = rotation @ designed_b
    gain = 1.7
    cross, h = los_convergence_constraint(
        position, velocity, los, designed, gain,
    )
    z1 = 1 - designed @ los
    a = (designed @ (np.eye(3) - np.outer(los, los)) @ velocity
         / np.linalg.norm(position))
    assert cross == pytest.approx(np.cross(designed, los))
    assert h == pytest.approx(a + gain*z1)
    g = rotation.T @ cross
    rate, _ = prioritize_los_rate((0.1, -0.2, 0.3), g, h, 2.0)
    step = 1e-6
    values = []
    for dt in (-step, step):
        p = position + dt*velocity
        r = rotation @ rodrigues(rate, dt*np.linalg.norm(rate))
        values.append(1 - (r @ designed_b) @ (-p / np.linalg.norm(p)))
    derivative = (values[1] - values[0]) / (2*step)
    assert derivative == pytest.approx(a - g @ rate, abs=1e-9)
    assert derivative <= -gain*z1 + max(0, h - 2*np.linalg.norm(g)) + 1e-9


def test_inner_loop_uses_current_constraint_and_ignores_legacy_omega1():
    config = OuterLoopConfig(0.1, (0.2, 0.2, 0.2), 0.3, 2.0, 40.0)
    outer = compute_outer_loop(
        (-10, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0), np.eye(3), config,
    )
    current = (np.array((0., 0., 1.)), 0.6)
    inner = compute_inner_loop(
        outer, np.eye(3), config, 1., los_constraint_e=current,
    )
    altered = compute_inner_loop(
        replace(outer, omega1_b=np.array((100., -200., 300.))),
        np.eye(3), config, 1., los_constraint_e=current,
    )
    expected, _ = prioritize_los_rate(inner.omega2_b, *current, 1.)
    assert inner.omega_d_b == pytest.approx(expected)
    assert inner.omega_d_b[2] >= 0.6
    assert altered.omega_d_b == pytest.approx(inner.omega_d_b)
    assert altered.thrust_n == inner.thrust_n
    assert config.los_lambda == 1.0


@pytest.mark.parametrize('gain', (0., -1., math.inf, math.nan))
def test_los_lambda_must_be_finite_and_positive(gain):
    config = OuterLoopConfig(
        0.1, (0.2, 0.2, 0.2), 0.3, 2.0, 40.0, los_lambda=gain,
    )
    with pytest.raises(ValueError, match='los_lambda'):
        config.validate()
