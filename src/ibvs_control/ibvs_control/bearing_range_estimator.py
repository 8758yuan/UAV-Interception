"""Estimate target range from time-aligned camera poses and image bearings.

The camera optical axis is +Z, and normalized image coordinates are X/Z,
Y/Z. Poses must refer to image capture time. Covariance accounts for image
noise conditional on the supplied camera poses; pose and calibration errors
must be included separately by the caller.
"""

from dataclasses import dataclass, replace
import math
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BearingSample:
    """One normalized image measurement and synchronized camera pose."""

    timestamp_s: float
    image_xy: Tuple[float, float]
    camera_position_enu_m: Tuple[float, float, float]
    camera_to_enu_rotation: np.ndarray


@dataclass(frozen=True)
class BearingRangeConfig:
    """Window size, image noise, and observability acceptance limits."""

    model: str = 'static'
    image_noise_std: float = 0.003
    min_samples: int = 6
    max_samples: int = 90
    max_age_s: float = 3.0
    min_parallax_deg: float = 0.5
    min_transverse_baseline_m: float = 0.25
    min_time_span_s: float = 0.5
    min_depth_m: float = 0.25
    max_condition_number: float = 1000.0
    max_reprojection_rms: float = 0.03
    outlier_reprojection_threshold: float = 0.06
    max_outlier_fraction: float = 0.2
    max_range_std_m: float = 5.0
    max_velocity_std_m_s: float = 2.0

    def validate(self) -> None:
        """Reject inconsistent or nonfinite configuration values."""
        if self.model not in ('static', 'constant_velocity'):
            raise ValueError('model must be static or constant_velocity')
        minimum = 3 if self.model == 'static' else 4
        if (
            isinstance(self.min_samples, bool)
            or not isinstance(self.min_samples, int)
            or self.min_samples < minimum
        ):
            raise ValueError(f'min_samples must be at least {minimum}')
        if (
            isinstance(self.max_samples, bool)
            or not isinstance(self.max_samples, int)
            or self.max_samples < self.min_samples
        ):
            raise ValueError('max_samples must be at least min_samples')
        positive = (
            'image_noise_std', 'max_age_s', 'min_time_span_s',
            'min_depth_m', 'max_condition_number',
            'max_reprojection_rms', 'outlier_reprojection_threshold',
            'max_range_std_m',
            'max_velocity_std_m_s',
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('min_parallax_deg', 'min_transverse_baseline_m'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.min_parallax_deg >= 180.0:
            raise ValueError('min_parallax_deg must be less than 180')
        if self.max_condition_number <= 1.0:
            raise ValueError('max_condition_number must exceed one')
        if (
            not math.isfinite(self.max_outlier_fraction)
            or not 0.0 <= self.max_outlier_fraction < 1.0
        ):
            raise ValueError('max_outlier_fraction must be in [0, 1)')


@dataclass(frozen=True)
class BearingRangeQuality:
    """Geometry and fit checks for one sliding-window estimate."""

    sample_count: int
    time_span_s: float
    inlier_count: int = 0
    outlier_count: int = 0
    parallax_deg: float = math.nan
    transverse_baseline_m: float = math.nan
    condition_number: float = math.inf
    reprojection_rms: float = math.nan
    max_reprojection_error: float = math.nan
    range_std_m: float = math.inf
    velocity_std_m_s: float = math.inf


@dataclass(frozen=True)
class BearingRangeEstimate:
    """Accepted estimate or a rejection with the available diagnostics.

    In static mode covariance_enu contains a 3x3 position block. In constant
    velocity mode it contains the full 6x6 covariance for [position, velocity]
    at reference_time_s, including their cross-covariance.
    """

    accepted: bool
    reason: str
    reference_time_s: Optional[float]
    target_position_enu_m: Optional[np.ndarray]
    target_velocity_enu_m_s: Optional[np.ndarray]
    covariance_enu: Optional[np.ndarray]
    quality: BearingRangeQuality


class BearingRangeEstimator:
    """Fit static or constant-velocity target state from recent bearings."""

    def __init__(self, config: Optional[BearingRangeConfig] = None) -> None:
        self.config = config or BearingRangeConfig()
        self.config.validate()
        self._samples = []

    def clear(self) -> None:
        """Forget all previous measurements, for example after target loss."""
        self._samples.clear()

    def add(self, sample: BearingSample) -> bool:
        """Add a newer capture-time sample; return False for stale stamps."""
        validated = _validated_sample(sample)
        if (
            self._samples
            and validated.timestamp_s <= self._samples[-1].timestamp_s
        ):
            return False
        self._samples.append(validated)
        cutoff = validated.timestamp_s - self.config.max_age_s
        self._samples = [
            item for item in self._samples if item.timestamp_s >= cutoff
        ][-self.config.max_samples:]
        return True

    def estimate(self) -> BearingRangeEstimate:
        """Estimate state and reject windows with weak or inconsistent data."""
        return self._estimate_samples(self._samples, allow_refit=True)

    def _estimate_samples(self, samples, allow_refit):
        """Fit a window, optionally clipping image outliers once."""
        count = len(samples)
        stamp = samples[-1].timestamp_s if samples else None
        span = (
            samples[-1].timestamp_s - samples[0].timestamp_s
            if count else 0.0
        )
        quality = BearingRangeQuality(count, span, inlier_count=count)

        def reject(reason, current_quality=quality):
            return BearingRangeEstimate(
                False, reason, stamp, None, None, None, current_quality
            )

        if count < self.config.min_samples:
            return reject('insufficient_samples')
        if (
            self.config.model == 'constant_velocity'
            and span < self.config.min_time_span_s
        ):
            return reject('insufficient_time_span')

        directions = np.array([
            item.camera_to_enu_rotation
            @ np.array((item.image_xy[0], item.image_xy[1], 1.0))
            for item in samples
        ])
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        positions = np.array([
            item.camera_position_enu_m for item in samples
        ])
        parallax, baseline = _geometry_quality(directions, positions)
        quality = _updated_quality(
            quality, parallax_deg=parallax,
            transverse_baseline_m=baseline,
        )
        if parallax < self.config.min_parallax_deg:
            return reject('insufficient_parallax', quality)
        if baseline < self.config.min_transverse_baseline_m:
            return reject('insufficient_transverse_baseline', quality)

        dynamic = self.config.model == 'constant_velocity'
        # Fit displacement across the window rather than velocity so the
        # linear system does not depend on the unit chosen for timestamps.
        normalized_times = np.array([
            (item.timestamp_s - stamp) / span if dynamic else 0.0
            for item in samples
        ])
        projections = (
            np.eye(3) - directions[:, :, None] * directions[:, None, :]
        )
        if dynamic:
            design = np.concatenate(
                (projections, normalized_times[:, None, None] * projections),
                axis=2,
            ).reshape((-1, 6))
        else:
            design = projections.reshape((-1, 3))
        right_hand = np.einsum('nij,nj->ni', projections, positions).ravel()
        state, _, rank, singular = np.linalg.lstsq(
            design, right_hand, rcond=None
        )
        dimension = 6 if dynamic else 3
        if rank < dimension:
            return reject('unobservable_geometry', quality)
        condition = float(singular[0] / singular[-1])
        quality = _updated_quality(quality, condition_number=condition)
        if condition > self.config.max_condition_number:
            return reject('ill_conditioned_geometry', quality)

        # Minimize the image-plane residual after the line-distance solution.
        # This also supplies a local image-space information matrix for P.
        measured = np.array([item.image_xy for item in samples])
        for _ in range(10):
            projection = _project(
                samples, state, normalized_times, dynamic,
                self.config.min_depth_m,
            )
            if projection is None:
                return reject('target_behind_camera', quality)
            predicted, jacobian = projection
            residual_xy = measured - predicted
            residual = residual_xy.ravel()
            errors = np.linalg.norm(residual_xy, axis=1)
            threshold = self.config.outlier_reprojection_threshold
            weights = np.minimum(
                1.0, threshold / np.maximum(errors, 1e-12)
            )
            row_weights = np.repeat(np.sqrt(weights), 2)
            step, _, _, _ = np.linalg.lstsq(
                row_weights[:, None] * jacobian,
                row_weights * residual,
                rcond=None,
            )
            if not np.all(np.isfinite(step)):
                return reject('numerical_failure', quality)
            if np.linalg.norm(step) < 1e-9:
                break
            # Backtracking retains positive depth and lowers image error.
            scale = 1.0
            current_cost = _robust_cost(errors, threshold)
            while scale >= 1.0 / 128.0:
                trial = state + scale * step
                trial_projection = _project(
                    samples, trial, normalized_times, dynamic,
                    self.config.min_depth_m,
                )
                if trial_projection is not None:
                    trial_errors = np.linalg.norm(
                        measured - trial_projection[0], axis=1
                    )
                    if _robust_cost(trial_errors, threshold) <= current_cost:
                        break
                scale *= 0.5
            if scale < 1.0 / 128.0:
                break
            state = trial

        projection = _project(
            samples, state, normalized_times, dynamic,
            self.config.min_depth_m,
        )
        if projection is None:
            return reject('target_behind_camera', quality)
        predicted, jacobian = projection
        residual = (measured - predicted).ravel()
        sample_errors = np.linalg.norm(measured - predicted, axis=1)
        rms = float(np.sqrt(np.mean(sample_errors ** 2)))
        outliers = sample_errors > self.config.outlier_reprojection_threshold
        outlier_count = int(np.count_nonzero(outliers))
        quality = _updated_quality(
            quality,
            inlier_count=count - outlier_count,
            outlier_count=outlier_count,
            max_reprojection_error=float(np.max(sample_errors)),
            reprojection_rms=rms,
        )
        if outlier_count:
            if (
                outlier_count / count > self.config.max_outlier_fraction
                or count - outlier_count < self.config.min_samples
            ):
                return reject('excess_outlier_fraction', quality)
            if not allow_refit:
                return reject('excess_sample_reprojection_error', quality)
            inlier_samples = [
                item for item, is_outlier in zip(samples, outliers)
                if not is_outlier
            ]
            result = self._estimate_samples(inlier_samples, allow_refit=False)
            return replace(result, quality=replace(
                result.quality,
                sample_count=count,
                inlier_count=len(inlier_samples),
                outlier_count=outlier_count,
                max_reprojection_error=float(np.max(sample_errors)),
            ))
        _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
        tolerance = np.finfo(float).eps * max(jacobian.shape) * singular[0]
        if singular[-1] <= tolerance:
            return reject('unobservable_geometry', quality)
        condition = float(singular[0] / singular[-1])
        quality = _updated_quality(
            quality, condition_number=max(quality.condition_number, condition),
        )
        if quality.condition_number > self.config.max_condition_number:
            return reject('ill_conditioned_geometry', quality)

        degrees_freedom = max(1, 2 * count - dimension)
        residual_variance = float(residual @ residual / degrees_freedom)
        variance = max(self.config.image_noise_std ** 2, residual_variance)
        covariance = (vt.T * (variance / singular ** 2)) @ vt
        if dynamic:
            transform = np.diag((1.0, 1.0, 1.0, 1.0 / span,
                                 1.0 / span, 1.0 / span))
            covariance = transform @ covariance @ transform.T
            velocity = state[3:] / span
            velocity_std = math.sqrt(max(
                0.0, float(np.linalg.eigvalsh(covariance[3:, 3:])[-1])
            ))
        else:
            velocity = None
            velocity_std = math.nan
        covariance = (covariance + covariance.T) / 2.0
        range_direction = state[:3] - positions[-1]
        range_direction /= np.linalg.norm(range_direction)
        range_std = math.sqrt(max(
            0.0,
            float(range_direction @ covariance[:3, :3] @ range_direction),
        ))
        quality = _updated_quality(
            quality, range_std_m=range_std,
            velocity_std_m_s=velocity_std,
        )
        if rms > self.config.max_reprojection_rms:
            return reject('excess_reprojection_error', quality)
        if range_std > self.config.max_range_std_m:
            return reject('uncertain_range', quality)
        if dynamic and velocity_std > self.config.max_velocity_std_m_s:
            return reject('uncertain_velocity', quality)
        return BearingRangeEstimate(
            accepted=True,
            reason='ok',
            reference_time_s=stamp,
            target_position_enu_m=state[:3].copy(),
            target_velocity_enu_m_s=(
                velocity.copy() if velocity is not None else None
            ),
            covariance_enu=covariance,
            quality=quality,
        )


def _validated_sample(sample: BearingSample) -> BearingSample:
    """Copy finite sample values and enforce a proper camera rotation."""
    stamp = float(sample.timestamp_s)
    image = np.asarray(sample.image_xy, dtype=float)
    position = np.asarray(sample.camera_position_enu_m, dtype=float)
    rotation = np.asarray(sample.camera_to_enu_rotation, dtype=float)
    if not math.isfinite(stamp):
        raise ValueError('timestamp_s must be finite')
    if image.shape != (2,) or not np.all(np.isfinite(image)):
        raise ValueError('image_xy must contain two finite values')
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(
            'camera_position_enu_m must contain three finite values'
        )
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError('camera_to_enu_rotation must be a finite 3x3 matrix')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError('camera_to_enu_rotation must be orthonormal')
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError('camera_to_enu_rotation must have determinant +1')
    return BearingSample(
        timestamp_s=stamp,
        image_xy=(float(image[0]), float(image[1])),
        camera_position_enu_m=tuple(float(value) for value in position),
        camera_to_enu_rotation=rotation.copy(),
    )


def _geometry_quality(directions, positions):
    """Measure angular separation and camera translation across the window."""
    cosines = np.clip(directions @ directions.T, -1.0, 1.0)
    parallax = math.degrees(math.acos(float(np.min(cosines))))
    mean_direction = directions[-1]
    offsets = positions - positions[0]
    transverse = offsets - np.outer(offsets @ mean_direction, mean_direction)
    differences = transverse[:, None, :] - transverse[None, :, :]
    baseline = float(np.max(np.linalg.norm(differences, axis=2)))
    return parallax, baseline


def _project(samples, state, normalized_times, dynamic, min_depth_m):
    """Return predicted normalized image coordinates and state Jacobian."""
    predicted = []
    jacobians = []
    for item, relative_time in zip(samples, normalized_times):
        target = state[:3]
        if dynamic:
            target = target + relative_time * state[3:]
        camera_xyz = item.camera_to_enu_rotation.T @ (
            target - item.camera_position_enu_m
        )
        depth = float(camera_xyz[2])
        if not np.all(np.isfinite(camera_xyz)) or depth <= min_depth_m:
            return None
        x, y = camera_xyz[:2] / depth
        predicted.append((x, y))
        image_jacobian = np.array((
            (1.0 / depth, 0.0, -x / depth),
            (0.0, 1.0 / depth, -y / depth),
        )) @ item.camera_to_enu_rotation.T
        if dynamic:
            image_jacobian = np.hstack((
                image_jacobian, relative_time * image_jacobian
            ))
        jacobians.append(image_jacobian)
    return np.asarray(predicted), np.vstack(jacobians)


def _updated_quality(quality, **updates):
    """Copy a frozen diagnostics record with selected fields updated."""
    values = vars(quality).copy()
    values.update(updates)
    return BearingRangeQuality(**values)


def _robust_cost(errors, threshold):
    """Huber cost for per-sample normalized image error magnitudes."""
    return float(np.sum(np.where(
        errors <= threshold,
        errors ** 2,
        2.0 * threshold * errors - threshold ** 2,
    )))
