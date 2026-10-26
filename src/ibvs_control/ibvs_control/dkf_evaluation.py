"""Evaluation-only Gazebo pose interpolation and streaming CSV statistics."""

import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

from ibvs_control.frames import quaternion_wxyz_to_rotation


STATE_NAMES = (
    'q_w', 'q_x', 'q_y', 'q_z',
    'p_r_e', 'p_r_n', 'p_r_u', 'v_r_e', 'v_r_n', 'v_r_u',
    'image_x', 'image_y', 'b_gyr_f', 'b_gyr_l', 'b_gyr_u',
    'b_acc_f', 'b_acc_l', 'b_acc_u',
)


def truth_between(before, after, time_ns, camera_rotation, camera_position):
    """Interpolate world poses; velocity is the interval's secant, not exact."""
    t0, pu0, qu0, pt0 = before
    t1, pu1, qu1, pt1 = after
    truth = np.full(18, np.nan)
    if not t0 <= time_ns <= t1 or t1 <= t0:
        return truth
    fraction = (time_ns - t0) / (t1 - t0)
    pu = pu0 + fraction * (pu1 - pu0)
    pt = pt0 + fraction * (pt1 - pt0)
    # Shortest-path SLERP avoids quaternion sign discontinuities.
    qu1 = qu1.copy()
    dot = float(np.dot(qu0, qu1))
    if dot < 0:
        qu1 = -qu1
        dot = -dot
    angle = np.arccos(np.clip(dot, -1, 1))
    if angle < 1e-8:
        q = (1 - fraction) * qu0 + fraction * qu1
    else:
        q = (np.sin((1 - fraction) * angle) * qu0
             + np.sin(fraction * angle) * qu1) / np.sin(angle)
    q /= np.linalg.norm(q)
    truth[:4] = q
    truth[4:7] = pu - pt
    truth[7:10] = ((pu1 - pu0) - (pt1 - pt0)) / ((t1 - t0) * 1e-9)
    pc = camera_rotation.T @ (
        quaternion_wxyz_to_rotation(q).T @ (pt - pu) - camera_position
    )
    if pc[2] > 0:
        truth[10:12] = pc[:2] / pc[2]
    return truth


class EvaluationCsv:
    """Keep every output, including unavailable truth, and summarize finite pairs."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.file = (self.directory / 'states.csv').open('w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ['publish_time_ns', 'imu_timestamp_us', 'state_time_ns',
             'initialized', 'valid', 'reason', 'truth_status',
             'truth_before_ns', 'truth_after_ns', 'clock_offset_us']
            + [f'{name}_{kind}' for name in STATE_NAMES
               for kind in ('true', 'estimate', 'error')]
        )
        self.count = np.zeros(18, dtype=int)
        self.squared = np.zeros(18)
        self.absolute = np.zeros(18)
        self.maximum = np.zeros(18)

    def write(self, metadata, truth, estimate):
        """Align quaternion sign to raw estimate; retain raw estimate components."""
        truth = np.asarray(truth, dtype=float).copy()
        estimate = np.asarray(estimate, dtype=float)
        if np.all(np.isfinite(truth[:4])) and np.dot(truth[:4], estimate[:4]) < 0:
            truth[:4] *= -1
        error = estimate - truth
        self.writer.writerow(list(metadata) + np.column_stack(
            (truth, estimate, error)).reshape(-1).tolist())
        self.file.flush()
        finite = np.isfinite(error)
        self.count += finite
        self.squared[finite] += error[finite] ** 2
        self.absolute[finite] += np.abs(error[finite])
        self.maximum[finite] = np.maximum(self.maximum[finite], np.abs(error[finite]))

    def summarize(self):
        """Write online statistics without rereading or retaining the CSV."""
        with (self.directory / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['state', 'count', 'rmse', 'mae', 'max_abs_error'])
            for i, name in enumerate(STATE_NAMES):
                n = self.count[i]
                writer.writerow([name, n,
                                 np.sqrt(self.squared[i] / n) if n else np.nan,
                                 self.absolute[i] / n if n else np.nan,
                                 self.maximum[i] if n else np.nan])

    def close(self):
        self.summarize()
        self.file.close()
        # Use the ROS interpreter without user-site NumPy overrides; system
        # Matplotlib and NumPy must use the same binary ABI. Only runs on exit.
        subprocess.run([
            sys.executable, '-s', str(Path(__file__).with_name('dkf_evaluation_report.py')),
            str(self.directory),
        ], check=True)
