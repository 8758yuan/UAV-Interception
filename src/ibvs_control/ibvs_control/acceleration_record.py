"""Read-only acceleration snapshots in Gazebo ENU, using simulation time."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ibvs_control.dkf_evaluation_report import write_excel


class AccelerationRecord:
    """Keep nearest real samples per half-second bin; never fabricate missing data."""

    def __init__(self, directory, prefix=''):
        self.directory = directory
        self.prefix = prefix
        self.origin = None
        self.end = None
        self.samples = [{}, {}]

    def add(self, kind, time_ns, acceleration):
        if self.origin is None or time_ns < self.origin:
            return
        if self.end is not None and time_ns > self.end:
            return
        index = (time_ns - self.origin + 250_000_000) // 500_000_000
        samples = self.samples[kind]
        distance = abs(time_ns - self.origin - index * 500_000_000)
        if index not in samples or distance < abs(samples[index][0] - self.origin - index * 500_000_000):
            samples[index] = (time_ns, acceleration)

    def save(self, now_ns):
        rows = [
            ['时间（秒）', '期望取样（秒）', '真值取样（秒）',
             '期望加速度 ENU（米/秒²）', '', '', '真实加速度 ENU（差分近似，米/秒²）', '', ''],
            ['', '', '', '东', '北', '上', '东', '北', '上'],
        ]
        if self.origin is not None:
            end = self.end if self.end is not None else now_ns
            for index in range(max(0, (end - self.origin) // 500_000_000) + 1):
                pair = [samples.get(index) for samples in self.samples]
                pair = [sample if sample and sample[0] <= end else None for sample in pair]
                rows.append([index * 0.5, *[
                    (sample[0] - self.origin) * 1e-9 if sample else '不可用'
                    for sample in pair], *[
                        float(value) if np.isfinite(value) else '不可用'
                        for sample in pair
                        for value in (sample[1] if sample else [np.nan] * 3)]])
        notes = [
            ['加速度记录说明'], ['项目', '说明'],
            ['记录范围', '首次 ACTIVE 至首次退出 ACTIVE；每 0.5 秒选择同一时间分箱内最近样本'],
            ['坐标系', 'Gazebo 世界 ENU（东、北、上），包含重力作用后的运动加速度'],
            ['真实值来源', 'Gazebo 连续三帧位置的二阶差分近似，时间为两段速度中点的中点'],
            ['期望值来源', '控制器：ControlDebug.acceleration_d；DKF 独立测试：位置参考轨迹的解析二阶导数（不是 PX4 内环期望加速度）'],
            ['缺失值', '不可用表示没有有效记录；不插值或补造数据'],
        ]
        write_excel(self.directory, {}, {}, comparison_rows=rows, summary_rows=notes,
                    filename=self.prefix + '每0.5秒加速度对比.xlsx', sheet_name='每0.5秒加速度对比')

    def plot_motion(self, velocity, acceleration, contact_time=None):
        """Use system Matplotlib/NumPy together, without temporary data files."""
        payload = {
            'directory': str(self.directory), 'prefix': self.prefix,
            'origin': self.origin, 'contact': contact_time,
            'velocity': [(int(t), np.asarray(v).tolist()) for t, v in velocity],
            'acceleration': [(int(t), np.asarray(v).tolist()) for t, v in acceleration],
        }
        subprocess.run([sys.executable, '-s', str(Path(__file__).resolve())],
                       input=json.dumps(payload), text=True, check=True)


def pose_acceleration(poses):
    """Second difference on unequal simulation-time steps; return its physical time."""
    (t0, p0), (t1, p1), (t2, p2) = poses
    dt0, dt1 = (t1 - t0) * 1e-9, (t2 - t1) * 1e-9
    acceleration = 2 * ((p2 - p1) / dt1 - (p1 - p0) / dt0) / (dt0 + dt1)
    return (t0 + 2 * t1 + t2) // 4, acceleration


def plot_motion(payload):
    """Draw full-record ENU components and magnitude, with event markers."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    origin = min((samples[0][0] for samples in
                  (payload['velocity'], payload['acceleration']) if samples), default=0)
    for key, unit, filename in (
            ('velocity', 'm/s', '速度.png'), ('acceleration', 'm/s²', '加速度.png')):
        samples = payload[key]
        figure = Figure(figsize=(13, 5), layout='constrained')
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        if samples:
            times = [(t - origin) * 1e-9 for t, _ in samples]
            values = np.array([v for _, v in samples])
            axis.plot(times, np.linalg.norm(values, axis=1), color='#17344e',
                      linewidth=2, label='Magnitude')
            for i, (label, color) in enumerate(zip(('x (East)', 'y (North)', 'z (Up)'),
                                                  ('#269ddd', '#ef8a37', '#ab73c4'))):
                axis.plot(times, values[:, i], color=color, linewidth=1, label=label)
        for time, label, style, color in (
                (payload['origin'], 'Active', '--', '#198e85'),
                (payload['contact'], 'Target contact', ':', '#8654b4')):
            if time is not None:
                elapsed = (time - origin) * 1e-9
                axis.axvline(elapsed, label=f'{label}: {elapsed:.2f} s',
                             linestyle=style, color=color)
        axis.set(title=f"{payload['prefix'].rstrip('_')} — {key.capitalize()} in ENU",
                 xlabel='Time since first motion sample (s)', ylabel=f'{key.capitalize()} ({unit})')
        axis.grid(alpha=.25)
        if samples or payload['origin'] is not None or payload['contact'] is not None:
            axis.legend(loc='upper left', ncol=2)
        figure.savefig(Path(payload['directory']) / (payload['prefix'] + filename), dpi=160)


if __name__ == '__main__':
    plot_motion(json.load(sys.stdin))
