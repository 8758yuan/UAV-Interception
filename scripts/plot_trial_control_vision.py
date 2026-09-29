"""Plot thrust command, earth-frame angular velocity, and image features."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from ibvs_control.frames import px4_quaternion_to_enu_flu_rotation
from interception_interfaces.msg import ControlDebug, VisionFeature
from px4_msgs.msg import VehicleOdometry, VehicleRatesSetpoint
from rclpy.serialization import deserialize_message
from ros_gz_interfaces.msg import Contacts


RATE_TOPIC = '/fmu/in/vehicle_rates_setpoint'
ODOMETRY_TOPIC = '/fmu/out/vehicle_odometry'
FEATURE_TOPIC = '/interception/vision/raw_feature'
DEBUG_TOPIC = '/interception/control/debug'
CONTACT_TOPIC = (
    '/world/default/model/ibvs_target/link/target_link/sensor/'
    'target_contact/contact'
)
TOPIC_TYPES = {
    RATE_TOPIC: VehicleRatesSetpoint,
    ODOMETRY_TOPIC: VehicleOdometry,
    FEATURE_TOPIC: VisionFeature,
    DEBUG_TOPIC: ControlDebug,
    CONTACT_TOPIC: Contacts,
}


def read_bag(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('cdr', 'cdr'),
    )
    available = {topic.name for topic in reader.get_all_topics_and_types()}
    missing = {RATE_TOPIC, ODOMETRY_TOPIC, FEATURE_TOPIC} - available
    if missing:
        raise ValueError(f'missing required topics: {sorted(missing)}')
    thrust = []
    thrust_newtons = []
    thrust_earth = []
    angular = []
    features = []
    events = {}
    first_stamp_ns = None
    last_time_s = 0.0
    latest_rotation = None
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        if first_stamp_ns is None:
            first_stamp_ns = stamp_ns
        time_s = (stamp_ns - first_stamp_ns) * 1e-9
        last_time_s = time_s
        if topic not in TOPIC_TYPES:
            continue
        message = deserialize_message(data, TOPIC_TYPES[topic])
        if topic == RATE_TOPIC:
            magnitude = -float(message.thrust_body[2])
            if math.isfinite(magnitude):
                thrust.append((time_s, magnitude))
        elif topic == ODOMETRY_TOPIC:
            body_frd = np.asarray(message.angular_velocity, dtype=float)
            if not np.all(np.isfinite(body_frd)):
                continue
            body_flu = body_frd * (1.0, -1.0, -1.0)
            rotation = px4_quaternion_to_enu_flu_rotation(message.q)
            latest_rotation = rotation
            earth = rotation @ body_flu
            angular.append((time_s, *earth, float(np.linalg.norm(earth))))
        elif topic == FEATURE_TOPIC:
            if message.valid and all(math.isfinite(value) for value in (
                message.x_norm, message.y_norm, message.area_px,
            )):
                x = float(message.x_norm)
                y = float(message.y_norm)
                features.append((time_s, x, y, math.hypot(x, y),
                                 float(message.area_px)))
            else:
                # NaNs break the line while the target is not detected.
                features.append((time_s, math.nan, math.nan,
                                 math.nan, math.nan))
        elif topic == DEBUG_TOPIC:
            if message.control_state == 'ACTIVE':
                events.setdefault('Active', time_s)
            if message.control_state in ('PRESTREAM', 'ACTIVE'):
                force_n = float(message.thrust_n)
                if message.valid and math.isfinite(force_n):
                    thrust_newtons.append((time_s, force_n))
                    if latest_rotation is not None:
                        force_earth = force_n * latest_rotation[:, 2]
                        thrust_earth.append((
                            time_s, *force_earth,
                            float(np.linalg.norm(force_earth)),
                        ))
        elif topic == CONTACT_TOPIC and message.contacts:
            events.setdefault('Target contact', time_s)
    return (thrust, thrust_newtons, thrust_earth, angular, features,
            events, last_time_s)


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def mark_events(ax, events):
    for name, color, style in (
        ('Active', '#21867a', '--'),
        ('Target contact', '#7a4ca0', ':'),
    ):
        if name in events:
            ax.axvline(events[name], color=color, linestyle=style,
                       linewidth=1.5, label=f'{name}: {events[name]:.2f} s')


def finish(fig, ax, trial_id, title, ylabel, end_s, path):
    ax.set(title=f'{trial_id} — {title}', xlabel='Time since bag start (s)',
           ylabel=ylabel)
    ax.set_xlim(0, end_s)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', ncol=2, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_thrust(path, trial_id, normalized, newtons, events, end_s):
    fig, (force_ax, normalized_ax) = plt.subplots(
        2, 1, figsize=(12, 7.4), sharex=True, layout='constrained',
    )
    force_ax.plot([row[0] for row in newtons], [row[1] for row in newtons],
                  color='#102a43', linewidth=1.4,
                  label='Controller thrust command')
    normalized_ax.plot(
        [row[0] for row in normalized], [row[1] for row in normalized],
        color='#0077b6', linewidth=1.4, label='PX4 normalized command',
    )
    for axis in (force_ax, normalized_ax):
        mark_events(axis, events)
        axis.grid(alpha=0.25)
        axis.set_xlim(0, end_s)
        axis.legend(loc='upper left', ncol=2, fontsize=8)
    force_ax.set(title=f'{trial_id} — Thrust command',
                 ylabel='Thrust force (N)')
    normalized_ax.set(xlabel='Time since bag start (s)',
                      ylabel='Normalized thrust (0–1)')
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_angular(path, trial_id, rows, events, end_s):
    fig, ax = plt.subplots(figsize=(12, 5.2), layout='constrained')
    time_s = [row[0] for row in rows]
    for index, label, color in (
        (4, 'Magnitude', '#102a43'), (1, 'x', '#0077b6'),
        (2, 'y', '#e07a28'), (3, 'z', '#8f56a7'),
    ):
        ax.plot(time_s, [row[index] for row in rows], color=color,
                linewidth=2.1 if index == 4 else 1.0, label=label)
    mark_events(ax, events)
    finish(fig, ax, trial_id, 'Angular velocity in earth-fixed x, y, z',
           'Angular velocity (rad/s)', end_s, path)


def plot_thrust_vector(path, trial_id, rows, events, end_s):
    fig, ax = plt.subplots(figsize=(12, 5.2), layout='constrained')
    time_s = [row[0] for row in rows]
    for index, label, color in (
        (4, 'Magnitude', '#102a43'), (1, 'x', '#0077b6'),
        (2, 'y', '#e07a28'), (3, 'z', '#8f56a7'),
    ):
        ax.plot(time_s, [row[index] for row in rows], color=color,
                linewidth=2.1 if index == 4 else 1.0, label=label)
    mark_events(ax, events)
    finish(fig, ax, trial_id, 'Thrust vector in earth-fixed x, y, z',
           'Thrust component (N)', end_s, path)


def plot_image(path, trial_id, rows, events, end_s):
    fig, (ax, area_ax) = plt.subplots(
        2, 1, figsize=(12, 7.4), sharex=True, layout='constrained',
        gridspec_kw={'height_ratios': [2, 1]},
    )
    time_s = [row[0] for row in rows]
    for index, label, color in (
        (1, 'Image x', '#0077b6'), (2, 'Image y', '#e07a28'),
        (3, 'Image error magnitude', '#102a43'),
    ):
        ax.plot(time_s, [row[index] for row in rows], color=color,
                linewidth=1.8 if index == 3 else 1.1, label=label)
    area_ax.plot(time_s, [row[4] for row in rows], color='#8f56a7',
                 linewidth=1.2, label='Detected target area')
    for axis in (ax, area_ax):
        mark_events(axis, events)
        axis.grid(alpha=0.25)
        axis.set_xlim(0, end_s)
        axis.legend(loc='upper left', ncol=2, fontsize=8)
    ax.set(title=f'{trial_id} — Image target features',
           ylabel='Normalized image coordinate')
    area_ax.set(xlabel='Time since bag start (s)', ylabel='Target area (pixels²)')
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.bag.parents[1] / 'plots' / args.bag.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (thrust, thrust_newtons, thrust_earth, angular, features,
     events, end_s) = read_bag(args.bag)
    write_csv(output_dir / 'thrust.csv',
              ('time_s', 'normalized_thrust_command'), thrust)
    write_csv(output_dir / 'thrust_newtons.csv',
              ('time_s', 'controller_thrust_command_n'), thrust_newtons)
    write_csv(output_dir / 'thrust_earth.csv',
              ('time_s', 'fx_earth_n', 'fy_earth_n', 'fz_earth_n',
               'thrust_magnitude_n'), thrust_earth)
    write_csv(output_dir / 'angular_velocity.csv',
              ('time_s', 'omega_x_earth_rad_s', 'omega_y_earth_rad_s',
               'omega_z_earth_rad_s', 'omega_magnitude_rad_s'), angular)
    write_csv(output_dir / 'image_features.csv',
              ('time_s', 'image_x_norm', 'image_y_norm', 'image_error_norm',
               'target_area_px'), features)
    plot_thrust(output_dir / 'thrust_vs_time.png', args.bag.name,
                thrust, thrust_newtons, events, end_s)
    plot_angular(output_dir / 'angular_velocity_vs_time.png', args.bag.name,
                 angular, events, end_s)
    plot_thrust_vector(output_dir / 'thrust_vector_vs_time.png', args.bag.name,
                       thrust_earth, events, end_s)
    plot_image(output_dir / 'image_features_vs_time.png', args.bag.name,
               features, events, end_s)
    print(f'thrust={len(thrust)}, thrust_n={len(thrust_newtons)}, '
          f'thrust_earth={len(thrust_earth)}, '
          f'angular={len(angular)}, '
          f'image={len(features)}; output={output_dir}')


if __name__ == '__main__':
    main()
