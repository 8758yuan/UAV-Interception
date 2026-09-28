"""Plot measured PX4 velocity and acceleration from a recorded P4 trial."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rosbag2_py
from interception_interfaces.msg import ControlDebug
from px4_msgs.msg import VehicleLocalPosition
from rclpy.serialization import deserialize_message
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import Bool


POSITION_TOPIC = '/fmu/out/vehicle_local_position_v1'
DEBUG_TOPIC = '/interception/control/debug'
CONTACT_TOPIC = (
    '/world/default/model/ibvs_target/link/target_link/sensor/'
    'target_contact/contact'
)
GREEN_TOPIC = '/interception/target/green_confirmed'


def read_trial(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('cdr', 'cdr'),
    )
    available_topics = {topic.name for topic in reader.get_all_topics_and_types()}
    types = {
        POSITION_TOPIC: VehicleLocalPosition,
        DEBUG_TOPIC: ControlDebug,
        CONTACT_TOPIC: Contacts,
        GREEN_TOPIC: Bool,
    }
    if POSITION_TOPIC not in available_topics:
        raise ValueError(f'{POSITION_TOPIC} is missing from {bag_path}')

    rows = []
    events = {}
    first_stamp_ns = None
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        if first_stamp_ns is None:
            first_stamp_ns = stamp_ns
        if topic not in (POSITION_TOPIC, DEBUG_TOPIC, CONTACT_TOPIC, GREEN_TOPIC):
            continue
        time_s = (stamp_ns - first_stamp_ns) * 1e-9
        message = deserialize_message(data, types[topic])
        if topic == POSITION_TOPIC:
            values = (
                message.vx, message.vy, message.vz,
                message.ax, message.ay, message.az,
            )
            if not all(math.isfinite(value) for value in values):
                continue
            vx_ned, vy_ned, vz_ned, ax_ned, ay_ned, az_ned = values
            # PX4 local position uses NED; this project uses ENU earth axes.
            vx, vy, vz = vy_ned, vx_ned, -vz_ned
            ax, ay, az = ay_ned, ax_ned, -az_ned
            rows.append((
                time_s, vx, vy, vz,
                math.sqrt(vx * vx + vy * vy + vz * vz),
                ax, ay, az,
                math.sqrt(ax * ax + ay * ay + az * az),
            ))
        elif topic == DEBUG_TOPIC:
            if message.control_state == 'ACTIVE':
                events.setdefault('Active', time_s)
            elif message.control_state == 'ABORT':
                events.setdefault('Safety abort', time_s)
        elif topic == CONTACT_TOPIC and message.contacts:
            events.setdefault('Target contact', time_s)
        elif topic == GREEN_TOPIC and message.data:
            events.setdefault('Green confirmation', time_s)
    if not rows:
        raise ValueError(f'no finite PX4 local-position samples in {bag_path}')
    return rows, events


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow((
            'time_s', 'vx_earth_m_s', 'vy_earth_m_s', 'vz_earth_m_s',
            'speed_m_s', 'ax_earth_m_s2', 'ay_earth_m_s2', 'az_earth_m_s2',
            'acceleration_m_s2',
        ))
        writer.writerows(rows)


def plot_series(path, trial_id, rows, events, indices, ylabel, title):
    time_s = [row[0] for row in rows]
    fig, ax = plt.subplots(figsize=(12, 5.2), layout='constrained')
    colors = ('#102a43', '#0077b6', '#e07a28', '#8f56a7')
    for index, label, color in zip(indices, ('Magnitude', 'x', 'y', 'z'), colors):
        ax.plot(
            time_s, [row[index] for row in rows],
            color=color, linewidth=2.1 if label == 'Magnitude' else 1.0,
            alpha=1.0 if label == 'Magnitude' else 0.78,
            label=label,
        )
    event_styles = (
        ('Active', '#21867a', '--'),
        ('Target contact', '#7a4ca0', ':'),
    )
    for name, color, style in event_styles:
        if name in events:
            ax.axvline(events[name], color=color, linestyle=style,
                       linewidth=1.5, label=f'{name}: {events[name]:.2f} s')
    ax.set(title=f'{trial_id} — {title}', xlabel='Time since bag start (s)',
           ylabel=ylabel)
    ax.set_xlim(0, max(time_s))
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', ncol=2, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.bag.parents[1] / 'plots' / args.bag.name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, events = read_trial(args.bag)
    write_csv(output_dir / 'kinematics.csv', rows)
    plot_series(
        output_dir / 'speed_vs_time.png', args.bag.name, rows, events,
        (4, 1, 2, 3), 'Velocity (m/s)', 'Velocity in earth-fixed x, y, z',
    )
    plot_series(
        output_dir / 'acceleration_vs_time.png', args.bag.name, rows, events,
        (8, 5, 6, 7), 'Acceleration (m/s²)',
        'Acceleration in earth-fixed x, y, z',
    )
    print(f'{len(rows)} samples; events={events}; output={output_dir}')


if __name__ == '__main__':
    main()
