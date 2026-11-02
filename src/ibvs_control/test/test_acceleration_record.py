"""Check acceleration math and the real workbook contents without simulation."""

from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np

from ibvs_control.acceleration_record import AccelerationRecord, pose_acceleration
from ibvs_control.dkf_standalone_test import position_offset_enu, trajectory_acceleration_enu


def test_pose_second_difference_unequal_steps():
    acceleration = np.array([2., -3., 4.])
    poses = [(int(t * 1e9), .5 * acceleration * t ** 2) for t in (0., .1, .3)]
    time, actual = pose_acceleration(poses)
    np.testing.assert_allclose(actual, acceleration)
    assert time == 125_000_000


def test_reference_acceleration_matches_position_derivative():
    dt = .001
    for case in 'ABCD':
        for t in (0., 3., 17.):
            numerical = (np.array(position_offset_enu(case, t + dt))
                         - 2 * np.array(position_offset_enu(case, t))
                         + np.array(position_offset_enu(case, t - dt))) / dt ** 2
            np.testing.assert_allclose(trajectory_acceleration_enu(case, t), numerical,
                                       atol=1e-7)


def test_workbook_sampling_missing_values_and_end(tmp_path):
    record = AccelerationRecord(tmp_path)
    record.origin = 1_000_000_000
    record.end = 2_100_000_000
    record.add(0, 1_400_000_000, [1., 2., 3.])
    record.add(0, 1_510_000_000, [4., 5., 6.])
    record.add(1, 1_490_000_000, [-1., -2., -3.])
    record.add(0, 2_300_000_000, [99.] * 3)
    record.save(100_000_000_000)
    with ZipFile(tmp_path / '每0.5秒加速度对比.xlsx') as archive:
        root = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    rows = root.findall('m:sheetData/m:row', ns)
    assert len(rows) == 5  # headers plus 0, 0.5, 1.0 seconds
    cells = rows[3].findall('m:c', ns)
    np.testing.assert_allclose([float(c.find('m:v', ns).text) for c in cells],
                               [0.5, 0.51, 0.49, 4., 5., 6., -1., -2., -3.])
    assert rows[4].findall('m:c', ns)[3].find('m:is/m:t', ns).text == '不可用'


def test_motion_outputs_only_workbook_and_two_plots(tmp_path):
    record = AccelerationRecord(tmp_path, 'trial_')
    record.origin = 1_000_000_000
    record.end = 2_000_000_000
    record.add(0, record.origin, [1., 0., 0.])
    record.add(1, record.origin, [1., 0., 0.])
    record.save(record.end)
    samples = [(1_000_000_000, np.array([1., 2., 3.])),
               (2_000_000_000, np.array([2., 3., 4.]))]
    record.plot_motion(samples, samples, 2_000_000_000)
    assert {p.name for p in tmp_path.iterdir()} == {
        'trial_每0.5秒加速度对比.xlsx', 'trial_速度.png', 'trial_加速度.png'}
    for path in tmp_path.glob('*.png'):
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
