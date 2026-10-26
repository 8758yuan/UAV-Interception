"""Evaluation checks use synthetic truth, not a closed-loop experiment."""

import csv

import numpy as np

from ibvs_control.dkf_evaluation import EvaluationCsv, truth_between


def test_time_frames_projection_and_unavailable_biases():
    q = np.array([1., 0., 0., 0.])
    before = (1_000_000_000, np.zeros(3), q, np.array([10., 0., 0.]))
    after = (3_000_000_000, np.array([2., 0., 0.]), -q, np.array([14., 2., 0.]))
    rotation = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    truth = truth_between(before, after, 2_000_000_000, rotation, np.array([0.14, 0., 0.]))
    np.testing.assert_allclose(truth[:4], q)
    np.testing.assert_allclose(truth[4:7], [-11., -1., 0.])
    np.testing.assert_allclose(truth[7:10], [-1., -1., 0.])
    np.testing.assert_allclose(truth[10:12], [-1 / 10.86, 0.])
    assert np.isnan(truth[12:]).all()
    assert np.isnan(truth_between(before, after, 0, rotation, np.zeros(3))).all()


def test_csv_all_states_sign_and_statistics(tmp_path):
    writer = EvaluationCsv(tmp_path / 'run')
    truth = np.zeros(18)
    truth[0] = -1
    truth[12:] = np.nan
    estimate = np.zeros(18)
    estimate[0] = 1
    estimate[4] = 2
    writer.write([1, 2, 3, True, True, 'test', 'test', 0, 4, 0], truth, estimate)
    writer.write([2, 3, 4, True, True, 'test', 'test', 0, 5, 0], truth, estimate)
    writer.close()
    rows = list(csv.DictReader((tmp_path / 'run' / 'states.csv').open()))
    assert len(rows) == 2
    assert len(rows[0]) == 10 + 18 * 3
    assert float(rows[0]['q_w_error']) == 0
    metrics = {r['state']: r for r in csv.DictReader(
        (tmp_path / 'run' / 'metrics.csv').open())}
    assert float(metrics['p_r_e']['rmse']) == 2
    assert float(metrics['p_r_e']['mae']) == 2
    assert float(metrics['p_r_e']['max_abs_error']) == 2
    assert metrics['b_gyr_f']['count'] == '0'
