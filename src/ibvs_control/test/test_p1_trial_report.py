"""Tests for structured P1 trial reports and gate evaluation."""

import csv
import json

import pytest

from ibvs_control.p1_trial_report import (
    build_trial_report,
    evaluate_acceptance,
    load_trial_reports,
    sanitize_trial_id,
    write_trial_report,
)
from ibvs_control.rate_test_sequence import RateTestResult


def _rate_results(passed=True):
    return tuple(
        RateTestResult(
            name=f'TEST_{index}',
            commanded_rate_rad_s=0.3,
            mean_rate_rad_s=0.25,
            samples=20,
            passed=passed,
            reason='ok' if passed else 'wrong_sign',
        )
        for index in range(6)
    )


def _report(trial_id, ended_utc, outcome='PASS', results=None):
    return build_trial_report(
        trial_id=trial_id,
        started_utc='2026-09-14T00:00:00.000+00:00',
        ended_utc=ended_utc,
        start_sim_time_s=10.0,
        end_sim_time_s=30.0,
        outcome=outcome,
        reason='ok' if outcome == 'PASS' else 'test_failed',
        terminal_phase='COMPLETE' if outcome == 'PASS' else 'ABORT',
        parameters={'hover_thrust': 0.727},
        max_horizontal_distance_m=0.4,
        max_relative_altitude_m=2.1,
        max_tilt_deg=14.0,
        rate_results=_rate_results() if results is None else results,
    )


def test_write_report_creates_json_and_csv(tmp_path) -> None:
    """One trial should create both detailed and tabular records."""
    report = _report('trial 01', '2026-09-14T00:01:00.000+00:00')
    json_path = write_trial_report(report, str(tmp_path))

    with json_path.open(encoding='utf-8') as stream:
        stored = json.load(stream)
    assert stored['trial_id'] == 'trial_01'
    assert len(stored['rate_results']) == 6

    with (tmp_path / 'p1_trials.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['outcome'] == 'PASS'
    assert rows[0]['passed_rate_tests'] == '6'


def test_load_reports_ignores_invalid_json(tmp_path) -> None:
    """A partial or corrupt file must not break gate evaluation."""
    write_trial_report(
        _report('valid', '2026-09-14T00:01:00.000+00:00'),
        str(tmp_path),
    )
    (tmp_path / 'p1_broken.json').write_text('{', encoding='utf-8')
    reports = load_trial_reports(str(tmp_path))
    assert [report['trial_id'] for report in reports] == ['valid']


def test_acceptance_requires_latest_consecutive_passes() -> None:
    """An abort should reset the acceptance streak."""
    reports = []
    for index in range(12):
        outcome = 'ABORT' if index == 3 else 'PASS'
        reports.append(
            _report(
                f'trial_{index:02d}',
                f'2026-09-14T00:{index:02d}:00.000+00:00',
                outcome,
            )
        )

    result = evaluate_acceptance(reports, required_passes=10)
    assert result.passed is False
    assert result.consecutive_passes == 8


def test_acceptance_rejects_incomplete_rate_results() -> None:
    """A nominal outcome without all six checks is not a valid PASS."""
    report = _report(
        'incomplete',
        '2026-09-14T00:01:00.000+00:00',
        results=_rate_results()[:5],
    )
    result = evaluate_acceptance([report], required_passes=1)
    assert result.passed is False


def test_trial_id_validation() -> None:
    """Identifiers should be safe and must retain useful content."""
    assert sanitize_trial_id(' run 01/left ') == 'run_01_left'
    with pytest.raises(ValueError):
        sanitize_trial_id('---')
