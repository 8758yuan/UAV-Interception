"""Tests for auditable P2 trial reports."""

import json

import pytest

from ibvs_control.p2_trial_report import (
    build_p2_trial_report,
    write_p2_trial_report,
)


def _report():
    return build_p2_trial_report(
        trial_id='p2 truth 01',
        started_utc='2026-09-15T00:00:00+00:00',
        ended_utc='2026-09-15T00:00:40+00:00',
        start_sim_time_s=10.0,
        end_sim_time_s=50.0,
        outcome='PASS',
        reason='hit',
        parameters={'speed_limit_m_s': 2.0},
        gate_evidence={
            'measured_p1_passes': 2,
            'required_p1_passes': 10,
            'p1_gate_waived': True,
        },
        metrics={'d_min_m': 0.4},
        transitions=[{'phase': 'ACTIVE', 'sim_time_s': 15.0}],
    )


def test_report_preserves_waiver_and_metrics(tmp_path) -> None:
    """The permanent record exposes measured evidence and the waiver."""
    path = write_p2_trial_report(_report(), str(tmp_path))
    stored = json.loads(path.read_text(encoding='utf-8'))
    assert stored['trial_id'] == 'p2_truth_01'
    assert stored['outcome'] == 'PASS'
    assert stored['gate_evidence']['measured_p1_passes'] == 2
    assert stored['gate_evidence']['p1_gate_waived'] is True
    assert stored['metrics']['d_min_m'] == 0.4


def test_report_does_not_overwrite_existing_trial(tmp_path) -> None:
    """Trial identifiers remain unique evidence keys."""
    write_p2_trial_report(_report(), str(tmp_path))
    with pytest.raises(FileExistsError):
        write_p2_trial_report(_report(), str(tmp_path))
