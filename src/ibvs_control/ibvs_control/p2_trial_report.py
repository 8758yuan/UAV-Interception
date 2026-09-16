"""Structured, auditable records for P2 truth-interception trials."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from ibvs_control.p1_trial_report import sanitize_trial_id


P2_SCHEMA_VERSION = '1.0'


def utc_now_iso() -> str:
    """Return the current UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def build_p2_trial_report(
    *,
    trial_id: str,
    started_utc: str,
    ended_utc: str,
    start_sim_time_s: float,
    end_sim_time_s: float,
    outcome: str,
    reason: str,
    parameters: Mapping[str, Any],
    gate_evidence: Mapping[str, Any],
    metrics: Mapping[str, Any],
    transitions: list,
    milestone: str = 'P2',
    control_mode: str = 'truth_closed_loop',
) -> dict:
    """Build one JSON-compatible trial report without hiding a waived gate."""
    if outcome not in ('PASS', 'ABORT'):
        raise ValueError('outcome must be PASS or ABORT')
    if not milestone or not control_mode:
        raise ValueError('milestone and control_mode must not be empty')
    return {
        'schema_version': P2_SCHEMA_VERSION,
        'milestone': milestone,
        'control_mode': control_mode,
        'trial_id': sanitize_trial_id(trial_id),
        'started_utc': started_utc,
        'ended_utc': ended_utc,
        'start_sim_time_s': float(start_sim_time_s),
        'end_sim_time_s': float(end_sim_time_s),
        'sim_duration_s': max(0.0, end_sim_time_s - start_sim_time_s),
        'outcome': outcome,
        'reason': reason,
        'gate_evidence': dict(gate_evidence),
        'parameters': dict(parameters),
        'metrics': dict(metrics),
        'transitions': list(transitions),
    }


def write_p2_trial_report(report: Mapping[str, Any], directory: str) -> Path:
    """Atomically write a P2 report at a deterministic trial path."""
    output_dir = Path(directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_id = sanitize_trial_id(str(report['trial_id']))
    path = output_dir / f'{trial_id}.json'
    if path.exists():
        raise FileExistsError(f'P2 trial report already exists: {path}')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)
    return path
