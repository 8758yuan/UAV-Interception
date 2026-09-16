"""Structured trial records and acceptance checks for the P1 flight gate."""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ibvs_control.rate_test_sequence import RateTestResult


SCHEMA_VERSION = '1.0'
CSV_FIELDS = (
    'ended_utc',
    'trial_id',
    'outcome',
    'reason',
    'terminal_phase',
    'sim_duration_s',
    'passed_rate_tests',
    'total_rate_tests',
    'max_horizontal_distance_m',
    'max_relative_altitude_m',
    'max_tilt_deg',
    'json_file',
)


@dataclass(frozen=True)
class P1Acceptance:
    """Result of checking the most recent consecutive P1 trials."""

    passed: bool
    required_passes: int
    consecutive_passes: int
    total_reports: int
    latest_trial_id: str


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp suitable for report metadata."""
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def build_trial_report(
    *,
    trial_id: str,
    started_utc: str,
    ended_utc: str,
    start_sim_time_s: float,
    end_sim_time_s: float,
    outcome: str,
    reason: str,
    terminal_phase: str,
    parameters: Mapping[str, Any],
    max_horizontal_distance_m: float,
    max_relative_altitude_m: float,
    max_tilt_deg: float,
    rate_results: Sequence[RateTestResult],
) -> Dict[str, Any]:
    """Build one JSON-serializable P1 trial report."""
    if outcome not in ('PASS', 'ABORT'):
        raise ValueError('outcome must be PASS or ABORT')
    clean_id = sanitize_trial_id(trial_id)
    duration_s = max(0.0, end_sim_time_s - start_sim_time_s)
    return {
        'schema_version': SCHEMA_VERSION,
        'milestone': 'P1',
        'trial_id': clean_id,
        'started_utc': started_utc,
        'ended_utc': ended_utc,
        'start_sim_time_s': float(start_sim_time_s),
        'end_sim_time_s': float(end_sim_time_s),
        'sim_duration_s': float(duration_s),
        'outcome': outcome,
        'reason': reason,
        'terminal_phase': terminal_phase,
        'parameters': dict(parameters),
        'safety_observations': {
            'max_horizontal_distance_m': float(
                max_horizontal_distance_m
            ),
            'max_relative_altitude_m': float(max_relative_altitude_m),
            'max_tilt_deg': float(max_tilt_deg),
        },
        'rate_results': [asdict(result) for result in rate_results],
    }


def sanitize_trial_id(trial_id: str) -> str:
    """Return a short filesystem-safe trial identifier."""
    clean_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', trial_id.strip())
    clean_id = clean_id.strip('._-')
    if not clean_id:
        raise ValueError('trial_id must contain a letter or number')
    return clean_id[:80]


def write_trial_report(
    report: Mapping[str, Any],
    results_directory: str,
) -> Path:
    """Atomically write a JSON report and append its row to the CSV index."""
    output_dir = Path(results_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_id = sanitize_trial_id(str(report['trial_id']))
    timestamp = _filename_timestamp(str(report['ended_utc']))
    json_path = output_dir / f'p1_{timestamp}_{trial_id}.json'
    suffix = 1
    while json_path.exists():
        json_path = output_dir / f'p1_{timestamp}_{trial_id}_{suffix:02d}.json'
        suffix += 1

    temporary_path = json_path.with_suffix('.json.tmp')
    with temporary_path.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    temporary_path.replace(json_path)

    _append_csv(output_dir / 'p1_trials.csv', report, json_path.name)
    return json_path


def load_trial_reports(results_directory: str) -> List[Dict[str, Any]]:
    """Load valid P1 JSON reports in chronological order."""
    output_dir = Path(results_directory).expanduser().resolve()
    reports = []
    for path in output_dir.glob('p1_*.json'):
        try:
            with path.open(encoding='utf-8') as stream:
                report = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue
        if report.get('milestone') != 'P1':
            continue
        report['_source_path'] = str(path)
        reports.append(report)
    reports.sort(key=lambda item: str(item.get('ended_utc', '')))
    return reports


def evaluate_acceptance(
    reports: Iterable[Mapping[str, Any]],
    required_passes: int = 10,
) -> P1Acceptance:
    """Require the newest trials to form a continuous PASS streak."""
    if required_passes <= 0:
        raise ValueError('required_passes must be greater than zero')
    ordered = sorted(
        reports,
        key=lambda item: str(item.get('ended_utc', '')),
    )
    consecutive = 0
    for report in reversed(ordered):
        if report.get('outcome') != 'PASS':
            break
        if not _six_rate_checks_passed(report):
            break
        consecutive += 1

    latest_id = str(ordered[-1].get('trial_id', '')) if ordered else ''
    return P1Acceptance(
        passed=consecutive >= required_passes,
        required_passes=required_passes,
        consecutive_passes=consecutive,
        total_reports=len(ordered),
        latest_trial_id=latest_id,
    )


def _six_rate_checks_passed(report: Mapping[str, Any]) -> bool:
    results = report.get('rate_results', [])
    return (
        isinstance(results, list)
        and len(results) == 6
        and all(result.get('passed') is True for result in results)
    )


def _filename_timestamp(timestamp: str) -> str:
    return re.sub(r'[^0-9]', '', timestamp)[:17] or 'unknown_time'


def _append_csv(
    csv_path: Path,
    report: Mapping[str, Any],
    json_filename: str,
) -> None:
    observations = report.get('safety_observations', {})
    results = report.get('rate_results', [])
    row = {
        'ended_utc': report.get('ended_utc', ''),
        'trial_id': report.get('trial_id', ''),
        'outcome': report.get('outcome', ''),
        'reason': report.get('reason', ''),
        'terminal_phase': report.get('terminal_phase', ''),
        'sim_duration_s': report.get('sim_duration_s', ''),
        'passed_rate_tests': sum(
            result.get('passed') is True for result in results
        ),
        'total_rate_tests': len(results),
        'max_horizontal_distance_m': observations.get(
            'max_horizontal_distance_m',
            '',
        ),
        'max_relative_altitude_m': observations.get(
            'max_relative_altitude_m',
            '',
        ),
        'max_tilt_deg': observations.get('max_tilt_deg', ''),
        'json_file': json_filename,
    }
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _parse_arguments(args: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check the latest consecutive P1 body-rate trials.',
    )
    parser.add_argument(
        '--results-directory',
        default='results/p1',
        help='directory containing p1_*.json trial files',
    )
    parser.add_argument(
        '--required-passes',
        type=int,
        default=10,
        help='number of consecutive latest PASS trials required',
    )
    return parser.parse_args(args)


def main(args: Sequence[str] = None) -> int:
    """Run the command-line P1 acceptance check."""
    options = _parse_arguments(args)
    reports = load_trial_reports(options.results_directory)
    result = evaluate_acceptance(reports, options.required_passes)
    status = 'PASS' if result.passed else 'NOT READY'
    print(
        f'P1 gate: {status}; consecutive_passes='
        f'{result.consecutive_passes}/{result.required_passes}; '
        f'total_reports={result.total_reports}; '
        f'latest_trial={result.latest_trial_id or "none"}'
    )
    return 0 if result.passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
