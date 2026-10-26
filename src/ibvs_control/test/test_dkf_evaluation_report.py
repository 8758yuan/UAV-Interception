"""Readable reports must preserve measured values and 0.5 s sampling semantics."""

import csv
from pathlib import Path
import subprocess
import sys

from ibvs_control.dkf_evaluation import STATE_NAMES
from ibvs_control.dkf_evaluation_report import STATE_COLUMNS, read_samples, snapshot_rows


def write_fixture(directory):
    directory.mkdir()
    headers = ['initialized', 'state_time_ns'] + [f'{name}_{kind}' for name in STATE_NAMES
                                               for kind in ('true', 'estimate', 'error')]
    with (directory / 'states.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, headers)
        writer.writeheader()
        writer.writerow({'initialized': 'False', 'state_time_ns': ''})
        # The 0.49 s record is closer to 0.5 s than the 0.52 s record.
        # There is no record near 1.5 s: the report must not fill this gap.
        for index, elapsed in enumerate((0, 490_000_000, 520_000_000,
                                         1_010_000_000, 2_000_000_000)):
            row = {'initialized': 'True', 'state_time_ns': 10_000_000_000 + elapsed}
            for name in STATE_NAMES:
                row.update({f'{name}_true': index,
                            f'{name}_estimate': index + 0.25,
                            f'{name}_error': 0.25})
            for name in STATE_NAMES[12:]:
                row[f'{name}_true'] = row[f'{name}_error'] = 'nan'
            writer.writerow(row)
    with (directory / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['state', 'count', 'rmse', 'mae', 'max_abs_error'])
        for name in STATE_NAMES:
            writer.writerow([name, 5, .25, .25, .25] if name in STATE_NAMES[:12]
                            else [name, 0, 'nan', 'nan', 'nan'])


def test_half_second_selects_real_samples_and_retains_missing_values(tmp_path):
    directory = tmp_path / 'recording'
    write_fixture(directory)
    rows, samples, origin = read_samples(directory)
    assert origin == 10_000_000_000
    assert len(rows) == 5
    assert samples[1]['_elapsed_ns'] == 490_000_000
    assert samples[3] is None
    output = list(snapshot_rows(samples))
    assert output[1][:2] == ['0.5', '0.490']
    position_index = 2 + 3 * [name for name, _, _ in STATE_COLUMNS].index('p_r_e')
    assert output[1][position_index:position_index + 3] == ['1.000', '1.250', '0.250']
    assert len(output[1]) == 2 + 18 * 3
    assert output[3][-3:] == ['不可用'] * 3
    assert output[0][-3:] == ['不可用', '0.250', '不可用']


def test_report_is_self_contained_and_leaves_raw_recording_unchanged(tmp_path):
    directory = tmp_path / 'recording'
    write_fixture(directory)
    original = (directory / 'states.csv').read_bytes()
    script = Path(__file__).parents[1] / 'ibvs_control' / 'dkf_evaluation_report.py'
    subprocess.run([sys.executable, '-s', str(script), str(directory)], check=True)
    page = (directory / '测试结果.html').read_text()
    assert '每隔 0.5 秒的数据对比' in page
    assert '<svg' in page
    assert '平均相差' in page
    assert '不可用' in page
    assert '<script src=' not in page
    assert 'publish_time_ns' not in page
    assert (directory / 'states.csv').read_bytes() == original
    from zipfile import ZipFile
    from xml.etree import ElementTree as ET

    assert not (directory / '每0.5秒对比.csv').exists()
    assert 'href="每0.5秒对比.xlsx"' in page
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(directory / '每0.5秒对比.xlsx') as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            ET.fromstring(archive.read(name))
        sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        summary = ET.fromstring(archive.read('xl/worksheets/sheet2.xml'))
    data_rows = sheet.findall('s:sheetData/s:row', ns)[2:]
    assert len(data_rows) == 5
    assert all(len(row) == 2 + 18 * 3 for row in data_rows)
    cells = {cell.attrib['r']: cell for row in data_rows for cell in row}
    assert cells['A4'].find('s:v', ns).text == '0.5'
    assert abs(float(cells['B4'].find('s:v', ns).text) - .49) < 1e-12
    assert float(cells['O4'].find('s:v', ns).text) == 1.
    assert cells['BD3'].find('s:is/s:t', ns).text == '不可用'
    assert cells['BC3'].find('s:v', ns) is not None  # Bias estimate stays numeric.
    assert all(cell.find('s:is/s:t', ns).text == '不可用'
               for cell in list(data_rows[3])[2:])
    pane = sheet.find('s:sheetViews/s:sheetView/s:pane', ns)
    assert pane.attrib['topLeftCell'] == 'C3'
    assert pane.attrib['xSplit'] == pane.attrib['ySplit'] == '2'
    assert len(sheet.findall('s:mergeCells/s:mergeCell', ns)) == 20
    assert len(summary.findall('s:sheetData/s:row', ns)) == 20


def test_report_filters_columns_without_removing_state_data(tmp_path):
    """Exercise real browser controls, header spans and sticky scroll behavior."""
    import shutil
    import pytest

    from ibvs_control.dkf_evaluation_report import snapshot_controls, snapshot_table

    chrome = shutil.which('google-chrome')
    if not chrome:
        pytest.skip('Chrome is required for the report interaction check')
    # Read the actual template script/CSS, so the check runs production behavior.
    script_path = Path(__file__).parents[1] / 'ibvs_control' / 'dkf_evaluation_report.py'
    source = script_path.read_text()
    css = source.split('<style>', 1)[1].split('</style>', 1)[0]
    script = source.split('</main><script>', 1)[1].split('</script></html>', 1)[0]
    check = r'''
const result = document.createElement('pre');
result.id = 'browser-check';
try {
  const assert = (condition, message) => { if(!condition) throw new Error(message); };
  const visible = selector => document.querySelectorAll('#snapshot-table ' + selector + ':not([hidden])').length;
  const firstRow = document.querySelector('#snapshot-table tbody tr');
  assert(firstRow.children.length === 56, 'all 18 states remain in each row');
  assert(visible('[data-state-heading]') === 6, 'default position and velocity');
  snapshotMode.value = 'error';
  snapshotMode.dispatchEvent(new Event('change'));
  assert(visible('tbody tr:first-child [data-kind]') === 6, 'error-only columns');
  assert([...document.querySelectorAll('[data-state-heading]')].every(h => h.colSpan === 1), 'error header spans');
  document.getElementById('show-all-states').click();
  assert(visible('[data-state-heading]') === 18, 'show all state groups');
  snapshotMode.value = 'all';
  snapshotMode.dispatchEvent(new Event('change'));
  assert(visible('tbody tr:first-child [data-kind]') === 54, 'all state triplets');
  const wrap = document.getElementById('snapshot-table');
  const timeCell = firstRow.children[0];
  const actualCell = firstRow.children[1];
  const timeX = timeCell.getBoundingClientRect().left;
  const actualX = actualCell.getBoundingClientRect().left;
  const header = document.querySelector('[data-state-heading=""]');
  const headerY = header.getBoundingClientRect().top;
  wrap.scrollLeft = 500;
  wrap.scrollTop = 100;
  assert(Math.abs(timeCell.getBoundingClientRect().left - timeX) < 1, 'fixed nominal time');
  assert(Math.abs(actualCell.getBoundingClientRect().left - actualX) < 1, 'fixed actual time');
  assert(Math.abs(header.getBoundingClientRect().top - headerY) < 1, 'fixed header');
  document.getElementById('show-main-states').click();
  assert(visible('[data-state-heading]') === 6, 'restore main states');
  const velocity = document.querySelector('input[value="velocity"]');
  velocity.checked = false;
  velocity.dispatchEvent(new Event('change'));
  assert(visible('[data-state-heading]') === 3, 'individual checkbox');
  assert(firstRow.children.length === 56, 'data retained after filtering');
  result.textContent = 'passed';
} catch(error) { result.textContent = error.message; }
document.body.appendChild(result);
'''
    # Enough rows for both horizontal and vertical scrolling.
    page = ('<!doctype html><meta charset="utf-8"><style>' + css + '</style><main>'
            '<select id="group"></select>' + snapshot_controls()
            + snapshot_table([['0.0', '0.000'] + ['1.000'] * 54] * 30)
            + '</main><script>' + script + check + '</script>')
    path = tmp_path / 'browser.html'
    path.write_text(page)
    browser = subprocess.run([
        chrome, '--headless', '--no-sandbox', '--disable-gpu', '--dump-dom', path.as_uri(),
    ], capture_output=True, text=True, check=True)
    assert '<pre id="browser-check">passed</pre>' in browser.stdout, browser.stdout[-1500:]


def test_event_boundaries_are_present_even_between_dkf_samples(tmp_path):
    import json

    directory = tmp_path / 'recording'
    write_fixture(directory)
    (directory / 'metadata.json').write_text(json.dumps({
        'recording_start_time_ns': 9_900_000_000,
        'recording_end_time_ns': 12_230_000_000,
    }))
    rows, samples, origin = read_samples(directory)
    snapshots = list(snapshot_rows(samples))
    assert origin == 9_900_000_000
    assert snapshots[0][:2] == ['0.0', '0.100']
    assert snapshots[-1][:2] == ['2.330', '2.100']
    assert snapshots[-1][2:] == list(snapshot_rows({0: rows[-1]}))[0][2:]
    assert [row[0] for row in snapshots] == ['0.0', '0.5', '1.0', '1.5', '2.0', '2.330']
