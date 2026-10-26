"""Create a Chinese, self-contained report from a completed DKF recording."""

import argparse
import csv
from html import escape
from io import StringIO
import math
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


# Names and ordering follow the actual 18-state observer, without conversions.
GROUPS = (
    ('position', '相对位置', '米', (
        ('p_r_e', '相对位置 X（东）'), ('p_r_n', '相对位置 Y（北）'),
        ('p_r_u', '相对位置 Z（上）'))),
    ('velocity', '相对速度', '米/秒', (
        ('v_r_e', '相对速度 X（东）'), ('v_r_n', '相对速度 Y（北）'),
        ('v_r_u', '相对速度 Z（上）'))),
    ('attitude', '姿态四元数', '无量纲', (
        ('q_w', '姿态 w'), ('q_x', '姿态 x'),
        ('q_y', '姿态 y'), ('q_z', '姿态 z'))),
    ('image', '图像坐标', '无量纲', (
        ('image_x', '图像水平坐标'), ('image_y', '图像垂直坐标'))),
    ('gyro', '陀螺仪偏置', '弧度/秒', (
        ('b_gyr_f', '陀螺仪偏置（前）'), ('b_gyr_l', '陀螺仪偏置（左）'),
        ('b_gyr_u', '陀螺仪偏置（上）'))),
    ('accel', '加速度计偏置', '米/秒²', (
        ('b_acc_f', '加速度计偏置（前）'), ('b_acc_l', '加速度计偏置（左）'),
        ('b_acc_u', '加速度计偏置（上）'))),
)
PERIOD_NS = 500_000_000


def display(value):
    """Use readable numbers and explicit missing values, never fabricated zeros."""
    number = float(value)
    if not math.isfinite(number):
        return '不可用'
    if number and abs(number) < 0.0005:
        return f'{number:.2e}'
    return f'{number:.3f}'


def read_samples(directory):
    """Use state time, excluding initialization with no physical time anchor."""
    with (directory / 'states.csv').open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream)
                if row['initialized'] == 'True' and row['state_time_ns']]
    metadata_path = directory / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    start = metadata.get('recording_start_time_ns')
    end = metadata.get('recording_end_time_ns')
    if start is not None:
        rows = [row for row in rows if int(row['state_time_ns']) >= start
                and (end is None or int(row['state_time_ns']) <= end)]
    if not rows:
        samples = {0: None, (end - start) / PERIOD_NS: None} if start is not None and end is not None else {}
        return [], samples, start or 0
    origin = start if start is not None else int(rows[0]['state_time_ns'])
    samples = {}
    for row in rows:
        elapsed_ns = int(row['state_time_ns']) - origin
        row['_elapsed_ns'] = elapsed_ns
        # Presentation sampling only: nearest actual output in each half-second bin.
        index = (elapsed_ns + PERIOD_NS // 2) // PERIOD_NS
        if index not in samples or abs(elapsed_ns - index * PERIOD_NS) < abs(
                samples[index]['_elapsed_ns'] - index * PERIOD_NS):
            samples[index] = row
    duration_ns = end - origin if end is not None else max(row['_elapsed_ns'] for row in rows)
    last_index = duration_ns // PERIOD_NS
    samples = {i: samples.get(i) for i in range(last_index + 1)}
    samples[0] = rows[0]
    samples[duration_ns / PERIOD_NS] = rows[-1]
    return rows, samples, origin


# Match the observer's q, p_r, v_r, image, gyro-bias, accel-bias order.
STATE_COLUMNS = tuple(
    (name, label, unit)
    for group_key in ('attitude', 'position', 'velocity', 'image', 'gyro', 'accel')
    for key, _, unit, states in GROUPS if key == group_key
    for name, label in states
)


def snapshot_rows(samples):
    """One physical sample per row, containing all 18 original state pairs."""
    for index, sample in samples.items():
        row = [f'{index * 0.5:.1f}' if float(index).is_integer() else f'{index * 0.5:.3f}',
               f'{sample["_elapsed_ns"] * 1e-9:.3f}' if sample else '不可用']
        for name, _, _ in STATE_COLUMNS:
            row.extend([display(sample[f'{name}_{kind}'])
                        for kind in ('true', 'estimate', 'error')]
                       if sample else ['不可用'] * 3)
        yield row


def write_excel(directory, samples, metrics):
    """Write the two-sheet XLSX using the standard Open XML ZIP format."""
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'

    def column(index):
        name = ''
        while index:
            index, digit = divmod(index - 1, 26)
            name = chr(65 + digit) + name
        return name

    def number(value):
        value = float(value)
        return value if math.isfinite(value) else '不可用'

    def worksheet(rows, freeze, merges=()):
        xml = [f'<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0">'
               f'<pane {freeze} state="frozen"/></sheetView></sheetViews>'
               '<sheetFormatPr defaultRowHeight="24"/><cols>'
               f'<col min="1" max="{max(map(len, rows))}" width="16" customWidth="1"/>'
               '</cols><sheetData>']
        for i, values in enumerate(rows, 1):
            header = i <= 2
            xml.append(f'<row r="{i}" ht="{36 if header else 24}" customHeight="1">')
            for j, value in enumerate(values, 1):
                style = 1 if header else (2 if i % 2 == 0 else 0)
                reference = f'{column(j)}{i}'
                if isinstance(value, (int, float)):
                    if value and abs(value) < 0.0005:
                        style = 4 if i % 2 == 0 else 3
                    xml.append(f'<c r="{reference}" s="{style}"><v>{value}</v></c>')
                else:
                    xml.append(f'<c r="{reference}" s="{style}" t="inlineStr">'
                               f'<is><t>{escape(str(value))}</t></is></c>')
            xml.append('</row>')
        xml.append('</sheetData>')
        if merges:
            xml.append(f'<mergeCells count="{len(merges)}">')
            xml.extend(f'<mergeCell ref="{cell_range}"/>' for cell_range in merges)
            xml.append('</mergeCells>')
        xml.append('</worksheet>')
        return ''.join(xml)

    top = ['时间（秒）', '实际取样（秒）']
    second = ['', '']
    merges = ['A1:A2', 'B1:B2']
    for i, (_, label, unit) in enumerate(STATE_COLUMNS):
        top.extend([f'{label}（{unit}）', '', ''])
        second.extend(['真实值', 'DKF估计值', '误差'])
        merges.append(f'{column(3 + i * 3)}1:{column(5 + i * 3)}1')
    comparisons = [top, second]
    for index, sample in samples.items():
        values = [index * 0.5, sample['_elapsed_ns'] * 1e-9 if sample else '不可用']
        for name, _, _ in STATE_COLUMNS:
            values.extend([number(sample[f'{name}_{kind}'])
                           for kind in ('true', 'estimate', 'error')]
                          if sample else ['不可用'] * 3)
        comparisons.append(values)
    summary = [
        ['DKF 误差汇总（全频率统计；速度真值为差分近似，偏置真值不可用）'],
        ['状态', '单位', '平均绝对误差', '最大绝对误差', '均方根误差', '有效比较次数'],
    ]
    for name, label, unit in STATE_COLUMNS:
        metric = metrics[name]
        summary.append([label, unit, number(metric['mae']), number(metric['max_abs_error']),
                        number(metric['rmse']), int(metric['count'])])
    relationships = 'http://schemas.openxmlformats.org/package/2006/relationships'
    document_rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    styles = f'<styleSheet xmlns="{ns}">'
    styles += ('<numFmts count="2"><numFmt numFmtId="164" formatCode="0.000"/>'
               '<numFmt numFmtId="165" formatCode="0.00E+00"/></numFmts>'
               '<fonts count="2"><font><sz val="11"/><name val="Microsoft YaHei"/></font>'
               '<font><b/><sz val="11"/><name val="Microsoft YaHei"/></font></fonts>'
               '<fills count="4"><fill><patternFill patternType="none"/></fill>'
               '<fill><patternFill patternType="gray125"/></fill>'
               '<fill><patternFill patternType="solid"><fgColor rgb="FFDCEEE9"/></patternFill></fill>'
               '<fill><patternFill patternType="solid"><fgColor rgb="FFF2F6F8"/></patternFill></fill>'
               '</fills><borders count="1"><border/></borders>'
               '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
               '<cellXfs count="5">')
    for num_format, font, fill in ((164, 0, 0), (0, 1, 2), (164, 0, 3),
                                   (165, 0, 0), (165, 0, 3)):
        styles += (f'<xf numFmtId="{num_format}" fontId="{font}" fillId="{fill}" '
                   'borderId="0" xfId="0" applyNumberFormat="1" applyAlignment="1">'
                   '<alignment horizontal="center" vertical="center" wrapText="1"/></xf>')
    styles += '</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'
    path = directory / '每0.5秒对比.xlsx'
    with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml',
                        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                        '<Default Extension="xml" ContentType="application/xml"/>'
                        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                        + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                                  for i in (1, 2)) + '</Types>')
        archive.writestr('_rels/.rels', f'<Relationships xmlns="{relationships}">'
                        f'<Relationship Id="rId1" Type="{document_rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="{document_rel}"><sheets>'
                        '<sheet name="每0.5秒对比" sheetId="1" r:id="rId1"/>'
                        '<sheet name="误差汇总" sheetId="2" r:id="rId2"/></sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', f'<Relationships xmlns="{relationships}">'
                        + ''.join(f'<Relationship Id="rId{i}" Type="{document_rel}/worksheet" Target="worksheets/sheet{i}.xml"/>'
                                  for i in (1, 2))
                        + f'<Relationship Id="rId3" Type="{document_rel}/styles" Target="styles.xml"/></Relationships>')
        archive.writestr('xl/styles.xml', styles)
        archive.writestr('xl/worksheets/sheet1.xml', worksheet(
            comparisons, 'xSplit="2" ySplit="2" topLeftCell="C3" activePane="bottomRight"', merges))
        archive.writestr('xl/worksheets/sheet2.xml', worksheet(
            summary, 'ySplit="2" topLeftCell="A3" activePane="bottomLeft"', ['A1:F1']))
    return path


def snapshot_table(rows):
    """Retain every state cell; filter only the report's visible columns."""
    group_for = {name: key for key, _, _, states in GROUPS for name, _ in states}
    head = ('<tr><th class="time-nominal" rowspan="2">时间（秒）</th>'
            '<th class="time-actual" rowspan="2">实际取样（秒）</th>')
    subhead = []
    columns = []
    for name, label, unit in STATE_COLUMNS:
        group = group_for[name]
        attrs = f'data-state-group="{group}"'
        hidden = '' if group in ('position', 'velocity') else ' hidden'
        head += (f'<th {attrs} data-state-heading colspan="3"{hidden}>'
                 f'{escape(label)}<br><small>{escape(unit)}</small></th>')
        for kind, title in (('true', '真实值'), ('estimate', '估计值'), ('error', '误差')):
            cell_attrs = f'{attrs} data-kind="{kind}"'
            columns.append((cell_attrs, hidden))
            subhead.append(f'<th {cell_attrs}{hidden}>{title}</th>')
    head += '</tr><tr>' + ''.join(subhead) + '</tr>'
    body = []
    for row in rows:
        cells = (f'<td class="time-nominal">{escape(row[0])}</td>'
                 f'<td class="time-actual">{escape(row[1])}</td>')
        cells += ''.join(f'<td {attrs}{hidden}>{escape(str(value))}</td>'
                         for value, (attrs, hidden) in zip(row[2:], columns))
        body.append('<tr>' + cells + '</tr>')
    return ('<div class="table-wrap wide" id="snapshot-table" tabindex="0" '
            'aria-label="每0.5秒全部状态对比，可横向滚动"><table><thead>' + head
            + '</thead><tbody>' + ''.join(body) + '</tbody></table></div>')


def snapshot_controls():
    checks = ''.join(
        f'<label><input type="checkbox" name="state-group" value="{key}"'
        + (' checked' if key in ('position', 'velocity') else '')
        + f'> {title}</label>' for key, title, _, _ in GROUPS
    )
    return (
        '<div class="snapshot-controls"><fieldset><legend>选择显示的状态</legend>'
        + checks + '</fieldset><div class="toolbar">'
        '<button type="button" id="show-all-states">显示全部状态</button>'
        '<button type="button" id="show-main-states">只看位置和速度</button>'
        '<label>显示内容：<select id="snapshot-mode">'
        '<option value="all">真实值＋估计值＋误差</option>'
        '<option value="error">只看误差</option></select></label></div>'
        '<p id="snapshot-status" aria-live="polite">当前显示 6 / 18 个状态</p></div>'
    )


def comparison_chart(rows, states, unit):
    """Render standard Matplotlib figures, embedded directly in the HTML."""
    if not rows:
        return '<p>尚无带状态时间的 DKF 数据，无法绘制曲线。</p>'
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_svg import FigureCanvasSVG

    with rc_context({'font.family': ['Noto Sans CJK JP', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'svg.fonttype': 'none'}):
        figure = Figure(figsize=(10, 2.25 * len(states)), layout='constrained')
        FigureCanvasSVG(figure)
        axes = figure.subplots(len(states), 1, squeeze=False)
        times = [row['_elapsed_ns'] * 1e-9 for row in rows]
        for axis, (name, label) in zip(axes[:, 0], states):
            axis.plot(times, [float(row[f'{name}_true']) for row in rows],
                      label='真实值', color='#167c80', linewidth=1.6)
            axis.plot(times, [float(row[f'{name}_estimate']) for row in rows],
                      label='DKF 估计值', color='#d26020', linewidth=1.3,
                      linestyle='--')
            axis.set_title(label, loc='left', fontsize=11)
            axis.set_ylabel(unit)
            axis.grid(alpha=0.18)
            axis.legend(loc='best', frameon=False)
        axes[-1, 0].set_xlabel('测试时间（秒）')
        image = StringIO()
        figure.savefig(image, format='svg')
        svg = image.getvalue()
        return svg[svg.index('<svg'):]


def table(headers, rows):
    head = ''.join(f'<th>{escape(str(value))}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in row)
                   + '</tr>' for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def generate_report(directory):
    """Generate the readable report and Excel workbook without changing raw files."""
    directory = Path(directory)
    rows, samples, _origin = read_samples(directory)
    metadata_path = directory / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    with (directory / 'metrics.csv').open(newline='') as stream:
        metrics = {row['state']: row for row in csv.DictReader(stream)}
    summary = []
    sections = []
    snapshots = list(snapshot_rows(samples))
    write_excel(directory, samples, metrics)
    for key, title, unit, states in GROUPS:
        summary_rows = []
        for name, label in states:
            metric = metrics[name]
            summary_rows.append([
                label, unit, display(metric['mae']), display(metric['max_abs_error']),
            ])
        summary_table = table(['状态', '单位', '平均相差', '最大相差'], summary_rows)
        if key in ('position', 'velocity'):
            summary.append(f'<h3>{title}</h3>{summary_table}')
        else:
            summary.append(f'<details><summary>{title}</summary>{summary_table}</details>')
        hidden = '' if key == 'position' else ' hidden'
        sections.append(
            f'<section class="comparison" data-group="{key}"{hidden}>'
            f'<h3>{title}：真实值与估计值</h3>{comparison_chart(rows, states, unit)}</section>'
        )
    duration = max(samples, default=0) * 0.5
    comparison_count = int(metrics['p_r_e']['count'])
    options = ''.join(f'<option value="{key}">{title}</option>' for key, title, _, _ in GROUPS)
    empty = '<p class="notice">本次没有可评价的相对位置真值，请检查原始记录中的真值或对时状态。</p>' if not comparison_count else ''
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>DKF 测试结果</title>
<style>
body{font:16px/1.65 system-ui,sans-serif;color:#213039;background:#f5f7f8;margin:0}
main{max-width:1100px;margin:30px auto;padding:28px;background:white;border-radius:12px}
h1{margin:0}h2{margin-top:32px}h3{font-size:18px}p{color:#52616a}.notice{background:#fff3d8;padding:12px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}
th,td{padding:10px 12px;border-bottom:1px solid #e3e9ec;text-align:right;white-space:nowrap}
th{background:#edf3f5;position:sticky;top:0}th:first-child,td:first-child{text-align:left}
.table-wrap{overflow:auto;max-height:560px;border:1px solid #e3e9ec;border-radius:6px}
details{margin:14px 0}summary{cursor:pointer;font-weight:600;padding:8px 0}
select{font:inherit;padding:8px 16px;border:1px solid #a8b9c1;border-radius:6px}
button{font:inherit;padding:8px 14px;background:#edf3f5;border:1px solid #a8b9c1;border-radius:6px;cursor:pointer}
button:hover{background:#dce9ed}fieldset{border:1px solid #dce4e8;border-radius:6px;display:flex;flex-wrap:wrap;gap:12px 22px}
fieldset label{white-space:nowrap;cursor:pointer}.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:12px}
#snapshot-status{font-size:14px;margin:8px 0}.wide table{border-collapse:separate;border-spacing:0}
.wide th,.wide td{box-sizing:border-box}
.wide th{text-align:center;z-index:3}
.wide thead tr:first-child th{top:0;height:64px}
.wide thead tr:nth-child(2) th{top:64px;height:40px}
.wide td{background:#fff}.wide tbody tr:nth-child(even) td{background:#f2f6f8}
.wide tbody tr:hover td{background:#e5eff4}
.wide .time-nominal,.wide .time-actual{position:sticky;z-index:2;background:#edf3f5}
.wide .time-nominal{left:0;min-width:96px;width:96px}
.wide .time-actual{left:96px;min-width:120px;width:120px;border-right:2px solid #a8b9c1}
.wide thead .time-nominal,.wide thead .time-actual{z-index:5}
.wide [data-state-heading],.wide [data-kind="true"]{border-left:2px solid #c5d5dc}
.wide th[data-state-group="position"]{background:#dceee9}
.wide th[data-state-group="velocity"]{background:#e1eafa}
.wide th[data-state-group="attitude"]{background:#eee4f6}
.wide th[data-state-group="image"]{background:#fff0d9}
.wide th[data-state-group="gyro"],.wide th[data-state-group="accel"]{background:#e8e9ec}
.wide td[data-kind="error"]{font-weight:600;color:#8b421f}
svg{width:100%;height:auto}a{color:#176c90}[hidden]{display:none}
@media(max-width:700px){main{margin:0;padding:16px}}
</style><main>'''
    page += f'<h1>DKF 测试结果</h1><p>本次测试：{escape(directory.name)}</p>'
    page += f'<p>记录时长 <b>{duration:.2f} 秒</b> · 相对位置有效比较 <b>{comparison_count} 次</b></p>{empty}'
    page += '<p>平均相差、最大相差越小，表示估计越接近真值。位置和速度按“无人机 − 目标”定义。</p>'
    page += '<p><a href="#snapshots">直接查看每 0.5 秒对比表 ↓</a></p>'
    page += '<h2>1. 误差汇总</h2>' + ''.join(summary)
    page += '<h2 id="snapshots">2. 每隔 0.5 秒的数据对比</h2>'
    page += '<p>每行仍包含同一时刻的全部 18 个状态，勾选只改变显示。默认看位置和速度；'
    page += '需要快速比较时，可切换“只看误差”。时间列和表头固定，向右滚动可查看其余列。</p>'
    page += snapshot_controls() + snapshot_table(snapshots)
    page += '<h2>3. 对比曲线</h2><label>选择要看的状态： <select id="group">' + options + '</select></label>'
    if metadata.get('recording_start_time_ns') is not None:
        if metadata.get('recording_end_time_ns') is not None:
            page += '<p>记录范围：进入 ACTIVE 至首次退出 ACTIVE，包含两个边界。时间从拦截开始计为 0 秒，最后一行包含结束时刻。</p>'
        else:
            page += '<p class="notice">未收到拦截结束信号，记录被中断；不将最后一个采样点冒充正常结束时刻。</p>'
    elif 'recording_window' in metadata:
        page += '<p class="notice">本次未收到进入 ACTIVE 的信号，没有记录拦截数据。</p>'
    else:
        page += '<p>旧记录未提供拦截起止信号，时间从首个已对时样本开始，最后一行保留最后一个样本。</p>'
    page += '<p>绿色实线是真实值，橙色虚线是 DKF 估计值；两条线越重合越好。'
    page += '中间每 0.5 秒选取最近的原始记录；首尾使用区间内最近的实际样本，同时列出实际取样时间，不插值、不补造缺失数据。缺失区间显示“不可用”。</p>'
    page += ''.join(sections)
    page += '<h2>数据说明</h2><p>相对速度真值由仿真位姿差分得到，是近似值。六个 IMU 偏置没有可靠真值，只显示估计值，无法评价精度。'
    page += '误差 = 估计值 − 真实值。汇总和曲线使用完整记录，0.5 秒表格仅用于方便阅读。</p>'
    page += '<p>'
    page += '<a href="每0.5秒对比.xlsx">下载 Excel 对比表</a> · <a href="states.csv">原始数据</a></p>'
    page += '''</main><script>
document.getElementById('group').addEventListener('change', function(){
  document.querySelectorAll('.comparison').forEach(section => {
    section.hidden = section.dataset.group !== this.value;
  });
});

const stateChecks = [...document.querySelectorAll('input[name="state-group"]')];
const snapshotMode = document.getElementById('snapshot-mode');
function updateSnapshotColumns(){
  const selected = new Set(stateChecks.filter(input => input.checked).map(input => input.value));
  const errorsOnly = snapshotMode.value === 'error';
  document.querySelectorAll('#snapshot-table [data-state-group]').forEach(cell => {
    cell.hidden = !selected.has(cell.dataset.stateGroup) ||
      (errorsOnly && cell.dataset.kind && cell.dataset.kind !== 'error');
    if (cell.hasAttribute('data-state-heading')) cell.colSpan = errorsOnly ? 1 : 3;
  });
  const visibleStates = document.querySelectorAll('#snapshot-table [data-state-heading]:not([hidden])').length;
  document.getElementById('snapshot-status').textContent =
    `当前显示 ${visibleStates} / 18 个状态 · ${errorsOnly ? '只看误差' : '真实值＋估计值＋误差'}`;
}
stateChecks.forEach(input => input.addEventListener('change', updateSnapshotColumns));
snapshotMode.addEventListener('change', updateSnapshotColumns);
document.getElementById('show-all-states').addEventListener('click', () => {
  stateChecks.forEach(input => input.checked = true);
  updateSnapshotColumns();
});
document.getElementById('show-main-states').addEventListener('click', () => {
  stateChecks.forEach(input => input.checked = ['position', 'velocity'].includes(input.value));
  updateSnapshotColumns();
});
updateSnapshotColumns();
</script></html>'''
    path = directory / '测试结果.html'
    path.write_text(page, encoding='utf-8')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(generate_report(parser.parse_args().directory))
