"""
Renders a brand's report as a single static HTML page — the payload for a
shareable link. This module only builds the HTML string; publishing it (as
a private Claude Artifact, with a real URL) happens outside the Streamlit
app, on request, since that's a capability of the assistant session, not
something the local app can call itself. See sync notes / project README
for the "ask Claude to publish a report" workflow.

No live data connections, no capabilities — this is a read-only snapshot
of the data at generation time, styled to the same design system as the
dashboard so a client opening the link doesn't get a jarring visual
handoff from the PDF/dashboard.
"""

from __future__ import annotations

import html as _html
import math
from datetime import datetime

import pandas as pd

from .metrics import _CCY_SYMBOLS

SEVERITY_LABEL = {"critical": "Critical", "watch": "Watch", "scale": "Scale candidate"}


def _esc(s) -> str:
    return _html.escape(str(s)) if s is not None else ""


def _money(v, ccy) -> str:
    if v is None or v != v:
        return "—"
    return f"{_CCY_SYMBOLS.get(ccy, ccy + ' ')}{v:,.2f}"


def _nice_ticks(max_v: float, count: int = 4) -> list[float]:
    if max_v <= 0:
        return [0, 1]
    rough = max_v / count
    mag = 10 ** int(math.floor(math.log10(rough))) if rough > 0 else 1
    for step in (1, 2, 5, 10):
        if rough / mag <= step:
            step_v = step * mag
            break
    else:
        step_v = 10 * mag
    top = step_v * (int(max_v / step_v) + 1)
    ticks = []
    v = 0.0
    while v <= top + 1e-9:
        ticks.append(round(v, 2))
        v += step_v
    return ticks


def _line_chart_svg(daily: pd.DataFrame, col: str, label: str, color: str) -> str:
    d = daily.dropna(subset=[col]) if col in daily.columns else daily
    if d.empty or len(d) < 2:
        return f'<div class="chart-empty">Not enough data for {_esc(label)}</div>'
    W, H, ML, MR, MT, MB = 320, 170, 44, 12, 10, 24
    plotW, plotH = W - ML - MR, H - MT - MB
    vals = d[col].tolist()
    dates = d["date"].tolist()
    max_v = max(vals) if vals else 1
    ticks = _nice_ticks(max_v * 1.15, 3)
    y_max = ticks[-1] or 1

    def x(i):
        return ML + (i * plotW) / (len(vals) - 1)

    def y(v):
        return MT + plotH - (v / y_max) * plotH

    grid = "".join(
        f'<line x1="{ML}" x2="{W-MR}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="var(--gridline)" stroke-width="1"/>'
        f'<text x="{ML-6}" y="{y(t)+3:.1f}" text-anchor="end" font-size="9" fill="var(--text-muted)">{t:,.0f}</text>'
        for t in ticks
    )
    path = " ".join(f'{"M" if i==0 else "L"}{x(i):.1f},{y(v):.1f}' for i, v in enumerate(vals))
    last_x, last_y = x(len(vals) - 1), y(vals[-1])
    def _fmt_date(d_):
        # "%-d" (no leading zero) isn't portable to Windows' strftime — format
        # with a zero-padded day and strip it manually instead.
        s = pd.Timestamp(d_).strftime("%b %d")
        month, day = s.split(" ")
        return f"{month} {day.lstrip('0') or '0'}"

    labels_x = "".join(
        f'<text x="{x(i):.1f}" y="{H-6}" text-anchor="middle" font-size="8.5" fill="var(--text-muted)">'
        f'{_fmt_date(d_)}</text>'
        for i, d_ in enumerate(dates) if i == 0 or i == len(dates) - 1 or i % max(1, len(dates)//4) == 0
    )
    return (
        f'<svg viewBox="0 0 {W} {H}" class="chart-svg" role="img" aria-label="{_esc(label)} over time">'
        f'{grid}'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3.5" fill="{color}" stroke="var(--surface)" stroke-width="2"/>'
        f'{labels_x}'
        f'</svg>'
    )


def build_report_html(brand: dict, date_start: str, date_end: str,
                       camp_agg: pd.DataFrame, econ: dict, daily: pd.DataFrame,
                       insights: list, forecast: dict | None = None) -> str:
    ccy = brand["currency"]
    is_transactional = brand["business_model"] == "transactional"
    metric_col = "roas" if is_transactional else "cpa"
    metric_label = "ROAS" if is_transactional else "CPA"

    kpi_cells = [
        ("Spend", _money(econ.get("spend"), ccy)),
        ("Conversions", f"{econ.get('conversions', 0):,.0f}"),
    ]
    if is_transactional:
        kpi_cells += [("ROAS", f"{econ['roas']:.2f}x" if econ.get("roas") is not None else "—"),
                      ("iROAS floor", f"{econ['iroas_floor']:.2f}x" if econ.get("iroas_floor") else "—")]
    else:
        kpi_cells += [("CAC / CPA", _money(econ.get("cac"), ccy)),
                      ("Target CPA", _money(brand.get("target_cpa"), ccy))]
    kpi_cells.append(("LTV : CAC", f"{econ['ltv_cac']:.2f}x" if econ.get("ltv_cac") else "—"))

    kpi_html = "".join(
        f'<div class="tile"><div class="tile-label">{_esc(l)}</div>'
        f'<div class="tile-value">{_esc(v)}</div></div>' for l, v in kpi_cells
    )

    rows_html = ""
    for _, r in camp_agg.sort_values("spend", ascending=False).iterrows():
        mv = r.get(metric_col)
        mstr = (_money(mv, ccy) if metric_col == "cpa" else (f"{mv:.2f}x" if mv is not None and mv == mv else "—"))
        rows_html += (
            f'<tr><td>{_esc(r["campaign"])}</td><td class="num">{_money(r["spend"], ccy)}</td>'
            f'<td class="num">{r["conversions"]:,.0f}</td><td class="num">{mstr}</td></tr>'
        )

    insights_html = ""
    if insights:
        for ins in insights:
            action_html = (f'<p><strong>&rarr; Suggested action:</strong> {_esc(ins.suggested_action)}</p>'
                            if ins.suggested_action else "")
            insights_html += (
                f'<div class="insight insight-{ins.severity}">'
                f'<div class="insight-head"><span class="pill pill-{ins.severity}">{SEVERITY_LABEL.get(ins.severity, ins.severity)}</span>'
                f'<strong>{_esc(ins.title)}</strong></div>'
                f'<p>{_esc(ins.detail)}</p>'
                f'{action_html}'
                f'<div class="insight-meta">metric: {_esc(ins.metric)} · threshold: {_esc(ins.threshold)} · formula: {_esc(ins.formula)}</div>'
                f'</div>'
            )
    else:
        insights_html = '<p class="muted">No threshold breaches this period.</p>'

    forecast_html = ""
    if forecast is not None:
        fmt_forecast = (lambda v: _money(v, ccy)) if metric_col == "cpa" else (lambda v: f"{v:.2f}x")
        fit_note = "a fairly consistent trend" if forecast["r_squared"] >= 0.5 else "noisy — treat loosely"
        forecast_html = (
            f'<p>Projecting the last {forecast["lookback_points"]} days\' momentum forward '
            f'(straight-line, not seasonality-aware): {_esc(metric_label)} is projected at '
            f'{fmt_forecast(forecast["projected_value_end"])} in 7 days '
            f'({forecast["pct_change_projected"]:+.1f}% vs. today), fit R²={forecast["r_squared"]} ({fit_note}).</p>'
        )
        target_val = brand.get("target_cpa") if metric_col == "cpa" else brand.get("target_roas")
        if target_val:
            breaches = (forecast["projected_value_end"] > target_val) if metric_col == "cpa" \
                else (forecast["projected_value_end"] < target_val)
            if breaches:
                forecast_html += (
                    f'<p><span class="pill pill-critical">Critical</span> At this trend, projected '
                    f'{_esc(metric_label)} in 7 days would be past the {fmt_forecast(target_val)} target — '
                    f'worth acting before it gets there, not after.</p>'
                )

    spend_chart = _line_chart_svg(daily, "spend", f"Spend ({ccy})", "#2a78d6")
    metric_chart = _line_chart_svg(daily, metric_col, metric_label, "#d95926")

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<title>{_esc(brand['name'])} Report</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{{
  color-scheme: light;
  --bg:#f5f7fb; --surface:#ffffff; --surface-2:#eef1f7;
  --text-primary:#10162b; --text-secondary:#4c5570; --text-muted:#8991a8;
  --border:rgba(16,22,43,0.10); --accent:#2a78d6; --gridline:#e4e8f0;
  --good:#0ca30c; --warning:#b8790a; --critical:#d03b3b;
  --good-wash:rgba(12,163,12,0.12); --warning-wash:rgba(184,121,10,0.14); --critical-wash:rgba(208,59,59,0.12);
}}
@media (prefers-color-scheme: dark){{
  :root:not([data-theme="light"]){{
    color-scheme: dark;
    --bg:#0a0f1c; --surface:#121a2c; --surface-2:#182338;
    --text-primary:#f4f6fb; --text-secondary:#b9c1d6; --text-muted:#838da6;
    --border:rgba(255,255,255,0.09); --accent:#3987e5; --gridline:#223049;
    --good:#0ca30c; --warning:#d9a01a; --critical:#e66767;
    --good-wash:rgba(12,163,12,0.18); --warning-wash:rgba(217,160,26,0.16); --critical-wash:rgba(230,103,103,0.16);
  }}
}}
:root[data-theme="dark"]{{
  color-scheme: dark;
  --bg:#0a0f1c; --surface:#121a2c; --surface-2:#182338;
  --text-primary:#f4f6fb; --text-secondary:#b9c1d6; --text-muted:#838da6;
  --border:rgba(255,255,255,0.09); --accent:#3987e5; --gridline:#223049;
  --good:#0ca30c; --warning:#d9a01a; --critical:#e66767;
  --good-wash:rgba(12,163,12,0.18); --warning-wash:rgba(217,160,26,0.16); --critical-wash:rgba(230,103,103,0.16);
}}
*{{box-sizing:border-box;}}
body{{background:var(--bg);color:var(--text-primary);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:14px;}}
.page{{max-width:880px;margin:0 auto;padding:32px 24px 64px;display:flex;flex-direction:column;gap:22px;}}
h1{{font-family:"Manrope",sans-serif;font-weight:800;font-size:24px;margin:0;text-wrap:balance;}}
.eyebrow{{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);}}
.meta{{color:var(--text-secondary);font-size:13px;margin:0;}}
.kpi-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;}}
.tile{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px;}}
.tile-label{{font-size:11px;color:var(--text-muted);font-weight:600;}}
.tile-value{{font-family:"Manrope",sans-serif;font-weight:700;font-size:18px;margin-top:4px;}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px 20px;}}
.card h2{{font-family:"Manrope",sans-serif;font-size:15px;margin:0 0 12px;}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
.chart-svg{{width:100%;height:auto;}}
.chart-empty{{color:var(--text-muted);font-size:12px;padding:40px 0;text-align:center;}}
table{{width:100%;border-collapse:collapse;}}
th,td{{text-align:left;padding:8px 10px;font-size:13px;border-bottom:1px solid var(--border);}}
th{{font-size:10.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:var(--text-muted);}}
.num{{text-align:right;font-variant-numeric:tabular-nums;}}
.insight{{border-left:3px solid var(--border);padding:8px 0 8px 12px;margin-bottom:10px;}}
.insight-critical{{border-color:var(--critical);}}
.insight-watch{{border-color:var(--warning);}}
.insight-scale{{border-color:var(--good);}}
.insight-head{{display:flex;align-items:center;gap:8px;margin-bottom:4px;}}
.insight p{{margin:0 0 4px;color:var(--text-secondary);}}
.insight-meta{{font-size:11px;color:var(--text-muted);}}
.pill{{font-size:10.5px;font-weight:700;padding:2px 8px;border-radius:999px;}}
.pill-critical{{background:var(--critical-wash);color:var(--critical);}}
.pill-watch{{background:var(--warning-wash);color:var(--warning);}}
.pill-scale{{background:var(--good-wash);color:var(--good);}}
.muted{{color:var(--text-muted);font-size:13px;}}
.footer{{text-align:center;color:var(--text-muted);font-size:11px;}}
@media (max-width:640px){{ .kpi-grid{{grid-template-columns:repeat(2,1fr);}} .chart-row{{grid-template-columns:1fr;}} }}
</style>
<div class="page">
  <div>
    <div class="eyebrow">Acquisition report</div>
    <h1>{_esc(brand['name'])}</h1>
    <p class="meta">{_esc(date_start)} to {_esc(date_end)} · generated {generated} · {_esc(brand['business_model'].replace('_',' ').title())}, {_esc(ccy)}, conversion = {_esc(brand['conversion_type'])}</p>
  </div>
  <div class="kpi-grid">{kpi_html}</div>
  <div class="card">
    <h2>Trend</h2>
    <div class="chart-row">{spend_chart}{metric_chart}</div>
  </div>
  <div class="card">
    <h2>Campaign performance</h2>
    <table><thead><tr><th>Campaign</th><th class="num">Spend</th><th class="num">Conversions</th><th class="num">{_esc(metric_label)}</th></tr></thead>
    <tbody>{rows_html}</tbody></table>
  </div>
  {f'<div class="card"><h2>Trend forecast</h2>{forecast_html}</div>' if forecast_html else ''}
  <div class="card">
    <h2>Growth insights</h2>
    {insights_html}
  </div>
  <p class="footer">Generated by the PPC Acquisition Intelligence prototype — a static snapshot, not a live view.</p>
</div>
"""
