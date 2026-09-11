"""
Client-/stakeholder-ready PDF export — one brand, one date range, with an
optional plain-language insights section. Pure-Python rendering (reportlab
+ matplotlib's non-interactive Agg backend), so it runs the same on a
headless machine as it does locally.
"""

from __future__ import annotations

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                 Table, TableStyle)

from .metrics import _CCY_SYMBOLS

NAVY = colors.HexColor("#10162b")
ACCENT = colors.HexColor("#2a78d6")
MUTED = colors.HexColor("#6b7280")
SEVERITY_COLOR = {"critical": colors.HexColor("#d03b3b"),
                   "watch": colors.HexColor("#b8790a"),
                   "scale": colors.HexColor("#0ca30c")}


def _money(v, ccy) -> str:
    if v is None or v != v:
        return "—"
    return f"{_CCY_SYMBOLS.get(ccy, ccy + ' ')}{v:,.2f}"


def _chart_image(daily: pd.DataFrame, metric_col: str, metric_label: str, currency: str) -> Image:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.3), dpi=150)
    for ax, col, label, color in (
        (axes[0], "spend", f"Spend ({currency})", "#2a78d6"),
        (axes[1], metric_col, metric_label, "#d95926"),
    ):
        d = daily.dropna(subset=[col]) if col in daily.columns else daily
        ax.plot(d["date"], d[col], color=color, linewidth=1.8, marker="o", markersize=3)
        ax.set_title(label, fontsize=9, color="#10162b", loc="left")
        ax.tick_params(labelsize=7, colors="#6b7280")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color="#e4e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=7.0 * inch, height=2.1 * inch)


def build_pdf_report(brand: dict, date_start: str, date_end: str,
                      camp_agg: pd.DataFrame, econ: dict, daily: pd.DataFrame,
                      insights: list, include_insights: bool = True) -> bytes:
    ccy = brand["currency"]
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=NAVY, fontSize=20)
    h2_style = ParagraphStyle("H2X", parent=styles["Heading2"], textColor=NAVY, spaceBefore=14)
    body_style = ParagraphStyle("BodyX", parent=styles["Normal"], textColor=colors.HexColor("#1f2937"))
    caption_style = ParagraphStyle("CaptionX", parent=styles["Normal"], textColor=MUTED, fontSize=9)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                             leftMargin=0.6 * inch, rightMargin=0.6 * inch)
    story = []

    story.append(Paragraph(f"{brand['name']} — Acquisition Report", title_style))
    story.append(Paragraph(
        f"{date_start} to {date_end} &nbsp;·&nbsp; generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"&nbsp;·&nbsp; {brand['business_model'].replace('_', ' ').title()}, {ccy}, "
        f"conversion = {brand['conversion_type']}",
        caption_style,
    ))
    story.append(Spacer(1, 12))

    # ---- KPI summary ----
    kpi_rows = [["Spend", "Conversions"]]
    kpi_vals = [[_money(econ.get("spend"), ccy), f"{econ.get('conversions', 0):,.0f}"]]
    if brand["business_model"] == "transactional":
        kpi_rows[0] += ["ROAS", "iROAS floor"]
        kpi_vals[0] += [f"{econ['roas']:.2f}x" if econ.get("roas") is not None else "—",
                         f"{econ['iroas_floor']:.2f}x" if econ.get("iroas_floor") else "—"]
    else:
        kpi_rows[0] += ["CAC / CPA", "Target CPA"]
        kpi_vals[0] += [_money(econ.get("cac"), ccy), _money(brand.get("target_cpa"), ccy)]
    kpi_rows[0] += ["LTV : CAC"]
    kpi_vals[0] += [f"{econ['ltv_cac']:.2f}x" if econ.get("ltv_cac") else "—"]

    kpi_table = Table([kpi_rows[0], kpi_vals[0]], hAlign="LEFT")
    kpi_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("FONTSIZE", (0, 1), (-1, 1), 14),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 1), (-1, 1), NAVY),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 0),
        ("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.HexColor("#e4e8f0")),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
    ]))
    story.append(kpi_table)

    # ---- trend chart ----
    if daily is not None and len(daily) >= 2:
        metric_col = "roas" if brand["business_model"] == "transactional" else "cpa"
        metric_label = metric_col.upper()
        story.append(Spacer(1, 6))
        story.append(_chart_image(daily, metric_col, metric_label, ccy))

    # ---- campaign table ----
    story.append(Paragraph("Campaign performance", h2_style))
    cols = ["campaign", "spend", "conversions", "cpa" if brand["business_model"] == "lead_gen" else "roas"]
    header = ["Campaign", "Spend", "Conversions", cols[3].upper()]
    data = [header]
    for _, r in camp_agg.sort_values("spend", ascending=False).iterrows():
        metric_val = r.get(cols[3])
        metric_str = (_money(metric_val, ccy) if cols[3] == "cpa"
                      else (f"{metric_val:.2f}x" if metric_val is not None and metric_val == metric_val else "—"))
        data.append([str(r["campaign"])[:42], _money(r["spend"], ccy), f"{r['conversions']:,.0f}", metric_str])

    camp_table = Table(data, hAlign="LEFT", colWidths=[3.0 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch])
    camp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f7")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#e4e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]))
    story.append(camp_table)

    # ---- insights ----
    if include_insights and insights:
        story.append(Paragraph("Growth insights", h2_style))
        for ins in insights:
            badge_color = SEVERITY_COLOR.get(ins.severity, MUTED)
            story.append(Paragraph(
                f'<font color="{badge_color.hexval()}">●</font> <b>{ins.title}</b>', body_style,
            ))
            story.append(Paragraph(ins.detail, body_style))
            story.append(Paragraph(
                f"metric: {ins.metric} · threshold: {ins.threshold} · formula: {ins.formula}", caption_style,
            ))
            story.append(Spacer(1, 8))
    elif include_insights:
        story.append(Paragraph("Growth insights", h2_style))
        story.append(Paragraph("No threshold breaches this period.", body_style))

    doc.build(story)
    return buf.getvalue()
