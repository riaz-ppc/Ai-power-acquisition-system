"""
Digest computation + small formatting helpers, factored out of app.py so
they're importable without pulling in Streamlit's runtime — used by both
the Digest tab / Portfolio view (inside the app) and the standalone
scheduled-digest script (scripts/send_digest.py, run from cron/CI, not
through Streamlit at all).
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from . import metrics


def money(v, ccy="USD"):
    if v is None or v != v:
        return "—"
    symbol = {"USD": "$", "GBP": "£", "EUR": "€"}.get(ccy, ccy + " ")
    return f"{symbol}{v:,.2f}"


def pct(v):
    return "—" if v is None or v != v else f"{v:+.1f}%"


def ratio(v, suffix="x"):
    return "—" if v is None or v != v else f"{v:.2f}{suffix}"


def brand_dict(row) -> dict:
    return dict(row) if row is not None else {}


_DRIVER_LABELS = {
    "cpm": "CPM (cost per 1,000 impressions)", "ctr": "CTR (click-through rate)",
    "cvr": "CVR (conversion rate)", "aov": "AOV (average order value)",
}


def _format_driver_value(key, v, ccy):
    if key in ("cpm", "aov"):
        return money(v, ccy)
    return f"{v * 100:.2f}%"


def _format_decomposition(decomp: dict, business_model: str, ccy: str) -> str:
    """Turns metrics.decompose_metric_change()'s output into one plain
    sentence naming the primary driver and, briefly, the others."""
    metric_name = "ROAS" if business_model == "transactional" else "CPA"
    verb = "worsened" if decomp["primary_driver_worsened_metric"] else "improved"
    primary = decomp["primary_driver"]
    p = decomp["drivers"][primary]
    text = (f"**Why:** mainly **{_DRIVER_LABELS[primary]}** — "
            f"{_format_driver_value(primary, p['baseline'], ccy)} → {_format_driver_value(primary, p['current'], ccy)} "
            f"({p['pct_change']:+.0f}%), which {verb} {metric_name} and accounts for "
            f"~{decomp['primary_driver_share_pct']:.0f}% of this move.")
    others = [k for k in decomp["drivers"] if k != primary]
    if others:
        bits = [f"{_DRIVER_LABELS[k].split(' (')[0]} {_format_driver_value(k, decomp['drivers'][k]['baseline'], ccy)} "
                f"→ {_format_driver_value(k, decomp['drivers'][k]['current'], ccy)}" for k in others]
        text += " Other drivers barely moved: " + "; ".join(bits) + "."
    return text


def compute_digest_items(df: pd.DataFrame, brand: dict, n: int = 7) -> list[dict]:
    """Every item comes straight from an already-tested metrics.* function —
    see the Digest tab and the portfolio view, both of which render this same
    list. Returns items sorted by priority (0=critical/1=watch/2=opportunity)
    then by dollar impact within a tier; [] means nothing worth flagging."""
    max_d = df["date"].max().date()
    cur_start, cur_end = max_d - timedelta(days=n - 1), max_d
    base_start, base_end = cur_start - timedelta(days=n), cur_start - timedelta(days=1)

    compare = metrics.compare_periods(df, cur_start, cur_end, base_start, base_end)
    camp_scoped = df[(df["date"] >= pd.Timestamp(cur_start)) & (df["date"] <= pd.Timestamp(cur_end))]
    camp_agg = metrics.aggregate(camp_scoped, by=["campaign"])
    full_daily = metrics.aggregate(df, by=["date"])

    items = []

    for ins in metrics.generate_insights(camp_agg, brand, compare):
        priority = {"critical": 0, "watch": 1, "scale": 2}.get(ins.severity, 1)
        detail = ins.detail + (f" → {ins.suggested_action}" if ins.suggested_action else "")
        items.append({"priority": priority, "impact": 0.0,
                       "badge": {"critical": "🔴", "watch": "🟡", "scale": "🟢"}.get(ins.severity, ""),
                       "title": ins.title, "detail": detail})

    forecast_metric = "roas" if brand["business_model"] == "transactional" else "cpa"
    forecast = metrics.forecast_trend(full_daily, forecast_metric)
    if forecast:
        target_val = brand["target_cpa"] if forecast_metric == "cpa" else brand["target_roas"]
        if target_val:
            fmt_val = (lambda v: money(v, brand["currency"])) if forecast_metric == "cpa" else ratio
            breaches = (forecast["projected_value_end"] > target_val) if forecast_metric == "cpa" \
                else (forecast["projected_value_end"] < target_val)
            if breaches:
                items.append({
                    "priority": 0, "impact": 0.0, "badge": "🔴",
                    "title": f"Blended {forecast_metric.upper()} trending toward a target breach",
                    "detail": f"At this trend, projected {forecast_metric.upper()} in 7 days "
                              f"({fmt_val(forecast['projected_value_end'])}) would be past your target "
                              f"({fmt_val(target_val)}) — worth acting before it gets there.",
                })

    realloc = metrics.budget_reallocation_view(df, camp_agg, brand)
    if realloc:
        best, worst = realloc[0], realloc[-1]
        if (best["ranking_metric"] is not None and worst["ranking_metric"] is not None
                and best["campaign"] != worst["campaign"]):
            gap_pct = abs(worst["ranking_metric"] - best["ranking_metric"]) / abs(best["ranking_metric"]) * 100 \
                if best["ranking_metric"] else 0
            if gap_pct >= 25:
                test_amount = worst["spend"] * 0.15
                items.append({
                    "priority": 2, "impact": test_amount, "badge": "🟢",
                    "title": f"Reallocate budget: {best['campaign']} over {worst['campaign']}",
                    "detail": f"A {gap_pct:.0f}% efficiency gap between campaigns — worth testing a shift "
                              f"of roughly {money(test_amount, brand['currency'])} from {worst['campaign']} "
                              f"to {best['campaign']}.",
                })

    cross_platform = metrics.cross_platform_reallocation_view(df, brand)
    if cross_platform:
        cp_best, cp_worst = cross_platform[0], cross_platform[-1]
        if (cp_best["ranking_metric"] is not None and cp_worst["ranking_metric"] is not None
                and cp_best["platform"] != cp_worst["platform"]):
            cp_gap_pct = abs(cp_worst["ranking_metric"] - cp_best["ranking_metric"]) / abs(cp_best["ranking_metric"]) * 100 \
                if cp_best["ranking_metric"] else 0
            if cp_gap_pct >= 25:
                cp_test_amount = cp_worst["spend"] * 0.15
                items.append({
                    "priority": 2, "impact": cp_test_amount, "badge": "🟢",
                    "title": f"Shift budget toward {cp_best['platform']} over {cp_worst['platform']}",
                    "detail": f"A {cp_gap_pct:.0f}% platform-level efficiency gap — worth testing a shift "
                              f"of roughly {money(cp_test_amount, brand['currency'])} toward {cp_best['platform']}.",
                })

    # Per-course platform asymmetries — the one signal the platform-level
    # reallocation above can't see. Only "shift": below-break-even courses
    # are already covered campaign-by-campaign by generate_insights().
    for course in [c for c in metrics.course_platform_view(camp_scoped, brand) if c["verdict"] == "shift"][:3]:
        items.append({
            "priority": 2, "impact": course["test_amount"], "badge": "🟢",
            "title": f"{course['course'].title()}: shift budget between platforms",
            "detail": course["detail"],
        })

    waste = metrics.keyword_waste_candidates(camp_scoped)
    if waste:
        total_waste_spend = sum(w["spend"] for w in waste)
        items.append({
            "priority": 1, "impact": total_waste_spend, "badge": "🟡",
            "title": f"{len(waste)} keyword(s) burning spend with no conversions",
            "detail": f"{money(total_waste_spend, brand['currency'])} of spend this period with a "
                      f"95%-confidence best case still below this account's own typical conversion rate. "
                      f"Biggest: {waste[0]['keyword']} ({money(waste[0]['spend'], brand['currency'])}).",
        })

    anomaly_metric = "roas" if brand["business_model"] == "transactional" else "cpa"
    flags = metrics.detect_anomalies(full_daily, anomaly_metric, z_thresh=2.0) if len(full_daily) >= 5 else []
    if flags:
        worst_flag = max(flags, key=lambda f: abs(f["z_score"]))
        d = pd.Timestamp(worst_flag["date"]).date().isoformat()
        items.append({
            "priority": 1, "impact": 0.0, "badge": "🟡",
            "title": f"{len(flags)} statistically unusual day(s) in {anomaly_metric.upper()}",
            "detail": f"Most extreme: {d} ({worst_flag['pct_change']:+.0f}% day-over-day, "
                      f"z={worst_flag['z_score']}).",
        })

    items.sort(key=lambda x: (x["priority"], -x["impact"]))
    return items
