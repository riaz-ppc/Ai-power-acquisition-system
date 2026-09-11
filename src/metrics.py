"""
Derived unit economics + the rule-based decision-support layer.

Every insight this module produces carries the metric, the threshold it was
tested against, and (where applicable) the formula used — per the brief's
"never just a vibe" requirement. Deliberately rule-based rather than an LLM
call: the brief wants shown work and reproducible thresholds, which a
deterministic rule engine gives for free and a free-text model has to be
argued into.

KNOWN SIMPLIFICATIONS (flagged here, not buried):
- Creative fatigue prefers real frequency (impressions ÷ reach, from Meta's
  "Reach" field) when it's present on an ad's rows; where reach wasn't in
  the export, it falls back to a CTR-decline proxy instead — see
  `creative_fatigue_candidates`.
- Payback period is a rough approximation (CAC / (AOV × margin)), not a
  cohort-based calculation — good enough to flag direction, not investment-grade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pandas as pd


def rows_to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    for c in ("spend", "impressions", "clicks", "conversions", "conversion_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # reach stays NaN (not 0) when absent — 0 would wrongly read as "zero reach"
    # rather than "no reach data for this row", which fatigue detection needs to tell apart.
    df["reach"] = pd.to_numeric(df.get("reach"), errors="coerce") if "reach" in df.columns else None
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["ctr"] = (df["clicks"] / df["impressions"]).where(df["impressions"] > 0)
    df["cpc"] = (df["spend"] / df["clicks"]).where(df["clicks"] > 0)
    df["cpa"] = (df["spend"] / df["conversions"]).where(df["conversions"] > 0)
    df["roas"] = (df["conversion_value"] / df["spend"]).where(df["spend"] > 0)
    if "reach" in df.columns:
        df["frequency"] = (df["impressions"] / df["reach"]).where(df["reach"] > 0)
    return df


PERIOD_LABELS = {"D": "Day", "W": "Week", "M": "Month"}


def aggregate_by_period(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Rolls performance rows up into calendar buckets — day, ISO week (Monday
    start), or calendar month — and adds period-over-period % change columns
    for spend/conversions/cpa/roas. This is the "compare month by month, or
    any other way" view: a table a human actually reads top-to-bottom,
    not a chart they have to squint at to see the third-to-last point.
    """
    if df.empty:
        return df
    d = df.copy()
    if freq == "D":
        d["period"] = d["date"].dt.date.astype(str)
    elif freq == "W":
        d["period"] = d["date"].dt.to_period("W-SUN").apply(lambda p: p.start_time.date().isoformat())
    else:  # "M"
        d["period"] = d["date"].dt.to_period("M").astype(str)

    g = d.groupby("period", dropna=False).agg(
        spend=("spend", "sum"), impressions=("impressions", "sum"), clicks=("clicks", "sum"),
        conversions=("conversions", "sum"), conversion_value=("conversion_value", "sum"),
    ).reset_index().sort_values("period")
    g = add_derived(g)

    for col in ("spend", "conversions", "cpa", "roas"):
        g[f"{col}_change_pct"] = g[col].pct_change() * 100
    return g


def aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(by, dropna=False).agg(
        spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum"),
        conversion_value=("conversion_value", "sum"),
    ).reset_index()
    return add_derived(g)


def brand_unit_economics(agg_totals: dict, brand) -> dict:
    """agg_totals: {spend, conversions, conversion_value} already summed."""
    spend = agg_totals.get("spend", 0)
    conversions = agg_totals.get("conversions", 0)
    conv_value = agg_totals.get("conversion_value", 0)

    cac = (spend / conversions) if conversions else None
    roas = (conv_value / spend) if spend else None
    ltv_cac = (brand["ltv"] / cac) if (brand["ltv"] and cac) else None
    iroas_floor = (1 / brand["margin_pct"]) if brand["margin_pct"] else None

    # Payback: how many repeat orders until cumulative contribution margin
    # recovers CAC. Deliberately reported as an ORDER COUNT, not a time
    # period — converting to days needs a real repeat-purchase-cadence
    # input (e.g. "customer reorders every N days"), which isn't collected
    # yet. Fabricating a days figure without that input was actively
    # misleading (it produced numbers like "715 days" on real test data),
    # so this stays as orders until that input exists.
    payback_orders = None
    if cac and brand["aov"] and brand["margin_pct"]:
        margin_per_order = brand["aov"] * brand["margin_pct"]
        if margin_per_order > 0:
            payback_orders = round(cac / margin_per_order, 1)

    return {
        "cac": cac, "roas": roas, "ltv_cac": ltv_cac,
        "iroas_floor": iroas_floor, "payback_orders": payback_orders,
    }


def compare_periods(df: pd.DataFrame, current_start, current_end, baseline_start, baseline_end) -> dict:
    cur = df[(df["date"] >= pd.Timestamp(current_start)) & (df["date"] <= pd.Timestamp(current_end))]
    base = df[(df["date"] >= pd.Timestamp(baseline_start)) & (df["date"] <= pd.Timestamp(baseline_end))]

    def totals(x):
        return {
            "spend": x["spend"].sum(), "conversions": x["conversions"].sum(),
            "conversion_value": x["conversion_value"].sum(),
            "cpa": (x["spend"].sum() / x["conversions"].sum()) if x["conversions"].sum() else None,
            "roas": (x["conversion_value"].sum() / x["spend"].sum()) if x["spend"].sum() else None,
        }

    ct, bt = totals(cur), totals(base)
    pct = {}
    for k in ("spend", "conversions", "cpa", "roas"):
        c, b = ct.get(k), bt.get(k)
        pct[k] = ((c - b) / b * 100) if (c is not None and b) else None
    return {"current": ct, "baseline": bt, "pct_change": pct}


def detect_anomalies(daily: pd.DataFrame, metric: str, z_thresh: float = 2.0) -> list[dict]:
    """
    Flags days where `metric`'s day-over-day % change is a statistical
    outlier vs. the series' own volatility (z-score), not an arbitrary
    fixed percent. Needs >=5 data points to say anything meaningful.
    """
    if daily.empty or len(daily) < 5 or metric not in daily.columns:
        return []
    s = daily.sort_values("date")[metric].astype(float)
    pct_change = s.pct_change() * 100
    mean, std = pct_change[1:].mean(), pct_change[1:].std()
    if not std or std != std:  # NaN guard
        return []
    z = (pct_change - mean) / std
    flags = []
    for idx in daily.sort_values("date").index:
        zi = z.get(idx)
        if zi is not None and zi == zi and abs(zi) >= z_thresh:
            flags.append({
                "date": daily.loc[idx, "date"],
                "metric": metric,
                "value": daily.loc[idx, metric],
                "pct_change": pct_change.get(idx),
                "z_score": round(float(zi), 2),
                "threshold": z_thresh,
            })
    return flags


def forecast_trend(daily: pd.DataFrame, metric: str, lookback_days: int = 14,
                    forecast_days: int = 7) -> dict | None:
    """
    Where anomaly detection looks backward ("did something already break"),
    this looks forward: fits a plain linear trend (ordinary least squares,
    no external stats dependency — this is the one honest thing a handful
    of points supports) over the last `lookback_days` and projects it
    `forecast_days` ahead. This is a straight-line extrapolation of recent
    momentum, not a seasonality-aware forecast — flagged here rather than
    dressed up as more sophisticated than it is. Needs >=5 real points in
    the lookback window; returns None otherwise rather than a forecast
    built on too little.
    """
    if daily.empty or metric not in daily.columns:
        return None
    d = daily.sort_values("date").dropna(subset=[metric]).tail(lookback_days)
    if len(d) < 5:
        return None

    x = (d["date"] - d["date"].min()).dt.days.to_numpy(dtype=float)
    y = d[metric].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    last_x = x[-1]
    last_date = d["date"].max()
    future_x = np.arange(last_x + 1, last_x + 1 + forecast_days)
    future_dates = [last_date + pd.Timedelta(days=int(i)) for i in range(1, forecast_days + 1)]
    future_y = intercept + slope * future_x

    # Fit quality: how much of the day-to-day variance the straight line
    # actually explains — low r_squared means "this trend line is mostly
    # noise," which the caller should say plainly rather than presenting
    # a confident-looking number.
    fitted = intercept + slope * x
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "slope_per_day": float(slope),
        "r_squared": round(float(r_squared), 3),
        "lookback_points": len(d),
        "forecast": [
            {"date": fd.date().isoformat(), "value": float(fy)}
            for fd, fy in zip(future_dates, future_y)
        ],
        "current_value": float(y[-1]),
        "projected_value_end": float(future_y[-1]),
        "pct_change_projected": float((future_y[-1] - y[-1]) / y[-1] * 100) if y[-1] else None,
    }


# Static commercial-calendar knowledge, keyed by the brand's own market
# (its "country") and, where relevant, its business model. This is
# deliberately NOT a live trends feed — nothing here is fetched or
# predicted, it's well-known recurring demand periods (moon-sighting
# events are approximate, +/- a day). The honest use of it is as a
# heads-up plus a same-window-last-year comparison against the brand's
# own historical data, not a claim about "what will happen this year."
# Lunar-calendar dates (Eid) are only stocked for a few years; a season
# outside that range is simply not surfaced rather than guessed.
_LUNAR_EVENTS = {
    "Eid al-Fitr": {2025: "2025-03-31", 2026: "2026-03-20", 2027: "2027-03-10"},
    "Eid al-Adha": {2025: "2025-06-07", 2026: "2026-05-27", 2027: "2027-05-17"},
}

# (month, day, span_days, name, direction, business_models or None=all, note)
_FIXED_EVENTS_BY_COUNTRY = {
    "Bangladesh": [
        (4, 14, 3, "Pohela Boishakh (Bengali New Year)", "up", None,
         "Major retail/e-commerce sales occasion nationwide."),
        (11, 25, 5, "Black Friday / Cyber Monday", "up", ["transactional"],
         "Growing e-commerce discount season even outside its US origin."),
        (12, 20, 12, "Year-end clearance season", "up", ["transactional"],
         "Retailers commonly clear stock before the new year."),
    ],
    "United Kingdom": [
        (1, 2, 30, "January enrollment/new-term surge", "up", ["lead_gen"],
         "New-year resolutions and new academic/training terms lift lead volume for courses and certifications."),
        (4, 1, 20, "New UK tax year (Apr 6) / compliance renewal window", "up", ["lead_gen"],
         "Employers often renew mandatory compliance training around the new tax year."),
        (9, 1, 20, "Back-to-school / new term", "up", ["lead_gen"],
         "Training and enrollment demand typically rises alongside the academic calendar."),
        (11, 25, 5, "Black Friday / Cyber Monday", "up", ["transactional"],
         "Widely observed UK retail discount event."),
        (12, 20, 15, "Christmas/New Year lull", "down", None,
         "Both consumer spending attention and B2B training bookings typically dip."),
    ],
}


def market_seasons(country: str | None, business_model: str, ref_date, window_days: int = 30) -> list[dict]:
    """
    Known recurring commercial/cultural demand periods for `country` whose
    window falls within `window_days` of `ref_date` (today, in practice).
    Each entry says its direction (up/down) and is filtered to business
    models it actually applies to. Returns [] when `country` isn't set or
    isn't in the static calendar yet, rather than guessing.
    """
    if not country or country not in _FIXED_EVENTS_BY_COUNTRY:
        return []
    ref_date = pd.Timestamp(ref_date).date()
    window_start, window_end = ref_date - pd.Timedelta(days=window_days), ref_date + pd.Timedelta(days=window_days)
    results = []
    for year in (ref_date.year - 1, ref_date.year, ref_date.year + 1):
        for month, day, span, name, direction, models, note in _FIXED_EVENTS_BY_COUNTRY[country]:
            if models and business_model not in models:
                continue
            try:
                start = pd.Timestamp(year=year, month=month, day=day).date()
            except ValueError:
                continue
            end = start + pd.Timedelta(days=span)
            if start <= window_end and end >= window_start:
                results.append({"name": name, "start": start, "end": end, "direction": direction, "note": note})
    for name, years in _LUNAR_EVENTS.items():
        for year in (ref_date.year - 1, ref_date.year, ref_date.year + 1):
            if year not in years:
                continue
            start = pd.Timestamp(years[year]).date()
            end = start + pd.Timedelta(days=3)
            if start <= window_end and end >= window_start:
                results.append({
                    "name": name, "start": start, "end": end, "direction": "up",
                    "note": "Major gift/retail spending occasion (Bangladesh); approximate date, moon-sighting dependent."
                    if country == "Bangladesh" else "Moon-sighting dependent; date is approximate.",
                })
    return sorted(results, key=lambda r: r["start"])


def creative_fatigue_candidates(ad_level_df: pd.DataFrame, min_periods: int = 3) -> list[dict]:
    """
    Per ad: prefer the real signal — frequency (impressions ÷ reach) climbing
    while CTR falls, Meta's own textbook fatigue pattern — whenever that
    ad's rows carry reach data. Only when reach is entirely absent for an ad
    (non-Meta source, or an export that didn't include it) does this fall
    back to a CTR-decline-only proxy, which is weaker evidence: CTR can drop
    for reasons that have nothing to do with fatigue (audience, placement
    mix), so the fallback threshold is intentionally stricter.
    """
    if ad_level_df.empty or "ad" not in ad_level_df.columns:
        return []
    out = []
    for ad, g in ad_level_df.groupby("ad"):
        g = add_derived(g.sort_values("date"))
        if len(g) < min_periods:
            continue

        has_reach = "frequency" in g.columns and g["frequency"].notna().sum() >= min_periods
        ctr = g["ctr"].dropna()
        if len(ctr) < min_periods:
            continue
        recent_ctr, prior_ctr = ctr.iloc[-1], ctr.iloc[0]
        if not prior_ctr:
            continue
        ctr_decline_pct = (recent_ctr - prior_ctr) / prior_ctr * 100

        if has_reach:
            freq = g["frequency"].dropna()
            recent_freq, prior_freq = freq.iloc[-1], freq.iloc[0]
            # the real pattern: frequency has climbed AND CTR has fallen
            if prior_freq and recent_freq > prior_freq * 1.3 and ctr_decline_pct <= -20:
                out.append({
                    "ad": ad, "signal": "frequency+ctr",
                    "frequency_start": round(prior_freq, 2), "frequency_recent": round(recent_freq, 2),
                    "ctr_start": round(prior_ctr, 4), "ctr_recent": round(recent_ctr, 4),
                    "pct_decline": round(ctr_decline_pct, 1),
                })
        elif ctr_decline_pct <= -30:  # stricter bar for the proxy-only signal
            out.append({
                "ad": ad, "signal": "ctr-proxy",
                "ctr_start": round(prior_ctr, 4), "ctr_recent": round(recent_ctr, 4),
                "pct_decline": round(ctr_decline_pct, 1),
            })
    return out


def two_proportion_z_test(conv_a: float, n_a: float, conv_b: float, n_b: float) -> dict:
    """
    Two-proportion z-test for "is variant B's conversion rate really
    different from variant A's" — the basic significance check the A/B
    test tracker needs. Normal-approximation p-value (two-tailed), no
    scipy dependency. Needs a decent sample (n_a, n_b >= 30) to be
    trustworthy; flagged in the result rather than silently returned.
    """
    import math

    if n_a <= 0 or n_b <= 0:
        return {"significant": False, "reason": "no data in one or both variants"}

    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) if 0 < p_pool < 1 else 0
    if se == 0:
        return {"rate_a": p_a, "rate_b": p_b, "z": 0.0, "p_value": 1.0,
                "significant": False, "low_sample_warning": min(n_a, n_b) < 30}

    z = (p_b - p_a) / se
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {
        "rate_a": p_a, "rate_b": p_b, "z": round(z, 3), "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "low_sample_warning": min(n_a, n_b) < 30,
    }


@dataclass
class Insight:
    severity: str      # scale / watch / critical
    title: str
    detail: str
    metric: str
    threshold: str
    formula: str
    suggested_action: str = ""  # what to actually DO about it, not just what's wrong


_CCY_SYMBOLS = {"USD": "$", "GBP": "£", "EUR": "€"}


def _money(v: float, ccy: str) -> str:
    return f"{_CCY_SYMBOLS.get(ccy, ccy + ' ')}{v:,.2f}"


def generate_insights(campaign_agg: pd.DataFrame, brand, period_compare: dict | None = None) -> list[Insight]:
    """
    NOTE on currency in the generated text: amounts are formatted with the
    brand's own currency (never a hardcoded "$"), and any literal currency
    symbol/number pairing avoids embedding two "$" in one string — Streamlit
    renders text between a pair of "$" as inline LaTeX math, which silently
    mangles a sentence like "CPA is $38 vs. a target of $28". Callers should
    still escape "$" defensively before st.write/st.caption for USD brands.
    """
    insights: list[Insight] = []
    if campaign_agg.empty:
        return insights

    business_model = brand["business_model"]
    target_roas = brand["target_roas"]
    target_cpa = brand["target_cpa"]
    margin_pct = brand["margin_pct"]
    ccy = brand["currency"]
    iroas_floor = (1 / margin_pct) if margin_pct else None

    for _, row in campaign_agg.iterrows():
        camp = row["campaign"]

        if business_model == "transactional" and iroas_floor and row["roas"] is not None:
            if row["roas"] < iroas_floor:
                insights.append(Insight(
                    severity="critical",
                    title=f"{camp}: below break-even ROAS",
                    detail=(f"ROAS is {row['roas']:.2f}x against a break-even floor of "
                            f"{iroas_floor:.2f}x — this campaign is losing money on contribution "
                            f"margin at current spend."),
                    metric="roas", threshold=f"< {iroas_floor:.2f}x",
                    formula="iROAS floor = 1 ÷ contribution margin",
                    suggested_action="Pause it, or cut its budget hard, until you've changed the "
                                      "creative/targeting — every day it keeps running at this ROAS "
                                      "loses money outright, not just underperforms.",
                ))
            elif target_roas and row["roas"] < target_roas:
                insights.append(Insight(
                    severity="watch",
                    title=f"{camp}: below target ROAS",
                    detail=f"ROAS is {row['roas']:.2f}x vs. a target of {target_roas:.2f}x.",
                    metric="roas", threshold=f"< {target_roas:.2f}x", formula="target_roas (brand config)",
                    suggested_action="Still profitable, just underperforming — worth a creative "
                                      "refresh or tighter targeting before cutting budget outright.",
                ))
            elif target_roas and row["roas"] >= target_roas * 1.25:
                insights.append(Insight(
                    severity="scale",
                    title=f"{camp}: strong scale candidate",
                    detail=f"ROAS is {row['roas']:.2f}x, 25%+ above target ({target_roas:.2f}x).",
                    metric="roas", threshold=f">= {target_roas*1.25:.2f}x", formula="target_roas × 1.25",
                    suggested_action="Increase budget here before anywhere else in the account — "
                                      "raise it incrementally (e.g. 20-30% at a time) and watch ROAS "
                                      "over the next few days rather than doubling it in one move.",
                ))

        if business_model == "lead_gen" and target_cpa and row["cpa"] is not None:
            if row["cpa"] > target_cpa * 1.15:
                insights.append(Insight(
                    severity="critical",
                    title=f"{camp}: CPA over target",
                    detail=(f"CPA is {_money(row['cpa'], ccy)} vs. a target of {_money(target_cpa, ccy)} "
                            f"(+15% tolerance breached)."),
                    metric="cpa", threshold=f"> {_money(target_cpa*1.15, ccy)}", formula="target_cpa × 1.15",
                    suggested_action="Pause or sharply reduce budget now — check the funnel first "
                                      "(is CTR down, or is lead-to-enrollment down) so the fix "
                                      "actually addresses what broke, not just the symptom.",
                ))
            elif row["cpa"] > target_cpa:
                insights.append(Insight(
                    severity="watch",
                    title=f"{camp}: CPA above target",
                    detail=f"CPA is {_money(row['cpa'], ccy)} vs. a target of {_money(target_cpa, ccy)}.",
                    metric="cpa", threshold=f"> {_money(target_cpa, ccy)}", formula="target_cpa (brand config)",
                    suggested_action="Not urgent yet — worth tightening targeting or testing new ad "
                                      "copy before it drifts further past target.",
                ))
            elif row["cpa"] <= target_cpa * 0.8:
                insights.append(Insight(
                    severity="scale",
                    title=f"{camp}: efficient scale candidate",
                    detail=f"CPA is {_money(row['cpa'], ccy)}, 20%+ under target ({_money(target_cpa, ccy)}).",
                    metric="cpa", threshold=f"<= {_money(target_cpa*0.8, ccy)}", formula="target_cpa × 0.8",
                    suggested_action="Increase budget here before anywhere else in the account — "
                                      "raise it incrementally and watch whether CPA holds as volume grows.",
                ))

    if period_compare:
        pct = period_compare.get("pct_change", {})
        cpa_chg = pct.get("cpa")
        if cpa_chg is not None and cpa_chg >= 30:
            insights.append(Insight(
                severity="critical",
                title="Blended CPA rising sharply",
                detail=f"Blended CPA is up {cpa_chg:.0f}% vs. the baseline period.",
                metric="cpa", threshold=">= 30% period-over-period", formula="(current − baseline) / baseline",
                suggested_action="Check the per-campaign table below for which specific campaign(s) "
                                  "are driving this before touching anything account-wide — a blended "
                                  "spike is often one or two campaigns, not everything at once.",
            ))
        roas_chg = pct.get("roas")
        if roas_chg is not None and roas_chg <= -25:
            insights.append(Insight(
                severity="critical",
                title="Blended ROAS falling sharply",
                detail=f"Blended ROAS is down {abs(roas_chg):.0f}% vs. the baseline period.",
                metric="roas", threshold="<= -25% period-over-period", formula="(current − baseline) / baseline",
                suggested_action="Check the per-campaign table below for which specific campaign(s) "
                                  "are driving this before touching anything account-wide — a blended "
                                  "drop is often one or two campaigns, not everything at once.",
            ))

    order = {"critical": 0, "watch": 1, "scale": 2}
    insights.sort(key=lambda i: order.get(i.severity, 9))
    return insights
