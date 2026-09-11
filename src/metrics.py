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


def _fit_marginal_efficiency(daily: pd.DataFrame, business_model: str) -> dict | None:
    """
    Shared curve-fit core for campaign_marginal_efficiency() and
    platform_marginal_efficiency() — fits a simple power-law response
    curve (value ~ a * spend^b — the standard simplified shape for a
    diminishing-returns media response, not a novel model) to a daily
    spend/value series, and reads the marginal cost/value of the NEXT
    dollar off the fitted elasticity `b`: b < 1 means diminishing
    returns (the margin is worse than the average so far); b > 1 means
    still scaling well (the margin is better than average).

    Deliberately returns None — rather than a number the data can't
    support — when: there's under 8 days of real spend+value; day-to-
    day spend barely varies (coefficient of variation < 0.15, i.e.
    nothing to fit a slope against); the fitted line explains under 30%
    of the variance (r_squared < 0.3); or, for a CPA series, the fitted
    elasticity is at or below zero (spend increases aren't tracking
    with conversions at all — inverting that into a "marginal CPA"
    would be nonsense, though a flat/negative elasticity is itself
    worth knowing and shows up via r_squared/None).
    """
    daily = daily[(daily["spend"] > 0) & (daily["value"] > 0)]
    if len(daily) < 8:
        return None

    spend = daily["spend"].to_numpy(dtype=float)
    value = daily["value"].to_numpy(dtype=float)
    cv = spend.std() / spend.mean() if spend.mean() else 0.0
    if cv < 0.15:
        return None

    log_spend, log_value = np.log(spend), np.log(value)
    b, log_a = np.polyfit(log_spend, log_value, 1)
    fitted = log_a + b * log_spend
    ss_res = np.sum((log_value - fitted) ** 2)
    ss_tot = np.sum((log_value - log_value.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    if r_squared < 0.3:
        return None

    avg_spend, avg_value = spend.mean(), value.mean()
    if business_model == "transactional":
        avg_metric = avg_value / avg_spend  # average ROAS
        marginal_metric = avg_metric * b    # marginal revenue per $ = b * avg ROAS
    else:
        if b <= 0:
            return None
        avg_metric = avg_spend / avg_value  # average CPA
        marginal_metric = avg_metric / b    # marginal cost per conversion

    return {
        "elasticity": round(float(b), 3),
        "r_squared": round(float(r_squared), 3),
        "avg_metric": float(avg_metric),
        "marginal_metric": float(marginal_metric),
        "days_used": int(len(daily)),
    }


def campaign_marginal_efficiency(df: pd.DataFrame, campaign: str, business_model: str,
                                  lookback_days: int = 60) -> dict | None:
    """Is the next dollar spent on THIS campaign still efficient? See
    _fit_marginal_efficiency() for the method and why it returns None
    rather than a number the data can't support."""
    if df.empty:
        return None
    value_col = "conversion_value" if business_model == "transactional" else "conversions"
    end = df["date"].max()
    start = end - pd.Timedelta(days=lookback_days)
    camp_df = df[(df["campaign"] == campaign) & (df["date"] >= start) & (df["date"] <= end)]
    daily = camp_df.groupby("date", dropna=False).agg(spend=("spend", "sum"), value=(value_col, "sum")).reset_index()
    return _fit_marginal_efficiency(daily, business_model)


def platform_marginal_efficiency(df: pd.DataFrame, platform: str, business_model: str,
                                  lookback_days: int = 60) -> dict | None:
    """Same question one level up: is the next dollar better spent on
    THIS platform (combining all its campaigns) than elsewhere — "Meta
    vs. Google/Microsoft," not "which campaign within one platform."
    Same method and same None-rather-than-guess guards as
    campaign_marginal_efficiency() — see _fit_marginal_efficiency()."""
    if df.empty:
        return None
    value_col = "conversion_value" if business_model == "transactional" else "conversions"
    end = df["date"].max()
    start = end - pd.Timedelta(days=lookback_days)
    plat_df = df[(df["platform"] == platform) & (df["date"] >= start) & (df["date"] <= end)]
    daily = plat_df.groupby("date", dropna=False).agg(spend=("spend", "sum"), value=(value_col, "sum")).reset_index()
    return _fit_marginal_efficiency(daily, business_model)


def _rank_by_marginal_efficiency(entity_agg: pd.DataFrame, entity_col: str, marginal_fn, brand,
                                  min_spend_share: float) -> list[dict]:
    """Shared ranking core for budget_reallocation_view() (entity =
    campaign) and cross_platform_reallocation_view() (entity =
    platform): ranks entities with a real spend share by the best
    available read on where the NEXT dollar is efficient — a marginal
    estimate where the entity's own history supports one, falling back
    to plain average CPA/ROAS, clearly labeled as such, where it
    doesn't. Basis for a reallocation suggestion, not a guaranteed-
    optimal split: says what's worth a look, not exactly how much to move."""
    if entity_agg.empty:
        return []
    total_spend = entity_agg["spend"].sum()
    if not total_spend:
        return []
    is_transactional = brand["business_model"] == "transactional"
    target = brand.get("target_roas") if is_transactional else brand.get("target_cpa")

    rows = []
    for _, r in entity_agg.iterrows():
        if r["spend"] < total_spend * min_spend_share:
            continue
        marginal = marginal_fn(r[entity_col])
        avg_metric = r.get("roas") if is_transactional else r.get("cpa")
        ranking_metric = marginal["marginal_metric"] if marginal else avg_metric
        efficient_at_target = None
        if target and ranking_metric is not None:
            efficient_at_target = (ranking_metric >= target) if is_transactional else (ranking_metric <= target)
        rows.append({
            entity_col: r[entity_col], "spend": float(r["spend"]),
            "avg_metric": float(avg_metric) if avg_metric is not None else None,
            "marginal_metric": marginal["marginal_metric"] if marginal else None,
            "elasticity": marginal["elasticity"] if marginal else None,
            "r_squared": marginal["r_squared"] if marginal else None,
            "basis": "marginal" if marginal else "average (not enough spend variation/history for a marginal estimate)",
            "ranking_metric": float(ranking_metric) if ranking_metric is not None else None,
            "efficient_at_target": efficient_at_target,
        })

    def _sort_key(row):
        if row["ranking_metric"] is None:
            return float("inf")
        return -row["ranking_metric"] if is_transactional else row["ranking_metric"]

    rows.sort(key=_sort_key)
    return rows


def budget_reallocation_view(df: pd.DataFrame, camp_agg: pd.DataFrame, brand,
                              min_spend_share: float = 0.05, lookback_days: int = 60) -> list[dict]:
    """Ranks this period's campaigns by where the next dollar is
    efficient — see _rank_by_marginal_efficiency()."""
    return _rank_by_marginal_efficiency(
        camp_agg, "campaign",
        lambda campaign: campaign_marginal_efficiency(df, campaign, brand["business_model"], lookback_days),
        brand, min_spend_share,
    )


def cross_platform_reallocation_view(df: pd.DataFrame, brand, min_spend_share: float = 0.05,
                                      lookback_days: int = 60) -> list[dict]:
    """
    Same ranking as budget_reallocation_view(), one level up: which
    PLATFORM — not which campaign within one — is the more efficient
    place for the next dollar right now. Only meaningful once a brand
    actually spends on more than one platform in this data; returns []
    otherwise rather than "ranking" a single platform against itself.
    """
    if df.empty:
        return []
    plat_agg = aggregate(df, by=["platform"])
    if len(plat_agg) < 2:
        return []
    return _rank_by_marginal_efficiency(
        plat_agg, "platform",
        lambda platform: platform_marginal_efficiency(df, platform, brand["business_model"], lookback_days),
        brand, min_spend_share,
    )


def campaign_totals(df: pd.DataFrame, campaign: str, start, end) -> dict:
    """Raw summed totals for one campaign over one date range — the plain
    building block decompose_metric_change() compares two of."""
    scoped = df[(df["campaign"] == campaign) & (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return {
        "spend": float(scoped["spend"].sum()), "impressions": float(scoped["impressions"].sum()),
        "clicks": float(scoped["clicks"].sum()), "conversions": float(scoped["conversions"].sum()),
        "conversion_value": float(scoped["conversion_value"].sum()),
    }


def decompose_metric_change(current: dict, baseline: dict, business_model: str) -> dict | None:
    """
    An insight can say a campaign's CPA/ROAS broke a threshold, but not
    WHY it moved — this decomposes the move into its funnel drivers
    using an exact identity, not a fitted/approximate attribution:
        CPA  = CPM ÷ (1000 × CTR × CVR)
        ROAS = (CTR × CVR × AOV × 1000) ÷ CPM
    (CPM = cost per 1,000 impressions, CTR = clicks/impressions,
    CVR = conversions/clicks, AOV = conversion_value/conversions.) Each
    driver's log-ratio (current vs. baseline) sums EXACTLY to the
    metric's own log-ratio — every dollar of the change is accounted
    for by one of these three (or four, for ROAS) numbers, nothing left
    as an unexplained residual.

    Returns None when either period has no real volume (zero
    impressions, clicks, conversions, or spend) — there's nothing
    honest to decompose from a period that didn't actually run.
    """
    def rates(totals):
        imp, clk, conv, spend, value = (totals["impressions"], totals["clicks"],
                                         totals["conversions"], totals["spend"], totals["conversion_value"])
        if not imp or not clk or not conv or not spend:
            return None
        return {"cpm": spend / imp * 1000, "ctr": clk / imp, "cvr": conv / clk,
                "aov": (value / conv) if conv else None}

    cur, base = rates(current), rates(baseline)
    if cur is None or base is None:
        return None

    driver_keys = ["cpm", "ctr", "cvr"] + (["aov"] if business_model == "transactional" else [])
    drivers = {}
    for key in driver_keys:
        cv, bv = cur[key], base[key]
        if not cv or not bv:
            return None
        drivers[key] = {"current": cv, "baseline": bv, "pct_change": (cv - bv) / bv * 100,
                         "log_ratio": float(np.log(cv / bv))}

    if business_model == "transactional":
        # log(ROAS_ratio) = log(ctr) + log(cvr) + log(aov) - log(cpm); positive = ROAS improved
        signed = {"ctr": 1, "cvr": 1, "aov": 1, "cpm": -1}
    else:
        # log(CPA_ratio) = log(cpm) - log(ctr) - log(cvr); positive = CPA worsened
        signed = {"cpm": 1, "ctr": -1, "cvr": -1}

    contributions = {k: signed[k] * drivers[k]["log_ratio"] for k in drivers}
    metric_log_ratio = sum(contributions.values())
    total_abs = sum(abs(v) for v in contributions.values()) or 1e-12
    primary = max(contributions, key=lambda k: abs(contributions[k]))
    # `signed` above defines metric_log_ratio as log(CPA_ratio) for lead_gen
    # (positive = CPA rose = worse) but log(ROAS_ratio) for transactional
    # (positive = ROAS rose = BETTER) — flip the read for transactional so
    # "worsened" means the same real-world thing in both cases.
    primary_pushed_worse = (contributions[primary] > 0) if business_model != "transactional" \
        else (contributions[primary] < 0)

    return {
        "metric_pct_change": (float(np.exp(metric_log_ratio)) - 1) * 100,
        "drivers": drivers,
        "contribution_share_pct": {k: abs(v) / total_abs * 100 for k, v in contributions.items()},
        "primary_driver": primary,
        "primary_driver_share_pct": abs(contributions[primary]) / total_abs * 100,
        "primary_driver_worsened_metric": primary_pushed_worse,
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


def keyword_waste_candidates(df: pd.DataFrame, min_clicks: int = 15) -> list[dict]:
    """
    Classic PPC hygiene: a keyword burning real clicks with zero
    conversions is usually the single highest-ROI thing to fix in a
    search account. Grounded with the "rule of three" — for zero
    observed successes in n trials, the upper bound of a 95% confidence
    interval on the true success rate is approximately 3/n — rather
    than an arbitrary flat click count. That upper bound is compared
    against THIS account's own blended conversion rate (from whichever
    keywords in the same window did convert), so the bar is calibrated
    to this specific brand and period, not a one-size-fits-all industry
    rule of thumb. A keyword only surfaces when even its best-case
    plausible conversion rate is still below what the rest of the
    account is actually achieving.

    Only Google/Microsoft carry keyword-level data — Meta has no
    keyword level at all. Needs >= min_clicks clicks to say anything;
    fewer than that and zero conversions could just be noise, not signal.
    """
    if df.empty or "keyword" not in df.columns:
        return []
    kw_df = df[df["keyword"].notna() & df["platform"].isin(["google", "microsoft"])]
    if kw_df.empty:
        return []

    agg = kw_df.groupby(["platform", "campaign", "keyword"], dropna=False).agg(
        spend=("spend", "sum"), clicks=("clicks", "sum"), conversions=("conversions", "sum")
    ).reset_index()

    converting = agg[agg["conversions"] > 0]
    total_clicks, total_conversions = converting["clicks"].sum(), converting["conversions"].sum()
    account_cvr = (total_conversions / total_clicks) if total_clicks else None

    candidates = []
    for _, r in agg.iterrows():
        if r["clicks"] < min_clicks or r["conversions"] > 0:
            continue
        upper_bound_cvr = 3 / r["clicks"]
        worth_reviewing = account_cvr is None or upper_bound_cvr < account_cvr
        if not worth_reviewing:
            continue
        candidates.append({
            "platform": r["platform"], "campaign": r["campaign"], "keyword": r["keyword"],
            "spend": float(r["spend"]), "clicks": int(r["clicks"]),
            "upper_bound_cvr_pct": round(upper_bound_cvr * 100, 2),
            "account_cvr_pct": round(float(account_cvr) * 100, 2) if account_cvr is not None else None,
        })

    candidates.sort(key=lambda c: -c["spend"])
    return candidates


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
    campaign: str | None = None  # which campaign this is about, if any — lets a caller
                                  # look up a root-cause decomposition without parsing `title`


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
                    campaign=camp,
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
                    campaign=camp,
                    title=f"{camp}: below target ROAS",
                    detail=f"ROAS is {row['roas']:.2f}x vs. a target of {target_roas:.2f}x.",
                    metric="roas", threshold=f"< {target_roas:.2f}x", formula="target_roas (brand config)",
                    suggested_action="Still profitable, just underperforming — worth a creative "
                                      "refresh or tighter targeting before cutting budget outright.",
                ))
            elif target_roas and row["roas"] >= target_roas * 1.25:
                insights.append(Insight(
                    severity="scale",
                    campaign=camp,
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
                    campaign=camp,
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
                    campaign=camp,
                    title=f"{camp}: CPA above target",
                    detail=f"CPA is {_money(row['cpa'], ccy)} vs. a target of {_money(target_cpa, ccy)}.",
                    metric="cpa", threshold=f"> {_money(target_cpa, ccy)}", formula="target_cpa (brand config)",
                    suggested_action="Not urgent yet — worth tightening targeting or testing new ad "
                                      "copy before it drifts further past target.",
                ))
            elif row["cpa"] <= target_cpa * 0.8:
                insights.append(Insight(
                    severity="scale",
                    campaign=camp,
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
