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
                ))
            elif target_roas and row["roas"] < target_roas:
                insights.append(Insight(
                    severity="watch",
                    title=f"{camp}: below target ROAS",
                    detail=f"ROAS is {row['roas']:.2f}x vs. a target of {target_roas:.2f}x.",
                    metric="roas", threshold=f"< {target_roas:.2f}x", formula="target_roas (brand config)",
                ))
            elif target_roas and row["roas"] >= target_roas * 1.25:
                insights.append(Insight(
                    severity="scale",
                    title=f"{camp}: strong scale candidate",
                    detail=f"ROAS is {row['roas']:.2f}x, 25%+ above target ({target_roas:.2f}x).",
                    metric="roas", threshold=f">= {target_roas*1.25:.2f}x", formula="target_roas × 1.25",
                ))

        if business_model == "lead_gen" and target_cpa and row["cpa"] is not None:
            if row["cpa"] > target_cpa * 1.15:
                insights.append(Insight(
                    severity="critical",
                    title=f"{camp}: CPA over target",
                    detail=(f"CPA is {_money(row['cpa'], ccy)} vs. a target of {_money(target_cpa, ccy)} "
                            f"(+15% tolerance breached)."),
                    metric="cpa", threshold=f"> {_money(target_cpa*1.15, ccy)}", formula="target_cpa × 1.15",
                ))
            elif row["cpa"] > target_cpa:
                insights.append(Insight(
                    severity="watch",
                    title=f"{camp}: CPA above target",
                    detail=f"CPA is {_money(row['cpa'], ccy)} vs. a target of {_money(target_cpa, ccy)}.",
                    metric="cpa", threshold=f"> {_money(target_cpa, ccy)}", formula="target_cpa (brand config)",
                ))
            elif row["cpa"] <= target_cpa * 0.8:
                insights.append(Insight(
                    severity="scale",
                    title=f"{camp}: efficient scale candidate",
                    detail=f"CPA is {_money(row['cpa'], ccy)}, 20%+ under target ({_money(target_cpa, ccy)}).",
                    metric="cpa", threshold=f"<= {_money(target_cpa*0.8, ccy)}", formula="target_cpa × 0.8",
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
            ))
        roas_chg = pct.get("roas")
        if roas_chg is not None and roas_chg <= -25:
            insights.append(Insight(
                severity="critical",
                title="Blended ROAS falling sharply",
                detail=f"Blended ROAS is down {abs(roas_chg):.0f}% vs. the baseline period.",
                metric="roas", threshold="<= -25% period-over-period", formula="(current − baseline) / baseline",
            ))

    order = {"critical": 0, "watch": 1, "scale": 2}
    insights.sort(key=lambda i: order.get(i.severity, 9))
    return insights
