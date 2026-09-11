"""
PPC Acquisition & Decision Intelligence — prototype.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from io import StringIO

import altair as alt
import pandas as pd
import streamlit as st

from src import db, mapping, metrics, normalize, orders, report

MAPPABLE_FIELDS = [f for f in mapping.CANONICAL_FIELDS if f != "level"]
LEVELS = ["campaign", "ad_set", "ad", "keyword"]


def _guess_field(header: str) -> str:
    h = header.strip().lower()
    guesses = [
        (("date", "day", "week", "month"), "date"),
        (("campaign",), "campaign"),
        (("ad group", "adset", "ad set"), "ad_set"),
        (("keyword",), "keyword"),
        (("spend", "cost", "amount spent"), "spend"),
        (("impression", "impr"), "impressions"),
        (("reach",), "reach"),
        (("click",), "clicks"),
        (("conversion", "lead", "result", "enrollment"), "conversions"),
        (("value", "revenue"), "conversion_value"),
        (("currency",), "currency"),
    ]
    for keys, field_name in guesses:
        if any(k in h for k in keys):
            return field_name
    return "ignore"

st.set_page_config(page_title="PPC Acquisition Intelligence", layout="wide")


def _check_password() -> bool:
    """Gate the whole app behind a single shared password, set via the
    APP_PASSWORD environment variable on the deployment (Render, not this
    repo — never hardcoded, never committed). Local dev with no
    APP_PASSWORD set stays open, so this never gets in the way of running
    it on your own machine. Session-scoped: each browser session that
    enters the correct password stays authenticated for that session only."""
    required = os.environ.get("APP_PASSWORD")
    if not required:
        return True
    if st.session_state.get("authenticated"):
        return True
    st.title("PPC Intelligence")
    pw = st.text_input("Password", type="password", key="pw_input")
    if pw:
        if pw == required:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not _check_password():
    st.stop()

db.init_db()

CURRENCIES = ["USD", "GBP", "BDT", "EUR"]


# --------------------------------------------------------------- helpers --

def money(v, ccy="USD"):
    if v is None or v != v:
        return "—"
    symbol = {"USD": "$", "GBP": "£", "EUR": "€"}.get(ccy, ccy + " ")
    return f"{symbol}{v:,.2f}"


def pct(v):
    return "—" if v is None or v != v else f"{v:+.1f}%"


def ratio(v, suffix="x"):
    return "—" if v is None or v != v else f"{v:.2f}{suffix}"


def md_safe(s: str) -> str:
    """Streamlit renders text between a pair of '$' as inline LaTeX math —
    any insight mentioning two dollar amounts (e.g. USD brands) would
    otherwise get silently mangled in st.write/st.caption. Escape before
    any insight text reaches markdown-rendering calls."""
    return s.replace("$", "\\$")


def brand_dict(row) -> dict:
    return dict(row) if row is not None else {}


# --------------------------------------------------------------- sidebar --

st.sidebar.title("PPC Intelligence")
brands = db.list_brands()
brand_names = {b["name"]: b["id"] for b in brands}

with st.sidebar.expander("+ New brand", expanded=(len(brands) == 0)):
    with st.form("new_brand"):
        name = st.text_input("Brand name")
        business_model = st.selectbox("Business model", ["transactional", "lead_gen"],
                                       help="transactional = e-commerce (ROAS/AOV); lead_gen = leads/enrollments (CPA)")
        conversion_type = st.text_input("Conversion type label", value="purchase" if business_model == "transactional" else "lead",
                                         help="e.g. purchase, lead, enrollment")
        currency = st.selectbox("Primary currency", CURRENCIES)
        col1, col2 = st.columns(2)
        with col1:
            margin_pct = st.number_input("Contribution margin (%)", 0.0, 100.0, 40.0) / 100
            aov = st.number_input("Avg. order value", 0.0, value=0.0)
        with col2:
            ltv = st.number_input("Customer/student LTV", 0.0, value=0.0)
            target_roas = st.number_input("Target ROAS (x)", 0.0, value=3.0) if business_model == "transactional" else None
        target_cpa = st.number_input("Target CPA", 0.0, value=50.0) if business_model == "lead_gen" else None
        target_payback_days = st.number_input("Target payback (days)", 0.0, value=90.0)
        if st.form_submit_button("Create brand"):
            if not name.strip():
                st.error("Brand name can't be empty.")
            elif db.get_brand_by_name(name.strip()):
                st.error(f"A brand named '{name}' already exists — pick a different name.")
            else:
                db.create_brand(
                    name=name.strip(), business_model=business_model, conversion_type=conversion_type,
                    currency=currency, margin_pct=margin_pct or None, aov=aov or None,
                    ltv=ltv or None, target_roas=target_roas, target_cpa=target_cpa,
                    target_payback_days=target_payback_days or None,
                )
                st.success(f"Created {name}")
                st.rerun()

if not brands:
    st.info("Create a brand in the sidebar to get started.")
    st.stop()

selected_name = st.sidebar.selectbox("Brand", list(brand_names.keys()))
brand_id = brand_names[selected_name]
brand = brand_dict(db.get_brand(brand_id))

st.sidebar.caption(
    f"{brand['business_model'].replace('_',' ').title()} · {brand['currency']} · "
    f"conversion = {brand['conversion_type']}"
)

tab_import, tab_dash, tab_insights, tab_tests, tab_recon, tab_export = st.tabs(
    ["📥 Import", "📊 Dashboard", "🧭 Insights", "🧪 A/B Tests", "💷 Reconciliation", "⚙️ Settings & Export"]
)

# ---------------------------------------------------------------- import --

with tab_import:
    st.subheader(f"Import PPC reports — {selected_name}")
    st.caption("Meta Ads Manager, Google Ads, or Microsoft Ads exports (CSV/XLSX). "
               "Platform and report level are auto-detected from the column headers.")

    uploaded = st.file_uploader("Drop export files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

    if uploaded:
        for f in uploaded:
            # Some real-world exports (a custom reporting-tool CSV, not a raw
            # platform-UI export) stack multiple tables in one file — see
            # normalize.split_multi_table_csv. XLSX files are read as a single
            # table; no evidence yet that xlsx exports have this shape.
            try:
                if f.name.lower().endswith(".csv"):
                    raw_text = f.read().decode("utf-8-sig")
                    blocks = normalize.split_multi_table_csv(raw_text)
                    sub_frames = [
                        (f"{f.name} — table {i+1}" if len(blocks) > 1 else f.name, pd.read_csv(StringIO(b)))
                        for i, b in enumerate(blocks)
                    ]
                else:
                    sub_frames = [(f.name, pd.read_excel(f))]
            except Exception as e:
                st.markdown(f"---\n**{f.name}**")
                st.error(f"Couldn't read this file at all: {e}")
                continue

            for sub_name, raw_df in sub_frames:
                key = sub_name  # unique per file+table, used to namespace widget keys below
                st.markdown(f"---\n**{sub_name}**")

                result = normalize.normalize_upload(raw_df, sub_name, brand["currency"])

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Detected platform", result.platform_label)
                c2.metric("Report level", result.level)
                c3.metric("Rows parsed", result.row_count)
                c4.metric("Date range", f"{result.date_start or '—'} → {result.date_end or '—'}")

                with st.expander("Detection confidence (why this platform?)"):
                    st.json(result.detection_scores)

                for w in result.warnings:
                    st.warning(w)

                if result.status == "failed" and result.platform_id is None:
                    st.error("Couldn't auto-detect the platform for this table.")
                    with st.expander(f"Map columns manually — {sub_name}", expanded=True):
                        st.caption("Generic CSV fallback: assign each column yourself. "
                                   "Required: date, campaign, spend.")
                        cols = list(raw_df.columns)
                        platform_label = st.text_input("Platform / source name", value="Generic export", key=f"plabel_{key}")
                        level_choice = st.selectbox("Report level", LEVELS, key=f"level_{key}")
                        choices = {}
                        grid = st.columns(2)
                        for i, col in enumerate(cols):
                            with grid[i % 2]:
                                choices[col] = st.selectbox(
                                    col, ["ignore"] + MAPPABLE_FIELDS,
                                    index=(["ignore"] + MAPPABLE_FIELDS).index(_guess_field(col)),
                                    key=f"map_{key}_{col}",
                                )
                        if st.button(f"Import with this mapping — {sub_name}", key=f"manual_import_{key}"):
                            m_result = normalize.normalize_manual_mapping(
                                raw_df, sub_name, brand["currency"], choices, level_choice, platform_label
                            )
                            for w in m_result.warnings:
                                st.warning(w)
                            if m_result.status == "failed":
                                st.error("Still can't import — fix the required-field mapping above.")
                            else:
                                import_id = db.create_import(
                                    brand_id=brand_id, platform=m_result.platform_id, level=m_result.level,
                                    filename=sub_name, date_start=m_result.date_start, date_end=m_result.date_end,
                                    row_count=m_result.row_count, status=m_result.status,
                                    unmapped_columns=m_result.unmapped_columns,
                                    notes="manual mapping: " + "; ".join(m_result.warnings),
                                )
                                db.insert_rows(import_id, brand_id, m_result.rows)
                                st.success(f"Imported {m_result.row_count} rows as '{platform_label}'.")
                                st.rerun()
                    continue

                rows_to_import = result.rows
                period_start = period_end = None
                if result.needs_period_date:
                    st.info("This table has no date column — pick the period it covers. "
                            "Every row will be recorded on the period's end date (daily "
                            "trend won't be available for this import, only period totals).")
                    default_end = date.today()
                    period_range = st.date_input(
                        "Period covered by this table", (default_end.replace(day=1), default_end),
                        key=f"period_{key}",
                    )
                    if len(period_range) == 2:
                        period_start, period_end = period_range
                        rows_to_import = [{**r, "date": str(period_end)} for r in result.rows]
                    else:
                        st.caption("Pick both a start and end date to continue.")

                confirm_disabled = result.needs_period_date and period_end is None
                date_start_for_import = str(period_start) if period_start else result.date_start
                date_end_for_import = str(period_end) if period_end else result.date_end

                if not confirm_disabled:
                    overlaps = db.find_overlapping_import(
                        brand_id, result.platform_id, date_start_for_import, date_end_for_import
                    )
                    if overlaps:
                        st.warning(
                            f"This date range overlaps {len(overlaps)} existing import(s) for "
                            f"{result.platform_label} already on file. Importing again will add "
                            f"duplicate rows unless you delete the old import first (Settings tab)."
                        )

                if st.button(f"Confirm import — {sub_name}", key=f"import_{key}", disabled=confirm_disabled):
                    import_id = db.create_import(
                        brand_id=brand_id, platform=result.platform_id, level=result.level,
                        filename=sub_name, date_start=date_start_for_import, date_end=date_end_for_import,
                        row_count=len(rows_to_import), status=result.status,
                        unmapped_columns=result.unmapped_columns,
                        notes="; ".join(result.warnings),
                    )
                    db.insert_rows(import_id, brand_id, rows_to_import)
                    st.success(f"Imported {len(rows_to_import)} rows.")
                    st.rerun()

    st.markdown("---")
    st.caption("**Not built yet:** automatic API sync (Meta/Google/Microsoft Marketing APIs) — "
               "a later phase, needs real hosting + OAuth app registration.")

# -------------------------------------------------------------- dashboard --

with tab_dash:
    st.subheader(f"Dashboard — {selected_name}")
    all_rows = db.rows_for_brand(brand_id)
    if not all_rows:
        st.info("No data yet — import a report first.")
    else:
        df = metrics.rows_to_df(all_rows)
        min_d, max_d = df["date"].min().date(), df["date"].max().date()
        default_start = max(min_d, max_d - timedelta(days=89))
        date_range = st.date_input("Date range", (default_start, max_d), min_value=min_d, max_value=max_d)

        # st.date_input returns a 1-tuple while the viewer has only picked the
        # start of the range (before clicking an end date) — unpacking it
        # unconditionally crashes the tab on that very common half-click, so
        # the rest of the tab is gated on having a real (start, end) pair.
        if len(date_range) != 2:
            st.info("Pick an end date to see this range.")
        else:
            start, end = date_range
            scoped = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
            totals = {
                "spend": scoped["spend"].sum(), "conversions": scoped["conversions"].sum(),
                "conversion_value": scoped["conversion_value"].sum(),
            }
            econ = metrics.brand_unit_economics(totals, brand)

            cols = st.columns(6)
            cols[0].metric("Spend", money(totals["spend"], brand["currency"]))
            cols[1].metric("Conversions", f"{totals['conversions']:,.0f}")
            if brand["business_model"] == "transactional":
                cols[2].metric("ROAS", ratio(econ["roas"]))
                cols[3].metric("iROAS floor", ratio(econ["iroas_floor"]) if econ["iroas_floor"] else "—")
            else:
                cols[2].metric("CAC / CPA", money(econ["cac"], brand["currency"]))
                cols[3].metric("Target CPA", money(brand["target_cpa"], brand["currency"]))
            cols[4].metric("LTV : CAC", ratio(econ["ltv_cac"]) if econ["ltv_cac"] else "—")
            payback_label = f"{econ['payback_orders']:.1f} orders" if econ["payback_orders"] else "—"
            cols[5].metric("Payback (orders to break even)", payback_label,
                            help="Repeat orders needed for cumulative contribution margin to recover CAC. "
                                 "Not a time period — that needs a repeat-purchase-cadence input this prototype doesn't collect yet.")

            st.markdown("#### Trend")
            daily = metrics.aggregate(scoped, by=["date"])
            left, right = st.columns(2)
            with left:
                chart = alt.Chart(daily).mark_line(point=True).encode(
                    x="date:T", y=alt.Y("spend:Q", title=f"Spend ({brand['currency']})"),
                    tooltip=["date:T", "spend:Q"],
                ).properties(height=260, title="Spend over time")
                st.altair_chart(chart, use_container_width=True)
            with right:
                metric_col = "roas" if brand["business_model"] == "transactional" else "cpa"
                chart2 = alt.Chart(daily.dropna(subset=[metric_col])).mark_line(point=True, color="#d95926").encode(
                    x="date:T", y=alt.Y(f"{metric_col}:Q", title=metric_col.upper()),
                    tooltip=["date:T", f"{metric_col}:Q"],
                ).properties(height=260, title=f"{metric_col.upper()} over time")
                st.altair_chart(chart2, use_container_width=True)

            st.markdown("#### Platform breakdown")
            plat = metrics.aggregate(scoped, by=["platform"])
            st.dataframe(plat.style.format({
                "spend": "{:,.2f}", "impressions": "{:,.0f}", "clicks": "{:,.0f}",
                "conversions": "{:,.0f}", "conversion_value": "{:,.2f}",
                "ctr": "{:.2%}", "cpc": "{:,.2f}", "cpa": "{:,.2f}", "roas": "{:.2f}x",
            }), use_container_width=True)

            st.markdown("#### Funnel")
            funnel_totals = scoped[["impressions", "clicks", "conversions"]].sum()
            fdf = pd.DataFrame({
                "stage": ["Impressions", "Clicks", "Conversions"],
                "value": [funnel_totals["impressions"], funnel_totals["clicks"], funnel_totals["conversions"]],
            })
            st.altair_chart(
                alt.Chart(fdf).mark_bar().encode(
                    y=alt.Y("stage:N", sort=None), x="value:Q", tooltip=["stage", "value"],
                    color=alt.Color("stage:N", legend=None),
                ).properties(height=180),
                use_container_width=True,
            )

            st.markdown("#### Campaign performance")
            camp = metrics.aggregate(scoped, by=["campaign"]).sort_values("spend", ascending=False)
            st.dataframe(camp.style.format({
                "spend": "{:,.2f}", "impressions": "{:,.0f}", "clicks": "{:,.0f}",
                "conversions": "{:,.0f}", "conversion_value": "{:,.2f}",
                "ctr": "{:.2%}", "cpc": "{:,.2f}", "cpa": "{:,.2f}", "roas": "{:.2f}x",
            }), use_container_width=True, height=320)

            if (scoped["level"] == "ad").any():
                fatigue = metrics.creative_fatigue_candidates(scoped[scoped["level"] == "ad"])
                if fatigue:
                    st.markdown("#### Creative fatigue signals")
                    st.caption("`frequency+ctr` = real signal (needs Meta's Reach field). "
                               "`ctr-proxy` = CTR-only fallback, weaker evidence.")
                    st.dataframe(pd.DataFrame(fatigue), use_container_width=True)

# --------------------------------------------------------------- insights --

with tab_insights:
    st.subheader(f"Insights — {selected_name}")
    all_rows = db.rows_for_brand(brand_id)
    if not all_rows:
        st.info("No data yet — import a report first.")
    else:
        df = metrics.rows_to_df(all_rows)
        max_d = df["date"].max().date()
        window = st.selectbox("Compare current period vs. baseline", ["Last 7 vs prior 7", "Last 30 vs prior 30"], index=1)
        n = 7 if "7" in window else 30
        cur_start, cur_end = max_d - timedelta(days=n - 1), max_d
        base_start, base_end = cur_start - timedelta(days=n), cur_start - timedelta(days=1)

        compare = metrics.compare_periods(df, cur_start, cur_end, base_start, base_end)
        camp_scoped = df[(df["date"] >= pd.Timestamp(cur_start)) & (df["date"] <= pd.Timestamp(cur_end))]
        camp_agg = metrics.aggregate(camp_scoped, by=["campaign"])

        insights = metrics.generate_insights(camp_agg, brand, compare)

        badge = {"critical": "🔴", "watch": "🟡", "scale": "🟢"}
        if not insights:
            st.success("No threshold breaches this period — nothing urgent to flag.")
        for ins in insights:
            with st.container(border=True):
                st.markdown(f"{badge.get(ins.severity,'')} **{md_safe(ins.title)}**")
                st.write(md_safe(ins.detail))
                st.caption(f"metric: `{ins.metric}` · threshold: `{md_safe(ins.threshold)}` · formula: `{ins.formula}`")

        st.markdown("---")
        st.caption("Period-over-period change (current vs. baseline):")
        pcols = st.columns(4)
        pcols[0].metric("Spend", money(compare["current"]["spend"], brand["currency"]), pct(compare["pct_change"]["spend"]))
        pcols[1].metric("Conversions", f"{compare['current']['conversions']:,.0f}", pct(compare["pct_change"]["conversions"]))
        pcols[2].metric("CPA", money(compare["current"]["cpa"], brand["currency"]), pct(compare["pct_change"]["cpa"]), delta_color="inverse")
        pcols[3].metric("ROAS", ratio(compare["current"]["roas"]), pct(compare["pct_change"]["roas"]))

        st.markdown("---")
        st.markdown("#### Anomalies")
        st.caption("Days where the day-over-day change is a statistical outlier vs. this series' own "
                   "volatility (z-score ≥ 2), not just an arbitrary percent move.")
        anomaly_metric = "roas" if brand["business_model"] == "transactional" else "cpa"
        full_daily = metrics.aggregate(df, by=["date"])
        if len(full_daily) < 5:
            st.info(f"Only {len(full_daily)} day(s) of history — need at least 5 for a z-score to mean anything.")
        else:
            flags = metrics.detect_anomalies(full_daily, anomaly_metric, z_thresh=2.0)
            if not flags:
                st.success(f"No statistically unusual day-over-day moves in {anomaly_metric.upper()} across "
                           f"{len(full_daily)} days of history.")
            else:
                for fl in flags:
                    d = pd.Timestamp(fl["date"]).date().isoformat()
                    val = money(fl["value"], brand["currency"]) if anomaly_metric == "cpa" else ratio(fl["value"])
                    st.warning(
                        f"**{d}** — {anomaly_metric.upper()} {val} "
                        f"({fl['pct_change']:+.0f}% day-over-day, z={fl['z_score']}, threshold=±{fl['threshold']})"
                    )

# ------------------------------------------------------------------ tests --

with tab_tests:
    st.subheader(f"A/B & creative tests — {selected_name}")

    with st.expander("+ Log a new test"):
        with st.form("new_test"):
            tname = st.text_input("Test name")
            hyp = st.text_area("Hypothesis")
            colA, colB = st.columns(2)
            with colA:
                a_label = st.text_input("Variant A label", value="Control")
                a_campaigns = st.text_area("Variant A campaign names (one per line)")
            with colB:
                b_label = st.text_input("Variant B label", value="Test")
                b_campaigns = st.text_area("Variant B campaign names (one per line)")
            if st.form_submit_button("Save test") and tname:
                db.create_test(
                    brand_id=brand_id, name=tname, hypothesis=hyp,
                    variant_a_label=a_label, variant_a_campaigns=json.dumps([c.strip() for c in a_campaigns.splitlines() if c.strip()]),
                    variant_b_label=b_label, variant_b_campaigns=json.dumps([c.strip() for c in b_campaigns.splitlines() if c.strip()]),
                    started_at=date.today().isoformat(), status="running",
                )
                st.success("Test saved.")
                st.rerun()

    tests = db.list_tests(brand_id)
    all_rows = db.rows_for_brand(brand_id)
    df = metrics.rows_to_df(all_rows) if all_rows else pd.DataFrame()

    if not tests:
        st.info("No tests logged yet.")
    for t in tests:
        with st.container(border=True):
            st.markdown(f"**{t['name']}** · {t['status']}")
            if t["hypothesis"]:
                st.caption(t["hypothesis"])
            a_camps, b_camps = json.loads(t["variant_a_campaigns"] or "[]"), json.loads(t["variant_b_campaigns"] or "[]")
            st.write(f"**{t['variant_a_label']}**: {', '.join(a_camps) or '—'}  \n**{t['variant_b_label']}**: {', '.join(b_camps) or '—'}")

            if not df.empty and a_camps and b_camps:
                a_df = df[df["campaign"].isin(a_camps)]
                b_df = df[df["campaign"].isin(b_camps)]
                a_conv, a_clicks = a_df["conversions"].sum(), a_df["clicks"].sum()
                b_conv, b_clicks = b_df["conversions"].sum(), b_df["clicks"].sum()
                test_result = metrics.two_proportion_z_test(a_conv, a_clicks, b_conv, b_clicks)

                c1, c2, c3 = st.columns(3)
                c1.metric(f"{t['variant_a_label']} conv. rate", f"{test_result.get('rate_a',0):.2%}" if a_clicks else "—")
                c2.metric(f"{t['variant_b_label']} conv. rate", f"{test_result.get('rate_b',0):.2%}" if b_clicks else "—")
                sig = test_result.get("significant")
                c3.metric("Statistically significant?", "Yes" if sig else "No",
                          f"p={test_result.get('p_value','—')}")
                if test_result.get("low_sample_warning"):
                    st.caption("⚠️ Sample size under 30 clicks on at least one side — treat this read cautiously.")
            else:
                st.caption("No matching campaign data yet for this test's variants.")

# -------------------------------------------------------- reconciliation --

with tab_recon:
    st.subheader(f"Revenue reconciliation — {selected_name}")
    st.caption("Cross-check actual order revenue against what each platform claims as conversion value. "
               "Platforms attribute from their own pixel, which can overstate (or understate) real revenue — "
               "this is the ground-truth check.")

    st.markdown("#### Import actual orders")
    order_files = st.file_uploader("Drop an order/sales export (needs a date and an amount column)",
                                    type=["csv"], accept_multiple_files=True, key="order_uploader")
    if order_files:
        for f in order_files:
            st.markdown(f"---\n**{f.name}**")
            try:
                raw_text = f.read().decode("utf-8-sig")
                blocks = normalize.split_multi_table_csv(raw_text)
                block_df = pd.read_csv(StringIO(blocks[0]))
            except Exception as e:
                st.error(f"Couldn't read this file: {e}")
                continue

            if not orders.looks_like_order_sheet(list(block_df.columns)):
                st.warning("This doesn't look like an order sheet (needs a date column and an amount "
                           "column, and no ad-metric columns like spend/impressions/clicks). Skipped.")
                continue

            o_result = orders.normalize_orders(block_df, f.name)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows parsed", o_result.row_count)
            c2.metric("Date range", f"{o_result.date_start or '—'} → {o_result.date_end or '—'}")
            c3.metric("Total amount", money(sum(r["amount"] for r in o_result.rows), brand["currency"]) if o_result.rows else "—")
            for w in o_result.warnings:
                st.warning(w)

            if o_result.rows and st.button(f"Import orders — {f.name}", key=f"order_import_{f.name}"):
                import_id = db.create_import(
                    brand_id=brand_id, platform="orders", level="order", filename=f.name,
                    date_start=o_result.date_start, date_end=o_result.date_end,
                    row_count=o_result.row_count, status=o_result.status,
                    unmapped_columns=o_result.unmapped_columns,
                )
                db.insert_orders(import_id, brand_id, o_result.rows)
                st.success(f"Imported {o_result.row_count} orders.")
                st.rerun()

    all_orders_unfiltered = db.orders_for_brand(brand_id)
    if not all_orders_unfiltered:
        st.info("No orders imported yet — drop an order/sales export above to get started.")
    else:
        st.markdown("---")
        st.markdown("#### Date range")
        st.caption("Scopes both the orders and the campaign data below to the same period — "
                   "reconciling different date ranges against each other isn't a fair comparison.")
        order_dates = [o["order_date"] for o in all_orders_unfiltered]
        recon_min_d = min(pd.to_datetime(order_dates)).date()
        recon_max_d = max(pd.to_datetime(order_dates)).date()
        recon_range = st.date_input("Reconciliation period", (recon_min_d, recon_max_d),
                                     min_value=recon_min_d, max_value=recon_max_d, key="recon_range")

        if len(recon_range) != 2:
            st.info("Pick an end date to continue.")
        else:
            recon_start, recon_end = str(recon_range[0]), str(recon_range[1])
            all_orders = db.orders_for_brand(brand_id, recon_start, recon_end)
            perf_rows = db.rows_for_brand(brand_id, recon_start, recon_end)

            st.markdown("---")
            st.markdown("#### Match campaign names")
            st.caption("Order sheets use shorthand campaign names that rarely match the platform's exact "
                       "names exactly. Every suggestion below is a guess — confirm or correct each one; "
                       "nothing is used for reconciliation until you save.")

            unmatched = db.unmatched_raw_campaigns(brand_id)
            campaigns_by_platform: dict[str, list[str]] = {}
            for r in perf_rows:
                campaigns_by_platform.setdefault(r["platform"], [])
                if r["campaign"] and r["campaign"] not in campaigns_by_platform[r["platform"]]:
                    campaigns_by_platform[r["platform"]].append(r["campaign"])
            all_known_campaigns = sorted({c for cs in campaigns_by_platform.values() for c in cs})

            if not all_known_campaigns:
                st.info("No campaign performance data in this date range — import that first "
                        "(Import tab), or widen the range above, so there's something to match orders against.")
            elif not unmatched:
                st.success("All order campaigns are matched.")
            else:
                order_rows_by_campaign: dict[str, str | None] = {}
                for o in all_orders:
                    if o["raw_campaign"] and o["raw_campaign"] not in order_rows_by_campaign:
                        order_rows_by_campaign[o["raw_campaign"]] = o["source"]

                with st.form("campaign_matches"):
                    choices = {}
                    options = ["ignore"] + all_known_campaigns
                    for raw in unmatched:
                        source = order_rows_by_campaign.get(raw)
                        platform_hint = orders.SOURCE_TO_PLATFORM.get((source or "").lower())
                        scoped = campaigns_by_platform.get(platform_hint, []) if platform_hint else all_known_campaigns
                        suggestion = orders.suggest_campaign_matches(
                            [raw], {platform_hint: scoped} if platform_hint else campaigns_by_platform
                        )[raw]
                        default_idx = options.index(suggestion) if suggestion in options else 0
                        label = f"{raw}" + (f"  (source: {source})" if source else "")
                        choices[raw] = st.selectbox(label, options, index=default_idx, key=f"match_{raw}")
                    if st.form_submit_button("Save matches"):
                        db.set_campaign_matches(brand_id, choices)
                        st.success("Saved.")
                        st.rerun()

            st.markdown("---")
            st.markdown("#### Reconciliation")
            odf = pd.DataFrame([dict(o) for o in all_orders])
            odf = odf[~odf["matched_campaign"].isin([None, "ignore"])]
            if odf.empty:
                st.info("No matched orders in this date range yet to reconcile — match campaigns above first.")
            else:
                order_totals = odf.groupby("matched_campaign")["amount"].sum().rename("actual_revenue")
                perf_df = metrics.rows_to_df(perf_rows)
                claimed_totals = metrics.aggregate(perf_df, by=["campaign"])[["campaign", "conversion_value"]].set_index("campaign")["conversion_value"].rename("platform_claimed")

                recon = pd.concat([order_totals, claimed_totals], axis=1).fillna(0.0).reset_index()
                recon.columns = ["campaign", "actual_revenue", "platform_claimed"]
                recon["delta"] = recon["actual_revenue"] - recon["platform_claimed"]
                recon["delta_pct"] = (recon["delta"] / recon["platform_claimed"].replace(0, pd.NA)) * 100
                recon = recon.sort_values("actual_revenue", ascending=False)

                t1, t2, t3 = st.columns(3)
                t1.metric("Actual revenue (orders)", money(recon["actual_revenue"].sum(), brand["currency"]))
                t2.metric("Platform-claimed revenue", money(recon["platform_claimed"].sum(), brand["currency"]))
                total_delta_pct = (recon["actual_revenue"].sum() - recon["platform_claimed"].sum()) / recon["platform_claimed"].sum() * 100 if recon["platform_claimed"].sum() else None
                t3.metric("Overall difference", pct(total_delta_pct))

                st.dataframe(recon.style.format({
                    "actual_revenue": "{:,.2f}", "platform_claimed": "{:,.2f}",
                    "delta": "{:,.2f}", "delta_pct": "{:+.1f}%",
                }), use_container_width=True)
                st.caption("Positive delta = platforms under-claimed vs. real revenue. Negative = platforms "
                           "over-claimed (common with pixel-based attribution).")

# ------------------------------------------------------------- settings ---

with tab_export:
    st.subheader(f"Settings & export — {selected_name}")

    with st.form("edit_brand"):
        st.write("**Targets & unit-economics assumptions**")
        col1, col2 = st.columns(2)
        with col1:
            margin_pct = st.number_input("Contribution margin (%)", 0.0, 100.0, (brand["margin_pct"] or 0.0) * 100) / 100
            aov = st.number_input("Avg. order value", 0.0, value=brand["aov"] or 0.0)
            ltv = st.number_input("Customer/student LTV", 0.0, value=brand["ltv"] or 0.0)
        with col2:
            target_roas = st.number_input("Target ROAS (x)", 0.0, value=brand["target_roas"] or 0.0)
            target_cpa = st.number_input("Target CPA", 0.0, value=brand["target_cpa"] or 0.0)
            target_payback_days = st.number_input("Target payback (days)", 0.0, value=brand["target_payback_days"] or 0.0)
        if st.form_submit_button("Save targets"):
            db.update_brand(brand_id, margin_pct=margin_pct or None, aov=aov or None, ltv=ltv or None,
                             target_roas=target_roas or None, target_cpa=target_cpa or None,
                             target_payback_days=target_payback_days or None)
            st.success("Saved.")
            st.rerun()

    st.markdown("---")
    st.write("**Imports on file**")
    imports = db.list_imports(brand_id)
    if imports:
        idf = pd.DataFrame([dict(i) for i in imports])[
            ["id", "platform", "level", "filename", "date_start", "date_end", "row_count", "status", "imported_at"]
        ]
        st.dataframe(idf, use_container_width=True)
        del_id = st.number_input("Import ID to delete", min_value=0, value=0, step=1)
        if st.button("Delete import") and del_id:
            if db.delete_import(int(del_id), brand_id):
                st.success(f"Deleted import {del_id}.")
                st.rerun()
            else:
                st.error(f"Import {del_id} doesn't belong to {selected_name} — nothing deleted.")
    else:
        st.caption("No imports yet.")

    st.markdown("---")
    st.write("**Export normalized data**")
    all_rows = db.rows_for_brand(brand_id)
    if all_rows:
        export_df = pd.DataFrame([dict(r) for r in all_rows])
        st.download_button("Download CSV (normalized rows)", export_df.to_csv(index=False),
                            file_name=f"{selected_name.replace(' ','_')}_normalized.csv", mime="text/csv")

    st.markdown("---")
    st.write("**Client-ready PDF report**")
    if not all_rows:
        st.caption("No data yet — import a report first.")
    else:
        rdf = metrics.rows_to_df(all_rows)
        r_min_d, r_max_d = rdf["date"].min().date(), rdf["date"].max().date()
        r_default_start = max(r_min_d, r_max_d - timedelta(days=29))
        r_date_range = st.date_input("Report date range", (r_default_start, r_max_d),
                                      min_value=r_min_d, max_value=r_max_d, key="report_range")
        include_insights_flag = st.checkbox("Include growth insights section", value=True)

        if len(r_date_range) != 2:
            st.info("Pick an end date to generate a report.")
        else:
            r_start, r_end = r_date_range
            r_scoped = rdf[(rdf["date"] >= pd.Timestamp(r_start)) & (rdf["date"] <= pd.Timestamp(r_end))]
            r_totals = {"spend": r_scoped["spend"].sum(), "conversions": r_scoped["conversions"].sum(),
                        "conversion_value": r_scoped["conversion_value"].sum()}
            r_econ = metrics.brand_unit_economics(r_totals, brand)
            r_econ["spend"] = r_totals["spend"]
            r_econ["conversions"] = r_totals["conversions"]
            r_camp_agg = metrics.aggregate(r_scoped, by=["campaign"])
            r_daily = metrics.aggregate(r_scoped, by=["date"])
            r_insights = metrics.generate_insights(r_camp_agg, brand) if include_insights_flag else []

            if st.button("Generate PDF report"):
                pdf_bytes = report.build_pdf_report(
                    brand, str(r_start), str(r_end), r_camp_agg, r_econ, r_daily,
                    r_insights, include_insights=include_insights_flag,
                )
                st.download_button(
                    "Download PDF", pdf_bytes,
                    file_name=f"{selected_name.replace(' ','_')}_{r_start}_{r_end}.pdf",
                    mime="application/pdf",
                )
    st.caption(
        "**Shareable web link:** this app can't create a public URL by itself (it only runs on "
        "your machine, and deploying it publicly is a separate decision). Ask Claude in chat to "
        "\"publish a shareable report for <brand>, <date range>\" and it generates this same report "
        "as a private link instead — src/report_html.py is what it uses."
    )
