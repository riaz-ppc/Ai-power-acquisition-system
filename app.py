"""
PPC Acquisition & Decision Intelligence — prototype.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from src import db, mapping, metrics, normalize, report

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

tab_import, tab_dash, tab_insights, tab_tests, tab_export = st.tabs(
    ["📥 Import", "📊 Dashboard", "🧭 Insights", "🧪 A/B Tests", "⚙️ Settings & Export"]
)

# ---------------------------------------------------------------- import --

with tab_import:
    st.subheader(f"Import PPC reports — {selected_name}")
    st.caption("Meta Ads Manager, Google Ads, or Microsoft Ads exports (CSV/XLSX). "
               "Platform and report level are auto-detected from the column headers.")

    uploaded = st.file_uploader("Drop export files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

    if uploaded:
        for f in uploaded:
            st.markdown(f"---\n**{f.name}**")
            try:
                if f.name.lower().endswith(".csv"):
                    raw_df = pd.read_csv(f)
                else:
                    raw_df = pd.read_excel(f)
            except Exception as e:
                st.error(f"Couldn't read this file at all: {e}")
                continue

            result = normalize.normalize_upload(raw_df, f.name, brand["currency"])

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
                st.error("Couldn't auto-detect the platform for this file.")
                with st.expander(f"Map columns manually — {f.name}", expanded=True):
                    st.caption("Generic CSV fallback: assign each column yourself. "
                               "Required: date, campaign, spend.")
                    cols = list(raw_df.columns)
                    platform_label = st.text_input("Platform / source name", value="Generic export", key=f"plabel_{f.name}")
                    level_choice = st.selectbox("Report level", LEVELS, key=f"level_{f.name}")
                    choices = {}
                    grid = st.columns(2)
                    for i, col in enumerate(cols):
                        with grid[i % 2]:
                            choices[col] = st.selectbox(
                                col, ["ignore"] + MAPPABLE_FIELDS,
                                index=(["ignore"] + MAPPABLE_FIELDS).index(_guess_field(col)),
                                key=f"map_{f.name}_{col}",
                            )
                    if st.button(f"Import with this mapping — {f.name}", key=f"manual_import_{f.name}"):
                        m_result = normalize.normalize_manual_mapping(
                            raw_df, f.name, brand["currency"], choices, level_choice, platform_label
                        )
                        for w in m_result.warnings:
                            st.warning(w)
                        if m_result.status == "failed":
                            st.error("Still can't import — fix the required-field mapping above.")
                        else:
                            import_id = db.create_import(
                                brand_id=brand_id, platform=m_result.platform_id, level=m_result.level,
                                filename=f.name, date_start=m_result.date_start, date_end=m_result.date_end,
                                row_count=m_result.row_count, status=m_result.status,
                                unmapped_columns=m_result.unmapped_columns,
                                notes="manual mapping: " + "; ".join(m_result.warnings),
                            )
                            db.insert_rows(import_id, brand_id, m_result.rows)
                            st.success(f"Imported {m_result.row_count} rows as '{platform_label}'.")
                            st.rerun()
                continue

            overlaps = db.find_overlapping_import(brand_id, result.platform_id, result.date_start, result.date_end)
            if overlaps:
                st.warning(
                    f"This date range overlaps {len(overlaps)} existing import(s) for "
                    f"{result.platform_label} already on file. Importing again will add "
                    f"duplicate rows unless you delete the old import first (Settings tab)."
                )

            if st.button(f"Confirm import — {f.name}", key=f"import_{f.name}"):
                import_id = db.create_import(
                    brand_id=brand_id, platform=result.platform_id, level=result.level,
                    filename=f.name, date_start=result.date_start, date_end=result.date_end,
                    row_count=result.row_count, status=result.status,
                    unmapped_columns=result.unmapped_columns,
                    notes="; ".join(result.warnings),
                )
                db.insert_rows(import_id, brand_id, result.rows)
                st.success(f"Imported {result.row_count} rows.")
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

# ------------------------------------------------------------- settings ---

with tab_export:
    st.subheader(f"Settings & export — {selected_name}")

    with st.form("edit_brand"):
        st.write("**Targets & unit-economics assumptions**")
        col1, col2 = st.columns(2)
        with col1:
            margin_pct = st.number_input("Contribution margin (%)", 0.0, 100.0, (brand["margin_pct"] or 0) * 100) / 100
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
