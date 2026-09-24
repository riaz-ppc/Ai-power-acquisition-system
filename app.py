"""
PPC Acquisition & Decision Intelligence — prototype.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import zipfile
from datetime import date, datetime, timedelta
from io import BytesIO, StringIO

import altair as alt
import anthropic
import pandas as pd
import streamlit as st

from src import ai_analyst, db, mapping, metrics, normalize, orders, report
from src.digest import (
    _format_decomposition, brand_dict, compute_digest_items, money, pct, ratio,
)

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


def _read_upload_to_frames(f) -> list[tuple[str, pd.DataFrame]]:
    """
    Turns one uploaded file into a list of (label, raw_dataframe) pairs —
    almost always one, but can be several: a CSV that stacks multiple
    tables (see normalize.split_multi_table_csv), an Excel workbook with
    more than one sheet (a Google Sheet with separate Google/Bing tabs
    exported as .xlsx would otherwise silently only read the first one),
    or a .zip containing one or more CSVs — how a real Microsoft/Bing Ads
    export actually arrived in practice, not a hypothetical.
    """
    name_lower = f.name.lower()

    if name_lower.endswith(".zip"):
        frames: list[tuple[str, pd.DataFrame]] = []
        with zipfile.ZipFile(BytesIO(f.read())) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv") and "__MACOSX" not in n]
            if not csv_names:
                raise ValueError("This zip doesn't contain any .csv files.")
            for entry_name in csv_names:
                raw_text = zf.read(entry_name).decode("utf-8-sig", errors="replace")
                blocks = normalize.split_multi_table_csv(raw_text)
                for i, b in enumerate(blocks):
                    label = f"{f.name}/{entry_name}" + (f" — table {i+1}" if len(blocks) > 1 else "")
                    frames.append((label, pd.read_csv(StringIO(b))))
        return frames

    if name_lower.endswith(".csv"):
        raw_text = f.read().decode("utf-8-sig")
        blocks = normalize.split_multi_table_csv(raw_text)
        return [
            (f"{f.name} — table {i+1}" if len(blocks) > 1 else f.name, pd.read_csv(StringIO(b)))
            for i, b in enumerate(blocks)
        ]

    # .xlsx / .xls — read every sheet, not just the default first one, and
    # strip any title/subtitle rows the same way a stacked CSV needs to.
    raw_sheets = pd.read_excel(f, sheet_name=None, header=None)
    frames = []
    multi_sheet = len(raw_sheets) > 1
    for sheet_name, raw_df in raw_sheets.items():
        cleaned = normalize.strip_title_rows_df(raw_df)
        if cleaned is None or cleaned.empty:
            continue
        label = f"{f.name} — {sheet_name}" if multi_sheet else f.name
        frames.append((label, cleaned))
    if not frames:
        raise ValueError("No usable sheet found in this workbook.")
    return frames


st.set_page_config(page_title="PPC Acquisition Intelligence", layout="wide")

# Theme: colors live in .streamlit/config.toml (Streamlit's own [theme]
# section only covers base/primary/background/text colors + a generic
# font keyword — not real typefaces or component-level styling). This
# fills the rest in: the same Manrope/IBM Plex Sans pairing already used
# for the shareable client report (src/report_html.py), so the two
# surfaces feel like one product, plus card-style KPI tiles, a colored
# tab underline, and softened corners on buttons/alerts/dataframes.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Manrope', sans-serif; font-weight: 800; letter-spacing: -0.01em; }

/* KPI tiles: st.metric as a small elevated card instead of bare text */
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.15);
}
[data-testid="stMetricLabel"] { font-weight: 500; opacity: 0.75; }

/* Tabs: a colored active underline instead of the default thin grey one */
[data-testid="stTabs"] button[role="tab"] { font-weight: 600; padding: 0.5rem 1rem; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #FF6B4A;
    border-bottom: 3px solid #FF6B4A;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: #FF6B4A; }

/* Buttons: accent fill on primary actions, a gentle lift on hover */
.stButton > button, .stDownloadButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: transform 0.05s ease-in-out;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }
button[kind="primary"] { background-color: #FF6B4A; border-color: #FF6B4A; }
button[kind="primary"]:hover { background-color: #ff7f61; border-color: #ff7f61; }

/* Softer corners on containers, alerts, and dataframes so the page reads
   as one consistent surface rather than a stack of default widgets */
[data-testid="stExpander"], [data-testid="stAlert"], [data-testid="stDataFrame"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px;
}

/* Sidebar: a faint divider so it reads as a distinct panel, not a grey slab */
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)


# Query param (not a cookie) that carries the login token across a page
# reload — see _check_password for why.
AUTH_QUERY_PARAM = "ppc_auth"


def _auth_token(password: str) -> str:
    # Keyed by the real password so the token is only ever valid for
    # whatever APP_PASSWORD currently is — the URL carries this derived
    # token, never the password itself, and rotating APP_PASSWORD
    # invalidates every outstanding link automatically.
    return hmac.new(password.encode(), b"ppc-intelligence-auth-v1", hashlib.sha256).hexdigest()


def _check_password() -> bool:
    """Gate the whole app behind a single shared password, set via the
    APP_PASSWORD environment variable on the deployment. Local dev with no
    APP_PASSWORD set stays open, so this never gets in the way of running
    it on your own machine.

    Login persists across a page reload via a token in the URL's query
    string (st.query_params), not a cookie. An earlier version used a
    signed cookie via a third-party component (streamlit_cookies_manager)
    whose cookie-read is async — mounts, then reports "ready" on a later
    rerun — and on Streamlit Community Cloud that handshake was landing
    sessions in the authenticated branch without ever actually matching a
    real cookie: a silent full bypass of the password gate. Query params
    are part of the very first request, available synchronously with no
    "is it ready yet?" window, so that failure mode can't recur here.

    Trade-off, deliberately accepted: the token is a bearer credential
    embedded in the URL — anyone who gets this exact URL (browser
    history, a pasted link, a screenshot) is in without ever knowing the
    password, until APP_PASSWORD is rotated. Don't share/paste this app's
    URL while logged in if that's a concern."""
    required = os.environ.get("APP_PASSWORD")
    if not required:
        return True
    if st.session_state.get("authenticated"):
        return True

    if st.query_params.get(AUTH_QUERY_PARAM) == _auth_token(required):
        st.session_state["authenticated"] = True
        return True

    st.title("PPC Intelligence")
    pw = st.text_input("Password", type="password", key="pw_input")
    if pw:
        if pw == required:
            st.session_state["authenticated"] = True
            st.query_params[AUTH_QUERY_PARAM] = _auth_token(required)
            return True
        else:
            st.error("Incorrect password.")
    return False


def _log_out():
    st.session_state["authenticated"] = False
    if AUTH_QUERY_PARAM in st.query_params:
        del st.query_params[AUTH_QUERY_PARAM]
    # Deliberately no st.rerun() here: removing the query param is itself
    # a message that still needs to reach the browser and update the
    # address bar — an immediate rerun races that message (same failure
    # shape as the old cookie-write timing bug). Rendering a message and
    # st.stop()-ing in THIS run gives the removal a chance to actually
    # land; the next reload then genuinely finds no token in the URL.
    st.info("Logged out. Reload the page to sign in again.")
    st.stop()


if not _check_password():
    st.stop()

db.init_db()

# ------------------------------------------------------------------ cache --
# Streamlit reruns this whole script on every widget interaction, and every
# tab body below runs on every rerun regardless of which tab is visible —
# so an uncached db.rows_for_brand(brand_id) was hitting Postgres with a
# brand-new connection 6-8 times (once per tab that needs it) for the exact
# same rows on every single click anywhere in the app. Caching these reads
# turns that into one real query per rerun (the rest are cache hits) and
# zero queries on reruns where nothing changed. TTL is just a safety net —
# every write below clears the cache immediately so nothing goes stale.
_CACHE_TTL = 300


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def cached_rows_for_brand(brand_id: int, start: str | None = None, end: str | None = None):
    return db.rows_for_brand(brand_id, start, end)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def cached_orders_for_brand(brand_id: int, start: str | None = None, end: str | None = None):
    return db.orders_for_brand(brand_id, start, end)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def cached_list_brands():
    return db.list_brands()


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def cached_list_tests(brand_id: int):
    return db.list_tests(brand_id)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def cached_unmatched_raw_campaigns(brand_id: int):
    return db.unmatched_raw_campaigns(brand_id)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def cached_list_imports(brand_id: int):
    return db.list_imports(brand_id)


def invalidate_brand_data_cache():
    """Call right after any write to performance_rows/orders/imports/tests
    so the very next read (typically the st.rerun() a few lines later)
    reflects it, instead of waiting out the TTL."""
    cached_rows_for_brand.clear()
    cached_orders_for_brand.clear()
    cached_list_tests.clear()
    cached_unmatched_raw_campaigns.clear()
    cached_list_imports.clear()


def invalidate_brands_cache():
    """Call after create_brand/update_brand — brands list/order/fields change."""
    cached_list_brands.clear()


CURRENCIES = ["USD", "GBP", "BDT", "EUR"]
# Markets with a stocked seasonal-calendar entry in metrics.market_seasons();
# "Other" is still a valid brand location, it just won't get seasonal context yet.
COUNTRIES = ["Bangladesh", "United Kingdom", "Other"]
# Currency alone identifies the market for every currency actually in use here
# (BDT is Bangladesh-only business, GBP is UK-only business) — no separate
# location field needed at brand-creation time. USD/EUR are genuinely
# ambiguous across countries, so they're left unset; set the market manually
# in Settings for a brand using one of those if seasonal context is wanted.
CURRENCY_TO_COUNTRY = {"BDT": "Bangladesh", "GBP": "United Kingdom"}


# --------------------------------------------------------------- helpers --

def md_safe(s: str) -> str:
    """Streamlit renders text between a pair of '$' as inline LaTeX math —
    any insight mentioning two dollar amounts (e.g. USD brands) would
    otherwise get silently mangled in st.write/st.caption. Escape before
    any insight text reaches markdown-rendering calls."""
    return s.replace("$", "\\$")


# --------------------------------------------------------------- sidebar --

st.sidebar.title("PPC Intelligence")
if os.environ.get("APP_PASSWORD") and st.sidebar.button("Log out"):
    _log_out()
brands = cached_list_brands()
brand_names = {b["name"]: b["id"] for b in brands}

with st.sidebar.expander("❓ How this app works", expanded=(len(brands) == 0)):
    st.markdown(
        "**1. Create a brand** below — one per business/account you advertise for.\n\n"
        "**2. Import** — drop the CSV/Excel export from Meta, Google, or Microsoft Ads for that brand. "
        "The platform, report level, and columns are all detected automatically.\n\n"
        "**3. Check Digest first** — one ranked list of the most important thing(s) to act on this week, "
        "pulled from everything below it. If you manage more than one brand, turn on **Portfolio view** "
        "just below to see every brand's top issue at once, without switching brands one at a time.\n\n"
        "**4. Dig deeper as needed** — Dashboard for the numbers, Insights for why something moved and "
        "what's worth scaling or cutting, Reconciliation to check real revenue against what the platform "
        "reports, A/B Tests to confirm a change actually worked, Settings to adjust targets or export data.\n\n"
        "Switch which brand you're looking at with the **Brand** dropdown below at any time."
    )

with st.sidebar.expander("+ New brand", expanded=(len(brands) == 0)):
    with st.form("new_brand"):
        name = st.text_input("Brand name")
        business_model = st.selectbox("Business model", ["transactional", "lead_gen"],
                                       help="transactional = e-commerce (ROAS/AOV); lead_gen = leads/enrollments (CPA)")
        conversion_type = st.text_input("Conversion type label", value="purchase" if business_model == "transactional" else "lead",
                                         help="e.g. purchase, lead, enrollment")
        currency = st.selectbox("Primary currency", CURRENCIES,
                                 help="Also identifies this brand's market (BDT → Bangladesh, GBP → United "
                                      "Kingdom) to surface known seasonal demand shifts — Eid, back-to-school, "
                                      "Black Friday, etc. — in the Insights tab. For USD/EUR, set the market "
                                      "manually in Settings after creating the brand, if wanted.")
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
                    currency=currency, country=CURRENCY_TO_COUNTRY.get(currency),
                    margin_pct=margin_pct or None, aov=aov or None,
                    ltv=ltv or None, target_roas=target_roas, target_cpa=target_cpa,
                    target_payback_days=target_payback_days or None,
                )
                invalidate_brands_cache()
                st.success(f"Created {name}")
                st.rerun()

if not brands:
    st.info("Create a brand in the sidebar to get started.")
    st.stop()

if len(brands) > 1 and st.sidebar.checkbox("🗞️ Portfolio view (all brands)"):
    st.title("Portfolio digest — all brands")
    st.caption("The same ranked signals as each brand's own Digest tab, one row per brand, so you can scan "
               "the whole portfolio without switching brands one at a time. Click into a brand's own Digest "
               "tab for the full ranked list — this shows only the single highest-priority signal per brand.")
    badge_rank = {"🔴": 0, "🟡": 1, "🟢": 2, "": 3}
    portfolio_rows = []
    for b in brands:
        b_rows = cached_rows_for_brand(b["id"])
        if not b_rows:
            portfolio_rows.append({"_sort": 3, "Brand": b["name"], "Status": "No data yet", "Top signal": "—"})
            continue
        b_df = metrics.rows_to_df(b_rows)
        b_items = compute_digest_items(b_df, brand_dict(b))
        if not b_items:
            portfolio_rows.append({"_sort": 3, "Brand": b["name"], "Status": "🟢 Nothing urgent", "Top signal": "—"})
        else:
            top = b_items[0]
            portfolio_rows.append({"_sort": badge_rank.get(top["badge"], 3), "Brand": b["name"],
                                    "Status": f"{top['badge']} {top['title']}", "Top signal": top["detail"]})
    portfolio_rows.sort(key=lambda r: r["_sort"])
    for r in portfolio_rows:
        with st.container(border=True):
            st.markdown(f"**{r['Brand']}** — {md_safe(r['Status'])}")
            if r["Top signal"] != "—":
                st.caption(md_safe(r["Top signal"]))
    st.stop()

selected_name = st.sidebar.selectbox("Brand", list(brand_names.keys()))
brand_id = brand_names[selected_name]
brand = brand_dict(db.get_brand(brand_id))

st.sidebar.caption(
    f"{brand['business_model'].replace('_',' ').title()} · {brand['currency']} · "
    f"conversion = {brand['conversion_type']}"
)

tab_import, tab_digest, tab_ai, tab_dash, tab_insights, tab_tests, tab_recon, tab_export = st.tabs(
    ["📥 Import", "🗞️ Digest", "🤖 AI Analyst", "📊 Dashboard", "🧭 Insights", "🧪 A/B Tests",
     "💷 Reconciliation", "⚙️ Settings & Export"]
)

# ---------------------------------------------------------------- import --

with tab_import:
    st.subheader(f"Import PPC reports — {selected_name}")
    st.caption("Meta Ads Manager, Google Ads, or Microsoft Ads exports — CSV, XLSX (every sheet is checked, "
               "not just the first), or a .zip containing CSVs. Platform, report level, and stacked/"
               "multi-table files are all auto-detected from the column headers.")

    uploaded = st.file_uploader("Drop export files", type=["csv", "xlsx", "xls", "zip"], accept_multiple_files=True)

    if uploaded:
        for f in uploaded:
            try:
                sub_frames = _read_upload_to_frames(f)
            except Exception as e:
                st.markdown(f"---\n**{f.name}**")
                st.error(f"Couldn't read this file at all: {e}")
                continue

            for sub_name, raw_df in sub_frames:
                key = sub_name  # unique per file+table, used to namespace widget keys below
                st.markdown(f"---\n**{sub_name}**")

                result = normalize.normalize_upload(raw_df, sub_name, brand["currency"])

                # st.metric clips long text (a platform name, "campaign") in a
                # narrow column — it's built for numbers. Plain text for the
                # labels, a real metric only for the one number here.
                info_col, rows_col = st.columns([3, 1])
                with info_col:
                    st.markdown(
                        f"**Detected platform:** {result.platform_label}  \n"
                        f"**Report level:** {result.level}  \n"
                        f"**Date range:** {result.date_start or '—'} → {result.date_end or '—'}"
                    )
                with rows_col:
                    st.metric("Rows parsed", result.row_count)

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
                                for tip in normalize.data_completeness_suggestions(
                                    m_result.rows, m_result.level, m_result.platform_id, brand["business_model"]
                                ):
                                    st.info(f"💡 {tip}")
                                import_id = db.create_import(
                                    brand_id=brand_id, platform=m_result.platform_id, level=m_result.level,
                                    filename=sub_name, date_start=m_result.date_start, date_end=m_result.date_end,
                                    row_count=m_result.row_count, status=m_result.status,
                                    unmapped_columns=m_result.unmapped_columns,
                                    notes="manual mapping: " + "; ".join(m_result.warnings),
                                )
                                db.insert_rows(import_id, brand_id, m_result.rows)
                                invalidate_brand_data_cache()
                                st.success(f"Imported {m_result.row_count} rows as '{platform_label}'.")
                                st.rerun()
                    continue

                completeness_tips = normalize.data_completeness_suggestions(
                    result.rows, result.level, result.platform_id, brand["business_model"]
                )
                if completeness_tips:
                    with st.expander(f"💡 {len(completeness_tips)} tip(s) for a more complete report"):
                        for tip in completeness_tips:
                            st.caption(f"• {tip}")

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

                overlaps = []
                if not confirm_disabled:
                    overlaps = db.find_overlapping_import(
                        brand_id, result.platform_id, date_start_for_import, date_end_for_import
                    )

                def _do_import(replace: bool):
                    if replace:
                        db.delete_rows_in_range(brand_id, result.platform_id, date_start_for_import, date_end_for_import)
                    import_id = db.create_import(
                        brand_id=brand_id, platform=result.platform_id, level=result.level,
                        filename=sub_name, date_start=date_start_for_import, date_end=date_end_for_import,
                        row_count=len(rows_to_import), status=result.status,
                        unmapped_columns=result.unmapped_columns,
                        notes="; ".join(result.warnings),
                    )
                    db.insert_rows(import_id, brand_id, rows_to_import)
                    invalidate_brand_data_cache()
                    st.success(f"Imported {len(rows_to_import)} rows"
                               + (" (old rows in this date range were replaced)." if replace else "."))
                    st.rerun()

                if overlaps:
                    st.warning(
                        f"This date range overlaps {len(overlaps)} existing import(s) for "
                        f"{result.platform_label} already on file — same platform, same dates. "
                        f"Choose how to handle it:"
                    )
                    oc1, oc2 = st.columns(2)
                    with oc1:
                        if st.button(f"Replace overlapping data & import", key=f"replace_{key}",
                                     disabled=confirm_disabled, type="primary"):
                            _do_import(replace=True)
                    with oc2:
                        if st.button(f"Add anyway (creates duplicates)", key=f"dupe_{key}",
                                     disabled=confirm_disabled):
                            _do_import(replace=False)
                else:
                    if st.button(f"Confirm import — {sub_name}", key=f"import_{key}", disabled=confirm_disabled):
                        _do_import(replace=False)

    st.markdown("---")
    st.caption("**Not built yet:** automatic API sync (Meta/Google/Microsoft Marketing APIs) — "
               "a later phase, needs real hosting + OAuth app registration.")

# ----------------------------------------------------------------- digest --

with tab_digest:
    st.subheader(f"Weekly digest — {selected_name}")
    all_rows = cached_rows_for_brand(brand_id)
    if not all_rows:
        st.info("No data yet — import a report first.")
    else:
        df = metrics.rows_to_df(all_rows)
        n = 7
        st.caption(f"Last {n} days vs. the {n} before that, synthesized from every other tab and ranked "
                   f"so the highest-priority thing is first — not a new analysis, a ranked reading of what's "
                   f"already computed elsewhere. Check the relevant tab for full detail on any item below.")

        items = compute_digest_items(df, brand, n)
        if not items:
            st.success("Nothing urgent this week — no threshold breaches, forecast warnings, reallocation "
                       "gaps, keyword waste, or anomalies detected.")
        else:
            shown = items[:5]
            for i, it in enumerate(shown, 1):
                with st.container(border=True):
                    st.markdown(f"{it['badge']} **{i}. {md_safe(it['title'])}**")
                    st.write(md_safe(it["detail"]))
            if len(items) > len(shown):
                st.caption(f"{len(items) - len(shown)} more signal(s) not shown here — "
                           f"see Insights and Reconciliation for full detail.")

# ------------------------------------------------------------- ai analyst --

with tab_ai:
    st.subheader(f"AI analyst — {selected_name}")
    st.caption("Ask anything about this brand's performance in plain English. Claude answers by running "
               "this app's own analyses — course view, insights, keyword waste, the digest — so every "
               "number it quotes comes from your imported data, not guesswork. It can read your data but "
               "can't change anything in your ad accounts.")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        st.info("To turn this on, add an **ANTHROPIC_API_KEY** — in Streamlit Cloud's app settings "
                "under Secrets (`ANTHROPIC_API_KEY = \"sk-ant-...\"`), or in `.env` locally. Get a key at "
                "console.anthropic.com. Each question typically costs a few pence in API usage.")
    else:
        ai_rows = cached_rows_for_brand(brand_id)
        if not ai_rows:
            st.info("No data yet — import a report first.")
        else:
            ai_df = metrics.rows_to_df(ai_rows)
            # Per-brand conversation: API history (tool calls included, needed for
            # follow-ups) and the display transcript are kept separately.
            hist_key, chat_key = f"ai_history_{brand_id}", f"ai_chat_{brand_id}"
            st.session_state.setdefault(hist_key, [])
            st.session_state.setdefault(chat_key, [])

            b1, b2 = st.columns([1, 1])
            want_briefing = b1.button("📝 Write this week's briefing", use_container_width=True)
            if b2.button("Clear conversation", use_container_width=True):
                st.session_state[hist_key], st.session_state[chat_key] = [], []
                st.rerun()

            if not st.session_state[chat_key]:
                st.caption("Try: *Which courses should I move budget between?* · *Why did ROAS change "
                           "vs last month?* · *What's wasting the most money right now?* · *Where would "
                           "an extra £500 do the most good?*")

            for msg in st.session_state[chat_key]:
                with st.chat_message(msg["role"]):
                    st.markdown(md_safe(msg["text"]))
                    if msg.get("meta"):
                        st.caption(msg["meta"])

            question = st.chat_input("Ask about campaigns, courses, budget, waste…", key=f"ai_input_{brand_id}")
            if want_briefing:
                question = ai_analyst.BRIEFING_QUESTION

            if question:
                shown_q = "Write this week's briefing" if want_briefing else question
                st.session_state[chat_key].append({"role": "user", "text": shown_q})
                with st.chat_message("user"):
                    st.markdown(md_safe(shown_q))
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Analysing your data…"):
                            reply = ai_analyst.ask(st.session_state[hist_key], question, ai_df, brand)
                        steps = ", ".join(dict.fromkeys(t.replace("get_", "").replace("_", " ")
                                                        for t in reply.tools_used))
                        meta = (f"Checked: {steps} · " if steps else "") + \
                               f"{reply.input_tokens + reply.output_tokens:,} tokens"
                        st.markdown(md_safe(reply.text))
                        st.caption(meta)
                        st.session_state[chat_key].append({"role": "assistant", "text": reply.text, "meta": meta})
                    except anthropic.AuthenticationError:
                        st.error("The Anthropic API key was rejected — check ANTHROPIC_API_KEY in your secrets.")
                    except anthropic.RateLimitError:
                        st.error("Hit the Anthropic API rate limit — wait a minute and try again.")
                    except anthropic.APIStatusError as e:
                        st.error(f"The Anthropic API returned an error ({e.status_code}) — try again shortly.")
                    except anthropic.APIConnectionError:
                        st.error("Couldn't reach the Anthropic API — check the network connection.")

# -------------------------------------------------------------- dashboard --

with tab_dash:
    st.subheader(f"Dashboard — {selected_name}")
    all_rows = cached_rows_for_brand(brand_id)
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

            # Six st.metric tiles in one st.columns(6) row clip both the label
            # ("Payback (orders to break even)") and long money/count values
            # ("$35,278.58", "1,574") at normal window widths — st.metric is
            # built for short numbers, not this much text in a sixth of the
            # row. A 3-wide x 2-row grid gives each tile double the room,
            # same fix already used for the Insights tab's period-over-
            # period row.
            krow1 = st.columns(3)
            krow2 = st.columns(3)
            kcols = [krow1[0], krow1[1], krow1[2], krow2[0], krow2[1], krow2[2]]
            kcols[0].metric("Spend", money(totals["spend"], brand["currency"]))
            kcols[1].metric("Conversions", f"{totals['conversions']:,.0f}")
            if brand["business_model"] == "transactional":
                kcols[2].metric("ROAS", ratio(econ["roas"]))
                kcols[3].metric("iROAS floor", ratio(econ["iroas_floor"]) if econ["iroas_floor"] else "—")
            else:
                kcols[2].metric("CAC / CPA", money(econ["cac"], brand["currency"]))
                kcols[3].metric("Target CPA", money(brand["target_cpa"], brand["currency"]))
            kcols[4].metric("LTV : CAC", ratio(econ["ltv_cac"]) if econ["ltv_cac"] else "—")
            payback_label = f"{econ['payback_orders']:.1f} orders" if econ["payback_orders"] else "—"
            kcols[5].metric("Payback (orders to break even)", payback_label,
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

            st.markdown("#### Compare by period")
            st.caption("Month-by-month (or week-by-week) view — a table you read top to bottom, "
                       "with the change vs. the prior period built in, rather than squinting at a chart.")
            period_choice = st.selectbox("Group by", ["Month", "Week"], key="dash_period_freq")
            period_df = metrics.aggregate_by_period(scoped, "M" if period_choice == "Month" else "W")
            if period_df is None or period_df.empty:
                st.caption("Not enough data in range to group.")
            else:
                metric_col_p = "roas" if brand["business_model"] == "transactional" else "cpa"
                display_cols = ["period", "spend", "conversions", metric_col_p,
                                 "spend_change_pct", "conversions_change_pct", f"{metric_col_p}_change_pct"]
                period_display = period_df[display_cols].rename(columns={
                    "period": period_choice, "spend": "Spend", "conversions": "Conversions",
                    metric_col_p: metric_col_p.upper(),
                    "spend_change_pct": "Spend Δ%", "conversions_change_pct": "Conversions Δ%",
                    f"{metric_col_p}_change_pct": f"{metric_col_p.upper()} Δ%",
                })
                fmt = {"Spend": "{:,.2f}", "Conversions": "{:,.0f}",
                       "Spend Δ%": "{:+.1f}%", "Conversions Δ%": "{:+.1f}%"}
                fmt[metric_col_p.upper()] = "{:.2f}x" if metric_col_p == "roas" else "{:,.2f}"
                fmt[f"{metric_col_p.upper()} Δ%"] = "{:+.1f}%"
                st.dataframe(period_display.style.format(fmt, na_rep="—"), use_container_width=True)

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
    all_rows = cached_rows_for_brand(brand_id)
    if not all_rows:
        st.info("No data yet — import a report first.")
    else:
        df = metrics.rows_to_df(all_rows)
        max_d = df["date"].max().date()
        window = st.selectbox("Compare current period vs. baseline",
                               ["Last 7 vs prior 7", "Last 30 vs prior 30", "Last 30 vs same 30 days last year"],
                               index=1)
        n = 7 if "7" in window else 30
        cur_start, cur_end = max_d - timedelta(days=n - 1), max_d
        if "last year" in window:
            base_start, base_end = cur_start - timedelta(days=365), cur_end - timedelta(days=365)
        else:
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
                if ins.campaign:
                    cur_totals = metrics.campaign_totals(df, ins.campaign, cur_start, cur_end)
                    base_totals = metrics.campaign_totals(df, ins.campaign, base_start, base_end)
                    decomp = metrics.decompose_metric_change(cur_totals, base_totals, brand["business_model"])
                    if decomp:
                        st.caption(md_safe(_format_decomposition(decomp, brand["business_model"], brand["currency"])))
                if ins.suggested_action:
                    st.markdown(f"**→ Suggested action:** {md_safe(ins.suggested_action)}")
                st.caption(f"metric: `{ins.metric}` · threshold: `{md_safe(ins.threshold)}` · formula: `{ins.formula}`")

        st.markdown("---")
        st.caption("Period-over-period change (current vs. baseline):")
        # Two rows of two, not four across — a long money string (e.g.
        # "$36,661.23") clips in a 4-wide st.metric column at normal screen
        # widths; this gives each tile roughly double the room.
        prow1 = st.columns(2)
        prow2 = st.columns(2)
        pcols = [prow1[0], prow1[1], prow2[0], prow2[1]]
        pcols[0].metric("Spend", money(compare["current"]["spend"], brand["currency"]), pct(compare["pct_change"]["spend"]))
        pcols[1].metric("Conversions", f"{compare['current']['conversions']:,.0f}", pct(compare["pct_change"]["conversions"]))
        pcols[2].metric("CPA", money(compare["current"]["cpa"], brand["currency"]), pct(compare["pct_change"]["cpa"]), delta_color="inverse")
        pcols[3].metric("ROAS", ratio(compare["current"]["roas"]), pct(compare["pct_change"]["roas"]))

        st.markdown("---")
        st.markdown("#### Trend forecast")
        st.caption("Where anomalies look backward, this looks forward: a straight-line projection of "
                   "the last 14 days' momentum, not a seasonality-aware forecast — flagged plainly "
                   "when the recent trend is too noisy for the line to mean much.")
        forecast_metric = "roas" if brand["business_model"] == "transactional" else "cpa"
        forecast_daily = metrics.aggregate(df, by=["date"])
        forecast = metrics.forecast_trend(forecast_daily, forecast_metric)
        if forecast is None:
            st.info("Not enough recent daily history to project a trend yet (need at least 5 days).")
        else:
            fmt_val = (lambda v: money(v, brand["currency"])) if forecast_metric == "cpa" else ratio
            hist_tail = forecast_daily.sort_values("date").dropna(subset=[forecast_metric]).tail(14)
            hist_chart_df = hist_tail[["date", forecast_metric]].rename(columns={forecast_metric: "value"})
            hist_chart_df["kind"] = "actual"
            fut_chart_df = pd.DataFrame(forecast["forecast"])
            fut_chart_df["date"] = pd.to_datetime(fut_chart_df["date"])
            fut_chart_df["kind"] = "projected"
            # connect the two lines visually at the last real point
            bridge = pd.DataFrame([{"date": hist_tail["date"].iloc[-1], "value": forecast["current_value"], "kind": "projected"}])
            combined = pd.concat([hist_chart_df, bridge, fut_chart_df], ignore_index=True)

            fc1, fc2 = st.columns([2, 1])
            with fc1:
                base = alt.Chart(combined).encode(x="date:T", y=alt.Y("value:Q", title=forecast_metric.upper()))
                actual_line = base.transform_filter("datum.kind == 'actual'").mark_line(point=True, color="#2a78d6")
                proj_line = base.transform_filter("datum.kind == 'projected'").mark_line(
                    point=True, strokeDash=[5, 4], color="#eb6834")
                st.altair_chart((actual_line + proj_line).properties(height=220), use_container_width=True)
            with fc2:
                st.metric(f"Projected {forecast_metric.upper()} in 7 days", fmt_val(forecast["projected_value_end"]),
                          pct(forecast["pct_change_projected"]), delta_color="inverse" if forecast_metric == "cpa" else "normal")
                fit_quality = "a fairly consistent trend" if forecast["r_squared"] >= 0.5 else "noisy — treat loosely"
                st.caption(f"Fit: R²={forecast['r_squared']} ({fit_quality}), based on last {forecast['lookback_points']} days.")

            target_val = brand["target_cpa"] if forecast_metric == "cpa" else brand["target_roas"]
            if target_val:
                breaches = (forecast["projected_value_end"] > target_val) if forecast_metric == "cpa" \
                    else (forecast["projected_value_end"] < target_val)
                if breaches:
                    st.warning(f"At this trend, projected {forecast_metric.upper()} in 7 days "
                               f"({fmt_val(forecast['projected_value_end'])}) would be past your target "
                               f"({fmt_val(target_val)}) — worth acting before it gets there, not after.")

        st.markdown("---")
        st.markdown("#### Market context")
        if not brand.get("country"):
            st.info("Add this brand's location/market in the Settings tab to unlock seasonal demand context here.")
        else:
            st.caption("A static calendar of known recurring demand periods for this brand's market — "
                       "not a live trends feed — cross-checked against this brand's own history for the "
                       "same window last year, where it has any.")
            seasons = metrics.market_seasons(brand["country"], brand["business_model"], date.today())
            if not seasons:
                st.success(f"No known seasonal demand shift for {brand['country']} in the surrounding 30 days.")
            else:
                season_metric = "roas" if brand["business_model"] == "transactional" else "cpa"
                for season in seasons:
                    with st.container(border=True):
                        icon = "📈" if season["direction"] == "up" else "📉"
                        active = season["start"] <= date.today() <= season["end"]
                        status = " · currently active" if active else (" · upcoming" if season["start"] > date.today() else " · recently ended")
                        st.markdown(f"{icon} **{season['name']}** — {season['start'].isoformat()} to {season['end'].isoformat()}{status}")
                        st.caption(season["note"])
                        last_year_start = season["start"].replace(year=season["start"].year - 1)
                        last_year_end = season["end"].replace(year=season["end"].year - 1)
                        this_year_end = min(season["end"], date.today())
                        if season["start"] <= date.today():
                            season_cmp = metrics.compare_periods(df, season["start"], this_year_end, last_year_start, last_year_end)
                            if season_cmp["baseline"]["spend"]:
                                cval, bval = season_cmp["current"][season_metric], season_cmp["baseline"][season_metric]
                                fmt = (lambda v: money(v, brand["currency"])) if season_metric == "cpa" else ratio
                                st.write(f"So far this window: {season_metric.upper()} "
                                         f"{fmt(cval) if cval is not None else '—'} vs. {fmt(bval) if bval is not None else '—'} "
                                         f"in the same window last year.")
                            else:
                                st.caption("No data from this brand for the same window last year yet — nothing to compare against.")
                        else:
                            last_year_cmp = metrics.compare_periods(df, last_year_start, last_year_end, last_year_start, last_year_end)
                            lval = last_year_cmp["current"][season_metric]
                            if last_year_cmp["current"]["spend"] and lval is not None:
                                fmt_val = money(lval, brand["currency"]) if season_metric == "cpa" else ratio(lval)
                                st.caption(f"Last year during this window, {season_metric.upper()} was {fmt_val} — a reference point, not a prediction.")

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

        st.markdown("---")
        st.markdown("#### Budget reallocation")
        st.caption("A campaign's average CPA/ROAS so far can look fine while it's already past the point "
                   "of diminishing returns — or look mediocre while still scaling well. Where a campaign's "
                   "own spend history varies enough to fit a response curve, this reads the MARGINAL cost/"
                   "value of the next pound/taka there, not just the average of what's already been spent. "
                   "Campaigns under 5% of this period's spend are skipped as too small to read reliably.")
        realloc_metric = "roas" if brand["business_model"] == "transactional" else "cpa"
        realloc = metrics.budget_reallocation_view(df, camp_agg, brand)
        if not realloc:
            st.info("No campaign in this period has enough spend share to rank.")
        else:
            fmt_realloc = (lambda v: money(v, brand["currency"])) if realloc_metric == "cpa" else ratio
            realloc_table = pd.DataFrame([{
                "Campaign": r["campaign"],
                "Spend": money(r["spend"], brand["currency"]),
                f"Avg {realloc_metric.upper()}": fmt_realloc(r["avg_metric"]) if r["avg_metric"] is not None else "—",
                f"Marginal {realloc_metric.upper()}": fmt_realloc(r["marginal_metric"]) if r["marginal_metric"] is not None else "—",
                "Basis": "marginal (curve fit)" if r["marginal_metric"] is not None else "average only",
                "Fit R²": r["r_squared"] if r["r_squared"] is not None else "—",
            } for r in realloc])
            st.dataframe(realloc_table, hide_index=True, use_container_width=True)

            best, worst = realloc[0], realloc[-1]
            if (best["ranking_metric"] is not None and worst["ranking_metric"] is not None
                    and best["campaign"] != worst["campaign"]):
                gap_pct = abs(worst["ranking_metric"] - best["ranking_metric"]) / abs(best["ranking_metric"]) * 100 \
                    if best["ranking_metric"] else 0
                if gap_pct >= 25:
                    test_amount = worst["spend"] * 0.15
                    st.warning(
                        f"**{best['campaign']}** looks like the more efficient place for the next pound/taka "
                        f"right now ({best['basis']}: {fmt_realloc(best['ranking_metric'])}) vs. "
                        f"**{worst['campaign']}** ({worst['basis']}: {fmt_realloc(worst['ranking_metric'])}) — "
                        f"a {gap_pct:.0f}% gap. Worth testing a shift of roughly "
                        f"{money(test_amount, brand['currency'])} (~15% of {worst['campaign']}'s spend this "
                        f"period) from {worst['campaign']} to {best['campaign']} and watching what happens — "
                        f"this is a suggested test size, not a guaranteed-optimal split."
                    )
                else:
                    st.success(f"No large efficiency gap between campaigns this period (best vs. worst "
                               f"differ by {gap_pct:.0f}%) — nothing worth disrupting budgets over yet.")

        cross_platform = metrics.cross_platform_reallocation_view(df, brand)
        if cross_platform:
            st.markdown("---")
            st.markdown("#### Cross-platform reallocation")
            st.caption("Same idea, one level up: not which campaign within a platform, but which "
                       "PLATFORM — Meta vs. Google/Microsoft — is the more efficient place for the "
                       "next pound/taka right now. Only shown once this brand actually spends on more "
                       "than one platform.")
            fmt_cp = (lambda v: money(v, brand["currency"])) if realloc_metric == "cpa" else ratio
            cp_table = pd.DataFrame([{
                "Platform": r["platform"],
                "Spend": money(r["spend"], brand["currency"]),
                f"Avg {realloc_metric.upper()}": fmt_cp(r["avg_metric"]) if r["avg_metric"] is not None else "—",
                f"Marginal {realloc_metric.upper()}": fmt_cp(r["marginal_metric"]) if r["marginal_metric"] is not None else "—",
                "Basis": "marginal (curve fit)" if r["marginal_metric"] is not None else "average only",
                "Fit R²": r["r_squared"] if r["r_squared"] is not None else "—",
            } for r in cross_platform])
            st.dataframe(cp_table, hide_index=True, use_container_width=True)

            cp_best, cp_worst = cross_platform[0], cross_platform[-1]
            if (cp_best["ranking_metric"] is not None and cp_worst["ranking_metric"] is not None
                    and cp_best["platform"] != cp_worst["platform"]):
                cp_gap_pct = abs(cp_worst["ranking_metric"] - cp_best["ranking_metric"]) / abs(cp_best["ranking_metric"]) * 100 \
                    if cp_best["ranking_metric"] else 0
                if cp_gap_pct >= 25:
                    cp_test_amount = cp_worst["spend"] * 0.15
                    st.warning(
                        f"**{cp_best['platform']}** looks like the more efficient platform for the next "
                        f"pound/taka right now ({cp_best['basis']}: {fmt_cp(cp_best['ranking_metric'])}) vs. "
                        f"**{cp_worst['platform']}** ({cp_worst['basis']}: {fmt_cp(cp_worst['ranking_metric'])}) — "
                        f"a {cp_gap_pct:.0f}% gap. Worth testing a shift of roughly "
                        f"{money(cp_test_amount, brand['currency'])} (~15% of {cp_worst['platform']}'s spend "
                        f"this period) toward {cp_best['platform']} and watching what happens — a suggested "
                        f"test size across platforms, not a guaranteed-optimal split."
                    )
                else:
                    st.success(f"No large efficiency gap between platforms this period (best vs. worst "
                               f"differ by {cp_gap_pct:.0f}%) — nothing worth shifting budget across "
                               f"platforms for yet.")

        course_scoped = df[(df["date"] >= pd.Timestamp(cur_start)) & (df["date"] <= pd.Timestamp(cur_end))]
        courses = metrics.course_platform_view(course_scoped, brand)
        if any(len(c["platforms"]) > 1 for c in courses):
            st.markdown("---")
            st.markdown("#### Courses across platforms")
            st.caption("Platform totals hide per-course differences — one course can do far better on "
                       "Bing than Google while the platforms look similar overall. Campaigns are grouped "
                       "into courses by name (ignoring notes in brackets, \"PMax\"/\"CAT\"/\"Training\", "
                       "version tags), so check the Campaigns column for anything grouped wrongly. "
                       "Break-even comes from this brand's margin/target in Settings — if a verdict "
                       "looks wrong, check that setting first.")
            verdict_label = {"shift": "🔀 Shift budget", "below_break_even": "🔴 Below break-even",
                             "expand": "🟢 Test on another platform", "healthy": "✅ Healthy",
                             "insufficient_data": "⚪ Too little spend"}
            for c in [c for c in courses if c["verdict"] in ("shift", "below_break_even", "expand")][:8]:
                st.markdown(f"**{verdict_label[c['verdict']]} — {md_safe(c['course'].title())}:** "
                            f"{md_safe(c['detail'])}")
            course_table = pd.DataFrame([{
                "Course": c["course"].title(),
                "Verdict": verdict_label[c["verdict"]],
                "Platform": p,
                "Spend": money(cell["spend"], brand["currency"]),
                "ROAS" if brand["business_model"] == "transactional" else "CPA":
                    metrics._fmt_efficiency(cell["efficiency"], brand["business_model"], brand["currency"]),
                "Campaigns": ", ".join(cell["campaigns"]),
            } for c in courses for p, cell in c["platforms"].items()])
            with st.expander(f"All {len(courses)} courses, by platform"):
                st.dataframe(course_table, hide_index=True, use_container_width=True)

        account_marginal = metrics.account_marginal_efficiency(df, brand["business_model"])
        if account_marginal:
            st.markdown("---")
            st.markdown("#### Account-wide budget capacity")
            st.caption("A different question from reallocation above: not which campaign or platform "
                       "should get the next pound, but whether growing TOTAL account spend still makes "
                       "sense at all — the same marginal-efficiency method, applied to the whole account "
                       "combined.")
            fmt_acct = (lambda v: money(v, brand["currency"])) if realloc_metric == "cpa" else ratio
            ac1, ac2 = st.columns(2)
            ac1.metric(f"Average {realloc_metric.upper()} (last {account_marginal['days_used']} days)",
                       fmt_acct(account_marginal["avg_metric"]))
            ac2.metric(f"Marginal {realloc_metric.upper()} (next pound/taka)",
                       fmt_acct(account_marginal["marginal_metric"]))
            fit_note = "still scaling well" if account_marginal["elasticity"] >= 1 else "diminishing returns"
            st.caption(f"Elasticity: {account_marginal['elasticity']} ({fit_note}), "
                       f"fit R²={account_marginal['r_squared']}.")

            acct_target = brand["target_cpa"] if realloc_metric == "cpa" else brand["target_roas"]
            if acct_target:
                marginal_ok = (account_marginal["marginal_metric"] <= acct_target) if realloc_metric == "cpa" \
                    else (account_marginal["marginal_metric"] >= acct_target)
                if marginal_ok:
                    st.success(f"There's still room to grow total budget — the next pound/taka account-wide "
                               f"would land at roughly {fmt_acct(account_marginal['marginal_metric'])}, still "
                               f"within your {fmt_acct(acct_target)} target.")
                else:
                    st.warning(f"Total account spend looks past the point of diminishing returns — the next "
                               f"pound/taka account-wide would land at roughly "
                               f"{fmt_acct(account_marginal['marginal_metric'])}, already past your "
                               f"{fmt_acct(acct_target)} target. Growth from here is more likely to come from "
                               f"reallocating toward the efficient campaigns/platforms already identified "
                               f"above than from simply raising the total budget.")

        kw_scoped = df[(df["date"] >= pd.Timestamp(cur_start)) & (df["date"] <= pd.Timestamp(cur_end))]
        if (kw_scoped["platform"].isin(["google", "microsoft"]) & kw_scoped["keyword"].notna()).any():
            st.markdown("---")
            st.markdown("#### Keyword waste")
            st.caption("Classic PPC hygiene: a keyword burning real clicks with zero conversions is usually "
                       "the single highest-ROI thing to fix in a search account. Grounded with the "
                       "\"rule of three\" — for zero conversions in n clicks, the upper bound of a 95% "
                       "confidence interval on the true conversion rate is ~3/n — compared against this "
                       "account's own blended conversion rate, not a flat industry rule of thumb. Only "
                       "Google/Microsoft carry keyword-level data.")
            waste = metrics.keyword_waste_candidates(kw_scoped)
            if not waste:
                st.success("No keywords with enough clicks and zero conversions to flag this period.")
            else:
                baseline = waste[0]["account_cvr_pct"]
                if baseline is not None:
                    st.caption(f"This account's blended conversion rate (from converting keywords this "
                               f"period): {baseline:.2f}%.")
                waste_table = pd.DataFrame([{
                    "Platform": w["platform"], "Campaign": w["campaign"], "Keyword": w["keyword"],
                    "Spend": money(w["spend"], brand["currency"]), "Clicks": w["clicks"],
                    "Best-case CVR (95% upper bound)": f"{w['upper_bound_cvr_pct']:.2f}%",
                } for w in waste])
                st.dataframe(waste_table, hide_index=True, use_container_width=True)
                total_waste_spend = sum(w["spend"] for w in waste)
                st.warning(f"{len(waste)} keyword(s), {money(total_waste_spend, brand['currency'])} of spend "
                           f"this period, with a 95%-confidence best case still below this account's own "
                           f"typical conversion rate — worth pausing or restructuring (tighter match type, "
                           f"a negative keyword) before spending more on them as-is.")

            with st.expander("Recurring word patterns (catches what single-keyword waste can't)"):
                st.caption("A word that shows up across several keyword phrases — each too thin on its own "
                           "to clear the bar above — can still be real waste in aggregate. This is a "
                           "search-term n-gram report's idea (a word like \"free\" across a dozen "
                           "zero-converting queries) applied to bid keyword text instead, since this app "
                           "doesn't import search-term reports — a weaker signal than real queries, but "
                           "still worth a negative-keyword review.")
                word_waste = metrics.keyword_word_waste(kw_scoped)
                if not word_waste:
                    st.success("No recurring words with enough combined clicks and zero conversions this period.")
                else:
                    word_table = pd.DataFrame([{
                        "Word": w["word"], "Appears in": f"{w['keyword_count']} keywords",
                        "Combined spend": money(w["spend"], brand["currency"]), "Combined clicks": w["clicks"],
                        "Best-case CVR (95% upper bound)": f"{w['upper_bound_cvr_pct']:.2f}%",
                    } for w in word_waste])
                    st.dataframe(word_table, hide_index=True, use_container_width=True)

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
                invalidate_brand_data_cache()
                st.success("Test saved.")
                st.rerun()

    tests = cached_list_tests(brand_id)
    all_rows = cached_rows_for_brand(brand_id)
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
            st.markdown(f"**Date range:** {o_result.date_start or '—'} → {o_result.date_end or '—'}")
            c1, c3 = st.columns(2)
            c1.metric("Rows parsed", o_result.row_count)
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
                invalidate_brand_data_cache()
                st.success(f"Imported {o_result.row_count} orders.")
                st.rerun()

    all_orders_unfiltered = cached_orders_for_brand(brand_id)
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
            all_orders = cached_orders_for_brand(brand_id, recon_start, recon_end)
            perf_rows = cached_rows_for_brand(brand_id, recon_start, recon_end)

            st.markdown("---")
            st.markdown("#### Match campaign names")
            st.caption("Order sheets use shorthand campaign names that rarely match the platform's exact "
                       "names exactly. Every suggestion below is a guess — confirm or correct each one; "
                       "nothing is used for reconciliation until you save.")

            unmatched = cached_unmatched_raw_campaigns(brand_id)
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
                        invalidate_brand_data_cache()
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
                camp_perf = metrics.aggregate(perf_df, by=["campaign"]).set_index("campaign")
                claimed_totals = camp_perf["conversion_value"].rename("platform_claimed")
                spend_totals = camp_perf["spend"].rename("spend")

                recon = pd.concat([order_totals, claimed_totals, spend_totals], axis=1).fillna(0.0).reset_index()
                recon.columns = ["campaign", "actual_revenue", "platform_claimed", "spend"]
                recon["delta"] = recon["actual_revenue"] - recon["platform_claimed"]
                recon["delta_pct"] = (recon["delta"] / recon["platform_claimed"].replace(0, pd.NA)) * 100
                # The ad platform's own "conversion value" is often just genuinely absent for
                # lead-gen accounts (Google/Bing don't assign a monetary value to a lead unless
                # that's separately configured) — spend is real regardless, so merging your own
                # order revenue against it gives a true ROI even when platform_claimed is 0,
                # which a lead-gen brand's CPA-based insights elsewhere in this app don't compute.
                recon["true_roas"] = (recon["actual_revenue"] / recon["spend"]).where(recon["spend"] > 0)
                recon["platform_roas"] = (recon["platform_claimed"] / recon["spend"]).where(recon["spend"] > 0)
                recon = recon.sort_values("actual_revenue", ascending=False)

                # Same truncation risk as the Dashboard KPI row this replaces the pattern of: 4
                # narrow st.metric columns clip a long label like "True ROI (actual revenue ÷
                # spend)" and a long money value like "$35,278.58" alike. 2x2 gives each tile
                # double the room.
                trow1 = st.columns(2)
                trow2 = st.columns(2)
                t1, t2, t3, t4 = trow1[0], trow1[1], trow2[0], trow2[1]
                t1.metric("Actual revenue (orders)", money(recon["actual_revenue"].sum(), brand["currency"]))
                t2.metric("Platform-claimed revenue", money(recon["platform_claimed"].sum(), brand["currency"]))
                total_delta_pct = (recon["actual_revenue"].sum() - recon["platform_claimed"].sum()) / recon["platform_claimed"].sum() * 100 if recon["platform_claimed"].sum() else None
                t3.metric("Revenue difference", pct(total_delta_pct))
                total_spend = recon["spend"].sum()
                t4.metric("True ROI (actual revenue ÷ spend)",
                          ratio(recon["actual_revenue"].sum() / total_spend) if total_spend else "—",
                          help="Your own order revenue against real spend — works even for lead-gen "
                               "brands, where the platform never tracked a conversion value at all.")

                st.dataframe(recon.style.format({
                    "actual_revenue": "{:,.2f}", "platform_claimed": "{:,.2f}", "spend": "{:,.2f}",
                    "delta": "{:,.2f}", "delta_pct": "{:+.1f}%",
                    "true_roas": lambda v: ratio(v), "platform_roas": lambda v: ratio(v),
                }), use_container_width=True)
                st.caption("delta/delta_pct: positive = platforms under-claimed vs. real revenue, negative = "
                           "platforms over-claimed (common with pixel-based attribution). true_roas: this "
                           "campaign's actual order revenue ÷ its spend — the honest ROI figure regardless of "
                           "whether the platform tracks conversion value at all. platform_roas: what the "
                           "platform's own claimed conversion value would imply, for comparison.")

                st.markdown("---")
                st.caption("The button below merges conversion_value = actual_revenue directly into this "
                           "brand's stored performance data for matched campaigns in this date range — but "
                           "ONLY where the platform's own conversion_value is currently 0/blank, so a "
                           "platform that genuinely tracks revenue (real pixel-based ROAS) is never silently "
                           "overwritten by potentially-incomplete order data. Once applied, ROAS-based "
                           "insights, root-cause decomposition, and budget reallocation elsewhere in this app "
                           "see your real revenue instead of a platform-reported zero. Re-importing the "
                           "platform file later resets conversion_value back to whatever the platform reports.")
                campaign_to_platform = {c: p for p, cs in campaigns_by_platform.items() for c in cs}
                if st.button("Merge conversion_value = actual_revenue"):
                    updates = []
                    skipped = []
                    for campaign in recon["campaign"]:
                        platform = campaign_to_platform.get(campaign)
                        if not platform:
                            skipped.append(campaign)
                            continue
                        camp_orders = odf[odf["matched_campaign"] == campaign]
                        amount_by_date = camp_orders.groupby("order_date")["amount"].sum().to_dict()
                        updates.extend(
                            (platform, campaign, order_date, amount)
                            for order_date, amount in amount_by_date.items()
                        )
                    total_updated = db.apply_actual_revenue(brand_id, updates)
                    msg = (f"Updated {total_updated} row(s) — conversion_value now reflects real order "
                           f"revenue wherever the platform reported none.")
                    if skipped:
                        msg += f" Skipped (couldn't tell which platform): {', '.join(skipped)}."
                    invalidate_brand_data_cache()
                    st.success(msg)
                    st.rerun()

# ------------------------------------------------------------- settings ---

with tab_export:
    st.subheader(f"Settings & export — {selected_name}")

    with st.form("edit_brand"):
        st.write("**Targets & unit-economics assumptions**")
        country_options = COUNTRIES if brand.get("country") in COUNTRIES else [brand.get("country") or "Other"] + COUNTRIES
        country = st.selectbox("Location / primary market", country_options,
                                index=country_options.index(brand.get("country")) if brand.get("country") in country_options else 0,
                                help="Used to surface known seasonal demand shifts for this market in the Insights tab.")
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
            db.update_brand(brand_id, country=country, margin_pct=margin_pct or None, aov=aov or None, ltv=ltv or None,
                             target_roas=target_roas or None, target_cpa=target_cpa or None,
                             target_payback_days=target_payback_days or None)
            invalidate_brands_cache()
            st.success("Saved.")
            st.rerun()

    st.markdown("---")
    st.write("**Imports on file**")
    imports = cached_list_imports(brand_id)
    if imports:
        idf = pd.DataFrame([dict(i) for i in imports])[
            ["id", "platform", "level", "filename", "date_start", "date_end", "row_count", "status", "imported_at"]
        ]
        st.dataframe(idf, use_container_width=True)
        del_id = st.number_input("Import ID to delete", min_value=0, value=0, step=1)
        if st.button("Delete import") and del_id:
            if db.delete_import(int(del_id), brand_id):
                invalidate_brand_data_cache()
                st.success(f"Deleted import {del_id}.")
                st.rerun()
            else:
                st.error(f"Import {del_id} doesn't belong to {selected_name} — nothing deleted.")
    else:
        st.caption("No imports yet.")

    st.markdown("---")
    st.write("**Possible campaign renames**")
    st.caption("A campaign renamed in the ad platform (a note added, a typo fixed) looks like a brand-new "
               "campaign to this app unless the two are linked — that breaks trend/forecast/root-cause "
               "history right at the rename. This checks for near-identical names whose active dates don't "
               "overlap (a real rename: the old name stops right around when the new one starts) and "
               "suggests merging them. Always a suggestion you confirm — nothing merges on its own.")
    rename_check_rows = cached_rows_for_brand(brand_id)
    if not rename_check_rows:
        st.caption("No data yet.")
    else:
        rename_df = metrics.rows_to_df(rename_check_rows)
        renames = normalize.detect_campaign_renames(rename_df)
        if not renames:
            st.success("No likely renames detected in this brand's campaign history.")
        else:
            for i, r in enumerate(renames):
                with st.container(border=True):
                    st.markdown(
                        f"**{md_safe(r['old_name'])}** ({r['old_range'][0]} → {r['old_range'][1]}) "
                        f"might be the same campaign as **{md_safe(r['new_name'])}** "
                        f"({r['new_range'][0]} → {r['new_range'][1]}) on {r['platform']} — "
                        f"{r['similarity']*100:.0f}% name match, {r['overlap_days']} day(s) overlap."
                    )
                    if st.button(f"Merge under '{r['new_name']}'", key=f"merge_rename_{i}"):
                        n = db.rename_campaign(brand_id, r["platform"], r["old_name"], r["new_name"])
                        invalidate_brand_data_cache()
                        st.success(f"Merged {n} row(s) — '{r['old_name']}' now reports as '{r['new_name']}'.")
                        st.rerun()

    st.markdown("---")
    st.write("**Export normalized data**")
    all_rows = cached_rows_for_brand(brand_id)
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
            # Forecast needs real lookback history, not just the report's own window (which could be
            # a single week) — fit against everything up to the report's end date, same as Insights tab.
            r_forecast_daily = metrics.aggregate(rdf[rdf["date"] <= pd.Timestamp(r_end)], by=["date"])
            r_forecast = metrics.forecast_trend(
                r_forecast_daily, "roas" if brand["business_model"] == "transactional" else "cpa"
            )

            if st.button("Generate PDF report"):
                pdf_bytes = report.build_pdf_report(
                    brand, str(r_start), str(r_end), r_camp_agg, r_econ, r_daily,
                    r_insights, include_insights=include_insights_flag, forecast=r_forecast,
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
