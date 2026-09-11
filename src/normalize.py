"""
Turns a raw uploaded DataFrame (whatever a platform exported) into canonical
performance rows. This is the layer that lets the rest of the app never know
or care whether a row came from Meta, Google, or Microsoft.

Explicitly does NOT silently drop anything it can't parse: unrecognized
columns, currency it had to assume, and any row it couldn't coerce to a
usable date/spend are all surfaced back to the caller.

Two entry points:
  - normalize_upload()        auto-detected platform (Meta/Google/Microsoft)
  - normalize_manual_mapping() generic CSV fallback — the user assigns each
    raw column to a canonical field themselves, for anything auto-detection
    can't confidently place. Both funnel through the same row-building and
    validation logic below, so neither path gets weaker guarantees than
    the other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import mapping


@dataclass
class NormalizeResult:
    rows: list[dict]
    platform_id: str | None
    platform_label: str
    detection_scores: dict[str, int]
    level: str
    unmapped_columns: list[str]
    currency: str | None
    currency_assumed: bool
    date_start: str | None
    date_end: str | None
    row_count: int
    dropped_row_count: int
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"   # ok / partial / failed
    needs_period_date: bool = False  # no date column at all — a whole-period
                                      # summary export (period totals, no daily
                                      # breakdown). Caller must ask the viewer
                                      # what date range this file covers before
                                      # rows can be stored (schema requires a date).


def split_multi_table_csv(raw_text: str) -> list[str]:
    """
    Some real-world exports aren't a raw platform-UI export at all — they're
    a custom report (a spreadsheet/Looker export) that stacks several tables
    in one CSV, each with its own title row, separated by blank lines. e.g.:

        HST Aug 2026 Google Campaign performance,,,,
        <blank line>
        Campaign,Clicks,Cost,...          <- real header
        ...data rows...
        <blank line>
        HST Aug 2026 Bing Campaign Performance,,,,
        <blank line>
        Campaign name,Clicks,Spend,...    <- a DIFFERENT table's header
        ...data rows...

    Some also stack a SECOND single-cell line under the title — a plain-text
    date range ("1 September 2026 - 7 September 2026") with no commas of its
    own — and skip the blank-line separator entirely between title and
    header. Both cases are handled the same way: any number of consecutive
    single-cell leading lines are dropped, not just one, until a line that
    actually looks like a header (more than one populated cell) is reached.

    Splits on blank-line-separated blocks first, then strips those leading
    single-cell lines from each block, leaving one clean CSV-text-per-table.
    A single ordinary CSV (the common case) round-trips as one block, title
    lines untouched, since its first line already has multiple cells.
    """
    lines = raw_text.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip(" ,\t") == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)

    cleaned = []
    for block in blocks:
        if len(block) < 2:
            continue  # a lone line (title with no table under it, stray blank) — not a table
        while len(block) > 2:
            first_cells = [c.strip() for c in block[0].split(",")]
            non_empty = [c for c in first_cells if c]
            if len(non_empty) <= 1:
                block = block[1:]  # drop a title/subtitle line, keep looking
            else:
                break  # this line has multiple populated cells — treat it as the real header
        cleaned.append("\n".join(block))
    return cleaned


_MONEY_RE = re.compile(r"[^0-9.\-]")


def _to_number(val) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val) if val == val else 0.0  # NaN check
    s = str(val).strip()
    if s in ("", "-", "--", "N/A", "n/a"):
        return 0.0
    s = _MONEY_RE.sub("", s)
    try:
        return float(s) if s not in ("", "-", ".") else 0.0
    except ValueError:
        return 0.0


def _to_iso_date(val) -> str | None:
    if val is None or str(val).strip() == "":
        return None
    try:
        return pd.to_datetime(val).date().isoformat()
    except Exception:
        return None


# Money-column values sometimes carry the currency as a literal symbol
# ("£3,417.47") rather than in a separate column or header — real exports
# from custom reporting tools (not the platforms' own UI) do this. Checked
# as a signal before ever falling back to the brand's configured default,
# so that default is a last resort, not a guess dressed up as detection.
_SYMBOL_CURRENCY = {"£": "GBP", "€": "EUR", "$": "USD", "₹": "INR", "৳": "BDT"}

_AGGREGATE_ROW_NAMES = {"total", "totals", "grand total", "sum", "-", "n/a"}


def _detect_symbol_currency(df: pd.DataFrame, header_map: dict) -> str | None:
    money_cols = [h for h, (f, _c) in header_map.items() if f in ("spend", "conversion_value") and h in df.columns]
    for h in money_cols:
        for val in df[h].dropna().astype(str).head(30):
            v = val.strip()
            if v and v[0] in _SYMBOL_CURRENCY:
                return _SYMBOL_CURRENCY[v[0]]
    return None


def _build_rows(df: pd.DataFrame, header_map: dict[str, tuple[str | None, str | None]],
                 platform_id: str, level: str,
                 brand_default_currency: str) -> tuple[list[dict], int, list[str], bool]:
    """Shared row-building core for both the auto-detected and manual-mapping
    paths. `header_map`: {raw_header: (canonical_field_or_None, captured_currency_or_None)}."""
    header_currency = next((c for _f, c in header_map.values() if c), None)
    explicit_currency_col = next((h for h, (f, _c) in header_map.items() if f == "currency"), None)
    symbol_currency = None if (header_currency or explicit_currency_col) else _detect_symbol_currency(df, header_map)
    fallback_currency = header_currency or symbol_currency or brand_default_currency
    currency_assumed = not header_currency and not explicit_currency_col and not symbol_currency

    campaign_col = next((h for h, (f, _c) in header_map.items() if f == "campaign"), None)
    has_date_col = any(f == "date" for f, _c in header_map.values())

    rows: list[dict] = []
    dropped = 0
    dates: list[str] = []

    for _, raw_row in df.iterrows():
        if campaign_col is not None:
            camp_val = raw_row.get(campaign_col)
            if camp_val is not None and str(camp_val).strip().lower() in _AGGREGATE_ROW_NAMES:
                dropped += 1  # a trailing summary row ("Total"), not a real campaign
                continue

        canon: dict = {f: None for f in mapping.CANONICAL_FIELDS}
        row_currency = None
        for raw_h, (canon_field, _captured) in header_map.items():
            if canon_field is None:
                continue
            val = raw_row.get(raw_h)
            if canon_field == "date":
                canon["date"] = _to_iso_date(val)
            elif canon_field == "currency":
                row_currency = str(val).strip() if val is not None else None
            elif canon_field in ("spend", "impressions", "clicks", "conversions", "conversion_value", "reach"):
                canon[canon_field] = _to_number(val)
            else:
                canon[canon_field] = None if val is None or str(val).strip() == "" else str(val).strip()

        canon["currency"] = row_currency or fallback_currency
        canon["level"] = level
        canon["platform"] = platform_id

        # A row missing its date is only a real parse failure when a date
        # column actually exists — a whole-period summary export (no daily
        # breakdown at all) legitimately has no date on any row, and the
        # caller fills one in afterward rather than every row being dropped.
        if (has_date_col and not canon["date"]) or canon["campaign"] in (None, ""):
            dropped += 1
            continue

        rows.append(canon)
        if canon["date"]:
            dates.append(canon["date"])

    needs_period_date = (not has_date_col) and len(rows) > 0
    return rows, dropped, dates, currency_assumed, needs_period_date


def _finalize(rows, dropped, dates, currency_assumed, unmapped, filename,
              platform_id, platform_label, detection_scores, level,
              brand_default_currency, needs_period_date=False) -> NormalizeResult:
    warnings: list[str] = []
    if needs_period_date:
        warnings.append(
            f"'{filename}' has no date column — it looks like a whole-period summary "
            "(one row per campaign, totals for the period), not a daily breakdown. "
            "Pick the date range this file covers before importing."
        )
    if unmapped:
        warnings.append(
            f"{len(unmapped)} column(s) weren't recognized and were left out: "
            + ", ".join(unmapped[:8]) + (", ..." if len(unmapped) > 8 else "")
        )
    if dropped:
        warnings.append(f"{dropped} row(s) were skipped — missing a usable date or campaign name.")
    if currency_assumed:
        warnings.append(
            f"No currency column or symbol found in '{filename}' — assumed the brand's "
            f"configured currency ({brand_default_currency}). Confirm this is correct."
        )

    status = "ok" if not warnings else ("partial" if rows else "failed")

    return NormalizeResult(
        rows=rows, platform_id=platform_id, platform_label=platform_label,
        detection_scores=detection_scores, level=level, unmapped_columns=unmapped,
        currency=(rows[0]["currency"] if rows else None), currency_assumed=currency_assumed,
        date_start=min(dates) if dates else None, date_end=max(dates) if dates else None,
        row_count=len(rows), dropped_row_count=dropped, warnings=warnings, status=status,
        needs_period_date=needs_period_date,
    )


def normalize_upload(df: pd.DataFrame, filename: str,
                      brand_default_currency: str) -> NormalizeResult:
    headers = list(df.columns)
    profile, scores = mapping.detect_platform(headers)

    if profile is None:
        return NormalizeResult(
            rows=[], platform_id=None, platform_label="Unrecognized",
            detection_scores=scores, level="unknown",
            unmapped_columns=headers, currency=None, currency_assumed=False,
            date_start=None, date_end=None, row_count=0, dropped_row_count=0,
            warnings=[
                f"Couldn't confidently identify the platform for '{filename}' from its columns. "
                "Use the manual mapping fallback below to import it as a generic report."
            ],
            status="failed",
        )

    headers_norm = {mapping._norm(h) for h in headers}
    level = profile.detect_level(headers_norm)
    header_map = mapping.map_headers(profile, headers)
    unmapped = [h for h in headers if h not in header_map]

    rows, dropped, dates, currency_assumed, needs_period_date = _build_rows(
        df, header_map, profile.id, level, brand_default_currency
    )
    return _finalize(rows, dropped, dates, currency_assumed, unmapped, filename,
                      profile.id, profile.label, scores, level, brand_default_currency,
                      needs_period_date)


def normalize_manual_mapping(df: pd.DataFrame, filename: str, brand_default_currency: str,
                              column_choices: dict[str, str], level: str,
                              platform_label: str = "Generic / manually mapped") -> NormalizeResult:
    """
    Generic CSV fallback. `column_choices`: {raw_header: canonical_field or
    "ignore"} as chosen by the user in the UI — required fields (date,
    campaign, spend) must be assigned or this refuses to import rather than
    guess.
    """
    headers = list(df.columns)
    missing_required = [
        f for f in mapping.REQUIRED_FOR_IMPORT
        if f not in column_choices.values()
    ]
    if missing_required:
        return NormalizeResult(
            rows=[], platform_id="generic", platform_label=platform_label,
            detection_scores={}, level=level, unmapped_columns=headers,
            currency=None, currency_assumed=False, date_start=None, date_end=None,
            row_count=0, dropped_row_count=0,
            warnings=[f"Map these required fields before importing: {', '.join(missing_required)}."],
            status="failed",
        )

    header_map = {
        h: (None if choice == "ignore" else choice, None)
        for h, choice in column_choices.items()
    }
    unmapped = [h for h, choice in column_choices.items() if choice == "ignore"]

    rows, dropped, dates, currency_assumed, needs_period_date = _build_rows(
        df, header_map, "generic", level, brand_default_currency
    )
    return _finalize(rows, dropped, dates, currency_assumed, unmapped, filename,
                      "generic", platform_label, {}, level, brand_default_currency,
                      needs_period_date)
