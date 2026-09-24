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

import difflib
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


def strip_title_rows_df(raw_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    The Excel-sheet equivalent of split_multi_table_csv's title-stripping —
    for a sheet read with header=None (so title/subtitle rows land as data
    instead of being wrongly treated as column names). Drops leading rows
    that have at most one non-empty cell, then promotes the first row that
    looks like a real header (more than one populated cell) to be the
    DataFrame's columns. Returns None if the sheet never finds such a row
    (e.g. a genuinely empty or single-column sheet).
    """
    df = raw_df.reset_index(drop=True)
    while len(df) > 2:
        first_row = df.iloc[0]
        non_empty = first_row.notna() & (first_row.astype(str).str.strip() != "")
        if non_empty.sum() <= 1:
            df = df.iloc[1:].reset_index(drop=True)
        else:
            break
    if len(df) < 2:
        return None
    header_row = df.iloc[0]
    if (header_row.notna() & (header_row.astype(str).str.strip() != "")).sum() <= 1:
        return None
    result = df.iloc[1:].reset_index(drop=True)
    result.columns = [str(c).strip() if pd.notna(c) else f"Unnamed_{i}" for i, c in enumerate(header_row)]
    return result


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


def data_completeness_suggestions(rows: list[dict], level: str, platform_id: str | None,
                                   business_model: str) -> list[str]:
    """
    Not a validation gate — these are plain-language suggestions for
    what would make THIS import (or the next one) more useful to the
    rest of the app, checked against what the rows actually contain
    rather than assumed. Only speaks up when a check finds something
    concrete and missing; returns [] rather than padding out a list for
    the sake of having something to say.
    """
    if not rows:
        return []
    suggestions: list[str] = []

    dates = {r["date"] for r in rows if r.get("date")}
    n_days = len(dates)
    if n_days < 14:
        suggestions.append(
            f"Only {n_days} distinct day(s) of data here — trend forecasting and anomaly "
            f"detection need at least 5 days to say anything at all, and get noticeably more "
            f"reliable with 14+. More history (or importing more files) sharpens both."
        )
    if n_days >= 2:
        span_start, span_end = min(dates), max(dates)
        expected_days = (pd.Timestamp(span_end) - pd.Timestamp(span_start)).days + 1
        if expected_days > n_days:
            missing = expected_days - n_days
            suggestions.append(
                f"{missing} day(s) between {span_start} and {span_end} have no rows at all — "
                f"gaps like this can distort day-over-day trend and anomaly reads, which assume "
                f"one row per day."
            )

    if business_model == "transactional":
        has_value = any((r.get("conversion_value") or 0) > 0 for r in rows)
        if not has_value:
            suggestions.append(
                "No revenue/conversion-value data in this file — ROAS-based insights, the "
                "break-even (iROAS) check, and budget reallocation all need it to say anything "
                "for this brand. Include a 'conversion value' / 'purchase value' column if your "
                "export tool offers one."
            )

    if platform_id == "meta" and level == "ad":
        has_reach = any(r.get("reach") for r in rows)
        if not has_reach:
            suggestions.append(
                "No 'Reach' column — creative fatigue detection falls back to a weaker CTR-only "
                "signal instead of the more reliable frequency-based one (impressions ÷ reach). "
                "Meta Ads Manager can add Reach as an export column."
            )

    if level == "campaign":
        if platform_id in ("google", "microsoft"):
            suggestions.append(
                "This is a campaign-level export — a keyword-level export would let a future "
                "keyword-waste check flag specific keywords burning spend with no conversions, "
                "not just the campaign as a whole."
            )
        elif platform_id == "meta":
            suggestions.append(
                "This is a campaign-level export — an ad-level export would let creative fatigue "
                "detection identify which specific ad is fading, not just the campaign."
            )

    zero_signal = sum(
        1 for r in rows
        if (r.get("spend") or 0) > 0 and not (r.get("impressions") or 0)
        and not (r.get("clicks") or 0) and not (r.get("conversions") or 0)
    )
    if zero_signal:
        suggestions.append(
            f"{zero_signal} row(s) have spend recorded but no impressions, clicks, or "
            f"conversions at all — worth checking whether this export left those columns out."
        )

    return suggestions


def _tokenize(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _token_similarity(shorter_tokens: list[str], longer_tokens: list[str]) -> float:
    """Each of the shorter name's tokens matched against its single best-
    fitting token in the longer name, then averaged — asymmetric on
    purpose: this asks "does the longer name explain every word of the
    shorter one," which is what a rename-plus-note looks like, not
    "are these two strings alike overall" (which false-positives on
    short common words and misses a whole appended note dragging down
    a naive average)."""
    if not shorter_tokens or not longer_tokens:
        return 0.0
    total = sum(
        max((difflib.SequenceMatcher(None, st, lt).ratio() for lt in longer_tokens), default=0.0)
        for st in shorter_tokens
    )
    return total / len(shorter_tokens)


def detect_campaign_renames(df: pd.DataFrame, min_similarity: float = 0.85,
                             max_overlap_days: int = 3) -> list[dict]:
    """
    Flags pairs of campaign names within the same platform that are
    probably the SAME campaign carried under a different literal string —
    a short note appended when relaunching, pausing, or tweaking it
    ("HSC PMX" -> "HSC PMX (relaunch march 4)") — rather than two
    genuinely different campaigns, so a rename doesn't silently reset
    that campaign's trend/forecast/decomposition history to zero.

    Two signals, both required: (1) every token of the SHORTER name
    matches well inside the LONGER name's tokens (>= min_similarity) —
    catches an appended note without being thrown off by short common
    words the way whole-string similarity would; (2) the two names'
    active date spans in this data overlap by no more than
    `max_overlap_days` — a real rename means the old name stops
    appearing right around when the new one starts, not that both ran
    side by side, which would mean they're genuinely different
    concurrent campaigns that just happen to share wording.

    Also requires each name to have dates of its own — the old name
    before, the new name after. Without that there's no "stopped here,
    started there" at all: whole-period summary imports put every
    campaign on the same single date, so two names that only ever appear
    together were running side by side in the same report (e.g. "LOLER
    Inspection" and "LOLER Training (pause ...)" — different courses),
    and the span-overlap check alone would wrongly pass them.

    Every result is a SUGGESTION for the viewer to confirm — nothing
    here merges anything on its own.
    """
    if df.empty:
        return []
    spans = df.groupby(["platform", "campaign"], dropna=False).agg(
        start=("date", "min"), end=("date", "max"), dates=("date", lambda s: frozenset(s))
    ).reset_index()

    candidates = []
    for platform, group in spans.groupby("platform"):
        recs = group.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                ra, rb = recs[i], recs[j]
                a, b = ra["campaign"], rb["campaign"]
                if not a or not b or a == b:
                    continue
                a_tokens, b_tokens = _tokenize(a), _tokenize(b)
                if not a_tokens or not b_tokens or a_tokens == b_tokens:
                    continue

                if len(a_tokens) <= len(b_tokens):
                    shorter_tokens, longer_tokens = a_tokens, b_tokens
                    shorter_rec, longer_rec = ra, rb
                else:
                    shorter_tokens, longer_tokens = b_tokens, a_tokens
                    shorter_rec, longer_rec = rb, ra

                sim = _token_similarity(shorter_tokens, longer_tokens)
                if sim < min_similarity:
                    continue

                s_start, s_end = pd.Timestamp(shorter_rec["start"]), pd.Timestamp(shorter_rec["end"])
                l_start, l_end = pd.Timestamp(longer_rec["start"]), pd.Timestamp(longer_rec["end"])
                overlap_start, overlap_end = max(s_start, l_start), min(s_end, l_end)
                overlap_days = max(0, (overlap_end - overlap_start).days + 1)
                if overlap_days > max_overlap_days:
                    continue
                if not (shorter_rec["dates"] - longer_rec["dates"]) or not (longer_rec["dates"] - shorter_rec["dates"]):
                    continue  # no before/after — the two names only ever ran together

                if (s_start, s_end) <= (l_start, l_end):
                    old_rec, new_rec = shorter_rec, longer_rec
                else:
                    old_rec, new_rec = longer_rec, shorter_rec

                candidates.append({
                    "platform": platform,
                    "old_name": old_rec["campaign"], "new_name": new_rec["campaign"],
                    "similarity": round(float(sim), 2), "overlap_days": int(overlap_days),
                    "old_range": (str(old_rec["start"]), str(old_rec["end"])),
                    "new_range": (str(new_rec["start"]), str(new_rec["end"])),
                })

    candidates.sort(key=lambda c: -c["similarity"])
    return candidates
