"""
Actual-order revenue reconciliation against platform-claimed conversion
value. Orders are a genuinely different shape from ad performance data (no
spend/impressions/clicks at all) — kept as their own table and detection
path rather than forced into the campaign-performance schema.

The real problem this exists for: platforms report conversion_value from
their own pixel/attribution, which can meaningfully overstate (or
understate) actual revenue. An order-level export (real transactions) is
the ground truth to check that against — but its campaign names are
usually shorthand ("Office PMX") that don't literally match the platform's
full campaign names ("Office Skills PMax (jan26)"), so matching them needs
a fuzzy step the viewer confirms, never a silent guess.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import pandas as pd

ORDER_HEADER_SYNONYMS = {
    "date": "order_date", "order date": "order_date", "created": "order_date",
    "order id": "order_id", "order #": "order_id", "id": "order_id",
    "amount": "amount", "total": "amount", "order total": "amount",
    "revenue": "amount", "actual revenue": "amount", "actual value": "amount",
    "campaign name": "campaign", "campaign": "campaign",
    "source": "source", "channel": "source", "platform": "source",
}
REQUIRED_ORDER_FIELDS = {"order_date", "amount"}

SOURCE_TO_PLATFORM = {
    "google": "google", "bing": "microsoft", "microsoft": "microsoft",
    "meta": "meta", "facebook": "meta", "fb": "meta",
}

_AMOUNT_RE = re.compile(r"[^0-9.\-]")


def _norm(h: str) -> str:
    # Underscores collapse to spaces too, not just whitespace — a raw
    # header can legitimately be spelled "actual_revenue" or "Actual
    # Revenue" for the same thing, and both should match one synonym key.
    return re.sub(r"[\s_]+", " ", str(h).strip().lower())


def looks_like_order_sheet(headers: list[str]) -> bool:
    """A real order export has a date and an amount but none of the ad
    metrics — that combination is what actually distinguishes it from a
    campaign performance report, not just sharing a 'Date' column name."""
    normed = {_norm(h) for h in headers}
    mapped_fields = {ORDER_HEADER_SYNONYMS[h] for h in normed if h in ORDER_HEADER_SYNONYMS}
    has_ad_metrics = any(h in normed for h in ("impressions", "clicks", "spend", "cost", "impr."))
    return REQUIRED_ORDER_FIELDS.issubset(mapped_fields) and not has_ad_metrics


@dataclass
class OrderImportResult:
    rows: list[dict]
    unmapped_columns: list[str]
    row_count: int
    dropped_row_count: int
    date_start: str | None
    date_end: str | None
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"  # ok / partial / failed


def normalize_orders(df: pd.DataFrame, filename: str) -> OrderImportResult:
    headers = list(df.columns)
    header_map = {h: ORDER_HEADER_SYNONYMS[_norm(h)] for h in headers if _norm(h) in ORDER_HEADER_SYNONYMS}
    unmapped = [h for h in headers if h not in header_map]

    rows: list[dict] = []
    dropped = 0
    dates: list[str] = []

    for _, raw_row in df.iterrows():
        rec = {"order_date": None, "order_id": None, "amount": None, "campaign": None, "source": None}
        for raw_h, field_name in header_map.items():
            val = raw_row.get(raw_h)
            if field_name == "order_date":
                try:
                    rec["order_date"] = pd.to_datetime(val).date().isoformat()
                except Exception:
                    rec["order_date"] = None
            elif field_name == "amount":
                s = _AMOUNT_RE.sub("", str(val).strip()) if val is not None else ""
                try:
                    rec["amount"] = float(s) if s not in ("", "-", ".") else None
                except ValueError:
                    rec["amount"] = None
            else:
                rec[field_name] = None if val is None or str(val).strip() == "" else str(val).strip()

        if not rec["order_date"] or rec["amount"] is None:
            dropped += 1
            continue
        rows.append(rec)
        dates.append(rec["order_date"])

    warnings = []
    if unmapped:
        warnings.append(f"{len(unmapped)} column(s) weren't recognized and were left out: " + ", ".join(unmapped[:6]))
    if dropped:
        warnings.append(f"{dropped} row(s) were skipped — missing a usable date or amount.")

    return OrderImportResult(
        rows=rows, unmapped_columns=unmapped, row_count=len(rows), dropped_row_count=dropped,
        date_start=min(dates) if dates else None, date_end=max(dates) if dates else None,
        warnings=warnings, status="ok" if not warnings else ("partial" if rows else "failed"),
    )


def suggest_campaign_matches(order_campaigns: list[str],
                              known_campaigns_by_platform: dict[str, list[str]]) -> dict[str, str | None]:
    """
    Best-effort fuzzy match from an order sheet's shorthand campaign name
    ("Office PMX") to a platform's real campaign name ("Office Skills PMax
    (jan26)"). Never trusted silently for the reconciliation numbers — the
    UI shows every suggestion and the viewer confirms or corrects it before
    it's used, since a wrong match would misattribute real revenue.

    `known_campaigns_by_platform`: {platform_id: [campaign names]} — when
    the order's own source narrows to one platform, only that platform's
    campaigns are considered (the same shorthand can exist on two
    platforms — "MS Office" on both a paused Google campaign and an
    active Bing one — so scoping by source avoids matching the wrong one).
    """
    suggestions: dict[str, str | None] = {}
    all_campaigns = [c for cs in known_campaigns_by_platform.values() for c in cs]

    for raw in order_campaigns:
        raw_l = raw.lower().strip()

        exact = next((c for c in all_campaigns if c.lower() == raw_l), None)
        if exact:
            suggestions[raw] = exact
            continue

        if not all_campaigns:
            suggestions[raw] = None
            continue

        # Whole-string similarity picks the wrong campaign on real data: "Office
        # PMX" scores closer to the unrelated "MS Office" than to the campaign
        # it actually shortens, "Office Skills PMax (jan26)", because raw
        # character overlap doesn't care that "PMX" is short for "PMax" inside
        # a much longer name. Scoring per-token instead — each raw token
        # matched against its single best-fitting token in the candidate, then
        # averaged — rewards a candidate that explains every word of the
        # shorthand, not just one that happens to share a common word.
        raw_tokens = _tokenize(raw)
        scored = [(c, _token_similarity(raw_tokens, _tokenize(c))) for c in all_campaigns]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_campaign, best_score = scored[0]
        suggestions[raw] = best_campaign if best_score >= 0.55 else None

    return suggestions


def _tokenize(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _token_similarity(raw_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not raw_tokens or not candidate_tokens:
        return 0.0
    total = 0.0
    for rt in raw_tokens:
        best = max((difflib.SequenceMatcher(None, rt, ct).ratio() for ct in candidate_tokens), default=0.0)
        total += best
    return total / len(raw_tokens)
