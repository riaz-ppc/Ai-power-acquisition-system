"""
Platform detection + column-mapping layer.

This is the piece the whole system is built on top of: every ad platform
names the same underlying metric differently, and export formats change
over time. Rather than hard-coding "if column == X" logic scattered through
the app, every platform gets one PlatformProfile here that says (a) how to
recognize its exports and (b) how to translate its columns into the
canonical schema. Adding a platform later means adding one profile, not
touching the parser.

Canonical row fields (what everything downstream reads):
  date, level, campaign, ad_set, ad, keyword,
  spend, impressions, clicks, conversions, conversion_value,
  currency, result_type

`result_type` keeps the platform's own label for what a "conversion" was
(e.g. Meta's "Purchases" vs "Leads") — useful context, but the brand's
configured conversion_type (purchase / lead / enrollment) is what the
analysis layer actually uses, since the same brand should read consistently
even if one report double-counts on-platform vs website results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

CANONICAL_FIELDS = [
    "date", "level", "campaign", "ad_set", "ad", "keyword",
    "spend", "impressions", "clicks", "conversions", "conversion_value", "reach",
    "currency", "result_type",
]

REQUIRED_FOR_IMPORT = ["date", "campaign", "spend"]


def _norm(h: str) -> str:
    """Lowercase, trim, collapse whitespace — for matching, never for display."""
    return re.sub(r"\s+", " ", str(h).strip().lower())


@dataclass
class PlatformProfile:
    id: str
    label: str
    # Header substrings/regexes that are strong, near-unique signals this
    # export came from this platform. Detection scores by how many hit.
    signals: list[str]
    # normalized-header -> canonical field, for exact/substring matches.
    # Value can be a plain canonical field name, or a (field, transform) pair.
    column_map: dict[str, str]
    # Regex-based column matches for headers with dynamic parts, e.g.
    # Meta's "Amount spent (BDT)". Each entry: (pattern, canonical_field,
    # optional group index that captures the currency code).
    regex_map: list[tuple[str, str, Optional[int]]] = field(default_factory=list)
    level_rules: list[tuple[str, str]] = field(default_factory=list)  # (header substr, level)

    def coverage(self, headers_norm: set[str]) -> int:
        """How many of this file's headers this profile actually knows how
        to map. More robust than a fixed signal list — a custom/trimmed
        export still scores well as long as the columns it DOES have are
        ones this platform is known to use."""
        n = 0
        for h in headers_norm:
            if h in self.column_map:
                n += 1
                continue
            if any(re.match(pat, h) for pat, _f, _g in self.regex_map):
                n += 1
        return n

    def signal_score(self, headers_norm: set[str]) -> int:
        s = 0
        for sig in self.signals:
            if any(sig in h for h in headers_norm):
                s += 1
        return s

    def score(self, headers_norm: set[str]) -> int:
        """Coverage carries most of the weight; signals (platform-distinctive
        column names/abbreviations) break ties between platforms that share
        a lot of generic column names (impressions/clicks/conversions)."""
        return self.coverage(headers_norm) + 2 * self.signal_score(headers_norm)

    def detect_level(self, headers_norm: set[str]) -> str:
        for substr, level in self.level_rules:
            if any(substr in h for h in headers_norm):
                return level
        return "campaign"


META = PlatformProfile(
    id="meta",
    label="Meta Ads Manager",
    signals=[
        "amount spent", "reporting starts", "reporting ends",
        "link clicks", "cost per results", "ad set name",
    ],
    column_map={
        "reporting starts": "date",
        "day": "date",
        "campaign name": "campaign",
        "ad set name": "ad_set",
        "adset name": "ad_set",
        "ad name": "ad",
        "impressions": "impressions",
        "reach": "reach",
        "link clicks": "clicks",
        "clicks (all)": "clicks",
        "results": "conversions",
        "result type": "result_type",
        "purchases": "conversions",
        "website purchases": "conversions",
        "purchases conversion value": "conversion_value",
        "website purchases conversion value": "conversion_value",
        "purchase roas (return on ad spend)": None,  # derived, recomputed downstream
        "currency": "currency",
    },
    regex_map=[
        (r"^amount spent \((\w+)\)$", "spend", 1),
        (r"^cpm.*\((\w+)\)$", None, 1),
        (r"^cost per results?\s*\((\w+)\)$", None, 1),
    ],
    level_rules=[
        ("ad name", "ad"),
        ("ad set name", "ad_set"),
        ("adset name", "ad_set"),
    ],
)

GOOGLE = PlatformProfile(
    id="google",
    label="Google Ads",
    signals=[
        "avg. cpc", "cost / conv.", "search impr. share", "conv. rate",
        "conv. value", "campaign", "auto-applied recommendations",
    ],
    column_map={
        "day": "date",
        "week": "date",
        "month": "date",
        "campaign": "campaign",
        "ad group": "ad_set",
        "keyword": "keyword",
        "currency code": "currency",
        "currency": "currency",
        "cost": "spend",
        "impressions": "impressions",
        "clicks": "clicks",
        "conversions": "conversions",
        "conv. value": "conversion_value",
        "all conv. value": "conversion_value",
        "revenue": "conversion_value",
        "conversion action": "result_type",
    },
    regex_map=[],
    level_rules=[
        ("keyword", "keyword"),
        ("ad group", "ad_set"),
    ],
)

MICROSOFT = PlatformProfile(
    id="microsoft",
    label="Microsoft Ads",
    signals=[
        "avg. cpc", "spend", "impr.", "network", "campaign name",
    ],
    column_map={
        "date": "date",
        "time period": "date",
        "campaign name": "campaign",
        "campaign": "campaign",
        "ad group name": "ad_set",
        "ad group": "ad_set",
        "keyword": "keyword",
        "currency code": "currency",
        "spend": "spend",
        "impressions": "impressions",
        "impr.": "impressions",
        "clicks": "clicks",
        "conversions": "conversions",
        "revenue": "conversion_value",
    },
    regex_map=[],
    level_rules=[
        ("keyword", "keyword"),
        ("ad group name", "ad_set"),
    ],
)

PLATFORMS = [META, GOOGLE, MICROSOFT]

MIN_COVERAGE = 2  # at least this many headers must be recognizably this platform's


def detect_platform(headers: list[str]) -> tuple[Optional[PlatformProfile], dict[str, int]]:
    """Returns (best-matching profile or None, {platform_id: score}) for transparency."""
    headers_norm = {_norm(h) for h in headers}
    scores = {p.id: p.score(headers_norm) for p in PLATFORMS}
    best = max(PLATFORMS, key=lambda p: scores[p.id])
    if best.coverage(headers_norm) < MIN_COVERAGE:
        return None, scores
    return best, scores


def map_headers(profile: PlatformProfile, headers: list[str]) -> dict[str, tuple[str, Optional[str]]]:
    """
    For each raw header, returns {raw_header: (canonical_field_or_None, captured_currency_or_None)}.
    A canonical_field of None means "recognized but intentionally dropped"
    (e.g. a platform-computed ratio we recompute ourselves downstream).
    A raw header with no entry at all means "unrecognized" — surfaced to the
    user rather than silently dropped.
    """
    out: dict[str, tuple[str, Optional[str]]] = {}
    for h in headers:
        hn = _norm(h)
        matched = False
        if hn in profile.column_map:
            out[h] = (profile.column_map[hn], None)
            matched = True
        if not matched:
            for pattern, canon_field, group_idx in profile.regex_map:
                m = re.match(pattern, hn)
                if m:
                    currency = m.group(group_idx) if group_idx else None
                    out[h] = (canon_field, currency.upper() if currency else None)
                    matched = True
                    break
        # unmatched headers are simply absent from `out` — caller reports them
    return out
