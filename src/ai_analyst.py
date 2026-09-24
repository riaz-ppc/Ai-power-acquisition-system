"""
AI analyst: Claude answers questions about a brand's PPC performance by
calling this app's own deterministic metrics functions as tools.

The rule engine in src/metrics.py stays the source of truth — every number
Claude cites comes back from a tool call, never from the model's own
arithmetic or memory. Claude's job is the part rules are bad at: choosing
which analyses answer the question, connecting results across them, and
explaining the trade-offs in plain language.

Read-only: every tool computes over rows already imported into this app's
database — the app has no write access to any ad account, so the prompt
tells Claude never to claim an action was taken.

Needs ANTHROPIC_API_KEY (or another credential the Anthropic SDK resolves).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from . import metrics
from .digest import compute_digest_items

MODEL = os.environ.get("AI_ANALYST_MODEL", "claude-opus-5")
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are a senior PPC analyst working inside a reporting app for a UK online-training \
business that advertises its courses on Google Ads, Microsoft (Bing) Ads and Meta. You help the user \
understand performance and decide what to do next.

How you work:
- Every figure you state must come from a tool result in this conversation. Never estimate, extrapolate \
or invent numbers; if the tools can't answer something, say what data is missing.
- Break-even verdicts depend on the brand's margin and targets (get_brand_settings). When a conclusion \
rests on them, say so — and if the margin isn't set, point that out before judging profitability.
- Be cautious with thin data: small spend or few conversions can swing ratios. Say when a signal is too \
weak to act on.
- Your tools only read data already imported into this app; you have no access to the ad accounts \
themselves, so never claim a change has been made.

How you answer:
- Lead with the direct answer, then the evidence, then specific next steps in priority order.
- Name campaigns and courses exactly as they appear in the data.
- Keep it concise and skimmable; use short lists or a small table when comparing things."""

TOOLS = [
    {
        "name": "get_brand_settings",
        "description": "The brand's business model (transactional = ROAS-driven, lead_gen = CPA-driven), "
                       "currency, contribution margin, and ROAS/CPA targets. Break-even verdicts elsewhere "
                       "are derived from these.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_account_overview",
        "description": "Totals for the latest period (spend, conversions, revenue, ROAS, CPA) overall and per "
                       "platform, with % change vs the equal-length period before it and the date range "
                       "covered. Start here for 'how are we doing' questions.",
        "input_schema": {
            "type": "object",
            "properties": {"period_days": {"type": "integer", "description": "Period length ending at the latest date in the data. Default 30."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_campaigns",
        "description": "Per-campaign spend, conversions, revenue, ROAS and CPA for the latest period, sorted "
                       "by spend. Optionally filter to one platform.",
        "input_schema": {
            "type": "object",
            "properties": {
                "period_days": {"type": "integer", "description": "Default 30."},
                "platform": {"type": "string", "enum": ["google", "microsoft", "meta"]},
                "limit": {"type": "integer", "description": "Max campaigns returned. Default 25."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_course_view",
        "description": "Campaigns grouped into courses across platforms, with a verdict per course: "
                       "shift (much better on one platform), below_break_even, expand (good on its only "
                       "platform, untested elsewhere), healthy, insufficient_data. Use for 'where should "
                       "budget go' and cross-platform questions.",
        "input_schema": {
            "type": "object",
            "properties": {"period_days": {"type": "integer", "description": "Default 30."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_insights",
        "description": "Rule-based findings for the latest period vs the previous one: campaigns below "
                       "break-even, big moves, scale candidates — each with its metric, threshold and a "
                       "suggested action.",
        "input_schema": {
            "type": "object",
            "properties": {"period_days": {"type": "integer", "description": "Default 30."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_keyword_waste",
        "description": "Keywords, and words recurring across keywords, spending with zero conversions where "
                       "even a 95%-confidence best case is below the account's own conversion rate. Only "
                       "Google/Microsoft keyword-level imports have this data.",
        "input_schema": {
            "type": "object",
            "properties": {"period_days": {"type": "integer", "description": "Default 30."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_weekly_digest",
        "description": "The app's ranked list of the most important things to act on in the last 7 days of "
                       "data (priority 0 = critical, 1 = watch, 2 = opportunity).",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def _r(v):
    """Round for compact, readable tool output; keep None/NaN as null."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    return round(float(v), 2)


class AnalystTools:
    """Executes tool calls against one brand's rows. Pure computation over
    already-imported data — no network, no writes."""

    def __init__(self, df: pd.DataFrame, brand: dict):
        self.df = df
        self.brand = brand

    def _window(self, days: int | None):
        days = max(1, int(days or 30))
        end = self.df["date"].max().normalize()
        start = end - timedelta(days=days - 1)
        base_end = start - timedelta(days=1)
        base_start = base_end - timedelta(days=days - 1)
        scoped = self.df[(self.df["date"] >= start) & (self.df["date"] <= end)]
        return scoped, start, end, base_start, base_end

    @staticmethod
    def _totals(d: pd.DataFrame) -> dict:
        spend, conv, rev = d["spend"].sum(), d["conversions"].sum(), d["conversion_value"].sum()
        return {"spend": _r(spend), "conversions": _r(conv), "revenue": _r(rev),
                "roas": _r(rev / spend) if spend else None, "cpa": _r(spend / conv) if conv else None}

    def get_brand_settings(self) -> dict:
        b = self.brand
        return {k: b.get(k) for k in ("name", "business_model", "conversion_type", "currency",
                                      "margin_pct", "aov", "ltv", "target_roas", "target_cpa")}

    def get_account_overview(self, period_days: int | None = None) -> dict:
        scoped, start, end, bs, be = self._window(period_days)
        compare = metrics.compare_periods(self.df, start, end, bs, be)
        return {
            "period": {"start": str(start.date()), "end": str(end.date())},
            "baseline_period": {"start": str(bs.date()), "end": str(be.date())},
            "totals": self._totals(scoped),
            "pct_change_vs_baseline": {k: _r(v) for k, v in compare["pct_change"].items()},
            "by_platform": {p: self._totals(g) for p, g in scoped.groupby("platform")},
            "note": "Imports without daily data are stored on their period's end date, so a period "
                    "contains whole monthly/weekly imports whose end date falls inside it.",
        }

    def get_campaigns(self, period_days: int | None = None, platform: str | None = None,
                      limit: int | None = None) -> dict:
        scoped, start, end, _, _ = self._window(period_days)
        if platform:
            scoped = scoped[scoped["platform"] == platform]
        agg = metrics.aggregate(scoped, by=["platform", "campaign"])
        if agg.empty:
            return {"period": {"start": str(start.date()), "end": str(end.date())}, "campaigns": []}
        agg = agg[agg["spend"] > 0].sort_values("spend", ascending=False).head(int(limit or 25))
        return {
            "period": {"start": str(start.date()), "end": str(end.date())},
            "campaigns": [{
                "platform": r["platform"], "campaign": r["campaign"], "spend": _r(r["spend"]),
                "conversions": _r(r["conversions"]), "revenue": _r(r["conversion_value"]),
                "roas": _r(r["roas"]), "cpa": _r(r["cpa"]),
            } for _, r in agg.iterrows()],
        }

    def get_course_view(self, period_days: int | None = None) -> dict:
        scoped, *_ = self._window(period_days)
        return {"courses": [{
            "course": c["course"], "verdict": c["verdict"], "detail": c["detail"],
            "total_spend": _r(c["total_spend"]),
            "platforms": {p: {"spend": _r(cell["spend"]), "revenue": _r(cell["revenue"]),
                              "conversions": _r(cell["conversions"]), "campaigns": cell["campaigns"]}
                          for p, cell in c["platforms"].items()},
        } for c in metrics.course_platform_view(scoped, self.brand)]}

    def get_insights(self, period_days: int | None = None) -> dict:
        scoped, start, end, bs, be = self._window(period_days)
        compare = metrics.compare_periods(self.df, start, end, bs, be)
        camp_agg = metrics.aggregate(scoped, by=["campaign"])
        return {"insights": [{
            "severity": i.severity, "title": i.title, "detail": i.detail, "metric": i.metric,
            "threshold": i.threshold, "suggested_action": i.suggested_action, "campaign": i.campaign,
        } for i in metrics.generate_insights(camp_agg, self.brand, compare)]}

    def get_keyword_waste(self, period_days: int | None = None) -> dict:
        scoped, *_ = self._window(period_days)
        has_keywords = bool(scoped["keyword"].notna().any()) if "keyword" in scoped.columns else False
        return {
            "has_keyword_level_data": has_keywords,
            "wasted_keywords": metrics.keyword_waste_candidates(scoped),
            "wasted_recurring_words": metrics.keyword_word_waste(scoped),
        }

    def get_weekly_digest(self) -> dict:
        return {"items": [{k: i[k] for k in ("priority", "title", "detail")}
                          for i in compute_digest_items(self.df, self.brand)]}

    def run(self, name: str, tool_input: dict) -> str:
        fn = getattr(self, name, None) if name in {t["name"] for t in TOOLS} else None
        if fn is None:
            raise ValueError(f"Unknown tool: {name}")
        return json.dumps(fn(**(tool_input or {})), default=str, sort_keys=True)


@dataclass
class AnalystReply:
    text: str
    tools_used: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    refused: bool = False


def _client():
    import anthropic
    return anthropic.Anthropic()


def ask(history: list[dict], question: str, df: pd.DataFrame, brand: dict, client=None) -> AnalystReply:
    """
    Runs one analyst turn. `history` is this conversation's prior API
    messages (tool calls included) and is extended in place, so the caller
    keeps it (e.g. in st.session_state) for follow-up questions. Returns the
    final answer plus which tools ran and the token usage, for transparency.
    """
    client = client or _client()
    tools = AnalystTools(df, brand)
    history.append({"role": "user", "content": question})
    reply = AnalystReply(text="")

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=history,
            thinking={"type": "adaptive"},
            cache_control={"type": "ephemeral"},
            # A classifier decline is re-run server-side on Anthropic's
            # recommended fallback model instead of failing the question.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            reply.input_tokens += (getattr(usage, "input_tokens", 0) or 0) \
                + (getattr(usage, "cache_read_input_tokens", 0) or 0) \
                + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            reply.output_tokens += getattr(usage, "output_tokens", 0) or 0

        if response.stop_reason == "refusal":
            history.pop()  # don't keep a question the model declined in the replayed history
            reply.refused = True
            reply.text = ("The AI analyst declined to answer this one. Try rephrasing the question "
                          "around the account's performance data.")
            return reply

        # `fallback` blocks are audit markers only — safe to drop from replayed history.
        content = [b for b in response.content if getattr(b, "type", None) != "fallback"]
        history.append({"role": "assistant", "content": content})

        if response.stop_reason != "tool_use":
            reply.text = "\n\n".join(b.text for b in content if getattr(b, "type", None) == "text").strip()
            if response.stop_reason == "max_tokens":
                reply.text += "\n\n_(Answer cut off at the length limit — ask a narrower follow-up.)_"
            return reply

        results = []
        for block in content:
            if getattr(block, "type", None) != "tool_use":
                continue
            reply.tools_used.append(block.name)
            try:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": tools.run(block.name, block.input)})
            except Exception as e:  # bad arguments or an empty slice — let Claude recover
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"Error: {e}", "is_error": True})
        history.append({"role": "user", "content": results})

    reply.text = ("Stopped after too many analysis steps without a final answer — try a more "
                  "specific question.")
    return reply


BRIEFING_QUESTION = (
    "Write my weekly PPC briefing for this brand: the three to five things that matter most right "
    "now, what to do about each, and anything I should watch next week. Check the digest, the course "
    "view and the account overview first."
)
