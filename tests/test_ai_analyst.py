import json
from types import SimpleNamespace as NS

import pytest

from src import ai_analyst, metrics


def _row(platform, campaign, spend, revenue, date="2026-08-31", conversions=5.0):
    return {
        "platform": platform, "campaign": campaign, "level": "campaign", "date": date,
        "spend": spend, "conversion_value": revenue, "conversions": conversions,
        "clicks": 50, "impressions": 1000, "keyword": None, "ad_set": None, "ad": None,
        "currency": "GBP", "reach": None, "result_type": None,
    }


BRAND = {"name": "EST", "business_model": "transactional", "conversion_type": "purchase",
         "currency": "GBP", "margin_pct": 0.85, "aov": None, "ltv": None,
         "target_roas": None, "target_cpa": None}


@pytest.fixture
def df():
    return metrics.rows_to_df([
        _row("google", "First Aid PMX", 1000, 2350),
        _row("microsoft", "First Aid CAT", 325, 312),
        _row("google", "First Aid PMX", 800, 1500, date="2026-07-31"),  # previous month
    ])


def text(t):
    return NS(type="text", text=t)


def tool_use(name, tool_input=None, id_="tu_1"):
    return NS(type="tool_use", name=name, input=tool_input or {}, id=id_)


def response(stop_reason, *content):
    return NS(stop_reason=stop_reason, content=list(content),
              usage=NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=0,
                       cache_creation_input_tokens=0))


class FakeClient:
    """Stands in for anthropic.Anthropic(): returns scripted responses and
    records every request so tests can inspect what was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def test_tool_round_then_answer(df):
    client = FakeClient(
        response("tool_use", text("Checking courses."), tool_use("get_course_view")),
        response("end_turn", text("Shift First Aid budget toward Google.")),
    )
    history = []
    reply = ai_analyst.ask(history, "Where should First Aid budget go?", df, BRAND, client=client)

    assert reply.text == "Shift First Aid budget toward Google."
    assert reply.tools_used == ["get_course_view"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]

    tool_result = history[2]["content"][0]
    courses = json.loads(tool_result["content"])["courses"]
    assert courses[0]["course"] == "first aid" and courses[0]["verdict"] == "shift"


def test_requests_use_opus_adaptive_thinking_and_server_fallbacks(df):
    client = FakeClient(response("end_turn", text("ok")))
    ai_analyst.ask([], "hi", df, BRAND, client=client)
    req = client.requests[0]
    assert req["model"] == "claude-opus-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req["betas"]
    assert {t["name"] for t in req["tools"]} == {t["name"] for t in ai_analyst.TOOLS}


def test_tool_errors_are_returned_to_claude_not_raised(df):
    client = FakeClient(
        response("tool_use", tool_use("get_campaigns", {"bogus_arg": 1})),
        response("end_turn", text("Recovered.")),
    )
    history = []
    reply = ai_analyst.ask(history, "q", df, BRAND, client=client)
    result = history[2]["content"][0]
    assert result["is_error"] is True
    assert reply.text == "Recovered."


def test_only_declared_tools_can_run(df):
    tools = ai_analyst.AnalystTools(df, BRAND)
    with pytest.raises(ValueError):
        tools.run("_window", {})
    with pytest.raises(ValueError):
        tools.run("__init__", {})


def test_refusal_is_reported_and_not_kept_in_history(df):
    client = FakeClient(response("refusal"))
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": [text("ok")]}]
    reply = ai_analyst.ask(history, "q", df, BRAND, client=client)
    assert reply.refused
    assert len(history) == 2  # the declined question was removed


def test_fallback_marker_blocks_are_not_replayed(df):
    client = FakeClient(response("end_turn", NS(type="fallback"), text("answer")))
    history = []
    ai_analyst.ask(history, "q", df, BRAND, client=client)
    assert [b.type for b in history[-1]["content"]] == ["text"]


def test_tool_loop_is_bounded(df):
    client = FakeClient(*[response("tool_use", tool_use("get_brand_settings", id_=f"t{i}"))
                          for i in range(ai_analyst.MAX_TOOL_ROUNDS)])
    reply = ai_analyst.ask([], "q", df, BRAND, client=client)
    assert "too many analysis steps" in reply.text
    assert len(client.requests) == ai_analyst.MAX_TOOL_ROUNDS


def test_account_overview_compares_latest_period_to_previous(df):
    overview = ai_analyst.AnalystTools(df, BRAND).get_account_overview(30)
    assert overview["period"] == {"start": "2026-08-02", "end": "2026-08-31"}
    assert overview["totals"]["spend"] == 1325.0  # July's import falls in the baseline, not here
    assert set(overview["by_platform"]) == {"google", "microsoft"}
    assert overview["pct_change_vs_baseline"]["spend"] == pytest.approx(65.62, abs=0.01)


def test_keyword_waste_reports_when_no_keyword_data(df):
    result = ai_analyst.AnalystTools(df, BRAND).get_keyword_waste()
    assert result == {"has_keyword_level_data": False, "wasted_keywords": [], "wasted_recurring_words": []}
