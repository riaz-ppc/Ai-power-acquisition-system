import pytest

from src import metrics


@pytest.mark.parametrize("google, bing, key", [
    ("AML Training (L 5 March)", "AML (Relanuch 14 July)", "aml"),
    ("LOLER Inspection", "Loler Inspection", "loler inspection"),
    ("First Aid PMX", "First Aid CAT", "first aid"),
    ("Food Hygiene PMax (reluanch 3Aug)", "Food Hygiene Cat (Relanuch 14 July)", "food hygiene"),
    ("COSHH", "COSHH (Relaunched- 12th Aug", "coshh"),  # unclosed paren
    ("Pest Control C.", "Pest Control C (flop)", "pest control c"),
    ("Safer Recruitment CAT", "Safer Recruitment Cat", "safer recruitment"),
])
def test_same_course_on_both_platforms_groups_together(google, bing, key):
    assert metrics.course_key(google) == metrics.course_key(bing) == key


def test_version_tags_are_stripped():
    assert metrics.course_key("Working at Height V-2.0 (flop)") == "working at height"


@pytest.mark.parametrize("a, b", [
    ("Food Hygiene Level 2 (2.0)", "Food Hygiene Level 3"),
    ("Food Hygiene Level 3", "Food Hygiene"),
    ("Ladder Safety", "Ladder Inspection"),
])
def test_genuinely_different_courses_stay_separate(a, b):
    assert metrics.course_key(a) != metrics.course_key(b)


def _row(platform, campaign, spend, revenue, conversions=1.0):
    return {
        "platform": platform, "campaign": campaign, "level": "campaign", "date": "2026-08-31",
        "spend": spend, "conversion_value": revenue, "conversions": conversions,
        "clicks": 10, "impressions": 100, "keyword": None, "ad_set": None, "ad": None,
        "currency": "GBP", "reach": None, "result_type": None,
    }


def _brand(margin=0.85, model="transactional", target_cpa=None):
    return {"business_model": model, "currency": "GBP", "margin_pct": margin,
            "target_roas": None, "target_cpa": target_cpa}


def _view(rows, brand=None, **kw):
    return {c["course"]: c for c in metrics.course_platform_view(metrics.rows_to_df(rows), brand or _brand(), **kw)}


def test_shift_when_one_platform_is_much_better():
    view = _view([_row("google", "First Aid PMX", 1000, 2350), _row("microsoft", "First Aid CAT", 325, 312)])
    assert view["first aid"]["verdict"] == "shift"
    assert "toward google" in view["first aid"]["detail"]


def test_shift_beats_below_break_even_when_a_better_platform_exists():
    # Both under a 2.5x floor (40% margin), but google is 2.4x better — moving
    # spend is more actionable than "cut".
    view = _view([_row("google", "First Aid PMX", 1000, 2350), _row("microsoft", "First Aid CAT", 325, 312)],
                 brand=_brand(margin=0.4))
    assert view["first aid"]["verdict"] == "shift"
    assert "40% margin" in view["first aid"]["detail"]


def test_below_break_even_names_its_margin_assumption():
    view = _view([_row("microsoft", "WFA", 325, 170), _row("google", "Other", 500, 5000)])
    assert view["wfa"]["verdict"] == "below_break_even"
    assert "85% margin" in view["wfa"]["detail"]


def test_zero_revenue_everywhere_is_below_break_even():
    view = _view([_row("google", "PPE", 85, 0, 0), _row("microsoft", "PPE", 37, 0, 0)], min_spend=30)
    assert view["ppe"]["verdict"] == "below_break_even"


def test_expand_mentions_a_thin_test_instead_of_calling_it_untried():
    view = _view([_row("microsoft", "AML (Relanuch 14 July)", 139, 902),
                  _row("google", "AML Training (L 5 March)", 35, 37)])
    aml = view["aml"]
    assert aml["verdict"] == "expand"
    assert "only £35.00 on google so far" in aml["detail"]


def test_below_min_spend_everywhere_is_insufficient_data():
    view = _view([_row("google", "HAVS", 5, 0, 0), _row("microsoft", "HAVS", 47, 157)])
    assert view["havs"]["verdict"] == "insufficient_data"


def test_zero_spend_courses_are_omitted():
    assert "paused" not in _view([_row("google", "Paused", 0, 0, 0), _row("google", "Live", 100, 500)])


def test_lead_gen_judges_by_cpa_against_target():
    brand = _brand(margin=None, model="lead_gen", target_cpa=30)
    view = _view([_row("google", "Mental Health", 300, 0, conversions=20),      # £15 CPA
                  _row("microsoft", "Mental Health", 300, 0, conversions=5)],   # £60 CPA
                 brand=brand)
    assert view["mental health"]["verdict"] == "shift"
    assert "£15.00 CPA on google" in view["mental health"]["detail"]


def test_actionable_verdicts_sort_first():
    view = metrics.course_platform_view(metrics.rows_to_df([
        _row("google", "Healthy", 500, 2000), _row("microsoft", "Healthy", 500, 2100),
        _row("google", "First Aid PMX", 1000, 2350), _row("microsoft", "First Aid CAT", 325, 312),
    ]), _brand())
    assert view[0]["verdict"] == "shift"
