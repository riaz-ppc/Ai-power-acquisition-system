import pytest

from src import metrics


def _kw_row(keyword, spend, clicks, conversions, platform="google"):
    return {
        "platform": platform, "campaign": "C1", "keyword": keyword, "level": "keyword",
        "date": "2026-09-15", "spend": spend, "clicks": clicks, "conversions": conversions,
        "impressions": clicks * 20, "conversion_value": conversions * 100,
        "ad_set": None, "ad": None, "currency": "GBP", "reach": None, "result_type": None,
    }


@pytest.fixture
def free_keywords_df():
    # "free" spread across three phrases: one clears the per-keyword bar
    # alone (20 clicks), two don't; one converting keyword sets the
    # account's baseline conversion rate at 5/30 = 16.67%.
    return metrics.rows_to_df([
        _kw_row("free online first aid course", 40, 20, 0),
        _kw_row("free fire safety training", 35, 18, 0),
        _kw_row("free haccp certificate", 25, 14, 0),
        _kw_row("level 3 haccp certification", 60, 30, 5),
    ])


def test_keyword_waste_flags_only_phrases_that_clear_the_bar_alone(free_keywords_df):
    waste = metrics.keyword_waste_candidates(free_keywords_df, min_clicks=15)
    assert [w["keyword"] for w in waste] == ["free online first aid course"]
    assert waste[0]["upper_bound_cvr_pct"] == 15.0  # rule of three: 3/20
    assert waste[0]["account_cvr_pct"] == 16.67


def test_keyword_word_waste_catches_a_word_spread_across_thin_phrases(free_keywords_df):
    words = metrics.keyword_word_waste(free_keywords_df, min_clicks=15)
    assert [w["word"] for w in words] == ["free"]
    assert words[0]["clicks"] == 52
    assert words[0]["keyword_count"] == 3
    assert words[0]["spend"] == 100.0


def test_keyword_word_waste_ignores_stopwords_and_converting_words(free_keywords_df):
    flagged = {w["word"] for w in metrics.keyword_word_waste(free_keywords_df, min_clicks=15)}
    assert "haccp" not in flagged       # appears in a converting keyword
    assert "online" not in flagged      # stopword
    assert "course" not in flagged      # stopword


def test_keyword_word_waste_needs_a_word_in_at_least_two_keywords():
    df = metrics.rows_to_df([
        _kw_row("unique wasteful phrase", 50, 40, 0),
        _kw_row("converting term", 50, 20, 5),
    ])
    assert metrics.keyword_word_waste(df, min_clicks=15) == []


def test_keyword_checks_ignore_meta_which_has_no_keyword_level():
    df = metrics.rows_to_df([_kw_row("free thing", 50, 40, 0, platform="meta"),
                             _kw_row("free stuff", 50, 40, 0, platform="meta")])
    assert metrics.keyword_waste_candidates(df) == []
    assert metrics.keyword_word_waste(df) == []


def test_z_test_no_data_is_never_significant():
    assert metrics.two_proportion_z_test(0, 0, 5, 100)["significant"] is False


def test_z_test_identical_rates_not_significant():
    result = metrics.two_proportion_z_test(10, 100, 10, 100)
    assert result["significant"] is False
    assert result["p_value"] == 1.0


def test_z_test_large_real_difference_is_significant():
    result = metrics.two_proportion_z_test(50, 1000, 100, 1000)
    assert result["significant"] is True
    assert result["z"] > 0  # B better than A


def test_z_test_flags_low_sample():
    assert metrics.two_proportion_z_test(1, 10, 5, 10)["low_sample_warning"] is True
    assert metrics.two_proportion_z_test(10, 100, 12, 100)["low_sample_warning"] is False
