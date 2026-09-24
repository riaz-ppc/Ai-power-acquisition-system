from datetime import date, timedelta

from src import digest, metrics


def test_money_formats_known_and_unknown_currencies():
    assert digest.money(1234.5, "GBP") == "£1,234.50"
    assert digest.money(10, "USD") == "$10.00"
    assert digest.money(10, "BDT") == "BDT 10.00"


def test_formatters_render_missing_values_as_dash():
    for fmt in (digest.money, digest.ratio, digest.pct):
        assert fmt(None) == "—"
        assert fmt(float("nan")) == "—"


def test_ratio_and_pct():
    assert digest.ratio(1.694) == "1.69x"
    assert digest.pct(12.34) == "+12.3%"
    assert digest.pct(-5) == "-5.0%"


def _campaign_rows():
    # Two campaigns over 14 days: one well under a 2.5x break-even ROAS
    # (40% margin), one comfortably above — enough for a ranked digest.
    start = date(2026, 9, 1)
    rows = []
    for i in range(14):
        d = (start + timedelta(days=i)).isoformat()
        for camp, spend, value in (("Loser", 100, 80), ("Winner", 100, 600)):
            rows.append({
                "platform": "google", "campaign": camp, "level": "campaign", "date": d,
                "spend": spend, "clicks": 50, "conversions": 5, "impressions": 1000,
                "conversion_value": value, "keyword": None, "ad_set": None, "ad": None,
                "currency": "GBP", "reach": None, "result_type": None,
            })
    return metrics.rows_to_df(rows)


BRAND = {
    "name": "Test", "business_model": "transactional", "currency": "GBP",
    "margin_pct": 0.4, "aov": None, "ltv": None, "target_roas": 3.0,
    "target_cpa": None, "target_payback_days": None, "country": None,
}


def test_digest_items_are_sorted_by_priority_then_impact():
    items = digest.compute_digest_items(_campaign_rows(), BRAND)
    assert items, "a campaign losing money should produce at least one item"
    keys = [(i["priority"], -i["impact"]) for i in items]
    assert keys == sorted(keys)


def test_digest_flags_the_losing_campaign_as_critical():
    items = digest.compute_digest_items(_campaign_rows(), BRAND)
    critical_titles = [i["title"] for i in items if i["priority"] == 0]
    assert any("Loser" in t for t in critical_titles)
    assert not any("Winner" in t and "below break-even" in t for t in critical_titles)


def test_digest_includes_cross_platform_course_shift():
    rows = []
    for i in range(7):
        d = (date(2026, 9, 1) + timedelta(days=i)).isoformat()
        for platform, campaign, value in (("google", "First Aid PMX", 300), ("microsoft", "First Aid CAT", 90)):
            rows.append({
                "platform": platform, "campaign": campaign, "level": "campaign", "date": d,
                "spend": 100, "clicks": 50, "conversions": 5, "impressions": 1000,
                "conversion_value": value, "keyword": None, "ad_set": None, "ad": None,
                "currency": "GBP", "reach": None, "result_type": None,
            })
    items = digest.compute_digest_items(metrics.rows_to_df(rows), BRAND)
    shift = [i for i in items if i["title"].startswith("First Aid: shift budget")]
    assert shift and "toward google" in shift[0]["detail"]
