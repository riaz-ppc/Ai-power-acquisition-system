from io import StringIO

import pandas as pd

from src import normalize

# Mirrors the shape of a real stacked Google + Bing custom report (title
# rows, blank separators, two different header vocabularies, £ symbols in
# money cells, a non-numeric "shared" revenue cell) with synthetic values.
STACKED_EXPORT = """Brand Aug 2026 Google Campaign performance,,,,,,,,,
,,,,,,,,,
Campaign,Campaign state,Campaign type,Clicks,Avg. CPC,Cost,Conversions,Cost / conv.,Revenue,
Alpha PMX,Enabled,Performance Max,1000,0.80,800.00,40,20.00,"£2,000.50",
Beta Search,Enabled,Search,200,1.50,300.00,10,30.00,shared,
,,,,,,,,,
,,,,,,,,,
 Brand Aug 2026 Bing Campaign Performance,,,,,,,,,
Campaign name,Campaign type,Clicks,CTR,Avg. CPC,Spend,CPA,Conversions,Revenue,
Gamma Search,Search,500,0.40%,0.60,300.00,30.00,10,£450.00,
"""


def test_single_csv_round_trips_as_one_block():
    csv = "Day,Campaign,Cost\n2026-08-01,A,10\n"
    blocks = normalize.split_multi_table_csv(csv)
    assert len(blocks) == 1
    assert blocks[0].splitlines()[0] == "Day,Campaign,Cost"


def test_stacked_export_splits_into_tables_with_titles_stripped():
    blocks = normalize.split_multi_table_csv(STACKED_EXPORT)
    assert len(blocks) == 2
    assert blocks[0].splitlines()[0].startswith("Campaign,Campaign state")
    assert blocks[1].splitlines()[0].startswith("Campaign name,Campaign type")


def test_title_plus_date_subtitle_without_blank_separator_is_stripped():
    csv = "Report title\n1 August 2026 - 31 August 2026\nCampaign,Clicks,Cost\nA,1,2\n"
    blocks = normalize.split_multi_table_csv(csv)
    assert blocks[0].splitlines()[0] == "Campaign,Clicks,Cost"


def test_to_number_handles_symbols_placeholders_and_text():
    assert normalize._to_number("£2,334.60") == 2334.60
    assert normalize._to_number("-") == 0.0
    assert normalize._to_number("shared") == 0.0
    assert normalize._to_number(float("nan")) == 0.0
    assert normalize._to_number(None) == 0.0


def _tables():
    return [pd.read_csv(StringIO(b)) for b in normalize.split_multi_table_csv(STACKED_EXPORT)]


def test_stacked_export_detects_google_then_microsoft_at_campaign_level():
    google, bing = (normalize.normalize_upload(t, f"t{i}", "USD") for i, t in enumerate(_tables()))

    assert google.platform_id == "google"
    assert bing.platform_id == "microsoft"
    assert google.level == bing.level == "campaign"
    assert google.row_count == 2 and bing.row_count == 1


def test_period_summary_without_date_column_asks_for_a_period():
    google = normalize.normalize_upload(_tables()[0], "t", "USD")
    assert google.needs_period_date


def test_currency_symbol_in_cells_beats_brand_default():
    google = normalize.normalize_upload(_tables()[0], "t", "USD")
    assert google.currency == "GBP"
    assert not google.currency_assumed


def test_money_values_parsed_and_non_numeric_revenue_becomes_zero():
    rows = {r["campaign"]: r for r in normalize.normalize_upload(_tables()[0], "t", "USD").rows}
    assert rows["Alpha PMX"]["spend"] == 800.0
    assert rows["Alpha PMX"]["conversion_value"] == 2000.50
    assert rows["Beta Search"]["conversion_value"] == 0.0


def test_non_ad_export_is_not_misdetected_as_a_platform():
    order_log = pd.DataFrame({"Date": ["2026-08-01"], "Campaign": ["A"], "Customer": ["x"]})
    result = normalize.normalize_upload(order_log, "orders.csv", "USD")
    assert result.platform_id is None
    assert result.status == "failed"
