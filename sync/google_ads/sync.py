"""
Google Ads API sync — pulls ad_group-level performance into the same
SQLite store and canonical schema the manual CSV import uses, so the
dashboard/insights code never has to know which path a row came from.

Run with the SEPARATE .venv-googleads environment (google-ads' own
dependencies conflict with the main app's Streamlit/protobuf versions —
that's why this lives in its own venv rather than inside src/).

UNTESTED against a live account: written against the documented
google-ads-python client shape, but there's no real Developer Token /
OAuth credentials to run it against yet. Expect to debug the first real
run together once you have credentials — this is the honest state, not
a guess dressed up as working code.

Setup (one-time, do this before the first sync):
  1. Apply for a Developer Token: Google Ads UI (your manager/MCC account)
     > Tools & Settings > API Center > Apply for Basic access.
  2. Google Cloud Console: create a project, enable "Google Ads API",
     create an OAuth 2.0 Client ID (type "Desktop app"), download its JSON
     as client_secret.json into this folder.
  3. Run generate_refresh_token.py (this folder, .venv-googleads) once —
     it opens your browser, you log in, it prints a refresh_token.
  4. Copy google-ads.yaml.example to google-ads.yaml and fill in all five
     values (developer_token, client_id, client_secret, refresh_token,
     login_customer_id).

Usage:
    ..\\..\\.venv-googleads\\Scripts\\python.exe sync.py \\
        --brand "High Skills Training" --customer-id 1234567890 --days 30
"""

from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import db  # noqa: E402

GAQL_QUERY = """
    SELECT
      segments.date,
      campaign.name,
      ad_group.name,
      customer.currency_code,
      metrics.cost_micros,
      metrics.impressions,
      metrics.clicks,
      metrics.conversions,
      metrics.conversions_value
    FROM ad_group
    WHERE segments.date DURING LAST_{days}_DAYS
      AND ad_group.status != 'REMOVED'
"""


def run_sync(brand_name: str, customer_id: str, days: int, config_path: pathlib.Path):
    from google.ads.googleads.client import GoogleAdsClient  # local import: only in this venv
    from google.ads.googleads.errors import GoogleAdsException

    brand = db.get_brand_by_name(brand_name)
    if brand is None:
        print(f"No brand named '{brand_name}' in the dashboard yet — create it there first.")
        sys.exit(1)
    brand = dict(brand)

    client = GoogleAdsClient.load_from_storage(path=str(config_path))
    ga_service = client.get_service("GoogleAdsService")
    query = GAQL_QUERY.format(days=days)

    rows: list[dict] = []
    dates: list[str] = []
    try:
        response = ga_service.search(customer_id=customer_id.replace("-", ""), query=query)
        for gr in response:
            spend = gr.metrics.cost_micros / 1_000_000
            row = {
                "date": gr.segments.date,
                "level": "ad_set",
                "campaign": gr.campaign.name,
                "ad_set": gr.ad_group.name,
                "ad": None,
                "keyword": None,
                "spend": spend,
                "impressions": gr.metrics.impressions,
                "clicks": gr.metrics.clicks,
                "conversions": gr.metrics.conversions,
                "conversion_value": gr.metrics.conversions_value,
                "currency": gr.customer.currency_code,
                "result_type": None,
                "platform": "google",
            }
            rows.append(row)
            dates.append(row["date"])
    except GoogleAdsException as ex:
        print(f"Google Ads API rejected the request (customer {customer_id}):")
        for error in ex.failure.errors:
            print(f"  - {error.message}")
        sys.exit(1)

    if not rows:
        print("No rows returned — check the customer id and date range, and that the "
              "account has spend in this window.")
        return

    date_start, date_end = min(dates), max(dates)

    # A sync run cleanly overwrites its window rather than accumulating
    # duplicates on every scheduled run (see db.delete_rows_in_range).
    db.delete_rows_in_range(brand["id"], "google", date_start, date_end)
    import_id = db.create_import(
        brand_id=brand["id"], platform="google", level="ad_set",
        filename=f"API sync {date_start}..{date_end}",
        date_start=date_start, date_end=date_end,
        row_count=len(rows), status="ok", unmapped_columns=[],
        notes=f"Google Ads API sync, customer {customer_id}, last {days} days",
    )
    db.insert_rows(import_id, brand["id"], rows)
    print(f"Synced {len(rows)} rows for '{brand_name}' ({date_start} -> {date_end}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Google Ads performance into the PPC dashboard.")
    parser.add_argument("--brand", required=True, help="Brand name exactly as created in the dashboard")
    parser.add_argument("--customer-id", required=True, help="Google Ads account id (digits, no dashes)")
    parser.add_argument("--days", type=int, default=30, help="Trailing window to pull/overwrite (default 30)")
    parser.add_argument("--config", default=str(HERE / "google-ads.yaml"), help="Path to google-ads.yaml")
    args = parser.parse_args()

    cfg = pathlib.Path(args.config)
    if not cfg.exists():
        print(f"Missing {cfg}. Copy google-ads.yaml.example to google-ads.yaml and fill it in first.")
        sys.exit(1)

    run_sync(args.brand, args.customer_id, args.days, cfg)
