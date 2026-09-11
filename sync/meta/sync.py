r"""
Meta Marketing API sync — pulls ad-level daily performance (including Reach,
which enables real frequency-based creative-fatigue detection) into the same
SQLite store and canonical schema manual CSV import uses.

Runs fine in the MAIN app venv — facebook-business doesn't have the
protobuf conflict google-ads does (verified: both installed together
without issue). No separate venv needed here.

UNTESTED against a live ad account: written against the documented
facebook-business SDK shape, but there's no real System User token to run
it against yet. Expect to debug the first real run together once you have
credentials.

Setup (one-time, do this before the first sync):
  1. developers.facebook.com > Create App (type: Business) > add the
     "Marketing API" product. Note the App ID and App Secret
     (Settings > Basic).
  2. Business Settings > Users > System Users > create one > Generate Token,
     scoped to "ads_read" on the ad account you want to sync. System User
     tokens don't expire the way personal user tokens do — the right choice
     for a scheduled job.
  3. Find the ad account id in Ads Manager (format act_XXXXXXXXXXXXX).
  4. Copy meta-config.yaml.example to meta-config.yaml and fill in
     app_id/app_secret/access_token/ad_account_id.
  5. Run `python sync.py --list-action-types --config meta-config.yaml`
     to see which action_type strings this account actually reports, and
     put the right one in conversion_action_type before the first real sync
     (don't guess — pixel/event setup varies per account).

Usage:
    ..\..\.venv\Scripts\python.exe sync.py --brand "Rozen Outfits" --days 30
    ..\..\.venv\Scripts\python.exe sync.py --list-action-types
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import db  # noqa: E402

FIELDS = [
    "campaign_name", "adset_name", "ad_name",
    "spend", "impressions", "reach", "clicks",
    "actions", "action_values", "account_currency", "date_start",
]


def _load_config(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _init_api(cfg: dict):
    from facebook_business.api import FacebookAdsApi
    FacebookAdsApi.init(cfg["app_id"], cfg["app_secret"], cfg["access_token"])


def _fetch_insights(cfg: dict, days: int):
    from facebook_business.adobjects.adaccount import AdAccount
    account = AdAccount(cfg["ad_account_id"])
    return account.get_insights(
        fields=FIELDS,
        params={
            "time_range": {"since": _n_days_ago(days), "until": _today()},
            "time_increment": 1,   # one row per ad per day
            "level": "ad",
            "limit": 500,
        },
    )


def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()


def _n_days_ago(n: int) -> str:
    import datetime
    return (datetime.date.today() - datetime.timedelta(days=n)).isoformat()


def list_action_types(cfg: dict, days: int = 7):
    """Debug helper: shows every action_type this account actually reports,
    so conversion_action_type in the config is a real choice, not a guess."""
    _init_api(cfg)
    insights = _fetch_insights(cfg, days)
    seen = set()
    for entry in insights:
        for a in entry.get("actions", []):
            seen.add(a["action_type"])
    if not seen:
        print(f"No actions found in the last {days} days — try a longer window or check the account has conversions.")
        return
    print("action_type values seen on this account:")
    for a in sorted(seen):
        print(f"  - {a}")
    print("\nPick the one that represents this brand's real conversion event "
          "and set it as conversion_action_type in meta-config.yaml.")


def build_rows(insights, conversion_action_type: str) -> list[dict]:
    rows = []
    for entry in insights:
        actions = {a["action_type"]: float(a["value"]) for a in entry.get("actions", [])}
        action_values = {a["action_type"]: float(a["value"]) for a in entry.get("action_values", [])}
        reach = entry.get("reach")
        rows.append({
            "date": entry["date_start"],
            "level": "ad",
            "campaign": entry.get("campaign_name"),
            "ad_set": entry.get("adset_name"),
            "ad": entry.get("ad_name"),
            "keyword": None,
            "spend": float(entry.get("spend", 0) or 0),
            "impressions": float(entry.get("impressions", 0) or 0),
            "clicks": float(entry.get("clicks", 0) or 0),
            "conversions": actions.get(conversion_action_type, 0.0),
            "conversion_value": action_values.get(conversion_action_type, 0.0),
            "reach": float(reach) if reach else None,
            "currency": entry.get("account_currency"),
            "result_type": conversion_action_type,
            "platform": "meta",
        })
    return rows


def run_sync(brand_name: str, days: int, cfg: dict):
    brand = db.get_brand_by_name(brand_name)
    if brand is None:
        print(f"No brand named '{brand_name}' in the dashboard yet — create it there first.")
        sys.exit(1)
    brand = dict(brand)

    if not cfg.get("conversion_action_type") or cfg["conversion_action_type"].startswith("INSERT_"):
        print("conversion_action_type isn't set in meta-config.yaml. Run:\n"
              "  python sync.py --list-action-types\n"
              "first to see what this account actually reports, then set it.")
        sys.exit(1)

    _init_api(cfg)
    insights = _fetch_insights(cfg, days)
    rows = build_rows(insights, cfg["conversion_action_type"])

    if not rows:
        print("No rows returned — check the ad account id and date range, and that "
              "the account has spend in this window.")
        return

    dates = [r["date"] for r in rows]
    date_start, date_end = min(dates), max(dates)

    db.delete_rows_in_range(brand["id"], "meta", date_start, date_end)
    import_id = db.create_import(
        brand_id=brand["id"], platform="meta", level="ad",
        filename=f"API sync {date_start}..{date_end}",
        date_start=date_start, date_end=date_end,
        row_count=len(rows), status="ok", unmapped_columns=[],
        notes=f"Meta Marketing API sync, {cfg['ad_account_id']}, last {days} days, "
              f"conversion_action_type={cfg['conversion_action_type']}",
    )
    db.insert_rows(import_id, brand["id"], rows)
    print(f"Synced {len(rows)} rows for '{brand_name}' ({date_start} -> {date_end}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Meta Ads performance into the PPC dashboard.")
    parser.add_argument("--brand", help="Brand name exactly as created in the dashboard")
    parser.add_argument("--days", type=int, default=30, help="Trailing window to pull/overwrite (default 30)")
    parser.add_argument("--config", default=str(HERE / "meta-config.yaml"), help="Path to meta-config.yaml")
    parser.add_argument("--list-action-types", action="store_true",
                         help="Debug: list this account's real action_type values instead of syncing")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f"Missing {cfg_path}. Copy meta-config.yaml.example to meta-config.yaml and fill it in first.")
        sys.exit(1)
    cfg = _load_config(cfg_path)

    if args.list_action_types:
        list_action_types(cfg)
    elif args.brand:
        run_sync(args.brand, args.days, cfg)
    else:
        parser.error("--brand is required unless using --list-action-types")
