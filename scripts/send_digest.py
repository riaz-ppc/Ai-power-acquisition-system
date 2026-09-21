"""
Sends the same ranked digest the app's own Digest tab shows, to Slack
and/or email, on a schedule — the "push it to me, I don't want to have
to remember to check" gap identified against commercial PPC tools
(Optmyzr, AgencyAnalytics, NinjaCat all treat this as table stakes).

Runs standalone (no Streamlit), so it can be driven by any scheduler —
GitHub Actions (see .github/workflows/weekly-digest.yml), a Render cron
job, plain cron, Task Scheduler. Needs DATABASE_URL like the app does,
plus at least one delivery method:

  SLACK_WEBHOOK_URL   an "Incoming Webhook" URL from a Slack app
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, DIGEST_EMAIL_TO
                       standard SMTP send (Gmail, SES, etc.)

Neither configured: prints the digest to stdout instead of sending
anything — lets you see exactly what would go out before wiring up
real credentials, rather than the script silently doing nothing.

Usage:
    python scripts/send_digest.py                 # every brand
    python scripts/send_digest.py --brand "EST"    # one brand only
"""

from __future__ import annotations

import argparse
import os
import smtplib
import sys
from email.mime.text import MIMEText
from urllib.request import Request, urlopen
import json as _json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows' default console codepage can't print the emoji badges the
# digest uses (e.g. 🔴) — force UTF-8 stdout so `--dry-run`/no-credential
# output works the same on Windows as it does on the Linux CI runner this
# is meant to run on. No-op where stdout already is UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import db, metrics  # noqa: E402
from src.digest import compute_digest_items  # noqa: E402

BADGE_RANK = {"🔴": 0, "🟡": 1, "🟢": 2, "": 3}


def render_brand_digest(brand: dict) -> str | None:
    """Plain-text digest for one brand, same items/ranking as the app's
    Digest tab. Returns None if there's no data yet or nothing to flag —
    callers skip brands with nothing worth sending."""
    rows = db.rows_for_brand(brand["id"])
    if not rows:
        return None
    df = metrics.rows_to_df(rows)
    items = compute_digest_items(df, brand)
    if not items:
        return None

    lines = [f"*{brand['name']}* — weekly digest ({len(items)} item(s))"]
    for i, item in enumerate(items[:7], start=1):
        badge = item["badge"] or "⚪"
        lines.append(f"{i}. {badge} *{item['title']}*")
        lines.append(f"   {item['detail']}")
    return "\n".join(lines)


def send_slack(webhook_url: str, text: str):
    payload = _json.dumps({"text": text}).encode("utf-8")
    req = Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=15) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"Slack webhook returned HTTP {resp.status}")


def send_email(subject: str, body: str):
    host, port = os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "587"))
    user, password = os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"]
    to = os.environ["DIGEST_EMAIL_TO"]

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to

    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(user, [to], msg.as_string())


def main():
    parser = argparse.ArgumentParser(description="Send the weekly PPC digest to Slack/email.")
    parser.add_argument("--brand", help="Only send for this brand (default: every brand)")
    parser.add_argument("--dry-run", action="store_true", help="Print instead of sending, even if credentials are set")
    args = parser.parse_args()

    if args.brand:
        found = db.get_brand_by_name(args.brand)
        brands = [dict(found)] if found else []
    else:
        brands = [dict(b) for b in db.list_brands()]
    if not brands:
        print(f"No brand named {args.brand!r} found." if args.brand else "No brands exist yet.")
        return

    sections = []
    for brand in brands:
        text = render_brand_digest(brand)
        if text:
            sections.append(text)

    if not sections:
        print("Nothing to flag for any brand this run — no digest sent.")
        return

    full_text = "\n\n".join(sections)
    slack_url = os.environ.get("SLACK_WEBHOOK_URL")
    has_smtp = all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "DIGEST_EMAIL_TO"))

    if args.dry_run or (not slack_url and not has_smtp):
        print(full_text)
        if not args.dry_run:
            print("\n[No SLACK_WEBHOOK_URL or SMTP_* env vars set — printed instead of sending.]")
        return

    if slack_url:
        send_slack(slack_url, full_text)
        print(f"Sent to Slack ({len(sections)} brand(s) with items).")
    if has_smtp:
        send_email("PPC Intelligence — weekly digest", full_text)
        print(f"Sent by email to {os.environ['DIGEST_EMAIL_TO']} ({len(sections)} brand(s) with items).")


if __name__ == "__main__":
    main()
