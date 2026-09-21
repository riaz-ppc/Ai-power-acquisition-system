# PPC Intelligence

PPC Acquisition & Decision Intelligence — a Streamlit app for uploading and
normalizing ad platform exports (Google Ads, Meta, Microsoft/Bing Ads) into
one schema, then reporting spend, conversions, and ROI across campaigns.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows; source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Create a `.env` file in the project root with:

```
DATABASE_URL=postgresql://...
APP_PASSWORD=your-password   # optional; leave unset to run without a login screen
```

`DATABASE_URL` is required — the app stores data in Postgres (e.g. Supabase)
rather than a local file.

## Running locally

```bash
streamlit run app.py
```

## Syncing ad platform data

- `sync/google_ads/` — pulls reports via the Google Ads API. Configure from
  `google-ads.yaml.example`.
- `sync/meta/` — pulls reports via the Meta Business API. Configure from
  `meta-config.yaml.example`.

## Scheduled digest (Slack/email)

`scripts/send_digest.py` sends the same ranked digest the app's Digest tab
shows, without anyone needing to open the app. `.github/workflows/weekly-digest.yml`
runs it every Monday via GitHub Actions — set these as repo secrets
(Settings > Secrets and variables > Actions) to enable delivery:

- `DATABASE_URL` — required, same one the app uses
- `SLACK_WEBHOOK_URL` — optional, a Slack "Incoming Webhook" URL
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `DIGEST_EMAIL_TO` —
  optional, set all five together for email delivery

With none of the optional secrets set, the workflow still runs and prints
the digest into the run's log instead of sending it — safe to merge and
schedule before any delivery credentials exist. Run it manually any time
from the Actions tab ("Run workflow"), or locally with
`python scripts/send_digest.py [--brand NAME] [--dry-run]`.

## Deployment

`render.yaml` defines an alternate deployment on Render; the primary path is
Streamlit Community Cloud. See the comments in `render.yaml` for details.
