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
ANTHROPIC_API_KEY=sk-ant-...  # optional; enables the AI Analyst tab
```

`DATABASE_URL` is required — the app stores data in Postgres (e.g. Supabase)
rather than a local file.

## Running locally

```bash
streamlit run app.py
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Tests cover the pure logic (import parsing/platform detection, the
statistical checks, the digest, course grouping, DB connection pooling)
and need no database.

## Syncing ad platform data

- `sync/google_ads/` — pulls reports via the Google Ads API. Configure from
  `google-ads.yaml.example`.
- `sync/meta/` — pulls reports via the Meta Business API. Configure from
  `meta-config.yaml.example`.

## AI analyst

The **🤖 AI Analyst** tab lets you ask questions in plain English ("which courses
should I move budget between?", "why did ROAS change vs last month?") and
generates a weekly briefing. Claude (`claude-opus-5`, adaptive thinking) answers
by calling this app's own analyses as tools — account overview, campaigns,
course view, insights, keyword waste, digest — so every number it quotes comes
from your imported data. It only reads data; it has no access to the ad accounts.

Enable it by setting `ANTHROPIC_API_KEY` (Streamlit Cloud: app settings → Secrets;
locally: `.env`). Without a key the tab shows setup instructions instead. Set
`AI_ANALYST_MODEL` to use a different Claude model. Code: `src/ai_analyst.py`.

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
