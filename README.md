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

## Deployment

`render.yaml` defines an alternate deployment on Render; the primary path is
Streamlit Community Cloud. See the comments in `render.yaml` for details.
