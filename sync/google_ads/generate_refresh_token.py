r"""
One-time local OAuth flow to get a Google Ads API refresh token.

Run this with the SEPARATE .venv-googleads environment (not the main app
venv — google-ads' dependencies conflict with Streamlit's, see the project
notes). This opens your default browser, asks you to log into the Google
account that administers your Ads account, and prints a refresh_token to
YOUR OWN terminal — nothing is sent anywhere else. Paste that value into
google-ads.yaml yourself; it is never typed into chat or seen by anything
but this script and your own config file.

Prerequisites (do these first, in Google Cloud Console):
  1. Create a project, enable the "Google Ads API".
  2. APIs & Services > Credentials > Create OAuth client ID > type "Desktop app".
  3. Download its JSON and save it in this folder as client_secret.json.

Usage:
    ..\..\.venv-googleads\Scripts\python.exe generate_refresh_token.py
"""

from __future__ import annotations

import pathlib
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/adwords"]
HERE = pathlib.Path(__file__).resolve().parent
CLIENT_SECRET_FILE = HERE / "client_secret.json"


def main():
    if not CLIENT_SECRET_FILE.exists():
        print(f"Missing {CLIENT_SECRET_FILE}.\n"
              "Download your OAuth client's JSON from Google Cloud Console "
              "(APIs & Services > Credentials) and save it there as client_secret.json.")
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), scopes=SCOPES)
    # access_type=offline + prompt=consent forces a refresh_token back even
    # on a re-consent; without it a second run can return none.
    credentials = flow.run_local_server(
        access_type="offline", prompt="consent", port=0
    )

    print("\n--- Success ---")
    print("Add this to your google-ads.yaml as refresh_token:\n")
    print(credentials.refresh_token)
    print("\n(This value is only printed here, in your own terminal.)")


if __name__ == "__main__":
    main()
