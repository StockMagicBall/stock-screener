"""
GitHub-backed persistence for the trade journal.

Streamlit Cloud's own filesystem is ephemeral -- it resets on redeploy, so
anything written directly to disk from within the app would vanish the next
time the repo updates. To make journal entries actually persist, this reads
and writes trade_journal.csv directly to the GitHub repo via GitHub's REST
API, using a personal access token stored in Streamlit's secrets manager
(Settings -> Secrets on share.streamlit.io -- NOT the same as GitHub Actions
secrets, which only that workflow can see).

Required Streamlit secrets (set in the Streamlit Cloud app's own Settings ->
Secrets page, as TOML):
    GITHUB_TOKEN = "ghp_..."          # personal access token, repo scope
    GITHUB_REPO  = "username/reponame"
"""

import base64
import json

import pandas as pd
import requests
import streamlit as st

JOURNAL_PATH = "trade_journal.csv"
JOURNAL_COLUMNS = [
    "ticker", "entry_date", "entry_price", "units", "total_cost",
    "stop_loss", "sell_trigger", "status", "exit_date", "exit_price",
    "realized_pnl", "realized_pnl_pct", "notes",
]


def _api_url(path: str) -> str:
    repo = st.secrets["GITHUB_REPO"]
    return f"https://api.github.com/repos/{repo}/contents/{path}"


def _headers() -> dict:
    token = st.secrets["GITHUB_TOKEN"]
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def load_journal() -> tuple[pd.DataFrame, str | None]:
    """Returns (dataframe, sha). sha is None if the file doesn't exist yet."""
    resp = requests.get(_api_url(JOURNAL_PATH), headers=_headers())
    if resp.status_code == 404:
        return pd.DataFrame(columns=JOURNAL_COLUMNS), None
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    from io import StringIO
    df = pd.read_csv(StringIO(content)) if content.strip() else pd.DataFrame(columns=JOURNAL_COLUMNS)
    for col in JOURNAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[JOURNAL_COLUMNS], data["sha"]


def save_journal(df: pd.DataFrame, sha: str | None, message: str) -> str:
    """Writes the dataframe back to GitHub. Returns the new sha."""
    csv_content = df.to_csv(index=False)
    encoded = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
    body = {"message": message, "content": encoded, "branch": "main"}
    if sha:
        body["sha"] = sha
    resp = requests.put(_api_url(JOURNAL_PATH), headers=_headers(), data=json.dumps(body))
    resp.raise_for_status()
    return resp.json()["content"]["sha"]
