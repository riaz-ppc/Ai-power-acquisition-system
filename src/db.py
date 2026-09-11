"""
Postgres persistence (originally SQLite; migrated when the app moved to a
hosted deployment that needed real persistent storage on a free tier —
see the project's deployment notes). Schema is plain relational SQL on
purpose — no ORM.

Requires DATABASE_URL (a standard postgres:// or postgresql:// connection
string — Supabase, Render Postgres, Neon, or a local Postgres all provide
one the same way). Local dev picks it up from a .env file if present
(never committed — see .gitignore); a hosted deployment sets it as a
platform secret/environment variable. There is deliberately no silent
fallback to a local file: a missing DATABASE_URL is a clear startup
error, not a quietly different (and easy to lose track of) data store.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()  # no-op if there's no local .env file (e.g. on a host that sets real env vars)

DATABASE_URL = os.environ.get("DATABASE_URL")

SCHEMA = """
CREATE TABLE IF NOT EXISTS brands (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    business_model TEXT NOT NULL CHECK (business_model IN ('transactional','lead_gen')),
    conversion_type TEXT NOT NULL,          -- purchase / lead / enrollment / ...
    currency TEXT NOT NULL DEFAULT 'USD',
    margin_pct REAL,                        -- contribution margin, 0-1, for iROAS floor
    aov REAL,                               -- average order value (transactional brands)
    ltv REAL,                               -- customer/student lifetime value
    target_roas REAL,
    target_cpa REAL,
    target_payback_days REAL,
    country TEXT,                           -- primary market, for seasonal/market-calendar context
    created_at TEXT DEFAULT (now()::text)
);

CREATE TABLE IF NOT EXISTS imports (
    id SERIAL PRIMARY KEY,
    brand_id INTEGER NOT NULL REFERENCES brands(id),
    platform TEXT NOT NULL,
    level TEXT NOT NULL,
    filename TEXT NOT NULL,
    imported_at TEXT DEFAULT (now()::text),
    date_start TEXT,
    date_end TEXT,
    row_count INTEGER,
    status TEXT NOT NULL DEFAULT 'ok',      -- ok / partial / failed
    unmapped_columns TEXT,                  -- json list, for "no silent data loss"
    notes TEXT
);

CREATE TABLE IF NOT EXISTS performance_rows (
    id SERIAL PRIMARY KEY,
    import_id INTEGER NOT NULL REFERENCES imports(id),
    brand_id INTEGER NOT NULL REFERENCES brands(id),
    date TEXT NOT NULL,
    platform TEXT NOT NULL,
    level TEXT NOT NULL,
    campaign TEXT,
    ad_set TEXT,
    ad TEXT,
    keyword TEXT,
    spend REAL DEFAULT 0,
    impressions REAL DEFAULT 0,
    clicks REAL DEFAULT 0,
    conversions REAL DEFAULT 0,
    conversion_value REAL DEFAULT 0,
    reach REAL,                             -- Meta only; enables real frequency-based fatigue detection
    currency TEXT,
    result_type TEXT
);

CREATE INDEX IF NOT EXISTS idx_rows_brand_date ON performance_rows(brand_id, date);
CREATE INDEX IF NOT EXISTS idx_rows_platform ON performance_rows(platform);

CREATE TABLE IF NOT EXISTS tests (
    id SERIAL PRIMARY KEY,
    brand_id INTEGER NOT NULL REFERENCES brands(id),
    name TEXT NOT NULL,
    hypothesis TEXT,
    variant_a_label TEXT,
    variant_a_campaigns TEXT,   -- json list of campaign names
    variant_b_label TEXT,
    variant_b_campaigns TEXT,
    started_at TEXT,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'running'  -- running / concluded
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    import_id INTEGER NOT NULL REFERENCES imports(id),
    brand_id INTEGER NOT NULL REFERENCES brands(id),
    order_date TEXT NOT NULL,
    order_id TEXT,
    amount REAL NOT NULL,
    raw_campaign TEXT,
    matched_campaign TEXT,          -- confirmed by the viewer, never auto-trusted
    source TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_brand_date ON orders(brand_id, order_date);

-- Idempotent: columns added after a table already existed elsewhere don't
-- need a separate migrations list in Postgres, IF NOT EXISTS covers it.
ALTER TABLE performance_rows ADD COLUMN IF NOT EXISTS reach REAL;
ALTER TABLE brands ADD COLUMN IF NOT EXISTS country TEXT;
"""


@contextmanager
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL isn't set. Local dev: put it in a .env file in the project "
            "root (never committed). Deployed: set it as a platform secret/env var."
        )
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)


# ---------------------------------------------------------------- brands --

def create_brand(**kwargs) -> int:
    fields = ["name", "business_model", "conversion_type", "currency", "margin_pct",
              "aov", "ltv", "target_roas", "target_cpa", "target_payback_days", "country"]
    cols = [f for f in fields if f in kwargs]
    placeholders = ",".join(["%s"] * len(cols))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO brands ({','.join(cols)}) VALUES ({placeholders}) RETURNING id",
                [kwargs[c] for c in cols],
            )
            return cur.fetchone()["id"]


def update_brand(brand_id: int, **kwargs):
    fields = ["business_model", "conversion_type", "currency", "margin_pct",
              "aov", "ltv", "target_roas", "target_cpa", "target_payback_days", "country"]
    cols = [f for f in fields if f in kwargs]
    if not cols:
        return
    set_clause = ",".join(f"{c}=%s" for c in cols)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE brands SET {set_clause} WHERE id=%s",
                [kwargs[c] for c in cols] + [brand_id],
            )


def list_brands() -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM brands ORDER BY name")
            return cur.fetchall()


def get_brand(brand_id: int) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM brands WHERE id=%s", (brand_id,))
            return cur.fetchone()


def get_brand_by_name(name: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM brands WHERE name=%s", (name,))
            return cur.fetchone()


# --------------------------------------------------------------- imports --

def find_overlapping_import(brand_id: int, platform: str, date_start: str, date_end: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM imports
                   WHERE brand_id=%s AND platform=%s
                     AND NOT (date_end < %s OR date_start > %s)""",
                (brand_id, platform, date_start, date_end),
            )
            return cur.fetchall()


def create_import(brand_id, platform, level, filename, date_start, date_end,
                   row_count, status, unmapped_columns: list[str], notes: str = "") -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO imports
                   (brand_id, platform, level, filename, date_start, date_end,
                    row_count, status, unmapped_columns, notes)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (brand_id, platform, level, filename, date_start, date_end,
                 row_count, status, json.dumps(unmapped_columns), notes),
            )
            return cur.fetchone()["id"]


def delete_import(import_id: int, brand_id: int) -> bool:
    """Scoped to brand_id on purpose — an import id typed into the wrong
    brand's Settings tab must refuse, not silently delete another brand's
    data. Returns False (nothing deleted) when the import doesn't belong
    to this brand."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT brand_id FROM imports WHERE id=%s", (import_id,))
            owner = cur.fetchone()
            if owner is None or owner["brand_id"] != brand_id:
                return False
            cur.execute("DELETE FROM performance_rows WHERE import_id=%s", (import_id,))
            cur.execute("DELETE FROM imports WHERE id=%s", (import_id,))
            return True


def delete_rows_in_range(brand_id: int, platform: str, date_start: str, date_end: str):
    """Used by API sync (not manual upload): a scheduled sync re-pulls a
    trailing window every run (ad platforms revise recent conversions for
    a few days after the fact), so it cleanly overwrites that window
    instead of accumulating duplicate rows like a one-off CSV import would."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM performance_rows WHERE brand_id=%s AND platform=%s AND date>=%s AND date<=%s",
                (brand_id, platform, date_start, date_end),
            )


def rename_campaign(brand_id: int, platform: str, old_name: str, new_name: str) -> int:
    """Merges an old campaign name's historical rows under a new/canonical
    name — for when the SAME real-world campaign got renamed in the ad
    platform (a note added, a typo fixed) rather than actually replaced,
    so trend/forecast/insights see one continuous campaign instead of two
    unrelated ones. Scoped to one platform since the same shorthand can
    legitimately exist on two different platforms. Returns rows affected.
    Never called silently — the caller confirms with the viewer first."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE performance_rows SET campaign=%s WHERE brand_id=%s AND platform=%s AND campaign=%s",
                (new_name, brand_id, platform, old_name),
            )
            return cur.rowcount


def list_imports(brand_id: int | None = None) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            if brand_id:
                cur.execute(
                    "SELECT * FROM imports WHERE brand_id=%s ORDER BY imported_at DESC",
                    (brand_id,),
                )
            else:
                cur.execute("SELECT * FROM imports ORDER BY imported_at DESC")
            return cur.fetchall()


def insert_rows(import_id: int, brand_id: int, rows: list[dict]):
    if not rows:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """INSERT INTO performance_rows
                   (import_id, brand_id, date, platform, level, campaign, ad_set, ad,
                    keyword, spend, impressions, clicks, conversions, conversion_value,
                    reach, currency, result_type)
                   VALUES (%(import_id)s, %(brand_id)s, %(date)s, %(platform)s, %(level)s, %(campaign)s,
                           %(ad_set)s, %(ad)s, %(keyword)s, %(spend)s, %(impressions)s, %(clicks)s,
                           %(conversions)s, %(conversion_value)s, %(reach)s, %(currency)s, %(result_type)s)""",
                [{**r, "import_id": import_id, "brand_id": brand_id, "reach": r.get("reach")} for r in rows],
            )


def rows_for_brand(brand_id: int, start: str | None = None, end: str | None = None):
    q = "SELECT * FROM performance_rows WHERE brand_id=%s"
    params: list = [brand_id]
    if start:
        q += " AND date >= %s"
        params.append(start)
    if end:
        q += " AND date <= %s"
        params.append(end)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchall()


# ------------------------------------------------------------------ tests --

def create_test(**kwargs) -> int:
    fields = ["brand_id", "name", "hypothesis", "variant_a_label", "variant_a_campaigns",
              "variant_b_label", "variant_b_campaigns", "started_at", "ended_at", "status"]
    cols = [f for f in fields if f in kwargs]
    placeholders = ",".join(["%s"] * len(cols))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO tests ({','.join(cols)}) VALUES ({placeholders}) RETURNING id",
                [kwargs[c] for c in cols],
            )
            return cur.fetchone()["id"]


def list_tests(brand_id: int) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM tests WHERE brand_id=%s ORDER BY started_at DESC", (brand_id,))
            return cur.fetchall()


# ----------------------------------------------------------------- orders --

def insert_orders(import_id: int, brand_id: int, rows: list[dict]):
    if not rows:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """INSERT INTO orders (import_id, brand_id, order_date, order_id, amount, raw_campaign, source)
                   VALUES (%(import_id)s, %(brand_id)s, %(order_date)s, %(order_id)s, %(amount)s, %(campaign)s, %(source)s)""",
                [{**r, "import_id": import_id, "brand_id": brand_id} for r in rows],
            )


def orders_for_brand(brand_id: int, start: str | None = None, end: str | None = None) -> list[dict]:
    q = "SELECT * FROM orders WHERE brand_id=%s"
    params: list = [brand_id]
    if start:
        q += " AND order_date >= %s"
        params.append(start)
    if end:
        q += " AND order_date <= %s"
        params.append(end)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchall()


def unmatched_raw_campaigns(brand_id: int) -> list[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT raw_campaign FROM orders WHERE brand_id=%s AND matched_campaign IS NULL AND raw_campaign IS NOT NULL",
                (brand_id,),
            )
            return [r["raw_campaign"] for r in cur.fetchall()]


def set_campaign_matches(brand_id: int, matches: dict[str, str | None]):
    """matches: {raw_campaign: matched_campaign_or_None}. None (or 'ignore')
    is stored as the literal string so it isn't re-suggested every time."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            for raw, matched in matches.items():
                cur.execute(
                    "UPDATE orders SET matched_campaign=%s WHERE brand_id=%s AND raw_campaign=%s",
                    (matched or "ignore", brand_id, raw),
                )
