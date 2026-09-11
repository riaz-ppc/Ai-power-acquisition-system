"""
SQLite persistence for the prototype. Schema is plain relational SQL on
purpose — no ORM — so moving to Postgres later is a driver swap plus a
`CREATE TABLE` translation, not a rewrite.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ppc_intelligence.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS brands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand_id INTEGER NOT NULL REFERENCES brands(id),
    platform TEXT NOT NULL,
    level TEXT NOT NULL,
    filename TEXT NOT NULL,
    imported_at TEXT DEFAULT (datetime('now')),
    date_start TEXT,
    date_end TEXT,
    row_count INTEGER,
    status TEXT NOT NULL DEFAULT 'ok',      -- ok / partial / failed
    unmapped_columns TEXT,                  -- json list, for "no silent data loss"
    notes TEXT
);

CREATE TABLE IF NOT EXISTS performance_rows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
"""


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Lightweight migrations for columns added after a table already existed on
# disk — CREATE TABLE IF NOT EXISTS doesn't alter an existing table, so new
# columns need an explicit, idempotent ADD COLUMN here.
_MIGRATIONS = [
    ("performance_rows", "reach", "ALTER TABLE performance_rows ADD COLUMN reach REAL"),
]


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        existing_tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        for table, column, ddl in _MIGRATIONS:
            if table not in existing_tables:
                continue
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if column not in cols:
                conn.execute(ddl)


# ---------------------------------------------------------------- brands --

def create_brand(**kwargs) -> int:
    fields = ["name", "business_model", "conversion_type", "currency", "margin_pct",
              "aov", "ltv", "target_roas", "target_cpa", "target_payback_days"]
    cols = [f for f in fields if f in kwargs]
    placeholders = ",".join(["?"] * len(cols))
    with get_conn() as conn:
        cur = conn.execute(
            f"INSERT INTO brands ({','.join(cols)}) VALUES ({placeholders})",
            [kwargs[c] for c in cols],
        )
        return cur.lastrowid


def update_brand(brand_id: int, **kwargs):
    fields = ["business_model", "conversion_type", "currency", "margin_pct",
              "aov", "ltv", "target_roas", "target_cpa", "target_payback_days"]
    cols = [f for f in fields if f in kwargs]
    if not cols:
        return
    set_clause = ",".join(f"{c}=?" for c in cols)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE brands SET {set_clause} WHERE id=?",
            [kwargs[c] for c in cols] + [brand_id],
        )


def list_brands() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM brands ORDER BY name").fetchall()


def get_brand(brand_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM brands WHERE id=?", (brand_id,)).fetchone()


def get_brand_by_name(name: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM brands WHERE name=?", (name,)).fetchone()


# --------------------------------------------------------------- imports --

def find_overlapping_import(brand_id: int, platform: str, date_start: str, date_end: str):
    with get_conn() as conn:
        return conn.execute(
            """SELECT * FROM imports
               WHERE brand_id=? AND platform=?
                 AND NOT (date_end < ? OR date_start > ?)""",
            (brand_id, platform, date_start, date_end),
        ).fetchall()


def create_import(brand_id, platform, level, filename, date_start, date_end,
                   row_count, status, unmapped_columns: list[str], notes: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO imports
               (brand_id, platform, level, filename, date_start, date_end,
                row_count, status, unmapped_columns, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (brand_id, platform, level, filename, date_start, date_end,
             row_count, status, json.dumps(unmapped_columns), notes),
        )
        return cur.lastrowid


def delete_import(import_id: int, brand_id: int) -> bool:
    """Scoped to brand_id on purpose — an import id typed into the wrong
    brand's Settings tab must refuse, not silently delete another brand's
    data. Returns False (nothing deleted) when the import doesn't belong
    to this brand."""
    with get_conn() as conn:
        owner = conn.execute("SELECT brand_id FROM imports WHERE id=?", (import_id,)).fetchone()
        if owner is None or owner["brand_id"] != brand_id:
            return False
        conn.execute("DELETE FROM performance_rows WHERE import_id=?", (import_id,))
        conn.execute("DELETE FROM imports WHERE id=?", (import_id,))
        return True


def delete_rows_in_range(brand_id: int, platform: str, date_start: str, date_end: str):
    """Used by API sync (not manual upload): a scheduled sync re-pulls a
    trailing window every run (ad platforms revise recent conversions for
    a few days after the fact), so it cleanly overwrites that window
    instead of accumulating duplicate rows like a one-off CSV import would."""
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM performance_rows WHERE brand_id=? AND platform=? AND date>=? AND date<=?",
            (brand_id, platform, date_start, date_end),
        )


def list_imports(brand_id: int | None = None) -> list[sqlite3.Row]:
    with get_conn() as conn:
        if brand_id:
            return conn.execute(
                "SELECT * FROM imports WHERE brand_id=? ORDER BY imported_at DESC",
                (brand_id,),
            ).fetchall()
        return conn.execute("SELECT * FROM imports ORDER BY imported_at DESC").fetchall()


def insert_rows(import_id: int, brand_id: int, rows: list[dict]):
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO performance_rows
               (import_id, brand_id, date, platform, level, campaign, ad_set, ad,
                keyword, spend, impressions, clicks, conversions, conversion_value,
                reach, currency, result_type)
               VALUES (:import_id, :brand_id, :date, :platform, :level, :campaign,
                       :ad_set, :ad, :keyword, :spend, :impressions, :clicks,
                       :conversions, :conversion_value, :reach, :currency, :result_type)""",
            [{**r, "import_id": import_id, "brand_id": brand_id, "reach": r.get("reach")} for r in rows],
        )


def rows_for_brand(brand_id: int, start: str | None = None, end: str | None = None):
    q = "SELECT * FROM performance_rows WHERE brand_id=?"
    params: list = [brand_id]
    if start:
        q += " AND date >= ?"
        params.append(start)
    if end:
        q += " AND date <= ?"
        params.append(end)
    with get_conn() as conn:
        return conn.execute(q, params).fetchall()


# ------------------------------------------------------------------ tests --

def create_test(**kwargs) -> int:
    fields = ["brand_id", "name", "hypothesis", "variant_a_label", "variant_a_campaigns",
              "variant_b_label", "variant_b_campaigns", "started_at", "ended_at", "status"]
    cols = [f for f in fields if f in kwargs]
    placeholders = ",".join(["?"] * len(cols))
    with get_conn() as conn:
        cur = conn.execute(
            f"INSERT INTO tests ({','.join(cols)}) VALUES ({placeholders})",
            [kwargs[c] for c in cols],
        )
        return cur.lastrowid


def list_tests(brand_id: int) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM tests WHERE brand_id=? ORDER BY started_at DESC", (brand_id,)
        ).fetchall()


# ----------------------------------------------------------------- orders --

def insert_orders(import_id: int, brand_id: int, rows: list[dict]):
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO orders (import_id, brand_id, order_date, order_id, amount, raw_campaign, source)
               VALUES (:import_id, :brand_id, :order_date, :order_id, :amount, :campaign, :source)""",
            [{**r, "import_id": import_id, "brand_id": brand_id} for r in rows],
        )


def orders_for_brand(brand_id: int, start: str | None = None, end: str | None = None) -> list[sqlite3.Row]:
    q = "SELECT * FROM orders WHERE brand_id=?"
    params: list = [brand_id]
    if start:
        q += " AND order_date >= ?"
        params.append(start)
    if end:
        q += " AND order_date <= ?"
        params.append(end)
    with get_conn() as conn:
        return conn.execute(q, params).fetchall()


def unmatched_raw_campaigns(brand_id: int) -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT raw_campaign FROM orders WHERE brand_id=? AND matched_campaign IS NULL AND raw_campaign IS NOT NULL",
            (brand_id,),
        ).fetchall()
        return [r["raw_campaign"] for r in rows]


def set_campaign_matches(brand_id: int, matches: dict[str, str | None]):
    """matches: {raw_campaign: matched_campaign_or_None}. None (or 'ignore')
    is stored as the literal string so it isn't re-suggested every time."""
    with get_conn() as conn:
        for raw, matched in matches.items():
            conn.execute(
                "UPDATE orders SET matched_campaign=? WHERE brand_id=? AND raw_campaign=?",
                (matched or "ignore", brand_id, raw),
            )
