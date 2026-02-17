import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional, Tuple


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS assets (
    asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    name TEXT,
    asset_type TEXT,
    exchange TEXT,
    currency TEXT,
    country TEXT,
    sector TEXT,
    industry TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fundamentals (
    symbol TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    market_cap REAL,
    avg_volume_20d REAL,
    dollar_volume_20d REAL,
    source TEXT,
    PRIMARY KEY (symbol, as_of_date)
);

CREATE TABLE IF NOT EXISTS prices_daily (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume REAL,
    source TEXT,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS returns_daily (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    ret REAL,
    return_type TEXT NOT NULL,
    PRIMARY KEY (symbol, date, return_type)
);

CREATE TABLE IF NOT EXISTS universe_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_date TEXT NOT NULL,
    criteria_json TEXT,
    count_assets INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS universe_members (
    snapshot_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    rank_liquidity INTEGER,
    PRIMARY KEY (snapshot_id, symbol),
    FOREIGN KEY (snapshot_id) REFERENCES universe_snapshots(snapshot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS basket_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    params_json TEXT,
    started_at TEXT,
    finished_at TEXT,
    FOREIGN KEY (snapshot_id) REFERENCES universe_snapshots(snapshot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS basket_candidates (
    run_id INTEGER NOT NULL,
    basket_id INTEGER NOT NULL,
    symbols_json TEXT NOT NULL,
    score REAL NOT NULL,
    aux_metrics_json TEXT,
    is_top_percentile INTEGER DEFAULT 0,
    PRIMARY KEY (run_id, basket_id),
    FOREIGN KEY (run_id) REFERENCES basket_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    kind TEXT NOT NULL,              -- 'prices' or 'fundamentals'
    symbol TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    note TEXT,
    PRIMARY KEY (kind, symbol)
);

CREATE TABLE IF NOT EXISTS basket_final (
    run_id INTEGER PRIMARY KEY,
    symbols_json TEXT NOT NULL,
    score REAL NOT NULL,
    explain_json TEXT,
    FOREIGN KEY (run_id) REFERENCES basket_runs(run_id) ON DELETE CASCADE
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


@contextmanager
def txn(conn: sqlite3.Connection):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def upsert_asset(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    cols = ["symbol", "name", "asset_type", "exchange", "currency", "country", "sector", "industry", "is_active"]
    vals = [row.get(c) for c in cols]
    placeholders = ",".join(["?"] * len(cols))
    update_set = ",".join([f"{c}=excluded.{c}" for c in cols[1:]]) + ",updated_at=datetime('now')"
    sql = f"""INSERT INTO assets ({",".join(cols)}) VALUES ({placeholders})
              ON CONFLICT(symbol) DO UPDATE SET {update_set};"""
    conn.execute(sql, vals)


def upsert_fundamental(
    conn: sqlite3.Connection,
    symbol: str,
    as_of_date: str,
    market_cap: Optional[float],
    avg_volume_20d: Optional[float],
    dollar_volume_20d: Optional[float],
    source: str,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO fundamentals
            (symbol, as_of_date, market_cap, avg_volume_20d, dollar_volume_20d, source)
            VALUES (?, ?, ?, ?, ?, ?);""",
        (symbol, as_of_date, market_cap, avg_volume_20d, dollar_volume_20d, source),
    )


def insert_prices(conn: sqlite3.Connection, symbol: str, rows: Iterable[Tuple], source: str) -> None:
    # rows: (date, open, high, low, close, adj_close, volume)
    conn.executemany(
        """INSERT OR REPLACE INTO prices_daily
            (symbol, date, open, high, low, close, adj_close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
        [(symbol, *r, source) for r in rows],
    )


def insert_returns(conn: sqlite3.Connection, symbol: str, rows: Iterable[Tuple[str, float]], return_type: str) -> None:
    # rows: (date, ret)
    conn.executemany(
        """INSERT OR REPLACE INTO returns_daily
            (symbol, date, ret, return_type)
            VALUES (?, ?, ?, ?);""",
        [(symbol, d, r, return_type) for (d, r) in rows],
    )


def create_universe_snapshot(conn: sqlite3.Connection, as_of_date: str, criteria_json: str, symbols: Iterable[str]) -> int:
    symbols_list = list(symbols)  # prevent consuming generators twice
    cur = conn.execute(
        """INSERT INTO universe_snapshots (as_of_date, criteria_json, count_assets)
            VALUES (?, ?, ?);""",
        (as_of_date, criteria_json, len(symbols_list)),
    )
    return int(cur.lastrowid)


def insert_universe_members(conn: sqlite3.Connection, snapshot_id: int, symbols_ranked: Iterable[Tuple[str, int]]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO universe_members (snapshot_id, symbol, rank_liquidity)
            VALUES (?, ?, ?);""",
        [(snapshot_id, s, r) for (s, r) in symbols_ranked],
    )


def create_run(conn: sqlite3.Connection, snapshot_id: int, params_json: str, started_at: str) -> int:
    cur = conn.execute(
        """INSERT INTO basket_runs (snapshot_id, params_json, started_at)
            VALUES (?, ?, ?);""",
        (snapshot_id, params_json, started_at),
    )
    return int(cur.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int, finished_at: str) -> None:
    conn.execute(
        """UPDATE basket_runs SET finished_at=? WHERE run_id=?;""",
        (finished_at, run_id),
    )


def insert_candidate(
    conn: sqlite3.Connection,
    run_id: int,
    basket_id: int,
    symbols_json: str,
    score: float,
    aux_metrics_json: str,
    is_top_percentile: int = 0,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO basket_candidates
            (run_id, basket_id, symbols_json, score, aux_metrics_json, is_top_percentile)
            VALUES (?, ?, ?, ?, ?, ?);""",
        (run_id, basket_id, symbols_json, score, aux_metrics_json, is_top_percentile),
    )


def upsert_final(conn: sqlite3.Connection, run_id: int, symbols_json: str, score: float, explain_json: str) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO basket_final (run_id, symbols_json, score, explain_json)
            VALUES (?, ?, ?, ?);""",
        (run_id, symbols_json, score, explain_json),
    )


def get_price_status(conn: sqlite3.Connection, symbol: str) -> Tuple[int, Optional[str]]:
    """Return (row_count, latest_date_iso or None) for prices_daily."""
    cur = conn.execute("SELECT COUNT(1), MAX(date) FROM prices_daily WHERE symbol=?;", (symbol,))
    cnt, mx = cur.fetchone()
    return int(cnt or 0), mx


def get_fundamental_status(conn: sqlite3.Connection, symbol: str, as_of_date: str) -> bool:
    cur = conn.execute("SELECT 1 FROM fundamentals WHERE symbol=? AND as_of_date=? LIMIT 1;", (symbol, as_of_date))
    return cur.fetchone() is not None


def upsert_ingestion_log(conn: sqlite3.Connection, kind: str, symbol: str, last_updated: str, note: str = "") -> None:
    conn.execute(
        """INSERT OR REPLACE INTO ingestion_log (kind, symbol, last_updated, note)
           VALUES (?, ?, ?, ?);""",
        (kind, symbol, last_updated, note),
    )
