# Uncorrelated Basket Finder (SQLite + Jupyter) — v3
# https://uncorrelated-asset-basket.streamlit.app/
Note: to run using the browser app, start by deselecting offline mode and allowing network

This project builds/uses a local SQLite database of asset metadata + daily prices, computes returns/correlation,
and searches for a 30-asset basket with minimal pairwise correlations (mean absolute correlation),
using simulated annealing with greedy-diversity seeds.

## Key v3 change: DB-first + offline cache mode

**Important reality check:** `yfinance` gets its data by calling Yahoo Finance web endpoints. There is **no** way
to use `yfinance` to populate data *without* network requests to Yahoo.  

So v3 implements what you likely want operationally:

- **DB-first behavior:** If your SQLite DB already has fresh + sufficient data, the pipeline does **no fetching**.
- **Offline mode (default):** If the DB is missing/stale, the pipeline will import **local cached CSV files** (no network).
- **Optional network mode:** You can enable `allow_network=True` to fetch missing data via yfinance (rate-limited).

## Cache layout (offline mode)

Place per-symbol CSVs here:

- `data_cache/prices/SPY.csv`
- `data_cache/prices/QQQ.csv`
- ...

CSV format: standard Yahoo columns work:
`Date,Open,High,Low,Close,Adj Close,Volume`

Optional metadata JSONs (for market cap / sector penalties, etc.):

- `data_cache/meta/SPY.json`

## Quickstart

1) Install:
- `pip install -r requirements.txt`

2) Open `basket_finder.ipynb` and run all cells.

### If you want zero network calls
Leave:
- `offline_mode=True`
- `allow_network=False`

and provide cached CSVs under `data_cache/prices/`.

### If you want it to fill missing data automatically
Set:
- `offline_mode=False`
- `allow_network=True`

(then it will call yfinance, which is a network pull).

## Outputs
- SQLite DB at `assets.sqlite`
- Final basket stored in DB (table: `basket_final`)
- Candidate baskets stored in DB (table: `basket_candidates`)
