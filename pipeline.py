from __future__ import annotations

import json, datetime as dt
from typing import List, Dict, Tuple
import pandas as pd
from config import Config
from logging_utils import log
import db as dbmod
from data import (
    fetch_metadata_yf,
    download_prices_yf,
    normalize_prices_df,
    load_prices_from_cache,
    load_metadata_from_cache,
)
from universe import rank_universe, AssetMeta, feasibility_penalties
from features import compute_returns, align_returns
from corr import corr_matrix
from optimizer import PenaltyInfo, generate_seeds_by_greedy_diversity, anneal_optimize
from report import summarize_basket

DEFAULT_SYMBOLS = [
    # Broad equity indices / style
    "SPY","QQQ","IWM","DIA","VTI","VEA","VWO",
    # Bonds
    "TLT","IEF","SHY","LQD","HYG","TIP",
    # Commodities / gold / energy
    "GLD","IAU","SLV","DBC","USO","UNG",
    # Real estate
    "VNQ",
    # Volatility-ish proxy (not perfect)
    "VXX",
    # Sector ETFs
    "XLF","XLK","XLE","XLV","XLI","XLP","XLY","XLB","XLU",
    # A few large equities (optional diversification)
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","JPM","XOM","JNJ"
]

def _is_fresh(latest_date_iso: str | None, freshness_days: int) -> bool:
    if not latest_date_iso:
        return False
    try:
        latest = dt.date.fromisoformat(latest_date_iso)
    except Exception:
        return False
    return (dt.date.today() - latest).days <= freshness_days

def run_pipeline(config: Config, seed_symbols: List[str] | None = None) -> Dict:
    """Main entrypoint.
    v3 behavior:
      - If DB already has fresh + sufficient price history, we skip any network/cache ingestion for that symbol.
      - If offline_mode=True (default), no network calls are made; cached CSVs are used only when DB is missing/stale.
      - Metadata is optional; if offline and no cached metadata, fundamentals are left null (soft penalties still work).
    """
    config.ensure_paths()
    seed_symbols = seed_symbols or DEFAULT_SYMBOLS

    log("Connecting DB...")
    conn = dbmod.connect(config.db_path)
    dbmod.init_db(conn)

    today = dt.date.today().isoformat()

    # ---- 1) Metadata ingestion (optional) ----
    metas: List[AssetMeta] = []
    meta_map_cache: Dict[str, dict] = load_metadata_from_cache(seed_symbols, config.cache_dir)

    if config.offline_mode or (not config.allow_network):
        # Offline: build AssetMeta from cache if available; else minimal meta
        for s in seed_symbols:
            m = meta_map_cache.get(s, {})
            metas.append(AssetMeta(
                symbol=s,
                name=m.get("name"),
                asset_type=m.get("asset_type"),
                exchange=m.get("exchange"),
                currency=m.get("currency"),
                country=m.get("country"),
                sector=m.get("sector"),
                industry=m.get("industry"),
                market_cap=m.get("market_cap"),
                avg_volume_20d=m.get("avg_volume_20d"),
                dollar_volume_20d=m.get("dollar_volume_20d"),
            ))
        log(f"Offline metadata: {sum(1 for m in metas if m.name)} / {len(metas)} symbols had cached meta.")
    else:
        log("Fetching metadata via yfinance (network)...")
        metas = fetch_metadata_yf(seed_symbols, min_delay_s=config.yf_min_delay_s)

    # Upsert assets + fundamentals only if not already present for today
    with dbmod.txn(conn):
        for m in metas:
            dbmod.upsert_asset(conn, {
                "symbol": m.symbol,
                "name": m.name,
                "asset_type": m.asset_type,
                "exchange": m.exchange,
                "currency": m.currency,
                "country": m.country,
                "sector": m.sector,
                "industry": m.industry,
                "is_active": 1
            })
            if not dbmod.get_fundamental_status(conn, m.symbol, today):
                dbmod.upsert_fundamental(conn, m.symbol, today, m.market_cap, m.avg_volume_20d, m.dollar_volume_20d, "cache" if (config.offline_mode or not config.allow_network) else "yfinance")
            dbmod.upsert_ingestion_log(conn, "fundamentals", m.symbol, dt.datetime.now().isoformat(timespec="seconds"), note="offline" if (config.offline_mode or not config.allow_network) else "yfinance")

    # ---- 2) Rank universe ----
    ranked = rank_universe(metas, config.soft_min_market_cap_usd, config.soft_min_dollar_volume_usd)
    ranked_metas = [m for (m,score) in ranked]

    def is_leveraged_inverse(meta: AssetMeta) -> bool:
        nm = (meta.name or "").lower()
        return any(x in nm for x in ["2x","3x","ultra","inverse","-1x","-2x","-3x","leveraged"])

    filtered = []
    for m in ranked_metas:
        if config.exclude_leveraged_inverse_etfs and is_leveraged_inverse(m):
            continue
        if (m.asset_type or "").lower() == "etf" and not config.include_etfs:
            continue
        if (m.asset_type or "").lower() == "equity" and not config.include_equities:
            continue
        filtered.append(m)

    filtered = filtered[:max(config.candidate_pool_size, config.basket_size)]
    universe_symbols = [m.symbol for m in filtered]
    log(f"Universe size after ranking/filtering: {len(universe_symbols)}")

    # ---- 3) Prices: decide what needs ingestion ----
    needs = []
    fresh = []
    with dbmod.txn(conn):
        for sym in universe_symbols:
            cnt, latest = dbmod.get_price_status(conn, sym)
            if cnt >= config.lookback_days and _is_fresh(latest, config.freshness_days):
                fresh.append(sym)
            else:
                needs.append(sym)
    log(f"Prices fresh in DB: {len(fresh)} | need update/import: {len(needs)}")

    # ---- 4) Ingest prices only for needed symbols ----
    px_dict: Dict[str, pd.DataFrame] = {}

    if needs:
        if config.offline_mode or (not config.allow_network):
            log("Offline mode: importing from local cache CSVs only for missing/stale symbols...")
            cache_px = load_prices_from_cache(needs, config.cache_dir)
            missing_cache = [s for s in needs if s not in cache_px]
            if missing_cache:
                log("WARNING: Missing cached CSVs for: " + ", ".join(missing_cache))
                log("These symbols will be dropped from the usable universe unless DB already contains enough history.")
            px_dict.update(cache_px)
        else:
            # Network mode: batch yfinance download; still throttling-friendly
            end = dt.date.today()
            start = end - dt.timedelta(days=int(config.lookback_days*1.6))
            start_s, end_s = str(start), str(end)

            log(f"Downloading prices via yfinance in batches (batch={config.yf_batch_size})...")
            remaining = needs[:]
            while remaining:
                batch = remaining[:config.yf_batch_size]
                remaining = remaining[config.yf_batch_size:]
                raw = download_prices_yf(batch, start=start_s, end=end_s, threads=False)
                batch_px = normalize_prices_df(raw)
                px_dict.update(batch_px)
                if remaining:
                    import time
                    time.sleep(config.yf_batch_sleep_s)

    # Persist prices + compute returns
    returns = {}
    with dbmod.txn(conn):
        for sym, df in px_dict.items():
            df = df.copy().sort_index()
            rows = []
            for idx, r in df.iterrows():
                rows.append((
                    idx.date().isoformat(),
                    float(r.get("open")) if pd.notna(r.get("open")) else None,
                    float(r.get("high")) if pd.notna(r.get("high")) else None,
                    float(r.get("low")) if pd.notna(r.get("low")) else None,
                    float(r.get("close")) if pd.notna(r.get("close")) else None,
                    float(r.get("adj_close")) if pd.notna(r.get("adj_close")) else None,
                    float(r.get("volume")) if pd.notna(r.get("volume")) else None,
                ))
            dbmod.insert_prices(conn, sym, rows, "cache" if (config.offline_mode or not config.allow_network) else "yfinance")
            dbmod.upsert_ingestion_log(conn, "prices", sym, dt.datetime.now().isoformat(timespec="seconds"),
                                       note="offline_cache" if (config.offline_mode or not config.allow_network) else "yfinance")

    # Build returns from DB if fresh symbols were skipped:
    # For simplicity, compute returns from in-memory price frames when available, else read from DB.
    # We'll read adj_close series for all universe symbols from DB to ensure completeness.
    log("Loading prices from DB for returns alignment...")
    price_series = {}
    with conn:
        for sym in universe_symbols:
            cur = conn.execute(
                "SELECT date, adj_close FROM prices_daily WHERE symbol=? ORDER BY date;",
                (sym,)
            )
            rows = cur.fetchall()
            if not rows:
                continue
            s = pd.Series(
                [r[1] for r in rows],
                index=pd.to_datetime([r[0] for r in rows]),
                name=sym
            ).dropna()
            if len(s) < config.lookback_days:
                continue
            df = pd.DataFrame({"adj_close": s})
            ret = compute_returns(df, config.return_type)
            returns[sym] = ret

    ret_df = align_returns(returns, config.lookback_days)
    universe_symbols = list(ret_df.columns)
    log(f"Aligned returns shape: {ret_df.shape} (symbols: {len(universe_symbols)})")
    if len(universe_symbols) < config.basket_size:
        raise RuntimeError(
            f"Not enough symbols with sufficient data to build basket of {config.basket_size}. "
            f"Tip: add cached CSVs under {config.cache_dir}/prices/ or set allow_network=True."
        )

    corr = corr_matrix(ret_df, method=config.corr_method)

    # Snapshot universe
    criteria = {
        "offline_mode": config.offline_mode,
        "allow_network": config.allow_network,
        "cache_dir": config.cache_dir,
        "freshness_days": config.freshness_days,
        "soft_min_market_cap_usd": config.soft_min_market_cap_usd,
        "soft_min_dollar_volume_usd": config.soft_min_dollar_volume_usd,
        "lookback_days": config.lookback_days,
        "return_type": config.return_type,
        "corr_method": config.corr_method,
        "candidate_pool_size": config.candidate_pool_size
    }
    with dbmod.txn(conn):
        snapshot_id = dbmod.create_universe_snapshot(conn, today, json.dumps(criteria), universe_symbols)
        symbols_ranked = [(s, i+1) for i,s in enumerate(universe_symbols)]
        dbmod.insert_universe_members(conn, snapshot_id, symbols_ranked)

    # Penalties maps
    cap_pen = {}
    dv_pen = {}
    sector_map = {}
    meta_map = {m.symbol: m for m in filtered}
    for s in universe_symbols:
        m = meta_map.get(s, AssetMeta(symbol=s))
        cpen, dpen = feasibility_penalties(m, config.soft_min_market_cap_usd, config.soft_min_dollar_volume_usd)
        cap_pen[s] = float(cpen)
        dv_pen[s] = float(dpen)
        sector_map[s] = m.sector or "Unknown"
    penalty_info = PenaltyInfo(cap_penalties=cap_pen, dv_penalties=dv_pen, sector_map=sector_map)
    lambdas = {
        "lambda_market_cap": config.lambda_market_cap,
        "lambda_dollar_vol": config.lambda_dollar_vol,
        "lambda_sector_concentration": config.lambda_sector_concentration,
    }

    # Optimizer run
    params = {
        "basket_size": config.basket_size,
        "n_seeds": config.n_seeds,
        "n_iterations": config.n_iterations,
        "objective": "abs_mean",
        "penalties": lambdas
    }
    started = dt.datetime.now().isoformat(timespec="seconds")
    with dbmod.txn(conn):
        run_id = dbmod.create_run(conn, snapshot_id, json.dumps(params), started)

    seeds = generate_seeds_by_greedy_diversity(corr, universe_symbols, config.basket_size, config.n_seeds, rng_seed=config.random_seed)
    candidates = []
    best = None
    best_score = 1e9
    best_aux = {}

    basket_id = 0
    for i, seed in enumerate(seeds):
        b, sc, aux = anneal_optimize(
            corr=corr,
            universe=universe_symbols,
            k=config.basket_size,
            penalty=penalty_info,
            lambdas=lambdas,
            base_metric="abs_mean",
            n_iter=config.n_iterations,
            rng_seed=config.random_seed + i + 1,
            seed_basket=seed
        )
        candidates.append({"basket_id": basket_id, "score": sc, "symbols": b, "aux": aux})
        if sc < best_score:
            best, best_score, best_aux = b, sc, aux
        basket_id += 1

    candidates_sorted = sorted(candidates, key=lambda x: x["score"])
    cutoff_idx = max(1, int(len(candidates_sorted) * config.top_percentile))
    top_set = set([c["basket_id"] for c in candidates_sorted[:cutoff_idx]])

    with dbmod.txn(conn):
        for c in candidates:
            dbmod.insert_candidate(
                conn,
                run_id=run_id,
                basket_id=c["basket_id"],
                symbols_json=json.dumps(c["symbols"]),
                score=float(c["score"]),
                aux_metrics_json=json.dumps(c["aux"]),
                is_top_percentile=1 if c["basket_id"] in top_set else 0
            )
        explain = {
            "best_aux": best_aux,
            "note": "Quantitative selection based on mean absolute correlation + soft penalties (market cap, dollar volume, sector concentration).",
            "offline_mode": config.offline_mode,
            "allow_network": config.allow_network
        }
        dbmod.upsert_final(conn, run_id, json.dumps(best), float(best_score), json.dumps(explain))
        finished = dt.datetime.now().isoformat(timespec="seconds")
        dbmod.finish_run(conn, run_id, finished)

    summary = summarize_basket(corr, best)

    return {
        "db_path": config.db_path,
        "snapshot_id": snapshot_id,
        "run_id": run_id,
        "universe_symbols": universe_symbols,
        "returns_df": ret_df,
        "corr": corr,
        "best_symbols": best,
        "best_score": best_score,
        "summary_df": summary,
        "candidates": candidates_sorted,
        "top_cutoff": cutoff_idx
    }
