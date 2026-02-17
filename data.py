from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import datetime as dt
import time
import json
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None
from logging_utils import log
from universe import AssetMeta

# -------------------------
# Offline / cache utilities
# -------------------------

def load_prices_from_cache(symbols: List[str], cache_dir: str) -> Dict[str, pd.DataFrame]:
    """Load cached Yahoo-style CSVs from {cache_dir}/prices/{SYMBOL}.csv"""
    out: Dict[str, pd.DataFrame] = {}
    base = Path(cache_dir) / "prices"
    for sym in symbols:
        fp = base / f"{sym}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        # Accept either 'Date' or 'date'
        date_col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
        if date_col is None:
            continue
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        # Normalize columns to expected names
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "open": rename[c] = "open"
            elif cl == "high": rename[c] = "high"
            elif cl == "low": rename[c] = "low"
            elif cl == "close": rename[c] = "close"
            elif cl in ("adj close", "adj_close", "adjclose"): rename[c] = "adj_close"
            elif cl == "volume": rename[c] = "volume"
        df = df.rename(columns=rename)
        if "adj_close" not in df.columns:
            # If only close exists, use close as adj_close (worst-case fallback)
            if "close" in df.columns:
                df["adj_close"] = df["close"]
            else:
                continue
        out[sym] = df[["open","high","low","close","adj_close","volume"]].copy() if all(c in df.columns for c in ["open","high","low","close","adj_close","volume"]) else df.copy()
    return out

def load_metadata_from_cache(symbols: List[str], cache_dir: str) -> Dict[str, dict]:
    """Load cached metadata JSONs from {cache_dir}/meta/{SYMBOL}.json"""
    out: Dict[str, dict] = {}
    base = Path(cache_dir) / "meta"
    for sym in symbols:
        fp = base / f"{sym}.json"
        if not fp.exists():
            continue
        try:
            out[sym] = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
    return out

# -------------------------
# yfinance network functions
# -------------------------

def _require_yfinance():
    if yf is None:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

def fetch_metadata_yf(symbols: List[str], min_delay_s: float = 0.3) -> List[AssetMeta]:
    """Network-based metadata fetch (may be rate-limited by Yahoo)."""
    _require_yfinance()
    metas: List[AssetMeta] = []
    for i, sym in enumerate(symbols):
        if i > 0:
            time.sleep(min_delay_s)
        try:
            t = yf.Ticker(sym)
            info = {}
            # Prefer fast_info first
            try:
                fi = getattr(t, "fast_info", None)
                if fi:
                    # fast_info is a mapping-like object
                    info.update(dict(fi))
            except Exception:
                pass
            try:
                # fallback (heavier)
                info2 = getattr(t, "info", {}) or {}
                info.update(info2)
            except Exception:
                pass

            name = info.get("shortName") or info.get("longName") or sym
            exchange = info.get("exchange")
            currency = info.get("currency")
            country = info.get("country")
            sector = info.get("sector")
            industry = info.get("industry")
            market_cap = info.get("marketCap")
            avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day") or info.get("averageVolume10days")
            last_price = info.get("regularMarketPrice") or info.get("previousClose") or info.get("last_price")
            dollar_vol = None
            if avg_volume is not None and last_price is not None:
                dollar_vol = float(avg_volume) * float(last_price)

            quote_type = info.get("quoteType")  # EQUITY, ETF, etc.
            asset_type = (quote_type.lower() if isinstance(quote_type, str) else None)

            metas.append(AssetMeta(
                symbol=sym,
                name=name,
                asset_type=asset_type,
                exchange=exchange,
                currency=currency,
                country=country,
                sector=sector,
                industry=industry,
                market_cap=float(market_cap) if market_cap is not None else None,
                avg_volume_20d=float(avg_volume) if avg_volume is not None else None,
                dollar_volume_20d=float(dollar_vol) if dollar_vol is not None else None,
            ))
        except Exception as e:
            log(f"Metadata fetch failed for {sym}: {e}")
            metas.append(AssetMeta(symbol=sym))
    return metas

def download_prices_yf(symbols: List[str], start: str, end: str, threads: bool = False) -> pd.DataFrame:
    _require_yfinance()
    return yf.download(
        tickers=" ".join(symbols),
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        threads=threads,
        progress=False,
    )

def normalize_prices_df(raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Convert yfinance download format -> dict(symbol -> df with normalized columns."""
    out: Dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out

    # MultiIndex columns: (field, symbol)
    if isinstance(raw.columns, pd.MultiIndex):
        symbols = sorted(set([c[1] for c in raw.columns]))
        for sym in symbols:
            cols = {}
            for f in ["Open","High","Low","Close","Adj Close","Volume"]:
                if (f, sym) in raw.columns:
                    cols[f] = raw[(f, sym)]
            if not cols:
                continue
            df = pd.DataFrame(cols).copy()
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                "Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
            })
            df = df.dropna(subset=["adj_close"])
            out[sym] = df
    else:
        df = raw.copy()
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
        })
        df = df.dropna(subset=["adj_close"])
        out["__single__"] = df
    return out
