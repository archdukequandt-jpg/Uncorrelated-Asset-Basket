from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def compute_returns(price_df: pd.DataFrame, return_type: str = "log_adj") -> pd.Series:
    """price_df columns should include adj_close."""
    px = price_df["adj_close"].astype(float)
    if return_type == "log_adj":
        r = np.log(px).diff()
    elif return_type == "simple_adj":
        r = px.pct_change()
    else:
        raise ValueError(f"Unknown return_type: {return_type}")
    return r.dropna()

def align_returns(returns: Dict[str, pd.Series], lookback: int) -> pd.DataFrame:
    """Align returns across symbols and keep last `lookback` rows."""
    df = pd.DataFrame(returns)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df = df.sort_index()
    if len(df) > lookback:
        df = df.iloc[-lookback:]
    # drop symbols with too much missing data
    min_coverage = 0.95
    keep = []
    for c in df.columns:
        coverage = df[c].notna().mean()
        if coverage >= min_coverage:
            keep.append(c)
    df = df[keep].dropna(axis=0, how="any")  # ensure full rows for correlation stability
    return df
