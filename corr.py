from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal

def corr_matrix(returns_df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    if method not in ("pearson","spearman"):
        raise ValueError("corr method must be 'pearson' or 'spearman'")
    return returns_df.corr(method=method)

def basket_score_abs_mean(corr: pd.DataFrame, symbols: list[str]) -> float:
    sub = corr.loc[symbols, symbols].values
    k = len(symbols)
    if k < 2:
        return 1.0
    # mean of upper triangle abs
    iu = np.triu_indices(k, k=1)
    vals = np.abs(sub[iu])
    return float(np.mean(vals)) if len(vals) else 1.0

def basket_score_abs_max(corr: pd.DataFrame, symbols: list[str]) -> float:
    sub = corr.loc[symbols, symbols].values
    k = len(symbols)
    if k < 2:
        return 1.0
    iu = np.triu_indices(k, k=1)
    vals = np.abs(sub[iu])
    return float(np.max(vals)) if len(vals) else 1.0
