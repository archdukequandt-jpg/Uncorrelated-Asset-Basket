from __future__ import annotations
import json
import pandas as pd

def summarize_basket(corr: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    sub = corr.loc[symbols, symbols].copy()
    # report each symbol's mean abs corr to others
    mean_abs = sub.abs().mean(axis=1)
    out = pd.DataFrame({"symbol": mean_abs.index, "mean_abs_corr": mean_abs.values})
    out = out.sort_values("mean_abs_corr")
    return out

def leaderboard_df(candidates: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(candidates).sort_values("score").reset_index(drop=True)
