from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math

@dataclass
class AssetMeta:
    symbol: str
    name: Optional[str] = None
    asset_type: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    country: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    avg_volume_20d: Optional[float] = None
    dollar_volume_20d: Optional[float] = None

def sigmoid_score(x: float, mid: float, scale: float) -> float:
    # maps to (0,1)
    z = (x - mid) / max(scale, 1e-9)
    return 1.0 / (1.0 + math.exp(-z))

def feasibility_penalties(meta: AssetMeta, soft_min_market_cap: float, soft_min_dollar_vol: float) -> Tuple[float,float]:
    # Penalty is high when below threshold, near 0 when above.
    cap = meta.market_cap if meta.market_cap is not None else 0.0
    dv = meta.dollar_volume_20d if meta.dollar_volume_20d is not None else 0.0

    cap_score = sigmoid_score(cap, soft_min_market_cap, soft_min_market_cap/2.0)
    dv_score  = sigmoid_score(dv, soft_min_dollar_vol, soft_min_dollar_vol/2.0)

    cap_pen = 1.0 - cap_score
    dv_pen  = 1.0 - dv_score
    return cap_pen, dv_pen

def rank_universe(metas: List[AssetMeta], soft_min_market_cap: float, soft_min_dollar_vol: float) -> List[Tuple[AssetMeta,float]]:
    ranked = []
    for m in metas:
        cap_pen, dv_pen = feasibility_penalties(m, soft_min_market_cap, soft_min_dollar_vol)
        # rank by *low* penalty and high dollar volume
        base = (1.0 - 0.6*cap_pen - 0.4*dv_pen)
        dv = m.dollar_volume_20d or 0.0
        score = base + 0.0000000001 * dv  # tiny tie-break
        ranked.append((m, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
