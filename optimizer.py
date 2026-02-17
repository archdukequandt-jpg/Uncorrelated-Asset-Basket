from __future__ import annotations
import random, math, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
import pandas as pd
from corr import basket_score_abs_mean, basket_score_abs_max

@dataclass
class PenaltyInfo:
    cap_penalties: Dict[str, float]
    dv_penalties: Dict[str, float]
    sector_map: Dict[str, str]

def sector_concentration_penalty(symbols: List[str], sector_map: Dict[str,str]) -> float:
    # Penalize if too many in same sector; simple Herfindahl index
    sectors = [sector_map.get(s) or "Unknown" for s in symbols]
    counts = {}
    for sec in sectors:
        counts[sec] = counts.get(sec, 0) + 1
    k = len(symbols)
    hhi = sum((c/k)**2 for c in counts.values())
    # normalize: minimum is ~1/n_sectors, max is 1
    return float(hhi)

def basket_objective(
    corr: pd.DataFrame,
    symbols: List[str],
    penalty: Optional[PenaltyInfo],
    lambdas: Dict[str,float],
    base_metric: str = "abs_mean",
) -> Tuple[float, Dict]:
    # Base correlation metric
    if base_metric == "abs_mean":
        base = basket_score_abs_mean(corr, symbols)
    elif base_metric == "abs_max":
        base = basket_score_abs_max(corr, symbols)
    else:
        raise ValueError("base_metric must be abs_mean or abs_max")

    aux = {"base_corr": base}

    if penalty is None:
        return base, aux

    cap_pen = float(np.mean([penalty.cap_penalties.get(s, 1.0) for s in symbols]))
    dv_pen  = float(np.mean([penalty.dv_penalties.get(s, 1.0) for s in symbols]))
    sec_pen = sector_concentration_penalty(symbols, penalty.sector_map)

    score = base             + lambdas.get("lambda_market_cap", 0.0) * cap_pen             + lambdas.get("lambda_dollar_vol", 0.0) * dv_pen             + lambdas.get("lambda_sector_concentration", 0.0) * sec_pen

    aux.update({
        "cap_pen": cap_pen,
        "dv_pen": dv_pen,
        "sector_pen": sec_pen,
        "score": score
    })
    return score, aux

def make_random_basket(universe: List[str], k: int, rng: random.Random) -> List[str]:
    return rng.sample(universe, k)

def anneal_optimize(
    corr: pd.DataFrame,
    universe: List[str],
    k: int,
    penalty: Optional[PenaltyInfo],
    lambdas: Dict[str,float],
    base_metric: str,
    n_iter: int = 3000,
    start_temp: float = 1.0,
    end_temp: float = 0.01,
    rng_seed: int = 42,
    seed_basket: Optional[List[str]] = None,
) -> Tuple[List[str], float, Dict]:
    rng = random.Random(rng_seed)
    current = seed_basket[:] if seed_basket is not None else make_random_basket(universe, k, rng)
    current_score, current_aux = basket_objective(corr, current, penalty, lambdas, base_metric)
    best = current[:]
    best_score = current_score
    best_aux = current_aux

    # Precompute for speed
    universe_set = set(universe)

    for it in range(n_iter):
        t = start_temp * ((end_temp / start_temp) ** (it / max(1, n_iter-1)))

        # propose: swap 1 asset
        out_idx = rng.randrange(k)
        out_sym = current[out_idx]
        remaining = set(current)
        remaining.remove(out_sym)
        # candidate pool for replacement
        candidates = list(universe_set - remaining)
        in_sym = rng.choice(candidates)
        proposal = current[:]
        proposal[out_idx] = in_sym

        prop_score, prop_aux = basket_objective(corr, proposal, penalty, lambdas, base_metric)
        delta = prop_score - current_score
        accept = (delta <= 0) or (rng.random() < math.exp(-delta / max(t, 1e-12)))

        if accept:
            current = proposal
            current_score = prop_score
            current_aux = prop_aux
            if current_score < best_score:
                best = current[:]
                best_score = current_score
                best_aux = current_aux

    return best, best_score, best_aux

def generate_seeds_by_greedy_diversity(corr: pd.DataFrame, universe: List[str], k: int, n_seeds: int, rng_seed: int = 42) -> List[List[str]]:
    rng = random.Random(rng_seed)
    seeds = []
    symbols = universe[:]
    if len(symbols) < k:
        raise ValueError("Universe smaller than basket size.")

    # Distance = 1 - abs(corr)
    abs_corr = corr.abs().fillna(1.0)

    for s in range(n_seeds):
        start = rng.choice(symbols)
        basket = [start]
        while len(basket) < k:
            # pick symbol that minimizes max abs corr to current basket
            remaining = [x for x in symbols if x not in basket]
            best_sym = None
            best_val = float("inf")
            for cand in remaining:
                mx = float(abs_corr.loc[cand, basket].max())
                if mx < best_val:
                    best_val = mx
                    best_sym = cand
            basket.append(best_sym)
        seeds.append(basket)
    return seeds
