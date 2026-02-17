"""LLM policy stub.

This project is designed so the quantitative engine does the scoring, while an LLM can
optionally propose search moves or suggest diversification priors.

By default (no API keys), we provide a deterministic, non-LLM policy.
"""
from __future__ import annotations
from typing import List, Dict

def propose_universe_tilts(current_symbols: List[str], metas: Dict[str, dict]) -> Dict:
    # Placeholder: suggests ensuring representation across broad asset types if metadata exists.
    # You can replace this with a call to an external LLM later.
    asset_types = {}
    for s in current_symbols:
        t = (metas.get(s, {}).get("asset_type") or "unknown").lower()
        asset_types[t] = asset_types.get(t, 0) + 1
    return {
        "note": "LLM policy stub: replace with an LLM call if desired.",
        "current_asset_type_counts": asset_types,
        "suggestion": "Consider mixing equities + bond ETFs + commodity ETFs + REITs for lower correlation."
    }
