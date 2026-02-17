from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Storage
    db_path: str = "assets.sqlite"
    data_dir: str = "data_cache"     # local cache directory (CSV/JSON) AND optional scratch
    cache_dir: str = "data_cache"    # alias; keep separate if you want

    # Ingestion mode
    # IMPORTANT: yfinance obtains data via Yahoo web endpoints; there is no fully-offline mode without
    # pre-downloaded local data. When offline_mode=True, the pipeline will ONLY read from local cache/DB.
    offline_mode: bool = True
    allow_network: bool = False  # when False, the pipeline will never call yfinance network functions.

    # Universe build
    include_etfs: bool = True
    include_equities: bool = True
    exclude_leveraged_inverse_etfs: bool = True

    # Soft filters (used as penalties, not hard constraints)
    soft_min_market_cap_usd: float = 100e6   # $100M
    soft_min_dollar_volume_usd: float = 1e6  # $1M/day, approx

    # Price history / features
    lookback_days: int = 252
    min_coverage: float = 0.95
    return_type: str = "log_adj"   # "log_adj" or "simple_adj"
    corr_method: str = "pearson"   # "pearson" or "spearman"

    # Incremental update rules
    # If DB has >= lookback_days rows for a symbol and latest date is within this many calendar days,
    # we treat it as "fresh" and skip any pulls/imports for that symbol.
    freshness_days: int = 5

    # Basket optimization
    basket_size: int = 30
    candidate_pool_size: int = 300  # after filtering/ranking
    n_seeds: int = 30               # seeds for optimizer
    n_iterations: int = 3000        # SA iterations per seed
    top_percentile: float = 0.01    # top 1%

    # Penalty weights
    lambda_market_cap: float = 0.10
    lambda_dollar_vol: float = 0.10
    lambda_sector_concentration: float = 0.05

    # yfinance throttling (only used when allow_network=True)
    yf_batch_size: int = 25
    yf_batch_sleep_s: float = 5.0
    yf_max_retries: int = 6
    yf_min_delay_s: float = 0.3  # small delay between metadata calls (if enabled)

    # Misc
    random_seed: int = 42

    def ensure_paths(self):
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        # Cache conventions:
        # - prices: {cache_dir}/prices/{SYMBOL}.csv (columns: Date,Open,High,Low,Close,Adj Close,Volume)
        # - metadata: {cache_dir}/meta/{SYMBOL}.json (optional)
        Path(self.cache_dir, "prices").mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir, "meta").mkdir(parents=True, exist_ok=True)
        return self
