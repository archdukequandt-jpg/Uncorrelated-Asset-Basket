import os
import io
import json
import contextlib
from pathlib import Path

import streamlit as st
import pandas as pd

from config import Config
from pipeline import run_pipeline, DEFAULT_SYMBOLS

st.set_page_config(page_title="Uncorrelated Basket Finder", layout="wide")

APP_DIR = Path(__file__).resolve().parent

st.title("Uncorrelated Basket Finder")
st.caption("Build a low-correlation basket using your local SQLite DB + optional local cache. Designed for GitHub/Streamlit Cloud.")

with st.expander("How to use", expanded=True):
    st.markdown(
        """
1. Put your database file (default: `assets.sqlite`) in the **same folder as this app**.
2. (Optional) Provide cached CSV prices under `data_cache/prices/` if you're running offline.
3. Click **Run pipeline** to rank the universe, load prices, compute returns/correlation, and optimize the basket.

**Notes**
- By default this project is configured to run **offline** (no network calls).
- If you enable network mode, the pipeline may attempt yfinance calls (not recommended on Streamlit Cloud).
        """
    )

# ----------------------------
# Sidebar: Paths + Mode
# ----------------------------
st.sidebar.header("Data / Mode")

default_db = str(APP_DIR / "assets.sqlite")
db_path = st.sidebar.text_input("SQLite DB path", value=os.environ.get("ASSETS_DB_PATH", default_db))

offline_mode = st.sidebar.checkbox("Offline mode", value=True, help="When ON, pipeline avoids network calls and uses DB/cache only.")
allow_network = st.sidebar.checkbox("Allow network", value=False, help="When ON, pipeline may call yfinance (not recommended on Streamlit Cloud).")

cache_dir = st.sidebar.text_input("Cache dir", value=str(APP_DIR / "data_cache"))
data_dir = st.sidebar.text_input("Data dir", value=str(APP_DIR / "data_cache"))

if not Path(db_path).exists():
    st.error(
        f"Database not found at: `{db_path}`\n\n"
        f"Put `assets.sqlite` in the same folder as `app.py` or change the path above."
    )
    st.stop()

# ----------------------------
# Sidebar: Universe / Filters
# ----------------------------
st.sidebar.header("Universe & Filters")

include_etfs = st.sidebar.checkbox("Include ETFs", value=True)
include_equities = st.sidebar.checkbox("Include Equities", value=True)
exclude_lev_inv = st.sidebar.checkbox("Exclude leveraged/inverse", value=True)

soft_min_market_cap = st.sidebar.number_input("Soft min market cap ($)", min_value=0.0, value=100_000_000.0, step=10_000_000.0)
soft_min_dollar_vol = st.sidebar.number_input("Soft min dollar volume ($/day)", min_value=0.0, value=1_000_000.0, step=100_000.0)

# ----------------------------
# Sidebar: Returns / Corr
# ----------------------------
st.sidebar.header("Returns / Correlation")

lookback_days = st.sidebar.number_input("Lookback days", min_value=60, max_value=3650, value=252, step=21)
min_coverage = st.sidebar.slider("Min coverage", min_value=0.50, max_value=1.00, value=0.95, step=0.01)

return_type = st.sidebar.selectbox("Return type", options=["log_adj", "simple_adj"], index=0)
corr_method = st.sidebar.selectbox("Correlation method", options=["pearson", "spearman"], index=0)

freshness_days = st.sidebar.number_input("Freshness days (DB skip rule)", min_value=0, max_value=60, value=5, step=1)

# ----------------------------
# Sidebar: Optimization
# ----------------------------
st.sidebar.header("Optimization")

basket_size = st.sidebar.number_input("Basket size", min_value=5, max_value=200, value=30, step=1)
candidate_pool_size = st.sidebar.number_input("Candidate pool size", min_value=50, max_value=5000, value=300, step=50)
n_seeds = st.sidebar.number_input("Number of seeds", min_value=1, max_value=200, value=30, step=1)
n_iterations = st.sidebar.number_input("Iterations per seed", min_value=200, max_value=200_000, value=3000, step=200)
top_percentile = st.sidebar.slider("Top percentile (DB flagging)", min_value=0.001, max_value=0.25, value=0.01, step=0.001)

st.sidebar.header("Penalty weights")
lambda_market_cap = st.sidebar.slider("λ Market cap", 0.0, 1.0, 0.10, 0.01)
lambda_dollar_vol = st.sidebar.slider("λ Dollar volume", 0.0, 1.0, 0.10, 0.01)
lambda_sector_conc = st.sidebar.slider("λ Sector concentration", 0.0, 1.0, 0.05, 0.01)

random_seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000_000, value=42, step=1)

# ----------------------------
# Seed symbols editor
# ----------------------------
st.subheader("Seed symbols")
seed_symbols_text = st.text_area(
    "Symbols (comma or newline separated)",
    value=",".join(DEFAULT_SYMBOLS),
    height=120
)
seed_symbols = [s.strip().upper() for s in seed_symbols_text.replace("\n", ",").split(",") if s.strip()]

col_run, col_export = st.columns([1, 1])

# Store results in session
if "result" not in st.session_state:
    st.session_state.result = None
if "logs" not in st.session_state:
    st.session_state.logs = ""

with col_run:
    run_clicked = st.button("Run pipeline", type="primary")

if run_clicked:
    cfg = Config(
        db_path=str(db_path),
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        offline_mode=bool(offline_mode),
        allow_network=bool(allow_network),

        include_etfs=bool(include_etfs),
        include_equities=bool(include_equities),
        exclude_leveraged_inverse_etfs=bool(exclude_lev_inv),

        soft_min_market_cap_usd=float(soft_min_market_cap),
        soft_min_dollar_volume_usd=float(soft_min_dollar_vol),

        lookback_days=int(lookback_days),
        min_coverage=float(min_coverage),
        return_type=str(return_type),
        corr_method=str(corr_method),
        freshness_days=int(freshness_days),

        basket_size=int(basket_size),
        candidate_pool_size=int(candidate_pool_size),
        n_seeds=int(n_seeds),
        n_iterations=int(n_iterations),
        top_percentile=float(top_percentile),

        lambda_market_cap=float(lambda_market_cap),
        lambda_dollar_vol=float(lambda_dollar_vol),
        lambda_sector_concentration=float(lambda_sector_conc),

        random_seed=int(random_seed),
    ).ensure_paths()

    st.info("Running pipeline… this can take a bit depending on DB size and iterations.")
    log_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(log_buf):
            result = run_pipeline(cfg, seed_symbols=seed_symbols)
        st.session_state.result = result
        st.session_state.logs = log_buf.getvalue()
        st.success("Pipeline complete.")
    except Exception as e:
        st.session_state.result = None
        st.session_state.logs = log_buf.getvalue()
        st.error(f"Pipeline failed: {e}")

# ----------------------------
# Tabs: Results / Logs / Data
# ----------------------------
tab_res, tab_logs, tab_data = st.tabs(["Results", "Logs", "Returns & Correlation"])

with tab_logs:
    st.subheader("Run logs")
    st.code(st.session_state.logs or "No logs yet.", language="text")

with tab_res:
    st.subheader("Best basket")

    result = st.session_state.result
    if not result:
        st.info("Run the pipeline to see results.")
    else:
        best_symbols = result["best_symbols"]
        best_score = result["best_score"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Basket size", len(best_symbols))
        c2.metric("Objective score", f"{best_score:.6f}")
        c3.metric("Run ID", str(result.get("run_id")))
        c4.metric("Snapshot ID", str(result.get("snapshot_id")))

        st.write("**Tickers**")
        st.dataframe(pd.DataFrame({"symbol": best_symbols}), use_container_width=True, height=420)

        # Show summary table if present
        summary_df = result.get("summary_df")
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            st.write("**Summary**")
            st.dataframe(summary_df, use_container_width=True)

        # Download tickers
        csv = pd.DataFrame({"symbol": best_symbols}).to_csv(index=False).encode("utf-8")
        st.download_button("Download basket CSV", data=csv, file_name="best_basket.csv", mime="text/csv")

        # Top candidates
        candidates = result.get("candidates", [])
        if candidates:
            st.write("**Top candidate baskets (by score)**")
            topn = min(25, len(candidates))
            cand_rows = []
            for c in candidates[:topn]:
                cand_rows.append({
                    "score": float(c.get("score", float("nan"))),
                    "symbols": ", ".join(c.get("symbols", [])[:60]) + ("…" if len(c.get("symbols", [])) > 60 else ""),
                    "basket_id": c.get("basket_id"),
                })
            st.dataframe(pd.DataFrame(cand_rows), use_container_width=True)

with tab_data:
    st.subheader("Returns matrix & correlation heatmap")

    result = st.session_state.result
    if not result:
        st.info("Run the pipeline to see returns/correlation.")
    else:
        ret_df = result.get("returns_df")
        corr = result.get("corr")

        if isinstance(ret_df, pd.DataFrame):
            st.write(f"Returns dataframe: `{ret_df.shape[0]}` rows × `{ret_df.shape[1]}` symbols")
            st.dataframe(ret_df.tail(30), use_container_width=True)

        if isinstance(corr, pd.DataFrame):
            st.write("Correlation matrix (best basket subset)")
            best_symbols = result["best_symbols"]
            corr_best = corr.loc[best_symbols, best_symbols]

            # Simple heatmap (matplotlib) without forcing colors
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.imshow(corr_best.values, aspect="auto")
            ax.set_title("Correlation Heatmap (Best Basket)")
            ax.set_xticks(range(len(best_symbols)))
            ax.set_yticks(range(len(best_symbols)))
            # avoid unreadable labels if big
            if len(best_symbols) <= 40:
                ax.set_xticklabels(best_symbols, rotation=90, fontsize=7)
                ax.set_yticklabels(best_symbols, fontsize=7)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig, clear_figure=True)

        st.caption("Tip: if you see 'Not enough symbols with sufficient data', add cached price CSVs in `data_cache/prices/` or enable network mode.")
