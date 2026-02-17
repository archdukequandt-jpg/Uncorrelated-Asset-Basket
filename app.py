import os
import io
import contextlib
from pathlib import Path

import streamlit as st
import pandas as pd

from config import Config
from pipeline import run_pipeline, DEFAULT_SYMBOLS

st.set_page_config(page_title="Uncorrelated Basket Finder", layout="wide")

APP_DIR = Path(__file__).resolve().parent

st.title("Uncorrelated Basket Finder")
st.caption("Build a low-correlation basket using SQLite + optional cache. GitHub/Streamlit Cloud ready.")

with st.expander("How to use", expanded=True):
    st.markdown(
        """
1. Put your database file (default: `assets.sqlite`) in the **same folder as this app**.
2. Click **Run pipeline** to rank the universe, load prices, compute returns/correlation, and optimize the basket.

**Mode**
- Default is **online** (network allowed) so the app can fetch missing data via yfinance.
- If you want fully offline runs, toggle **Offline mode** ON and provide cached CSVs under `data_cache/prices/`.
        """
    )

# ----------------------------
# Sidebar: Paths + Mode
# ----------------------------
st.sidebar.header("Data / Mode")

default_db = str(APP_DIR / "assets.sqlite")
db_path = st.sidebar.text_input("SQLite DB path", value=os.environ.get("ASSETS_DB_PATH", default_db))

# Online-first defaults to avoid "not enough symbols" in small DBs
offline_mode = st.sidebar.checkbox("Offline mode", value=False, help="When ON, pipeline avoids network calls and uses DB/cache only.")
allow_network = st.sidebar.checkbox("Allow network", value=True, help="When ON, pipeline may call yfinance to fill missing/stale symbols.")

cache_dir = st.sidebar.text_input("Cache dir", value=str(APP_DIR / "data_cache"))
data_dir = st.sidebar.text_input("Data dir", value=str(APP_DIR / "data_cache"))

if not Path(db_path).exists():
    st.error(
        f"Database not found at: `{db_path}`\n\n"
        f"Put `assets.sqlite` in the same folder as `app.py` or change the path above."
    )
    st.stop()

# ----------------------------
# Sidebar: Returns / Corr / Optimization (keep it simple)
# ----------------------------
st.sidebar.header("Returns / Correlation")
lookback_days = st.sidebar.number_input("Lookback days", min_value=30, max_value=3650, value=252, step=21)
return_type = st.sidebar.selectbox("Return type", options=["log_adj", "simple_adj"], index=0)
corr_method = st.sidebar.selectbox("Correlation method", options=["pearson", "spearman"], index=0)

st.sidebar.header("Optimization")
basket_size = st.sidebar.number_input("Basket size", min_value=3, max_value=200, value=5, step=1)
candidate_pool_size = st.sidebar.number_input("Candidate pool size", min_value=20, max_value=5000, value=300, step=50)
n_seeds = st.sidebar.number_input("Seeds", min_value=1, max_value=200, value=20, step=1)
n_iterations = st.sidebar.number_input("Iterations per seed", min_value=200, max_value=200_000, value=2000, step=200)
top_percentile = st.sidebar.slider("Top percentile", min_value=0.001, max_value=0.25, value=0.01, step=0.001)

st.sidebar.header("Penalty weights")
lambda_market_cap = st.sidebar.slider("λ Market cap", 0.0, 1.0, 0.10, 0.01)
lambda_dollar_vol = st.sidebar.slider("λ Dollar volume", 0.0, 1.0, 0.10, 0.01)
lambda_sector_conc = st.sidebar.slider("λ Sector concentration", 0.0, 1.0, 0.05, 0.01)

random_seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000_000, value=42, step=1)

# ----------------------------
# Seed symbols
# ----------------------------
st.subheader("Seed symbols")
seed_symbols_text = st.text_area(
    "Symbols (comma or newline separated)",
    value=",".join(DEFAULT_SYMBOLS),
    height=110
)
seed_symbols = [s.strip().upper() for s in seed_symbols_text.replace("\n", ",").split(",") if s.strip()]

if "result" not in st.session_state:
    st.session_state.result = None
if "logs" not in st.session_state:
    st.session_state.logs = ""

run_clicked = st.button("Run pipeline", type="primary")

if run_clicked:
    cfg = Config(
        db_path=str(db_path),
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        offline_mode=bool(offline_mode),
        allow_network=bool(allow_network),

        lookback_days=int(lookback_days),
        return_type=str(return_type),
        corr_method=str(corr_method),

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

    st.info("Running pipeline…")
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

tab_res, tab_logs, tab_data = st.tabs(["Results", "Logs", "Correlation"])

with tab_logs:
    st.code(st.session_state.logs or "No logs yet.", language="text")

with tab_res:
    result = st.session_state.result
    if not result:
        st.info("Run the pipeline to see results.")
    else:
        best_symbols = result["best_symbols"]
        best_score = result["best_score"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Basket size", len(best_symbols))
        c2.metric("Objective score", f"{best_score:.6f}")
        c3.metric("Run ID", str(result.get("run_id")))

        st.dataframe(pd.DataFrame({"symbol": best_symbols}), use_container_width=True, height=400)
        csv = pd.DataFrame({"symbol": best_symbols}).to_csv(index=False).encode("utf-8")
        st.download_button("Download basket CSV", data=csv, file_name="best_basket.csv", mime="text/csv")

with tab_data:
    result = st.session_state.result
    if not result:
        st.info("Run the pipeline to see correlation.")
    else:
        corr = result.get("corr")
        best = result.get("best_symbols", [])
        if isinstance(corr, pd.DataFrame) and best:
            corr_best = corr.loc[best, best]
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.imshow(corr_best.values, aspect="auto")
            ax.set_title("Correlation Heatmap (Best Basket)")
            if len(best) <= 40:
                ax.set_xticks(range(len(best)))
                ax.set_yticks(range(len(best)))
                ax.set_xticklabels(best, rotation=90, fontsize=7)
                ax.set_yticklabels(best, fontsize=7)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig, clear_figure=True)
