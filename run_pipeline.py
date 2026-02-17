from config import Config
from pipeline import run_pipeline

if __name__ == "__main__":
    cfg = Config(db_path="assets.sqlite").ensure_paths()
    result = run_pipeline(cfg)
    print("Best basket:", result["best_symbols"])
    print("Best score:", result["best_score"])
