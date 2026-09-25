"""How the notebooks talk to each other: file paths, the column lists they agree on, and the train/test split.

All cleaning, feature and model logic lives in the notebooks:
    model_names.ipynb -> data/fixed_models.csv -> data_cleaning.ipynb
    data_cleaning.ipynb -> CLEAN_CSV -> EDA.ipynb
                                     -> modeling.ipynb (feature engineering) -> FEATURES_CSV, BEST_PARAMS_PATH
                                                                             -> final_model.ipynb -> MODEL_PATH

The dashboard only reads files the notebooks wrote:
    final_model.ipynb, dashboard_cleaning.ipynb, dashboard_duplicates.ipynb -> DASHBOARD_DIR (small CSVs) -> dashboard/
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
CLEAN_CSV = ROOT / "data" / "vehicles_clean.csv"
FEATURES_CSV = ROOT / "data" / "vehicles_features.csv"
BEST_PARAMS_PATH = ROOT / "data" / "best_params.json"
MODEL_PATH = ROOT / "models" / "final_model.joblib"
DASHBOARD_DIR = ROOT / "data" / "dashboard"

# columns of FEATURES_CSV
base_num_cols = ["age", "log_odometer", "miles_per_year", "cylinders_num", "is_classic"]  # iteration 1
keyword_cols = ["needs_work", "high_trim", "lifted", "diesel_words", "one_owner", "carfax",   # iteration 2,
                "warranty", "leather", "sunroof", "navigation", "turbo"]                     # from the description
listing_cols = ["is_dealer", "has_vin", "desc_len", "lat", "long"] + keyword_cols           # iteration 2
num_cols = base_num_cols + listing_cols
cat_cols = ["manufacturer", "condition", "fuel", "title_status", "transmission",
            "drive", "type", "paint_color", "state"]
cat_features = ["model"] + cat_cols


def load_split():
    """FEATURES_CSV -> the same 80/20 split every time. y is log1p(price)."""
    df = pd.read_csv(FEATURES_CSV)
    X = df.drop(columns=["price"])
    y = np.log1p(df["price"])
    return train_test_split(X, y, test_size=0.2, random_state=42)
