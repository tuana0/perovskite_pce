import itertools
import random
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
INPUT = BASE / "spacelis_model_hazir_veri.csv"
OUT_ALL = BASE / "feature_combo_search_results.csv"
OUT_TOP = BASE / "feature_combo_search_top20.csv"
OUT_REPORT = BASE / "feature_combo_search_report.txt"
TARGET = "JV_default_PCE"

RANDOM_STATE = 42
MAX_PER_K = 220  # sampled combinations per k when full enumeration is too large
TOP_POOL_SIZE = 18  # pool to sample combinations from


def load_data():
    df = pd.read_csv(INPUT)
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET]).copy()

    leak = [c for c in X.columns if c.startswith("JV_default_") or c.startswith("JV_reverse_scan_")]
    for c in ["JV_measured", "JV_average_over_n_number_of_cells", "JV_certified_values"]:
        if c in X.columns:
            leak.append(c)
    X = X.drop(columns=leak, errors="ignore")
    X = X.dropna(axis=1, how="all").copy()
    X = X.select_dtypes(include=[np.number, "bool"]).copy()
    X = X.fillna(0.0)

    # physical target bounds
    m = (y >= 0) & (y <= 40)
    return X.loc[m].reset_index(drop=True), y.loc[m].reset_index(drop=True)


def rank_features(X, y, n=TOP_POOL_SIZE):
    model = RandomForestRegressor(
        n_estimators=120, random_state=RANDOM_STATE, n_jobs=1, min_samples_leaf=2
    )
    model.fit(X, y)
    imp = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp.head(n)["feature"].tolist(), imp


def sample_combinations(features, k, max_count=MAX_PER_K):
    all_count = int(math.comb(len(features), k))
    if all_count <= max_count:
        return list(itertools.combinations(features, k))
    rng = random.Random(RANDOM_STATE + k)
    combos = set()
    while len(combos) < max_count:
        c = tuple(sorted(rng.sample(features, k)))
        combos.add(c)
    return list(combos)


def eval_combo(X, y, cols):
    Xs = X.loc[:, list(cols)]
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=RANDOM_STATE)
    m = RandomForestRegressor(
        n_estimators=80, random_state=RANDOM_STATE, n_jobs=1, min_samples_leaf=2
    )
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    return (
        r2_score(yte, p),
        mean_absolute_error(yte, p),
        np.sqrt(mean_squared_error(yte, p)),
    )


def main():
    X, y = load_data()
    top_features, imp_df = rank_features(X, y)

    rows = []
    for k in [2, 3, 4, 5]:
        combos = sample_combinations(top_features, k, MAX_PER_K)
        for cols in combos:
            r2, mae, rmse = eval_combo(X, y, cols)
            rows.append(
                {
                    "k": k,
                    "features": " | ".join(cols),
                    "r2": r2,
                    "mae": mae,
                    "rmse": rmse,
                }
            )

    res = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    res.to_csv(OUT_ALL, index=False)
    res.head(20).to_csv(OUT_TOP, index=False)

    best = res.iloc[0]
    lines = [
        "Feature Combination Search Report",
        "=" * 33,
        f"Rows used: {len(X)}",
        f"Feature pool size: {len(top_features)}",
        f"Pool features: {', '.join(top_features)}",
        f"Total combinations evaluated: {len(res)}",
        "",
        f"Best combo (k={int(best['k'])}):",
        best["features"],
        f"R2={best['r2']:.4f}, MAE={best['mae']:.4f}, RMSE={best['rmse']:.4f}",
        "",
        "Top 5 combos:",
    ]
    for _, r in res.head(5).iterrows():
        lines.append(
            f"- k={int(r['k'])}, R2={r['r2']:.4f}: {r['features']}"
        )
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
