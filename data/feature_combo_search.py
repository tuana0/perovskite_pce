import itertools
import random
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

BASE = Path(__file__).resolve().parent
INPUT = BASE / "perovskite_data.csv"
OUT_ALL = BASE / "feature_combo_search_results.csv"
OUT_TOP = BASE / "feature_combo_search_top20.csv"
OUT_REPORT = BASE / "feature_combo_search_report.txt"
TARGET = "JV_default_PCE"
GROUP_COL = "Ref_DOI_number"

RANDOM_STATE = 42
MAX_PER_K = 500
TOP_POOL_SIZE = 40


def parse_first_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in {"unknown", "nan", "none"}:
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except ValueError:
        return np.nan


def load_data():
    df = pd.read_csv(INPUT, low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if GROUP_COL not in df.columns:
        raise ValueError(f"Missing group column: {GROUP_COL}")

    y = pd.to_numeric(df[TARGET], errors="coerce")
    groups = df[GROUP_COL].fillna("Unknown_DOI")
    X = df.drop(columns=[TARGET]).copy()

    leak = [
        c
        for c in X.columns
        if c.startswith("JV_")
        or c.startswith("EQE_")
        or c.startswith("Stabilised_performance_")
        or c.startswith("Stability_")
    ]
    for c in ["JV_measured", "JV_average_over_n_number_of_cells", "JV_certified_values"]:
        if c in X.columns:
            leak.append(c)
    X = X.drop(columns=leak, errors="ignore")

    drop_ref_cols = [c for c in X.columns if c.startswith("Ref_")]
    X = X.drop(columns=drop_ref_cols, errors="ignore")

    num_parts = {}
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            num_parts[c] = pd.to_numeric(s, errors="coerce")
        else:
            parsed = s.map(parse_first_float)
            if parsed.notna().mean() >= 0.75:
                num_parts[c] = parsed

    X = pd.DataFrame(num_parts, index=df.index)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    min_non_null = max(10, int(len(X) * 0.01))
    keep_cols = X.columns[X.notna().sum() >= min_non_null]
    X = X.loc[:, keep_cols]
    X = X.fillna(X.median(numeric_only=True))

    m = y.notna() & (y >= 0) & (y <= 40)
    return (
        X.loc[m].reset_index(drop=True),
        y.loc[m].reset_index(drop=True),
        groups.loc[m].reset_index(drop=True),
    )


def rank_features(X, y, n=TOP_POOL_SIZE):
    model = RandomForestRegressor(
        n_estimators=140, random_state=RANDOM_STATE, n_jobs=1, min_samples_leaf=2
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


def eval_combo(X, y, groups, cols):
    Xs = X.loc[:, list(cols)]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(gss.split(Xs, y, groups=groups))
    Xtr, Xte = Xs.iloc[tr], Xs.iloc[te]
    ytr, yte = y.iloc[tr], y.iloc[te]
    gtr = groups.iloc[tr]

    cv = GroupKFold(n_splits=3)
    cv_scores = []
    for tr_cv, va_cv in cv.split(Xtr, ytr, groups=gtr):
        xtr_cv, xva_cv = Xtr.iloc[tr_cv], Xtr.iloc[va_cv]
        ytr_cv, yva_cv = ytr.iloc[tr_cv], ytr.iloc[va_cv]
        m_cv = RandomForestRegressor(
            n_estimators=120, random_state=RANDOM_STATE, n_jobs=1, min_samples_leaf=2
        )
        m_cv.fit(xtr_cv, ytr_cv)
        p_cv = m_cv.predict(xva_cv)
        cv_scores.append(r2_score(yva_cv, p_cv))

    m = RandomForestRegressor(
        n_estimators=180, random_state=RANDOM_STATE, n_jobs=1, min_samples_leaf=2
    )
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    return (
        float(np.mean(cv_scores)),
        float(np.std(cv_scores)),
        r2_score(yte, p),
        mean_absolute_error(yte, p),
        np.sqrt(mean_squared_error(yte, p)),
    )


def main():
    X, y, groups = load_data()
    top_features, _ = rank_features(X, y)

    rows = []
    for k in [3, 4, 5, 6, 7, 8]:
        combos = sample_combinations(top_features, k, MAX_PER_K)
        for cols in combos:
            cv_r2_mean, cv_r2_std, holdout_r2, mae, rmse = eval_combo(X, y, groups, cols)
            rows.append(
                {
                    "k": k,
                    "features": " | ".join(cols),
                    "cv_r2_mean": cv_r2_mean,
                    "cv_r2_std": cv_r2_std,
                    "r2": holdout_r2,
                    "mae": mae,
                    "rmse": rmse,
                }
            )

    res = pd.DataFrame(rows).sort_values(["cv_r2_mean", "r2"], ascending=False).reset_index(drop=True)
    res.to_csv(OUT_ALL, index=False)
    res.head(20).to_csv(OUT_TOP, index=False)

    best = res.iloc[0]
    lines = [
        "Feature Combination Search Report (DOI Group-Safe)",
        "=" * 50,
        f"Rows used: {len(X)}",
        f"Groups used: {groups.nunique()}",
        f"Feature pool size: {len(top_features)}",
        f"Pool features: {', '.join(top_features)}",
        f"Total combinations evaluated: {len(res)}",
        "",
        f"Best combo (k={int(best['k'])}):",
        best["features"],
        f"CV R2={best['cv_r2_mean']:.4f} +- {best['cv_r2_std']:.4f}",
        f"Holdout R2={best['r2']:.4f}, MAE={best['mae']:.4f}, RMSE={best['rmse']:.4f}",
        "",
        "Top 5 combos:",
    ]
    for _, r in res.head(5).iterrows():
        lines.append(
            f"- k={int(r['k'])}, CV R2={r['cv_r2_mean']:.4f}, Holdout R2={r['r2']:.4f}: {r['features']}"
        )

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()



