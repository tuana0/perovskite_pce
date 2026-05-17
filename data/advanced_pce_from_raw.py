import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import joblib


BASE = Path(__file__).resolve().parent
RAW = BASE / "perovskite_data.csv"
OUT_REPORT = BASE / "advanced_pce_report.txt"
OUT_TABLE = BASE / "advanced_pce_results.csv"
OUT_MODEL = BASE / "advanced_best_pce_model.pkl"

TARGET = "JV_default_PCE"
GROUP_COL = "Ref_DOI_number"

# Performance or post-measurement fields that can leak target information
LEAK_PREFIXES = (
    "JV_default_",
    "JV_reverse_scan_",
    "JV_forward_scan_",
    "Stabilised_performance_",
    "EQE_integrated_",
    "Stability_PCE_",
    "Outdoor_PCE_",
)
LEAK_EXACT = {
    "JV_measured",
    "JV_average_over_n_number_of_cells",
    "JV_certified_values",
}

# High-level text/id fields to skip
SKIP_PREFIXES = ("Ref_",)
SKIP_EXACT = {
    "Ref_internal_sample_id",
    "Ref_free_text_comment",
    "Ref_original_filename_data_upload",
    "JV_link_raw_data",
    "EQE_link_raw_data",
    "Stability_link_raw_data_for_stability_trace",
    "Outdoor_link_raw_data_for_outdoor_trace",
    "Outdoor_link_detailed_weather_data",
    "Outdoor_link_spectral_data",
    "Outdoor_link_irradiance_data",
}


def parse_first_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in {"unknown", "nan", "none", "false", "true"}:
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return np.nan
    return np.nan


def is_leak(col):
    if col == TARGET:
        return False
    if col in LEAK_EXACT:
        return True
    return col.startswith(LEAK_PREFIXES)


def is_skip(col):
    if col in SKIP_EXACT:
        return True
    return col.startswith(SKIP_PREFIXES)


def main():
    df = pd.read_csv(RAW, low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target: {TARGET}")

    df = df[df[TARGET].notna()].copy()
    df = df[df[TARGET] <= 40.0].copy()
    df = df[df[TARGET] >= 0.0].copy()

    # Build candidate columns
    all_cols = [c for c in df.columns if c != TARGET and not is_leak(c) and not is_skip(c)]

    numeric_cols = []
    categorical_cols = []
    tmp_numeric = {}

    for c in all_cols:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            numeric_cols.append(c)
            continue
        parsed = col.map(parse_first_float)
        ratio = parsed.notna().mean()
        # Keep as numeric if parseable enough
        if ratio >= 0.65:
            numeric_cols.append(c)
            tmp_numeric[c] = parsed
        else:
            cardinality = col.nunique(dropna=True)
            if cardinality <= 120:
                categorical_cols.append(c)

    X = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        if c in tmp_numeric:
            X[c] = tmp_numeric[c]
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce")

    for c in categorical_cols:
        X[c] = df[c].astype(str).fillna("Unknown")

    y = pd.to_numeric(df[TARGET], errors="coerce")
    groups = df[GROUP_COL].fillna("Unknown_DOI") if GROUP_COL in df.columns else pd.Series(["g"] * len(df), index=df.index)

    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    groups = groups.loc[mask]

    num_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_features = [c for c in X.columns if c not in num_features]

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=40))]), cat_features),
        ]
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
    g_train = groups.iloc[tr_idx]

    models = [
        ("RF_raw", RandomForestRegressor(n_estimators=220, min_samples_leaf=2, random_state=42, n_jobs=1)),
        (
            "XGB_raw",
            XGBRegressor(
                n_estimators=700,
                max_depth=9,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]

    gkf = GroupKFold(n_splits=3)
    rows = []
    best = None
    best_score = -np.inf

    for name, est in models:
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        h_r2 = r2_score(y_test, pred)
        h_mae = mean_absolute_error(y_test, pred)
        h_rmse = np.sqrt(mean_squared_error(y_test, pred))

        cv = cross_validate(
            pipe,
            X_train,
            y_train,
            groups=g_train,
            cv=gkf,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
            n_jobs=1,
        )
        cv_r2 = cv["test_r2"]
        cv_mae = -cv["test_mae"]
        cv_rmse = -cv["test_rmse"]

        rows.append(
            {
                "model": name,
                "holdout_r2": h_r2,
                "holdout_mae": h_mae,
                "holdout_rmse": h_rmse,
                "cv_r2_mean": cv_r2.mean(),
                "cv_r2_std": cv_r2.std(),
                "cv_mae_mean": cv_mae.mean(),
                "cv_rmse_mean": cv_rmse.mean(),
            }
        )
        if cv_r2.mean() > best_score:
            best_score = cv_r2.mean()
            best = (name, pipe)

    out = pd.DataFrame(rows).sort_values(["cv_r2_mean", "holdout_r2"], ascending=False).reset_index(drop=True)
    out.to_csv(OUT_TABLE, index=False)
    joblib.dump(best[1], OUT_MODEL)

    lines = [
        "Advanced PCE Modeling Report (Raw + Group Split)",
        "=" * 48,
        f"Rows used: {len(X)}",
        f"Numeric features: {len(num_features)}",
        f"Categorical features: {len(cat_features)}",
        f"Total input columns after filtering: {X.shape[1]}",
        "",
        "Results (ranked by CV R2):",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"- {r['model']}: holdout R2={r['holdout_r2']:.4f}, CV R2={r['cv_r2_mean']:.4f} +- {r['cv_r2_std']:.4f}, "
            f"MAE={r['holdout_mae']:.4f}, RMSE={r['holdout_rmse']:.4f}"
        )
    lines.append("")
    lines.append(f"Best model: {best[0]}")
    lines.append(f"Saved model: {OUT_MODEL.name}")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

