import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from xgboost import XGBRegressor
import joblib

BASE = Path(__file__).resolve().parent
RAW = BASE / "perovskite_data.csv"
READY = BASE / "spacelis_model_hazir_veri.csv"
OUT_REPORT = BASE / "tuned_pce_report.txt"
OUT_TABLE = BASE / "tuned_pce_results.csv"
OUT_MODEL = BASE / "tuned_best_pce_model.pkl"
TARGET = "JV_default_PCE"


def load_groups_from_raw():
    raw = pd.read_csv(RAW, usecols=["Ref_DOI_number", TARGET], low_memory=False)
    raw = raw[raw[TARGET].notna()].copy()
    return raw["Ref_DOI_number"].fillna("Unknown_DOI").reset_index(drop=True)


def main():
    df = pd.read_csv(READY)
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET]).copy()

    drop_cols = [c for c in X.columns if c.startswith("JV_default_") or c.startswith("JV_reverse_scan_")]
    for c in ["JV_measured", "JV_average_over_n_number_of_cells", "JV_certified_values"]:
        if c in X.columns:
            drop_cols.append(c)
    X = X.drop(columns=drop_cols, errors="ignore")
    X = X.dropna(axis=1, how="all").fillna(0.0)

    mask = (y >= 0.0) & (y <= 40.0)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    groups = load_groups_from_raw()
    if len(groups) >= len(y):
        groups = groups.iloc[: len(y)].reset_index(drop=True)
    else:
        groups = pd.Series(["Unknown_DOI"] * len(y))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]
    g_train = groups.iloc[tr]

    cv = GroupKFold(n_splits=3)

    rf = RandomForestRegressor(random_state=42, n_jobs=1)
    rf_space = {
        "n_estimators": [140, 180, 240],
        "max_depth": [None, 16, 24, 32],
        "min_samples_split": [2, 4, 8, 12],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": ["sqrt", 0.5, 0.7, 1.0],
    }
    rf_search = RandomizedSearchCV(
        rf,
        param_distributions=rf_space,
        n_iter=5,
        scoring="r2",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=0,
    )
    rf_search.fit(X_train, y_train, groups=g_train)
    rf_best = rf_search.best_estimator_
    rf_pred = rf_best.predict(X_test)

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    xgb_space = {
        "n_estimators": [180, 260, 360],
        "max_depth": [5, 7, 9, 11],
        "learning_rate": [0.02, 0.04, 0.06, 0.08],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "min_child_weight": [1, 3, 5, 8],
    }
    xgb_search = RandomizedSearchCV(
        xgb,
        param_distributions=xgb_space,
        n_iter=5,
        scoring="r2",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=0,
    )
    xgb_search.fit(X_train, y_train, groups=g_train)
    xgb_best = xgb_search.best_estimator_
    xgb_pred = xgb_best.predict(X_test)

    rows = [
        {
            "model": "RF_tuned",
            "cv_r2_best": rf_search.best_score_,
            "holdout_r2": r2_score(y_test, rf_pred),
            "holdout_mae": mean_absolute_error(y_test, rf_pred),
            "holdout_rmse": np.sqrt(mean_squared_error(y_test, rf_pred)),
            "best_params": str(rf_search.best_params_),
        },
        {
            "model": "XGB_tuned",
            "cv_r2_best": xgb_search.best_score_,
            "holdout_r2": r2_score(y_test, xgb_pred),
            "holdout_mae": mean_absolute_error(y_test, xgb_pred),
            "holdout_rmse": np.sqrt(mean_squared_error(y_test, xgb_pred)),
            "best_params": str(xgb_search.best_params_),
        },
    ]
    res = pd.DataFrame(rows).sort_values(["cv_r2_best", "holdout_r2"], ascending=False).reset_index(drop=True)
    res.to_csv(OUT_TABLE, index=False)

    best_model_name = res.loc[0, "model"]
    best_model = rf_best if best_model_name == "RF_tuned" else xgb_best
    joblib.dump(best_model, OUT_MODEL)

    report = [
        "Tuned PCE Model Report",
        "=" * 24,
        f"Rows used: {len(X)}",
        f"Features used: {X.shape[1]}",
        "",
    ]
    for _, r in res.iterrows():
        report.append(
            f"- {r['model']}: CV R2={r['cv_r2_best']:.4f}, Holdout R2={r['holdout_r2']:.4f}, "
            f"MAE={r['holdout_mae']:.4f}, RMSE={r['holdout_rmse']:.4f}"
        )
        report.append(f"  params: {r['best_params']}")
    report.append("")
    report.append(f"Best model saved: {OUT_MODEL.name} ({best_model_name})")
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
