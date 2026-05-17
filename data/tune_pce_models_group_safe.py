import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

BASE = Path(__file__).resolve().parent
RAW = BASE / "perovskite_data.csv"
OUT_REPORT = BASE / "tuned_pce_group_safe_report.txt"
OUT_TABLE = BASE / "tuned_pce_group_safe_results.csv"
OUT_MODEL = BASE / "tuned_pce_group_safe_best_model.pkl"

TARGET = "JV_default_PCE"
GROUP_COL = "Ref_DOI_number"

LEAKAGE_PREFIXES = ("JV_default_", "JV_reverse_scan_", "JV_forward_scan_")
LEAKAGE_EXACT = {
    "JV_measured",
    "JV_average_over_n_number_of_cells",
    "JV_certified_values",
    "Stabilised_performance_PCE",
    "Stabilised_performance_Vmp",
    "Stabilised_performance_Jmp",
    "EQE_integrated_Jsc",
}
SANITY_THRESHOLDS = {
    "JV_default_FF": 1.0,
    "JV_default_Voc": 2.0,
    "JV_default_Jsc": 50.0,
    "JV_default_PCE": 40.0,
}
DROP_PREFIXES = ("Ref_",)


def is_leakage_col(col: str) -> bool:
    if col in LEAKAGE_EXACT:
        return True
    if col == TARGET:
        return False
    return col.startswith(LEAKAGE_PREFIXES)


def load_xyg():
    df = pd.read_csv(RAW, low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(f"Target column not found: {TARGET}")
    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column not found: {GROUP_COL}")

    df = df[df[TARGET].notna()].copy()
    for col, thr in SANITY_THRESHOLDS.items():
        if col in df.columns:
            df = df[df[col] <= thr]
    df = df[(df[TARGET] >= 0.0) & (df[TARGET] <= 40.0)].copy()

    leakage_cols = [c for c in df.columns if is_leakage_col(c)]
    drop_ref_cols = [c for c in df.columns if c.startswith(DROP_PREFIXES)]

    X = df.drop(columns=[TARGET] + leakage_cols + drop_ref_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number, "bool"]).replace([np.inf, -np.inf], np.nan)
    # Remove columns that are entirely missing to avoid repeated imputer warnings.
    X = X.dropna(axis=1, how="all")
    # Drop extremely sparse columns; these often become all-missing inside CV folds.
    min_non_null = max(10, int(len(X) * 0.01))
    keep_cols = X.columns[X.notna().sum() >= min_non_null]
    X = X.loc[:, keep_cols]
    y = pd.to_numeric(df[TARGET], errors="coerce")
    groups = df[GROUP_COL].fillna("Unknown_DOI")

    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    groups = groups.loc[mask].reset_index(drop=True)
    return X, y, groups


def make_search(estimator, param_space, cv):
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )
    return RandomizedSearchCV(
        pipe,
        param_distributions=param_space,
        n_iter=30,
        scoring="r2",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=0,
    )


def main():
    X, y, groups = load_xyg()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]
    g_train = groups.iloc[tr]

    cv = GroupKFold(n_splits=3)

    rf = RandomForestRegressor(random_state=42, n_jobs=1)
    rf_space = {
        "model__n_estimators": [180, 260, 360, 500],
        "model__max_depth": [None, 16, 24, 32, 40],
        "model__min_samples_split": [2, 4, 8, 12],
        "model__min_samples_leaf": [1, 2, 4, 6],
        "model__max_features": ["sqrt", 0.5, 0.7, 1.0],
    }

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    xgb_space = {
        "model__n_estimators": [220, 320, 450, 650],
        "model__max_depth": [5, 7, 9, 11],
        "model__learning_rate": [0.02, 0.04, 0.06, 0.08],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0, 8.0],
        "model__min_child_weight": [1, 3, 5, 8],
    }

    rf_search = make_search(rf, rf_space, cv)
    rf_search.fit(X_train, y_train, groups=g_train)
    rf_pred = rf_search.best_estimator_.predict(X_test)

    xgb_search = make_search(xgb, xgb_space, cv)
    xgb_search.fit(X_train, y_train, groups=g_train)
    xgb_pred = xgb_search.best_estimator_.predict(X_test)

    rows = [
        {
            "model": "RF_tuned_group_safe",
            "cv_r2_best": rf_search.best_score_,
            "holdout_r2": r2_score(y_test, rf_pred),
            "holdout_mae": mean_absolute_error(y_test, rf_pred),
            "holdout_rmse": np.sqrt(mean_squared_error(y_test, rf_pred)),
            "best_params": str(rf_search.best_params_),
        },
        {
            "model": "XGB_tuned_group_safe",
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
    best_model = rf_search.best_estimator_ if best_model_name == "RF_tuned_group_safe" else xgb_search.best_estimator_
    joblib.dump(best_model, OUT_MODEL)

    report = [
        "Tuned PCE Model Report (Group-Safe)",
        "=" * 34,
        f"Rows used: {len(X)}",
        f"Features used: {X.shape[1]}",
        f"Group column: {GROUP_COL}",
        f"Unique groups: {groups.nunique()}",
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
