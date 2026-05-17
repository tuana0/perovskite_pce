import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib

BASE = Path(__file__).resolve().parent
INPUT = BASE / "spacelis_model_hazir_veri.csv"
OUT_RESULTS = BASE / "pce_model_benchmark_results.csv"
OUT_REPORT = BASE / "pce_model_benchmark_report.txt"
BEST_MODEL_PATH = BASE / "spacelis_best_pce_model.pkl"

TARGET = "JV_default_PCE"
LEAKAGE_PREFIXES = ("JV_default_", "JV_reverse_scan_")
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


def is_leakage_col(col: str) -> bool:
    if col in LEAKAGE_EXACT:
        return True
    if col == TARGET:
        return False
    return col.startswith(LEAKAGE_PREFIXES)


def load_xy():
    df = pd.read_csv(INPUT)
    df = df.drop(columns=df.columns[df.isna().all()].tolist())
    df = df.drop_duplicates().reset_index(drop=True)
    for col, thr in SANITY_THRESHOLDS.items():
        if col in df.columns:
            df = df[df[col] <= thr]
    df = df[df[TARGET].notna()].copy()

    leakage_cols = [c for c in df.columns if is_leakage_col(c)]
    X = df.drop(columns=[TARGET] + leakage_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number, "bool"]).replace([np.inf, -np.inf], np.nan)
    y = df[TARGET].astype(float)
    return X, y


def evaluate_model(name, estimator, X_train, X_test, y_train, y_test, cv):
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    hold_mae = mean_absolute_error(y_test, pred)
    hold_rmse = np.sqrt(mean_squared_error(y_test, pred))
    hold_r2 = r2_score(y_test, pred)

    cv_scores = cross_validate(
        pipe,
        pd.concat([X_train, X_test], axis=0),
        pd.concat([y_train, y_test], axis=0),
        cv=cv,
        scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
        n_jobs=1,
    )
    return {
        "model": name,
        "holdout_mae": hold_mae,
        "holdout_rmse": hold_rmse,
        "holdout_r2": hold_r2,
        "cv_mae_mean": (-cv_scores["test_mae"]).mean(),
        "cv_rmse_mean": (-cv_scores["test_rmse"]).mean(),
        "cv_r2_mean": cv_scores["test_r2"].mean(),
        "cv_r2_std": cv_scores["test_r2"].std(),
        "pipeline": pipe,
    }


def main():
    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    models = [
        ("RandomForest", RandomForestRegressor(n_estimators=160, random_state=42, n_jobs=1, min_samples_leaf=2)),
        ("ExtraTrees", ExtraTreesRegressor(n_estimators=180, random_state=42, n_jobs=1, min_samples_leaf=2)),
        (
            "XGBoost",
            XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=1,
                objective="reg:squarederror",
            ),
        ),
        (
            "CatBoost",
            CatBoostRegressor(
                iterations=600,
                depth=8,
                learning_rate=0.05,
                loss_function="RMSE",
                eval_metric="R2",
                random_seed=42,
                verbose=False,
            ),
        ),
        ("Ridge", Ridge(alpha=2.0)),
        ("KNN", KNeighborsRegressor(n_neighbors=25, weights="distance")),
    ]

    results = []
    for name, estimator in models:
        results.append(evaluate_model(name, estimator, X_train, X_test, y_train, y_test, cv))

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "pipeline"} for r in results])
    results_df = results_df.sort_values(["cv_r2_mean", "holdout_r2"], ascending=False).reset_index(drop=True)
    results_df.to_csv(OUT_RESULTS, index=False)

    best_name = results_df.loc[0, "model"]
    best_pipeline = next(r["pipeline"] for r in results if r["model"] == best_name)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    lines = [
        "PCE Model Benchmark Report",
        "=" * 28,
        f"Dataset rows: {len(X)}",
        f"Features: {X.shape[1]}",
        "",
        "Models ranked by CV R2:",
    ]
    for _, row in results_df.iterrows():
        lines.append(
            f"- {row['model']}: holdout R2={row['holdout_r2']:.4f}, CV R2={row['cv_r2_mean']:.4f} +- {row['cv_r2_std']:.4f}, "
            f"holdout MAE={row['holdout_mae']:.4f}, holdout RMSE={row['holdout_rmse']:.4f}"
        )
    lines.append("")
    lines.append(f"Best model saved: {BEST_MODEL_PATH.name} ({best_name})")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
