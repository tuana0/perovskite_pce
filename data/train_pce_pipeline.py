import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

BASE = Path(__file__).resolve().parent
INPUT = BASE / "spacelis_model_hazir_veri.csv"
CLEAN_OUT = BASE / "spacelis_model_hazir_veri_clean_for_pce.csv"
MODEL_OUT = BASE / "spacelis_rf_pce_clean.pkl"
REPORT_OUT = BASE / "pce_model_report.txt"
FI_OUT = BASE / "pce_feature_importance_top30.csv"

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


def main() -> None:
    df = pd.read_csv(INPUT)
    if TARGET not in df.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    start_rows, start_cols = df.shape
    all_nan_cols = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=all_nan_cols)

    dup_count = int(df.duplicated().sum())
    if dup_count:
        df = df.drop_duplicates().reset_index(drop=True)

    for col, thr in SANITY_THRESHOLDS.items():
        if col in df.columns:
            df = df[df[col] <= thr]

    df = df[df[TARGET].notna()].copy()

    leakage_cols = [c for c in df.columns if is_leakage_col(c)]
    X = df.drop(columns=[TARGET] + leakage_cols, errors="ignore")
    y = df[TARGET].astype(float)

    X = X.select_dtypes(include=[np.number, "bool"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=120,
                    random_state=42,
                    n_jobs=1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
        n_jobs=1,
    )

    cv_mae = -cv_scores["test_mae"]
    cv_rmse = -cv_scores["test_rmse"]
    cv_r2 = cv_scores["test_r2"]

    clean_df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    clean_df.to_csv(CLEAN_OUT, index=False)
    joblib.dump(model, MODEL_OUT)

    rf = model.named_steps["rf"]
    fi = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    fi.head(30).to_csv(FI_OUT, index=False)

    lines = [
        "PCE Model Training Report",
        "=" * 30,
        f"Input file: {INPUT.name}",
        f"Rows x Cols (raw): {start_rows} x {start_cols}",
        f"Fully-NaN columns removed: {len(all_nan_cols)}",
        f"Duplicate rows removed: {dup_count}",
        f"Rows after cleaning: {len(df)}",
        f"Leakage columns excluded: {len(leakage_cols)}",
        f"Features used: {X.shape[1]}",
        f"Target: {TARGET}",
        "",
        "Holdout (20%) metrics:",
        f"MAE  : {mae:.4f}",
        f"RMSE : {rmse:.4f}",
        f"R2   : {r2:.4f}",
        "",
        "3-Fold CV metrics (mean +- std):",
        f"MAE  : {cv_mae.mean():.4f} +- {cv_mae.std():.4f}",
        f"RMSE : {cv_rmse.mean():.4f} +- {cv_rmse.std():.4f}",
        f"R2   : {cv_r2.mean():.4f} +- {cv_r2.std():.4f}",
        "",
        f"Saved cleaned dataset: {CLEAN_OUT.name}",
        f"Saved model: {MODEL_OUT.name}",
        f"Saved top features: {FI_OUT.name}",
    ]

    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
