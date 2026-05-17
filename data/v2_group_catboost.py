import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import joblib

BASE = Path(__file__).resolve().parent
RAW = BASE / "perovskite_data.csv"
OUT_REPORT = BASE / "v2_group_catboost_report.txt"
OUT_PRED = BASE / "v2_group_catboost_holdout_predictions.csv"
OUT_MODEL = BASE / "v2_group_catboost_model.cbm"
TARGET = "JV_default_PCE"
GROUP_COL = "Ref_DOI_number"

LEAK_PREFIXES = (
    "JV_default_",
    "JV_reverse_scan_",
    "JV_forward_scan_",
    "Stabilised_performance_",
    "EQE_integrated_",
    "Stability_PCE_",
    "Outdoor_PCE_",
)
LEAK_EXACT = {"JV_measured", "JV_average_over_n_number_of_cells", "JV_certified_values"}
SKIP_PREFIXES = ("Ref_",)


def parse_first_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in {"unknown", "nan", "none"}:
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return np.nan
    return np.nan


def is_leak(c):
    if c == TARGET:
        return False
    if c in LEAK_EXACT:
        return True
    return c.startswith(LEAK_PREFIXES)


def build_features(df):
    candidates = [c for c in df.columns if c != TARGET and not is_leak(c) and not c.startswith(SKIP_PREFIXES)]
    num_parts = {}
    cat_cols = []
    for c in candidates:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            miss = s.isna().mean()
            if miss < 0.98:
                num_parts[c] = pd.to_numeric(s, errors="coerce")
            continue
        parsed = s.map(parse_first_float)
        parsed_ratio = parsed.notna().mean()
        nunique = s.nunique(dropna=True)
        miss = s.isna().mean()
        if parsed_ratio >= 0.75 and miss < 0.98:
            num_parts[c] = parsed
        elif 2 <= nunique <= 80 and miss < 0.98:
            cat_cols.append(c)

    X_num = pd.DataFrame(num_parts, index=df.index)
    for c in X_num.columns:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")
    X_cat = pd.DataFrame(index=df.index)
    for c in cat_cols:
        X_cat[c] = df[c].astype(str).replace({"nan": "Unknown", "None": "Unknown"}).fillna("Unknown")

    X = pd.concat([X_num, X_cat], axis=1)
    # Drop columns that are nearly empty after parsing
    keep = X.columns[X.isna().mean() < 0.995].tolist()
    X = X[keep]
    active_cat_cols = [c for c in cat_cols if c in X.columns]
    return X, active_cat_cols


def fold_score(X, y, groups, params, cat_cols):
    gkf = GroupKFold(n_splits=2)
    fold_r2 = []
    fold_mae = []
    fold_rmse = []
    cat_idx = [i for i, c in enumerate(X.columns) if c in cat_cols]
    for tr, va in gkf.split(X, y, groups):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        model = CatBoostRegressor(**params)
        model.fit(Xtr, ytr, cat_features=cat_idx, eval_set=(Xva, yva), use_best_model=True, verbose=False)
        p = model.predict(Xva)
        fold_r2.append(r2_score(yva, p))
        fold_mae.append(mean_absolute_error(yva, p))
        fold_rmse.append(np.sqrt(mean_squared_error(yva, p)))
    return np.mean(fold_r2), np.std(fold_r2), np.mean(fold_mae), np.mean(fold_rmse)


def main():
    df = pd.read_csv(RAW, low_memory=False)
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].between(0.0, 40.0)].copy()
    df = df.drop_duplicates().reset_index(drop=True)
    groups = df[GROUP_COL].fillna("Unknown_DOI") if GROUP_COL in df.columns else pd.Series(["g"] * len(df))

    X, cat_cols = build_features(df)
    y = df[TARGET].reset_index(drop=True)
    X = X.reset_index(drop=True)
    groups = groups.reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]
    g_train = groups.iloc[tr]

    configs = [
        {"name": "cb_a", "iterations": 420, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 4.0},
        {"name": "cb_b", "iterations": 300, "depth": 7, "learning_rate": 0.07, "l2_leaf_reg": 3.0},
    ]
    base = {
        "loss_function": "RMSE",
        "eval_metric": "R2",
        "random_seed": 42,
        "nan_mode": "Min",
        "subsample": 0.85,
    }

    rows = []
    best_cfg = None
    best_cv = -1e9
    for cfg in configs:
        params = base | cfg
        params.pop("name")
        cv_r2_mean, cv_r2_std, cv_mae, cv_rmse = fold_score(X_train, y_train, g_train, params, cat_cols)
        rows.append({"model": cfg["name"], "cv_r2_mean": cv_r2_mean, "cv_r2_std": cv_r2_std, "cv_mae": cv_mae, "cv_rmse": cv_rmse, "params": str(params)})
        if cv_r2_mean > best_cv:
            best_cv = cv_r2_mean
            best_cfg = params

    cat_idx = [i for i, c in enumerate(X.columns) if c in cat_cols]
    best_model = CatBoostRegressor(**best_cfg)
    best_model.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_test, y_test), use_best_model=True, verbose=False)
    pred = best_model.predict(X_test)
    hold_r2 = r2_score(y_test, pred)
    hold_mae = mean_absolute_error(y_test, pred)
    hold_rmse = np.sqrt(mean_squared_error(y_test, pred))

    out = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)
    out_path = BASE / "v2_group_catboost_cv_results.csv"
    out.to_csv(out_path, index=False)

    pred_out = pd.DataFrame({"y_true": y_test.values, "y_pred": pred})
    pred_out.to_csv(OUT_PRED, index=False)
    best_model.save_model(str(OUT_MODEL))
    joblib.dump({"features": list(X.columns), "cat_idx": cat_idx}, BASE / "v2_group_catboost_metadata.pkl")

    lines = [
        "V2 Group CatBoost Report",
        "=" * 24,
        f"Rows used: {len(X)}",
        f"Features used: {X.shape[1]}",
        f"Categorical feature count: {len(cat_idx)}",
        "",
        "CV candidates:",
    ]
    for _, r in out.iterrows():
        lines.append(f"- {r['model']}: CV R2={r['cv_r2_mean']:.4f} +- {r['cv_r2_std']:.4f}, CV MAE={r['cv_mae']:.4f}, CV RMSE={r['cv_rmse']:.4f}")
    lines += [
        "",
        f"Holdout R2: {hold_r2:.4f}",
        f"Holdout MAE: {hold_mae:.4f}",
        f"Holdout RMSE: {hold_rmse:.4f}",
        f"Saved model: {OUT_MODEL.name}",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
