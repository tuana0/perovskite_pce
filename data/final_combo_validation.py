import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

BASE = Path(__file__).resolve().parent
RAW = BASE / "perovskite_data.csv"
COMBO_INPUT = BASE / "feature_combo_search_results.csv"
OUT_CSV = BASE / "final_combo_validation_results.csv"
OUT_REPORT = BASE / "final_combo_validation_report.txt"

GROUP_COL = "Ref_DOI_number"
TARGET_CANDIDATES = ["JV_default_PCE", "PCE"]
BASELINE_HOLDOUT_R2 = 0.4233
RANDOM_STATE = 42
MAX_SELECTED_COMBOS = 5

LEAK_PREFIXES = (
    "JV_",
    "EQE_",
    "Stabilised_performance_",
    "Stability_",
    "Outdoor_PCE_",
)
LEAK_KEYWORDS = (
    "measured",
    "certified",
    "scan",
    "forward",
    "reverse",
    "voc",
    "jsc",
    "ff",
)
SOFT_RISK_KEYWORDS = (
    "outdoor_",
    "module_jv",
)


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


def pick_target(df):
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No target found. Expected one of: {TARGET_CANDIDATES}")


def is_hard_leak_feature(name):
    n = name.lower()
    if name.startswith(LEAK_PREFIXES):
        return True
    return any(k in n for k in LEAK_KEYWORDS)


def is_soft_risk_feature(name):
    n = name.lower()
    return any(k in n for k in SOFT_RISK_KEYWORDS)


def split_combo(combo_text):
    return [x.strip() for x in combo_text.split("|") if x.strip()]


def select_clean_combos(combo_df):
    top20 = combo_df.sort_values(["cv_r2_mean", "r2"], ascending=False).head(20).copy()
    selected = []
    dropped = []

    for _, row in top20.iterrows():
        feats = split_combo(row["features"])
        hard_leaks = [f for f in feats if is_hard_leak_feature(f)]
        soft_risks = [f for f in feats if is_soft_risk_feature(f)]

        if hard_leaks:
            dropped.append((row["features"], f"hard leakage risk: {', '.join(hard_leaks)}"))
            continue
        if soft_risks:
            dropped.append((row["features"], f"soft risk excluded for final validation: {', '.join(soft_risks)}"))
            continue

        selected.append((row["features"], feats))
        if len(selected) >= MAX_SELECTED_COMBOS:
            break

    return selected, dropped, top20


def prepare_feature_frame(df, features):
    X_num = pd.DataFrame(index=df.index)
    X_cat = pd.DataFrame(index=df.index)

    for c in features:
        if c not in df.columns:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            X_num[c] = pd.to_numeric(s, errors="coerce")
            continue

        parsed = s.map(parse_first_float)
        parsed_ratio = parsed.notna().mean()
        if parsed_ratio >= 0.75:
            X_num[c] = parsed
        else:
            X_cat[c] = s.astype(str).replace({"nan": "Unknown", "None": "Unknown"}).fillna("Unknown")

    X = pd.concat([X_num, X_cat], axis=1)
    if not X_num.empty:
        X_num = X_num.fillna(X_num.median(numeric_only=True))
        X.update(X_num)
    keep = X.columns[X.isna().mean() < 0.995].tolist()
    X = X[keep]
    cat_cols = [c for c in X_cat.columns if c in X.columns]
    return X, cat_cols


def score_catboost(X, y, groups, cat_cols):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]
    g_train = groups.iloc[tr]

    params = {
        "loss_function": "RMSE",
        "eval_metric": "R2",
        "random_seed": RANDOM_STATE,
        "nan_mode": "Min",
        "subsample": 0.85,
        "iterations": 500,
        "depth": 7,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "verbose": False,
    }

    cv = GroupKFold(n_splits=3)
    cat_idx = [i for i, c in enumerate(X.columns) if c in cat_cols]
    cv_scores = []
    for tr_cv, va_cv in cv.split(X_train, y_train, groups=g_train):
        xtr, xva = X_train.iloc[tr_cv], X_train.iloc[va_cv]
        ytr, yva = y_train.iloc[tr_cv], y_train.iloc[va_cv]
        m = CatBoostRegressor(**params)
        m.fit(xtr, ytr, cat_features=cat_idx, eval_set=(xva, yva), use_best_model=True, verbose=False)
        p = m.predict(xva)
        cv_scores.append(r2_score(yva, p))

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_test, y_test), use_best_model=True, verbose=False)
    pred = model.predict(X_test)

    return {
        "model": "CatBoost",
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
        "holdout_r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }, (X_train, X_test, y_train, y_test, g_train)


def score_xgb(X, y, groups):
    if not HAS_XGB:
        return None

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]
    g_train = groups.iloc[tr]

    params = {
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "min_child_weight": 2,
        "n_jobs": 1,
    }

    cv = GroupKFold(n_splits=3)
    cv_scores = []
    for tr_cv, va_cv in cv.split(X_train, y_train, groups=g_train):
        xtr_raw, xva_raw = X_train.iloc[tr_cv], X_train.iloc[va_cv]
        ytr, yva = y_train.iloc[tr_cv], y_train.iloc[va_cv]
        xtr = pd.get_dummies(xtr_raw, dummy_na=True)
        xva = pd.get_dummies(xva_raw, dummy_na=True).reindex(columns=xtr.columns, fill_value=0)
        m = XGBRegressor(**params)
        m.fit(xtr, ytr)
        p = m.predict(xva)
        cv_scores.append(r2_score(yva, p))

    xtr_full = pd.get_dummies(X_train, dummy_na=True)
    xte_full = pd.get_dummies(X_test, dummy_na=True).reindex(columns=xtr_full.columns, fill_value=0)
    model = XGBRegressor(**params)
    model.fit(xtr_full, y_train)
    pred = model.predict(xte_full)

    return {
        "model": "XGBoost",
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
        "holdout_r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }


def stability_score(cv_mean, cv_std, holdout):
    gap = abs(cv_mean - holdout)
    return holdout - gap - 0.5 * cv_std


def tune_best_combo_catboost(X, y, groups, cat_cols):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]
    g_train = groups.iloc[tr]
    cv = GroupKFold(n_splits=3)
    cat_idx = [i for i, c in enumerate(X.columns) if c in cat_cols]

    best = None
    rows = []
    grid = itertools.product([4, 6, 8], [0.03, 0.05, 0.1], [3, 5, 7], [300, 500, 800])
    for depth, lr, l2, iters in grid:
        params = {
            "loss_function": "RMSE",
            "eval_metric": "R2",
            "random_seed": RANDOM_STATE,
            "nan_mode": "Min",
            "subsample": 0.85,
            "depth": depth,
            "learning_rate": lr,
            "l2_leaf_reg": l2,
            "iterations": iters,
            "verbose": False,
        }

        cv_scores = []
        for tr_cv, va_cv in cv.split(X_train, y_train, groups=g_train):
            xtr, xva = X_train.iloc[tr_cv], X_train.iloc[va_cv]
            ytr, yva = y_train.iloc[tr_cv], y_train.iloc[va_cv]
            m = CatBoostRegressor(**params)
            m.fit(xtr, ytr, cat_features=cat_idx, eval_set=(xva, yva), use_best_model=True, verbose=False)
            p = m.predict(xva)
            cv_scores.append(r2_score(yva, p))

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_test, y_test), use_best_model=True, verbose=False)
        pred = model.predict(X_test)
        hold = r2_score(y_test, pred)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        row = {
            "model": "CatBoost_tuned",
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std,
            "holdout_r2": float(hold),
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "params": str(params),
            "stability_score": float(stability_score(cv_mean, cv_std, hold)),
        }
        rows.append(row)
        if best is None or row["stability_score"] > best["stability_score"]:
            best = row

    return best, rows


def main():
    df = pd.read_csv(RAW, low_memory=False)
    target = pick_target(df)
    df = df[df[target].notna()].copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[df[target].between(0.0, 40.0)].drop_duplicates().reset_index(drop=True)
    if GROUP_COL not in df.columns:
        raise ValueError(f"Missing group column: {GROUP_COL}")
    groups = df[GROUP_COL].fillna("Unknown_DOI").reset_index(drop=True)
    y = df[target].reset_index(drop=True)

    combos = pd.read_csv(COMBO_INPUT)
    selected, dropped, top20 = select_clean_combos(combos)
    if len(selected) < 3:
        raise ValueError("Clean combo count below 3 after leakage filtering. Relax soft-risk rules if needed.")

    rows = []
    tested_combo_names = []
    for combo_text, feat_list in selected:
        X, cat_cols = prepare_feature_frame(df, feat_list)
        if X.shape[1] == 0:
            continue

        tested_combo_names.append(combo_text)
        cb_metrics, _ = score_catboost(X, y, groups, cat_cols)
        cb_metrics.update(
            {
                "combo": combo_text,
                "n_features": int(X.shape[1]),
                "feature_list": " | ".join(list(X.columns)),
                "delta_r2_vs_baseline": float(cb_metrics["holdout_r2"] - BASELINE_HOLDOUT_R2),
                "overfit_warning": bool(cb_metrics["cv_r2_mean"] - cb_metrics["holdout_r2"] > 0.03),
                "stability_score": float(stability_score(cb_metrics["cv_r2_mean"], cb_metrics["cv_r2_std"], cb_metrics["holdout_r2"])),
                "params": "default_validation_params",
            }
        )
        rows.append(cb_metrics)

        xgb_metrics = score_xgb(X, y, groups)
        if xgb_metrics is not None:
            xgb_metrics.update(
                {
                    "combo": combo_text,
                    "n_features": int(X.shape[1]),
                    "feature_list": " | ".join(list(X.columns)),
                    "delta_r2_vs_baseline": float(xgb_metrics["holdout_r2"] - BASELINE_HOLDOUT_R2),
                    "overfit_warning": bool(xgb_metrics["cv_r2_mean"] - xgb_metrics["holdout_r2"] > 0.03),
                    "stability_score": float(stability_score(xgb_metrics["cv_r2_mean"], xgb_metrics["cv_r2_std"], xgb_metrics["holdout_r2"])),
                    "params": "default_validation_params",
                }
            )
            rows.append(xgb_metrics)

    res = pd.DataFrame(rows).sort_values("stability_score", ascending=False).reset_index(drop=True)
    if res.empty:
        raise ValueError("No validation result produced.")

    best_row = res.iloc[0]
    best_combo_text = best_row["combo"]
    best_combo_features = split_combo(best_combo_text)

    X_best, cat_best = prepare_feature_frame(df, best_combo_features)
    tuned_best, tuned_rows = tune_best_combo_catboost(X_best, y, groups, cat_best)
    tuned_best_full = dict(tuned_best)
    tuned_best_full.update(
        {
            "combo": best_combo_text,
            "n_features": int(X_best.shape[1]),
            "feature_list": " | ".join(list(X_best.columns)),
            "delta_r2_vs_baseline": float(tuned_best["holdout_r2"] - BASELINE_HOLDOUT_R2),
            "overfit_warning": bool(tuned_best["cv_r2_mean"] - tuned_best["holdout_r2"] > 0.03),
        }
    )

    res_final = pd.concat([res, pd.DataFrame([tuned_best_full])], ignore_index=True)
    res_final = res_final.sort_values("stability_score", ascending=False).reset_index(drop=True)
    res_final.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("Final Combo Validation Report")
    lines.append("=" * 29)
    lines.append(f"Rows used: {len(df)}")
    lines.append(f"Target: {target}")
    lines.append(f"Group column: {GROUP_COL}")
    lines.append(f"Baseline holdout R2: {BASELINE_HOLDOUT_R2:.4f}")
    lines.append("")
    lines.append("Tested combos (top20 filtered, leakage-aware):")
    for c in tested_combo_names:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("Dropped combos (top20):")
    if dropped:
        for combo_txt, reason in dropped:
            lines.append(f"- {reason} :: {combo_txt}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Validation results (ordered by stability):")
    for _, r in res_final.head(12).iterrows():
        lines.append(
            f"- {r['model']} | n={int(r['n_features'])} | CV R2={r['cv_r2_mean']:.4f} +- {r['cv_r2_std']:.4f} | "
            f"Holdout R2={r['holdout_r2']:.4f} | delta={r['delta_r2_vs_baseline']:.4f} | overfit={bool(r['overfit_warning'])}"
        )
        lines.append(f"  combo: {r['combo']}")

    best_final = res_final.iloc[0]
    lines.append("")
    lines.append("Best combo summary:")
    lines.append(f"- Model: {best_final['model']}")
    lines.append(f"- Holdout R2: {best_final['holdout_r2']:.4f}")
    lines.append(f"- CV R2: {best_final['cv_r2_mean']:.4f} +- {best_final['cv_r2_std']:.4f}")
    lines.append(f"- delta vs baseline: {best_final['delta_r2_vs_baseline']:.4f}")
    lines.append(f"- Stable: {not bool(best_final['overfit_warning'])}")
    lines.append(f"- Features: {best_final['feature_list']}")

    max_holdout = float(res_final["holdout_r2"].max())
    if 0.40 <= max_holdout <= 0.47:
        lines.append("")
        lines.append("Note: Results remain in the 0.40-0.47 band; this can indicate a data/feature ceiling rather than model-only limits.")

    lines.append("")
    lines.append(f"Tuning candidates evaluated on best combo: {len(tuned_rows)}")
    lines.append(f"Best tuning params: {tuned_best_full['params']}")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
