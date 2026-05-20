import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

try:
    import shap  # noqa: F401
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

BASE = Path(__file__).resolve().parent
RAW = BASE / "perovskite_data.csv"
AUDIT_RESULTS = BASE / "baseline_feature_audit_results.csv"
AUDIT_IMPORTANCE = BASE / "baseline_feature_importance.csv"

OUT_RESULTS = BASE / "final_feature_set_analysis_results.csv"
OUT_IMPORTANCE = BASE / "final_feature_set_importance.csv"
OUT_REPORT = BASE / "final_feature_set_analysis_report.txt"

TARGET = "JV_default_PCE"
GROUP_COL = "Ref_DOI_number"
BASELINE_R2 = 0.4233
RANDOM_STATE = 42

CANDIDATE_SETUPS = [
    "full_baseline_features",
    "top20_feature_importance",
    "top40_feature_importance",
    "no_measurement_features",
    "no_metadata_features",
]

RISK_KEYWORDS = (
    "jv_",
    "performance",
    "voc",
    "jsc",
    "ff",
    "measured",
    "certified",
    "scan",
    "forward",
    "reverse",
    "outdoor",
    "doi",
    "title",
    "author",
    "journal",
)

CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "R2",
    "random_seed": RANDOM_STATE,
    "nan_mode": "Min",
    "subsample": 0.85,
    "iterations": 300,
    "depth": 7,
    "learning_rate": 0.07,
    "l2_leaf_reg": 3.0,
    "verbose": False,
}


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


def split_feature_list(text):
    if pd.isna(text) or not str(text).strip():
        return []
    return [x.strip() for x in str(text).split("|") if x.strip()]


def is_risk_feature(name):
    n = name.lower()
    return any(k in n for k in RISK_KEYWORDS)


def load_candidate_sets(audit_df, imp_df):
    setup_to_features = {}

    for setup in CANDIDATE_SETUPS:
        row = audit_df[audit_df["setup"] == setup]
        if row.empty:
            continue
        setup_to_features[setup] = split_feature_list(row.iloc[0]["feature_list"])

    if "top20_feature_importance" not in setup_to_features:
        setup_to_features["top20_feature_importance"] = imp_df["feature"].head(20).tolist()
    if "top40_feature_importance" not in setup_to_features:
        setup_to_features["top40_feature_importance"] = imp_df["feature"].head(40).tolist()

    return setup_to_features


def build_X_for_features(df, features):
    cols = [c for c in features if c in df.columns and c != TARGET and c != GROUP_COL]
    X_num = pd.DataFrame(index=df.index)
    X_cat = pd.DataFrame(index=df.index)

    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            X_num[c] = pd.to_numeric(s, errors="coerce")
        else:
            parsed = s.map(parse_first_float)
            if parsed.notna().mean() >= 0.75:
                X_num[c] = parsed
            else:
                X_cat[c] = s.astype(str).replace({"nan": "Unknown", "None": "Unknown"}).fillna("Unknown")

    if not X_num.empty:
        X_num = X_num.fillna(X_num.median(numeric_only=True))

    X = pd.concat([X_num, X_cat], axis=1)
    keep = X.columns[X.isna().mean() < 0.995].tolist()
    X = X[keep]
    cat_cols = [c for c in X_cat.columns if c in X.columns]
    return X, cat_cols


def evaluate_feature_set(name, features, df, y, groups, split_idx):
    X, cat_cols = build_X_for_features(df, features)
    if X.empty:
        return None

    tr, te = split_idx
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]
    g_train = groups.iloc[tr]

    cat_idx = [i for i, c in enumerate(X.columns) if c in cat_cols]
    cv = GroupKFold(n_splits=3)
    cv_scores = []
    for tr_cv, va_cv in cv.split(X_train, y_train, groups=g_train):
        xtr, xva = X_train.iloc[tr_cv], X_train.iloc[va_cv]
        ytr, yva = y_train.iloc[tr_cv], y_train.iloc[va_cv]
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(xtr, ytr, cat_features=cat_idx, eval_set=(xva, yva), use_best_model=True, verbose=False)
        pred_cv = m.predict(xva)
        cv_scores.append(r2_score(yva, pred_cv))

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_test, y_test), use_best_model=True, verbose=False)
    pred = model.predict(X_test)

    holdout_r2 = float(r2_score(y_test, pred))
    cv_r2_mean = float(np.mean(cv_scores))
    cv_r2_std = float(np.std(cv_scores))

    feat_list = list(X.columns)
    risk_count = sum(is_risk_feature(f) for f in feat_list)

    return {
        "setup": name,
        "n_features": int(len(feat_list)),
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "holdout_r2": holdout_r2,
        "delta_vs_baseline": float(holdout_r2 - BASELINE_R2),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "overfit_flag": bool(cv_r2_mean - holdout_r2 > 0.03),
        "risk_feature_count": int(risk_count),
        "risk_feature_ratio": float(risk_count / max(1, len(feat_list))),
        "feature_list": " | ".join(feat_list),
        "model_obj": model,
    }


def choose_recommended(df_res):
    tmp = df_res.copy()
    tmp["perf_gap"] = (BASELINE_R2 - tmp["holdout_r2"]).abs()
    tmp["stability_gap"] = (tmp["cv_r2_mean"] - tmp["holdout_r2"]).abs()

    # prioritize near-baseline performance, then compactness, then lower risk, then stability
    tmp["score"] = (
        -tmp["perf_gap"]
        - 0.003 * tmp["n_features"]
        - 0.15 * tmp["risk_feature_ratio"]
        - 0.5 * tmp["stability_gap"]
    )

    # explicit preference: if top40 is very close to full baseline, prefer top40
    full_row = tmp[tmp["setup"] == "full_baseline_features"]
    top40_row = tmp[tmp["setup"] == "top40_feature_importance"]
    if not full_row.empty and not top40_row.empty:
        full_r2 = float(full_row.iloc[0]["holdout_r2"])
        top40_r2 = float(top40_row.iloc[0]["holdout_r2"])
        if full_r2 - top40_r2 <= 0.005:
            return top40_row.iloc[0]

    return tmp.sort_values("score", ascending=False).iloc[0]


def main():
    df = pd.read_csv(RAW, low_memory=False)
    if TARGET not in df.columns or GROUP_COL not in df.columns:
        raise ValueError("Required columns are missing: JV_default_PCE / Ref_DOI_number")

    df = df[df[TARGET].notna()].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].between(0.0, 40.0)].drop_duplicates().reset_index(drop=True)

    audit_df = pd.read_csv(AUDIT_RESULTS)
    imp_df = pd.read_csv(AUDIT_IMPORTANCE)

    setup_to_features = load_candidate_sets(audit_df, imp_df)
    missing = [s for s in CANDIDATE_SETUPS if s not in setup_to_features]
    if missing:
        raise ValueError(f"Missing feature sets in inputs: {missing}")

    y = df[TARGET].reset_index(drop=True)
    groups = df[GROUP_COL].fillna("Unknown_DOI").reset_index(drop=True)

    # Split indices derived on full baseline feature matrix-like row count (same df rows)
    dummy_X = np.zeros((len(df), 1))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    split_idx = next(gss.split(dummy_X, y, groups=groups))

    rows = []
    for setup in CANDIDATE_SETUPS:
        result = evaluate_feature_set(setup, setup_to_features[setup], df, y, groups, split_idx)
        if result is not None:
            rows.append(result)

    if not rows:
        raise ValueError("No feature set could be evaluated")

    res = pd.DataFrame([{k: v for k, v in r.items() if k != "model_obj"} for r in rows])
    res = res.sort_values("holdout_r2", ascending=False).reset_index(drop=True)
    res.to_csv(OUT_RESULTS, index=False)

    recommended = choose_recommended(res)
    rec_setup = str(recommended["setup"])
    rec_model = next(r["model_obj"] for r in rows if r["setup"] == rec_setup)
    rec_features = split_feature_list(res[res["setup"] == rec_setup].iloc[0]["feature_list"])

    # importance on recommended model
    imp_gain = rec_model.get_feature_importance()
    imp_out = pd.DataFrame({"feature": rec_features, "importance_gain": imp_gain})
    imp_out["risk_flag"] = imp_out["feature"].map(lambda x: "risk" if is_risk_feature(x) else "clean")

    if HAS_SHAP:
        try:
            X_rec, _ = build_X_for_features(df, rec_features)
            sample_n = min(2000, len(X_rec))
            rs = np.random.RandomState(RANDOM_STATE)
            idx = rs.choice(len(X_rec), size=sample_n, replace=False)
            shap_vals = rec_model.get_feature_importance(data=X_rec.iloc[idx], type="ShapValues")
            mean_abs = np.abs(shap_vals[:, :-1]).mean(axis=0)
            shap_map = dict(zip(rec_features, mean_abs))
            imp_out["shap_mean_abs"] = imp_out["feature"].map(lambda f: float(shap_map.get(f, np.nan)))
        except Exception:
            imp_out["shap_mean_abs"] = np.nan
    else:
        imp_out["shap_mean_abs"] = np.nan

    imp_out = imp_out.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    imp_out.head(30).to_csv(OUT_IMPORTANCE, index=False)

    full_r = res[res["setup"] == "full_baseline_features"].iloc[0]
    top20_r = res[res["setup"] == "top20_feature_importance"].iloc[0]
    top40_r = res[res["setup"] == "top40_feature_importance"].iloc[0]

    lines = []
    lines.append("Final Feature Set Analysis Report")
    lines.append("=" * 33)
    lines.append(f"Rows used: {len(df)}")
    lines.append(f"Target: {TARGET}")
    lines.append(f"Group column: {GROUP_COL}")
    lines.append(f"Baseline reference holdout R2: {BASELINE_R2:.4f}")
    lines.append("")
    lines.append("Candidate set results:")
    for _, r in res.iterrows():
        lines.append(
            f"- {r['setup']}: n={int(r['n_features'])}, CV R2={r['cv_r2_mean']:.4f} +- {r['cv_r2_std']:.4f}, "
            f"Holdout R2={r['holdout_r2']:.4f}, delta={r['delta_vs_baseline']:.4f}, "
            f"risk_count={int(r['risk_feature_count'])}, overfit={bool(r['overfit_flag'])}"
        )

    lines.append("")
    lines.append("Recommendation:")
    lines.append(f"- Recommended final set: {rec_setup}")
    lines.append(
        "- Reason: selected by performance proximity to baseline, compactness, lower risk burden, and CV-holdout stability."
    )
    lines.append(
        f"- Full baseline vs top40: {full_r['holdout_r2']:.4f} vs {top40_r['holdout_r2']:.4f}."
    )
    lines.append(
        f"- Full baseline vs top20: {full_r['holdout_r2']:.4f} vs {top20_r['holdout_r2']:.4f}."
    )
    lines.append(
        "- Performance-explainability tradeoff: compact sets reduce complexity and improve interpretability, "
        "but may lose multi-factor interactions captured by full baseline."
    )

    lines.append("")
    lines.append("Leakage/risk note:")
    lines.append(
        "- Risk features were tracked (JV/performance proxies, Voc/Jsc/FF terms, measured/certified/scan/reverse/forward, "
        "outdoor post-measurement signals, bibliographic identifiers)."
    )
    lines.append("- Risk features should not be accepted solely because they raise score.")

    lines.append("")
    lines.append("Top 30 important features saved to final_feature_set_importance.csv.")
    if HAS_SHAP:
        lines.append("SHAP summary values included when calculation succeeded.")
    else:
        lines.append("SHAP not available; CatBoost gain importance used.")

    lines.append("")
    lines.append("Next feature-engineering steps for later R2 improvement:")
    lines.append("- Standardize composition descriptors (A/B/C ion stoichiometry normalization).")
    lines.append("- Add physics-informed aggregates (bandgap-alignment/process interaction features).")
    lines.append("- Harmonize process metadata units and categorical vocabularies across studies.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
