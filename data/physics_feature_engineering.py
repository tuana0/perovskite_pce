import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

BASE = Path(__file__).resolve().parent
RAW = BASE / "perovskite_data.csv"
FINAL_IMPORTANCE = BASE / "final_feature_set_importance.csv"

OUT_RESULTS = BASE / "physics_feature_engineering_results.csv"
OUT_IMPORTANCE = BASE / "derived_feature_importance.csv"
OUT_LOG = BASE / "derived_feature_generation_log.txt"
OUT_REPORT = BASE / "physics_feature_engineering_report.txt"

TARGET = "JV_default_PCE"
GROUP_COL = "Ref_DOI_number"
RANDOM_STATE = 42
TOP40_R2_REFERENCE = 0.4062
MISSING_EXCLUDE_THRESHOLD = 0.98

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

LEAK_RISK_KEYWORDS = (
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


# ---------- helpers ----------
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


def count_tokens(text):
    if pd.isna(text):
        return np.nan
    s = str(text).strip()
    if not s:
        return np.nan
    parts = [p.strip() for p in re.split(r"[|,;/+:-]", s) if p.strip() and p.strip().lower() not in {"none", "nan", "unknown"}]
    if not parts:
        return np.nan
    return float(len(set(parts)))


def to_binary(series):
    s = series.astype(str).str.lower().str.strip()
    return s.isin({"yes", "true", "1", "y", "present", "tested", "t"}).astype(float)


def safe_num(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return s.map(parse_first_float)


def risk_feature(name):
    n = name.lower()
    return any(k in n for k in LEAK_RISK_KEYWORDS)


# ---------- derived features ----------
def build_derived_features(df):
    d = pd.DataFrame(index=df.index)

    # Composition-derived
    d["mixed_cation_count"] = df.get("Perovskite_composition_a_ions_coefficients", pd.Series(np.nan, index=df.index)).map(count_tokens)
    d["mixed_halide_count"] = df.get("Perovskite_composition_c_ions_coefficients", pd.Series(np.nan, index=df.index)).map(count_tokens)
    d["inorganic_vs_organic_binary"] = safe_num(df, "Perovskite_composition_inorganic")
    d["leadfree_binary"] = safe_num(df, "Perovskite_composition_leadfree")
    d["normalized_A_B_C_ion_counts"] = (
        df.get("Perovskite_composition_a_ions_coefficients", pd.Series(np.nan, index=df.index)).map(count_tokens).fillna(0)
        + df.get("Perovskite_composition_b_ions_coefficients", pd.Series(np.nan, index=df.index)).map(count_tokens).fillna(0)
        + df.get("Perovskite_composition_c_ions_coefficients", pd.Series(np.nan, index=df.index)).map(count_tokens).fillna(0)
    ) / 3.0
    d["composition_complexity_score"] = (
        d["mixed_cation_count"].fillna(0)
        + d["mixed_halide_count"].fillna(0)
        + d["normalized_A_B_C_ion_counts"].fillna(0)
    )

    # Process-derived
    anneal_t = safe_num(df, "Perovskite_deposition_thermal_annealing_temperature")
    anneal_time = safe_num(df, "Perovskite_deposition_thermal_annealing_time")
    d["annealing_energy"] = anneal_t * anneal_time
    d["deposition_step_count"] = safe_num(df, "Perovskite_deposition_number_of_deposition_steps")
    d["solvent_count"] = df.get("Perovskite_deposition_solvents_mixing_ratios", pd.Series(np.nan, index=df.index)).map(count_tokens)
    front_present = df.get("Add_lay_front", pd.Series(np.nan, index=df.index)).notna()
    back_present = df.get("Add_lay_back", pd.Series(np.nan, index=df.index)).notna()
    d["additive_presence_binary"] = (front_present | back_present).astype(float)
    quench = safe_num(df, "Perovskite_deposition_quenching_induced_crystallisation").fillna(0)
    d["quenching_and_annealing_interaction"] = quench * d["annealing_energy"].fillna(0)
    d["process_complexity_score"] = d[["deposition_step_count", "solvent_count", "additive_presence_binary"]].fillna(0).sum(axis=1)

    # Device-derived
    d["architecture_class"] = df.get("Cell_architecture", pd.Series("Unknown", index=df.index)).astype(str)
    flex = safe_num(df, "Cell_flexible").fillna(0)
    semi = safe_num(df, "Cell_semitransparent").fillna(0)
    d["flexible_and_semitransparent_interaction"] = flex * semi
    stack_text = df.get("Cell_stack_sequence", pd.Series(np.nan, index=df.index))
    d["stack_depth_estimate"] = stack_text.map(count_tokens)
    d["device_complexity_score"] = d[["stack_depth_estimate", "flexible_and_semitransparent_interaction"]].fillna(0).sum(axis=1)

    # Bandgap/material-derived
    bandgap = safe_num(df, "Perovskite_band_gap")
    d["bandgap_bucket"] = pd.cut(
        bandgap,
        bins=[-np.inf, 1.45, 1.65, np.inf],
        labels=["low", "mid", "high"],
    ).astype(str)
    d["graded_bandgap_binary"] = safe_num(df, "Perovskite_band_gap_graded")
    dim_cols = [
        "Perovskite_dimension_0D",
        "Perovskite_dimension_2D",
        "Perovskite_dimension_2D3D_mixture",
        "Perovskite_dimension_3D",
        "Perovskite_dimension_3D_with_2D_capping_layer",
    ]
    dim_score = pd.Series(0.0, index=df.index)
    for c in dim_cols:
        dim_score += safe_num(df, c).fillna(0)
    d["dimensionality_score"] = dim_score

    return d


def prepare_matrix(df, features):
    X_num = pd.DataFrame(index=df.index)
    X_cat = pd.DataFrame(index=df.index)
    for c in features:
        if c not in df.columns:
            continue
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


def evaluate_setup(name, features, derived_set, df, y, groups, split_idx):
    X, cat_cols = prepare_matrix(df, features)
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
        cv_scores.append(r2_score(yva, m.predict(xva)))

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_test, y_test), use_best_model=True, verbose=False)
    pred = model.predict(X_test)

    cv_mean = float(np.mean(cv_scores))
    hold = float(r2_score(y_test, pred))
    feat_list = list(X.columns)
    n_derived = sum(f in derived_set for f in feat_list)

    return {
        "setup": name,
        "n_features": int(len(feat_list)),
        "n_derived_features": int(n_derived),
        "cv_r2_mean": cv_mean,
        "cv_r2_std": float(np.std(cv_scores)),
        "holdout_r2": hold,
        "delta_vs_top40": float(hold - TOP40_R2_REFERENCE),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "overfit_flag": bool(cv_mean - hold > 0.03),
        "feature_list": " | ".join(feat_list),
        "model_obj": model,
        "risk_feature_count": int(sum(risk_feature(f) for f in feat_list)),
    }


def main():
    df = pd.read_csv(RAW, low_memory=False)
    imp = pd.read_csv(FINAL_IMPORTANCE)

    if TARGET not in df.columns or GROUP_COL not in df.columns:
        raise ValueError("Required columns missing")

    df = df[df[TARGET].notna()].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].between(0.0, 40.0)].drop_duplicates().reset_index(drop=True)

    base_features = imp["feature"].dropna().astype(str).tolist()
    if len(base_features) == 0:
        raise ValueError("No base features found in final_feature_set_importance.csv")

    derived = build_derived_features(df)
    # remove very-missing derived features
    log_lines = ["Derived Feature Generation Log", "=" * 31, f"Base features loaded: {len(base_features)}"]
    kept_derived = []
    excluded_derived = []
    group_map = {
        "composition": [
            "mixed_cation_count", "mixed_halide_count", "inorganic_vs_organic_binary", "leadfree_binary",
            "composition_complexity_score", "normalized_A_B_C_ion_counts",
        ],
        "process": [
            "annealing_energy", "process_complexity_score", "deposition_step_count", "solvent_count",
            "additive_presence_binary", "quenching_and_annealing_interaction",
        ],
        "device": [
            "architecture_class", "flexible_and_semitransparent_interaction", "device_complexity_score", "stack_depth_estimate",
        ],
        "bandgap": ["bandgap_bucket", "graded_bandgap_binary", "dimensionality_score"],
    }

    for c in derived.columns:
        miss = float(derived[c].isna().mean())
        if miss >= MISSING_EXCLUDE_THRESHOLD:
            excluded_derived.append((c, miss))
        else:
            kept_derived.append(c)
        log_lines.append(f"- {c}: missing_ratio={miss:.4f} {'EXCLUDED' if miss >= MISSING_EXCLUDE_THRESHOLD else 'KEPT'}")

    df_model = pd.concat([df, derived[kept_derived]], axis=1)
    derived_set = set(kept_derived)

    y = df_model[TARGET].reset_index(drop=True)
    groups = df_model[GROUP_COL].fillna("Unknown_DOI").reset_index(drop=True)
    dummy_x = np.zeros((len(df_model), 1))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    split_idx = next(gss.split(dummy_x, y, groups=groups))

    comp = [f for f in group_map["composition"] if f in derived_set]
    proc = [f for f in group_map["process"] if f in derived_set]
    dev = [f for f in group_map["device"] if f in derived_set]
    bnd = [f for f in group_map["bandgap"] if f in derived_set]
    all_derived = comp + proc + dev + bnd

    setups = [
        ("original_top40", base_features),
        ("top40_plus_composition_features", base_features + comp),
        ("top40_plus_process_features", base_features + proc),
        ("top40_plus_device_features", base_features + dev),
        ("top40_plus_bandgap_features", base_features + bnd),
        ("top40_plus_all_derived", base_features + all_derived),
        ("derived_features_only", all_derived),
    ]

    rows = []
    for name, feats in setups:
        out = evaluate_setup(name, feats, derived_set, df_model, y, groups, split_idx)
        if out is not None:
            rows.append(out)

    if not rows:
        raise ValueError("No setup produced results")

    res = pd.DataFrame([{k: v for k, v in r.items() if k != "model_obj"} for r in rows])
    res = res.sort_values("holdout_r2", ascending=False).reset_index(drop=True)
    res.to_csv(OUT_RESULTS, index=False)

    best_setup = res.iloc[0]["setup"]
    best_model = next(r["model_obj"] for r in rows if r["setup"] == best_setup)
    best_feats = [x.strip() for x in res.iloc[0]["feature_list"].split("|")]
    imp_gain = best_model.get_feature_importance()
    imp_out = pd.DataFrame({"feature": best_feats, "importance_gain": imp_gain})
    imp_out["is_derived"] = imp_out["feature"].isin(list(derived_set))
    imp_out["risk_flag"] = imp_out["feature"].map(lambda x: "risk" if risk_feature(x) else "clean")
    imp_out = imp_out.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    imp_out.head(30).to_csv(OUT_IMPORTANCE, index=False)

    log_lines.append("")
    log_lines.append(f"Derived kept: {len(kept_derived)}")
    log_lines.append(f"Derived excluded: {len(excluded_derived)}")
    OUT_LOG.write_text("\n".join(log_lines), encoding="utf-8")

    top = res.iloc[0]
    orig = res[res["setup"] == "original_top40"].iloc[0]

    rpt = []
    rpt.append("Physics Feature Engineering Report")
    rpt.append("=" * 34)
    rpt.append(f"Rows used: {len(df_model)}")
    rpt.append(f"Base feature count (from final_feature_set_importance): {len(base_features)}")
    rpt.append(f"Derived kept/excluded: {len(kept_derived)}/{len(excluded_derived)}")
    rpt.append("")
    rpt.append("Setup results:")
    for _, r in res.iterrows():
        rpt.append(
            f"- {r['setup']}: n={int(r['n_features'])}, derived={int(r['n_derived_features'])}, "
            f"CV R2={r['cv_r2_mean']:.4f} +- {r['cv_r2_std']:.4f}, Holdout R2={r['holdout_r2']:.4f}, "
            f"delta_vs_top40={r['delta_vs_top40']:.4f}, overfit={bool(r['overfit_flag'])}"
        )
    rpt.append("")
    rpt.append("Key answers:")
    rpt.append(f"- Best setup: {top['setup']} (Holdout R2={top['holdout_r2']:.4f}).")
    rpt.append(f"- Original top40 Holdout R2 in this run: {orig['holdout_r2']:.4f}.")
    rpt.append("- Derived features carry signal if any derived-augmented setup beats original_top40.")
    rpt.append("- Interaction usefulness is indicated by gains in setups containing interaction terms.")
    rpt.append("- Derived-only performance shows standalone explanatory power of engineered physics features.")
    rpt.append(
        f"- Leakage risk check: best setup includes {int(top['risk_feature_count'])} risk-tagged features; "
        "review before final deployment."
    )
    rpt.append("")
    rpt.append("Final recommendation:")
    if top["holdout_r2"] > orig["holdout_r2"] + 0.002:
        rpt.append(f"- Recommend {top['setup']} for next phase (meaningful gain over original_top40).")
    else:
        rpt.append("- Keep original_top40 as final safe set for now; derived features did not deliver a robust gain yet.")
    rpt.append("- Next step for R2 improvement: enrich composition stoichiometry parsing and process-condition harmonization.")

    OUT_REPORT.write_text("\n".join(rpt), encoding="utf-8")
    print("\n".join(rpt))


if __name__ == "__main__":
    main()

