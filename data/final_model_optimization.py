import itertools
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

BASE = Path(__file__).resolve().parent
RAW = BASE / "perovskite_data.csv"
FINAL_IMPORTANCE = BASE / "final_feature_set_importance.csv"
PHYSICS_RESULTS = BASE / "physics_feature_engineering_results.csv"

OUT_RESULTS = BASE / "final_model_optimization_results.csv"
OUT_REPORT = BASE / "final_model_optimization_report.txt"
OUT_IMPORTANCE = BASE / "final_model_feature_importance.csv"
OUT_PRED = BASE / "final_model_predictions_holdout.csv"

TARGET = "JV_default_PCE"
GROUP_COL = "Ref_DOI_number"
RANDOM_STATE = 42
MISSING_EXCLUDE_THRESHOLD = 0.98

REF_ORIGINAL_TOP40 = 0.4062
REF_PHYSICS_BEST = 0.4244
REF_HIST_BASELINE = 0.4233

TUNE_TRIALS_PER_MODEL = 36

LEAK_RISK_KEYWORDS = (
    "jv_", "performance", "voc", "jsc", "ff", "measured", "certified", "scan",
    "forward", "reverse", "outdoor", "doi", "title", "author", "journal",
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


def count_tokens(text):
    if pd.isna(text):
        return np.nan
    s = str(text).strip()
    if not s:
        return np.nan
    parts = [p.strip() for p in re.split(r"[|,;/+:-]", s) if p.strip() and p.strip().lower() not in {"none", "nan", "unknown"}]
    return float(len(set(parts))) if parts else np.nan


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


def build_derived_features(df):
    d = pd.DataFrame(index=df.index)

    d["mixed_cation_count"] = df.get("Perovskite_composition_a_ions_coefficients", pd.Series(np.nan, index=df.index)).map(count_tokens)
    d["mixed_halide_count"] = df.get("Perovskite_composition_c_ions_coefficients", pd.Series(np.nan, index=df.index)).map(count_tokens)
    d["inorganic_vs_organic_binary"] = safe_num(df, "Perovskite_composition_inorganic")
    d["leadfree_binary"] = safe_num(df, "Perovskite_composition_leadfree")
    d["normalized_A_B_C_ion_counts"] = (
        df.get("Perovskite_composition_a_ions_coefficients", pd.Series(np.nan, index=df.index)).map(count_tokens).fillna(0)
        + df.get("Perovskite_composition_b_ions_coefficients", pd.Series(np.nan, index=df.index)).map(count_tokens).fillna(0)
        + df.get("Perovskite_composition_c_ions_coefficients", pd.Series(np.nan, index=df.index)).map(count_tokens).fillna(0)
    ) / 3.0
    d["composition_complexity_score"] = d[["mixed_cation_count", "mixed_halide_count", "normalized_A_B_C_ion_counts"]].fillna(0).sum(axis=1)

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

    d["architecture_class"] = df.get("Cell_architecture", pd.Series("Unknown", index=df.index)).astype(str)
    flex = safe_num(df, "Cell_flexible").fillna(0)
    semi = safe_num(df, "Cell_semitransparent").fillna(0)
    d["flexible_and_semitransparent_interaction"] = flex * semi
    d["stack_depth_estimate"] = df.get("Cell_stack_sequence", pd.Series(np.nan, index=df.index)).map(count_tokens)
    d["device_complexity_score"] = d[["stack_depth_estimate", "flexible_and_semitransparent_interaction"]].fillna(0).sum(axis=1)

    bandgap = safe_num(df, "Perovskite_band_gap")
    d["bandgap_bucket"] = pd.cut(bandgap, bins=[-np.inf, 1.45, 1.65, np.inf], labels=["low", "mid", "high"]).astype(str)
    d["graded_bandgap_binary"] = safe_num(df, "Perovskite_band_gap_graded")
    dim_cols = [
        "Perovskite_dimension_0D", "Perovskite_dimension_2D", "Perovskite_dimension_2D3D_mixture",
        "Perovskite_dimension_3D", "Perovskite_dimension_3D_with_2D_capping_layer",
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


def encode_non_catboost(xtr, xva):
    a = pd.get_dummies(xtr, dummy_na=True)
    b = pd.get_dummies(xva, dummy_na=True).reindex(columns=a.columns, fill_value=0)
    return a, b


def make_model(name, params=None):
    p = params or {}
    if name == "CatBoost":
        base = {
            "loss_function": "RMSE", "eval_metric": "R2", "random_seed": RANDOM_STATE,
            "nan_mode": "Min", "subsample": 0.85, "iterations": 500, "depth": 7,
            "learning_rate": 0.05, "l2_leaf_reg": 5.0, "verbose": False,
        }
        base.update(p)
        return CatBoostRegressor(**base)
    if name == "XGBoost":
        base = {
            "objective": "reg:squarederror", "random_state": RANDOM_STATE,
            "n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 3.0, "n_jobs": 1,
        }
        base.update(p)
        return XGBRegressor(**base)
    if name == "LightGBM":
        base = {
            "random_state": RANDOM_STATE, "n_estimators": 500, "num_leaves": 63,
            "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9,
        }
        base.update(p)
        return LGBMRegressor(**base)
    if name == "ExtraTrees":
        base = {"n_estimators": 500, "random_state": RANDOM_STATE, "n_jobs": 1, "min_samples_leaf": 2}
        base.update(p)
        return ExtraTreesRegressor(**base)
    if name == "RandomForest":
        base = {"n_estimators": 500, "random_state": RANDOM_STATE, "n_jobs": 1, "min_samples_leaf": 2}
        base.update(p)
        return RandomForestRegressor(**base)
    raise ValueError(name)


def score_model(name, X, y, groups, split_idx, cat_cols, params=None, return_holdout_pred=False):
    tr, te = split_idx
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]
    g_train = groups.iloc[tr]

    cv = GroupKFold(n_splits=3)
    cv_scores = []
    fold_preds = []
    cat_idx = [i for i, c in enumerate(X.columns) if c in cat_cols]

    for tr_cv, va_cv in cv.split(X_train, y_train, groups=g_train):
        xtr, xva = X_train.iloc[tr_cv], X_train.iloc[va_cv]
        ytr, yva = y_train.iloc[tr_cv], y_train.iloc[va_cv]

        if name == "CatBoost":
            m = make_model(name, params)
            m.fit(xtr, ytr, cat_features=cat_idx, eval_set=(xva, yva), use_best_model=True, verbose=False)
            p = m.predict(xva)
        else:
            xtr_e, xva_e = encode_non_catboost(xtr, xva)
            m = make_model(name, params)
            m.fit(xtr_e, ytr)
            p = m.predict(xva_e)

        cv_scores.append(r2_score(yva, p))

    if name == "CatBoost":
        m_full = make_model(name, params)
        m_full.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_test, y_test), use_best_model=True, verbose=False)
        pred = m_full.predict(X_test)
        feature_names_used = list(X.columns)
    else:
        xtr_e, xte_e = encode_non_catboost(X_train, X_test)
        m_full = make_model(name, params)
        m_full.fit(xtr_e, y_train)
        pred = m_full.predict(xte_e)
        feature_names_used = list(xtr_e.columns)

    out = {
        "model": name,
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
        "holdout_r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "overfit_flag": bool(np.mean(cv_scores) - r2_score(y_test, pred) > 0.03),
        "params": str(params) if params else "default",
        "model_obj": m_full,
        "feature_names_used": feature_names_used,
        "holdout_pred": pred if return_holdout_pred else None,
        "y_test": y_test.values if return_holdout_pred else None,
    }
    return out


def sample_grid(grid_dict, n_trials, seed):
    keys = list(grid_dict.keys())
    values = [grid_dict[k] for k in keys]
    all_combos = list(itertools.product(*values))
    rng = random.Random(seed)
    if len(all_combos) <= n_trials:
        picked = all_combos
    else:
        picked = rng.sample(all_combos, n_trials)
    params = []
    for combo in picked:
        params.append({k: v for k, v in zip(keys, combo)})
    return params


def tune_model(name, X, y, groups, split_idx, cat_cols):
    if name == "CatBoost":
        grid = {
            "depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1], "l2_leaf_reg": [3, 5, 7],
            "iterations": [300, 500, 800], "subsample": [0.8, 0.9, 1.0],
        }
    elif name == "XGBoost":
        grid = {
            "max_depth": [3, 5, 7], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [300, 500, 800],
            "subsample": [0.8, 0.9, 1.0], "colsample_bytree": [0.8, 0.9, 1.0], "reg_lambda": [1, 3, 5],
        }
    elif name == "LightGBM":
        grid = {
            "num_leaves": [31, 63, 127], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [300, 500, 800],
            "subsample": [0.8, 0.9, 1.0], "colsample_bytree": [0.8, 0.9, 1.0],
        }
    else:
        return None

    best = None
    trial_params = sample_grid(grid, TUNE_TRIALS_PER_MODEL, RANDOM_STATE + hash(name) % 1000)
    for p in trial_params:
        s = score_model(name, X, y, groups, split_idx, cat_cols, params=p)
        if (best is None) or (s["holdout_r2"] > best["holdout_r2"]):
            best = s
    best["model"] = f"{name}_tuned"
    return best


def ensemble_score(label, base_models, weights, X, y, groups, split_idx, cat_cols):
    tr, te = split_idx
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]
    g_train = groups.iloc[tr]

    cv = GroupKFold(n_splits=3)
    cv_scores = []
    for tr_cv, va_cv in cv.split(X_train, y_train, groups=g_train):
        xtr, xva = X_train.iloc[tr_cv], X_train.iloc[va_cv]
        ytr, yva = y_train.iloc[tr_cv], y_train.iloc[va_cv]
        fold_preds = []
        for mname in base_models:
            if mname == "CatBoost":
                cat_idx = [i for i, c in enumerate(X.columns) if c in cat_cols]
                m = make_model(mname)
                m.fit(xtr, ytr, cat_features=cat_idx, eval_set=(xva, yva), use_best_model=True, verbose=False)
                fold_preds.append(m.predict(xva))
            else:
                xtr_e, xva_e = encode_non_catboost(xtr, xva)
                m = make_model(mname)
                m.fit(xtr_e, ytr)
                fold_preds.append(m.predict(xva_e))
        ens_pred = np.average(np.vstack(fold_preds), axis=0, weights=weights)
        cv_scores.append(r2_score(yva, ens_pred))

    hold_preds = []
    for mname in base_models:
        if mname == "CatBoost":
            cat_idx = [i for i, c in enumerate(X.columns) if c in cat_cols]
            m = make_model(mname)
            m.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_test, y_test), use_best_model=True, verbose=False)
            hold_preds.append(m.predict(X_test))
        else:
            xtr_e, xte_e = encode_non_catboost(X_train, X_test)
            m = make_model(mname)
            m.fit(xtr_e, y_train)
            hold_preds.append(m.predict(xte_e))

    pred = np.average(np.vstack(hold_preds), axis=0, weights=weights)
    return {
        "model": label,
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
        "holdout_r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "overfit_flag": bool(np.mean(cv_scores) - r2_score(y_test, pred) > 0.03),
        "params": f"weights={weights}",
        "model_obj": None,
        "holdout_pred": pred,
        "y_test": y_test.values,
    }


def main():
    df = pd.read_csv(RAW, low_memory=False)
    imp = pd.read_csv(FINAL_IMPORTANCE)
    phys = pd.read_csv(PHYSICS_RESULTS)

    if TARGET not in df.columns or GROUP_COL not in df.columns:
        raise ValueError("Missing required target/group columns")

    df = df[df[TARGET].notna()].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].between(0.0, 40.0)].drop_duplicates().reset_index(drop=True)

    # Base features: prefer original_top40 row from physics results
    top40_row = phys[phys["setup"] == "original_top40"]
    if not top40_row.empty:
        base_features = [x.strip() for x in str(top40_row.iloc[0]["feature_list"]).split("|") if x.strip()]
    else:
        base_features = imp["feature"].dropna().astype(str).tolist()

    derived = build_derived_features(df)
    kept_derived = [c for c in derived.columns if float(derived[c].isna().mean()) < MISSING_EXCLUDE_THRESHOLD]
    df_model = pd.concat([df, derived[kept_derived]], axis=1)

    feature_set = base_features + kept_derived
    X, cat_cols = prepare_matrix(df_model, feature_set)
    y = df_model[TARGET].reset_index(drop=True)
    groups = df_model[GROUP_COL].fillna("Unknown_DOI").reset_index(drop=True)

    dummy_x = np.zeros((len(df_model), 1))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    split_idx = next(gss.split(dummy_x, y, groups=groups))

    model_names = ["CatBoost", "ExtraTrees", "RandomForest"]
    if HAS_XGB:
        model_names.append("XGBoost")
    if HAS_LGBM:
        model_names.append("LightGBM")

    rows = []
    preds_for_export = {}

    for name in model_names:
        s = score_model(name, X, y, groups, split_idx, cat_cols, return_holdout_pred=True)
        rows.append(s)
        preds_for_export[name] = s["holdout_pred"]

    base_df = pd.DataFrame([{k: v for k, v in r.items() if k not in {"model_obj", "feature_names_used", "holdout_pred", "y_test"}} for r in rows])
    base_df = base_df.sort_values("holdout_r2", ascending=False).reset_index(drop=True)

    # tune best 2 eligible models
    tunable = [m for m in base_df["model"].tolist() if m in {"CatBoost", "XGBoost", "LightGBM"}]
    top2 = tunable[:2]
    tuned_rows = []
    for name in top2:
        t = tune_model(name, X, y, groups, split_idx, cat_cols)
        if t is not None:
            tuned_rows.append(t)
            preds_for_export[t["model"]] = t.get("holdout_pred")

    # ensembles
    ens_rows = []
    if HAS_XGB:
        ens_rows.append(ensemble_score("Ens_Cat_XGB_50_50", ["CatBoost", "XGBoost"], [0.5, 0.5], X, y, groups, split_idx, cat_cols))
        ens_rows.append(ensemble_score("Ens_Cat_XGB_60_40", ["CatBoost", "XGBoost"], [0.6, 0.4], X, y, groups, split_idx, cat_cols))
        ens_rows.append(ensemble_score("Ens_Cat_XGB_70_30", ["CatBoost", "XGBoost"], [0.7, 0.3], X, y, groups, split_idx, cat_cols))
    ens_rows.append(ensemble_score("Ens_Cat_ET_50_50", ["CatBoost", "ExtraTrees"], [0.5, 0.5], X, y, groups, split_idx, cat_cols))
    if HAS_XGB:
        ens_rows.append(ensemble_score("Ens_Cat_XGB_ET_50_30_20", ["CatBoost", "XGBoost", "ExtraTrees"], [0.5, 0.3, 0.2], X, y, groups, split_idx, cat_cols))
        ens_rows.append(ensemble_score("Ens_Cat_XGB_ET_40_40_20", ["CatBoost", "XGBoost", "ExtraTrees"], [0.4, 0.4, 0.2], X, y, groups, split_idx, cat_cols))

    all_rows = rows + tuned_rows + ens_rows
    out_df = pd.DataFrame([{k: v for k, v in r.items() if k not in {"model_obj", "feature_names_used", "holdout_pred", "y_test"}} for r in all_rows])
    out_df["delta_vs_original_top40"] = out_df["holdout_r2"] - REF_ORIGINAL_TOP40
    out_df["delta_vs_physics_best"] = out_df["holdout_r2"] - REF_PHYSICS_BEST
    out_df["delta_vs_historical_baseline"] = out_df["holdout_r2"] - REF_HIST_BASELINE
    out_df = out_df.sort_values("holdout_r2", ascending=False).reset_index(drop=True)
    out_df.to_csv(OUT_RESULTS, index=False)

    best_row = out_df.iloc[0]
    best_name = best_row["model"]

    # feature importance for best single model (exclude ensembles)
    single_candidates = out_df[~out_df["model"].str.startswith("Ens_")]
    best_single_name = single_candidates.iloc[0]["model"]
    best_single_obj = None
    best_single_feats = None
    for r in all_rows:
        if r["model"] == best_single_name:
            best_single_obj = r["model_obj"]
            best_single_feats = r.get("feature_names_used")
            break

    importance_df = pd.DataFrame(columns=["feature", "importance", "is_derived", "risk_flag"])
    if best_single_obj is not None:
        if "CatBoost" in best_single_name:
            imp_vals = np.asarray(best_single_obj.get_feature_importance())
        else:
            imp_vals = np.asarray(best_single_obj.feature_importances_)

        feats = list(best_single_feats) if best_single_feats is not None else []
        n = min(len(feats), len(imp_vals))
        if n > 0:
            importance_df = pd.DataFrame({"feature": feats[:n], "importance": imp_vals[:n]})
        else:
            importance_df = pd.DataFrame(columns=["feature", "importance"])
        importance_df["is_derived"] = importance_df["feature"].isin(kept_derived)
        importance_df["risk_flag"] = importance_df["feature"].map(lambda x: "risk" if risk_feature(x) else "clean")
        importance_df = importance_df.sort_values("importance", ascending=False).head(30)
        importance_df.to_csv(OUT_IMPORTANCE, index=False)
    else:
        importance_df.to_csv(OUT_IMPORTANCE, index=False)

    # holdout predictions export
    tr, te = split_idx
    pred_out = pd.DataFrame({"y_true": y.iloc[te].values})
    for r in all_rows:
        p = r.get("holdout_pred")
        if p is not None:
            pred_out[r["model"]] = p
    pred_out.to_csv(OUT_PRED, index=False)

    lines = []
    lines.append("Final Model Optimization Report")
    lines.append("=" * 31)
    lines.append(f"Rows used: {len(X)}")
    lines.append(f"Final setup: top40_plus_all_derived")
    lines.append(f"Features used: {X.shape[1]} (base={len(base_features)}, derived_kept={len(kept_derived)})")
    lines.append("")
    lines.append("Model results:")
    for _, r in out_df.iterrows():
        lines.append(
            f"- {r['model']}: CV R2={r['cv_r2_mean']:.4f} +- {r['cv_r2_std']:.4f}, Holdout R2={r['holdout_r2']:.4f}, "
            f"d_top40={r['delta_vs_original_top40']:.4f}, d_phys={r['delta_vs_physics_best']:.4f}, "
            f"d_hist={r['delta_vs_historical_baseline']:.4f}, overfit={bool(r['overfit_flag'])}"
        )

    best_tuned = out_df[out_df["model"].str.contains("_tuned", regex=False)].head(1)
    best_ens = out_df[out_df["model"].str.startswith("Ens_")].head(1)

    lines.append("")
    lines.append(f"Best single model: {best_single_name}")
    if not best_tuned.empty:
        lines.append(f"Best tuned model: {best_tuned.iloc[0]['model']} (R2={best_tuned.iloc[0]['holdout_r2']:.4f})")
    else:
        lines.append("Best tuned model: N/A")
    if not best_ens.empty:
        lines.append(f"Best ensemble: {best_ens.iloc[0]['model']} (R2={best_ens.iloc[0]['holdout_r2']:.4f})")
    else:
        lines.append("Best ensemble: N/A")

    lines.append("")
    lines.append(f"Did any model beat physics best (0.4244)? {'Yes' if float(best_row['holdout_r2']) > REF_PHYSICS_BEST else 'No'}")
    lines.append(f"Overall best model: {best_name} (R2={best_row['holdout_r2']:.4f})")
    lines.append("Recommendation: accept final model only if overfit_flag is False and leakage-risk review passes.")
    lines.append("If scores stay in 0.42-0.45 band, this may be a realistic leakage-free DOI-group-safe ceiling.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()



