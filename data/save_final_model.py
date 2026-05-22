import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

BASE = Path(__file__).resolve().parent
RAW = BASE / "perovskite_data.csv"
FINAL_IMPORTANCE = BASE / "final_feature_set_importance.csv"

OUT_DIR = BASE / "final_artifacts"
OUT_MODEL = OUT_DIR / "final_model.cbm"
OUT_FEATURES = OUT_DIR / "final_feature_list.csv"
OUT_META = OUT_DIR / "final_model_metadata.json"

TARGET = "JV_default_PCE"
GROUP_COL = "Ref_DOI_number"
RANDOM_STATE = 42
MISSING_EXCLUDE_THRESHOLD = 0.98

# Best tuned CatBoost params from final_model_optimization_results.csv
CATBOOST_TUNED_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "R2",
    "random_seed": RANDOM_STATE,
    "nan_mode": "Min",
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5,
    "iterations": 800,
    "subsample": 1.0,
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW, low_memory=False)
    imp = pd.read_csv(FINAL_IMPORTANCE)

    if TARGET not in df.columns or GROUP_COL not in df.columns:
        raise ValueError("Missing required columns: JV_default_PCE / Ref_DOI_number")

    df = df[df[TARGET].notna()].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].between(0.0, 40.0)].drop_duplicates().reset_index(drop=True)

    base_features = imp["feature"].dropna().astype(str).tolist()
    if not base_features:
        raise ValueError("No base features found in final_feature_set_importance.csv")

    derived = build_derived_features(df)
    kept_derived = [c for c in derived.columns if float(derived[c].isna().mean()) < MISSING_EXCLUDE_THRESHOLD]
    df_model = pd.concat([df, derived[kept_derived]], axis=1)

    feature_set = base_features + kept_derived
    X, cat_cols = prepare_matrix(df_model, feature_set)
    y = df_model[TARGET].reset_index(drop=True)

    cat_idx = [i for i, c in enumerate(X.columns) if c in cat_cols]

    model = CatBoostRegressor(**CATBOOST_TUNED_PARAMS)
    model.fit(X, y, cat_features=cat_idx, verbose=False)
    model.save_model(str(OUT_MODEL))

    pd.DataFrame({"feature": list(X.columns)}).to_csv(OUT_FEATURES, index=False)

    meta = {
        "model_name": "CatBoost_tuned",
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "target": TARGET,
        "group_column": GROUP_COL,
        "feature_set_name": "top40_plus_all_derived",
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_base_features_input": int(len(base_features)),
        "n_derived_features_kept": int(len(kept_derived)),
        "catboost_params": CATBOOST_TUNED_PARAMS,
        "artifacts": {
            "model": OUT_MODEL.name,
            "feature_list": OUT_FEATURES.name,
            "metadata": OUT_META.name,
        },
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_MODEL}")
    print(f"Saved: {OUT_FEATURES}")
    print(f"Saved: {OUT_META}")


if __name__ == "__main__":
    main()
