from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
CAND = BASE / "inverse_design_safe_descriptor_candidates.csv"
TOP50 = BASE / "inverse_design_safe_descriptor_top50.csv"
RAW = BASE / "perovskite_data.csv"

OUT_ENRICH = BASE / "candidate_profile_enrichment.csv"
OUT_RULES = BASE / "candidate_profile_rules.txt"
OUT_REPORT = BASE / "candidate_profile_report.txt"

DESCRIPTORS = [
    "A_site_variance",
    "cation_entropy",
    "X_site_variance",
    "bandgap_bucket",
    "bandgap_distance_from_1p7",
    "quenching_binary",
    "annealing_energy",
    "annealing_time_bucket",
    "annealing_temperature_bucket",
    "stack_depth_estimate",
    "device_complexity_score",
    "architecture_class",
    "solvent_count",
    "process_complexity_score",
    "composition_complexity_score",
]


def to_bool_like(series):
    m = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    return series.astype(str).str.strip().str.lower().map(m)


def classify_feature_type(ds, gp, feature):
    if feature not in ds.columns or feature not in gp.columns:
        return "missing"
    d_num = pd.to_numeric(ds[feature], errors="coerce")
    g_num = pd.to_numeric(gp[feature], errors="coerce")
    d_bool = to_bool_like(ds[feature]).notna().mean()
    g_bool = to_bool_like(gp[feature]).notna().mean()
    if d_bool > 0.9 and g_bool > 0.9:
        return "boolean"
    if d_num.notna().mean() > 0.8 and g_num.notna().mean() > 0.8:
        return "numeric"
    return "categorical"


def safe_mode(series):
    s = series.dropna()
    if s.empty:
        return "NA"
    return str(s.mode().iloc[0])


def fmt_or_na(x):
    if x is None:
        return "NA"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "NA"
    return x


def summarize_numeric(ds, gp, feature, subgroup):
    d = pd.to_numeric(ds[feature], errors="coerce")
    g = pd.to_numeric(gp[feature], errors="coerce")

    if d.notna().sum() == 0 or g.notna().sum() == 0:
        return {
            "subgroup": subgroup,
            "feature": feature,
            "feature_type": "numeric",
            "dataset_mean_or_mode": "NA",
            "candidate_mean_or_mode": "NA",
            "dataset_median_or_frequency": "NA",
            "candidate_median_or_frequency": "NA",
            "difference": "NA",
            "enrichment_ratio": "NA",
            "standardized_difference": "NA",
            "interpretation": "missing_numeric_values",
        }

    d_mean = float(d.mean())
    g_mean = float(g.mean())
    d_med = float(d.median())
    g_med = float(g.median())
    diff = g_mean - d_mean
    ratio = np.nan if abs(d_mean) < 1e-12 else g_mean / d_mean
    d_std = float(d.std())
    z = np.nan if np.isnan(d_std) or abs(d_std) < 1e-12 else diff / d_std

    if np.isnan(z):
        interp = "no_standardized_difference_available"
    elif z > 0.2:
        interp = "enriched_above_dataset"
    elif z < -0.2:
        interp = "depleted_vs_dataset"
    else:
        interp = "near_dataset_baseline"

    return {
        "subgroup": subgroup,
        "feature": feature,
        "feature_type": "numeric",
        "dataset_mean_or_mode": d_mean,
        "candidate_mean_or_mode": g_mean,
        "dataset_median_or_frequency": d_med,
        "candidate_median_or_frequency": g_med,
        "difference": diff,
        "enrichment_ratio": ratio,
        "standardized_difference": z,
        "interpretation": interp,
    }


def summarize_categorical_like(ds, gp, feature, subgroup, is_boolean=False):
    if is_boolean:
        d_bool = to_bool_like(ds[feature])
        g_bool = to_bool_like(gp[feature])
        d_mode = safe_mode(d_bool.map({True: "True", False: "False"}))
        g_mode = safe_mode(g_bool.map({True: "True", False: "False"}))
        d_freq = float((d_bool == True).mean()) if d_bool.notna().sum() > 0 else np.nan
        g_freq = float((g_bool == True).mean()) if g_bool.notna().sum() > 0 else np.nan
        diff = g_freq - d_freq if not (np.isnan(d_freq) or np.isnan(g_freq)) else np.nan
        ratio = np.nan if np.isnan(d_freq) or abs(d_freq) < 1e-12 else g_freq / d_freq
        if np.isnan(diff):
            interp = "missing_boolean_values"
        elif diff > 0:
            interp = "true_ratio_enriched"
        elif diff < 0:
            interp = "true_ratio_depleted"
        else:
            interp = "same_true_ratio"
        return {
            "subgroup": subgroup,
            "feature": feature,
            "feature_type": "boolean",
            "dataset_mean_or_mode": d_mode,
            "candidate_mean_or_mode": g_mode,
            "dataset_median_or_frequency": d_freq,
            "candidate_median_or_frequency": g_freq,
            "difference": diff,
            "enrichment_ratio": ratio,
            "standardized_difference": "NA",
            "interpretation": interp,
        }

    d = ds[feature].fillna("NA").astype(str)
    g = gp[feature].fillna("NA").astype(str)
    d_mode = safe_mode(d)
    g_mode = safe_mode(g)
    d_mode_freq = float((d == d_mode).mean()) if len(d) > 0 else np.nan
    g_mode_freq = float((g == g_mode).mean()) if len(g) > 0 else np.nan
    g_on_d_mode = float((g == d_mode).mean()) if len(g) > 0 else np.nan
    diff = g_on_d_mode - d_mode_freq if not (np.isnan(g_on_d_mode) or np.isnan(d_mode_freq)) else np.nan
    ratio = np.nan if np.isnan(d_mode_freq) or abs(d_mode_freq) < 1e-12 else g_on_d_mode / d_mode_freq
    interp = "same_mode" if d_mode == g_mode else "mode_shifted"
    return {
        "subgroup": subgroup,
        "feature": feature,
        "feature_type": "categorical",
        "dataset_mean_or_mode": d_mode,
        "candidate_mean_or_mode": g_mode,
        "dataset_median_or_frequency": d_mode_freq,
        "candidate_median_or_frequency": g_mode_freq,
        "difference": diff,
        "enrichment_ratio": ratio,
        "standardized_difference": "NA",
        "interpretation": interp,
    }


def summarize_feature(ds, gp, feature, subgroup):
    if feature not in ds.columns or feature not in gp.columns:
        return {
            "subgroup": subgroup,
            "feature": feature,
            "feature_type": "missing",
            "dataset_mean_or_mode": "NA",
            "candidate_mean_or_mode": "NA",
            "dataset_median_or_frequency": "NA",
            "candidate_median_or_frequency": "NA",
            "difference": "NA",
            "enrichment_ratio": "NA",
            "standardized_difference": "NA",
            "interpretation": "missing_descriptor",
        }
    ftype = classify_feature_type(ds, gp, feature)
    if ftype == "numeric":
        return summarize_numeric(ds, gp, feature, subgroup)
    if ftype == "boolean":
        return summarize_categorical_like(ds, gp, feature, subgroup, is_boolean=True)
    return summarize_categorical_like(ds, gp, feature, subgroup, is_boolean=False)


def get_enrich_row(df, subgroup, feature):
    x = df[(df["subgroup"] == subgroup) & (df["feature"] == feature)]
    return None if x.empty else x.iloc[0]


def build_rules(enrich_df):
    target_group = "high_score_lower_risk_all"
    mapped = [get_enrich_row(enrich_df, target_group, f) for f in DESCRIPTORS]
    mapped = [m for m in mapped if m is not None]
    rules = []
    for i, r in enumerate(mapped[:10], start=1):
        rules.append(
            {
                "rule": i,
                "pattern": f"{r['feature']} shows {r['interpretation']} pattern in low-risk high-score candidates.",
                "evidence": (
                    f"dataset={r['dataset_mean_or_mode']}, candidates={r['candidate_mean_or_mode']}, "
                    f"diff={r['difference']}, enrich={r['enrichment_ratio']}, std_diff={r['standardized_difference']}"
                ),
                "group": "high_score_lower_risk",
                "interpretation": "Model links this descriptor with high-score low-risk profiles.",
                "caution": "Treat as model-guided hypothesis, not direct experimental truth.",
            }
        )
    while len(rules) < 10:
        idx = len(rules) + 1
        rules.append(
            {
                "rule": idx,
                "pattern": "Cross-type consistency check placeholder.",
                "evidence": "NA",
                "group": "existing + perturbation + recombination",
                "interpretation": "More data-quality harmonization may be needed for this descriptor.",
                "caution": "Do not treat as actionable without descriptor-level validation.",
            }
        )
    return rules


def section(lines, title):
    lines.append(title)
    lines.append("-" * len(title))


def main():
    cand = pd.read_csv(CAND, low_memory=False)
    _ = pd.read_csv(TOP50, low_memory=False)
    raw = pd.read_csv(RAW, low_memory=False)
    baseline = raw.copy()
    for feat in DESCRIPTORS:
        if feat not in baseline.columns and feat in cand.columns:
            baseline[feat] = cand[feat]

    low = cand[cand["risk_bucket"] == "high_score_lower_risk"].copy()
    low_exist = low[low["candidate_type"] == "existing"].copy()
    low_pert = low[low["candidate_type"] == "perturbation"].copy()
    low_reco = low[low["candidate_type"] == "recombination"].copy()

    groups = {
        "high_score_lower_risk_all": low,
        "existing_low_risk_high_score": low_exist,
        "perturbation_low_risk_high_score": low_pert,
        "recombination_low_risk_high_score": low_reco,
    }

    rows = []
    for subgroup, sdf in groups.items():
        for feat in DESCRIPTORS:
            rows.append(summarize_feature(baseline, sdf, feat, subgroup))

    enrich = pd.DataFrame(rows)
    for col in [
        "dataset_mean_or_mode",
        "candidate_mean_or_mode",
        "dataset_median_or_frequency",
        "candidate_median_or_frequency",
        "difference",
        "enrichment_ratio",
        "standardized_difference",
    ]:
        enrich[col] = enrich[col].apply(fmt_or_na)
    enrich.to_csv(OUT_ENRICH, index=False)

    rules = build_rules(enrich)
    rule_lines = []
    for rr in rules:
        rule_lines.append(f"Rule {rr['rule']}")
        rule_lines.append(f"Pattern: {rr['pattern']}")
        rule_lines.append(f"Evidence: {rr['evidence']}")
        rule_lines.append(f"Candidate group: {rr['group']}")
        rule_lines.append(f"Interpretation: {rr['interpretation']}")
        rule_lines.append(f"Experimental caution: {rr['caution']}")
        rule_lines.append("")
    OUT_RULES.write_text("\n".join(rule_lines), encoding="utf-8")

    top10 = low.sort_values("predicted_PCE", ascending=False).head(10)
    all_group = enrich[enrich["subgroup"] == "high_score_lower_risk_all"].copy()
    miss = all_group[all_group["feature_type"] == "missing"]["feature"].tolist()
    num_part = all_group[all_group["feature_type"] == "numeric"].copy()
    cat_part = all_group[all_group["feature_type"].isin(["categorical", "boolean"])].copy()
    num_part["abs_z"] = pd.to_numeric(num_part["standardized_difference"], errors="coerce").abs()
    num_part = num_part.sort_values("abs_z", ascending=False)

    lines = []
    section(lines, "Executive summary")
    lines.append("This analysis extracts descriptor-level profiles from existing safe_descriptor outputs.")
    lines.append("No new model training or new candidate generation is performed.")
    lines.append("These are model-guided design hypotheses, not final experimental recipes.")
    lines.append("")

    section(lines, "Candidate group counts")
    lines.append(f"Total candidates: {len(cand)}")
    lines.append(f"high_score_lower_risk: {len(low)}")
    lines.append(f"existing: {len(low_exist)}")
    lines.append(f"perturbation: {len(low_pert)}")
    lines.append(f"recombination: {len(low_reco)}")
    lines.append("")

    section(lines, "Enriched numerical descriptors")
    if num_part.empty:
        lines.append("NA")
    else:
        for _, r in num_part.iterrows():
            lines.append(
                f"{r['feature']}: dataset={r['dataset_mean_or_mode']}, candidate={r['candidate_mean_or_mode']}, "
                f"diff={r['difference']}, z={r['standardized_difference']}, enrich={r['enrichment_ratio']}, interp={r['interpretation']}"
            )
    lines.append("")

    section(lines, "Enriched categorical/boolean descriptors")
    if cat_part.empty:
        lines.append("NA")
    else:
        for _, r in cat_part.iterrows():
            lines.append(
                f"{r['feature']}: dataset_mode={r['dataset_mean_or_mode']}, candidate_mode={r['candidate_mean_or_mode']}, "
                f"dataset_freq={r['dataset_median_or_frequency']}, candidate_freq={r['candidate_median_or_frequency']}, "
                f"enrich={r['enrichment_ratio']}, interp={r['interpretation']}"
            )
    lines.append("")

    section(lines, "Top 10 low-risk high-score candidates")
    if top10.empty:
        lines.append("NA")
    else:
        for _, r in top10.iterrows():
            lines.append(
                f"{r['candidate_id']} | type={r['candidate_type']} | predicted_PCE={r['predicted_PCE']:.4f} | risk={r.get('risk_score', np.nan):.3f}"
            )
    lines.append("")

    section(lines, "Design rules")
    for rr in rules:
        lines.append(f"Rule {rr['rule']}: {rr['pattern']}")
    lines.append("")

    section(lines, "Experimental prioritization guidance")
    lines.append("1. bandgap-related hypothesis: prioritize candidate profiles with enriched bandgap_bucket / low bandgap_distance_from_1p7.")
    lines.append("2. quenching/process hypothesis: test quenching_binary-enriched process windows first.")
    lines.append("3. annealing hypothesis: evaluate annealing_energy and annealing bucket windows around enriched ranges.")
    lines.append("4. device complexity hypothesis: compare high vs moderate device_complexity_score under same composition/process.")
    lines.append("5. composition diversity hypothesis: validate A_site_variance/cation_entropy/X_site_variance enriched ranges.")
    lines.append("")

    section(lines, "Limitations")
    lines.append("Model preferences are statistical and may not transfer directly to new labs/material systems.")
    lines.append("Descriptor gaps and missing values can bias enrichment interpretation.")
    lines.append("Use these outputs as screening hypotheses before experimental confirmation.")
    lines.append(f"Missing descriptor: {', '.join(miss) if miss else 'None'}")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
