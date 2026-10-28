from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2_fourlevel_no_concurrent_hamd17 as base


from public_paths import ROOT
OUT = ROOT / "outputs" / "step2_fourlevel_no_concurrent_hamd17_education3"
OLD_OUT = ROOT / "outputs" / "step2_fourlevel_no_concurrent_hamd17"
MODEL_ID = "FOUR_LEVEL_ZI_ORDINAL_ES_GEE_NO_CONCURRENT_HAMD17_EDUCATION3_V1"
N_PARAMETERS = 44
EDUCATION_MAP = {
    "小学": "Junior high or below",
    "初中": "Junior high or below",
    "高中/中专/职高": "High school/vocational",
    "大学/大专": "College or above",
    "研究生及以上": "College or above",
}
EDUCATION_LEVELS = ["Junior high or below", "High school/vocational", "College or above"]
EDUCATION_TERM = "C(education3, Treatment(reference='College or above'))"
RHS = base.RHS.replace("C(education)", EDUCATION_TERM)


def log(message: str) -> None:
    print(f"Four-level education3 refit: {message}", flush=True)


def parameter_names(Gdf: pd.DataFrame, Xdf: pd.DataFrame) -> list[str]:
    return (
        [f"mixing_gamma:{c}" for c in Gdf.columns]
        + ["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"]
        + [f"ordinal_beta:{c}" for c in Xdf.columns]
    )


def load_and_preflight():
    version = json.loads(base.VERSION_RECORD.read_text(encoding="utf-8"))
    freeze_hash = base.sha256(base.FREEZE)
    df = pd.read_csv(base.FREEZE)
    original_education = df["education"].copy(deep=True)
    df["education3"] = df["education"].map(EDUCATION_MAP)
    df["education3"] = pd.Categorical(df["education3"], categories=EDUCATION_LEVELS)
    checks = []

    def check(name, observed, expected, passed):
        checks.append({"check": name, "observed": observed, "expected": expected, "pass": bool(passed)})

    check("freeze SHA256", freeze_hash, base.EXPECTED_FREEZE_SHA256, freeze_hash.lower() == base.EXPECTED_FREEZE_SHA256)
    check("source SHA256", version.get("source_sha256"), base.EXPECTED_SOURCE_SHA256, version.get("source_sha256", "").lower() == base.EXPECTED_SOURCE_SHA256)
    check("participants", df["participant_ID"].nunique(), base.EXPECTED_PARTICIPANTS, df["participant_ID"].nunique() == base.EXPECTED_PARTICIPANTS)
    check("person-visits", len(df), base.EXPECTED_VISITS, len(df) == base.EXPECTED_VISITS)
    duplicate_keys = int(df.duplicated(["participant_ID", "time"]).sum())
    check("duplicate participant_ID + time", duplicate_keys, 0, duplicate_keys == 0)
    check("original education retained", bool(original_education.equals(df["education"])), True, original_education.equals(df["education"]))
    check("original education missing visits", int(df["education"].isna().sum()), 0, not df["education"].isna().any())
    check("derived education missing visits", int(df["education3"].isna().sum()), 0, not df["education3"].isna().any())
    check("derived education levels", json.dumps([str(x) for x in df["education3"].cat.categories]), json.dumps(EDUCATION_LEVELS), [str(x) for x in df["education3"].cat.categories] == EDUCATION_LEVELS)
    participant_multicategory = int((df.groupby("participant_ID", observed=True)["education3"].nunique() > 1).sum())
    check("participants changing education3 across visits", participant_multicategory, 0, participant_multicategory == 0)

    if "burden_ord4" not in df.columns or df["burden_ord4"].isna().any():
        raise RuntimeError("PREFLIGHT FAIL: burden_ord4 missing or incomplete")
    df["burden_ord4"] = df["burden_ord4"].astype(int)
    levels = sorted(df["burden_ord4"].unique().tolist())
    counts = df["burden_ord4"].value_counts().sort_index().to_dict()
    check("outcome levels", levels, [0, 1, 2, 3], levels == [0, 1, 2, 3])
    check("outcome counts", json.dumps(counts, sort_keys=True), json.dumps(base.EXPECTED_COUNTS, sort_keys=True), counts == base.EXPECTED_COUNTS)

    Gdf, Xdf = b1.make_design(df, mix_rhs=RHS, ord_rhs=RHS)
    names = parameter_names(Gdf, Xdf)
    forbidden = [c for c in [*Gdf.columns, *Xdf.columns] if "concurrent_HAMD" in c or "site" in c.lower() or "province" in c.lower()]
    missing_mixing_domains = [c for c in base.DOMAIN_VARS if c not in Gdf.columns]
    missing_ordinal_domains = [c for c in base.DOMAIN_VARS if c not in Xdf.columns]
    education_columns_G = [c for c in Gdf.columns if "education3" in c]
    education_columns_X = [c for c in Xdf.columns if "education3" in c]
    check("mixing columns", Gdf.shape[1], 21, Gdf.shape[1] == 21)
    check("ordinal columns", Xdf.shape[1], 20, Xdf.shape[1] == 20)
    check("threshold count", base.Q, 3, base.Q == 3)
    check("total parameters", len(names), N_PARAMETERS, len(names) == N_PARAMETERS)
    check("unique parameter names", len(set(names)), N_PARAMETERS, len(set(names)) == N_PARAMETERS)
    check("education3 mixing contrasts", len(education_columns_G), 2, len(education_columns_G) == 2)
    check("education3 ordinal contrasts", len(education_columns_X), 2, len(education_columns_X) == 2)
    reference_coefficient_present = any("[T.College or above]" in c for c in education_columns_G + education_columns_X)
    check("College or above reference absent as coefficient", reference_coefficient_present, False, not reference_coefficient_present)
    check("forbidden design terms", json.dumps(forbidden), "[]", not forbidden)
    check("nine mixing domains", json.dumps(missing_mixing_domains), "[]", not missing_mixing_domains)
    check("nine ordinal domains", json.dumps(missing_ordinal_domains), "[]", not missing_ordinal_domains)
    check("threshold positions", json.dumps(names[21:24]), json.dumps(["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"]), names[21:24] == ["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"])

    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(checks)
    audit.to_csv(OUT / "01_preflight_hard_checks.csv", index=False, encoding="utf-8-sig")
    category_audit = (
        df.assign(education3=df["education3"].astype("string"))
        .groupby("education3", dropna=False)
        .agg(participants=("participant_ID", "nunique"), person_visits=("participant_ID", "size"))
        .reindex(EDUCATION_LEVELS)
        .reset_index()
    )
    category_audit["missing_person_visits"] = 0
    category_audit.to_csv(OUT / "02_education3_category_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "category": ["0", "1", "2", ">=3"],
        "n": [counts.get(i, 0) for i in range(4)],
        "percent": [counts.get(i, 0) / len(df) * 100 for i in range(4)],
    }).to_csv(OUT / "03_outcome_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "index_zero_based": range(len(names)),
        "index_one_based": range(1, len(names) + 1),
        "block": ["mixing"] * 21 + ["threshold"] * 3 + ["ordinal"] * 20,
        "parameter": names,
    }).to_csv(OUT / "04_parameter_index_map.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"component": "mixing", "columns": Gdf.shape[1], "RHS": RHS},
        {"component": "ordered thresholds", "columns": 3, "RHS": "a0 + positive cumulative exp(log_delta) increments"},
        {"component": "ordinal", "columns": Xdf.shape[1], "RHS": RHS},
    ]).to_csv(OUT / "04b_design_audit.csv", index=False, encoding="utf-8-sig")
    base.atomic_json(OUT / "00_preflight_manifest.json", {
        "model_id": MODEL_ID,
        "freeze_id": base.FREEZE_ID,
        "freeze_sha256": freeze_hash,
        "participants": int(df["participant_ID"].nunique()),
        "person_visits": int(len(df)),
        "education_reference": "College or above",
        "education_category_counts": category_audit.to_dict(orient="records"),
        "outcome_counts": {str(k): int(v) for k, v in counts.items()},
        "mixing_parameters": int(Gdf.shape[1]),
        "thresholds": 3,
        "ordinal_parameters": int(Xdf.shape[1]),
        "total_parameters": len(names),
        "all_checks_pass": bool(audit["pass"].all()),
    })
    if not audit["pass"].all():
        raise RuntimeError(f"PREFLIGHT FAIL: {audit.loc[~audit['pass'], 'check'].tolist()}")
    return df, Gdf, Xdf, Gdf.to_numpy(float), Xdf.to_numpy(float), audit


def configure_base() -> None:
    base.OUT = OUT
    base.MODEL_ID = MODEL_ID
    base.N_PARAMETERS = N_PARAMETERS
    base.RHS = RHS
    base.parameter_names = parameter_names
    base.load_and_preflight = load_and_preflight
    base.engine.OUT = OUT
    base.engine.log = log
    base.log = log


def compare_with_five_category() -> dict:
    old_effects = pd.read_csv(OLD_OUT / "10_domain_effects_robust_wald.csv")
    new_effects = pd.read_csv(OUT / "10_domain_effects_robust_wald.csv")
    effects = new_effects.merge(old_effects, on=["Domain", "Component"], suffixes=("_education3", "_education5"))
    effects["OR_ratio_education3_vs_education5"] = effects["OR_education3"] / effects["OR_education5"]
    effects["log_OR_difference"] = np.log(effects["OR_education3"]) - np.log(effects["OR_education5"])
    effects["P_difference"] = effects["Wald_P_education3"] - effects["Wald_P_education5"]
    effects["p_lt_0_05_same"] = (effects["Wald_P_education3"] < 0.05) == (effects["Wald_P_education5"] < 0.05)
    effects.to_csv(OUT / "18_effect_comparison_vs_education5.csv", index=False, encoding="utf-8-sig")

    old_prob = pd.read_csv(OLD_OUT / "11_adjusted_joint_probabilities.csv")
    new_prob = pd.read_csv(OUT / "11_adjusted_joint_probabilities.csv")
    probability_rows = []
    for domain in b1.DOMAINS:
        old = old_prob.loc[old_prob["Domain"].eq(domain)].iloc[0]
        new = new_prob.loc[new_prob["Domain"].eq(domain)].iloc[0]
        for cutoff in (1, 3):
            key = f"RD_burden_ge{cutoff}_domain1_minus_domain0"
            probability_rows.append({
                "Domain": domain,
                "cutoff": f"burden>={cutoff}",
                "RD_education3": float(new[key]),
                "RD_education5": float(old[key]),
                "RD_difference_education3_minus_education5": float(new[key] - old[key]),
            })
    probability_comparison = pd.DataFrame(probability_rows)
    probability_comparison.to_csv(OUT / "19_probability_difference_comparison_vs_education5.csv", index=False, encoding="utf-8-sig")

    full_gate = json.loads((OUT / "12_full_sample_gate.json").read_text(encoding="utf-8"))
    old_gate = json.loads((OLD_OUT / "12_full_sample_gate.json").read_text(encoding="utf-8"))
    boot = pd.read_csv(OUT / "13_bootstrap_5_replicate_diagnostics.csv")
    old_boot = pd.read_csv(OLD_OUT / "13_bootstrap_5_replicate_diagnostics.csv")
    full_diag = pd.read_csv(OUT / "05_full_sample_11_fit_diagnostics.csv")
    old_full_diag = pd.read_csv(OLD_OUT / "05_full_sample_11_fit_diagnostics.csv")
    strict = full_diag[full_diag["strict_valid"].astype(str).str.lower().eq("true")].copy()
    old_strict = old_full_diag[old_full_diag["strict_valid"].astype(str).str.lower().eq("true")].copy()
    boundary = (boot["min_pi"] < 1e-5) | (boot["max_pi"] > 1 - 1e-10)
    old_boundary = (old_boot["min_pi"] < 1e-5) | (old_boot["max_pi"] > 1 - 1e-10)
    high_condition = strict["jacobian_condition_number"] > 1e8
    old_high_condition = old_strict["jacobian_condition_number"] > 1e8
    competing = bool((strict["objective"].max() - strict["objective"].min() > 1e-3) and (strict["max_abs_parameter"] .max() > 5))
    comparison = {
        "education3_selected_max_abs_mixing": float(full_gate["max_abs_mixing_coefficient"]),
        "education5_selected_max_abs_mixing": float(old_gate["max_abs_mixing_coefficient"]),
        "education3_full_sample_high_mixing_strict_fits": int((strict["max_abs_mixing_coefficient"] > 5).sum()),
        "education5_full_sample_high_mixing_strict_fits": int((old_strict["max_abs_mixing_coefficient"] > 5).sum()),
        "education3_full_sample_high_condition_strict_fits": int(high_condition.sum()),
        "education5_full_sample_high_condition_strict_fits": int(old_high_condition.sum()),
        "education3_bootstrap_extreme_mixing_selected": int((boot["max_abs_mixing_coefficient"] > 5).sum()),
        "education5_bootstrap_extreme_mixing_selected": int((old_boot["max_abs_mixing_coefficient"] > 5).sum()),
        "education3_bootstrap_boundary_probability_selected": int(boundary.sum()),
        "education5_bootstrap_boundary_probability_selected": int(old_boundary.sum()),
        "max_abs_log_OR_difference": float(effects["log_OR_difference"].abs().max()),
        "same_p_lt_0_05_classification": int(effects["p_lt_0_05_same"].sum()),
        "max_abs_RD_difference_burden_ge1_or_ge3": float(probability_comparison["RD_difference_education3_minus_education5"].abs().max()),
        "competing_full_sample_basin": competing,
    }
    base.atomic_json(OUT / "20_stability_comparison_summary.json", comparison)
    return comparison


def finalize_report() -> None:
    comparison = compare_with_five_category()
    summary_path = OUT / "17_final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    boot = pd.read_csv(OUT / "13_bootstrap_5_replicate_diagnostics.csv")
    diagnostics = pd.read_csv(OUT / "14_bootstrap_5_all_start_diagnostics.csv")
    all_selected = bool(boot["selected"].astype(str).str.lower().eq("true").all())
    complete = bool(len(diagnostics) == 55 and (boot["starts_executed"] == 11).all())
    boundary = (boot["min_pi"] < 1e-5) | (boot["max_pi"] > 1 - 1e-10)
    high_mixing = boot["max_abs_mixing_coefficient"] > 5
    if not all_selected or not complete:
        status = "FAIL"
    elif bool(boundary.any() or high_mixing.any() or comparison["competing_full_sample_basin"]):
        status = "REVIEW REQUIRED"
    else:
        status = "PASS"
    summary.update({
        "final_status": status,
        "boundary_assessment_not_coefficient_only": True,
        "selected_bootstrap_boundary_probability_replicates": int(boundary.sum()),
        "selected_bootstrap_extreme_mixing_replicates": int(high_mixing.sum()),
        "competing_full_sample_basin": bool(comparison["competing_full_sample_basin"]),
    })
    base.atomic_json(summary_path, summary)

    effect_cmp = pd.read_csv(OUT / "18_effect_comparison_vs_education5.csv")
    prob_cmp = pd.read_csv(OUT / "19_probability_difference_comparison_vs_education5.csv")
    category = pd.read_csv(OUT / "02_education3_category_audit.csv")
    full_gate = json.loads((OUT / "12_full_sample_gate.json").read_text(encoding="utf-8"))
    report = [
        "# Three-Level Education Refit Comparison Report",
        "",
        f"## Final status: {status}",
        "",
        "## Education audit",
        "",
    ]
    for _, row in category.iterrows():
        report.append(f"- {row['education3']}: {int(row['participants'])} participants / {int(row['person_visits'])} person-visits; missing 0.")
    report += [
        "- Reference group: College or above.",
        "- Original education retained; frozen rows and participant-time keys unchanged.",
        "",
        "## Full-sample and stability findings",
        "",
        f"- Parameter blocks: mixing 21 + thresholds 3 + ordinal 20 = 44.",
        f"- Full-sample strict-valid fits: {full_gate['strict_valid_fits']}/11; selected maximum |gamma|: {full_gate['max_abs_mixing_coefficient']:.6g}.",
        f"- Full-sample high-mixing strict fits: {comparison['education3_full_sample_high_mixing_strict_fits']} versus {comparison['education5_full_sample_high_mixing_strict_fits']} with five-level education.",
        f"- Selected bootstrap extreme-mixing solutions: {comparison['education3_bootstrap_extreme_mixing_selected']}/5 versus {comparison['education5_bootstrap_extreme_mixing_selected']}/5.",
        f"- Selected bootstrap boundary-probability solutions: {comparison['education3_bootstrap_boundary_probability_selected']}/5 versus {comparison['education5_bootstrap_boundary_probability_selected']}/5.",
        "- Boundary assessment used mixing probabilities, objective basins, Jacobian conditioning, covariance validity, and coefficient magnitudes; it was not based on a coefficient cutoff alone.",
        "",
        "## Clinical-effect comparison",
        "",
        f"- Maximum absolute log-OR change across 18 effects: {comparison['max_abs_log_OR_difference']:.6g}.",
        f"- Same raw-P < 0.05 classification: {comparison['same_p_lt_0_05_classification']}/18 effects.",
        f"- Maximum absolute change in the domain-present minus domain-absent probability difference for burden>=1 or burden>=3: {comparison['max_abs_RD_difference_burden_ge1_or_ge3']:.6g}.",
        "- Detailed effect and probability comparisons are in `18_effect_comparison_vs_education5.csv` and `19_probability_difference_comparison_vs_education5.csv`.",
        "",
        "This five-replicate cluster bootstrap is a numerical stability trial only, not formal bootstrap inference. No two-level education model, other outcome model, or Step 3 analysis was run.",
    ]
    (OUT / "Final_Model_Report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_full() -> None:
    configure_base()
    base.run_full_sample()


def run_bootstrap() -> None:
    configure_base()
    base.run_bootstrap(5)
    finalize_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    if args.bootstrap:
        run_bootstrap()
    else:
        run_full()




