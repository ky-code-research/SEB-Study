from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

import pandas as pd
from scipy.stats import chi2, f, norm, t

import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2_fourlevel_concurrent_hamd17_education3 as step2_m1
import step2_fourlevel_no_concurrent_hamd17_education3 as step2_m2
import step3_domain_time_burden_4799_preliminary as step3
import step3_independent_model_core as step3_core
from site_clustered_sandwich_gate import (
    covariance_diagnostics,
    fay_df_v1,
    fay_graubard,
    score_hessian_contributions,
    sym,
    unit_contrast,
)


from public_paths import ROOT
FREEZE = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "repaired_natural_complete_case_analysis_data.csv"
OUT = ROOT / "outputs" / "hospital_clustered_GEE_sensitivity_2026-10-23_final"
EXPECTED_HASH = "6844f512b5122beabc9905456e1a9a2111f6318c398a984232068b22c22fd16c"
EXPECTED_WEEKS = {"Week2": 1381, "Week8": 1259, "Week12": 1146, "Week24": 1013}
M = 21
FORMAL_DOMAIN_LABELS = {
    "Sleep": "Sleep",
    "Gastrointestinal": "Gastrointestinal",
    "Other psychiatric": "Other symptom",
    "Nervous system": "Nervous system",
    "Cardiac": "Heart",
    "Eye/Ear": "Eye/Ear",
    "Skin": "Skin",
    "Sexual": "Sexual functioning",
    "Genital/Urinary": "Genital/Urinary",
}
FORMAL_DOMAIN_ORDER = list(FORMAL_DOMAIN_LABELS.values())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh(pvals: pd.Series) -> pd.Series:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return pd.Series(out, index=pvals.index)


def json_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.copy().astype(object)
    for col in clean.columns:
        clean[col] = clean[col].map(lambda v: "Inf" if isinstance(v, (float, np.floating)) and np.isposinf(v) else ("-Inf" if isinstance(v, (float, np.floating)) and np.isneginf(v) else (None if isinstance(v, (float, np.floating)) and np.isnan(v) else v)))
    return clean.to_dict("records")


def scalar_inference(estimate: float, se: float, critical: float, distribution: str, df: float | None = None) -> dict:
    stat = estimate / se if se > 0 else np.nan
    if distribution == "normal":
        p = 2 * norm.sf(abs(stat))
    elif distribution == "t":
        p = 2 * t.sf(abs(stat), df)
    else:
        raise ValueError(distribution)
    return {
        "SE": se,
        "statistic": stat,
        "P": p,
        "coef_CI_low": estimate - critical * se,
        "coef_CI_high": estimate + critical * se,
        "OR": float(np.exp(estimate)),
        "OR_CI_low": float(np.exp(estimate - critical * se)),
        "OR_CI_high": float(np.exp(estimate + critical * se)),
    }


def model_matrices(theta, y, gdf, xdf, df):
    score_obs, patient_scores, site_scores, a_by_site, a, aggregation_diff = score_hessian_contributions(
        theta, y, gdf.to_numpy(float), xdf.to_numpy(float), df["participant_ID"], df["site"]
    )
    fg = fay_graubard(site_scores, a_by_site, a)
    patient_u = patient_scores.to_numpy(float)
    v_patient = sym(fg["Ainv"] @ (patient_u.T @ patient_u) @ fg["Ainv"].T)
    return {
        "score_obs": score_obs,
        "patient_scores": patient_scores,
        "site_scores": site_scores,
        "a_by_site": a_by_site,
        "a": a,
        "fg": fg,
        "V_patient": v_patient,
        "aggregation_diff": aggregation_diff,
    }


def influence_for_scalar(label, component, domain, contrast, mats, site_sizes):
    fg = mats["fg"]
    vals = []
    for site in fg["sites"]:
        psi0 = float(contrast @ fg["Ainv"] @ fg["U"][site])
        psifg = float(contrast @ fg["Ainv"] @ fg["adjusted_U"][site])
        vals.append((site, psi0, psifg))
    den0 = sum(x[1] ** 2 for x in vals)
    denfg = sum(x[2] ** 2 for x in vals)
    rows = []
    for site, psi0, psifg in vals:
        rows.append({
            "stage": label, "component": component, "target": domain, "site": site,
            "participants": int(site_sizes.loc[site, "participants"]),
            "person_visits": int(site_sizes.loc[site, "person_visits"]),
            "one_step_coefficient_shift_if_site_removed": -psi0,
            "CR0_variance_share": psi0 ** 2 / den0 if den0 else np.nan,
            "FG_variance_share": psifg ** 2 / denfg if denfg else np.nan,
        })
    return rows


def step2_results(df, y):
    g, x = b1.make_design(df, step2_m1.RHS, step2_m1.RHS)
    names = step2_m1.parameter_names(g, x)
    saved = pd.read_csv(ROOT / "outputs" / "robust_sandwich_covariance_repair_2026-10-09" / "step2" / "model_1_parameter_estimates_repaired.csv")
    if saved["parameter"].tolist() != names:
        raise RuntimeError("Step 2 locked parameter names do not match reconstructed design")
    theta = saved["estimate"].to_numpy(float)
    mats = model_matrices(theta, y, g, x, df)
    patient_se = np.sqrt(np.clip(np.diag(mats["V_patient"]), 0, np.inf))
    lock_diff = float(np.max(np.abs(patient_se - saved["robust_SE"].to_numpy(float))))
    site_sizes = df.assign(site_key=df["site"].astype(str)).groupby("site_key", observed=True).agg(
        participants=("participant_ID", "nunique"), person_visits=("participant_ID", "size")
    ).sort_index()
    pg = g.shape[1]
    rows, influence = [], []
    for component, offset, columns in [
        ("Zero-Inflated Component", 0, list(g.columns)),
        ("Burden Severity", pg + 3, list(x.columns)),
    ]:
        for internal_domain in b1.DOMAINS:
            domain = FORMAL_DOMAIN_LABELS[internal_domain]
            var = b1.DOMAIN_VAR[internal_domain]
            idx = offset + columns.index(var)
            contrast = unit_contrast(len(theta), idx)
            estimate = float(theta[idx])
            fg_df = fay_df_v1(mats["a"], mats["a_by_site"], mats["fg"], contrast)
            methods = {
                "Primary patient-clustered normal": (mats["V_patient"], "normal", None),
                "Hospital CR0 normal": (mats["fg"]["V_CR0"], "normal", None),
                "Hospital CR0 t(20)": (mats["fg"]["V_CR0"], "t", M - 1),
                "Hospital CR1 normal": (mats["fg"]["V_CR1"], "normal", None),
                "Hospital CR1 t(20)": (mats["fg"]["V_CR1"], "t", M - 1),
                "Hospital Fay-Graubard t(df)": (mats["fg"]["V_FG"], "t", fg_df),
            }
            for method, (vcov, distribution, dfi) in methods.items():
                se = math.sqrt(max(float(contrast @ vcov @ contrast), 0.0))
                critical = norm.ppf(0.975) if distribution == "normal" else t.ppf(0.975, dfi)
                inf = scalar_inference(estimate, se, critical, distribution, dfi)
                rows.append({
                    "stage": "Step 2 Model 1", "component": component, "domain": domain,
                    "parameter": names[idx], "coefficient": estimate, "OR_definition": "exp(gamma)" if component.startswith("Zero") else "exp(beta)",
                    "method": method, "reference_distribution": "Normal" if distribution == "normal" else f"t(df={dfi:.9g})",
                    "effective_df": np.nan if dfi is None else dfi, **inf,
                })
            influence.extend(influence_for_scalar("Step 2 Model 1", component, domain, contrast, mats, site_sizes))
    result = pd.DataFrame(rows)
    for method in result["method"].unique():
        for component in result["component"].unique():
            mask = (result["method"] == method) & (result["component"] == component)
            result.loc[mask, "BH_FDR_P_within_9_domains"] = bh(result.loc[mask, "P"])
    method_order = [
        "Primary patient-clustered normal", "Hospital CR0 normal", "Hospital CR0 t(20)",
        "Hospital CR1 normal", "Hospital CR1 t(20)", "Hospital Fay-Graubard t(df)",
    ]
    result["component_order"] = pd.Categorical(result["component"], categories=["Zero-Inflated Component", "Burden Severity"], ordered=True)
    result["domain_order"] = pd.Categorical(result["domain"], categories=FORMAL_DOMAIN_ORDER, ordered=True)
    result["method_order"] = pd.Categorical(result["method"], categories=method_order, ordered=True)
    result = result.sort_values(["component_order", "domain_order", "method_order"]).drop(columns=["component_order", "domain_order", "method_order"])
    return result, pd.DataFrame(influence), mats, lock_diff, names, theta


def wald(beta, vcov):
    rank = int(np.linalg.matrix_rank(vcov, tol=np.linalg.svd(vcov, compute_uv=False)[0] * 1e-8))
    w = float(beta @ np.linalg.pinv(vcov) @ beta)
    return w, rank


def step3_results(df, y):
    df3 = df.copy()
    df3["concurrent_HAMD17_z"] = (pd.to_numeric(df3["concurrent_HAMD17"], errors="raise") - step3_core.HAMD_ONLY_MEAN) / step3_core.HAMD_ONLY_SD_DDOF0
    gi_var = b1.DOMAIN_VAR["Gastrointestinal"]
    rhs = step3.rhs_for(step2_m1.RHS, gi_var)
    g, x = b1.make_design(df3, rhs, rhs)
    names = step3.parameter_names(g, x)
    saved = pd.read_csv(ROOT / "outputs" / "step3_independent_model1_with_hamd17_2026-10-09" / "domains" / "Gastrointestinal" / "selected_named_parameters.csv")
    if saved["parameter"].tolist() != names:
        raise RuntimeError("Step 3 locked parameter names do not match reconstructed design")
    theta = saved["estimate"].to_numpy(float)
    mats = model_matrices(theta, y, g, x, df3)
    official_params = pd.read_csv(ROOT / "outputs" / "step3_independent_model1_with_hamd17_2026-10-09" / "06_selected_parameter_estimates.csv")
    official_params = official_params[(official_params["Model"] == "Model 1") & (official_params["Domain"] == "Gastrointestinal")]
    patient_se = np.sqrt(np.clip(np.diag(mats["V_patient"]), 0, np.inf))
    lock_diff = float(np.max(np.abs(patient_se - official_params["robust_SE"].to_numpy(float))))
    official_joint = pd.read_csv(ROOT / "outputs" / "step3_independent_model1_with_hamd17_2026-10-09" / "09_global_interactions.csv")
    official_joint = official_joint[(official_joint["Model"] == "Model 1") & (official_joint["Domain"] == "Gastrointestinal")]
    pg = g.shape[1]
    rows, influence_rows, term_df_rows = [], [], []
    for component, offset, columns in [
        ("Zero-Inflated Component", 0, list(g.columns)),
        ("Burden Severity", pg + 3, list(x.columns)),
    ]:
        indices = [offset + columns.index(step3.interaction_column(gi_var, week)) for week in step3.NONREF_TIMES]
        L = np.vstack([unit_contrast(len(theta), i) for i in indices])
        beta = L @ theta
        for week, i, l in zip(step3.NONREF_TIMES, indices, L):
            dfi = fay_df_v1(mats["a"], mats["a_by_site"], mats["fg"], l)
            term_df_rows.append({"component": component, "term": names[i], "coefficient": theta[i], "FG_effective_df": dfi})
        for method, vcov, ref in [
            ("Primary patient-clustered", mats["V_patient"], "Chi-square(3)"),
            ("Hospital CR0 asymptotic", mats["fg"]["V_CR0"], "Chi-square(3)"),
            ("Hospital CR0 small-sample", mats["fg"]["V_CR0"], "Approximate F(3,20)"),
            ("Hospital CR1 asymptotic", mats["fg"]["V_CR1"], "Chi-square(3)"),
            ("Hospital CR1 small-sample", mats["fg"]["V_CR1"], "Approximate F(3,20)"),
            ("Hospital Fay-Graubard diagnostic", mats["fg"]["V_FG"], "Chi-square(3), asymptotic only"),
        ]:
            block = sym(L @ vcov @ L.T)
            W, rank = wald(beta, block)
            if ref.startswith("Approximate F"):
                p = float(f.sf(W / 3.0, 3, M - 1))
                statistic = W / 3.0
            else:
                p = float(chi2.sf(W, 3))
                statistic = W
            rows.append({
                "stage": "Step 3 Model 1", "domain": "Gastrointestinal", "component": component,
                "method": method, "reference_distribution": ref, "statistic": statistic,
                "underlying_Wald_chi2": W, "numerator_df": 3, "denominator_df": M - 1 if ref.startswith("Approximate F") else np.nan,
                "raw_P": p, "covariance_block_rank": rank,
                "covariance_block_finite": bool(np.isfinite(block).all()),
                "covariance_block_min_eigenvalue": float(np.linalg.eigvalsh(block).min()),
                "covariance_block_condition": float(np.linalg.cond(block)),
                "inference_label": "Exploratory/Approximate inference" if method != "Primary patient-clustered" else "Locked primary inference",
            })
        for site in mats["fg"]["sites"]:
            z0 = L @ mats["fg"]["Ainv"] @ mats["fg"]["U"][site]
            zfg = L @ mats["fg"]["Ainv"] @ mats["fg"]["adjusted_U"][site]
            influence_rows.append({"component": component, "site": site, "CR0_joint_trace_contribution": float(z0 @ z0), "FG_joint_trace_contribution": float(zfg @ zfg), "one_step_interaction_vector_shift_norm": float(np.linalg.norm(z0))})
    result = pd.DataFrame(rows)
    for method in result["method"].unique():
        mask = result["method"] == method
        result.loc[mask, "BH_FDR_P_within_2_GI_component_tests"] = bh(result.loc[mask, "raw_P"])
    influence = pd.DataFrame(influence_rows)
    influence["CR0_joint_trace_share"] = influence.groupby("component")["CR0_joint_trace_contribution"].transform(lambda s: s / s.sum())
    influence["FG_joint_trace_share"] = influence.groupby("component")["FG_joint_trace_contribution"].transform(lambda s: s / s.sum())
    # Reproduce official joint tests exactly before any hospital inference is accepted.
    reproduced = result[result["method"] == "Primary patient-clustered"].copy()
    map_component = {"Zero-Inflated Component": "Zero-Inflated", "Burden Severity": "Ordinal"}
    reproduced["official_component"] = reproduced["component"].map(map_component)
    reproduced = reproduced.merge(official_joint[["Component", "Wald_chi2", "raw_P"]], left_on="official_component", right_on="Component", how="left")
    max_w_diff = float(np.max(np.abs(reproduced["underlying_Wald_chi2"] - reproduced["Wald_chi2"])))
    max_p_diff = float(np.max(np.abs(reproduced["raw_P_x"] - reproduced["raw_P_y"])))
    return result, influence, pd.DataFrame(term_df_rows), mats, lock_diff, max_w_diff, max_p_diff, names, theta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FREEZE)
    df["education3"] = pd.Categorical(df["education"].map(step2_m2.EDUCATION_MAP), categories=step2_m2.EDUCATION_LEVELS)
    week_counts = df["time"].value_counts().to_dict()
    lock = {
        "freeze_file": str(FREEZE), "freeze_sha256": sha256(FREEZE),
        "participants": int(df["participant_ID"].nunique()), "person_visits": int(len(df)),
        "hospital_sites": int(df["site"].nunique()), "week_counts": week_counts,
        "participant_site_mapping_unique": bool(df.groupby("participant_ID")["site"].nunique().eq(1).all()),
        "participant_time_unique": bool(not df.duplicated(["participant_ID", "time"]).any()),
    }
    if not (lock["freeze_sha256"] == EXPECTED_HASH and lock["participants"] == 1468 and lock["person_visits"] == 4799 and lock["hospital_sites"] == 21 and all(week_counts.get(k) == v for k, v in EXPECTED_WEEKS.items()) and lock["participant_site_mapping_unique"] and lock["participant_time_unique"]):
        raise RuntimeError(f"Frozen data lock failed: {lock}")
    y = df["burden_ord4"].to_numpy(int)
    s2, s2_influence, s2_mats, s2_se_diff, s2_names, s2_theta = step2_results(df, y)
    s3, s3_influence, s3_term_df, s3_mats, s3_se_diff, s3_w_diff, s3_p_diff, s3_names, s3_theta = step3_results(df, y)
    if max(s2_se_diff, s3_se_diff) >= 1e-8 or max(s3_w_diff, s3_p_diff) >= 1e-8:
        raise RuntimeError("Formal patient-clustered inference reproduction failed")

    cov_diag = []
    for stage, mats in [("Step 2 Model 1", s2_mats), ("Step 3 Model 1 Gastrointestinal", s3_mats)]:
        for key in ["V_patient", "V_CR0", "V_CR1", "V_FG"]:
            vcov = mats[key] if key == "V_patient" else mats["fg"][key]
            row = covariance_diagnostics(vcov, f"{stage}/{key}")
            row.update({"stage": stage, "parameters": vcov.shape[0], "cluster_count": 1468 if key == "V_patient" else 21})
            cov_diag.append(row)
    s2_target_names = s2.loc[s2["method"] == "Primary patient-clustered normal", "parameter"].tolist()
    s2_target_idx = [s2_names.index(name) for name in s2_target_names]
    for key in ["V_CR0", "V_CR1", "V_FG"]:
        block = s2_mats["fg"][key][np.ix_(s2_target_idx, s2_target_idx)]
        row = covariance_diagnostics(block, f"Step 2 Model 1/{key}_18-target-block")
        row.update({"stage": "Step 2 Model 1", "parameters": 18, "cluster_count": 21})
        cov_diag.append(row)
    s3_target_idx = [s3_names.index(name) for name in s3_term_df["term"].tolist()]
    for key in ["V_CR0", "V_CR1", "V_FG"]:
        full = s3_mats["fg"][key]
        for component, idx in [("Zero-Inflated Component", s3_target_idx[:3]), ("Burden Severity", s3_target_idx[3:])]:
            block = full[np.ix_(idx, idx)]
            row = covariance_diagnostics(block, f"Step 3 Model 1/{key}_{component}_3df-block")
            row.update({"stage": "Step 3 Model 1 Gastrointestinal", "parameters": 3, "cluster_count": 21})
            cov_diag.append(row)
    cov_diag = pd.DataFrame(cov_diag)

    # Main comparison table: locked primary normal CI vs hospital CR1 with t(20) CI.
    prim = s2[s2["method"] == "Primary patient-clustered normal"].copy()
    hosp = s2[s2["method"] == "Hospital CR1 t(20)"].copy()
    main = prim[["component", "domain", "OR", "OR_CI_low", "OR_CI_high", "P", "BH_FDR_P_within_9_domains"]].merge(
        hosp[["component", "domain", "OR", "OR_CI_low", "OR_CI_high", "SE", "P", "BH_FDR_P_within_9_domains"]],
        on=["component", "domain"], suffixes=("_primary", "_hospital")
    )
    main["Primary model OR (95% CI)"] = main.apply(lambda r: f"{r.OR_primary:.2f} ({r.OR_CI_low_primary:.2f}-{r.OR_CI_high_primary:.2f})", axis=1)
    main["Hospital-clustered OR (95% CI)"] = main.apply(lambda r: f"{r.OR_hospital:.2f} ({r.OR_CI_low_hospital:.2f}-{r.OR_CI_high_hospital:.2f})", axis=1)
    main["Primary_BH_significant"] = main["BH_FDR_P_within_9_domains_primary"] < 0.05
    main["Hospital_BH_significant"] = main["BH_FDR_P_within_9_domains_hospital"] < 0.05
    main["Conclusion_consistency"] = np.where(main["Primary_BH_significant"] == main["Hospital_BH_significant"], "Consistent", "Changed")
    main["domain_order"] = pd.Categorical(main["domain"], categories=FORMAL_DOMAIN_ORDER, ordered=True)
    main["component_order"] = pd.Categorical(main["component"], categories=["Zero-Inflated Component", "Burden Severity"], ordered=True)
    main = main.sort_values(["component_order", "domain_order"]).drop(columns=["domain_order", "component_order"])

    # Maximum site influence summaries.
    s2_inf_summary = s2_influence.loc[s2_influence.groupby(["stage", "component", "target"])["CR0_variance_share"].idxmax()].copy()
    s3_inf_summary = s3_influence.loc[s3_influence.groupby("component")["CR0_joint_trace_share"].idxmax()].copy()
    site_sizes = df.assign(site_key=df["site"].astype(str)).groupby("site_key").agg(participants=("participant_ID", "nunique"), person_visits=("participant_ID", "size")).reset_index().rename(columns={"site_key": "site"})

    s2.to_csv(OUT / "Step2_full_precision.csv", index=False, encoding="utf-8-sig")
    s3.to_csv(OUT / "Step3_GI_global_full_precision.csv", index=False, encoding="utf-8-sig")
    main.to_csv(OUT / "Supplementary_Table_main_data.csv", index=False, encoding="utf-8-sig")
    s2_influence.to_csv(OUT / "Step2_site_influence_all.csv", index=False, encoding="utf-8-sig")
    s2_inf_summary.to_csv(OUT / "Step2_max_site_influence.csv", index=False, encoding="utf-8-sig")
    s3_influence.to_csv(OUT / "Step3_site_influence_all.csv", index=False, encoding="utf-8-sig")
    s3_inf_summary.to_csv(OUT / "Step3_max_site_influence.csv", index=False, encoding="utf-8-sig")
    s3_term_df.to_csv(OUT / "Step3_FG_scalar_term_df.csv", index=False, encoding="utf-8-sig")
    cov_diag.to_csv(OUT / "Covariance_diagnostics.csv", index=False, encoding="utf-8-sig")
    site_sizes.to_csv(OUT / "Hospital_site_sizes.csv", index=False, encoding="utf-8-sig")
    lock.update({
        "step2_max_abs_patient_SE_diff": s2_se_diff,
        "step3_max_abs_patient_SE_diff": s3_se_diff,
        "step3_max_abs_primary_joint_Wald_diff": s3_w_diff,
        "step3_max_abs_primary_joint_P_diff": s3_p_diff,
        "step2_max_abs_total_score": float(np.max(np.abs(np.sum(s2_mats["score_obs"], axis=0)))),
        "step3_max_abs_total_score": float(np.max(np.abs(np.sum(s3_mats["score_obs"], axis=0)))),
        "step2_patient_to_site_score_aggregation_diff": s2_mats["aggregation_diff"],
        "step3_patient_to_site_score_aggregation_diff": s3_mats["aggregation_diff"],
        "step2_CR1_multiplier": s2_mats["fg"]["CR1_multiplier"],
        "step3_CR1_multiplier": s3_mats["fg"]["CR1_multiplier"],
        "optimization_or_refitting_performed": False,
    })
    (OUT / "analysis_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "main": json_records(main), "step2": json_records(s2), "step3": json_records(s3),
        "covariance": json_records(cov_diag), "step2Influence": json_records(s2_inf_summary),
        "step3Influence": json_records(s3_inf_summary), "step3FgDf": json_records(s3_term_df),
        "siteSizes": json_records(site_sizes), "lock": lock,
    }
    (OUT / "workbook_payload.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "out": str(OUT), "lock": lock,
        "main_changed": main[main["Conclusion_consistency"] == "Changed"][["component", "domain", "Primary_BH_significant", "Hospital_BH_significant"]].to_dict("records"),
        "step3_CR1_t20": s3[s3["method"] == "Hospital CR1 small-sample"][["component", "underlying_Wald_chi2", "raw_P", "BH_FDR_P_within_2_GI_component_tests"]].to_dict("records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




