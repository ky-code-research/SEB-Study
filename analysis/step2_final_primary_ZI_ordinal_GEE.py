from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit
from statsmodels.stats.multitest import multipletests

import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2B3_fixed_sample_covariate_attenuation as b3


ROOT = b1.ROOT
OUT = ROOT / "outputs" / "step2_final_primary_ZI_ordinal_GEE"
SOURCE = b1.SOURCE

EXPECTED_PARTICIPANTS = 1468
EXPECTED_VISITS = 4761
PRIMARY_RHS = b3.MODEL_RHS["M3"]
DOMAIN_VARS = [b1.DOMAIN_VAR[d] for d in b1.DOMAINS]


def log(msg: str) -> None:
    print(f"Step2 final primary: {msg}", flush=True)


def names_for(G_cols: list[str], X_cols: list[str]) -> list[str]:
    return (
        [f"mixing_gamma:{c}" for c in G_cols]
        + ["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"]
        + [f"ordinal_beta:{c}" for c in X_cols]
    )


def make_starts(y, G, X) -> list[dict]:
    beta0, raw0, method = b1.fit_ordinal_start(y, X)
    starts = []
    for extra in [0.01, 0.03, 0.05, 0.10, 0.15, 0.30, 0.50]:
        gamma = np.zeros(G.shape[1])
        gamma[0] = math.log((1.0 - extra) / extra)
        starts.append({
            "start": f"ordinary_start_extra_zero_{extra:.2f}",
            "theta": np.r_[gamma, raw0, beta0],
            "start_method": method,
        })
    rng = np.random.default_rng(20260929)
    scale = np.r_[np.repeat(0.12, G.shape[1]), np.repeat(0.06, 3), np.repeat(0.06, X.shape[1])]
    for i in range(1, 6):
        base = starts[i % len(starts)]["theta"]
        starts.append({
            "start": f"random_perturbation_{i}",
            "theta": base + rng.normal(0, scale),
            "start_method": method,
        })
    return starts


def theta_summary(start, theta, y, G, X, success, message, nit, start_method=None) -> dict:
    ll, score = b1.obs_loglik_score(theta, y, G, X)
    gamma, _, _, alpha, _ = b1.unpack(theta, G.shape[1], X.shape[1])
    pi = expit(np.asarray(G) @ gamma)
    return {
        "start": start,
        "success": bool(success),
        "message": str(message),
        "iterations": int(nit),
        "objective": float(ll.sum()),
        "objective_per_observation": float(ll.mean()),
        "max_absolute_score_gradient": float(np.max(np.abs(score.sum(axis=0)))),
        "threshold_alpha0": float(alpha[0]),
        "threshold_alpha1": float(alpha[1]),
        "threshold_alpha2": float(alpha[2]),
        "thresholds_ordered": bool(alpha[0] < alpha[1] < alpha[2]),
        "median_pi": float(np.median(pi)),
        "min_pi": float(pi.min()),
        "max_pi": float(pi.max()),
        "start_method": start_method,
    }


def fit_primary(y, G, X):
    rows = []
    fits = []
    for st in make_starts(y, G, X):
        res = optimize.minimize(
            lambda th: b1.neg_obj_grad(th, y, G, X),
            st["theta"],
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 3000, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50},
        )
        row = theta_summary(st["start"], res.x, y, G, X, res.success, res.message, res.nit, st["start_method"])
        rows.append(row)
        fits.append({"theta": res.x, "summary": row})
        log(f"{st['start']} success={res.success} objective={row['objective']:.6f} score={row['max_absolute_score_gradient']:.3g}")

    best = max(fits, key=lambda z: z["summary"]["objective"])
    ref = optimize.minimize(
        lambda th: b1.neg_obj_grad(th, y, G, X),
        best["theta"],
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 80, "maxcor": 50},
    )
    ref_row = theta_summary("refined_from_best_start", ref.x, y, G, X, ref.success, ref.message, ref.nit, "refined_BFGS")
    rows.append(ref_row)
    log(f"refined success={ref.success} objective={ref_row['objective']:.6f} score={ref_row['max_absolute_score_gradient']:.3g}")
    return ref.x, pd.DataFrame(rows)


def domain_inference(theta, se, G_cols, X_cols):
    rows = []
    pg = len(G_cols)
    for domain in b1.DOMAINS:
        var = b1.DOMAIN_VAR[domain]
        gi = G_cols.index(var)
        gamma = float(theta[gi])
        gamma_se = float(se[gi])
        z = gamma / gamma_se if gamma_se > 0 else np.nan
        rows.append({
            "Panel": "Panel A. Zero-inflation component",
            "Domain": domain,
            "component": "mixing",
            "parameter": var,
            "Coefficient": gamma,
            "Robust SE": gamma_se,
            "OR": math.exp(-gamma),
            "95% CI": f"{math.exp(-(gamma + 1.96 * gamma_se)):.3f}, {math.exp(-(gamma - 1.96 * gamma_se)):.3f}",
            "CI low": math.exp(-(gamma + 1.96 * gamma_se)),
            "CI high": math.exp(-(gamma - 1.96 * gamma_se)),
            "Raw P": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "OR definition": "OR_extra_zero=exp(-gamma)",
        })

        xi = pg + 3 + X_cols.index(var)
        beta = float(theta[xi])
        beta_se = float(se[xi])
        z = beta / beta_se if beta_se > 0 else np.nan
        rows.append({
            "Panel": "Panel B. Ordinary ordinal burden component",
            "Domain": domain,
            "component": "full_ordinal",
            "parameter": var,
            "Coefficient": beta,
            "Robust SE": beta_se,
            "OR": math.exp(beta),
            "95% CI": f"{math.exp(beta - 1.96 * beta_se):.3f}, {math.exp(beta + 1.96 * beta_se):.3f}",
            "CI low": math.exp(beta - 1.96 * beta_se),
            "CI high": math.exp(beta + 1.96 * beta_se),
            "Raw P": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "OR definition": "OR_higher_burden=exp(beta)",
        })
    out = pd.DataFrame(rows)
    out["BH-FDR P"] = np.nan
    for component, idx in out.groupby("component").groups.items():
        idx = list(idx)
        p = out.loc[idx, "Raw P"].to_numpy(float)
        mask = np.isfinite(p)
        q = np.full(len(p), np.nan)
        if mask.any():
            q[mask] = multipletests(p[mask], method="fdr_bh")[1]
        out.loc[idx, "BH-FDR P"] = q
    return out


def sample_qc(base, long, analysis, mapping, zmap, sha) -> dict[str, pd.DataFrame]:
    required = [
        "participant_ID", "site", "time", "visit_order", "burden_ord4", "age_z", "sex",
        "education", "first_episode", "baseline_med_class", "baseline_HAMD17_z",
        "baseline_SDS_z", "concurrent_HAMD17_z",
    ] + DOMAIN_VARS
    valid = long["burden_ord4"].isin([0, 1, 2, 3])
    sample = pd.DataFrame([
        {"stage": "baseline workbook sheet", "participants": base["participant_ID"].nunique(), "person_visits": np.nan},
        {"stage": "long rows with valid FIBSER burden", "participants": long.loc[valid, "participant_ID"].nunique(), "person_visits": int(valid.sum())},
        {"stage": "primary complete-case analysis", "participants": analysis["participant_ID"].nunique(), "person_visits": len(analysis)},
    ])
    missing_rows = []
    for c in required:
        missing_rows.append({
            "variable": c,
            "missing_rows_in_long": int(long[c].isna().sum()),
            "nonmissing_rows_in_long": int(long[c].notna().sum()),
            "missing_among_valid_FIBSER": int(long.loc[valid, c].isna().sum()),
        })
    exclusions = pd.DataFrame([{
        "valid_FIBSER_person_visits": int(valid.sum()),
        "complete_case_person_visits": len(analysis),
        "missingness_excluded_person_visits": int(valid.sum() - len(analysis)),
        "valid_FIBSER_participants": long.loc[valid, "participant_ID"].nunique(),
        "complete_case_participants": analysis["participant_ID"].nunique(),
        "participants_lost_from_missingness": long.loc[valid, "participant_ID"].nunique() - analysis["participant_ID"].nunique(),
        "complete_case_rule": "drop rows missing outcome, time, all 9 PRISE domains, age, sex, education, first_episode, baseline_HAMD17, baseline_SDS, baseline_antidepressant_class, or concurrent_HAMD",
    }])
    outcome = analysis["burden_ord4"].value_counts().reindex([0, 1, 2, 3], fill_value=0).rename_axis("burden_ord4").reset_index(name="n")
    outcome["category"] = outcome["burden_ord4"].map({0: "0", 1: "1", 2: "2", 3: ">=3"})
    source = pd.DataFrame([{
        "source_file": str(SOURCE),
        "SHA256": sha,
        "province_site_QC_only": "province/site may remain in data but are not in primary model",
        "primary_model_excludes_site": True,
        "primary_model_excludes_province": True,
    }])
    return {
        "source_hash": source,
        "sample_construction": sample,
        "missingness_exclusions": exclusions,
        "variable_missingness": pd.DataFrame(missing_rows),
        "outcome_distribution": outcome,
        "variable_mapping": mapping,
        "z_standardization": zmap,
    }


def covariance_conditioning(V: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = []
    wanted = []
    for v in DOMAIN_VARS:
        wanted.append(f"mixing_gamma:{v}")
        wanted.append(f"ordinal_beta:{v}")
    for block, block_names in [("full_covariance", names), ("18-domain scientific covariance block", wanted)]:
        idx = [names.index(n) for n in block_names if n in names]
        sub = ((V + V.T) / 2.0)[np.ix_(idx, idx)]
        eig = np.linalg.eigvalsh(sub)
        rows.append({
            "block": block,
            "parameters": len(idx),
            "finite": bool(np.isfinite(sub).all()),
            "PSD": bool(eig.min() >= -1e-8),
            "min_eigenvalue": float(eig.min()),
            "max_eigenvalue": float(eig.max()),
            "condition_number": float(np.linalg.cond(sub)),
        })
    idx = [names.index(n) for n in wanted if n in names]
    sub = ((V + V.T) / 2.0)[np.ix_(idx, idx)]
    d = np.sqrt(np.maximum(np.diag(sub), 0.0))
    corr = sub / np.outer(d, d)
    corr[~np.isfinite(corr)] = np.nan
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    rows.append({
        "block": "18-domain scientific covariance block correlations",
        "parameters": len(idx),
        "finite": bool(np.isfinite(corr).all()),
        "PSD": np.nan,
        "min_eigenvalue": np.nan,
        "max_eigenvalue": np.nan,
        "condition_number": np.nan,
        "max_abs_correlation": float(np.nanmax(np.abs(off))),
    })
    return pd.DataFrame(rows)


def previous_m3_qc(final_domains: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prev_path = ROOT / "outputs" / "step2B3_fixed_sample_covariate_attenuation" / "02_nested_model_domain_results.xlsx"
    prev = pd.read_excel(prev_path, sheet_name="domain_results")
    prev = prev.loc[prev["model"].eq("M3")].copy()
    cur = final_domains.copy()
    cur = cur.rename(columns={"Coefficient": "coefficient", "Robust SE": "robust_SE", "Raw P": "raw_P"})
    cur["model"] = "FinalPrimary"
    keep = ["Domain", "component", "parameter", "coefficient", "robust_SE", "OR", "raw_P", "BH-FDR P"]
    comp = cur[keep].merge(prev[["Domain", "component", "parameter", "coefficient", "robust_SE", "OR", "raw_P", "BH_FDR"]], on=["Domain", "component", "parameter"], suffixes=("_final", "_previous_M3"))
    comp["coefficient_difference"] = comp["coefficient_final"] - comp["coefficient_previous_M3"]
    comp["SE_difference"] = comp["robust_SE_final"] - comp["robust_SE_previous_M3"]
    comp["direction_agreement"] = np.sign(comp["coefficient_final"]) == np.sign(comp["coefficient_previous_M3"])
    coef_tol = 1e-4
    se_tol = 1e-4
    reproduced = bool(
        comp["coefficient_difference"].abs().max() < coef_tol
        and comp["SE_difference"].abs().max() < se_tol
        and comp["direction_agreement"].all()
    )
    summary = pd.DataFrame([{
        "previous_source": str(prev_path),
        "rows_compared": len(comp),
        "max_absolute_coefficient_difference": float(comp["coefficient_difference"].abs().max()),
        "max_SE_difference": float(comp["SE_difference"].abs().max()),
        "coefficient_material_difference_threshold": coef_tol,
        "SE_material_difference_threshold": se_tol,
        "direction_agreement_count": int(comp["direction_agreement"].sum()),
        "direction_agreement_all": bool(comp["direction_agreement"].all()),
        "QC": "approximately reproduced previous M3; no material difference" if reproduced else "CHECK: material difference",
    }])
    return summary, comp


def make_report(sample_gate, fit_best, cov_diag, prev_summary, domains):
    mix = domains.loc[domains["component"].eq("mixing"), ["Domain", "Coefficient", "Robust SE", "OR", "95% CI", "Raw P", "BH-FDR P"]]
    ordn = domains.loc[domains["component"].eq("full_ordinal"), ["Domain", "Coefficient", "Robust SE", "OR", "95% CI", "Raw P", "BH-FDR P"]]
    raw_mix = int((mix["Raw P"] < 0.05).sum())
    fdr_mix = int((mix["BH-FDR P"] < 0.05).sum())
    raw_ord = int((ordn["Raw P"] < 0.05).sum())
    fdr_ord = int((ordn["BH-FDR P"] < 0.05).sum())
    return f"""# Step2 Final Primary ZI Ordinal GEE Report

## Sample Gate

{sample_gate.to_markdown(index=False)}

## Primary Model

Specification: zero-inflated proportional-odds ordinal LIM-GEE, working independence, participant-level cluster robust sandwich.

Both components used: time + 9 PRISE domains + age + sex + education + first_episode + baseline_HAMD + baseline_SDS + baseline_antidepressant_class + concurrent_HAMD.

Hospital site and province were excluded from the primary model.

## Best Refined Fit

{fit_best.to_markdown(index=False)}

## Covariance Diagnostics

{cov_diag.to_markdown(index=False)}

## QC Against Previous Frozen M3

{prev_summary.to_markdown(index=False)}

## Panel A. Zero-inflation Component

OR is `OR_extra_zero=exp(-gamma)`; OR < 1 means lower odds of belonging to the additional-zero state.

{mix.to_markdown(index=False)}

## Panel B. Ordinary Ordinal Burden Component

OR is `OR_higher_burden=exp(beta)`; OR > 1 means greater odds of a higher burden category within the ordinary ordinal component.

{ordn.to_markdown(index=False)}

## Multiplicity Counts

- Zero-inflation component: raw P<.05 = {raw_mix}/9; BH-FDR<.05 = {fdr_mix}/9.
- Ordinary ordinal burden component: raw P<.05 = {raw_ord}/9; BH-FDR<.05 = {fdr_ord}/9.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sha = b1.sha256(SOURCE)
    log("rebuilding data independently from workbook")
    base, long, analysis, mapping, zmap = b1.build_dataset()
    sample_books = sample_qc(base, long, analysis, mapping, zmap, sha)
    b1.write_book(OUT / "01_primary_sample_QC.xlsx", sample_books)

    sample_gate = pd.DataFrame([{
        "expected_participants": EXPECTED_PARTICIPANTS,
        "actual_participants": analysis["participant_ID"].nunique(),
        "expected_person_visits": EXPECTED_VISITS,
        "actual_person_visits": len(analysis),
        "gate_pass": bool(analysis["participant_ID"].nunique() == EXPECTED_PARTICIPANTS and len(analysis) == EXPECTED_VISITS),
    }])
    if not bool(sample_gate["gate_pass"].iloc[0]):
        (OUT / "Step2_final_primary_report.md").write_text(
            "# Step2 Final Primary ZI Ordinal GEE Report\n\nSTOP: sample gate failed.\n\n"
            + sample_gate.to_markdown(index=False),
            encoding="utf-8",
        )
        raise SystemExit("STOP: final primary sample did not match expected 1468 participants / 4761 person-visits")

    log("building primary M3 design without site/province")
    G, X = b1.make_design(analysis, mix_rhs=PRIMARY_RHS, ord_rhs=PRIMARY_RHS)
    G_cols, X_cols = list(G.columns), list(X.columns)
    forbidden = [c for c in G_cols + X_cols if "site" in c.lower() or "province" in c.lower()]
    if forbidden:
        raise SystemExit(f"STOP: forbidden site/province terms in primary model: {forbidden}")

    y = analysis["burden_ord4"].to_numpy()
    log(f"fitting primary model with {G.shape[1]} mixing columns and {X.shape[1]} ordinal columns")
    theta, fit_qc = fit_primary(y, G, X)

    log("computing participant-level robust sandwich covariance")
    sw = b1.sandwich(theta, y, G, X, analysis["participant_ID"])
    names = names_for(G_cols, X_cols)
    param = b1.param_table(theta, sw["se"], G_cols, X_cols)
    param["parameter_full_name"] = names
    cov_diag = covariance_conditioning(sw["V"], names)
    best_fit = fit_qc.tail(1).copy()
    model_diag = pd.concat([
        best_fit.assign(diagnostic_type="best_refined_fit"),
        pd.DataFrame([{
            "diagnostic_type": "sandwich",
            "success": bool(best_fit["success"].iloc[0]),
            "objective": float(best_fit["objective"].iloc[0]),
            "max_absolute_score_gradient": float(best_fit["max_absolute_score_gradient"].iloc[0]),
            "thresholds_ordered": bool(best_fit["thresholds_ordered"].iloc[0]),
            "median_pi": float(best_fit["median_pi"].iloc[0]),
            "min_pi": float(best_fit["min_pi"].iloc[0]),
            "max_pi": float(best_fit["max_pi"].iloc[0]),
            "covariance_finite": bool(sw["finite"]),
            "covariance_PSD": bool(sw["V_positive_semidefinite"]),
            "full_covariance_condition_number": float(cov_diag.loc[cov_diag["block"].eq("full_covariance"), "condition_number"].iloc[0]),
            "domain_block_condition_number": float(cov_diag.loc[cov_diag["block"].eq("18-domain scientific covariance block"), "condition_number"].iloc[0]),
            "max_correlation_within_domain_block": float(cov_diag.loc[cov_diag["block"].eq("18-domain scientific covariance block correlations"), "max_abs_correlation"].iloc[0]),
        }])
    ], ignore_index=True, sort=False)

    domains = domain_inference(theta, sw["se"], G_cols, X_cols)
    pub_cols = ["Domain", "Coefficient", "Robust SE", "OR", "95% CI", "Raw P", "BH-FDR P"]
    panel_a = domains.loc[domains["component"].eq("mixing"), pub_cols].copy()
    panel_b = domains.loc[domains["component"].eq("full_ordinal"), pub_cols].copy()

    log("reading previous frozen M3 only after independent fit for QC")
    prev_summary, prev_compare = previous_m3_qc(domains)
    if prev_summary["QC"].iloc[0] == "CHECK: material difference":
        b1.write_book(OUT / "02_primary_model_diagnostics.xlsx", {
            "fit_multistart": fit_qc,
            "primary_model_diagnostics": model_diag,
            "covariance_conditioning": cov_diag,
            "parameters": param,
        })
        b1.write_book(OUT / "03_final_domain_inference.xlsx", {
            "Panel A Zero-inflation": panel_a,
            "Panel B Ordinary ordinal": panel_b,
            "all_domain_inference": domains,
        })
        b1.write_book(OUT / "04_primary_vs_previous_M3_QC.xlsx", {
            "summary": prev_summary,
            "coefficient_comparison": prev_compare,
        })
        raise SystemExit("STOP: independent final primary fit differs materially from previous frozen M3")

    b1.write_book(OUT / "02_primary_model_diagnostics.xlsx", {
        "fit_multistart": fit_qc,
        "primary_model_diagnostics": model_diag,
        "covariance_conditioning": cov_diag,
        "parameters": param,
    })
    b1.write_book(OUT / "03_final_domain_inference.xlsx", {
        "Panel A Zero-inflation": panel_a,
        "Panel B Ordinary ordinal": panel_b,
        "all_domain_inference": domains,
    })
    b1.write_book(OUT / "04_primary_vs_previous_M3_QC.xlsx", {
        "summary": prev_summary,
        "coefficient_comparison": prev_compare,
    })
    report = make_report(sample_gate, best_fit, cov_diag, prev_summary, domains)
    (OUT / "Step2_final_primary_report.md").write_text(report, encoding="utf-8")
    log(f"completed outputs: {OUT}")


if __name__ == "__main__":
    main()


