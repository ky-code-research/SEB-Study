from __future__ import annotations

import json
import math
import os
import platform
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import optimize, stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


from public_paths import ROOT
DEFAULT_OUT = ROOT / "outputs" / "binary_logistic_GEE_tolerability_S6_S7_2026-10-19_final"
OUT = Path(os.environ.get("S6_S7_OUTPUT_DIR", str(DEFAULT_OUT)))
if OUT.exists():
    raise FileExistsError(f"Output directory already exists; choose a new directory: {OUT}")

# The validated reconstruction module reads only the original Excel workbook.
# Redirect its import-time audit directory to this run's new directory.
os.environ["BINARY_GEE_OUTPUT_DIR"] = str(OUT)
import run_binary_logistic_gee_sensitivity as rebuild  # noqa: E402
import secondary_prise_tolerability_qc as tolqc  # noqa: E402


FREEZE_ID = rebuild.FREEZE_ID
EXPECTED_SOURCE_SHA256 = rebuild.EXPECTED_SOURCE_SHA256
TARGET_DOMAINS = [
    "Sleep",
    "Gastrointestinal",
    "Nervous system",
    "Heart",
    "Eye/Ear",
    "Skin",
    "Sexual functioning",
    "Genital/Urinary",
]
DISPLAY_TO_QC = {
    "Sleep": "Sleep",
    "Gastrointestinal": "Gastrointestinal",
    "Nervous system": "Nervous system",
    "Heart": "Cardiac",
    "Eye/Ear": "Eye/Ear",
    "Skin": "Skin",
    "Sexual functioning": "Sexual functioning",
    "Genital/Urinary": "Genital/Urinary",
}
DISPLAY_TO_VAR = {name: var for name, var in rebuild.DOMAINS}
LEVELS = ["None", "Tolerable", "Distressing"]
CONTRASTS = ["Tolerable vs None", "Distressing vs None", "Distressing vs Tolerable"]
OUTCOMES = [
    ("any_burden", "S6", "Any burden (FIBSER 1-6)"),
    ("moderate_burden", "S7", "Moderate or greater burden (FIBSER 3-6)"),
]
SECONDARY = (
    ROOT
    / "outputs"
    / "secondary_analysis_PRISE_tolerability_ESGEE_2026-10-11_plotting_contrasts_2026-10-12_preliminary"
    / "Secondary_Plotting_Contrasts_48.csv"
)


def bh_adjust_with_na(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna() & np.isfinite(values.astype(float))
    if valid.any():
        result.loc[valid] = multipletests(values.loc[valid].astype(float), method="fdr_bh")[1]
    return result


def complete_separation_lp(x: np.ndarray, y: np.ndarray) -> dict:
    signed = np.where(y == 1, 1.0, -1.0)
    res = optimize.linprog(
        c=np.zeros(x.shape[1]),
        A_ub=-(signed[:, None] * x),
        b_ub=-np.ones(len(y)),
        bounds=[(None, None)] * x.shape[1],
        method="highs",
    )
    return {
        "complete_separation_detected": bool(res.success),
        "lp_status": int(res.status),
        "lp_message": str(res.message),
    }


def model_formula(outcome: str, target_var: str) -> str:
    other_vars = [v for _, v in rebuild.DOMAINS if v != target_var]
    return (
        f"{outcome} ~ C(target_tolerability, Treatment(reference='None')) + "
        + " + ".join(other_vars)
        + " + C(time, Treatment(reference='Week2')) + age_z + "
        "C(sex, Treatment(reference='Female')) + "
        "C(education3, Treatment(reference='College or above')) + "
        "C(first_episode, Treatment(reference='FirstEpisode')) + baseline_HAMD17_z + baseline_SDS_z + "
        "C(baseline_med_class, Treatment(reference='SNRI')) + concurrent_HAMD17_z"
    )


def build_analysis():
    analysis, long, occ_map, sheet_audit, conflicts, zmap, sheet_names, source_hash = rebuild.rebuild_analysis()
    freeze_gate = rebuild.stop_gate(analysis)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("STOP: source hash changed")
    if len(occ_map) != 72 or not occ_map["status"].eq("PASS").all():
        raise RuntimeError("STOP: 72-item occurrence mapping gate failed")

    tol_collapsed, tol_map, tol_sheet_keys = tolqc.build_raw_followup()
    if len(tol_map) != 64 or not tol_map["status"].eq("PASS").all():
        raise RuntimeError("STOP: 64-item tolerability mapping gate failed")
    if not tol_sheet_keys["status"].eq("PASS").all():
        raise RuntimeError("STOP: duplicate patient-time key in a source sheet")

    merge_cols = ["participant_ID", "time"]
    for display in TARGET_DOMAINS:
        qcd = DISPLAY_TO_QC[display]
        qvar = tolqc.MODEL_VAR[qcd]
        expected = DISPLAY_TO_VAR[display]
        if qvar != expected:
            raise RuntimeError(f"STOP: variable-name mapping mismatch for {display}: {qvar} != {expected}")
        merge_cols += [f"{qvar}__occ_raw", f"{qvar}__tol_raw"]

    tol_collapsed = tol_collapsed[merge_cols].copy()
    tol_collapsed["time"] = tol_collapsed["time"].astype(str)
    analysis = analysis.copy()
    analysis["time"] = analysis["time"].astype(str)
    merged = analysis.merge(tol_collapsed, on=["participant_ID", "time"], how="left", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise RuntimeError("STOP: tolerance reconstruction failed patient/time matching")
    merged = merged.drop(columns="_merge")
    if merged.duplicated(["participant_ID", "time"]).any():
        raise RuntimeError("STOP: duplicate patient/time after raw merge")

    coding_rows = []
    for display in TARGET_DOMAINS:
        var = DISPLAY_TO_VAR[display]
        raw_occ = merged[f"{var}__occ_raw"]
        occurrence_expected = raw_occ.map({1.0: 1.0, 2.0: 0.0})
        occ_match = merged[var].astype(float).eq(occurrence_expected.astype(float))
        if not occ_match.all():
            raise RuntimeError(f"STOP: raw occurrence to analysis mismatch for {display}")
        level, issue = tolqc.classify_tolerability(raw_occ, merged[f"{var}__tol_raw"])
        merged[f"tol__{var}"] = level
        merged[f"tol_issue__{var}"] = issue
        contradictions = issue.str.startswith("contradiction").sum()
        invalid = issue.str.startswith("invalid").sum()
        if contradictions or invalid:
            raise RuntimeError(f"STOP: contradiction/invalid tolerance coding for {display}: {contradictions}/{invalid}")
        counts = level.value_counts().to_dict()
        issues = issue.loc[issue.ne("")].value_counts().to_dict()
        coding_rows.append(
            {
                "Domain": display,
                "Analysis variable": var,
                "Occurrence mapping match": bool(occ_match.all()),
                "Classified visits": int(level.notna().sum()),
                "Excluded visits": int(level.isna().sum()),
                "None": int(counts.get("None", 0)),
                "Tolerable": int(counts.get("Tolerable", 0)),
                "Distressing": int(counts.get("Distressing", 0)),
                "True missing after present": int(issues.get("true_missing_tolerability_after_present", 0)),
                "Missing occurrence/not assessed": int(issues.get("missing_occurrence_or_not_assessed", 0)),
                "Contradictions": int(contradictions),
                "Invalid codes": int(invalid),
                "Status": "PASS",
            }
        )

    merged["time"] = pd.Categorical(merged["time"], categories=rebuild.WEEKS, ordered=True)
    return (
        merged,
        freeze_gate,
        occ_map,
        tol_map,
        tol_sheet_keys,
        pd.DataFrame(coding_rows),
        sheet_audit,
        conflicts,
        zmap,
        sheet_names,
        source_hash,
    )


def design_vif(xdf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = np.asarray(xdf, dtype=float)
    rank = int(np.linalg.matrix_rank(x))
    rows = []
    for j, col in enumerate(xdf.columns):
        if col == "Intercept":
            continue
        try:
            value = float(variance_inflation_factor(x, j))
        except Exception:
            value = np.inf
        rows.append({"Parameter": col, "VIF": value})
    vif = pd.DataFrame(rows)
    nonconstant = [c for c in xdf.columns if c != "Intercept" and float(xdf[c].std(ddof=0)) > 0]
    sx = np.asarray(xdf[nonconstant], dtype=float)
    sx = (sx - sx.mean(axis=0)) / sx.std(axis=0, ddof=0)
    summary = {
        "Design rows": int(xdf.shape[0]),
        "Design columns": int(xdf.shape[1]),
        "Design rank": rank,
        "Full rank": bool(rank == xdf.shape[1]),
        "Scaled condition number": float(np.linalg.cond(sx)),
        "Maximum VIF": float(vif["VIF"].max()),
    }
    return vif, summary


def cooccurrence_rows(d: pd.DataFrame, outcome_label: str, target_domain: str) -> list[dict]:
    out = []
    domain_vars = [v for _, v in rebuild.DOMAINS]
    corr = d[domain_vars].astype(float).corr()
    name = {v: n for n, v in rebuild.DOMAINS}
    for i, v1 in enumerate(domain_vars):
        for v2 in domain_vars[i + 1 :]:
            tab = pd.crosstab(d[v1].astype(int), d[v2].astype(int)).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            out.append(
                {
                    "Outcome": outcome_label,
                    "Target domain model": target_domain,
                    "Domain 1": name[v1],
                    "Domain 2": name[v2],
                    "Neither": int(tab.loc[0, 0]),
                    "Domain 2 only": int(tab.loc[0, 1]),
                    "Domain 1 only": int(tab.loc[1, 0]),
                    "Both": int(tab.loc[1, 1]),
                    "Phi correlation": float(corr.loc[v1, v2]),
                }
            )
    return out


def safe_exp(x: float) -> float:
    return float(np.exp(np.clip(x, -700, 700)))


def fit_one(analysis: pd.DataFrame, outcome: str, table_id: str, outcome_label: str, domain: str):
    target_var = DISPLAY_TO_VAR[domain]
    d = analysis.loc[analysis[f"tol__{target_var}"].notna()].copy()
    d["target_tolerability"] = pd.Categorical(d[f"tol__{target_var}"], categories=LEVELS)
    formula = model_formula(outcome, target_var)
    ydf, xdf = patsy.dmatrices(formula, d, return_type="dataframe")
    if len(xdf) != len(d) or xdf.isna().any().any() or ydf.isna().any().any():
        raise RuntimeError(f"STOP: model-matrix mismatch for {table_id}/{domain}")
    tol_cols = {
        level: next((c for c in xdf.columns if "target_tolerability" in c and f"[T.{level}]" in c), None)
        for level in ["Tolerable", "Distressing"]
    }
    if any(v is None for v in tol_cols.values()):
        raise RuntimeError(f"STOP: target tolerance parameters missing for {table_id}/{domain}")
    expected_other = [v for _, v in rebuild.DOMAINS if v != target_var]
    if not all(v in xdf.columns for v in expected_other) or target_var in xdf.columns:
        raise RuntimeError(f"STOP: other-domain design mapping failed for {table_id}/{domain}")

    vif, design = design_vif(xdf)
    if not design["Full rank"]:
        raise RuntimeError(f"STOP: rank-deficient design for {table_id}/{domain}")
    y = np.asarray(ydf).reshape(-1).astype(int)
    separation = complete_separation_lp(np.asarray(xdf, float), y)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        model = sm.GEE.from_formula(
            formula,
            groups="participant_ID",
            data=d,
            family=sm.families.Binomial(),
            cov_struct=Exchangeable(),
        )
        result = model.fit(maxiter=400, ctol=1e-9, cov_type="robust")
        # ResourceWarning about a delayed Excel file-handle cleanup can surface
        # during fitting but is unrelated to GEE estimation; retain model/numeric
        # warnings only.
        fit_warnings = [str(w.message) for w in ws if not isinstance(w.message, ResourceWarning)]

    cov_df = result.cov_params()
    cov = np.asarray(cov_df, dtype=float)
    cov_finite = bool(np.isfinite(cov).all())
    cov_symmetric = bool(np.allclose(cov, cov.T, rtol=1e-8, atol=1e-10))
    min_eig = float(np.linalg.eigvalsh((cov + cov.T) / 2).min()) if cov_finite else np.nan
    cov_valid = bool(cov_finite and cov_symmetric and min_eig >= -1e-8)

    tab = pd.crosstab(d["target_tolerability"], d[outcome]).reindex(index=LEVELS, columns=[0, 1], fill_value=0)
    table_rows = []
    level_sparse = {}
    for level in LEVELS:
        n0, n1 = int(tab.loc[level, 0]), int(tab.loc[level, 1])
        level_sparse[level] = n0 < 5 or n1 < 5
        table_rows.append(
            {
                "Outcome": outcome_label,
                "Table": table_id,
                "Domain": domain,
                "Tolerability": level,
                "Non-events": n0,
                "Events": n1,
                "Total": n0 + n1,
                "Event percentage": 100 * n1 / (n0 + n1) if n0 + n1 else np.nan,
                "Sparse cell <5": bool(n0 < 5 or n1 < 5),
                "Zero cell": bool(n0 == 0 or n1 == 0),
            }
        )

    time_rows = []
    for week in rebuild.WEEKS:
        for level in LEVELS:
            g = d.loc[(d["time"].astype(str) == week) & d["target_tolerability"].eq(level)]
            ev = int(g[outcome].sum())
            time_rows.append(
                {
                    "Outcome": outcome_label,
                    "Table": table_id,
                    "Domain": domain,
                    "Time": week,
                    "Tolerability": level,
                    "Non-events": int(len(g) - ev),
                    "Events": ev,
                    "Total": int(len(g)),
                    "Sparse event cell <5": bool(ev < 5),
                    "Zero event cell": bool(ev == 0),
                }
            )

    contrast_specs = [
        ("Tolerable vs None", {tol_cols["Tolerable"]: 1.0}, ["Tolerable"]),
        ("Distressing vs None", {tol_cols["Distressing"]: 1.0}, ["Distressing"]),
        (
            "Distressing vs Tolerable",
            {tol_cols["Distressing"]: 1.0, tol_cols["Tolerable"]: -1.0},
            ["Distressing", "Tolerable"],
        ),
    ]
    contrast_rows = []
    trace_rows = []
    zcrit = stats.norm.ppf(0.975)
    for contrast, weights, involved_levels in contrast_specs:
        c = np.zeros(len(result.params))
        for param, weight in weights.items():
            c[result.params.index.get_loc(param)] = weight
        estimate = float(c @ result.params.to_numpy(float))
        variance = float(c @ cov @ c) if cov_finite else np.nan
        se = math.sqrt(variance) if np.isfinite(variance) and variance > 0 else np.nan
        z = estimate / se if np.isfinite(se) and se > 0 else np.nan
        p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        low, high = estimate - zcrit * se, estimate + zcrit * se
        estimate_finite = bool(np.isfinite([estimate, se, p, low, high]).all())
        extreme_or = bool(abs(estimate) > math.log(20)) if np.isfinite(estimate) else True
        large_se = bool(se > 2) if np.isfinite(se) else True
        wide_ci = bool((high - low) > 2 * math.log(10)) if estimate_finite else True
        zero_relevant = any(int(tab.loc[level, 0]) == 0 or int(tab.loc[level, 1]) == 0 for level in involved_levels)
        sparse_relevant = any(level_sparse[level] for level in involved_levels)
        reliable = bool(
            estimate_finite
            and cov_valid
            and not separation["complete_separation_detected"]
            and not zero_relevant
            and not large_se
        )
        flags = []
        if sparse_relevant:
            flags.append("relevant outcome cell <5")
        if zero_relevant:
            flags.append("relevant zero outcome cell")
        if extreme_or:
            flags.append("extreme OR (<0.05 or >20)")
        if large_se:
            flags.append("large robust SE (>2)")
        if wide_ci:
            flags.append("CI spans >100-fold")
        if not estimate_finite:
            flags.append("non-finite contrast")
        if separation["complete_separation_detected"]:
            flags.append("complete separation")
        contrast_rows.append(
            {
                "Table": table_id,
                "Outcome": outcome_label,
                "PRISE domain": domain,
                "Contrast": contrast,
                "Coefficient contrast": estimate,
                "Contrast variance": variance,
                "Robust SE": se,
                "Wald z": z,
                "Adjusted OR": safe_exp(estimate) if estimate_finite else np.nan,
                "CI low": safe_exp(low) if estimate_finite else np.nan,
                "CI high": safe_exp(high) if estimate_finite else np.nan,
                "P value": p if reliable else np.nan,
                "Raw P before reliability masking": p,
                "Reliably estimable": reliable,
                "Contrast flags": "None" if not flags else "; ".join(flags),
                "OR direction": "OR > 1 indicates higher odds of reaching the binary FIBSER burden threshold",
            }
        )
        trace_rows.append(
            {
                "Table": table_id,
                "Outcome": outcome_label,
                "Domain": domain,
                "Raw columns": f"{target_var}__occ_raw + {target_var}__tol_raw",
                "Target analysis variable": f"tol__{target_var}",
                "Design parameters": "; ".join(f"{k}*{v:g}" for k, v in weights.items()),
                "Result parameters found": bool(all(k in result.params.index for k in weights)),
                "Variance formula": "c' V_robust c; D-vs-T = Var(D)+Var(T)-2Cov(D,T)",
                "OR/CI/P transformation": "exp(c'beta); exp(c'beta +/- 1.959964*SE); two-sided Wald P",
                "Trace status": "PASS" if reliable else "REVIEW REQUIRED",
            }
        )

    params = pd.DataFrame(
        {
            "Table": table_id,
            "Outcome": outcome_label,
            "Domain": domain,
            "Parameter": result.params.index,
            "Coefficient": result.params.to_numpy(float),
            "Robust SE": result.bse.to_numpy(float),
            "Raw P": result.pvalues.to_numpy(float),
        }
    )
    vif.insert(0, "Domain", domain)
    vif.insert(0, "Outcome", outcome_label)
    vif.insert(0, "Table", table_id)

    target_param_mask = params["Parameter"].str.contains("target_tolerability", regex=False)
    target_max_se = float(params.loc[target_param_mask, "Robust SE"].max())
    any_sparse = any(r["Sparse cell <5"] for r in table_rows)
    any_zero = any(r["Zero cell"] for r in table_rows)
    finite_params = bool(np.isfinite(params[["Coefficient", "Robust SE", "Raw P"]].to_numpy(float)).all())
    stable = bool(
        getattr(result, "converged", False)
        and design["Full rank"]
        and cov_valid
        and finite_params
        and not separation["complete_separation_detected"]
        and not any_zero
        and target_max_se <= 2
        and len(fit_warnings) == 0
    )
    flags = []
    if not getattr(result, "converged", False): flags.append("did not converge")
    if not design["Full rank"]: flags.append("rank deficient")
    if not cov_valid: flags.append("invalid robust covariance")
    if not finite_params: flags.append("non-finite parameter output")
    if separation["complete_separation_detected"]: flags.append("complete separation")
    if any_zero: flags.append("zero target-tolerability-by-outcome cell")
    if any_sparse: flags.append("sparse target-tolerability-by-outcome cell <5")
    if target_max_se > 2: flags.append("target robust SE >2")
    if fit_warnings: flags.append("fit warning")
    summary = {
        "Table": table_id,
        "Outcome": outcome_label,
        "Domain": domain,
        "Status": "PASS" if stable else "REVIEW REQUIRED",
        "Converged": bool(getattr(result, "converged", False)),
        "Participants": int(d["participant_ID"].nunique()),
        "Person-visits": int(len(d)),
        "Excluded visits": int(len(analysis) - len(d)),
        "Events": int(d[outcome].sum()),
        "Non-events": int((1 - d[outcome]).sum()),
        "Events per fitted parameter": float(d[outcome].sum() / xdf.shape[1]),
        "Working correlation": float(np.asarray(result.cov_struct.dep_params).reshape(-1)[0]),
        "Iterations": int(len(result.fit_history.get("params", []))),
        "Robust covariance finite": cov_finite,
        "Robust covariance symmetric": cov_symmetric,
        "Robust covariance minimum eigenvalue": min_eig,
        "Robust covariance valid": cov_valid,
        "Complete separation detected": separation["complete_separation_detected"],
        "Separation LP status": separation["lp_status"],
        "Separation LP message": separation["lp_message"],
        **design,
        "Maximum target robust SE": target_max_se,
        "Sparse overall 2x3 cell <5": any_sparse,
        "Zero overall 2x3 cell": any_zero,
        "Fit warnings": "None" if not fit_warnings else " | ".join(fit_warnings),
        "Diagnostic flags": "None" if not flags else "; ".join(flags),
        "Formula": formula,
    }
    return {
        "contrasts": contrast_rows,
        "summary": summary,
        "table": table_rows,
        "time": time_rows,
        "params": params,
        "vif": vif,
        "trace": trace_rows,
        "cooccurrence": cooccurrence_rows(d, outcome_label, domain),
    }


def compare_secondary(results: pd.DataFrame) -> pd.DataFrame:
    if not SECONDARY.exists():
        raise FileNotFoundError(f"Comparison-only Secondary result is missing: {SECONDARY}")
    sec = pd.read_csv(SECONDARY)
    sec = sec.rename(columns={
        "Domain": "Secondary domain",
        "Component": "Secondary component",
        "Contrast": "Contrast",
        "OR": "Secondary OR",
        "CI_low": "Secondary CI low",
        "CI_high": "Secondary CI high",
        "raw_Wald_P": "Secondary raw P",
        "FDR_P": "Secondary FDR P",
        "Model_diagnostic_status": "Secondary model status",
    })
    sec["PRISE domain"] = sec["Secondary domain"].replace({"Cardiac": "Heart"})
    keep = [
        "PRISE domain", "Contrast", "Secondary component", "Secondary OR", "Secondary CI low",
        "Secondary CI high", "Secondary raw P", "Secondary FDR P", "Secondary model status",
    ]
    # Each key occurs twice on the left (S6/S7) and twice on the right
    # (zero-inflated/ordinal components), intentionally yielding 96 rows.
    merged = results.merge(sec[keep], on=["PRISE domain", "Contrast"], how="left", validate="many_to_many")
    if len(merged) != 96 or merged["Secondary component"].isna().any():
        raise RuntimeError("STOP: S6/S7 to Secondary comparison mapping failed")
    merged["Binary direction"] = np.where(merged["Adjusted OR"] > 1, "higher", np.where(merged["Adjusted OR"] < 1, "lower", "null"))
    # Extra-zero OR <1 and ordinal OR >1 both correspond to higher burden.
    merged["Secondary burden direction"] = np.where(
        merged["Secondary component"].eq("Zero-inflated"),
        np.where(merged["Secondary OR"] < 1, "higher", np.where(merged["Secondary OR"] > 1, "lower", "null")),
        np.where(merged["Secondary OR"] > 1, "higher", np.where(merged["Secondary OR"] < 1, "lower", "null")),
    )
    merged["Direction agreement"] = merged["Binary direction"].eq(merged["Secondary burden direction"])
    merged["Binary raw significant"] = merged["P value"].lt(0.05)
    merged["Binary FDR24 significant"] = merged["FDR-adjusted P value"].lt(0.05)
    merged["Secondary raw significant"] = merged["Secondary raw P"].lt(0.05)
    merged["Secondary FDR significant"] = merged["Secondary FDR P"].lt(0.05)
    merged["Significance agreement (raw)"] = merged["Binary raw significant"].eq(merged["Secondary raw significant"])
    merged["Parameter note"] = "Binary threshold ORs and zero-inflated/ordinal component ORs estimate different parameters; magnitude is not directly comparable."
    return merged


def write_report(results: pd.DataFrame, models: pd.DataFrame, coding: pd.DataFrame, source_hash: str, comparison: pd.DataFrame):
    lines = [
        "# Supplementary Tables S6 and S7: Binary Logistic GEE tolerability sensitivity analyses",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: `experiment-agent`",
        "- Origin Mode: `validate`",
        f"- Origin Date: `{datetime.now().date().isoformat()}`",
        "- Verification Status: `VERIFIED` (raw-data reconstruction and reproducible model rerun completed)",
        "",
        "## Source and frozen-sample gate",
        "",
        f"- Freeze ID: `{FREEZE_ID}`",
        f"- Raw Excel SHA-256: `{source_hash}`",
        "- Frozen sample reproduced: 1,468 participants and 4,799 person-visits; Week 2/8/12/24 = 1,381/1,259/1,146/1,013.",
        "- FIBSER 0/1/2/≥3 = 3,464/584/512/239; S6/S7 full-freeze events = 1,335/239.",
        "- Occurrence mapping: 72/72 PASS. Tolerability mapping: 64/64 PASS.",
        "- Each model excludes only visits with unclassified target-domain tolerability after the frozen complete-case gate.",
        "",
        "## Model",
        "",
        "Sixteen domain-specific binary logistic GEE models used a binomial family, logit link, participant clusters, exchangeable working correlation, and robust sandwich covariance. Each model includes target-domain tolerability (None reference; Tolerable; Distressing), binary occurrence of the other eight PRISE domains, categorical time (Week 2 reference), age, sex, education, first-episode status, baseline HAMD-17, baseline SDS, baseline antidepressant class (SNRI reference), and concurrent HAMD-17. No interactions were fitted.",
        "",
        "All three contrasts were derived from the same fitted model. Distressing vs Tolerable used the complete robust covariance matrix: Var(D)+Var(T)-2Cov(D,T). BH-FDR was applied to 24 prespecified contrasts separately for S6 and S7. A joint 48-test BH result is diagnostic only.",
        "",
        "## Model stability",
        "",
    ]
    for _, r in models.iterrows():
        lines.append(f"- {r['Table']} {r['Domain']}: {r['Status']} — {r['Diagnostic flags']}.")
    lines += ["", "## Statistical findings", ""]
    for table_id in ["S6", "S7"]:
        sub = results.loc[results["Table"].eq(table_id)]
        raw = sub.loc[sub["P value"].lt(0.05), ["PRISE domain", "Contrast"]]
        fdr = sub.loc[sub["FDR-adjusted P value"].lt(0.05), ["PRISE domain", "Contrast"]]
        joint = sub.loc[sub["FDR-adjusted P value (48-test diagnostic)"].lt(0.05), ["PRISE domain", "Contrast"]]
        fmt = lambda x: "; ".join(f"{a} — {b}" for a, b in x.itertuples(index=False, name=None)) or "None"
        lines.append(f"- {table_id} raw P<0.05: {fmt(raw)}.")
        lines.append(f"- {table_id} FDR24 P<0.05: {fmt(fdr)}.")
        lines.append(f"- {table_id} joint-FDR48 P<0.05: {fmt(joint)}.")
    lines += ["", "## Secondary comparison", ""]
    for table_id in ["S6", "S7"]:
        sub = comparison.loc[comparison["Table"].eq(table_id)]
        lines.append(
            f"- {table_id}: direction agreement with the Secondary component-specific estimates in "
            f"{int(sub['Direction agreement'].sum())}/{len(sub)} rows; raw-significance agreement in "
            f"{int(sub['Significance agreement (raw)'].sum())}/{len(sub)} rows."
        )
    lines += [
        "",
        "Binary threshold ORs, zero-inflated extra-zero ORs, and conditional ordinal ORs are not the same estimand. Only burden-direction and qualitative significance/stability are compared.",
        "",
        "## Interpretation and fallacy scan",
        "",
        "These are longitudinal associations, not causal effects. The analysis checks confidence-interval precision, sparse cells, multiple testing, clustering, complete-case selection, attrition, and possible separation. It does not infer a monotonic tolerability gradient. Concurrent symptom severity and reverse causality remain plausible. Person-visit models do not support ecological claims, and no post hoc model or threshold selection was used.",
        "",
        "## Reproducibility",
        "",
        "The accompanying Python script rebuilds both the occurrence and tolerability variables from the original Excel workbook, reruns all 16 models, computes the three contrasts and both BH families, and writes auditable CSV/JSON intermediates used to create the Excel deliverables.",
    ]
    (OUT / "S6_S7_Analysis_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    (
        analysis, freeze_gate, occ_map, tol_map, tol_sheet_keys, coding, sheet_audit,
        conflicts, zmap, sheet_names, source_hash,
    ) = build_analysis()

    collected = {k: [] for k in ["contrasts", "summary", "table", "time", "params", "vif", "trace", "cooccurrence"]}
    for outcome, table_id, label in OUTCOMES:
        for domain in TARGET_DOMAINS:
            fitted = fit_one(analysis, outcome, table_id, label, domain)
            for key in collected:
                value = fitted[key]
                collected[key].extend(value if isinstance(value, list) else [value])

    results = pd.DataFrame(collected["contrasts"])
    results["FDR-adjusted P value"] = np.nan
    for table_id in ["S6", "S7"]:
        idx = results.index[results["Table"].eq(table_id)]
        # Preserve all 24 planned rows; NA comparisons stay NA and are not silently redefined.
        results.loc[idx, "FDR-adjusted P value"] = bh_adjust_with_na(results.loc[idx, "P value"]).to_numpy()
    results["FDR-adjusted P value (48-test diagnostic)"] = bh_adjust_with_na(results["P value"])
    results["OR (95% CI)"] = results.apply(
        lambda r: "NA" if pd.isna(r["Adjusted OR"]) else f"{r['Adjusted OR']:.2f} ({r['CI low']:.2f}, {r['CI high']:.2f})",
        axis=1,
    )
    results["P value display"] = results["P value"].map(lambda x: "NA" if pd.isna(x) else ("<0.001" if x < 0.001 else f"{x:.3f}"))
    results["FDR P display"] = results["FDR-adjusted P value"].map(lambda x: "NA" if pd.isna(x) else ("<0.001" if x < 0.001 else f"{x:.3f}"))
    results["Domain order"] = results["PRISE domain"].map({d: i for i, d in enumerate(TARGET_DOMAINS)})
    results["Contrast order"] = results["Contrast"].map({c: i for i, c in enumerate(CONTRASTS)})
    results = results.sort_values(["Table", "Domain order", "Contrast order"]).reset_index(drop=True)

    models = pd.DataFrame(collected["summary"])
    contingency = pd.DataFrame(collected["table"])
    by_time = pd.DataFrame(collected["time"])
    params = pd.concat(collected["params"], ignore_index=True)
    vif = pd.concat(collected["vif"], ignore_index=True)
    trace = pd.DataFrame(collected["trace"])
    cooccurrence = pd.DataFrame(collected["cooccurrence"])
    comparison = compare_secondary(results)

    exports = {
        "S6_results.csv": results.loc[results["Table"].eq("S6")],
        "S7_results.csv": results.loc[results["Table"].eq("S7")],
        "Model_Summary.csv": models,
        "Target_2x3.csv": contingency,
        "Time_Stratified_Sparsity.csv": by_time,
        "Parameter_Estimates.csv": params,
        "VIF.csv": vif,
        "Contrast_Trace.csv": trace,
        "Domain_Cooccurrence.csv": cooccurrence,
        "Freeze_Gate.csv": freeze_gate,
        "Occurrence_Mapping_72.csv": occ_map,
        "Tolerability_Mapping_64.csv": tol_map,
        "Tolerability_Sheet_Key_Audit.csv": tol_sheet_keys,
        "Tolerability_Coding_Audit.csv": coding,
        "Raw_Sheet_Audit.csv": sheet_audit,
        "Raw_Collapse_Conflicts.csv": conflicts,
        "Standardization_Audit.csv": zmap,
        "S6_S7_vs_Secondary.csv": comparison,
    }
    for name, frame in exports.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    write_report(results, models, coding, source_hash, comparison)
    manifest = {
        "generated": datetime.now().astimezone().isoformat(),
        "freeze_id": FREEZE_ID,
        "source": str(rebuild.SOURCE),
        "source_sha256": source_hash,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "model": "Binary logistic GEE; exchangeable; participant-cluster robust sandwich covariance",
        "formal_fdr": "BH separately within 24 S6 and 24 S7 contrasts",
        "diagnostic_fdr": "BH jointly over 48 S6/S7 contrasts",
        "source_sheets": sheet_names,
        "all_model_status": models[["Table", "Domain", "Status"]].to_dict("records"),
    }
    (OUT / "Run_Manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "participants": int(analysis["participant_ID"].nunique()),
        "visits": int(len(analysis)),
        "models": len(models),
        "stable_models": int(models["Status"].eq("PASS").sum()),
        "reliable_contrasts": int(results["Reliably estimable"].sum()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()




