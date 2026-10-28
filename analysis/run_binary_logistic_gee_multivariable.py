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
DEFAULT_OUT = ROOT / "outputs" / "binary_logistic_GEE_multivariable_S4_S5_2026-10-18"
OUT = Path(os.environ.get("MULTIVARIABLE_GEE_OUTPUT_DIR", str(DEFAULT_OUT)))
if OUT.exists():
    raise FileExistsError(f"Output directory already exists; choose a new directory: {OUT}")
os.environ["BINARY_GEE_OUTPUT_DIR"] = str(OUT)

# This module contains the verified raw-Excel reconstruction functions used for
# the preceding single-domain analysis. Importing it creates only the new OUT
# directory specified above; no saved cleaned data or model output is read.
import run_binary_logistic_gee_sensitivity as rebuild  # noqa: E402


OUT.mkdir(parents=True, exist_ok=True)
FREEZE_ID = rebuild.FREEZE_ID
DOMAINS = rebuild.DOMAINS
DOMAIN_VARS = [var for _, var in DOMAINS]
PREVIOUS_SINGLE = (
    ROOT / "outputs" / "binary_logistic_GEE_sensitivity_S4_S5_2026-10-18" / "Model_Results.csv"
)


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    if not np.isfinite(pvalues).all():
        raise RuntimeError("Non-finite P value encountered before BH-FDR adjustment")
    return multipletests(pvalues, alpha=0.05, method="fdr_bh")[1]


def full_formula(outcome: str) -> str:
    domain_terms = " + ".join(DOMAIN_VARS)
    return (
        f"{outcome} ~ {domain_terms} + C(time, Treatment(reference='Week2')) + age_z + "
        "C(sex, Treatment(reference='Female')) + "
        "C(education3, Treatment(reference='College or above')) + "
        "C(first_episode, Treatment(reference='FirstEpisode')) + baseline_HAMD17_z + baseline_SDS_z + "
        "C(baseline_med_class, Treatment(reference='SNRI')) + concurrent_HAMD17_z"
    )


def complete_separation_lp(x: np.ndarray, y: np.ndarray) -> dict:
    """Sufficient diagnostic for complete linear separation of the full matrix."""
    signed = np.where(y == 1, 1.0, -1.0)
    a_ub = -(signed[:, None] * x)
    b_ub = -np.ones(len(y))
    res = optimize.linprog(
        c=np.zeros(x.shape[1]),
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * x.shape[1],
        method="highs",
    )
    return {
        "complete_separation_detected": bool(res.success),
        "lp_status": int(res.status),
        "lp_message": str(res.message),
    }


def design_diagnostics(xdf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = np.asarray(xdf, dtype=float)
    rank = int(np.linalg.matrix_rank(x))
    nonconstant = [c for c in xdf.columns if c != "Intercept" and float(xdf[c].std(ddof=0)) > 0]
    scaled = np.asarray(xdf[nonconstant], dtype=float)
    scaled = (scaled - scaled.mean(axis=0)) / scaled.std(axis=0, ddof=0)
    scaled_condition = float(np.linalg.cond(scaled))
    vif_rows = []
    for idx, col in enumerate(xdf.columns):
        if col == "Intercept":
            continue
        vif_rows.append({"Parameter": col, "VIF": float(variance_inflation_factor(x, idx))})
    vif = pd.DataFrame(vif_rows)
    summary = {
        "design_rows": int(xdf.shape[0]),
        "design_columns": int(xdf.shape[1]),
        "design_rank": rank,
        "full_rank": bool(rank == xdf.shape[1]),
        "scaled_condition_number": scaled_condition,
        "max_vif": float(vif["VIF"].max()),
        "max_domain_vif": float(vif.loc[vif["Parameter"].isin(DOMAIN_VARS), "VIF"].max()),
    }
    return vif, summary


def domain_cooccurrence(analysis: pd.DataFrame):
    domain = analysis[DOMAIN_VARS].astype(int)
    corr = domain.corr(method="pearson")
    present = domain.sum().rename_axis("Analysis variable").reset_index(name="Present visits")
    name_map = {var: name for name, var in DOMAINS}
    present["Domain"] = present["Analysis variable"].map(name_map)
    rows = []
    for i, (name_i, var_i) in enumerate(DOMAINS):
        for j in range(i + 1, len(DOMAINS)):
            name_j, var_j = DOMAINS[j]
            tab = pd.crosstab(domain[var_i], domain[var_j]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            rows.append(
                {
                    "Domain 1": name_i,
                    "Domain 2": name_j,
                    "Neither": int(tab.loc[0, 0]),
                    "Domain 2 only": int(tab.loc[0, 1]),
                    "Domain 1 only": int(tab.loc[1, 0]),
                    "Both present": int(tab.loc[1, 1]),
                    "Phi correlation": float(corr.loc[var_i, var_j]),
                }
            )
    corr_named = corr.rename(index=name_map, columns=name_map).rename_axis("Domain").reset_index()
    return present[["Domain", "Analysis variable", "Present visits"]], pd.DataFrame(rows), corr_named


def fit_multivariable(analysis: pd.DataFrame, outcome: str, outcome_label: str):
    formula = full_formula(outcome)
    ydf, xdf = patsy.dmatrices(formula, analysis, return_type="dataframe")
    y = np.asarray(ydf).reshape(-1).astype(int)
    x = np.asarray(xdf, dtype=float)
    if len(xdf) != len(analysis) or ydf.isna().any().any() or xdf.isna().any().any():
        raise RuntimeError(f"STOP: model matrix row/missing mismatch for {outcome_label}")
    if any(v not in xdf.columns for v in DOMAIN_VARS):
        raise RuntimeError(f"STOP: at least one domain coefficient is absent from {outcome_label} design matrix")
    vif, design = design_diagnostics(xdf)
    if not design["full_rank"]:
        raise RuntimeError(f"STOP: rank-deficient design matrix for {outcome_label}")
    separation = complete_separation_lp(x, y)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        model = sm.GEE.from_formula(
            formula,
            groups="participant_ID",
            data=analysis,
            family=sm.families.Binomial(),
            cov_struct=Exchangeable(),
        )
        result = model.fit(maxiter=300, ctol=1e-9, cov_type="robust")
        fit_warnings = [str(w.message) for w in ws]
    cov = np.asarray(result.cov_params(), dtype=float)
    cov_finite = bool(np.isfinite(cov).all())
    cov_symmetric = bool(np.allclose(cov, cov.T, rtol=1e-8, atol=1e-10))
    min_eig = float(np.linalg.eigvalsh((cov + cov.T) / 2).min()) if cov_finite else np.nan
    covariance_valid = bool(cov_finite and cov_symmetric and min_eig >= -1e-8)
    domain_rows = []
    trace_rows = []
    for final_name, var in DOMAINS:
        beta = float(result.params[var])
        se = float(result.bse[var])
        p = float(result.pvalues[var])
        low = beta - stats.norm.ppf(0.975) * se
        high = beta + stats.norm.ppf(0.975) * se
        tab = pd.crosstab(analysis[var].astype(int), analysis[outcome].astype(int)).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        counts = [int(tab.loc[0, 0]), int(tab.loc[0, 1]), int(tab.loc[1, 0]), int(tab.loc[1, 1])]
        zero_cell = any(v == 0 for v in counts)
        extreme = bool(abs(beta) > math.log(20))
        large_se = bool(se > 2)
        finite = bool(np.isfinite([beta, se, p, low, high]).all())
        flags = []
        if zero_cell:
            flags.append("zero domain-by-outcome cell")
        if extreme:
            flags.append("extreme adjusted OR (<0.05 or >20)")
        if large_se:
            flags.append("large robust SE (>2)")
        if not finite:
            flags.append("non-finite domain estimate")
        domain_rows.append(
            {
                "Outcome": outcome_label,
                "Domain": final_name,
                "Analysis variable": var,
                "Coefficient": beta,
                "Robust SE": se,
                "Adjusted OR": float(np.exp(beta)),
                "CI low": float(np.exp(low)),
                "CI high": float(np.exp(high)),
                "Raw P": p,
                "Domain absent/outcome 0": counts[0],
                "Domain absent/outcome 1": counts[1],
                "Domain present/outcome 0": counts[2],
                "Domain present/outcome 1": counts[3],
                "Zero cell": zero_cell,
                "Extreme OR": extreme,
                "Large SE": large_se,
                "Domain flags": "None" if not flags else "; ".join(flags),
            }
        )
        trace_rows.append(
            {
                "Outcome": outcome_label,
                "Final domain": final_name,
                "Raw-to-analysis mapping": f"72-item verified mapping -> {var} (0 absent, 1 present)",
                "Design matrix parameter": var,
                "Result parameter": str(result.params.index[result.params.index.get_loc(var)]),
                "OR transformation": "exp(coefficient)",
                "CI transformation": "exp(coefficient +/- 1.959964 x robust SE)",
                "P-value source": "two-sided robust Wald test",
                "Trace status": "PASS",
            }
        )
    domain_df = pd.DataFrame(domain_rows)
    domain_df["FDR-adjusted P (9-test family)"] = bh_adjust(domain_df["Raw P"].to_numpy())
    parameter_df = pd.DataFrame(
        {
            "Outcome": outcome_label,
            "Parameter": result.params.index,
            "Coefficient": result.params.to_numpy(float),
            "Robust SE": result.bse.to_numpy(float),
            "Raw P": result.pvalues.to_numpy(float),
        }
    )
    summary = {
        "Outcome": outcome_label,
        "Converged": bool(getattr(result, "converged", False)),
        "Participants": int(analysis["participant_ID"].nunique()),
        "Person-visits": int(len(analysis)),
        "Events": int(analysis[outcome].sum()),
        "Non-events": int((1 - analysis[outcome]).sum()),
        "Events per fitted parameter": float(analysis[outcome].sum() / xdf.shape[1]),
        "Working correlation": float(np.asarray(result.cov_struct.dep_params).reshape(-1)[0]),
        "Iterations": int(len(result.fit_history.get("params", []))),
        "Robust covariance finite": cov_finite,
        "Robust covariance symmetric": cov_symmetric,
        "Robust covariance minimum eigenvalue": min_eig,
        "Robust covariance valid": covariance_valid,
        "Complete separation detected": separation["complete_separation_detected"],
        "Separation LP status": separation["lp_status"],
        "Separation LP message": separation["lp_message"],
        "Design rows": design["design_rows"],
        "Design columns": design["design_columns"],
        "Design rank": design["design_rank"],
        "Full rank": design["full_rank"],
        "Scaled condition number": design["scaled_condition_number"],
        "Maximum VIF": design["max_vif"],
        "Maximum domain VIF": design["max_domain_vif"],
        "Maximum absolute domain coefficient": float(domain_df["Coefficient"].abs().max()),
        "Maximum domain robust SE": float(domain_df["Robust SE"].max()),
        "Minimum events among domain-present visits": int(domain_df["Domain present/outcome 1"].min()),
        "Fit warnings": "None" if not fit_warnings else " | ".join(fit_warnings),
        "Formula": formula,
    }
    summary["Stable"] = bool(
        summary["Converged"]
        and summary["Robust covariance valid"]
        and summary["Full rank"]
        and not summary["Complete separation detected"]
        and not domain_df["Zero cell"].any()
        and not domain_df["Extreme OR"].any()
        and not domain_df["Large SE"].any()
        and summary["Fit warnings"] == "None"
    )
    vif.insert(0, "Outcome", outcome_label)
    return domain_df, pd.DataFrame([summary]), parameter_df, vif, pd.DataFrame(trace_rows)


def build_comparison(multivariable: pd.DataFrame) -> pd.DataFrame:
    if not PREVIOUS_SINGLE.exists():
        raise FileNotFoundError(f"Comparison-only single-domain result is missing: {PREVIOUS_SINGLE}")
    single = pd.read_csv(PREVIOUS_SINGLE)
    if len(single) != 18:
        raise RuntimeError("Comparison-only single-domain file does not contain 18 rows")
    if set(single["Participants"].astype(int)) != {1468} or set(single["Person-visits"].astype(int)) != {4799}:
        raise RuntimeError("Comparison-only single-domain sample counts do not match the current freeze")
    keep = single[["Outcome", "Domain", "Adjusted OR", "CI low", "CI high", "P value"]].rename(
        columns={
            "Adjusted OR": "Single-domain OR",
            "CI low": "Single-domain CI low",
            "CI high": "Single-domain CI high",
            "P value": "Single-domain raw P",
        }
    )
    multi = multivariable[
        ["Outcome", "Domain", "Adjusted OR", "CI low", "CI high", "Raw P", "FDR-adjusted P (9-test family)"]
    ].rename(
        columns={
            "Adjusted OR": "Multivariable OR",
            "CI low": "Multivariable CI low",
            "CI high": "Multivariable CI high",
            "Raw P": "Multivariable raw P",
            "FDR-adjusted P (9-test family)": "Multivariable FDR P (9 tests)",
        }
    )
    out = keep.merge(multi, on=["Outcome", "Domain"], how="outer", validate="one_to_one", indicator=True)
    if not out["_merge"].eq("both").all():
        raise RuntimeError("Single-domain and multivariable result rows did not match one-to-one")
    out = out.drop(columns="_merge")
    out["OR ratio multivariable/single"] = out["Multivariable OR"] / out["Single-domain OR"]
    out["Absolute log-OR change"] = (np.log(out["Multivariable OR"]) - np.log(out["Single-domain OR"])).abs()
    out["Direction changed"] = (out["Single-domain OR"] - 1) * (out["Multivariable OR"] - 1) < 0
    out["Raw significance changed"] = (out["Single-domain raw P"] < 0.05) != (out["Multivariable raw P"] < 0.05)
    out["Change interpretation"] = np.select(
        [
            out["Direction changed"],
            out["Raw significance changed"] & (out["Multivariable raw P"] >= 0.05),
            out["Raw significance changed"] & (out["Multivariable raw P"] < 0.05),
            out["OR ratio multivariable/single"] <= 0.8,
            out["OR ratio multivariable/single"] >= 1.2,
        ],
        [
            "Direction changed",
            "Lost raw significance after mutual domain adjustment",
            "Gained raw significance after mutual domain adjustment",
            "OR attenuated by at least 20%",
            "OR increased by at least 20%",
        ],
        default="Direction/significance unchanged; OR change <20%",
    )
    return out


def write_summary(
    analysis: pd.DataFrame,
    mapping: pd.DataFrame,
    gate: pd.DataFrame,
    model_summary: pd.DataFrame,
    results: pd.DataFrame,
    comparison: pd.DataFrame,
    co_pairs: pd.DataFrame,
    source_hash: str,
):
    lines = [
        "Multivariable binary logistic GEE update for Supplementary Tables S4 and S5",
        "=" * 78,
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        f"Freeze ID: {FREEZE_ID}",
        f"Raw source: {rebuild.SOURCE}",
        f"Raw source SHA-256: {source_hash}",
        "Analysis input: rebuilt from the raw Excel workbook using the verified reconstruction functions. Previous single-domain results were read only after fitting, for the requested comparison workbook.",
        "",
        "Data gates",
        "----------",
    ]
    lines.extend(f"{r['check']}: {r['observed']} vs {r['expected']} [{r['status']}]" for _, r in gate.iterrows())
    lines += [
        f"PRISE mapping rows: {len(mapping)}; PASS rows: {int(mapping['status'].eq('PASS').sum())}.",
        "",
        "Model specification",
        "-------------------",
        "Two models only: S4 any burden (1-6 vs 0) and S5 moderate or greater burden (3-6 vs 0-2).",
        "Each model includes all nine PRISE domains simultaneously plus time, age, sex, education, first-episode status, baseline HAMD-17, baseline SDS, baseline antidepressant class, and concurrent HAMD-17.",
        "Binomial family; logit link; participant clustering; exchangeable working correlation; robust sandwich covariance. No domain-time or domain-domain interactions.",
        "References: domain absent, Week 2, female, college or above, first episode, and SNRI.",
        "BH-FDR: nine domain P values adjusted separately within S4 and S5. An additional 18-test joint BH result is diagnostic only.",
        "",
        "Model diagnostics",
        "-----------------",
    ]
    for _, r in model_summary.iterrows():
        lines.append(
            f"{r['Outcome']}: converged={r['Converged']}; stable={r['Stable']}; events={r['Events']}; "
            f"working correlation={r['Working correlation']:.6g}; rank={r['Design rank']}/{r['Design columns']}; "
            f"scaled condition number={r['Scaled condition number']:.6g}; max domain VIF={r['Maximum domain VIF']:.6g}; "
            f"complete separation={r['Complete separation detected']}; max domain SE={r['Maximum domain robust SE']:.6g}; "
            f"min domain-present events={r['Minimum events among domain-present visits']}; covariance valid={r['Robust covariance valid']}."
        )
    max_pair = co_pairs.loc[co_pairs["Phi correlation"].abs().idxmax()]
    lines += [
        f"Largest absolute pairwise domain phi correlation: {max_pair['Domain 1']} with {max_pair['Domain 2']} = {max_pair['Phi correlation']:.6g}.",
        "",
        "Multiplicity and comparison summary",
        "-----------------------------------",
    ]
    for outcome in results["Outcome"].drop_duplicates():
        sub = results.loc[results["Outcome"].eq(outcome)]
        raw = ", ".join(sub.loc[sub["Raw P"] < 0.05, "Domain"])
        fdr9 = ", ".join(sub.loc[sub["FDR-adjusted P (9-test family)"] < 0.05, "Domain"])
        fdr18 = ", ".join(sub.loc[sub["FDR-adjusted P (18-test family)"] < 0.05, "Domain"])
        lines.append(f"{outcome}: raw P<0.05 = {raw or 'None'}; FDR9<0.05 = {fdr9 or 'None'}; FDR18<0.05 = {fdr18 or 'None'}.")
    changed = comparison.loc[
        comparison["Direction changed"] | comparison["Raw significance changed"] | (comparison["OR ratio multivariable/single"] <= 0.8) | (comparison["OR ratio multivariable/single"] >= 1.2)
    ]
    lines.append(f"Comparison rows with direction/significance change or >=20% OR change: {len(changed)} of 18.")
    lines += [
        "",
        "Parameter trace",
        "---------------",
        "For each domain: verified original Excel occurrence column -> participant/time collapse -> binary occurrence variable -> full-rank design-matrix column with the same analysis-variable name -> named GEE coefficient -> exp(beta), robust Wald CI and raw P -> BH-adjusted P -> formal table row.",
        "",
        "Software",
        "--------",
        f"Python {platform.python_version()}; pandas {pd.__version__}; NumPy {np.__version__}; SciPy {scipy.__version__}; statsmodels {statsmodels.__version__}; patsy {patsy.__version__}",
    ]
    (OUT / "Analysis_Notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    analysis, long, mapping, sheet_audit, conflicts, zmap, sheet_names, source_hash = rebuild.rebuild_analysis()
    gate = rebuild.stop_gate(analysis)
    if len(mapping) != 72 or not mapping["status"].eq("PASS").all():
        raise RuntimeError("STOP: 72-item PRISE mapping gate failed")
    if not conflicts.empty:
        raise RuntimeError("STOP: discordant nonmissing values found across paired follow-up sheets")
    present, co_pairs, co_matrix = domain_cooccurrence(analysis)

    result_frames, summary_frames, parameter_frames, vif_frames, trace_frames = [], [], [], [], []
    for outcome, label in [
        ("any_burden", "Any burden (FIBSER burden 1-6)"),
        ("moderate_burden", "Moderate or greater burden (FIBSER burden 3-6)"),
    ]:
        domain_df, summary_df, parameter_df, vif_df, trace_df = fit_multivariable(analysis, outcome, label)
        result_frames.append(domain_df)
        summary_frames.append(summary_df)
        parameter_frames.append(parameter_df)
        vif_frames.append(vif_df)
        trace_frames.append(trace_df)
        print(f"FIT {label} | stable={summary_df.iloc[0]['Stable']} | converged={summary_df.iloc[0]['Converged']}", flush=True)

    results = pd.concat(result_frames, ignore_index=True)
    results["FDR-adjusted P (18-test family)"] = bh_adjust(results["Raw P"].to_numpy())
    model_summary = pd.concat(summary_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    vif = pd.concat(vif_frames, ignore_index=True)
    trace = pd.concat(trace_frames, ignore_index=True)
    comparison = build_comparison(results)

    results.to_csv(OUT / "Multivariable_Domain_Results.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(OUT / "Model_Summary.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(OUT / "All_Model_Parameters.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(OUT / "VIF_Diagnostics.csv", index=False, encoding="utf-8-sig")
    trace.to_csv(OUT / "Parameter_Trace.csv", index=False, encoding="utf-8-sig")
    present.to_csv(OUT / "Domain_Prevalence.csv", index=False, encoding="utf-8-sig")
    co_pairs.to_csv(OUT / "Domain_Cooccurrence_Pairs.csv", index=False, encoding="utf-8-sig")
    co_matrix.to_csv(OUT / "Domain_Phi_Correlation_Matrix.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "Single_vs_Multivariable_Comparison.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(OUT / "Freeze_Gate.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(OUT / "PRISE_Mapping_Audit.csv", index=False, encoding="utf-8-sig")
    sheet_audit.to_csv(OUT / "Source_Sheet_Audit.csv", index=False, encoding="utf-8-sig")
    zmap.to_csv(OUT / "Standardization_Constants.csv", index=False, encoding="utf-8-sig")
    write_summary(analysis, mapping, gate, model_summary, results, comparison, co_pairs, source_hash)

    payload = {
        "metadata": {
            "freeze_id": FREEZE_ID,
            "source": str(rebuild.SOURCE),
            "source_sha256": source_hash,
            "participants": int(analysis["participant_ID"].nunique()),
            "person_visits": int(len(analysis)),
            "week_counts": {k: int(v) for k, v in analysis["time"].astype(str).value_counts().to_dict().items()},
            "any_events": int(analysis["any_burden"].sum()),
            "moderate_events": int(analysis["moderate_burden"].sum()),
        },
        "results": results.to_dict(orient="records"),
        "model_summary": model_summary.to_dict(orient="records"),
        "parameters": parameters.to_dict(orient="records"),
        "vif": vif.to_dict(orient="records"),
        "trace": trace.to_dict(orient="records"),
        "domain_prevalence": present.to_dict(orient="records"),
        "cooccurrence_pairs": co_pairs.to_dict(orient="records"),
        "cooccurrence_matrix": co_matrix.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
        "freeze_gate": gate.to_dict(orient="records"),
        "mapping": mapping.to_dict(orient="records"),
    }
    (OUT / "workbook_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OUTPUT_DIR={OUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "RUN_FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise




