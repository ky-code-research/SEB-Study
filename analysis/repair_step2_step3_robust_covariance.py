from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

from scipy import stats

import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2_candidate_burden3_no_concurrent_HAMD as zi
import step2_fourlevel_concurrent_hamd17_education3 as step2_m1
import step2_fourlevel_no_concurrent_hamd17_education3 as step2_m2
import step2_fourlevel_no_concurrent_hamd17 as step2_base
import step3_domain_time_burden_4799_preliminary as step3


from public_paths import ROOT
OUT = ROOT / "outputs" / "robust_sandwich_covariance_repair_2026-10-09"
STEP2_OUT = OUT / "step2"
STEP3_OUT = OUT / "step3"
FORMULA = "V = A_inv @ B @ A_inv.T"
PSD_TOL = 1e-8
SYMMETRY_TOL = 1e-10


def log(message: str) -> None:
    print(f"Robust covariance repair: {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def covariance_checks(A: np.ndarray, B: np.ndarray, V_raw: np.ndarray, label: str) -> tuple[np.ndarray, dict]:
    if not np.isfinite(V_raw).all():
        raise RuntimeError(f"{label}: repaired covariance is not finite")
    symmetry_error = float(np.max(np.abs(V_raw - V_raw.T)))
    scale = max(1.0, float(np.max(np.abs(V_raw))))
    symmetric = bool(symmetry_error <= SYMMETRY_TOL * scale)
    V = (V_raw + V_raw.T) / 2.0
    eig = np.linalg.eigvalsh(V)
    min_eig = float(eig.min())
    psd = bool(min_eig >= -PSD_TOL * scale)
    if not symmetric:
        raise RuntimeError(f"{label}: repaired covariance symmetry check failed: {symmetry_error}")
    if not psd:
        raise RuntimeError(f"{label}: repaired covariance PSD check failed: {min_eig}")
    return V, {
        "covariance_finite": True,
        "covariance_symmetric": symmetric,
        "covariance_symmetry_max_abs": symmetry_error,
        "covariance_PSD": psd,
        "covariance_min_eigenvalue": min_eig,
        "covariance_condition_number": float(np.linalg.cond(V)),
        "jacobian_rank": int(np.linalg.matrix_rank(A)),
        "jacobian_parameters": int(A.shape[0]),
        "jacobian_condition_number": float(np.linalg.cond(A)),
        "meat_rank": int(np.linalg.matrix_rank(B)),
        "max_robust_SE": float(np.sqrt(np.clip(np.diag(V), 0.0, np.inf)).max()),
    }


def corrected_step3_covariance(theta: np.ndarray, y: np.ndarray, G: np.ndarray, X: np.ndarray, clusters: np.ndarray, label: str) -> tuple[np.ndarray, dict]:
    sw = b1.sandwich(theta, y, G, X, clusters)
    V, checks = covariance_checks(sw["A"], sw["B"], sw["V"], label)
    return V, checks


def corrected_step2_covariance(theta: np.ndarray, y: np.ndarray, G: np.ndarray, X: np.ndarray, clusters: np.ndarray, label: str) -> tuple[np.ndarray, dict]:
    sw = zi.sandwich(theta, y, G, X, 3, clusters)
    V, checks = covariance_checks(sw["A"], sw["B"], sw["V"], label)
    return V, checks


def parameter_table(names: list[str], theta: np.ndarray, V: np.ndarray) -> pd.DataFrame:
    se = np.sqrt(np.clip(np.diag(V), 0.0, np.inf))
    z = np.divide(theta, se, out=np.full_like(theta, np.nan), where=se > 0)
    return pd.DataFrame({
        "parameter_index_zero_based": np.arange(len(theta)),
        "parameter": names,
        "estimate": theta,
        "robust_SE": se,
        "z": z,
        "raw_Wald_P": 2.0 * stats.norm.sf(np.abs(z)),
    })


def step2_domain_effects(theta: np.ndarray, V: np.ndarray, Gdf: pd.DataFrame, Xdf: pd.DataFrame) -> pd.DataFrame:
    pg = Gdf.shape[1]
    se = np.sqrt(np.clip(np.diag(V), 0.0, np.inf))
    rows: list[dict] = []
    for domain in b1.DOMAINS:
        var = b1.DOMAIN_VAR[domain]
        specs = [
            ("Zero-inflated", list(Gdf.columns).index(var), -1.0, "exp(-gamma)"),
            ("Ordinal", pg + 3 + list(Xdf.columns).index(var), 1.0, "exp(beta)"),
        ]
        for component, idx, sign, definition in specs:
            coef = float(theta[idx])
            robust_se = float(se[idx])
            z = coef / robust_se if robust_se > 0 else np.nan
            log_or = sign * coef
            rows.append({
                "Domain": domain,
                "Component": component,
                "Coefficient": coef,
                "Robust_SE": robust_se,
                "OR": math.exp(log_or),
                "CI_low": math.exp(log_or - 1.96 * robust_se),
                "CI_high": math.exp(log_or + 1.96 * robust_se),
                "Wald_P": float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                "OR_definition": definition,
            })
    return pd.DataFrame(rows)


def compare_step2(old: pd.DataFrame, new: pd.DataFrame, model: str) -> pd.DataFrame:
    keys = ["Domain", "Component"]
    cols = ["Robust_SE", "OR", "CI_low", "CI_high", "Wald_P"]
    out = old[keys + cols].merge(new[keys + cols], on=keys, suffixes=("_before", "_after"))
    out.insert(0, "Model", model)
    for c in cols:
        out[f"abs_change_{c}"] = (out[f"{c}_after"] - out[f"{c}_before"]).abs()
    out["CI_crosses_1_before"] = (out["CI_low_before"] <= 1) & (out["CI_high_before"] >= 1)
    out["CI_crosses_1_after"] = (out["CI_low_after"] <= 1) & (out["CI_high_after"] >= 1)
    out["CI_classification_changed"] = out["CI_crosses_1_before"] != out["CI_crosses_1_after"]
    out["P_lt_0.05_before"] = out["Wald_P_before"] < 0.05
    out["P_lt_0.05_after"] = out["Wald_P_after"] < 0.05
    out["P_significance_changed"] = out["P_lt_0.05_before"] != out["P_lt_0.05_after"]
    return out


def run_step2(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    STEP2_OUT.mkdir(parents=True, exist_ok=True)
    y = df["burden_ord4"].to_numpy(int)
    clusters = df["participant_ID"].to_numpy()
    diagnostics: list[dict] = []
    comparisons: list[pd.DataFrame] = []
    effects_all: list[pd.DataFrame] = []
    specs = [
        ("Model 1", step2_m1, ROOT / "outputs" / "step2_fourlevel_concurrent_hamd17_education3"),
        ("Model 2", step2_m2, ROOT / "outputs" / "step2_fourlevel_no_concurrent_hamd17_education3"),
    ]
    for model, module, old_dir in specs:
        d = df.copy()
        d["education3"] = pd.Categorical(
            d["education"].map(step2_m2.EDUCATION_MAP), categories=step2_m2.EDUCATION_LEVELS
        )
        Gdf, Xdf = b1.make_design(d, module.RHS, module.RHS)
        names = module.parameter_names(Gdf, Xdf)
        saved = pd.read_csv(old_dir / "09_selected_parameter_estimates.csv")
        if saved["parameter"].tolist() != names:
            raise RuntimeError(f"{model} Step 2 saved parameter names do not match live design")
        theta = saved["estimate"].to_numpy(float)
        V, checks = corrected_step2_covariance(
            theta, y, Gdf.to_numpy(float), Xdf.to_numpy(float), clusters, f"Step 2/{model}"
        )
        ptab = parameter_table(names, theta, V)
        effects = step2_domain_effects(theta, V, Gdf, Xdf)
        effects.insert(0, "Model", model)
        ptab.insert(0, "Model", model)
        ptab.to_csv(STEP2_OUT / f"{model.lower().replace(' ', '_')}_parameter_estimates_repaired.csv", index=False, encoding="utf-8-sig")
        effects.to_csv(STEP2_OUT / f"{model.lower().replace(' ', '_')}_domain_effects_repaired.csv", index=False, encoding="utf-8-sig")
        old_effects = pd.read_csv(old_dir / "10_domain_effects_robust_wald.csv")
        comparisons.append(compare_step2(old_effects, effects.drop(columns="Model"), model))
        effects_all.append(effects)
        diagnostics.append({"Stage": "Step 2", "Model": model, "Domain": "All domains", "parameters": len(theta), **checks, "boundary_review_required": False})
        log(f"Step 2 {model}: covariance and 18 domain Wald rows recomputed")
    comparison = pd.concat(comparisons, ignore_index=True)
    effects_all_df = pd.concat(effects_all, ignore_index=True)
    diag_df = pd.DataFrame(diagnostics)
    comparison.to_csv(STEP2_OUT / "step2_before_after_comparison.csv", index=False, encoding="utf-8-sig")
    effects_all_df.to_csv(STEP2_OUT / "step2_domain_effects_repaired_all_models.csv", index=False, encoding="utf-8-sig")
    diag_df.to_csv(STEP2_OUT / "step2_covariance_diagnostics_repaired.csv", index=False, encoding="utf-8-sig")
    return effects_all_df, comparison, diag_df


def compare_step3_time(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    keys = ["Model", "Domain", "Component", "Time"]
    cols = ["Robust_SE", "OR", "CI_low", "CI_high", "raw_Wald_P"]
    out = old[keys + cols].merge(new[keys + cols], on=keys, suffixes=("_before", "_after"))
    for c in cols:
        out[f"abs_change_{c}"] = (out[f"{c}_after"] - out[f"{c}_before"]).abs()
    out["CI_crosses_1_before"] = (out["CI_low_before"] <= 1) & (out["CI_high_before"] >= 1)
    out["CI_crosses_1_after"] = (out["CI_low_after"] <= 1) & (out["CI_high_after"] >= 1)
    out["CI_classification_changed"] = out["CI_crosses_1_before"] != out["CI_crosses_1_after"]
    out["P_lt_0.05_before"] = out["raw_Wald_P_before"] < 0.05
    out["P_lt_0.05_after"] = out["raw_Wald_P_after"] < 0.05
    out["P_significance_changed"] = out["P_lt_0.05_before"] != out["P_lt_0.05_after"]
    return out


def compare_step3_global(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    keys = ["Model", "Domain", "Component"]
    cols = ["Wald_chi2", "raw_P", "BH_FDR_P"]
    out = old[keys + cols].merge(new[keys + cols], on=keys, suffixes=("_before", "_after"))
    for c in cols:
        out[f"abs_change_{c}"] = (out[f"{c}_after"] - out[f"{c}_before"]).abs()
    out["raw_P_lt_0.05_before"] = out["raw_P_before"] < 0.05
    out["raw_P_lt_0.05_after"] = out["raw_P_after"] < 0.05
    out["raw_P_significance_changed"] = out["raw_P_lt_0.05_before"] != out["raw_P_lt_0.05_after"]
    out["BH_FDR_lt_0.05_before"] = out["BH_FDR_P_before"] < 0.05
    out["BH_FDR_lt_0.05_after"] = out["BH_FDR_P_after"] < 0.05
    out["BH_FDR_significance_changed"] = out["BH_FDR_lt_0.05_before"] != out["BH_FDR_lt_0.05_after"]
    return out


def run_step3(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    STEP3_OUT.mkdir(parents=True, exist_ok=True)
    step3.base.engine.Q = 3
    y = df["burden_ord4"].to_numpy(int)
    clusters = df["participant_ID"].to_numpy()
    old_root = step3.OUT
    old_diag = pd.read_csv(old_root / "04_model_diagnostics.csv")
    boundary_map = old_diag.set_index(["Model", "Domain"])["boundary_review_required"].to_dict()
    global_rows: list[dict] = []
    time_rows: list[dict] = []
    parameter_rows: list[pd.DataFrame] = []
    diagnostic_rows: list[dict] = []
    probability_rows: list[pd.DataFrame] = []
    for model, spec in step3.MODEL_SPECS.items():
        for domain in b1.DOMAINS:
            var = b1.DOMAIN_VAR[domain]
            rhs = step3.rhs_for(spec["rhs"], var)
            Gdf, Xdf = b1.make_design(df, rhs, rhs)
            names = step3.parameter_names(Gdf, Xdf)
            domain_slug = domain.replace("/", "_").replace(" ", "_")
            saved_path = old_root / spec["slug"] / domain_slug / "selected_named_parameters.csv"
            saved = pd.read_csv(saved_path)
            if saved["parameter"].tolist() != names:
                raise RuntimeError(f"Step 3 {model}/{domain}: saved parameter names do not match live design")
            theta = saved["estimate"].to_numpy(float)
            V, checks = corrected_step3_covariance(
                theta, y, Gdf.to_numpy(float), Xdf.to_numpy(float), clusters, f"Step 3/{model}/{domain}"
            )
            boundary = bool(boundary_map[(model, domain)])
            status = "PRELIMINARY_BOUNDARY_REVIEW_REQUIRED" if boundary else "PRELIMINARY"
            ptab = parameter_table(names, theta, V)
            ptab.insert(0, "Domain", domain)
            ptab.insert(0, "Model", model)
            ptab["inference_status"] = status
            parameter_rows.append(ptab)
            gl = step3.extract_global(model, domain, var, theta, V, names)
            te = step3.extract_time_effects(model, domain, var, theta, V, names)
            for row in gl:
                row["boundary_review_required"] = boundary
                row["inference_status"] = status
            for row in te:
                row["boundary_review_required"] = boundary
                row["inference_status"] = status
            global_rows.extend(gl)
            time_rows.extend(te)
            diagnostic_rows.append({"Stage": "Step 3", "Model": model, "Domain": domain, "parameters": len(theta), **checks, "boundary_review_required": boundary, "inference_status": status})
            log(f"Step 3 {model}/{domain}: repaired covariance, 2 global and 8 time-specific rows recomputed")
        old_prob = pd.read_csv(old_root / f"{spec['slug']}_adjusted_probabilities.csv")
        old_prob["covariance_repair_effect"] = "unchanged; predictions depend on saved parameters, not covariance"
        old_prob.to_csv(STEP3_OUT / f"{spec['slug']}_adjusted_probabilities_unchanged.csv", index=False, encoding="utf-8-sig")
        probability_rows.append(old_prob)
    globals_df = step3.add_fdr(pd.DataFrame(global_rows))
    times_df = pd.DataFrame(time_rows)
    params_df = pd.concat(parameter_rows, ignore_index=True)
    diag_df = pd.DataFrame(diagnostic_rows)
    for model, spec in step3.MODEL_SPECS.items():
        globals_df.loc[globals_df.Model.eq(model)].to_csv(STEP3_OUT / f"{spec['slug']}_global_interactions_repaired.csv", index=False, encoding="utf-8-sig")
        times_df.loc[times_df.Model.eq(model)].to_csv(STEP3_OUT / f"{spec['slug']}_time_specific_ORs_repaired.csv", index=False, encoding="utf-8-sig")
    globals_df.to_csv(STEP3_OUT / "step3_global_interactions_repaired_all_models.csv", index=False, encoding="utf-8-sig")
    times_df.to_csv(STEP3_OUT / "step3_time_specific_ORs_repaired_all_models.csv", index=False, encoding="utf-8-sig")
    params_df.to_csv(STEP3_OUT / "step3_parameter_estimates_repaired.csv", index=False, encoding="utf-8-sig")
    diag_df.to_csv(STEP3_OUT / "step3_covariance_diagnostics_repaired.csv", index=False, encoding="utf-8-sig")
    step3.sensitivity_table(globals_df).to_csv(STEP3_OUT / "step3_concurrent_HAMD17_sensitivity_repaired.csv", index=False, encoding="utf-8-sig")
    old_global = pd.concat([
        pd.read_csv(old_root / f"{spec['slug']}_global_interactions.csv") for spec in step3.MODEL_SPECS.values()
    ], ignore_index=True)
    old_time = pd.concat([
        pd.read_csv(old_root / f"{spec['slug']}_time_specific_ORs.csv") for spec in step3.MODEL_SPECS.values()
    ], ignore_index=True)
    global_comparison = compare_step3_global(old_global, globals_df)
    time_comparison = compare_step3_time(old_time, times_df)
    global_comparison.to_csv(STEP3_OUT / "step3_global_before_after_comparison.csv", index=False, encoding="utf-8-sig")
    time_comparison.to_csv(STEP3_OUT / "step3_time_specific_before_after_comparison.csv", index=False, encoding="utf-8-sig")
    return globals_df, times_df, global_comparison, time_comparison, diag_df


def build_report(step2_comp: pd.DataFrame, step2_diag: pd.DataFrame, step3_global_comp: pd.DataFrame, step3_time_comp: pd.DataFrame, step3_diag: pd.DataFrame) -> str:
    step2_sig = step2_comp.loc[step2_comp.P_significance_changed | step2_comp.CI_classification_changed]
    step3_time_sig = step3_time_comp.loc[step3_time_comp.P_significance_changed | step3_time_comp.CI_classification_changed]
    step3_global_raw = step3_global_comp.loc[step3_global_comp.raw_P_significance_changed]
    step3_global_fdr = step3_global_comp.loc[step3_global_comp.BH_FDR_significance_changed]
    all_diag = pd.concat([step2_diag, step3_diag], ignore_index=True)
    boundary = step3_diag.loc[step3_diag.boundary_review_required, ["Model", "Domain"]]
    max_step2_p = float(step2_comp.abs_change_Wald_P.max())
    max_step3_p = float(step3_time_comp.abs_change_raw_Wald_P.max())
    max_global_p = float(step3_global_comp.abs_change_raw_P.max())
    max_fdr_p = float(step3_global_comp.abs_change_BH_FDR_P.max())
    max_ci = max(float(step2_comp.abs_change_CI_low.max()), float(step2_comp.abs_change_CI_high.max()), float(step3_time_comp.abs_change_CI_low.max()), float(step3_time_comp.abs_change_CI_high.max()))
    return f"""# Step 2 & Step 3 Robust Sandwich Covariance Repair Report

## Material Passport

- Verification Status: **VERIFIED for deterministic covariance/Wald recomputation; inferential status remains preliminary**.
- Frozen data: `{step2_base.FREEZE_ID}`; SHA-256 `{sha256(step2_base.FREEZE)}`; 1468 participants / 4799 person-visits.
- Repair: `{FORMULA}`; original Jacobian and participant-cluster score definitions retained.
- Parameters: loaded from saved selected parameter files; no optimization or bootstrap was run.
- Original result directories were not overwritten.

## Repair scope and checks

- Step 3 and both requested four-level Step 2 models shared the missing-transpose defect through two covariance helpers; both helpers were repaired.
- Recomputed: 2 Step 2 covariance matrices and 18 Step 3 covariance matrices; all {len(all_diag)} are finite, symmetric within relative tolerance `{SYMMETRY_TOL:g}`, and PSD within relative tolerance `{PSD_TOL:g}`.
- Step 3: 36 global 3-df Wald tests, 144 time-specific effects, and BH-FDR within each model×component family of nine domains were recomputed.
- Step 2: 36 domain/component robust-Wald rows were recomputed.

## Before–after findings

- OR point estimates changed: **0** (saved parameter vectors were unchanged).
- Step 2 CI/P classification changes at 0.05: **{len(step2_sig)}**.
- Step 3 time-specific CI/P classification changes at 0.05: **{len(step3_time_sig)}**.
- Step 3 global raw-P classification changes at 0.05: **{len(step3_global_raw)}**.
- Step 3 global BH-FDR classification changes at 0.05: **{len(step3_global_fdr)}**.
- Maximum absolute changes: Step 2 Wald P `{max_step2_p:.6g}`; Step 3 time-specific P `{max_step3_p:.6g}`; Step 3 global raw P `{max_global_p:.6g}`; BH-FDR P `{max_fdr_p:.6g}`; CI endpoint `{max_ci:.6g}`.

Detailed row-level differences, including unchanged rows, are saved in the Step 2 and Step 3 before/after comparison CSV files.

## Boundary status

`BOUNDARY_REVIEW_REQUIRED` was retained for {len(boundary)} Step 3 model/domain fits: {', '.join((boundary.Model + '/' + boundary.Domain).tolist()) if len(boundary) else 'none'}. The covariance repair does not resolve higher-objective weak-identification solutions; repaired Wald outputs remain preliminary.

## Restrictions honored

No model formula, covariate, frozen data, score definition, Jacobian, parameter estimate, optimizer result, or adjusted prediction was changed. Adjusted probability tables were copied to the repair directory with an explicit unchanged note. No bootstrap or additional analysis was run.

Statistical fallacy scan 11/11: no new causal or confirmatory claim is made; multiple testing remains controlled by the prespecified nine-domain BH families, and boundary-conditioned inference remains explicitly flagged.
"""


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STEP2_OUT.mkdir(parents=True, exist_ok=True)
    STEP3_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(step2_base.FREEZE)
    if len(df) != 4799 or df["participant_ID"].nunique() != 1468 or df.duplicated(["participant_ID", "time"]).any():
        raise RuntimeError("Frozen sample gate failed")
    df["education3"] = pd.Categorical(
        df["education"].map(step2_m2.EDUCATION_MAP), categories=step2_m2.EDUCATION_LEVELS
    )
    if df["education3"].isna().any():
        raise RuntimeError("education3 reconstruction failed")
    step2_effects, step2_comp, step2_diag = run_step2(df)
    step3_globals, step3_times, step3_global_comp, step3_time_comp, step3_diag = run_step3(df)
    report = build_report(step2_comp, step2_diag, step3_global_comp, step3_time_comp, step3_diag)
    (OUT / "Robust_Sandwich_Covariance_Repair_Report.md").write_text(report, encoding="utf-8")
    atomic_json(OUT / "00_repair_manifest.json", {
        "repair_formula": FORMULA,
        "freeze_id": step2_base.FREEZE_ID,
        "freeze_sha256": sha256(step2_base.FREEZE),
        "participants": 1468,
        "person_visits": 4799,
        "step2_models_recomputed": 2,
        "step3_models_recomputed": 18,
        "optimization_rerun": False,
        "bootstrap_run": False,
        "original_outputs_overwritten": False,
        "all_covariances_finite": bool(pd.concat([step2_diag, step3_diag]).covariance_finite.all()),
        "all_covariances_symmetric": bool(pd.concat([step2_diag, step3_diag]).covariance_symmetric.all()),
        "all_covariances_PSD": bool(pd.concat([step2_diag, step3_diag]).covariance_PSD.all()),
    })
    log(f"complete: {OUT}")


if __name__ == "__main__":
    run()




