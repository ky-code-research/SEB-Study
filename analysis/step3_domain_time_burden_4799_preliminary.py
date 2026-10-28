from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Compatibility for the workstation SciPy build.
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests

import model_comparison_gate_fibser as legacy
import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2_fourlevel_no_concurrent_hamd17 as base
import step2_fourlevel_no_concurrent_hamd17_education3 as model2_base
import step2_fourlevel_concurrent_hamd17_education3 as model1_base


from public_paths import ROOT
OUT = ROOT / "outputs" / "step3_domain_time_burden_4799_preliminary"
FREEZE_ID = base.FREEZE_ID
EXPECTED_PARTICIPANTS = 1468
EXPECTED_VISITS = 4799
EXPECTED_COUNTS = {0: 3464, 1: 584, 2: 512, 3: 239}
TIMES = ["Week2", "Week8", "Week12", "Week24"]
NONREF_TIMES = ["Week8", "Week12", "Week24"]
TIME_TERM = "C(time, Treatment(reference='Week2'))"
Q = 3
SCORE_TOL = 5e-2
SEED = 20261009
DOMAIN_VARS = [b1.DOMAIN_VAR[d] for d in b1.DOMAINS]
MAPPING_AUDIT = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "PRISE_Mapping_Verification.csv"

MODEL_SPECS = {
    "Model 1": {
        "slug": "model1_with_concurrent_hamd17",
        "rhs": model1_base.RHS,
        "expected_parameters": 52,
        "expected_g": 25,
        "expected_x": 24,
        "step2_parameters": ROOT / "outputs" / "step2_fourlevel_concurrent_hamd17_education3" / "09_selected_parameter_estimates.csv",
        "concurrent_hamd17": True,
    },
    "Model 2": {
        "slug": "model2_without_concurrent_hamd17",
        "rhs": model2_base.RHS,
        "expected_parameters": 50,
        "expected_g": 24,
        "expected_x": 23,
        "step2_parameters": ROOT / "outputs" / "step2_fourlevel_no_concurrent_hamd17_education3" / "09_selected_parameter_estimates.csv",
        "concurrent_hamd17": False,
    },
}


def log(message: str) -> None:
    print(f"Step3 preliminary: {message}", flush=True)


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def parameter_names(gdf: pd.DataFrame, xdf: pd.DataFrame) -> list[str]:
    return (
        [f"mixing_gamma:{c}" for c in gdf.columns]
        + ["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"]
        + [f"ordinal_beta:{c}" for c in xdf.columns]
    )


def rhs_for(base_rhs: str, domain_var: str) -> str:
    return f"{base_rhs} + {domain_var}:{TIME_TERM}"


def interaction_column(domain_var: str, week: str) -> str:
    return f"{domain_var}:{TIME_TERM}[T.{week}]"


def time_column(week: str) -> str:
    return f"{TIME_TERM}[T.{week}]"


def vector(names: list[str], name: str) -> np.ndarray:
    if name not in names:
        raise RuntimeError(f"PARAMETER ALIGNMENT FAIL: {name!r} absent")
    out = np.zeros(len(names), dtype=float)
    out[names.index(name)] = 1.0
    return out


def load_data() -> pd.DataFrame:
    version = json.loads(base.VERSION_RECORD.read_text(encoding="utf-8"))
    freeze_hash = base.sha256(base.FREEZE)
    df = pd.read_csv(base.FREEZE)
    df["education3"] = pd.Categorical(
        df["education"].map(model2_base.EDUCATION_MAP),
        categories=model2_base.EDUCATION_LEVELS,
    )
    checks: list[dict] = []

    def check(name, observed, expected, passed) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "pass": bool(passed)})

    check("freeze ID", FREEZE_ID, "PRISE_REPAIRED_NATURAL_CC_1468P_4799V_2026-10-08", FREEZE_ID == "PRISE_REPAIRED_NATURAL_CC_1468P_4799V_2026-10-08")
    check("freeze SHA256", freeze_hash, base.EXPECTED_FREEZE_SHA256, freeze_hash.lower() == base.EXPECTED_FREEZE_SHA256)
    check("source SHA256", version.get("source_sha256"), base.EXPECTED_SOURCE_SHA256, version.get("source_sha256", "").lower() == base.EXPECTED_SOURCE_SHA256)
    check("participants", df["participant_ID"].nunique(), EXPECTED_PARTICIPANTS, df["participant_ID"].nunique() == EXPECTED_PARTICIPANTS)
    check("person-visits", len(df), EXPECTED_VISITS, len(df) == EXPECTED_VISITS)
    duplicates = int(df.duplicated(["participant_ID", "time"]).sum())
    check("unique participant_ID + time", duplicates, 0, duplicates == 0)
    counts = df["burden_ord4"].value_counts().sort_index().astype(int).to_dict()
    check("burden counts", json.dumps(counts, sort_keys=True), json.dumps(EXPECTED_COUNTS, sort_keys=True), counts == EXPECTED_COUNTS)
    check("time levels", json.dumps(sorted(df["time"].unique())), json.dumps(sorted(TIMES)), set(df["time"].unique()) == set(TIMES))
    check("education3 complete", int(df["education3"].isna().sum()), 0, not df["education3"].isna().any())
    changing = int((df.groupby("participant_ID", observed=True)["education3"].nunique() > 1).sum())
    check("education3 constant within participant", changing, 0, changing == 0)

    mapping = pd.read_csv(MAPPING_AUDIT)
    mapping_domains = set(mapping.loc[mapping["status"].astype(str).str.upper().eq("PASS"), "final_model_variable"].dropna().astype(str))
    check("nine repaired occurrence variables in mapping audit", json.dumps(sorted(mapping_domains & set(DOMAIN_VARS))), json.dumps(sorted(DOMAIN_VARS)), set(DOMAIN_VARS).issubset(mapping_domains))
    for column in DOMAIN_VARS:
        observed = set(pd.to_numeric(df[column], errors="coerce").dropna().unique())
        valid = observed.issubset({0, 1, 0.0, 1.0}) and not df[column].isna().any()
        check(f"binary complete occurrence variable: {column}", json.dumps(sorted(observed)), "[0, 1]", valid)

    audit = pd.DataFrame(checks)
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / "01_frozen_sample_and_domain_checks.csv", index=False, encoding="utf-8-sig")
    if not audit["pass"].all():
        raise RuntimeError(f"STOP: frozen-sample/domain gate failed: {audit.loc[~audit['pass'], 'check'].tolist()}")
    df["burden_ord4"] = df["burden_ord4"].astype(int)
    return df


def preflight_designs(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    maps: list[pd.DataFrame] = []
    for model, spec in MODEL_SPECS.items():
        step2 = pd.read_csv(spec["step2_parameters"])
        if step2["parameter"].duplicated().any():
            raise RuntimeError(f"STOP: duplicate Step 2 parameter names in {model}")
        for domain in b1.DOMAINS:
            var = b1.DOMAIN_VAR[domain]
            rhs = rhs_for(spec["rhs"], var)
            gdf, xdf = b1.make_design(df, rhs, rhs)
            names = parameter_names(gdf, xdf)
            interactions_g = [c for c in gdf.columns if ":" in c]
            interactions_x = [c for c in xdf.columns if ":" in c]
            expected_interactions = [interaction_column(var, w) for w in NONREF_TIMES]
            forbidden = [c for c in [*gdf.columns, *xdf.columns] if "site" in c.lower() or "province" in c.lower()]
            edu_g = [c for c in gdf.columns if "education3" in c]
            edu_x = [c for c in xdf.columns if "education3" in c]
            hamd_g = [c for c in gdf.columns if c == "concurrent_HAMD17_z"]
            hamd_x = [c for c in xdf.columns if c == "concurrent_HAMD17_z"]
            checks = {
                "g_columns_ok": gdf.shape[1] == spec["expected_g"],
                "x_columns_ok": xdf.shape[1] == spec["expected_x"],
                "parameters_ok": len(names) == spec["expected_parameters"],
                "unique_names": len(names) == len(set(names)),
                "mixing_interactions_exact": set(interactions_g) == set(expected_interactions),
                "ordinal_interactions_exact": set(interactions_x) == set(expected_interactions),
                "nine_domains_both": all(v in gdf.columns and v in xdf.columns for v in DOMAIN_VARS),
                "education_two_contrasts_both": len(edu_g) == 2 and len(edu_x) == 2,
                "college_reference": not any("[T.College or above]" in c for c in edu_g + edu_x),
                "concurrent_hamd_rule": (hamd_g == ["concurrent_HAMD17_z"] and hamd_x == ["concurrent_HAMD17_z"]) if spec["concurrent_hamd17"] else (not hamd_g and not hamd_x),
                "no_site_or_province": not forbidden,
            }
            row = {
                "Model": model,
                "Domain": domain,
                "domain_variable": var,
                "mixing_columns": gdf.shape[1],
                "thresholds": Q,
                "ordinal_columns": xdf.shape[1],
                "total_parameters": len(names),
                "all_design_checks_pass": all(checks.values()),
                **checks,
                "mixing_columns_json": json.dumps(list(gdf.columns), ensure_ascii=False),
                "ordinal_columns_json": json.dumps(list(xdf.columns), ensure_ascii=False),
            }
            rows.append(row)
            maps.append(pd.DataFrame({
                "Model": model,
                "Domain": domain,
                "parameter_index_zero_based": range(len(names)),
                "parameter": names,
            }))
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "02_design_matrix_checks.csv", index=False, encoding="utf-8-sig")
    pd.concat(maps, ignore_index=True).to_csv(OUT / "03_parameter_name_maps.csv", index=False, encoding="utf-8-sig")
    if not audit["all_design_checks_pass"].all():
        failed = audit.loc[~audit["all_design_checks_pass"], ["Model", "Domain"]].to_dict(orient="records")
        raise RuntimeError(f"STOP: design/parameter alignment gate failed: {failed}")
    return audit


def mapped_step2_start(spec: dict, names: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    source = pd.read_csv(spec["step2_parameters"])
    lookup = source.set_index("parameter")["estimate"].astype(float).to_dict()
    theta = np.array([lookup.get(name, 0.0) for name in names], dtype=float)
    mapping = pd.DataFrame({
        "target_parameter": names,
        "source_parameter_exact_match": [name if name in lookup else "" for name in names],
        "source_present": [name in lookup for name in names],
        "initial_value": theta,
        "rule": ["exact_name_match" if name in lookup else "new_interaction_zero" for name in names],
    })
    expected_new = {
        f"mixing_gamma:{interaction_column(var, week)}"
        for var in DOMAIN_VARS for week in NONREF_TIMES
    } | {
        f"ordinal_beta:{interaction_column(var, week)}"
        for var in DOMAIN_VARS for week in NONREF_TIMES
    }
    actual_new = set(mapping.loc[~mapping["source_present"], "target_parameter"])
    if len(actual_new) != 6 or not actual_new.issubset(expected_new):
        raise RuntimeError(f"STOP: exact-name warm-start mapping failed; new parameters={sorted(actual_new)}")
    return theta, mapping


def candidate_row(fit_number: int, start: dict, res, assessment: dict, jac: dict) -> dict:
    strict = bool(
        res.success
        and assessment["strict_score_converged"]
        and assessment["parameter_valid"]
        and jac["jacobian_finite"]
        and jac["jacobian_full_rank"]
    )
    return {
        "fit_number": fit_number,
        "start": start["start"],
        "start_method": start["method"],
        "program_success": bool(res.success),
        "message": str(res.message),
        "iterations": int(res.nit),
        **assessment,
        **jac,
        "strict_valid": strict,
        "selected": False,
    }


def load_saved_candidates(checkpoint_dir: Path, names: list[str]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    candidates: list[dict] = []
    for diag_path in sorted(checkpoint_dir.glob("start_*_diagnostics.csv")):
        number = int(diag_path.stem.split("_")[1])
        param_path = checkpoint_dir / f"start_{number:02d}_named_parameters.csv"
        if not param_path.exists():
            continue
        row = pd.read_csv(diag_path).iloc[0].to_dict()
        saved = pd.read_csv(param_path).sort_values("parameter_index_zero_based")
        if len(saved) != len(names) or saved["parameter"].tolist() != names:
            raise RuntimeError(f"STOP: checkpoint parameter mismatch at {param_path}")
        theta = saved["estimate"].to_numpy(float)
        rows.append(row)
        candidates.append({"theta": theta, "row": row})
    return rows, candidates


def fit_model(model: str, domain: str, y: np.ndarray, gdf: pd.DataFrame, xdf: pd.DataFrame, clusters: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = MODEL_SPECS[model]
    g, x = gdf.to_numpy(float), xdf.to_numpy(float)
    names = parameter_names(gdf, xdf)
    warm, mapping = mapped_step2_start(spec, names)
    generated = legacy.build_zi_starts(y, g, x, Q + 1)
    if len(generated) != 10:
        raise RuntimeError(f"Expected 10 Step 2 starts, got {len(generated)}")
    starts = [{"start": "mapped_step2_by_exact_name", "theta": warm, "method": "exact_parameter_name_map"}]
    starts.extend({"start": s["start"], "theta": s["theta"], "method": s["method"]} for s in generated)

    model_dir = OUT / spec["slug"] / domain.replace("/", "_").replace(" ", "_")
    checkpoint_dir = model_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(model_dir / "initial_parameter_mapping.csv", index=False, encoding="utf-8-sig")
    rows, candidates = load_saved_candidates(checkpoint_dir, names)
    completed = {int(r["fit_number"]) for r in rows}

    for fit_number, start in enumerate(starts, 1):
        if fit_number in completed:
            log(f"{model} / {domain}: start {fit_number}/11 checkpoint loaded")
            continue
        res = optimize.minimize(
            lambda th: b1.neg_obj_grad(th, y, g, x),
            start["theta"],
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 4000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 100, "maxcor": 80},
        )
        assessment = base.engine.basic_assessment(res.x, y, g, x)
        _, jac = base.engine.jacobian_assessment(res.x, y, g, x)
        row = candidate_row(fit_number, start, res, assessment, jac)
        pd.DataFrame([row]).to_csv(checkpoint_dir / f"start_{fit_number:02d}_diagnostics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({
            "fit_number": fit_number,
            "parameter_index_zero_based": range(len(names)),
            "parameter": names,
            "estimate": res.x,
        }).to_csv(checkpoint_dir / f"start_{fit_number:02d}_named_parameters.csv", index=False, encoding="utf-8-sig")
        rows.append(row)
        candidates.append({"theta": res.x, "row": row})
        log(f"{model} / {domain}: start {fit_number}/11 success={res.success} strict={row['strict_valid']} objective={assessment['objective']:.6f} score={assessment['max_abs_score']:.4g}")

    strict = [c for c in candidates if bool(c["row"]["strict_valid"])]
    if not strict:
        raise RuntimeError(f"STOP: {model}/{domain} has no strict-valid multi-start solution")
    best = max(strict, key=lambda c: (float(c["row"]["objective"]), -float(c["row"]["max_abs_score"])))

    refined_number = 12
    refined_diag = checkpoint_dir / "start_12_diagnostics.csv"
    refined_param = checkpoint_dir / "start_12_named_parameters.csv"
    if refined_diag.exists() and refined_param.exists():
        saved = pd.read_csv(refined_param).sort_values("parameter_index_zero_based")
        if len(saved) != len(names) or saved["parameter"].tolist() != names:
            raise RuntimeError(f"STOP: refined checkpoint parameter mismatch for {model}/{domain}")
        refined_row = pd.read_csv(refined_diag).iloc[0].to_dict()
        refined_theta = saved["estimate"].to_numpy(float)
    else:
        start = {"start": "refined_from_best_strict_raw", "method": str(best["row"]["start"])}
        res = optimize.minimize(
            lambda th: b1.neg_obj_grad(th, y, g, x),
            best["theta"],
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 8000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 120, "maxcor": 100},
        )
        assessment = base.engine.basic_assessment(res.x, y, g, x)
        _, jac = base.engine.jacobian_assessment(res.x, y, g, x)
        refined_row = candidate_row(refined_number, start, res, assessment, jac)
        refined_theta = res.x
        pd.DataFrame([refined_row]).to_csv(refined_diag, index=False, encoding="utf-8-sig")
        pd.DataFrame({
            "fit_number": refined_number,
            "parameter_index_zero_based": range(len(names)),
            "parameter": names,
            "estimate": refined_theta,
        }).to_csv(refined_param, index=False, encoding="utf-8-sig")
        log(f"{model} / {domain}: refined strict={refined_row['strict_valid']} objective={assessment['objective']:.6f} score={assessment['max_abs_score']:.4g}")

    all_candidates = [*candidates, {"theta": refined_theta, "row": refined_row}]
    strict_all = [c for c in all_candidates if bool(c["row"]["strict_valid"])]
    if not strict_all:
        raise RuntimeError(f"STOP: {model}/{domain} has no strict-valid candidate after refinement")

    # Apply the already-documented Step 2 boundary rule before Wald inference:
    # rank/conditioning prefilter, followed by local-curvature and covariance checks.
    # Higher-objective weak-identification candidates remain recorded and trigger
    # BOUNDARY_REVIEW_REQUIRED; they are never silently used for Wald inference.
    selection_rows: list[dict] = []
    selected = None
    ordered = sorted(strict_all, key=lambda c: (float(c["row"]["objective"]), -float(c["row"]["max_abs_score"])), reverse=True)
    for candidate in ordered:
        row = candidate["row"]
        preeligible = bool(
            row.get("jacobian_finite", False)
            and row.get("jacobian_full_rank", False)
            and float(row.get("jacobian_condition_number", np.inf)) <= 1e12
            and float(row.get("jacobian_min_singular_value", 0.0)) > 1e-8
        )
        audit = {
            "Model": model,
            "Domain": domain,
            "fit_number": int(row["fit_number"]),
            "start": row["start"],
            "objective": float(row["objective"]),
            "max_abs_score": float(row["max_abs_score"]),
            "rank_condition_prefilter": preeligible,
            "curvature_positive_definite": False,
            "covariance_finite": False,
            "covariance_PSD": False,
            "eligible_for_preliminary_Wald": False,
            "classification": "WEAK_IDENTIFICATION",
        }
        if preeligible:
            sw_candidate = b1.sandwich(candidate["theta"], y, g, x, clusters)
            avec = np.linalg.eigvalsh((sw_candidate["A"] + sw_candidate["A"].T) / 2.0)
            vvec = np.linalg.eigvalsh((sw_candidate["V"] + sw_candidate["V"].T) / 2.0)
            curvature = bool(np.isfinite(avec).all() and avec.min() > 1e-8)
            cov_finite = bool(sw_candidate["finite"])
            cov_psd = bool(np.isfinite(vvec).all() and vvec.min() >= -1e-8)
            eligible = bool(curvature and cov_finite and cov_psd)
            audit.update({
                "curvature_positive_definite": curvature,
                "information_min_eigenvalue": float(avec.min()),
                "covariance_finite": cov_finite,
                "covariance_PSD": cov_psd,
                "covariance_min_eigenvalue": float(vvec.min()),
                "max_robust_SE": float(sw_candidate["se"].max()),
                "eligible_for_preliminary_Wald": eligible,
                "classification": "REGULAR_OR_FINITE_BOUNDARY" if eligible else "WEAK_IDENTIFICATION",
            })
            if eligible and selected is None:
                selected = candidate
        selection_rows.append(audit)
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError(f"STOP: {model}/{domain} has no locally identified finite candidate eligible for preliminary Wald")
    selected_number = int(selected["row"]["fit_number"])
    selected_objective = float(selected["row"]["objective"])
    highest_strict_objective = max(float(c["row"]["objective"]) for c in strict_all)
    review_required = bool(highest_strict_objective > selected_objective + 1e-6)
    selection_audit = pd.DataFrame(selection_rows)
    selection_audit["selected_for_preliminary_Wald"] = selection_audit["fit_number"].eq(selected_number)
    selection_audit["boundary_review_required"] = review_required
    selection_audit["highest_strict_objective"] = highest_strict_objective
    selection_audit["selected_regular_objective"] = selected_objective
    selection_audit.to_csv(model_dir / "boundary_selection_audit.csv", index=False, encoding="utf-8-sig")
    all_rows = pd.DataFrame([c["row"] for c in all_candidates]).sort_values("fit_number")
    all_rows["selected"] = all_rows["fit_number"].astype(int).eq(selected_number)
    all_rows["boundary_review_required"] = review_required
    all_rows.to_csv(model_dir / "all_start_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "parameter_index_zero_based": range(len(names)),
        "parameter": names,
        "estimate": selected["theta"],
    }).to_csv(model_dir / "selected_named_parameters.csv", index=False, encoding="utf-8-sig")
    return selected["theta"], all_rows, mapping, selection_audit


def lincomb(theta: np.ndarray, covariance: np.ndarray, contrast: np.ndarray) -> tuple[float, float, float]:
    estimate = float(contrast @ theta)
    variance = float(contrast @ covariance @ contrast)
    se = math.sqrt(max(variance, 0.0))
    p = float(2 * stats.norm.sf(abs(estimate / se))) if se > 0 else np.nan
    return estimate, se, p


def safe_exp(value: float) -> float:
    """Exponentiate without hiding boundary-driven infinite Wald limits."""
    if value > math.log(np.finfo(float).max):
        return math.inf
    if value < math.log(np.nextafter(0.0, 1.0)):
        return 0.0
    return math.exp(value)


def wald3(theta: np.ndarray, covariance: np.ndarray, contrasts: list[np.ndarray]) -> tuple[float, int, float]:
    lmat = np.vstack(contrasts)
    estimates = lmat @ theta
    cov = lmat @ covariance @ lmat.T
    rank = int(np.linalg.matrix_rank(cov, tol=1e-10))
    if rank != 3:
        raise RuntimeError(f"STOP: global interaction covariance rank is {rank}, expected 3")
    statistic = float(estimates.T @ np.linalg.pinv(cov) @ estimates)
    return statistic, rank, float(stats.chi2.sf(statistic, rank))


def extract_global(model: str, domain: str, var: str, theta: np.ndarray, covariance: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for component, prefix in [("Zero-Inflated", "mixing_gamma"), ("Ordinal", "ordinal_beta")]:
        contrasts = [vector(names, f"{prefix}:{interaction_column(var, week)}") for week in NONREF_TIMES]
        chi2, df, p = wald3(theta, covariance, contrasts)
        rows.append({"Model": model, "Domain": domain, "Component": component, "Wald_chi2": chi2, "df": df, "raw_P": p})
    return rows


def extract_time_effects(model: str, domain: str, var: str, theta: np.ndarray, covariance: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for component, prefix, sign, definition in [
        ("Zero-Inflated", "mixing_gamma", -1.0, "exp(-gamma); odds of latent additional-zero state"),
        ("Ordinal", "ordinal_beta", 1.0, "exp(beta); OR>1 indicates higher ordinary ordinal burden"),
    ]:
        for week in TIMES:
            contrast = vector(names, f"{prefix}:{var}")
            if week != "Week2":
                contrast += vector(names, f"{prefix}:{interaction_column(var, week)}")
            estimate, se, p = lincomb(theta, covariance, contrast)
            log_or = sign * estimate
            rows.append({
                "Model": model,
                "Domain": domain,
                "Component": component,
                "Time": week,
                "linear_predictor_coefficient": estimate,
                "Robust_SE": se,
                "OR": safe_exp(log_or),
                "CI_low": safe_exp(log_or - 1.96 * se),
                "CI_high": safe_exp(log_or + 1.96 * se),
                "raw_Wald_P": p,
                "OR_definition": definition,
            })
    return rows


def set_fixed_time(matrix: np.ndarray, columns: list[str], week: str) -> None:
    for nonref in NONREF_TIMES:
        col = time_column(nonref)
        if col not in columns:
            raise RuntimeError(f"STOP: time column missing during prediction: {col}")
        matrix[:, columns.index(col)] = float(week == nonref)


def adjusted_probabilities(model: str, domain: str, var: str, theta: np.ndarray, gdf: pd.DataFrame, xdf: pd.DataFrame) -> list[dict]:
    rows = []
    gcols, xcols = list(gdf.columns), list(xdf.columns)
    for week in TIMES:
        matrices = []
        for present in [0, 1]:
            g = gdf.to_numpy(float).copy()
            x = xdf.to_numpy(float).copy()
            set_fixed_time(g, gcols, week)
            set_fixed_time(x, xcols, week)
            g[:, gcols.index(var)] = float(present)
            x[:, xcols.index(var)] = float(present)
            for nonref in NONREF_TIMES:
                ic = interaction_column(var, nonref)
                g[:, gcols.index(ic)] = float(present == 1 and week == nonref)
                x[:, xcols.index(ic)] = float(present == 1 and week == nonref)
            matrices.append((g, x))
        p0 = base.engine.predict(theta, matrices[0][0], matrices[0][1])[0]
        p1 = base.engine.predict(theta, matrices[1][0], matrices[1][1])[0]
        ge1_0 = p0[:, 1:].sum(axis=1)
        ge1_1 = p1[:, 1:].sum(axis=1)
        rows.append({
            "Model": model,
            "Domain": domain,
            "Time": week,
            "P_burden_ge1_domain_absent": float(ge1_0.mean()),
            "P_burden_ge1_domain_present": float(ge1_1.mean()),
            "Adjusted_probability_difference_present_minus_absent": float((ge1_1 - ge1_0).mean()),
            "standardization": "all 4799 frozen person-visits; time fixed; all other covariates retained at observed values",
            "mean_probability_sum_absent": float(p0.sum(axis=1).mean()),
            "mean_probability_sum_present": float(p1.sum(axis=1).mean()),
        })
    return rows


def covariance_diagnostics(model: str, domain: str, theta: np.ndarray, y: np.ndarray, gdf: pd.DataFrame, xdf: pd.DataFrame, clusters: np.ndarray) -> tuple[np.ndarray, dict, pd.DataFrame]:
    g, x = gdf.to_numpy(float), xdf.to_numpy(float)
    sw = b1.sandwich(theta, y, g, x, clusters)
    covariance = (sw["V"] + sw["V"].T) / 2.0
    eig = np.linalg.eigvalsh(covariance)
    a_eig = np.linalg.eigvalsh((sw["A"] + sw["A"].T) / 2.0)
    assessment = base.engine.basic_assessment(theta, y, g, x)
    _, jac = base.engine.jacobian_assessment(theta, y, g, x)
    names = parameter_names(gdf, xdf)
    params = b1.param_table(theta, sw["se"], list(gdf.columns), list(xdf.columns))
    if params["parameter"].tolist() != names:
        raise RuntimeError(f"STOP: OR/CI parameter-table alignment failed for {model}/{domain}")
    int_names = {
        f"mixing_gamma:{interaction_column(b1.DOMAIN_VAR[domain], w)}" for w in NONREF_TIMES
    } | {
        f"ordinal_beta:{interaction_column(b1.DOMAIN_VAR[domain], w)}" for w in NONREF_TIMES
    }
    focal = params[params["parameter"].isin(int_names)]
    if len(focal) != 6:
        raise RuntimeError(f"STOP: expected six focal interaction parameters for {model}/{domain}, got {len(focal)}")
    boundary = bool(
        assessment["max_abs_parameter"] >= 50
        or assessment["max_abs_mixing_coefficient"] > 8
        or assessment["min_pi"] <= 1e-8
        or assessment["max_pi"] >= 1 - 1e-8
        or assessment["min_ordinary_category_probability"] <= 1e-12
        or jac["jacobian_condition_number"] > 1e12
        or float(sw["se"].max()) > 5
    )
    row = {
        "Model": model,
        "Domain": domain,
        "parameters": len(theta),
        **assessment,
        **jac,
        "covariance_finite": bool(sw["finite"]),
        "covariance_PSD": bool(sw["V_positive_semidefinite"]),
        "information_positive_definite": bool(np.isfinite(a_eig).all() and a_eig.min() > 1e-8),
        "information_min_eigenvalue": float(a_eig.min()),
        "covariance_min_eigenvalue": float(eig.min()),
        "covariance_condition_number": float(np.linalg.cond(covariance)),
        "max_robust_SE": float(sw["se"].max()),
        "max_focal_interaction_SE": float(focal["robust_SE"].max()),
        "boundary_solution_flag": boundary,
        "selected_solution_usable_preliminary": bool(
            assessment["strict_score_converged"]
            and assessment["parameter_valid"]
            and jac["jacobian_full_rank"]
            and jac["jacobian_finite"]
            and sw["finite"]
            and sw["V_positive_semidefinite"]
            and np.isfinite(a_eig).all()
            and a_eig.min() > 1e-8
        ),
    }
    return covariance, row, params


def add_fdr(global_tests: pd.DataFrame) -> pd.DataFrame:
    result = global_tests.copy()
    result["BH_FDR_P"] = np.nan
    # Nine domains are corrected separately within each model and model component.
    for (_, _), idx in result.groupby(["Model", "Component"], sort=False).groups.items():
        if len(idx) != 9:
            raise RuntimeError("STOP: BH-FDR family does not contain exactly nine domain tests")
        result.loc[idx, "BH_FDR_P"] = multipletests(result.loc[idx, "raw_P"].to_numpy(float), method="fdr_bh")[1]
    result["BH_FDR_significant_0.05"] = result["BH_FDR_P"] < 0.05
    return result


def sensitivity_table(global_tests: pd.DataFrame) -> pd.DataFrame:
    m1 = global_tests[global_tests["Model"].eq("Model 1")].drop(columns="Model")
    m2 = global_tests[global_tests["Model"].eq("Model 2")].drop(columns="Model")
    out = m1.merge(m2, on=["Domain", "Component"], suffixes=("_Model1", "_Model2"))
    out["raw_P_signal_changed"] = (out["raw_P_Model1"] < 0.05) != (out["raw_P_Model2"] < 0.05)
    out["BH_FDR_signal_changed"] = (out["BH_FDR_P_Model1"] < 0.05) != (out["BH_FDR_P_Model2"] < 0.05)
    out["sensitivity_label"] = np.select(
        [out["BH_FDR_signal_changed"], out["raw_P_signal_changed"]],
        ["sensitive after BH-FDR", "nominal-P sensitive only"],
        default="not classification-sensitive",
    )
    return out


def write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)


def make_report(global_tests: pd.DataFrame, diagnostics: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    lines = [
        "# Step 3 Preliminary Robust-Wald Report",
        "",
        "> **PRELIMINARY / EXPLORATORY ONLY.** The custom ES-GEE formal variance-inference issue remains unresolved. These participant-cluster robust Wald results are not confirmatory.",
        "",
        "## Scope and consistency gates",
        "",
        "- Frozen data: `PRISE_REPAIRED_NATURAL_CC_1468P_4799V_2026-10-08`; 1468 participants / 4799 person-visits.",
        "- FIBSER burden counts (0/1/2/>=3): 3464/584/512/239.",
        "- Eighteen separate models: nine focal-domain interaction models under each of two adjustment sets.",
        "- Both model parts contain the same focal domain x categorical-time interaction; Week 2 is the reference.",
        "- Model 1 has 52 parameters per focal-domain model; Model 2 has 50. Every coefficient was mapped/extracted by complete parameter name.",
        "- Education reference: College or above. Site/province were excluded.",
        "- Probability standardization fixes time and domain status for all 4799 frozen rows while retaining all other covariates at their observed values.",
        "",
        "## Numerical diagnostics",
        "",
        f"- Usable selected solutions: {int(diagnostics['selected_solution_usable_preliminary'].sum())}/18.",
        f"- Selected boundary flags: {int(diagnostics['boundary_solution_flag'].sum())}/18.",
        f"- Jacobian full-rank solutions: {int(diagnostics['jacobian_full_rank'].sum())}/18.",
        f"- Finite covariance matrices: {int(diagnostics['covariance_finite'].sum())}/18.",
        "",
        "## Global domain x time interactions",
        "",
    ]
    for model in MODEL_SPECS:
        lines.extend([f"### {model}", ""])
        sub = global_tests[global_tests["Model"].eq(model)]
        for component in ["Zero-Inflated", "Ordinal"]:
            comp = sub[sub["Component"].eq(component)]
            fdr = comp.loc[comp["BH_FDR_P"] < 0.05, "Domain"].tolist()
            nominal = comp.loc[(comp["raw_P"] < 0.05) & (comp["BH_FDR_P"] >= 0.05), "Domain"].tolist()
            lines.append(f"- {component}: BH-FDR signals = {', '.join(fdr) if fdr else 'none'}; raw-P-only signals = {', '.join(nominal) if nominal else 'none'}.")
        lines.append("")
    changed = sensitivity[sensitivity["sensitivity_label"].ne("not classification-sensitive")]
    lines.extend(["## Sensitivity to concurrent HAMD-17 adjustment", ""])
    if changed.empty:
        lines.append("No domain-component global interaction changed its raw-P<0.05 or BH-FDR<0.05 classification between Model 1 and Model 2.")
    else:
        lines.append(changed[["Domain", "Component", "raw_P_Model1", "BH_FDR_P_Model1", "raw_P_Model2", "BH_FDR_P_Model2", "sensitivity_label"]].to_markdown(index=False))
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Time-specific ORs and adjusted joint probabilities describe the fitted interaction patterns. They should be interpreted only after the corresponding 3-df global interaction test, and no result here should be presented as final confirmatory inference.",
        "",
    ])
    return "\n".join(lines)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base.engine.Q = Q
    base.engine.SCORE_TOL = SCORE_TOL
    log("running frozen-sample and occurrence-variable gates")
    df = load_data()
    design_audit = preflight_designs(df)
    log("all 18 design/parameter preflight rows PASS")
    y = df["burden_ord4"].to_numpy(int)
    clusters = df["participant_ID"].to_numpy()

    all_fit_rows: list[pd.DataFrame] = []
    all_diagnostics: list[dict] = []
    all_params: list[pd.DataFrame] = []
    all_selection_rows: list[pd.DataFrame] = []
    global_rows: list[dict] = []
    effect_rows: list[dict] = []
    probability_rows: list[dict] = []

    for model, spec in MODEL_SPECS.items():
        for domain in b1.DOMAINS:
            var = b1.DOMAIN_VAR[domain]
            log(f"fitting {model} / {domain}")
            rhs = rhs_for(spec["rhs"], var)
            gdf, xdf = b1.make_design(df, rhs, rhs)
            names = parameter_names(gdf, xdf)
            if len(names) != spec["expected_parameters"]:
                raise RuntimeError(f"STOP: live design changed after preflight for {model}/{domain}")
            theta, fit_diag, _, selection_audit = fit_model(model, domain, y, gdf, xdf, clusters)
            fit_diag.insert(0, "Domain", domain)
            fit_diag.insert(0, "Model", model)
            all_fit_rows.append(fit_diag)
            all_selection_rows.append(selection_audit)
            covariance, diag, params = covariance_diagnostics(model, domain, theta, y, gdf, xdf, clusters)
            diag["boundary_review_required"] = bool(selection_audit["boundary_review_required"].iloc[0])
            diag["highest_strict_objective"] = float(selection_audit["highest_strict_objective"].iloc[0])
            diag["selected_regular_objective"] = float(selection_audit["selected_regular_objective"].iloc[0])
            diag["objective_gap_weak_minus_selected_regular"] = diag["highest_strict_objective"] - diag["selected_regular_objective"]
            if not diag["selected_solution_usable_preliminary"]:
                raise RuntimeError(f"STOP: selected solution failed Jacobian/covariance gate for {model}/{domain}")
            all_diagnostics.append(diag)
            params.insert(0, "Domain", domain)
            params.insert(0, "Model", model)
            all_params.append(params)
            global_rows.extend(extract_global(model, domain, var, theta, covariance, names))
            effect_rows.extend(extract_time_effects(model, domain, var, theta, covariance, names))
            probability_rows.extend(adjusted_probabilities(model, domain, var, theta, gdf, xdf))
            log(f"completed {model} / {domain}")

    diagnostics = pd.DataFrame(all_diagnostics)
    globals_df = add_fdr(pd.DataFrame(global_rows))
    effects_df = pd.DataFrame(effect_rows)
    probabilities_df = pd.DataFrame(probability_rows)
    sensitivity = sensitivity_table(globals_df)
    fits_df = pd.concat(all_fit_rows, ignore_index=True)
    params_df = pd.concat(all_params, ignore_index=True)
    selection_df = pd.concat(all_selection_rows, ignore_index=True)

    # Final one-to-one output cardinality and probability checks.
    expected_sizes = {
        "global tests": (len(globals_df), 36),
        "time-specific OR rows": (len(effects_df), 144),
        "probability rows": (len(probabilities_df), 72),
        "diagnostic rows": (len(diagnostics), 18),
    }
    failed_sizes = {k: v for k, v in expected_sizes.items() if v[0] != v[1]}
    if failed_sizes:
        raise RuntimeError(f"STOP: final-table cardinality mismatch: {failed_sizes}")
    if not np.allclose(probabilities_df["mean_probability_sum_absent"], 1.0, atol=1e-10) or not np.allclose(probabilities_df["mean_probability_sum_present"], 1.0, atol=1e-10):
        raise RuntimeError("STOP: predicted joint probabilities do not sum to one")
    if effects_df[["OR", "CI_low", "CI_high", "raw_Wald_P"]].isna().any().any():
        raise RuntimeError("STOP: missing OR/CI/P values in final table")

    diagnostics.to_csv(OUT / "04_model_diagnostics.csv", index=False, encoding="utf-8-sig")
    fits_df.to_csv(OUT / "05_all_multistart_diagnostics.csv", index=False, encoding="utf-8-sig")
    params_df.to_csv(OUT / "06_selected_parameter_estimates.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUT / "07_concurrent_HAMD17_sensitivity.csv", index=False, encoding="utf-8-sig")
    selection_df.to_csv(OUT / "08_boundary_selection_audit.csv", index=False, encoding="utf-8-sig")

    for model, spec in MODEL_SPECS.items():
        label = spec["slug"]
        g = globals_df[globals_df["Model"].eq(model)].copy()
        e = effects_df[effects_df["Model"].eq(model)].copy()
        p = probabilities_df[probabilities_df["Model"].eq(model)].copy()
        write_excel(OUT / f"{label}_global_interactions.xlsx", {"global_interactions": g})
        write_excel(OUT / f"{label}_time_specific_ORs.xlsx", {"time_specific_ORs": e})
        write_excel(OUT / f"{label}_adjusted_probabilities.xlsx", {"adjusted_probabilities": p})
        g.to_csv(OUT / f"{label}_global_interactions.csv", index=False, encoding="utf-8-sig")
        e.to_csv(OUT / f"{label}_time_specific_ORs.csv", index=False, encoding="utf-8-sig")
        p.to_csv(OUT / f"{label}_adjusted_probabilities.csv", index=False, encoding="utf-8-sig")

    write_excel(OUT / "Step3_preliminary_complete_results.xlsx", {
        "design_checks": design_audit,
        "model_diagnostics": diagnostics,
        "global_interactions": globals_df,
        "time_specific_ORs": effects_df,
        "adjusted_probabilities": probabilities_df,
        "HAMD17_sensitivity": sensitivity,
        "boundary_selection": selection_df,
    })
    report = make_report(globals_df, diagnostics, sensitivity)
    (OUT / "Step3_preliminary_comparison_report.md").write_text(report, encoding="utf-8")
    atomic_json(OUT / "00_run_manifest.json", {
        "status": "COMPLETE_PRELIMINARY",
        "inference_label": "preliminary exploratory participant-cluster robust Wald",
        "formal_variance_issue_resolved": False,
        "freeze_id": FREEZE_ID,
        "freeze_sha256": base.sha256(base.FREEZE),
        "participants": EXPECTED_PARTICIPANTS,
        "person_visits": EXPECTED_VISITS,
        "models": 18,
        "model1_parameters_each": 52,
        "model2_parameters_each": 50,
        "global_tests": len(globals_df),
        "time_specific_OR_rows": len(effects_df),
        "probability_rows": len(probabilities_df),
        "all_selected_solutions_usable": bool(diagnostics["selected_solution_usable_preliminary"].all()),
        "boundary_review_required_models": int(selection_df.groupby(["Model", "Domain"])["boundary_review_required"].max().sum()),
    })
    log(f"COMPLETE_PRELIMINARY: {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the 18-model preliminary Step 3 analysis")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Use --run to execute the 18-model analysis")
    run()


if __name__ == "__main__":
    main()




