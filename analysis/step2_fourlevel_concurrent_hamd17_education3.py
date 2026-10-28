from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

from scipy import optimize

import model_comparison_gate_fibser as legacy
import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2B3_fixed_sample_covariate_attenuation as b3
import step2_fourlevel_no_concurrent_hamd17 as base
import step2_fourlevel_no_concurrent_hamd17_education3 as education3


from public_paths import ROOT
OUT = ROOT / "outputs" / "step2_fourlevel_concurrent_hamd17_education3"
COMPARATOR_OUT = ROOT / "outputs" / "step2_fourlevel_no_concurrent_hamd17_education3"
MAPPING_AUDIT = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "PRISE_Mapping_Verification.csv"
MODEL_ID = "FOUR_LEVEL_ZI_ORDINAL_ES_GEE_CONCURRENT_HAMD17_EDUCATION3_V1"
N_PARAMETERS = 46
RHS = b3.MODEL_RHS["M3"].replace("C(education)", education3.EDUCATION_TERM)


def log(message: str) -> None:
    print(f"Four-level concurrent-HAMD17 education3: {message}", flush=True)


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
    df["education3"] = pd.Categorical(
        df["education"].map(education3.EDUCATION_MAP),
        categories=education3.EDUCATION_LEVELS,
    )
    checks: list[dict] = []

    def check(name, observed, expected, passed):
        checks.append({"check": name, "observed": observed, "expected": expected, "pass": bool(passed)})

    check("freeze SHA256", freeze_hash, base.EXPECTED_FREEZE_SHA256, freeze_hash.lower() == base.EXPECTED_FREEZE_SHA256)
    check("source SHA256", version.get("source_sha256"), base.EXPECTED_SOURCE_SHA256, version.get("source_sha256", "").lower() == base.EXPECTED_SOURCE_SHA256)
    check("participants", df["participant_ID"].nunique(), base.EXPECTED_PARTICIPANTS, df["participant_ID"].nunique() == base.EXPECTED_PARTICIPANTS)
    check("person-visits", len(df), base.EXPECTED_VISITS, len(df) == base.EXPECTED_VISITS)
    duplicate_keys = int(df.duplicated(["participant_ID", "time"]).sum())
    check("duplicate participant_ID + time", duplicate_keys, 0, duplicate_keys == 0)

    df["burden_ord4"] = pd.to_numeric(df["burden_ord4"], errors="coerce")
    levels = sorted(df["burden_ord4"].dropna().astype(int).unique().tolist())
    counts = df["burden_ord4"].value_counts().sort_index().astype(int).to_dict()
    check("outcome complete", int(df["burden_ord4"].isna().sum()), 0, not df["burden_ord4"].isna().any())
    check("outcome levels", levels, [0, 1, 2, 3], levels == [0, 1, 2, 3])
    check("outcome counts", json.dumps(counts, sort_keys=True), json.dumps(base.EXPECTED_COUNTS, sort_keys=True), counts == base.EXPECTED_COUNTS)
    df["burden_ord4"] = df["burden_ord4"].astype(int)

    check("original education complete", int(df["education"].isna().sum()), 0, not df["education"].isna().any())
    check("derived education3 complete", int(df["education3"].isna().sum()), 0, not df["education3"].isna().any())
    check("education3 levels", json.dumps(list(df["education3"].cat.categories)), json.dumps(education3.EDUCATION_LEVELS), list(df["education3"].cat.categories) == education3.EDUCATION_LEVELS)
    changing = int((df.groupby("participant_ID", observed=True)["education3"].nunique() > 1).sum())
    check("participants changing education3", changing, 0, changing == 0)

    mapping = pd.read_csv(MAPPING_AUDIT)
    expected_domains = set(base.DOMAIN_VARS)
    mapping_domains = set(mapping["final_model_variable"].dropna().astype(str))
    mapping_pass = mapping["status"].astype(str).str.upper().eq("PASS").all()
    check("corrected PRISE mapping audit all PASS", bool(mapping_pass), True, bool(mapping_pass))
    check("corrected PRISE mapping contains nine domains", json.dumps(sorted(expected_domains & mapping_domains)), json.dumps(sorted(expected_domains)), expected_domains.issubset(mapping_domains))

    continuous = ["age_z", "baseline_HAMD17_z", "baseline_SDS_z", "concurrent_HAMD17_z"]
    for column in continuous:
        values = pd.to_numeric(df[column], errors="coerce")
        check(f"{column} finite and complete", int((~np.isfinite(values)).sum()), 0, np.isfinite(values).all())

    Gdf, Xdf = b1.make_design(df, mix_rhs=RHS, ord_rhs=RHS)
    names = parameter_names(Gdf, Xdf)
    all_columns = [*Gdf.columns, *Xdf.columns]
    forbidden = [c for c in all_columns if "site" in c.lower() or "province" in c.lower() or ":" in c]
    missing_mix_domains = [c for c in base.DOMAIN_VARS if c not in Gdf.columns]
    missing_ord_domains = [c for c in base.DOMAIN_VARS if c not in Xdf.columns]
    education_g = [c for c in Gdf.columns if "education3" in c]
    education_x = [c for c in Xdf.columns if "education3" in c]
    hamd_g = [c for c in Gdf.columns if c == "concurrent_HAMD17_z"]
    hamd_x = [c for c in Xdf.columns if c == "concurrent_HAMD17_z"]
    med_g = [c for c in Gdf.columns if "baseline_med_class" in c]
    med_x = [c for c in Xdf.columns if "baseline_med_class" in c]
    check("mixing regression parameters", Gdf.shape[1], 22, Gdf.shape[1] == 22)
    check("ordinal regression parameters", Xdf.shape[1], 21, Xdf.shape[1] == 21)
    check("ordered thresholds", base.Q, 3, base.Q == 3)
    check("total parameters", len(names), N_PARAMETERS, len(names) == N_PARAMETERS)
    check("unique parameter names", len(set(names)), N_PARAMETERS, len(set(names)) == N_PARAMETERS)
    check("concurrent HAMD-17 in mixing once", json.dumps(hamd_g), '["concurrent_HAMD17_z"]', hamd_g == ["concurrent_HAMD17_z"])
    check("concurrent HAMD-17 in ordinal once", json.dumps(hamd_x), '["concurrent_HAMD17_z"]', hamd_x == ["concurrent_HAMD17_z"])
    check("nine mixing domains", json.dumps(missing_mix_domains), "[]", not missing_mix_domains)
    check("nine ordinal domains", json.dumps(missing_ord_domains), "[]", not missing_ord_domains)
    check("education3 mixing contrasts", len(education_g), 2, len(education_g) == 2)
    check("education3 ordinal contrasts", len(education_x), 2, len(education_x) == 2)
    check("College or above reference absent", any("[T.College or above]" in c for c in education_g + education_x), False, not any("[T.College or above]" in c for c in education_g + education_x))
    check("SNRI medication reference mixing", json.dumps(med_g), '["C(baseline_med_class)[T.SSRI]"]', med_g == ["C(baseline_med_class)[T.SSRI]"])
    check("SNRI medication reference ordinal", json.dumps(med_x), '["C(baseline_med_class)[T.SSRI]"]', med_x == ["C(baseline_med_class)[T.SSRI]"])
    check("no site province or interactions", json.dumps(forbidden), "[]", not forbidden)
    check("threshold positions", json.dumps(names[22:25]), json.dumps(["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"]), names[22:25] == ["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"])

    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(checks)
    audit.to_csv(OUT / "01_preflight_hard_checks.csv", index=False, encoding="utf-8-sig")
    category = (
        df.assign(education3=df["education3"].astype("string"))
        .groupby("education3", dropna=False)
        .agg(participants=("participant_ID", "nunique"), person_visits=("participant_ID", "size"))
        .reindex(education3.EDUCATION_LEVELS)
        .reset_index()
    )
    category.to_csv(OUT / "02_education3_category_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"category": ["0", "1", "2", ">=3"], "n": [counts.get(i, 0) for i in range(4)]}).to_csv(OUT / "03_outcome_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "index_zero_based": range(N_PARAMETERS),
        "index_one_based": range(1, N_PARAMETERS + 1),
        "block": ["mixing"] * 22 + ["threshold"] * 3 + ["ordinal"] * 21,
        "parameter": names,
    }).to_csv(OUT / "04_parameter_index_map.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"component": "mixing", "columns": Gdf.shape[1], "RHS": RHS},
        {"component": "ordered thresholds", "columns": 3, "RHS": "a0 + cumulative positive exp(log_delta) increments"},
        {"component": "ordinal", "columns": Xdf.shape[1], "RHS": RHS},
    ]).to_csv(OUT / "04b_design_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"variable": c, "rule": "standardized on rebuilt long dataset before complete-case restriction; population SD ddof=0", "frozen_mean": float(df[c].mean()), "frozen_sd_ddof0": float(df[c].std(ddof=0))}
        for c in continuous
    ]).to_csv(OUT / "04c_standardization_audit.csv", index=False, encoding="utf-8-sig")
    base.atomic_json(OUT / "00_preflight_manifest.json", {
        "model_id": MODEL_ID,
        "freeze_id": base.FREEZE_ID,
        "freeze_sha256": freeze_hash,
        "source_sha256": version.get("source_sha256"),
        "participants": int(df["participant_ID"].nunique()),
        "person_visits": int(len(df)),
        "outcome_counts": {str(k): int(v) for k, v in counts.items()},
        "education_reference": "College or above",
        "mixing_parameters": int(Gdf.shape[1]),
        "ordinal_parameters": int(Xdf.shape[1]),
        "thresholds": 3,
        "total_parameters": len(names),
        "concurrent_HAMD17_in_both_components": bool(hamd_g and hamd_x),
        "initialization_rule": "map previous model estimates by exact parameter name; never by fixed index",
        "all_checks_pass": bool(audit["pass"].all()),
    })
    if not audit["pass"].all():
        raise RuntimeError(f"PREFLIGHT FAIL: {audit.loc[~audit['pass'], 'check'].tolist()}")
    return df, Gdf, Xdf, Gdf.to_numpy(float), Xdf.to_numpy(float), audit


def mapped_previous_start(names: list[str]) -> np.ndarray:
    source_path = COMPARATOR_OUT / "09_selected_parameter_estimates.csv"
    source = pd.read_csv(source_path)
    if source["parameter"].duplicated().any():
        raise RuntimeError("Previous-model parameter names are not unique")
    lookup = source.set_index("parameter")["estimate"].astype(float).to_dict()
    values = np.array([lookup.get(name, 0.0) for name in names], dtype=float)
    mapping = pd.DataFrame({
        "target_index_zero_based": range(len(names)),
        "target_parameter": names,
        "source_parameter_exact_match": [name if name in lookup else "" for name in names],
        "source_present": [name in lookup for name in names],
        "initial_value": values,
        "rule": ["exact_name_match" if name in lookup else "new_parameter_initialized_at_zero" for name in names],
    })
    mapping.to_csv(OUT / "04d_initial_parameter_name_mapping.csv", index=False, encoding="utf-8-sig")
    new_names = mapping.loc[~mapping["source_present"], "target_parameter"].tolist()
    expected_new = ["mixing_gamma:concurrent_HAMD17_z", "ordinal_beta:concurrent_HAMD17_z"]
    if new_names != expected_new:
        raise RuntimeError(f"Name-mapped initialization mismatch; new parameters were {new_names}")
    return values


def fit_full_sample_named(y, G, X):
    names = parameter_names(_LIVE_GDF, _LIVE_XDF)
    mapped = mapped_previous_start(names)
    generated = legacy.build_zi_starts(y, G, X, base.Q + 1)
    if len(generated) != 10:
        raise RuntimeError(f"Expected 10 generated starts, found {len(generated)}")
    starts = [{"start": "mapped_previous_no_concurrent_by_name", "method": "exact_parameter_name_map", "theta": mapped}, *generated]
    checkpoint_dir = OUT / "full_sample_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {
        "model_id": MODEL_ID,
        "freeze_sha256": base.EXPECTED_FREEZE_SHA256,
        "starts_requested": 11,
        "completed_starts": [],
        "status": "RUNNING",
    }
    if checkpoint.get("model_id") != MODEL_ID or checkpoint.get("freeze_sha256") != base.EXPECTED_FREEZE_SHA256 or checkpoint.get("starts_requested") != 11:
        raise RuntimeError("Existing full-sample checkpoint is incompatible")
    rows, candidates = [], []
    pd.DataFrame([{"fit_number": i, "start": s["start"], "start_method": s["method"]} for i, s in enumerate(starts, 1)]).to_csv(OUT / "04e_full_sample_prespecified_starts.csv", index=False, encoding="utf-8-sig")
    for fit_number, start in enumerate(starts, 1):
        diag_path = checkpoint_dir / f"start_{fit_number:02d}_diagnostics.csv"
        params_path = checkpoint_dir / f"start_{fit_number:02d}_named_parameters.csv"
        if fit_number in checkpoint.get("completed_starts", []) and diag_path.exists() and params_path.exists():
            row = pd.read_csv(diag_path).iloc[0].to_dict()
            saved = pd.read_csv(params_path).sort_values("parameter_index_zero_based")
            if len(saved) != N_PARAMETERS or saved["parameter"].tolist() != names:
                raise RuntimeError(f"Full-sample checkpoint start {fit_number} parameter name mismatch")
            theta = saved["estimate"].to_numpy(float)
            assessment = base.engine.basic_assessment(theta, y, G, X)
            _, jac = base.engine.jacobian_assessment(theta, y, G, X)
            strict = str(row["strict_valid"]).lower() == "true"
            log(f"full-sample start {fit_number}/11 restored from checkpoint")
        else:
            res = optimize.minimize(
                lambda th: base.engine.zi.neg_obj_grad(th, y, G, X, base.Q),
                start["theta"], method="L-BFGS-B", jac=True,
                options={"maxiter": 8000, "ftol": 1e-13, "gtol": 1e-9, "maxls": 120, "maxcor": 100},
            )
            theta = res.x
            assessment = base.engine.basic_assessment(theta, y, G, X)
            _, jac = base.engine.jacobian_assessment(theta, y, G, X)
            strict = bool(res.success and assessment["strict_score_converged"] and assessment["parameter_valid"] and jac["jacobian_full_rank"] and jac["jacobian_finite"])
            row = {"fit_number": fit_number, "start": start["start"], "start_method": start["method"], "program_success": bool(res.success), "message": str(res.message), "iterations": int(res.nit), **assessment, **jac, "strict_valid": strict, "selected": False}
            pd.DataFrame([row]).to_csv(diag_path, index=False, encoding="utf-8-sig")
            pd.DataFrame({"fit_number": fit_number, "start": start["start"], "parameter_index_zero_based": range(N_PARAMETERS), "parameter": names, "estimate": theta}).to_csv(params_path, index=False, encoding="utf-8-sig")
            checkpoint["completed_starts"] = sorted(set(checkpoint.get("completed_starts", [])) | {fit_number})
            checkpoint["status"] = "RUNNING"
            base.atomic_json(checkpoint_path, checkpoint)
            log(f"full-sample start {fit_number}/11 {start['start']}: success={res.success}, strict={strict}, objective={assessment['objective']:.6f}")
        rows.append(row)
        candidates.append({"theta": theta, "assessment": assessment, "jacobian": jac, "strict_valid": strict, "row": row})
    selected = base.engine.select_best(candidates)
    rows[selected["row"]["fit_number"] - 1]["selected"] = True
    checkpoint["selected_fit_number"] = int(selected["row"]["fit_number"])
    checkpoint["completed_starts"] = list(range(1, 12))
    checkpoint["status"] = "COMPLETE"
    base.atomic_json(checkpoint_path, checkpoint)
    return selected["theta"], pd.DataFrame(rows), candidates


def backfill_full_sample_checkpoint() -> None:
    """Create per-start resume artifacts from a completed run without refitting."""
    diagnostics = pd.read_csv(OUT / "05_full_sample_11_fit_diagnostics.csv")
    parameters = pd.read_csv(OUT / "06_full_sample_all_start_named_parameters.csv")
    checkpoint_dir = OUT / "full_sample_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for fit_number in range(1, 12):
        diagnostics.loc[diagnostics["fit_number"].eq(fit_number)].to_csv(checkpoint_dir / f"start_{fit_number:02d}_diagnostics.csv", index=False, encoding="utf-8-sig")
        parameters.loc[parameters["fit_number"].eq(fit_number)].to_csv(checkpoint_dir / f"start_{fit_number:02d}_named_parameters.csv", index=False, encoding="utf-8-sig")
    selected_fit = int(diagnostics.loc[diagnostics["selected"].astype(str).str.lower().eq("true"), "fit_number"].iloc[0])
    base.atomic_json(checkpoint_dir / "checkpoint.json", {
        "model_id": MODEL_ID,
        "freeze_sha256": base.EXPECTED_FREEZE_SHA256,
        "starts_requested": 11,
        "completed_starts": list(range(1, 12)),
        "selected_fit_number": selected_fit,
        "status": "COMPLETE",
    })


def configure() -> None:
    global _LIVE_GDF, _LIVE_XDF
    _, _LIVE_GDF, _LIVE_XDF, _, _, _ = load_and_preflight()
    base.OUT = OUT
    base.MODEL_ID = MODEL_ID
    base.N_PARAMETERS = N_PARAMETERS
    base.RHS = RHS
    base.parameter_names = parameter_names
    base.load_and_preflight = load_and_preflight
    base.engine.OUT = OUT
    base.engine.log = log
    base.engine.fit_full_sample = fit_full_sample_named
    base.log = log


def audit_full_sample() -> None:
    diagnostics = pd.read_csv(OUT / "05_full_sample_11_fit_diagnostics.csv")
    strict = diagnostics[diagnostics["strict_valid"].astype(str).str.lower().eq("true")].copy()
    strict["high_mixing"] = strict["max_abs_mixing_coefficient"] > 5
    strict["boundary_probability"] = (strict["min_pi"] < 1e-5) | (strict["max_pi"] > 1 - 1e-10)
    selected_obj = float(strict.loc[strict["selected"].astype(str).str.lower().eq("true"), "objective"].iloc[0])
    strict["objective_gap_from_selected"] = strict["objective"] - selected_obj
    strict.to_csv(OUT / "12a_full_sample_local_solution_audit.csv", index=False, encoding="utf-8-sig")
    regular = strict.loc[~strict["high_mixing"] & ~strict["boundary_probability"]]
    high = strict.loc[strict["high_mixing"]]
    boundary = strict.loc[strict["boundary_probability"]]
    best_regular = float(regular["objective"].max()) if len(regular) else -np.inf
    best_high = float(high["objective"].max()) if len(high) else -np.inf
    best_boundary = float(boundary["objective"].max()) if len(boundary) else -np.inf
    selected = strict.loc[strict["selected"].astype(str).str.lower().eq("true")].iloc[0]
    gate_path = OUT / "12_full_sample_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate.update({
        "jacobian_min_singular_value": float(selected["jacobian_min_singular_value"]),
        "selected_min_pi": float(selected["min_pi"]),
        "selected_max_pi": float(selected["max_pi"]),
        "selected_boundary_probability": bool(selected["boundary_probability"]),
        "high_mixing_competing_local_solution": bool(len(high) and best_high >= selected_obj - 1e-3),
        "boundary_solution_higher_than_best_regular": bool(len(boundary) and (not len(regular) or best_boundary > best_regular + 1e-8)),
        "best_regular_objective": None if not len(regular) else best_regular,
        "best_high_mixing_objective": None if not len(high) else best_high,
        "best_boundary_objective": None if not len(boundary) else best_boundary,
    })
    numerical_review = bool(gate["jacobian_condition_number"] > 1e8 or gate["selected_boundary_probability"] or gate["high_mixing_competing_local_solution"] or gate["boundary_solution_higher_than_best_regular"])
    gate["status"] = "FAIL" if not gate["full_sample_numerical_gate_pass"] else ("REVIEW REQUIRED" if gate["extreme_mixing_abs_gt_5"] or numerical_review else "PASS")
    base.atomic_json(gate_path, gate)


def compare_results_and_finalize() -> None:
    new_effects = pd.read_csv(OUT / "10_domain_effects_robust_wald.csv")
    old_effects = pd.read_csv(COMPARATOR_OUT / "10_domain_effects_robust_wald.csv")
    effects = new_effects.merge(old_effects, on=["Domain", "Component"], suffixes=("_with_concurrent", "_without_concurrent"))
    effects["OR_ratio_with_vs_without"] = effects["OR_with_concurrent"] / effects["OR_without_concurrent"]
    effects["log_OR_difference"] = np.log(effects["OR_with_concurrent"]) - np.log(effects["OR_without_concurrent"])
    effects.to_csv(OUT / "18_effect_comparison_vs_no_concurrent_HAMD17.csv", index=False, encoding="utf-8-sig")

    new_prob = pd.read_csv(OUT / "11_adjusted_joint_probabilities.csv")
    old_prob = pd.read_csv(COMPARATOR_OUT / "11_adjusted_joint_probabilities.csv")
    probabilities = new_prob.merge(old_prob, on="Domain", suffixes=("_with_concurrent", "_without_concurrent"))
    for cutoff in (1, 3):
        key = f"RD_burden_ge{cutoff}_domain1_minus_domain0"
        probabilities[f"{key}_difference_with_minus_without"] = probabilities[f"{key}_with_concurrent"] - probabilities[f"{key}_without_concurrent"]
    probabilities.to_csv(OUT / "19_joint_probability_comparison_vs_no_concurrent_HAMD17.csv", index=False, encoding="utf-8-sig")

    df, Gdf, Xdf, G, X, _ = load_and_preflight()
    y = df["burden_ord4"].to_numpy(int)
    ids = df["participant_ID"].unique()
    pid = df["participant_ID"].to_numpy()
    names = parameter_names(Gdf, Xdf)
    covariance_rows, basin_rows = [], []
    all_diag = pd.read_csv(OUT / "14_bootstrap_5_all_start_diagnostics.csv")
    all_params = pd.read_csv(OUT / "15_bootstrap_5_all_start_named_parameters.csv")
    for rep in range(1, 6):
        draw = np.random.default_rng(base.SEED + rep).choice(ids, len(ids), replace=True)
        idx_parts = [np.flatnonzero(pid == ident) for ident in draw]
        idx = np.concatenate(idx_parts)
        clusters = np.concatenate([np.repeat(j, len(part)) for j, part in enumerate(idx_parts)])
        diag = all_diag.loc[all_diag["bootstrap_replicate"].eq(rep)].copy()
        selected_row = diag.loc[diag["selected"].astype(str).str.lower().eq("true")].iloc[0]
        start_number = int(selected_row["start_number"])
        theta_rows = all_params.loc[(all_params["bootstrap_replicate"].eq(rep)) & (all_params["start_number"].eq(start_number))].sort_values("parameter_index_zero_based")
        if len(theta_rows) != N_PARAMETERS or theta_rows["parameter"].tolist() != names:
            raise RuntimeError(f"Bootstrap {rep}: saved parameter name map mismatch")
        theta = theta_rows["estimate"].to_numpy(float)
        _, cov = base.engine.selected_covariance(theta, y[idx], G[idx], X[idx], clusters)
        row = {"bootstrap_replicate": rep, "selected_start_number": start_number, "jacobian_min_singular_value": float(selected_row.get("jacobian_min_singular_value", np.nan)), **cov.iloc[0].to_dict()}
        covariance_rows.append(row)
        valid = diag[diag["basic_valid"].astype(str).str.lower().eq("true")].copy()
        valid["high_mixing"] = valid["max_abs_mixing_coefficient"] > 5
        valid["boundary_probability"] = (valid["min_pi"] < 1e-5) | (valid["max_pi"] > 1 - 1e-10)
        best_regular = valid.loc[~valid["high_mixing"] & ~valid["boundary_probability"], "objective"].max()
        best_high = valid.loc[valid["high_mixing"], "objective"].max()
        best_boundary = valid.loc[valid["boundary_probability"], "objective"].max()
        basin_rows.append({
            "bootstrap_replicate": rep,
            "best_objective": float(valid["objective"].max()),
            "best_regular_objective": float(best_regular) if np.isfinite(best_regular) else np.nan,
            "best_high_mixing_objective": float(best_high) if np.isfinite(best_high) else np.nan,
            "best_boundary_objective": float(best_boundary) if np.isfinite(best_boundary) else np.nan,
            "high_mixing_competing_within_1e-3": bool(np.isfinite(best_high) and best_high >= valid["objective"].max() - 1e-3),
            "boundary_higher_than_regular": bool(np.isfinite(best_boundary) and (not np.isfinite(best_regular) or best_boundary > best_regular + 1e-8)),
        })
    covariances = pd.DataFrame(covariance_rows)
    basins = pd.DataFrame(basin_rows)
    covariances.to_csv(OUT / "20_bootstrap_selected_covariance_diagnostics.csv", index=False, encoding="utf-8-sig")
    basins.to_csv(OUT / "21_bootstrap_local_solution_audit.csv", index=False, encoding="utf-8-sig")

    gate = json.loads((OUT / "12_full_sample_gate.json").read_text(encoding="utf-8"))
    reps = pd.read_csv(OUT / "13_bootstrap_5_replicate_diagnostics.csv")
    complete = len(all_diag) == 55 and len(reps) == 5 and (reps["starts_executed"] == 11).all() and reps["selected"].astype(str).str.lower().eq("true").all()
    covariance_ok = covariances["jacobian_full_rank"].all() and covariances["covariance_finite"].all() and covariances["covariance_PSD"].all()
    boundary = (reps["min_pi"] < 1e-5) | (reps["max_pi"] > 1 - 1e-10)
    high_mixing = reps["max_abs_mixing_coefficient"] > 5
    review = bool(gate["status"] == "REVIEW REQUIRED" or boundary.any() or high_mixing.any() or basins["high_mixing_competing_within_1e-3"].any() or basins["boundary_higher_than_regular"].any() or (covariances["jacobian_condition_number"] > 1e8).any())
    status = "FAIL" if (not gate["full_sample_numerical_gate_pass"] or not complete or not covariance_ok) else ("REVIEW REQUIRED" if review else "PASS")
    summary = {
        "model_id": MODEL_ID,
        "final_status": status,
        "full_sample_status": gate["status"],
        "full_sample_all_11_starts": bool(gate["all_11_fits_executed"]),
        "bootstrap_replicates_completed": int(len(reps)),
        "bootstrap_total_start_fits": int(len(all_diag)),
        "bootstrap_selected_full_rank_PSD": int((covariances["jacobian_full_rank"] & covariances["covariance_PSD"]).sum()),
        "bootstrap_selected_boundary_probability": int(boundary.sum()),
        "bootstrap_selected_high_mixing": int(high_mixing.sum()),
        "bootstrap_high_mixing_competing_basins": int(basins["high_mixing_competing_within_1e-3"].sum()),
        "bootstrap_boundary_higher_than_regular": int(basins["boundary_higher_than_regular"].sum()),
        "max_abs_log_OR_change_vs_no_concurrent": float(effects["log_OR_difference"].abs().max()),
        "max_abs_RD_change_vs_no_concurrent": float(max(probabilities["RD_burden_ge1_domain1_minus_domain0_difference_with_minus_without"].abs().max(), probabilities["RD_burden_ge3_domain1_minus_domain0_difference_with_minus_without"].abs().max())),
        "formal_bootstrap_inference": False,
        "FDR_performed": False,
        "step3_performed": False,
    }
    base.atomic_json(OUT / "22_final_summary.json", summary)

    report = [
        "# Step 2 四级 Zero-Inflated Ordinal ES-GEE（同期 HAMD-17＋教育三分类）",
        "",
        f"## 最终判定：{status}",
        "",
        "- 冻结样本：1468 participants / 4799 person-visits；participant_ID＋time 无重复。",
        "- FIBSER burden：0 / 1 / 2 / ≥3 = 3464 / 584 / 512 / 239。",
        "- 参数：mixing 22＋threshold 3＋ordinal 21＝46；同期 HAMD-17 同时进入两个部分。",
        "- 教育参照：College or above；九个修复后的 PRISE 域同时进入两个部分；未加入 site、province 或 domain×time。",
        f"- 全样本：11/11 起点；strict-valid {gate['strict_valid_fits']}/11；目标函数 {gate['selected_objective']:.9f}。",
        f"- Jacobian：full rank={gate['jacobian_full_rank']}，condition={gate['jacobian_condition_number']:.6g}，最小奇异值={gate['jacobian_min_singular_value']:.6g}。",
        f"- 协方差：finite={gate['covariance_finite']}，PSD={gate['covariance_PSD']}，condition={gate['covariance_condition_number']:.6g}。",
        f"- 全样本 mixing：max |gamma|={gate['max_abs_mixing_coefficient']:.6g}，min/max π={gate['selected_min_pi']:.6g}/{gate['selected_max_pi']:.6g}。",
        f"- Bootstrap 稳定性试运行：5/5 replicates，55/55 起点；选定解 full-rank＋PSD {summary['bootstrap_selected_full_rank_PSD']}/5。",
        f"- Bootstrap 选定边界概率解 {summary['bootstrap_selected_boundary_probability']}/5；极端 mixing 解 {summary['bootstrap_selected_high_mixing']}/5。",
        f"- 与去同期 HAMD-17 模型相比：18 项效应最大 |Δlog OR|={summary['max_abs_log_OR_change_vs_no_concurrent']:.6g}；联合概率差最大绝对变化={summary['max_abs_RD_change_vs_no_concurrent']:.6g}。",
        "",
        "18 项 OR、95% Wald CI 与原始 Wald P 见 `10_domain_effects_robust_wald.csv`；九域 P(burden≥1)、P(burden≥3) 与概率差见 `11_adjusted_joint_probabilities.csv`。本次 5 次 cluster bootstrap 仅为数值稳定性试运行，不构成正式 bootstrap 推断；未做 FDR 或 Step 3。",
    ]
    (OUT / "Final_Model_Report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_full() -> None:
    configure()
    base.run_full_sample()
    audit_full_sample()


def run_bootstrap() -> None:
    configure()
    base.run_bootstrap(5)
    compare_results_and_finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    run_bootstrap() if args.bootstrap else run_full()




