from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

from scipy import optimize, stats

import model_comparison_gate_fibser as legacy
import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2_fourlevel_no_concurrent_hamd17 as base
import step3_domain_time_burden_4799_preliminary as common


from public_paths import ROOT
FREEZE_ID = "PRISE_REPAIRED_NATURAL_CC_1468P_4799V_2026-10-08"
EXPECTED_PARTICIPANTS = 1468
EXPECTED_VISITS = 4799
EXPECTED_COUNTS = {0: 3464, 1: 584, 2: 512, 3: 239}
HAMD_ONLY_N = 5443
HAMD_ONLY_MEAN = 10.528201359544369
HAMD_ONLY_SD_DDOF0 = 6.439452465864943
Q = 3
SCORE_TOL = 5e-2


def log(model: str, message: str) -> None:
    print(f"Step3 independent {model}: {message}", flush=True)


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def threshold_parameter_names(q: int) -> list[str]:
    if q < 1:
        raise ValueError("The ordinal model requires at least one threshold")
    return ["threshold_raw:a0"] + [f"threshold_raw:log_delta{j}" for j in range(1, q)]


def parameter_names(gdf: pd.DataFrame, xdf: pd.DataFrame, q: int) -> list[str]:
    return (
        [f"mixing_gamma:{c}" for c in gdf.columns]
        + threshold_parameter_names(q)
        + [f"ordinal_beta:{c}" for c in xdf.columns]
    )


def load_frozen_data(out: Path, spec: dict) -> pd.DataFrame:
    use_concurrent_hamd17 = bool(spec["concurrent_hamd17"])
    q = int(spec.get("q", Q))
    outcome_column = str(spec.get("outcome_column", "burden_ord4"))
    expected_counts = {int(k): int(v) for k, v in spec.get("outcome_counts", EXPECTED_COUNTS).items()}
    version = json.loads(base.VERSION_RECORD.read_text(encoding="utf-8"))
    freeze_hash = base.sha256(base.FREEZE)
    df = pd.read_csv(base.FREEZE)
    df["education3"] = pd.Categorical(
        df["education"].map(common.model2_base.EDUCATION_MAP),
        categories=common.model2_base.EDUCATION_LEVELS,
    )
    checks: list[dict] = []

    def check(name, observed, expected, passed) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "pass": bool(passed)})

    check("freeze ID", base.FREEZE_ID, FREEZE_ID, base.FREEZE_ID == FREEZE_ID)
    check("freeze SHA256", freeze_hash, base.EXPECTED_FREEZE_SHA256, freeze_hash.lower() == base.EXPECTED_FREEZE_SHA256)
    check("source SHA256", version.get("source_sha256"), base.EXPECTED_SOURCE_SHA256, version.get("source_sha256", "").lower() == base.EXPECTED_SOURCE_SHA256)
    check("participants", df["participant_ID"].nunique(), EXPECTED_PARTICIPANTS, df["participant_ID"].nunique() == EXPECTED_PARTICIPANTS)
    check("person-visits", len(df), EXPECTED_VISITS, len(df) == EXPECTED_VISITS)
    check("unique participant_ID + time", int(df.duplicated(["participant_ID", "time"]).sum()), 0, not df.duplicated(["participant_ID", "time"]).any())
    counts4 = df["burden_ord4"].value_counts().sort_index().astype(int).to_dict()
    check("frozen four-level burden counts", json.dumps(counts4, sort_keys=True), json.dumps(EXPECTED_COUNTS, sort_keys=True), counts4 == EXPECTED_COUNTS)
    df["burden_ord4"] = df["burden_ord4"].astype(int)
    if outcome_column == "burden_ord3":
        df[outcome_column] = np.select(
            [df["burden_ord4"].eq(0), df["burden_ord4"].eq(1)],
            [0, 1],
            default=2,
        ).astype(int)
    if outcome_column not in df:
        raise RuntimeError(f"STOP: requested outcome column is absent: {outcome_column}")
    counts = df[outcome_column].value_counts().sort_index().astype(int).to_dict()
    check(f"analysis outcome counts: {outcome_column}", json.dumps(counts, sort_keys=True), json.dumps(expected_counts, sort_keys=True), counts == expected_counts)
    check("threshold count matches outcome categories", q, len(expected_counts) - 1, q == len(expected_counts) - 1)
    check("education3 complete", int(df["education3"].isna().sum()), 0, not df["education3"].isna().any())
    if use_concurrent_hamd17:
        original_z = pd.to_numeric(df["concurrent_HAMD17_z"], errors="coerce").copy()
        raw = pd.to_numeric(df["concurrent_HAMD17"], errors="coerce")
        df["concurrent_HAMD17_z"] = (raw - HAMD_ONLY_MEAN) / HAMD_ONLY_SD_DDOF0
        check("concurrent HAMD raw complete", int(raw.isna().sum()), 0, not raw.isna().any())
        check("HAMD-only standardization reference N", HAMD_ONLY_N, 5443, HAMD_ONLY_N == 5443)
        check("HAMD-only standardization mean", HAMD_ONLY_MEAN, 10.528201359544369, HAMD_ONLY_MEAN == 10.528201359544369)
        check("HAMD-only standardization SD ddof=0", HAMD_ONLY_SD_DDOF0, 6.439452465864943, HAMD_ONLY_SD_DDOF0 == 6.439452465864943)
        check("HAMD z changed from contaminated frozen column", float(np.max(np.abs(df["concurrent_HAMD17_z"] - original_z))), ">0", bool(np.max(np.abs(df["concurrent_HAMD17_z"] - original_z)) > 0))
    else:
        check("concurrent HAMD excluded by model specification", "excluded", "excluded", True)
    for column in common.DOMAIN_VARS:
        observed = set(pd.to_numeric(df[column], errors="coerce").dropna().unique())
        check(f"binary complete occurrence variable: {column}", json.dumps(sorted(observed)), "[0, 1]", observed.issubset({0, 1, 0.0, 1.0}) and not df[column].isna().any())
    audit = pd.DataFrame(checks)
    audit.to_csv(out / "01_frozen_sample_and_domain_checks.csv", index=False, encoding="utf-8-sig")
    if not audit["pass"].all():
        raise RuntimeError(f"STOP: frozen data gate failed: {audit.loc[~audit['pass'], 'check'].tolist()}")
    df[outcome_column] = df[outcome_column].astype(int)
    return df


def preflight_designs(model: str, spec: dict, df: pd.DataFrame, out: Path) -> pd.DataFrame:
    q = int(spec.get("q", Q))
    rows: list[dict] = []
    maps: list[pd.DataFrame] = []
    for domain in b1.DOMAINS:
        var = b1.DOMAIN_VAR[domain]
        rhs = common.rhs_for(spec["rhs"], var)
        gdf, xdf = b1.make_design(df, rhs, rhs)
        names = parameter_names(gdf, xdf, q)
        expected_interactions = {common.interaction_column(var, w) for w in common.NONREF_TIMES}
        hamd_g = [c for c in gdf.columns if c == "concurrent_HAMD17_z"]
        hamd_x = [c for c in xdf.columns if c == "concurrent_HAMD17_z"]
        checks = {
            "mixing_columns_ok": gdf.shape[1] == spec["expected_g"],
            "ordinal_columns_ok": xdf.shape[1] == spec["expected_x"],
            "parameters_ok": len(names) == spec["expected_parameters"],
            "unique_parameter_names": len(names) == len(set(names)),
            "mixing_interactions_exact": {c for c in gdf.columns if ":" in c} == expected_interactions,
            "ordinal_interactions_exact": {c for c in xdf.columns if ":" in c} == expected_interactions,
            "nine_domains_both": all(v in gdf.columns and v in xdf.columns for v in common.DOMAIN_VARS),
            "education_two_contrasts_both": len([c for c in gdf.columns if "education3" in c]) == 2 and len([c for c in xdf.columns if "education3" in c]) == 2,
            "college_reference": not any("[T.College or above]" in c for c in [*gdf.columns, *xdf.columns]),
            "concurrent_hamd_rule": (hamd_g == ["concurrent_HAMD17_z"] and hamd_x == ["concurrent_HAMD17_z"]) if spec["concurrent_hamd17"] else (not hamd_g and not hamd_x),
            "no_site_or_province": not any("site" in c.lower() or "province" in c.lower() for c in [*gdf.columns, *xdf.columns]),
        }
        rows.append({
            "Model": model,
            "Domain": domain,
            "mixing_columns": gdf.shape[1],
            "thresholds": q,
            "ordinal_columns": xdf.shape[1],
            "total_parameters": len(names),
            "all_design_checks_pass": all(checks.values()),
            **checks,
            "mixing_columns_json": json.dumps(list(gdf.columns), ensure_ascii=False),
            "ordinal_columns_json": json.dumps(list(xdf.columns), ensure_ascii=False),
        })
        maps.append(pd.DataFrame({
            "Model": model,
            "Domain": domain,
            "parameter_index_zero_based": range(len(names)),
            "parameter": names,
        }))
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "02_design_matrix_checks.csv", index=False, encoding="utf-8-sig")
    pd.concat(maps, ignore_index=True).to_csv(out / "03_parameter_name_maps.csv", index=False, encoding="utf-8-sig")
    if not audit["all_design_checks_pass"].all():
        raise RuntimeError("STOP: data -> design -> parameter-name alignment gate failed")
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
    missing = mapping.loc[~mapping["source_present"], "target_parameter"].tolist()
    if len(missing) != 6 or not all(":" in name for name in missing):
        raise RuntimeError(f"STOP: warm-start mapping mismatch: {missing}")
    return theta, mapping


def candidate_diagnostics(fit_number: int, start: dict, res, theta: np.ndarray, y: np.ndarray, g: np.ndarray, x: np.ndarray, clusters: np.ndarray, q: int) -> tuple[dict, np.ndarray]:
    assessment = base.engine.basic_assessment(theta, y, g, x)
    _, jac = base.engine.jacobian_assessment(theta, y, g, x)
    sw = base.engine.zi.sandwich(theta, y, g, x, q, clusters)
    a_eig = np.linalg.eigvalsh((sw["A"] + sw["A"].T) / 2.0)
    v = (sw["V"] + sw["V"].T) / 2.0
    v_eig = np.linalg.eigvalsh(v)
    strict = bool(res.success and assessment["strict_score_converged"] and assessment["parameter_valid"] and jac["jacobian_finite"] and jac["jacobian_full_rank"])
    row = {
        "fit_number": fit_number,
        "start": start["start"],
        "start_method": start["method"],
        "program_success": bool(res.success),
        "message": str(res.message),
        "iterations": int(res.nit),
        **assessment,
        **jac,
        "covariance_formula": "V = A_inv @ B @ A_inv.T",
        "covariance_finite": bool(np.isfinite(v).all()),
        "covariance_PSD": bool(np.isfinite(v_eig).all() and v_eig.min() >= -1e-8),
        "covariance_min_eigenvalue": float(v_eig.min()),
        "covariance_condition_number": float(np.linalg.cond(v)),
        "max_robust_SE": float(np.sqrt(np.clip(np.diag(v), 0, np.inf)).max()),
        "information_min_eigenvalue": float(a_eig.min()),
        "information_positive_definite": bool(np.isfinite(a_eig).all() and a_eig.min() > 1e-8),
        "strict_valid": strict,
        "selected": False,
    }
    return row, v


def fit_domain(model: str, spec: dict, domain: str, y: np.ndarray, clusters: np.ndarray, gdf: pd.DataFrame, xdf: pd.DataFrame, out: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q = int(spec.get("q", Q))
    g, x = gdf.to_numpy(float), xdf.to_numpy(float)
    names = parameter_names(gdf, xdf, q)
    warm, mapping = mapped_step2_start(spec, names)
    generated = legacy.build_zi_starts(y, g, x, q + 1)
    if len(generated) != 10:
        raise RuntimeError(f"STOP: expected 10 generated starts, got {len(generated)}")
    starts = [{"start": "mapped_step2_by_exact_name", "theta": warm, "method": "exact_parameter_name_map"}]
    starts.extend({"start": s["start"], "theta": s["theta"], "method": s["method"]} for s in generated)
    if len(starts) != 11:
        raise RuntimeError("STOP: exactly 11 starts required")

    domain_dir = out / "domains" / domain.replace("/", "_").replace(" ", "_")
    domain_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(domain_dir / "initial_parameter_mapping.csv", index=False, encoding="utf-8-sig")
    candidates: list[dict] = []
    extreme_rows: list[dict] = []
    for fit_number, start in enumerate(starts, 1):
        res = optimize.minimize(
            lambda th: base.engine.zi.neg_obj_grad(th, y, g, x, q),
            start["theta"],
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 4000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 100, "maxcor": 80},
        )
        row, covariance = candidate_diagnostics(fit_number, start, res, res.x, y, g, x, clusters, q)
        candidates.append({"theta": res.x, "row": row, "covariance": covariance})
        pd.DataFrame([row]).to_csv(domain_dir / f"start_{fit_number:02d}_diagnostics.csv", index=False, encoding="utf-8-sig")
        params = pd.DataFrame({"fit_number": fit_number, "parameter_index_zero_based": range(len(names)), "parameter": names, "estimate": res.x})
        params.to_csv(domain_dir / f"start_{fit_number:02d}_named_parameters.csv", index=False, encoding="utf-8-sig")
        mix = params.iloc[: g.shape[1]].assign(abs_estimate=lambda z: z["estimate"].abs()).sort_values("abs_estimate", ascending=False).head(5)
        extreme_rows.extend({"Model": model, "Domain": domain, **r} for r in mix.to_dict(orient="records"))
        log(model, f"{domain}: start {fit_number}/11 success={res.success} strict={row['strict_valid']} objective={row['objective']:.6f} score={row['max_abs_score']:.4g}")

    # All eleven starts have completed before refinement or selection.
    strict = [c for c in candidates if c["row"]["strict_valid"]]
    if not strict:
        raise RuntimeError(f"STOP: {model}/{domain} has no strict-valid start")
    best = max(strict, key=lambda c: (float(c["row"]["objective"]), -float(c["row"]["max_abs_score"])))
    refined_start = {"start": "refined_from_best_strict_raw", "theta": best["theta"], "method": best["row"]["start"]}
    res = optimize.minimize(
        lambda th: base.engine.zi.neg_obj_grad(th, y, g, x, q),
        refined_start["theta"],
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 8000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 120, "maxcor": 100},
    )
    refined_row, refined_cov = candidate_diagnostics(12, refined_start, res, res.x, y, g, x, clusters, q)
    candidates.append({"theta": res.x, "row": refined_row, "covariance": refined_cov})
    pd.DataFrame([refined_row]).to_csv(domain_dir / "start_12_refinement_diagnostics.csv", index=False, encoding="utf-8-sig")

    strict_all = [c for c in candidates if c["row"]["strict_valid"]]
    ordered = sorted(strict_all, key=lambda c: (float(c["row"]["objective"]), -float(c["row"]["max_abs_score"])), reverse=True)
    selection_rows: list[dict] = []
    selected = None
    for candidate in ordered:
        row = candidate["row"]
        preeligible = bool(row["jacobian_finite"] and row["jacobian_full_rank"] and row["jacobian_condition_number"] <= 1e12 and row["jacobian_min_singular_value"] > 1e-8)
        eligible = bool(preeligible and row["information_positive_definite"] and row["covariance_finite"] and row["covariance_PSD"])
        if eligible and selected is None:
            selected = candidate
        selection_rows.append({
            "Model": model,
            "Domain": domain,
            "fit_number": int(row["fit_number"]),
            "start": row["start"],
            "objective": float(row["objective"]),
            "max_abs_score": float(row["max_abs_score"]),
            "max_abs_mixing_coefficient": float(row["max_abs_mixing_coefficient"]),
            "rank_condition_prefilter": preeligible,
            "curvature_positive_definite": bool(row["information_positive_definite"]),
            "covariance_finite": bool(row["covariance_finite"]),
            "covariance_PSD": bool(row["covariance_PSD"]),
            "eligible_for_preliminary_Wald": eligible,
            "classification": "REGULAR_OR_FINITE_BOUNDARY" if eligible else "WEAK_IDENTIFICATION",
        })
    if selected is None:
        raise RuntimeError(f"STOP: {model}/{domain} has no locally identified finite candidate")
    selected_number = int(selected["row"]["fit_number"])
    selected_objective = float(selected["row"]["objective"])
    highest_strict_objective = max(float(c["row"]["objective"]) for c in strict_all)
    review_required = bool(highest_strict_objective > selected_objective + 1e-6)
    selection = pd.DataFrame(selection_rows)
    selection["selected_for_preliminary_Wald"] = selection["fit_number"].eq(selected_number)
    selection["boundary_review_required"] = review_required
    selection["highest_strict_objective"] = highest_strict_objective
    selection["selected_regular_objective"] = selected_objective
    selection.to_csv(domain_dir / "boundary_selection_audit.csv", index=False, encoding="utf-8-sig")
    all_rows = pd.DataFrame([c["row"] for c in candidates]).sort_values("fit_number")
    all_rows["selected"] = all_rows["fit_number"].eq(selected_number)
    all_rows["boundary_review_required"] = review_required
    all_rows.to_csv(domain_dir / "all_start_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(extreme_rows).to_csv(domain_dir / "largest_mixing_parameters_by_start.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"parameter_index_zero_based": range(len(names)), "parameter": names, "estimate": selected["theta"]}).to_csv(domain_dir / "selected_named_parameters.csv", index=False, encoding="utf-8-sig")
    return selected["theta"], all_rows, selection, pd.DataFrame(extreme_rows)


def adjusted_probabilities(model: str, domain: str, var: str, theta: np.ndarray, gdf: pd.DataFrame, xdf: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    gcols, xcols = list(gdf.columns), list(xdf.columns)
    for week in common.TIMES:
        matrices = []
        for present in [0, 1]:
            g = gdf.to_numpy(float).copy()
            x = xdf.to_numpy(float).copy()
            common.set_fixed_time(g, gcols, week)
            common.set_fixed_time(x, xcols, week)
            g[:, gcols.index(var)] = float(present)
            x[:, xcols.index(var)] = float(present)
            for nonref in common.NONREF_TIMES:
                interaction = common.interaction_column(var, nonref)
                g[:, gcols.index(interaction)] = float(present == 1 and week == nonref)
                x[:, xcols.index(interaction)] = float(present == 1 and week == nonref)
            matrices.append((g, x))
        p_absent = base.engine.predict(theta, matrices[0][0], matrices[0][1])[0]
        p_present = base.engine.predict(theta, matrices[1][0], matrices[1][1])[0]
        ge1_absent = p_absent[:, 1:].sum(axis=1)
        ge1_present = p_present[:, 1:].sum(axis=1)
        rows.append({
            "Model": model,
            "Domain": domain,
            "Time": week,
            "P_burden_ge1_domain_absent": float(ge1_absent.mean()),
            "P_burden_ge1_domain_present": float(ge1_present.mean()),
            "Adjusted_probability_difference_present_minus_absent": float((ge1_present - ge1_absent).mean()),
            "standardization": "all 4799 frozen person-visits; time fixed; all other covariates retained at observed values",
            "mean_probability_sum_absent": float(p_absent.sum(axis=1).mean()),
            "mean_probability_sum_present": float(p_present.sum(axis=1).mean()),
        })
    return rows


def covariance_diagnostics(model: str, domain: str, theta: np.ndarray, y: np.ndarray, gdf: pd.DataFrame, xdf: pd.DataFrame, clusters: np.ndarray, q: int) -> tuple[np.ndarray, dict, pd.DataFrame]:
    g, x = gdf.to_numpy(float), xdf.to_numpy(float)
    sw = base.engine.zi.sandwich(theta, y, g, x, q, clusters)
    covariance = (sw["V"] + sw["V"].T) / 2.0
    eig = np.linalg.eigvalsh(covariance)
    a_eig = np.linalg.eigvalsh((sw["A"] + sw["A"].T) / 2.0)
    assessment = base.engine.basic_assessment(theta, y, g, x)
    _, jac = base.engine.jacobian_assessment(theta, y, g, x)
    names = parameter_names(gdf, xdf, q)
    se = np.asarray(sw["se"], dtype=float)
    params = pd.DataFrame({"parameter": names, "estimate": theta, "robust_SE": se})
    params["z"] = params["estimate"] / params["robust_SE"]
    params["raw_p"] = 2 * stats.norm.sf(np.abs(params["z"]))
    interaction_names = {
        f"mixing_gamma:{common.interaction_column(b1.DOMAIN_VAR[domain], week)}"
        for week in common.NONREF_TIMES
    } | {
        f"ordinal_beta:{common.interaction_column(b1.DOMAIN_VAR[domain], week)}"
        for week in common.NONREF_TIMES
    }
    focal = params[params["parameter"].isin(interaction_names)]
    if len(focal) != 6:
        raise RuntimeError(f"STOP: expected six focal interaction parameters for {model}/{domain}, got {len(focal)}")
    boundary = bool(
        assessment["max_abs_parameter"] >= 50
        or assessment["max_abs_mixing_coefficient"] > 8
        or assessment["min_pi"] <= 1e-8
        or assessment["max_pi"] >= 1 - 1e-8
        or assessment["min_ordinary_category_probability"] <= 1e-12
        or jac["jacobian_condition_number"] > 1e12
        or float(se.max()) > 5
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
        "max_robust_SE": float(se.max()),
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


def make_report(model: str, spec: dict, diagnostics: pd.DataFrame, globals_df: pd.DataFrame, comparison_note: str) -> str:
    boundary = diagnostics.loc[diagnostics["boundary_solution_flag"] | diagnostics["boundary_review_required"], "Domain"].tolist()
    display = globals_df.copy()
    cols = list(display.columns)
    table_lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in display.iterrows():
        table_lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in cols) + " |")
    global_table = "\n".join(table_lines)
    return "\n".join([
        f"# Step 3 Independent Refit — {model}",
        "",
        "> **PRELIMINARY / EXPLORATORY ONLY.**",
        "",
        f"- Frozen data: `{FREEZE_ID}`; {EXPECTED_PARTICIPANTS} participants / {EXPECTED_VISITS} visits.",
        f"- Independent process PID: {os.getpid()}.",
        f"- Outcome: {spec.get('outcome_label', 'FIBSER burden 0/1/2/>=3')}; counts {spec.get('outcome_counts', EXPECTED_COUNTS)}.",
        f"- Parameters per domain: {spec['expected_parameters']} (mixing {spec['expected_g']} + ordinal {spec['expected_x']} + thresholds {spec.get('q', Q)}).",
        "- Eleven prespecified starts were completed for every domain before refinement and candidate selection.",
        "- Sandwich covariance: `V = A_inv @ B @ A_inv.T`.",
        "- Education reference: College or above; site/province excluded.",
        "- Concurrent HAMD-17: " + (f"included in both parts; HAMD-only constants N={HAMD_ONLY_N}, mean={HAMD_ONLY_MEAN:.15g}, SD(ddof=0)={HAMD_ONLY_SD_DDOF0:.15g}." if spec["concurrent_hamd17"] else "completely absent from both design matrices and parameter maps."),
        "",
        "## Numerical status",
        "",
        f"- Selected usable solutions: {int(diagnostics['selected_solution_usable_preliminary'].sum())}/9.",
        f"- Selected boundary/boundary-review domains: {', '.join(boundary) if boundary else 'none'}.",
        "- All candidates are retained in `05_all_multistart_diagnostics.csv`; strict-valid selection candidates are additionally retained in `08_boundary_selection_audit.csv`.",
        "",
        "## Global interactions",
        "",
        global_table,
        "",
        "## Comparison scope",
        "",
        comparison_note,
        "",
    ])


def run_independent(model: str, spec: dict, out: Path) -> None:
    if out.exists():
        raise RuntimeError(f"STOP: independent output directory already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    q = int(spec.get("q", Q))
    outcome_column = str(spec.get("outcome_column", "burden_ord4"))
    base.engine.Q = q
    base.engine.SCORE_TOL = SCORE_TOL
    log(model, f"starting clean process PID={os.getpid()} output={out}")
    df = load_frozen_data(out, spec)
    design = preflight_designs(model, spec, df, out)
    y = df[outcome_column].to_numpy(int)
    clusters = df["participant_ID"].to_numpy()
    fits: list[pd.DataFrame] = []
    selections: list[pd.DataFrame] = []
    extremes: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    params: list[pd.DataFrame] = []
    global_rows: list[dict] = []
    effect_rows: list[dict] = []
    probability_rows: list[dict] = []
    for domain in b1.DOMAINS:
        var = b1.DOMAIN_VAR[domain]
        rhs = common.rhs_for(spec["rhs"], var)
        gdf, xdf = b1.make_design(df, rhs, rhs)
        names = parameter_names(gdf, xdf, q)
        theta, fit_rows, selection, extreme = fit_domain(model, spec, domain, y, clusters, gdf, xdf, out)
        fit_rows.insert(0, "Domain", domain)
        fit_rows.insert(0, "Model", model)
        fits.append(fit_rows)
        selections.append(selection)
        extremes.append(extreme)
        covariance, diag, param = covariance_diagnostics(model, domain, theta, y, gdf, xdf, clusters, q)
        diag["boundary_review_required"] = bool(selection["boundary_review_required"].iloc[0])
        diag["highest_strict_objective"] = float(selection["highest_strict_objective"].iloc[0])
        diag["selected_regular_objective"] = float(selection["selected_regular_objective"].iloc[0])
        diag["objective_gap_weak_minus_selected_regular"] = diag["highest_strict_objective"] - diag["selected_regular_objective"]
        if not diag["selected_solution_usable_preliminary"]:
            raise RuntimeError(f"STOP: selected solution failed inference gate for {model}/{domain}")
        diagnostics.append(diag)
        param.insert(0, "Domain", domain)
        param.insert(0, "Model", model)
        params.append(param)
        global_rows.extend(common.extract_global(model, domain, var, theta, covariance, names))
        effect_rows.extend(common.extract_time_effects(model, domain, var, theta, covariance, names))
        probability_rows.extend(adjusted_probabilities(model, domain, var, theta, gdf, xdf))
        log(model, f"completed domain {domain}")

    diagnostics_df = pd.DataFrame(diagnostics)
    globals_df = common.add_fdr(pd.DataFrame(global_rows))
    effects_df = pd.DataFrame(effect_rows)
    probabilities_df = pd.DataFrame(probability_rows)
    fits_df = pd.concat(fits, ignore_index=True)
    selections_df = pd.concat(selections, ignore_index=True)
    extremes_df = pd.concat(extremes, ignore_index=True)
    params_df = pd.concat(params, ignore_index=True)
    if len(globals_df) != 18 or len(effects_df) != 72 or len(probabilities_df) != 36 or len(diagnostics_df) != 9:
        raise RuntimeError("STOP: final result cardinality mismatch")
    if not (fits_df.groupby("Domain")["fit_number"].apply(lambda s: set(range(1, 12)).issubset(set(s))).all()):
        raise RuntimeError("STOP: at least one domain did not complete all 11 starts")
    if effects_df[["OR", "CI_low", "CI_high", "raw_Wald_P"]].isna().any().any():
        raise RuntimeError("STOP: OR/CI/P mapping produced missing values")
    if not np.allclose(probabilities_df[["mean_probability_sum_absent", "mean_probability_sum_present"]], 1.0, atol=1e-10):
        raise RuntimeError("STOP: predicted probabilities do not sum to one")

    diagnostics_df.to_csv(out / "04_model_diagnostics.csv", index=False, encoding="utf-8-sig")
    fits_df.to_csv(out / "05_all_multistart_diagnostics.csv", index=False, encoding="utf-8-sig")
    params_df.to_csv(out / "06_selected_parameter_estimates.csv", index=False, encoding="utf-8-sig")
    extremes_df.to_csv(out / "07_extreme_mixing_parameters_by_start.csv", index=False, encoding="utf-8-sig")
    selections_df.to_csv(out / "08_boundary_selection_audit.csv", index=False, encoding="utf-8-sig")
    globals_df.to_csv(out / "09_global_interactions.csv", index=False, encoding="utf-8-sig")
    effects_df.to_csv(out / "10_time_specific_ORs.csv", index=False, encoding="utf-8-sig")
    probabilities_df.to_csv(out / "11_adjusted_probabilities.csv", index=False, encoding="utf-8-sig")
    common.write_excel(out / "Step3_independent_results.xlsx", {
        "design_checks": design,
        "model_diagnostics": diagnostics_df,
        "global_interactions": globals_df,
        "time_specific_ORs": effects_df,
        "adjusted_probabilities": probabilities_df,
        "boundary_selection": selections_df,
    })
    comparison_note = "Comparison with the prior combined-process run is generated only after both independent processes complete."
    (out / "Step3_independent_report.md").write_text(make_report(model, spec, diagnostics_df, globals_df, comparison_note), encoding="utf-8")
    _write_json(out / "00_run_manifest.json", {
        "status": "COMPLETE_PRELIMINARY",
        "inference_label": "PRELIMINARY",
        "freeze_id": FREEZE_ID,
        "freeze_sha256": base.sha256(base.FREEZE),
        "process_id": os.getpid(),
        "elapsed_seconds": time.time() - started,
        "model": model,
        "domains": 9,
        "prespecified_starts_per_domain": 11,
        "refinement_per_domain": 1,
        "parameters_per_domain": spec["expected_parameters"],
        "outcome_column": outcome_column,
        "outcome_label": spec.get("outcome_label"),
        "outcome_counts": spec.get("outcome_counts"),
        "thresholds": q,
        "covariance_formula": "V = A_inv @ B @ A_inv.T",
        "concurrent_hamd17": spec["concurrent_hamd17"],
        "hamd_only_reference_n": HAMD_ONLY_N if spec["concurrent_hamd17"] else None,
        "hamd_only_mean": HAMD_ONLY_MEAN if spec["concurrent_hamd17"] else None,
        "hamd_only_sd_ddof0": HAMD_ONLY_SD_DDOF0 if spec["concurrent_hamd17"] else None,
        "all_selected_solutions_usable": bool(diagnostics_df["selected_solution_usable_preliminary"].all()),
    })
    log(model, f"COMPLETE_PRELIMINARY elapsed={time.time() - started:.1f}s")





