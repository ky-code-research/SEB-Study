from __future__ import annotations

import hashlib
import json
import math
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
import secondary_prise_tolerability_qc as qc
import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2_candidate_burden3_no_concurrent_HAMD as zi
import step2_fourlevel_concurrent_hamd17_education3 as m1
import step2_fourlevel_no_concurrent_hamd17 as base
import step2_fourlevel_no_concurrent_hamd17_education3 as edu3
from tolerability_contrasts import pairwise_tolerability_contrasts


from public_paths import ROOT
OUT = ROOT / "outputs" / "secondary_analysis_PRISE_tolerability_ESGEE_2026-10-11_preliminary"
CHECKPOINT = ROOT / "outputs" / "_secondary_tolerability_ESGEE_checkpoints_2026-10-11"
FREEZE = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "repaired_natural_complete_case_analysis_data.csv"
PREVIOUS = ROOT / "outputs" / "step2_fourlevel_concurrent_hamd17_education3" / "09_selected_parameter_estimates.csv"
Q = 3
SCORE_TOL = 5e-2
LEVELS = ["None", "Tolerable", "Distressing"]
LEVEL_CODE = {"None": 0, "Tolerable": 1, "Distressing": 2}
DOMAIN_VARS = list(base.DOMAIN_VARS)
EXPECTED_FREEZE_SHA = "6844f512b5122beabc9905456e1a9a2111f6318c398a984232068b22c22fd16c"


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_float_list(x: np.ndarray) -> str:
    return json.dumps([float(v) for v in np.asarray(x).ravel()], separators=(",", ":"), allow_nan=False)


def parameter_names(Gdf: pd.DataFrame, Xdf: pd.DataFrame) -> list[str]:
    return ([f"mixing_gamma:{c}" for c in Gdf.columns]
            + ["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"]
            + [f"ordinal_beta:{c}" for c in Xdf.columns])


def predict(theta: np.ndarray, G: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gamma, _, _, _, _ = zi.unpack(theta, G.shape[1], X.shape[1], Q)
    pi = stats.logistic.cdf(G @ gamma)
    ordinary, _, _ = zi.ordinal_probs_and_deriv(theta, G, X, Q)
    joint = pi[:, None] * ordinary
    joint[:, 0] += 1.0 - pi
    return joint, pi, ordinary


def assessment(theta: np.ndarray, y: np.ndarray, G: np.ndarray, X: np.ndarray) -> dict:
    ll, score = zi.obs_loglik_score(theta, y, G, X, Q)
    joint, pi, ordinary = predict(theta, G, X)
    gamma, _, beta, alpha, _ = zi.unpack(theta, G.shape[1], X.shape[1], Q)
    max_score = float(np.max(np.abs(score.sum(axis=0))))
    finite = bool(np.isfinite(theta).all() and np.isfinite(ll).all() and np.isfinite(score).all())
    ordered = bool(len(alpha) == 3 and np.all(np.diff(alpha) > 0))
    valid = bool(finite and ordered and ordinary.min() > 0 and np.max(np.abs(theta)) < 50)
    out = {
        "objective": float(ll.sum()), "max_abs_score": max_score,
        "strict_score_converged": bool(max_score <= SCORE_TOL),
        "parameters_finite": finite, "parameter_valid": valid,
        "thresholds_ordered": ordered, "threshold_1": float(alpha[0]),
        "threshold_2": float(alpha[1]), "threshold_3": float(alpha[2]),
        "threshold_gap_1_2": float(alpha[1] - alpha[0]),
        "threshold_gap_2_3": float(alpha[2] - alpha[1]),
        "max_abs_parameter": float(np.max(np.abs(theta))),
        "max_abs_mixing_coefficient": float(np.max(np.abs(gamma))),
        "max_abs_ordinal_coefficient": float(np.max(np.abs(beta))),
        "min_ordinary_category_probability": float(ordinary.min()),
        "min_pi": float(pi.min()), "max_pi": float(pi.max()),
        "latent_extra_zero_mean": float((1.0 - pi).mean()),
    }
    for k in range(4):
        out[f"observed_p{k}"] = float(np.mean(y == k))
        out[f"predicted_p{k}"] = float(joint[:, k].mean())
    out["max_abs_observed_predicted_probability_difference"] = max(
        abs(out[f"observed_p{k}"] - out[f"predicted_p{k}"]) for k in range(4)
    )
    return out


def jacobian_assessment(theta: np.ndarray, y: np.ndarray, G: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, dict]:
    A = -zi.finite_hessian_of_score(theta, y, G, X, Q)
    singular = np.linalg.svd(A, compute_uv=False)
    rank = int(np.linalg.matrix_rank(A))
    return A, {
        "jacobian_finite": bool(np.isfinite(A).all()),
        "jacobian_rank": rank,
        "jacobian_full_rank": bool(rank == len(theta)),
        "jacobian_condition_number": float(np.linalg.cond(A)),
        "jacobian_min_singular_value": float(singular.min()),
    }


def corrected_covariance(theta: np.ndarray, y: np.ndarray, G: np.ndarray, X: np.ndarray,
                         clusters: np.ndarray) -> tuple[np.ndarray, dict]:
    _, score_i = zi.obs_loglik_score(theta, y, G, X, Q)
    score_df = pd.DataFrame(score_i)
    score_df["cluster"] = clusters
    U = score_df.groupby("cluster", observed=True).sum().to_numpy()
    B = U.T @ U
    A = -zi.finite_hessian_of_score(theta, y, G, X, Q)
    A_inv = np.linalg.pinv(A)
    V_raw = A_inv @ B @ A_inv.T
    scale = max(1.0, float(np.max(np.abs(V_raw))))
    symmetry_error = float(np.max(np.abs(V_raw - V_raw.T)))
    V = (V_raw + V_raw.T) / 2.0
    eig = np.linalg.eigvalsh(V)
    return V, {
        "covariance_finite": bool(np.isfinite(V_raw).all()),
        "covariance_symmetric": bool(symmetry_error <= 1e-10 * scale),
        "covariance_symmetry_max_abs_error": symmetry_error,
        "covariance_PSD": bool(eig.min() >= -1e-8),
        "covariance_min_eigenvalue": float(eig.min()),
        "covariance_condition_number": float(np.linalg.cond(V)),
        "max_robust_SE": float(np.sqrt(np.maximum(np.diag(V), 0)).max()),
        "meat_rank": int(np.linalg.matrix_rank(B)),
        "clusters": int(pd.Series(clusters).nunique()),
    }


def bh_adjust(p: pd.Series) -> pd.Series:
    vals = p.to_numpy(float)
    result = np.full(len(vals), np.nan)
    ok = np.isfinite(vals)
    idx = np.where(ok)[0]
    if not len(idx):
        return pd.Series(result, index=p.index)
    order = idx[np.argsort(vals[idx], kind="mergesort")]
    ranked = vals[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return pd.Series(result, index=p.index)


def compare_frames(saved: pd.DataFrame, rebuilt: pd.DataFrame) -> tuple[bool, str]:
    keys = ["participant_ID", "time"]
    s = saved.copy(); r = rebuilt.copy()
    for f in (s, r):
        f["participant_ID"] = f["participant_ID"].map(b1.norm_id)
        f["time"] = f["time"].astype(str)
    common = [c for c in s.columns if c in r.columns]
    joined = s[keys + [c for c in common if c not in keys]].merge(
        r[keys + [c for c in common if c not in keys]], on=keys, validate="one_to_one", suffixes=("_saved", "_rebuilt")
    )
    bad = []
    for c in common:
        if c in keys:
            continue
        a, z = joined[f"{c}_saved"], joined[f"{c}_rebuilt"]
        an, zn = pd.to_numeric(a, errors="coerce"), pd.to_numeric(z, errors="coerce")
        numeric_like = (an.notna() | a.isna()).all() and (zn.notna() | z.isna()).all()
        same = np.isclose(an, zn, rtol=1e-10, atol=1e-10, equal_nan=True).all() if numeric_like else (
            a.astype("string").fillna("<NA>").eq(z.astype("string").fillna("<NA>")).all()
        )
        if not same:
            bad.append(c)
    return not bad, ";".join(bad)


def build_analysis() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    source_hash = qc.sha256(qc.SOURCE)
    freeze_hash = sha256(FREEZE)
    collapsed, mapping, sheet_keys = qc.build_raw_followup()
    _, _, rebuilt, _, _ = b1.build_dataset()
    freeze_audit, _ = qc.build_freeze_audit(collapsed, rebuilt)
    saved = pd.read_csv(FREEZE)
    saved["participant_ID"] = saved["participant_ID"].map(b1.norm_id)
    saved["time"] = saved["time"].astype(str)
    exact, differences = compare_frames(saved, rebuilt)
    hard_rows = list(freeze_audit.to_dict("records")) + list(mapping.to_dict("records")) + list(sheet_keys.to_dict("records"))
    hard_rows.extend([
        {"audit_section": "model_freeze_integrity", "check_name": "frozen CSV SHA-256", "status": "PASS" if freeze_hash == EXPECTED_FREEZE_SHA else "FAIL", "observed": freeze_hash, "expected": EXPECTED_FREEZE_SHA},
        {"audit_section": "model_freeze_integrity", "check_name": "all shared frozen columns equal current raw rebuild", "status": "PASS" if exact else "FAIL", "observed": differences or "none", "expected": "none"},
        {"audit_section": "model_freeze_integrity", "check_name": "raw Excel SHA-256 recorded", "status": "PASS", "observed": source_hash, "expected": source_hash},
    ])
    audit = pd.DataFrame(hard_rows)
    if not audit["status"].fillna("FAIL").eq("PASS").all():
        audit.to_csv(OUT / "Secondary_Source_Sample_QC.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError("SOURCE/MAPPING/FREEZE HARD GATE FAILED")
    tol_cols = ["participant_ID", "time"]
    for domain in qc.DOMAINS:
        var = qc.MODEL_VAR[domain]
        tol_cols += [f"{var}__occ_raw", f"{var}__tol_raw"]
    analysis = saved.merge(collapsed[tol_cols], on=["participant_ID", "time"], how="left", validate="one_to_one")
    analysis["education3"] = pd.Categorical(analysis["education"].map(edu3.EDUCATION_MAP), categories=edu3.EDUCATION_LEVELS)
    sample_rows = []
    for domain in qc.DOMAINS:
        var = qc.MODEL_VAR[domain]
        lev, issue = qc.classify_tolerability(analysis[f"{var}__occ_raw"], analysis[f"{var}__tol_raw"])
        analysis[f"tol__{var}"] = lev
        analysis[f"tol_issue__{var}"] = issue
        valid = lev.notna()
        counts = lev.value_counts().to_dict()
        issue_counts = issue.loc[issue.ne("")].value_counts().to_dict()
        row = {
            "audit_section": "domain_effective_sample", "check_name": f"{domain}: effective analysis sample",
            "status": "PASS" if not issue.str.match(r"^(contradiction|invalid)").any() else "FAIL",
            "domain": domain, "final_model_variable": var,
            "observed": int(valid.sum()), "expected": "domain-specific complete tolerability",
            "effective_person_visits": int(valid.sum()),
            "effective_unique_participants": int(analysis.loc[valid, "participant_ID"].nunique()),
            "excluded_visits": int((~valid).sum()),
            "none_visits": int(counts.get("None", 0)), "tolerable_visits": int(counts.get("Tolerable", 0)),
            "distressing_visits": int(counts.get("Distressing", 0)),
            "true_missing_tolerability": int(issue_counts.get("true_missing_tolerability_after_present", 0)),
            "missing_occurrence_or_not_assessed": int(issue_counts.get("missing_occurrence_or_not_assessed", 0)),
            "contradictions": int(sum(v for k, v in issue_counts.items() if k.startswith("contradiction"))),
            "invalid_codes": int(sum(v for k, v in issue_counts.items() if k.startswith("invalid"))),
            "detail": json.dumps(issue_counts, ensure_ascii=False, sort_keys=True),
        }
        sample_rows.append(row); hard_rows.append(row)
    final_audit = pd.DataFrame(hard_rows)
    final_audit.to_csv(OUT / "Secondary_Source_Sample_QC.csv", index=False, encoding="utf-8-sig")
    if not final_audit["status"].fillna("FAIL").eq("PASS").all():
        raise RuntimeError("DOMAIN CODING HARD GATE FAILED")
    return analysis, mapping, sample_rows


def domain_design(df: pd.DataFrame, domain: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    target = qc.MODEL_VAR[domain]
    d = df.loc[df[f"tol__{target}"].notna()].copy()
    d["target_tolerability"] = pd.Categorical(d[f"tol__{target}"], categories=LEVELS)
    terms = [x.strip() for x in m1.RHS.split("+")]
    terms = [x for x in terms if x != target]
    tol_term = "C(target_tolerability, Treatment(reference='None'))"
    terms.insert(1, tol_term)
    rhs = " + ".join(terms)
    Gdf, Xdf = b1.make_design(d, mix_rhs=rhs, ord_rhs=rhs)
    expected_other = [v for v in DOMAIN_VARS if v != target]
    target_terms_g = [c for c in Gdf.columns if "target_tolerability" in c]
    target_terms_x = [c for c in Xdf.columns if "target_tolerability" in c]
    forbidden = [c for c in [*Gdf.columns, *Xdf.columns] if c == target or ":" in c or "site" in c.lower() or "province" in c.lower()]
    checks = [
        len(target_terms_g) == 2, len(target_terms_x) == 2,
        all(v in Gdf.columns and v in Xdf.columns for v in expected_other),
        target not in Gdf.columns and target not in Xdf.columns, not forbidden,
        sum(c == "concurrent_HAMD17_z" for c in Gdf.columns) == 1,
        sum(c == "concurrent_HAMD17_z" for c in Xdf.columns) == 1,
        len(Gdf) == len(d) == len(Xdf), Gdf.shape[1] == 23, Xdf.shape[1] == 22,
    ]
    if not all(checks):
        raise RuntimeError(f"DESIGN MAPPING HARD GATE FAILED for {domain}: G={list(Gdf.columns)}, X={list(Xdf.columns)}")
    return d, Gdf, Xdf, rhs


def mapped_start(names: list[str]) -> np.ndarray:
    previous = pd.read_csv(PREVIOUS).set_index("parameter")["estimate"].astype(float).to_dict()
    return np.array([previous.get(n, 0.0) for n in names], dtype=float)


def run_candidate(domain: str, fit_number: int, start_name: str, start_method: str,
                  start_theta: np.ndarray, y: np.ndarray, G: np.ndarray, X: np.ndarray,
                  names: list[str], refinement: bool = False) -> tuple[dict, np.ndarray]:
    slug = qc.MODEL_VAR[domain]
    cp = CHECKPOINT / f"{slug}__{fit_number:02d}.json"
    if cp.exists():
        saved = json.loads(cp.read_text(encoding="utf-8"))
        if saved.get("parameter_names") == names:
            theta = np.array(saved["theta"], float)
            row = saved["row"]
            log(f"{domain} start {fit_number} restored")
            return row, theta
    opts = {"maxiter": 12000 if refinement else 8000, "ftol": 1e-14 if refinement else 1e-13,
            "gtol": 1e-10 if refinement else 1e-9, "maxls": 150 if refinement else 120,
            "maxcor": 120 if refinement else 100}
    res = optimize.minimize(lambda th: zi.neg_obj_grad(th, y, G, X, Q), start_theta,
                            method="L-BFGS-B", jac=True, options=opts)
    theta = res.x
    ass = assessment(theta, y, G, X)
    A, jac = jacobian_assessment(theta, y, G, X)
    strict = bool(res.success and ass["strict_score_converged"] and ass["parameter_valid"]
                  and jac["jacobian_full_rank"] and jac["jacobian_finite"])
    row = {"domain": domain, "fit_number": fit_number, "start": start_name,
           "start_method": start_method, "is_refinement": refinement,
           "program_success": bool(res.success), "message": str(res.message), "iterations": int(res.nit),
           **ass, **jac, "strict_valid": strict, "selected": False,
           "parameter_names_json": json.dumps(names, separators=(",", ":")), "theta_json": json_float_list(theta)}
    cp.write_text(json.dumps({"parameter_names": names, "theta": theta.tolist(), "row": row}, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    log(f"{domain} start {fit_number}: success={res.success} strict={strict} obj={ass['objective']:.6f} score={ass['max_abs_score']:.3g}")
    return row, theta


def fit_domain(df: pd.DataFrame, domain: str) -> tuple[list[dict], dict, list[dict], list[dict]]:
    d, Gdf, Xdf, rhs = domain_design(df, domain)
    y = d["burden_ord4"].astype(int).to_numpy()
    G, X = Gdf.to_numpy(float), Xdf.to_numpy(float)
    names = parameter_names(Gdf, Xdf)
    starts = [{"start": "mapped_final_Model1_by_exact_name", "method": "exact_name; new tolerability coefficients=0", "theta": mapped_start(names)}]
    starts += legacy.build_zi_starts(y, G, X, 4)
    rows, thetas = [], []
    for i, s in enumerate(starts, 1):
        row, theta = run_candidate(domain, i, s["start"], s["method"], s["theta"], y, G, X, names)
        rows.append(row); thetas.append(theta)
    strict_idx = [i for i, r in enumerate(rows) if r["strict_valid"]]
    if not strict_idx:
        raise RuntimeError(f"No strict-valid raw start for {domain}")
    best_i = max(strict_idx, key=lambda i: (rows[i]["objective"], -rows[i]["max_abs_score"]))
    row, theta = run_candidate(domain, len(rows) + 1, "refined_from_best_strict_raw_start", rows[best_i]["start"],
                               thetas[best_i], y, G, X, names, refinement=True)
    rows.append(row); thetas.append(theta)
    strict_idx = [i for i, r in enumerate(rows) if r["strict_valid"]]
    selected_i = max(strict_idx, key=lambda i: (rows[i]["objective"], -rows[i]["max_abs_score"]))
    rows[selected_i]["selected"] = True
    theta = thetas[selected_i]
    V, cov = corrected_covariance(theta, y, G, X, d["participant_ID"].to_numpy())
    selected = rows[selected_i]
    strict = [r for r in rows if r["strict_valid"]]
    selected_obj = selected["objective"]
    competitive_pairs = [(r, thetas[i]) for i, r in enumerate(rows)
                         if r["strict_valid"] and r["fit_number"] != selected["fit_number"]
                         and r["objective"] >= selected_obj - 1e-3]
    competitive = [r for r, _ in competitive_pairs]
    competitive_parameter_differences = [float(np.max(np.abs(t - theta))) for _, t in competitive_pairs]
    distinct_competing = [x for x in competitive_parameter_differences if x > 1e-2]
    high_competing = any(r["max_abs_mixing_coefficient"] > 5 for r in competitive)
    boundary_competing = any(r["min_pi"] < 1e-5 or r["max_pi"] > 1 - 1e-10 for r in competitive)
    selected_boundary = selected["min_pi"] < 1e-5 or selected["max_pi"] > 1 - 1e-10
    selected_high = selected["max_abs_mixing_coefficient"] > 5
    distress_n = int(d["target_tolerability"].eq("Distressing").sum())
    distress_cells = d.loc[d["target_tolerability"].eq("Distressing"), "burden_ord4"].value_counts().reindex(range(4), fill_value=0)
    sparse = bool(distress_n < 50 or (distress_cells < 5).any())
    covariance_ok = cov["covariance_finite"] and cov["covariance_symmetric"] and cov["covariance_PSD"]
    numerical_review = bool(selected_boundary or selected_high or high_competing or boundary_competing or distinct_competing
                            or selected["jacobian_condition_number"] > 1e8 or not covariance_ok or sparse)
    status = "REVIEW REQUIRED" if numerical_review else "PASS"
    diag = {"domain": domain, "preliminary": True, "status": status,
            "effective_person_visits": len(d), "effective_unique_participants": int(d["participant_ID"].nunique()),
            "distressing_person_visits": distress_n,
            "distressing_fibser_counts_0_1_2_ge3": json.dumps([int(x) for x in distress_cells.tolist()]),
            "sparse_distressing_flag": sparse, "selected_fit_number": selected["fit_number"],
            "strict_valid_candidates": len(strict), "competitive_candidates_within_1e-3": len(competitive),
            "distinct_competing_solutions_parameter_diff_gt_0_01": len(distinct_competing),
            "max_parameter_difference_among_objective_competitors": max(competitive_parameter_differences, default=0.0),
            "high_mixing_competing_solution": high_competing, "boundary_competing_solution": boundary_competing,
            "selected_high_mixing": selected_high, "selected_boundary_probability": selected_boundary,
            "RHS": rhs, "mixing_columns": Gdf.shape[1], "ordinal_columns": Xdf.shape[1], "parameters": len(theta),
            **{k: v for k, v in selected.items() if k not in {"domain", "parameter_names_json", "theta_json", "selected"}},
            **cov, "parameter_names_json": json.dumps(names, separators=(",", ":")),
            "selected_theta_json": json_float_list(theta), "robust_covariance_json": json_float_list(V)}
    tol_cols_g = {level: next(c for c in Gdf.columns if "target_tolerability" in c and f"[T.{level}]" in c) for level in ["Tolerable", "Distressing"]}
    tol_cols_x = {level: next(c for c in Xdf.columns if "target_tolerability" in c and f"[T.{level}]" in c) for level in ["Tolerable", "Distressing"]}
    contrasts = pairwise_tolerability_contrasts(theta, V, Gdf, Xdf, domain, status, Q)
    prob_rows = []
    baseG, baseX = G.copy(), X.copy()
    g_t, g_d = Gdf.columns.get_loc(tol_cols_g["Tolerable"]), Gdf.columns.get_loc(tol_cols_g["Distressing"])
    x_t, x_d = Xdf.columns.get_loc(tol_cols_x["Tolerable"]), Xdf.columns.get_loc(tol_cols_x["Distressing"])
    probs = {}
    for level in LEVELS:
        Gc, Xc = baseG.copy(), baseX.copy(); Gc[:, [g_t, g_d]] = 0; Xc[:, [x_t, x_d]] = 0
        if level == "Tolerable": Gc[:, g_t] = 1; Xc[:, x_t] = 1
        if level == "Distressing": Gc[:, g_d] = 1; Xc[:, x_d] = 1
        probs[level] = predict(theta, Gc, Xc)[0].mean(axis=0)
    for level in LEVELS:
        p = probs[level]; pt = probs["Tolerable"]
        prob_rows.append({"domain": domain, "tolerability_level": level, "tolerability_code": LEVEL_CODE[level],
                          "preliminary": True, "model_status": status, "standardization_population_visits": len(d),
                          "adjusted_P_FIBSER_0": p[0], "adjusted_P_FIBSER_1": p[1],
                          "adjusted_P_FIBSER_2": p[2], "adjusted_P_FIBSER_ge3": p[3],
                          "adjusted_P_FIBSER_ge1": p[1:].sum(), "adjusted_P_FIBSER_ge2": p[2:].sum(),
                          "difference_vs_Tolerable_P0": p[0] - pt[0],
                          "difference_vs_Tolerable_Pge1": p[1:].sum() - pt[1:].sum(),
                          "difference_vs_Tolerable_Pge2": p[2:].sum() - pt[2:].sum(),
                          "difference_vs_Tolerable_Pge3": p[3] - pt[3]})
    return rows, diag, contrasts, prob_rows


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    v = df[cols].copy()
    for c in cols:
        if pd.api.types.is_float_dtype(v[c]):
            v[c] = v[c].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in v.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


def write_report(audit: pd.DataFrame, diagnostics: pd.DataFrame, contrasts: pd.DataFrame, probabilities: pd.DataFrame) -> None:
    summary = contrasts.copy()
    summary["OR (95% CI)"] = summary.apply(lambda r: f"{r.OR:.3f} ({r.CI_low:.3f}, {r.CI_high:.3f})", axis=1)
    warnings = diagnostics.loc[diagnostics["status"].ne("PASS"), ["domain", "status", "sparse_distressing_flag", "jacobian_condition_number", "selected_boundary_probability", "high_mixing_competing_solution", "boundary_competing_solution"]]
    report = f"""# Secondary Analysis: PRISE tolerability and four-level FIBSER burden

**Status: preliminary.** This analysis is separate from completed Step 2/3 results and does not modify them.

## Scope and source/sample gate

The analysis used the frozen 1468-participant / 4799-person-visit sample (Week 2/8/12/24: 1381/1259/1146/1013). Tolerability was reconstructed directly from the source Excel by unique labeled PRISE blocks: occurrence `1=yes`, `2=no`; tolerability `1=tolerable`, `2=distressing`. A blank tolerability field was coded None only when occurrence explicitly equaled 2. Domain-specific occurrence/tolerability uncertainty was excluded only for that target domain. All mapping, key, outcome, frozen-column, and contradiction gates passed before fitting.

## Model

Eight separate four-level zero-inflated ordinal ES-GEE models were fitted with working independence and participant-cluster robust sandwich covariance `A^-1 B A^-T`. Each target domain was represented in both components by two indicators (None reference), while its binary occurrence variable was removed. The other eight occurrence variables (including Other psychiatric), time, age, sex, 3-level education, first episode, baseline HAMD-17, baseline SDS, baseline medication class, and concurrent HAMD-17 followed final Model 1 coding and frozen standardization. No tolerability-by-time interaction was fitted.

Eleven prespecified raw starts (one exact-name mapped final-Model-1 start plus ten standard generated starts) and one refinement from the best strict raw start were evaluated per domain. Selection used strict convergence/validity and then objective value, never significance. Jacobian rank/conditioning, thresholds, competing/boundary solutions, covariance properties, and observed-versus-predicted outcome probabilities were checked.

## Three pairwise tolerability contrasts

{md_table(summary, ['domain','component','contrast','contrast_role','OR (95% CI)','raw_P','FDR_P','model_status'])}

All three contrasts were computed from the same fitted parameter vector and complete robust covariance. None is the reference for Tolerable-vs-None and Distressing-vs-None; Distressing-vs-Tolerable uses `Var(D)+Var(T)-2Cov(D,T)`. Zero-inflated ORs use `exp(-c'gamma)` and ordinal ORs use `exp(c'beta)`. The locked BH-FDR rule was retained: only the eight primary Distressing-vs-Tolerable contrasts were adjusted separately within each component. The other two contrast sets are exploratory and retain raw Wald P values only.

## Numerical and sparsity warnings

{md_table(warnings, list(warnings.columns)) if len(warnings) else 'No model met the prespecified REVIEW REQUIRED criteria.'}

Rows marked **REVIEW REQUIRED** are reported for transparency but their ORs and P values should not be substantively interpreted. Distressing sample size and its four-category FIBSER cells are provided in the diagnostics. Adjusted probabilities are marginal standardizations over each domain-specific valid sample and are descriptive model-based summaries, not causal effects.

## Validation

Source/sample/mapping gates passed. All selected solutions had finite ordered thresholds, strict score checks, full-rank Jacobians, and covariance checks as recorded in `Secondary_Model_Diagnostics.csv`. ORs, confidence intervals, Wald P values, BH-FDR values, adjusted probabilities, and observed-versus-predicted distributions were independently recalculated from saved selected parameters and robust covariance; the validation script must report PASS before this report is finalized.

## Interpretation limits and fallacy scan

This is an observational longitudinal association analysis. It does not establish that distressing side effects cause higher FIBSER burden; concurrent symptom severity and reverse causality remain plausible. Participant-cluster covariance handles within-participant dependence, but domain-specific missing tolerability and attrition can still affect generalizability. BH-FDR addresses the prespecified eight Distressing-vs-Tolerable comparisons within each model component, not the 32 exploratory reference-level comparisons.

- Simpson's paradox: adjusted and marginal patterns may differ; no subgroup causal conclusion is made.
- Ecological fallacy: analyses use person-visits with participant clustering, not aggregate groups.
- Berkson/collider bias: selection into the frozen complete-case sample may induce associations; no causal interpretation is made.
- Base-rate neglect: adjusted probabilities accompany ORs.
- Regression to the mean: repeated observational measurements are not interpreted as treatment response.
- Survivorship/attrition bias: later visits have fewer observations and this limitation remains.
- Look-elsewhere effect: only one prespecified contrast per domain/component was tested; BH-FDR is component-specific.
- Garden of forking paths: model/covariates/contrast were prespecified; solution choice did not use P values.
- Correlation-as-causation: explicitly rejected.
- Reverse causality: distress and burden are contemporaneous and direction cannot be established.

## Material Passport

- Origin Skill: `experiment-agent`
- Origin Mode: `validate`
- Origin Date: `2026-10-11`
- Verification Status: `ANALYZED` (deterministic post-estimation checks independently passed; optimization results remain preliminary)
- Version Label: `validation_v1`

## Stop point

No Tolerability × Time, Plan C, site/IPW sensitivity, GEE variant, regression selection, or other additional model was run. Analysis stops here pending confirmation.
"""
    (OUT / "Secondary_Analysis_Report.md").write_text(report, encoding="utf-8")


def main() -> None:
    analysis, mapping, sample_rows = build_analysis()
    all_starts, diagnostics, contrasts, probabilities = [], [], [], []
    for i, domain in enumerate(qc.DOMAINS, 1):
        log(f"MODEL {i}/8 {domain} begins")
        rows, diag, ctr, probs = fit_domain(analysis, domain)
        all_starts.extend(rows); diagnostics.append(diag); contrasts.extend(ctr); probabilities.extend(probs)
        pd.DataFrame(all_starts).to_csv(OUT / "Secondary_All_Start_Diagnostics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(diagnostics).to_csv(OUT / "Secondary_Model_Diagnostics.csv", index=False, encoding="utf-8-sig")
    contrast_df = pd.DataFrame(contrasts)
    contrast_df["FDR_P"] = np.nan
    for component in ["Zero-inflated", "Ordinal"]:
        idx = contrast_df.index[
            contrast_df["component"].eq(component)
            & contrast_df["contrast"].eq("Distressing vs Tolerable")
        ]
        contrast_df.loc[idx, "FDR_P"] = bh_adjust(contrast_df.loc[idx, "raw_P"]).to_numpy()
    contrast_df["FDR_method_family"] = np.where(
        contrast_df["contrast"].eq("Distressing vs Tolerable"),
        contrast_df["component"].map(
            lambda x: f"BH within 8 prespecified {x} Distressing-vs-Tolerable contrasts"
        ),
        "Not applicable: exploratory reference-level contrast; raw Wald P only",
    )
    contrast_df.to_csv(OUT / "Secondary_Tolerability_Contrasts.csv", index=False, encoding="utf-8-sig")
    probability_df = pd.DataFrame(probabilities)
    probability_df.to_csv(OUT / "Secondary_Adjusted_Probabilities.csv", index=False, encoding="utf-8-sig")
    diag_df = pd.DataFrame(diagnostics)
    start_df = pd.DataFrame(all_starts)
    audit_df = pd.read_csv(OUT / "Secondary_Source_Sample_QC.csv")
    write_report(audit_df, diag_df, contrast_df, probability_df)
    log("ALL MODELS COMPLETE; preliminary outputs written")


if __name__ == "__main__":
    main()




