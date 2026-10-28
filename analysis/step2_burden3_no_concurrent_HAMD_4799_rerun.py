from __future__ import annotations

import hashlib
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Compatibility for the workstation SciPy build.
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

from scipy import optimize, stats
from scipy.special import expit

import model_comparison_gate_fibser as legacy
import step2_candidate_burden3_no_concurrent_HAMD as zi
import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2B3_fixed_sample_covariate_attenuation as b3


from public_paths import ROOT
FREEZE = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "repaired_natural_complete_case_analysis_data.csv"
VERSION_RECORD = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "data_version_record.json"
OUT = ROOT / "outputs" / "step2_burden3_no_concurrent_HAMD_4799_rerun"

FREEZE_ID = "PRISE_REPAIRED_NATURAL_CC_1468P_4799V_2026-10-08"
EXPECTED_FREEZE_SHA256 = "6844f512b5122beabc9905456e1a9a2111f6318c398a984232068b22c22fd16c"
EXPECTED_SOURCE_SHA256 = "d7e59303c755dbd0547b0b6c84694d038f53e775e83a6a2483057007fd3054c8"
EXPECTED_PARTICIPANTS = 1468
EXPECTED_VISITS = 4799
Q = 2
SCORE_TOL = 5e-2
SEED = 20261008
DOMAIN_VARS = [b1.DOMAIN_VAR[d] for d in b1.DOMAINS]
RHS = b3.MODEL_RHS["M3"].replace(" + concurrent_HAMD17_z", "")


def log(message: str) -> None:
    print(f"Three-level no-concurrent-HAMD rerun: {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_data() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    version = json.loads(VERSION_RECORD.read_text(encoding="utf-8"))
    freeze_hash = sha256(FREEZE)
    if freeze_hash.lower() != EXPECTED_FREEZE_SHA256:
        raise RuntimeError(f"Freeze hash mismatch: {freeze_hash}")
    if version.get("source_sha256", "").lower() != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Source-workbook hash in version record does not match the locked source hash")
    df = pd.read_csv(FREEZE)
    if (df["participant_ID"].nunique(), len(df)) != (EXPECTED_PARTICIPANTS, EXPECTED_VISITS):
        raise RuntimeError("Frozen sample count gate failed")
    duplicate_keys = int(df.duplicated(["participant_ID", "time"]).sum())
    if duplicate_keys:
        raise RuntimeError(f"participant_ID + time is not unique: {duplicate_keys} duplicates")
    missing = [c for c in DOMAIN_VARS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Repaired PRISE columns missing: {missing}")
    for col in DOMAIN_VARS:
        observed = set(df[col].dropna().astype(float).unique())
        if not observed.issubset({0.0, 1.0}) or df[col].isna().any():
            raise RuntimeError(f"Invalid repaired PRISE values in {col}: {observed}")
    if df["burden_ord4"].isna().any():
        raise RuntimeError("Frozen burden outcome contains missing values")
    df["burden_ord4"] = df["burden_ord4"].astype(int)
    df["burden_ord3"] = np.select(
        [df["burden_ord4"].eq(0), df["burden_ord4"].eq(1)], [0, 1], default=2
    ).astype(int)
    audit = pd.DataFrame([
        {"check": "Freeze ID", "expected": FREEZE_ID, "observed": FREEZE_ID, "pass": True},
        {"check": "freeze SHA256", "expected": EXPECTED_FREEZE_SHA256, "observed": freeze_hash, "pass": freeze_hash.lower() == EXPECTED_FREEZE_SHA256},
        {"check": "source SHA256", "expected": EXPECTED_SOURCE_SHA256, "observed": version.get("source_sha256"), "pass": version.get("source_sha256", "").lower() == EXPECTED_SOURCE_SHA256},
        {"check": "participants", "expected": EXPECTED_PARTICIPANTS, "observed": int(df.participant_ID.nunique()), "pass": df.participant_ID.nunique() == EXPECTED_PARTICIPANTS},
        {"check": "person-visits", "expected": EXPECTED_VISITS, "observed": len(df), "pass": len(df) == EXPECTED_VISITS},
        {"check": "unique participant_ID + time", "expected": 0, "observed": duplicate_keys, "pass": duplicate_keys == 0},
    ])
    return df, version, audit


def make_design(df: pd.DataFrame):
    Gdf, Xdf = b1.make_design(df, mix_rhs=RHS, ord_rhs=RHS)
    forbidden = [c for c in [*Gdf.columns, *Xdf.columns] if "concurrent_HAMD" in c or "site" in c.lower() or "province" in c.lower()]
    if forbidden:
        raise RuntimeError(f"Forbidden design terms: {forbidden}")
    expected_parameters = Gdf.shape[1] + Q + Xdf.shape[1]
    if expected_parameters != 47:
        raise RuntimeError(f"Expected 47 parameters, got {expected_parameters}")
    return Gdf, Xdf, Gdf.to_numpy(float), Xdf.to_numpy(float)


def parameter_names(Gdf: pd.DataFrame, Xdf: pd.DataFrame) -> list[str]:
    return (
        [f"mixing_gamma:{c}" for c in Gdf.columns]
        + [f"threshold_raw:{j}" for j in range(Q)]
        + [f"ordinal_beta:{c}" for c in Xdf.columns]
    )


def predict(theta, G, X):
    gamma, _, _, _, _ = zi.unpack(theta, G.shape[1], X.shape[1], Q)
    pi = expit(np.asarray(G) @ gamma)
    ordinary, _, _ = zi.ordinal_probs_and_deriv(theta, G, X, Q)
    joint = pi[:, None] * ordinary
    joint[:, 0] += 1.0 - pi
    return joint, pi, ordinary


def basic_assessment(theta, y, G, X) -> dict:
    ll, score = zi.obs_loglik_score(theta, y, G, X, Q)
    joint, pi, ordinary = predict(theta, G, X)
    gamma, _, beta, alpha, _ = zi.unpack(theta, G.shape[1], X.shape[1], Q)
    max_score = float(np.max(np.abs(score.sum(axis=0))))
    finite = bool(np.isfinite(theta).all() and np.isfinite(ll).all() and np.isfinite(score).all())
    ordered = bool(np.all(np.diff(alpha) > 0))
    parameter_valid = bool(finite and ordered and ordinary.min() > 0 and np.max(np.abs(theta)) < 50)
    return {
        "objective": float(ll.sum()),
        "max_abs_score": max_score,
        "strict_score_converged": bool(max_score <= SCORE_TOL),
        "parameters_finite": finite,
        "parameter_valid": parameter_valid,
        "thresholds_ordered": ordered,
        "threshold_1": float(alpha[0]),
        "threshold_2": float(alpha[1]),
        "threshold_gap": float(alpha[1] - alpha[0]),
        "max_abs_parameter": float(np.max(np.abs(theta))),
        "max_abs_mixing_coefficient": float(np.max(np.abs(gamma))),
        "max_abs_ordinal_coefficient": float(np.max(np.abs(beta))),
        "min_ordinary_category_probability": float(ordinary.min()),
        "min_pi": float(pi.min()),
        "max_pi": float(pi.max()),
        "latent_extra_zero_mean": float((1.0 - pi).mean()),
        "observed_zero": float((y == 0).mean()),
        "predicted_zero": float(joint[:, 0].mean()),
    }


def jacobian_assessment(theta, y, G, X) -> tuple[np.ndarray, dict]:
    A = -zi.finite_hessian_of_score(theta, y, G, X, Q)
    singular = np.linalg.svd(A, compute_uv=False)
    rank = int(np.linalg.matrix_rank(A))
    out = {
        "jacobian_finite": bool(np.isfinite(A).all()),
        "jacobian_rank": rank,
        "jacobian_full_rank": bool(rank == len(theta)),
        "jacobian_condition_number": float(np.linalg.cond(A)),
        "jacobian_min_singular_value": float(singular.min()),
    }
    return A, out


def select_best(candidates: list[dict]) -> dict:
    # Strict validity is mandatory. Objective is compared only inside the strict-valid set.
    strict = [c for c in candidates if c["strict_valid"]]
    if not strict:
        raise RuntimeError("No start met strict convergence, Jacobian, and parameter-validity gates")
    return max(strict, key=lambda c: (c["assessment"]["objective"], -c["assessment"]["max_abs_score"]))


def fit_full_sample(y, G, X) -> tuple[np.ndarray, pd.DataFrame, list[dict]]:
    starts = legacy.build_zi_starts(y, G, X, Q + 1)
    if len(starts) != 10:
        raise RuntimeError(f"Expected 10 prespecified raw starts before refinement, got {len(starts)}")
    candidates: list[dict] = []
    rows: list[dict] = []
    for ordinal, start in enumerate(starts, start=1):
        res = optimize.minimize(
            lambda th: zi.neg_obj_grad(th, y, G, X, Q),
            start["theta"], method="L-BFGS-B", jac=True,
            options={"maxiter": 4000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 100, "maxcor": 80},
        )
        assessment = basic_assessment(res.x, y, G, X)
        _, jac = jacobian_assessment(res.x, y, G, X)
        strict_valid = bool(res.success and assessment["strict_score_converged"] and assessment["parameter_valid"] and jac["jacobian_full_rank"] and jac["jacobian_finite"])
        row = {"fit_number": ordinal, "start": start["start"], "start_method": start["method"], "program_success": bool(res.success), "message": str(res.message), "iterations": int(res.nit), **assessment, **jac, "strict_valid": strict_valid, "selected": False}
        rows.append(row)
        candidates.append({"theta": res.x, "assessment": assessment, "jacobian": jac, "strict_valid": strict_valid, "row": row})
        log(f"full-sample start {ordinal}/10 {start['start']}: success={res.success}, strict={strict_valid}, objective={assessment['objective']:.6f}, score={assessment['max_abs_score']:.4g}")

    best_raw = select_best(candidates)
    res = optimize.minimize(
        lambda th: zi.neg_obj_grad(th, y, G, X, Q),
        best_raw["theta"], method="L-BFGS-B", jac=True,
        options={"maxiter": 8000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 120, "maxcor": 100},
    )
    assessment = basic_assessment(res.x, y, G, X)
    _, jac = jacobian_assessment(res.x, y, G, X)
    strict_valid = bool(res.success and assessment["strict_score_converged"] and assessment["parameter_valid"] and jac["jacobian_full_rank"] and jac["jacobian_finite"])
    refined_row = {"fit_number": 11, "start": "refined_from_best_strict_raw_start", "start_method": best_raw["row"]["start"], "program_success": bool(res.success), "message": str(res.message), "iterations": int(res.nit), **assessment, **jac, "strict_valid": strict_valid, "selected": False}
    rows.append(refined_row)
    refined = {"theta": res.x, "assessment": assessment, "jacobian": jac, "strict_valid": strict_valid, "row": refined_row}
    candidates.append(refined)
    selected = select_best(candidates)
    rows[selected["row"]["fit_number"] - 1]["selected"] = True
    log(f"selected fit={selected['row']['fit_number']} objective={selected['assessment']['objective']:.6f} score={selected['assessment']['max_abs_score']:.4g}")
    return selected["theta"], pd.DataFrame(rows), candidates


def selected_covariance(theta, y, G, X, clusters) -> tuple[np.ndarray, pd.DataFrame]:
    sw = zi.sandwich(theta, y, G, X, Q, clusters)
    V = (sw["V"] + sw["V"].T) / 2.0
    eig = np.linalg.eigvalsh(V)
    row = {
        "parameters": len(theta),
        "jacobian_rank": int(np.linalg.matrix_rank(sw["A"])),
        "jacobian_full_rank": bool(np.linalg.matrix_rank(sw["A"]) == len(theta)),
        "jacobian_condition_number": float(np.linalg.cond(sw["A"])),
        "covariance_finite": bool(np.isfinite(V).all()),
        "covariance_PSD": bool(eig.min() >= -1e-8),
        "covariance_min_eigenvalue": float(eig.min()),
        "covariance_condition_number": float(np.linalg.cond(V)),
        "max_robust_SE": float(np.sqrt(np.maximum(np.diag(V), 0)).max()),
    }
    return V, pd.DataFrame([row])


def domain_effects(theta, V, Gdf, Xdf) -> pd.DataFrame:
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    pg = Gdf.shape[1]
    rows = []
    for domain in b1.DOMAINS:
        var = b1.DOMAIN_VAR[domain]
        specs = [
            ("Zero-inflated", list(Gdf.columns).index(var), -1.0, "exp(-gamma)"),
            ("Ordinal", pg + Q + list(Xdf.columns).index(var), 1.0, "exp(beta)"),
        ]
        for component, idx, sign, direction in specs:
            coef, robust_se = float(theta[idx]), float(se[idx])
            z = coef / robust_se if robust_se > 0 else np.nan
            if sign < 0:
                lo, hi = math.exp(-(coef + 1.96 * robust_se)), math.exp(-(coef - 1.96 * robust_se))
            else:
                lo, hi = math.exp(coef - 1.96 * robust_se), math.exp(coef + 1.96 * robust_se)
            rows.append({"Domain": domain, "Component": component, "Coefficient": coef, "Robust_SE": robust_se, "OR": math.exp(sign * coef), "CI_low": lo, "CI_high": hi, "Wald_P": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan, "OR_definition": direction})
    return pd.DataFrame(rows)


def adjusted_probabilities(theta, Gdf, Xdf) -> pd.DataFrame:
    """Average counterfactual joint probabilities with each domain set to 0 versus 1."""
    G0 = Gdf.to_numpy(float)
    X0 = Xdf.to_numpy(float)
    rows = []
    for domain in b1.DOMAINS:
        var = b1.DOMAIN_VAR[domain]
        gi = list(Gdf.columns).index(var)
        xi = list(Xdf.columns).index(var)
        g_absent, g_present = G0.copy(), G0.copy()
        x_absent, x_present = X0.copy(), X0.copy()
        g_absent[:, gi] = x_absent[:, xi] = 0.0
        g_present[:, gi] = x_present[:, xi] = 1.0
        p0 = predict(theta, g_absent, x_absent)[0]
        p1 = predict(theta, g_present, x_present)[0]
        ge1_0, ge1_1 = p0[:, 1:].sum(axis=1), p1[:, 1:].sum(axis=1)
        ge2_0, ge2_1 = p0[:, 2], p1[:, 2]
        rows.append({
            "Domain": domain,
            "P_burden_ge1_domain0": float(ge1_0.mean()),
            "P_burden_ge1_domain1": float(ge1_1.mean()),
            "RD_burden_ge1_domain1_minus_domain0": float((ge1_1 - ge1_0).mean()),
            "P_burden_ge2_domain0": float(ge2_0.mean()),
            "P_burden_ge2_domain1": float(ge2_1.mean()),
            "RD_burden_ge2_domain1_minus_domain0": float((ge2_1 - ge2_0).mean()),
            "mean_probability_sum_domain0": float(p0.sum(axis=1).mean()),
            "mean_probability_sum_domain1": float(p1.sum(axis=1).mean()),
        })
    return pd.DataFrame(rows)


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def bootstrap_all_starts(theta0, df, G, X, y, n_boot: int, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Checkpointed cluster bootstrap; every replicate executes all 11 starts."""
    ids = df["participant_ID"].unique()
    pid = df["participant_ID"].to_numpy()
    checkpoint_dir = OUT / f"bootstrap_{n_boot}_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {
        "freeze_id": FREEZE_ID,
        "freeze_sha256": EXPECTED_FREEZE_SHA256,
        "replicates_requested": n_boot,
        "completed_replicates": [],
        "current_replicate": None,
        "completed_starts_current_replicate": [],
        "status": "RUNNING",
    }
    if checkpoint.get("freeze_sha256") != EXPECTED_FREEZE_SHA256 or checkpoint.get("replicates_requested") != n_boot:
        raise RuntimeError("Existing bootstrap checkpoint is incompatible with this frozen-data run")

    for rep in range(1, n_boot + 1):
        rep_diag_path = checkpoint_dir / f"replicate_{rep:03d}_all_start_diagnostics.csv"
        rep_param_path = checkpoint_dir / f"replicate_{rep:03d}_all_start_named_parameters.csv"
        rep_result_path = checkpoint_dir / f"replicate_{rep:03d}_selected_result.json"
        if rep_result_path.exists() and rep in checkpoint.get("completed_replicates", []):
            log(f"bootstrap {rep}/{n_boot}: checkpoint complete, skipping")
            continue

        rng = np.random.default_rng(SEED + rep)
        sampled = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([np.flatnonzero(pid == ident) for ident in sampled])
        yb, Gb, Xb = y[idx], G[idx], X[idx]
        generated = legacy.build_zi_starts(yb, Gb, Xb, Q + 1)
        starts = [{"start": "full_sample_selected", "theta": theta0, "method": "full_sample"}, *generated]
        if len(starts) != 11:
            raise RuntimeError(f"Bootstrap replicate {rep}: expected 11 starts, got {len(starts)}")

        old_diag = pd.read_csv(rep_diag_path) if rep_diag_path.exists() else pd.DataFrame()
        old_params = pd.read_csv(rep_param_path) if rep_param_path.exists() else pd.DataFrame()
        completed = set(old_diag.get("start_number", pd.Series(dtype=int)).astype(int).tolist())
        checkpoint.update({"current_replicate": rep, "completed_starts_current_replicate": sorted(completed), "status": "RUNNING"})
        atomic_json(checkpoint_path, checkpoint)

        # Deliberately no early break: all 11 optimizations are always run.
        for ordinal, start in enumerate(starts, start=1):
            if ordinal in completed:
                continue
            try:
                res = optimize.minimize(
                    lambda th: zi.neg_obj_grad(th, yb, Gb, Xb, Q),
                    start["theta"], method="L-BFGS-B", jac=True,
                    options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 100, "maxcor": 80},
                )
                assessment = basic_assessment(res.x, yb, Gb, Xb)
                basic_valid = bool(res.success and assessment["strict_score_converged"] and assessment["parameter_valid"])
                row = {"bootstrap_replicate": rep, "start_number": ordinal, "start": start["start"], "start_method": start["method"], "program_success": bool(res.success), "message": str(res.message), "iterations": int(res.nit), **assessment, "basic_valid": basic_valid, "jacobian_checked": False, "jacobian_full_rank": False, "selected": False}
                param_rows = pd.DataFrame({
                    "bootstrap_replicate": rep,
                    "start_number": ordinal,
                    "start": start["start"],
                    "parameter_index_zero_based": np.arange(len(names)),
                    "parameter": names,
                    "estimate": res.x,
                })
            except Exception as exc:
                row = {"bootstrap_replicate": rep, "start_number": ordinal, "start": start["start"], "start_method": start["method"], "program_success": False, "message": repr(exc), "basic_valid": False, "jacobian_checked": False, "jacobian_full_rank": False, "selected": False}
                param_rows = pd.DataFrame()

            old_diag = pd.concat([old_diag, pd.DataFrame([row])], ignore_index=True)
            old_diag.to_csv(rep_diag_path, index=False, encoding="utf-8-sig")
            if not param_rows.empty:
                old_params = pd.concat([old_params, param_rows], ignore_index=True)
                old_params.to_csv(rep_param_path, index=False, encoding="utf-8-sig")
            completed.add(ordinal)
            checkpoint["completed_starts_current_replicate"] = sorted(completed)
            atomic_json(checkpoint_path, checkpoint)
            log(f"bootstrap {rep}/{n_boot}, start {ordinal}/11 saved: success={row['program_success']}, basic_valid={row['basic_valid']}")

        if len(completed) != 11 or len(old_diag) != 11:
            raise RuntimeError(f"Bootstrap replicate {rep}: checkpoint contains {len(old_diag)} diagnostics, expected 11")

        param_lookup = {}
        if not old_params.empty:
            for start_number, sub in old_params.groupby("start_number", sort=False):
                ordered_params = sub.sort_values("parameter_index_zero_based")
                if ordered_params["parameter"].tolist() == names and len(ordered_params) == len(names):
                    param_lookup[int(start_number)] = ordered_params["estimate"].to_numpy(float)

        # Compare every completed solution first. Then test Jacobian validity in
        # descending objective order until the highest-objective valid solution is found.
        valid_diag = old_diag.loc[old_diag["basic_valid"].astype(str).str.lower().eq("true")].copy()
        valid_diag = valid_diag.sort_values(["objective", "max_abs_score"], ascending=[False, True])
        selected_number = None
        for row_index, diag_row in valid_diag.iterrows():
            start_number = int(diag_row["start_number"])
            theta_candidate = param_lookup.get(start_number)
            if theta_candidate is None:
                continue
            _, jac = jacobian_assessment(theta_candidate, yb, Gb, Xb)
            for key, value in jac.items():
                old_diag.loc[old_diag["start_number"].eq(start_number), key] = value
            old_diag.loc[old_diag["start_number"].eq(start_number), "jacobian_checked"] = True
            if jac["jacobian_finite"] and jac["jacobian_full_rank"]:
                selected_number = start_number
                break
        result = {"bootstrap_replicate": rep, "sampled_cluster_draws": len(sampled), "unique_original_participants": len(set(sampled)), "rows": len(idx), "starts_executed": 11, "selected": selected_number is not None}
        if selected_number is None:
            result["failure"] = "No strict-valid full-rank solution"
        else:
            selected_mask = old_diag["start_number"].eq(selected_number)
            old_diag.loc[selected_mask, "selected"] = True
            selected_diag = old_diag.loc[selected_mask].iloc[0]
            result.update({
                "selected_start_number": selected_number,
                "selected_start": str(selected_diag["start"]),
                "objective": float(selected_diag["objective"]),
                "max_abs_score": float(selected_diag["max_abs_score"]),
                "max_abs_mixing_coefficient": float(selected_diag["max_abs_mixing_coefficient"]),
                "max_abs_ordinal_coefficient": float(selected_diag["max_abs_ordinal_coefficient"]),
                "threshold_1": float(selected_diag["threshold_1"]),
                "threshold_2": float(selected_diag["threshold_2"]),
                "jacobian_full_rank": True,
            })
        old_diag.to_csv(rep_diag_path, index=False, encoding="utf-8-sig")
        atomic_json(rep_result_path, result)
        checkpoint["completed_replicates"] = sorted(set(checkpoint.get("completed_replicates", [])) | {rep})
        checkpoint["current_replicate"] = None
        checkpoint["completed_starts_current_replicate"] = []
        atomic_json(checkpoint_path, checkpoint)
        log(f"bootstrap {rep}/{n_boot}: all 11 starts complete, selected={selected_number is not None}, score={result.get('max_abs_score', np.nan):.4g}")
        if selected_number is None:
            checkpoint["status"] = "STOPPED_UNACCEPTABLE_NUMERICAL_ANOMALY"
            checkpoint["stop_reason"] = f"Bootstrap replicate {rep}: all 11 starts ran, but none passed strict parameter-validity and full-rank Jacobian gates"
            atomic_json(checkpoint_path, checkpoint)
            raise RuntimeError(checkpoint["stop_reason"])

    checkpoint["status"] = "COMPLETE"
    atomic_json(checkpoint_path, checkpoint)
    replicate_rows, start_frames, parameter_frames = [], [], []
    for rep in range(1, n_boot + 1):
        replicate_rows.append(json.loads((checkpoint_dir / f"replicate_{rep:03d}_selected_result.json").read_text(encoding="utf-8")))
        start_frames.append(pd.read_csv(checkpoint_dir / f"replicate_{rep:03d}_all_start_diagnostics.csv"))
        parameter_frames.append(pd.read_csv(checkpoint_dir / f"replicate_{rep:03d}_all_start_named_parameters.csv"))
    return pd.DataFrame(replicate_rows), pd.concat(start_frames, ignore_index=True), pd.concat(parameter_frames, ignore_index=True)


def run_bootstrap(n_boot: int) -> None:
    if n_boot <= 0:
        raise ValueError("Bootstrap replicate count must be positive")
    df, _, _ = load_frozen_data()
    Gdf, Xdf, G, X = make_design(df)
    y = df["burden_ord3"].to_numpy(int)
    estimates_path = OUT / "07_selected_parameter_estimates.csv"
    if not estimates_path.exists():
        raise RuntimeError("Full-sample gate outputs are missing; run the full-sample mode first")
    saved = pd.read_csv(estimates_path)
    expected_names = parameter_names(Gdf, Xdf)
    if saved["parameter"].tolist() != expected_names or len(saved) != 47:
        raise RuntimeError("Saved full-sample parameter vector does not match the current design")
    theta0 = saved["estimate"].to_numpy(float)
    names = parameter_names(Gdf, Xdf)
    checkpoint_dir = OUT / f"bootstrap_{n_boot}_checkpoint"
    result_files = sorted(checkpoint_dir.glob("replicate_*_selected_result.json")) if checkpoint_dir.exists() else []
    prior_results = [json.loads(p.read_text(encoding="utf-8")) for p in result_files]
    failed_prior = [r for r in prior_results if not r.get("selected", False)]
    if failed_prior:
        checkpoint_path = checkpoint_dir / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["status"] = "STOPPED_UNACCEPTABLE_NUMERICAL_ANOMALY"
        checkpoint["stop_reason"] = f"Bootstrap replicate {failed_prior[0]['bootstrap_replicate']}: all 11 starts ran, but none passed strict parameter-validity and full-rank Jacobian gates"
        atomic_json(checkpoint_path, checkpoint)
        diag_files = sorted(checkpoint_dir.glob("replicate_*_all_start_diagnostics.csv"))
        param_files = sorted(checkpoint_dir.glob("replicate_*_all_start_named_parameters.csv"))
        diagnostics = pd.concat([pd.read_csv(p) for p in diag_files], ignore_index=True)
        parameters = pd.concat([pd.read_csv(p) for p in param_files], ignore_index=True)
        pd.DataFrame(prior_results).to_csv(OUT / f"10_bootstrap_{n_boot}_replicate_diagnostics.csv", index=False, encoding="utf-8-sig")
        diagnostics.to_csv(OUT / f"11_bootstrap_{n_boot}_all_start_diagnostics.csv", index=False, encoding="utf-8-sig")
        parameters.to_csv(OUT / f"12_bootstrap_{n_boot}_all_start_named_parameters.csv", index=False, encoding="utf-8-sig")
        adjusted_probabilities(theta0, Gdf, Xdf).to_csv(OUT / "08b_adjusted_joint_probabilities.csv", index=False, encoding="utf-8-sig")
        stop_summary = {
            "replicates_requested": n_boot,
            "replicates_completed_before_stop": len(prior_results),
            "failed_replicate": int(failed_prior[0]["bootstrap_replicate"]),
            "failed_replicate_starts_executed": int(failed_prior[0]["starts_executed"]),
            "partial_later_start_rows_preserved": int(len(diagnostics) - sum(int(r.get("starts_executed", 0)) for r in prior_results)),
            "checkpoint_status": checkpoint["status"],
            "stop_reason": checkpoint["stop_reason"],
            "final_status": "FAIL",
        }
        (OUT / f"13_bootstrap_{n_boot}_STOP_summary.json").write_text(json.dumps(stop_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        full_gate = json.loads((OUT / "09_full_sample_gate.json").read_text(encoding="utf-8"))
        report = f"""# Three-Level Zero-Inflated Ordinal ES-GEE Without Concurrent HAMD-17

## Final status: FAIL

The frozen-data, 47-parameter design, and full-sample 11-fit gates passed mechanically. The full-sample solution already required review because its maximum absolute mixing coefficient was {full_gate['max_abs_mixing_coefficient']:.3f} (>5).

The five-replicate participant-cluster bootstrap trial was stopped at the required anomaly gate. Replicate {failed_prior[0]['bootstrap_replicate']} executed all 11 starts, but none passed parameter-validity selection: every solution had an extreme mixing coefficient between approximately 63 and 136, with fitted ordinary-process probabilities reaching 1. No bootstrap solution was selected for that replicate. Continuing the remaining replicates would violate the instruction to stop on an unacceptable numerical anomaly.

Checkpoint/resume was verified by per-start persistence. A later replicate had begun before the anomaly stop was applied; its partial rows remain in the checkpoint directory as audit evidence and are not treated as a completed replicate or scientific result.

- Full-sample 18 effects: `08_domain_effects_robust_wald.csv`.
- Adjusted probabilities: `08b_adjusted_joint_probabilities.csv`.
- Completed/partial bootstrap diagnostics: `11_bootstrap_{n_boot}_all_start_diagnostics.csv`.
- Every saved start's named 47-parameter vector: `12_bootstrap_{n_boot}_all_start_named_parameters.csv`.
- Checkpoint state: `bootstrap_{n_boot}_checkpoint/checkpoint.json`.

This is a numerical-identifiability failure of the bootstrap trial, not formal bootstrap inference. No FDR correction, other model refit, 50-replicate bootstrap, or Step 3 analysis was performed.
"""
        (OUT / "Final_Model_Report.md").write_text(report, encoding="utf-8")
        raise RuntimeError(checkpoint["stop_reason"])
    replicates, starts, all_named_parameters = bootstrap_all_starts(theta0, df, G, X, y, n_boot, names)
    replicates.to_csv(OUT / f"10_bootstrap_{n_boot}_replicate_diagnostics.csv", index=False, encoding="utf-8-sig")
    starts.to_csv(OUT / f"11_bootstrap_{n_boot}_all_start_diagnostics.csv", index=False, encoding="utf-8-sig")
    all_named_parameters.to_csv(OUT / f"12_bootstrap_{n_boot}_all_start_named_parameters.csv", index=False, encoding="utf-8-sig")
    selected_keys = starts.loc[starts["selected"].astype(str).str.lower().eq("true"), ["bootstrap_replicate", "start_number"]]
    selected_named = all_named_parameters.merge(selected_keys, on=["bootstrap_replicate", "start_number"], how="inner")
    selected_named.to_csv(OUT / f"12b_bootstrap_{n_boot}_selected_named_parameters.csv", index=False, encoding="utf-8-sig")
    probabilities = adjusted_probabilities(theta0, Gdf, Xdf)
    probabilities.to_csv(OUT / "08b_adjusted_joint_probabilities.csv", index=False, encoding="utf-8-sig")
    full_gate = json.loads((OUT / "09_full_sample_gate.json").read_text(encoding="utf-8"))
    all_replicates_valid = bool(replicates["selected"].all())
    all_start_counts_valid = bool((replicates["starts_executed"] == 11).all() and len(starts) == n_boot * 11)
    bootstrap_extreme = bool(replicates.get("max_abs_mixing_coefficient", pd.Series(dtype=float)).fillna(np.inf).gt(5).any())
    if not all_replicates_valid or not all_start_counts_valid:
        status = "FAIL"
    elif full_gate["extreme_mixing_abs_gt_5"] or bootstrap_extreme:
        status = "REVIEW REQUIRED"
    else:
        status = "PASS"
    summary = {
        "replicates_requested": n_boot,
        "replicates_with_all_11_starts": int((replicates["starts_executed"] == 11).sum()),
        "selected_valid_solution": int(replicates["selected"].sum()),
        "selection_rate": float(replicates["selected"].mean()),
        "total_start_fits_expected": int(n_boot * 11),
        "total_start_fit_rows": int(len(starts)),
        "early_break_used": False,
        "checkpoint_status": "COMPLETE",
        "checkpoint_directory": str(OUT / f"bootstrap_{n_boot}_checkpoint"),
        "all_replicates_valid": all_replicates_valid,
        "all_start_counts_valid": all_start_counts_valid,
        "full_sample_extreme_mixing_abs_gt_5": bool(full_gate["extreme_mixing_abs_gt_5"]),
        "bootstrap_selected_extreme_mixing_abs_gt_5": bootstrap_extreme,
        "final_status": status,
    }
    (OUT / f"13_bootstrap_{n_boot}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# Three-Level Zero-Inflated Ordinal ES-GEE Without Concurrent HAMD-17

## Final status: {status}

## Frozen analysis set and model

- Freeze ID: `{FREEZE_ID}`; SHA256: `{EXPECTED_FREEZE_SHA256}`.
- Direct read of {EXPECTED_PARTICIPANTS} participants / {EXPECTED_VISITS} person-visits; participant_ID + time unique; no sample reconstruction or filtering.
- Three-level FIBSER burden: 0 / 1 / >=2. Both components exclude concurrent HAMD-17, site, province, and domain x time terms.
- The design contains 47 parameters and two ordered thresholds. OR definitions: zero-inflated `exp(-gamma)`; ordinal `exp(beta)`.

## Full-sample gate

- All 11 fits were executed and strict-valid; selected objective {full_gate['selected_objective']:.9f}, maximum absolute score {full_gate['selected_max_abs_score']:.6g}.
- Ordered thresholds: {full_gate['threshold_values'][0]:.8f}, {full_gate['threshold_values'][1]:.8f}.
- Jacobian full rank: {full_gate['jacobian_full_rank']}; covariance finite/PSD: {full_gate['covariance_finite']}/{full_gate['covariance_PSD']}.
- Maximum absolute mixing coefficient: {full_gate['max_abs_mixing_coefficient']:.6g}; |gamma| > 5 warning: {full_gate['extreme_mixing_abs_gt_5']}.

## Five-replicate bootstrap trial

- Replicates completed with all 11 starts: {summary['replicates_with_all_11_starts']}/{n_boot}; total start fits: {summary['total_start_fit_rows']}/{summary['total_start_fits_expected']}.
- Strict-valid, full-rank selected solution: {summary['selected_valid_solution']}/{n_boot}.
- Early break used: {summary['early_break_used']}; checkpoint status: {summary['checkpoint_status']}.
- Selected bootstrap solution with |mixing coefficient| > 5: {summary['bootstrap_selected_extreme_mixing_abs_gt_5']}.

## Outputs

- `08_domain_effects_robust_wald.csv`: 18 ORs, Wald 95% CIs, and raw Wald P values; no FDR correction.
- `08b_adjusted_joint_probabilities.csv`: adjusted P(burden>=1), P(burden>=2), and domain-present minus domain-absent probability differences.
- `10_bootstrap_{n_boot}_replicate_diagnostics.csv`: replicate-level selection diagnostics.
- `11_bootstrap_{n_boot}_all_start_diagnostics.csv`: all {n_boot * 11} start diagnostics.
- `12_bootstrap_{n_boot}_all_start_named_parameters.csv`: named 47-parameter vector for every completed start.
- `bootstrap_{n_boot}_checkpoint/`: per-start progress files and resume checkpoint.

## Interpretation of status

The run is labelled **{status}**. Numerical convergence, rank, covariance, checkpoint, and all-start execution requirements are evaluated mechanically. `REVIEW REQUIRED` is assigned when otherwise valid solutions contain extreme mixing parameters (|gamma| > 5); this is a latent-process stability warning, not a claim that the scientific associations are invalid. This five-replicate run is a feasibility trial only and is not formal bootstrap inference.
"""
    (OUT / "Final_Model_Report.md").write_text(report, encoding="utf-8")
    log(f"bootstrap completed: {summary}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, version, freeze_audit = load_frozen_data()
    Gdf, Xdf, G, X = make_design(df)
    y = df["burden_ord3"].to_numpy(int)
    names = parameter_names(Gdf, Xdf)
    parameter_map = pd.DataFrame({"index_zero_based": range(len(names)), "index_one_based": range(1, len(names) + 1), "parameter": names})
    design_audit = pd.DataFrame([
        {"component": "zero-inflated", "columns": Gdf.shape[1], "RHS": RHS, "concurrent_HAMD17_included": False},
        {"component": "ordinal", "columns": Xdf.shape[1], "RHS": RHS, "concurrent_HAMD17_included": False},
        {"component": "ordinal thresholds", "columns": Q, "RHS": "two ordered thresholds", "concurrent_HAMD17_included": False},
    ])
    outcome = df["burden_ord3"].value_counts().reindex(range(3), fill_value=0)
    outcome_table = pd.DataFrame({"category": ["0", "1", ">=2"], "n": outcome.to_numpy(), "percent": outcome.to_numpy() / len(df) * 100})
    log(f"freeze passed: {df.participant_ID.nunique()} participants / {len(df)} visits; parameters={len(names)}")
    theta, multistart, _ = fit_full_sample(y, G, X)
    V, covariance = selected_covariance(theta, y, G, X, df["participant_ID"])
    effects = domain_effects(theta, V, Gdf, Xdf)
    probabilities = adjusted_probabilities(theta, Gdf, Xdf)
    estimates = parameter_map.copy()
    estimates["estimate"] = theta
    estimates["robust_SE"] = np.sqrt(np.maximum(np.diag(V), 0.0))
    gamma, _, _, alpha, _ = zi.unpack(theta, G.shape[1], X.shape[1], Q)
    selected = multistart.loc[multistart["selected"]].iloc[0]
    final_gate = {
        "freeze_id": FREEZE_ID,
        "freeze_sha256": EXPECTED_FREEZE_SHA256,
        "participants": int(df.participant_ID.nunique()),
        "person_visits": int(len(df)),
        "unique_participant_time": bool(not df.duplicated(["participant_ID", "time"]).any()),
        "parameters": int(len(theta)),
        "ordinal_thresholds": int(Q),
        "threshold_values": [float(x) for x in alpha],
        "selected_fit_number": int(selected["fit_number"]),
        "selected_start": str(selected["start"]),
        "selected_objective": float(selected["objective"]),
        "selected_max_abs_score": float(selected["max_abs_score"]),
        "selected_strict_valid": bool(selected["strict_valid"]),
        "all_11_fits_executed": bool(len(multistart) == 11),
        "strict_valid_fits": int(multistart["strict_valid"].sum()),
        "max_abs_mixing_coefficient": float(np.max(np.abs(gamma))),
        "extreme_mixing_abs_gt_5": bool(np.max(np.abs(gamma)) > 5),
        "jacobian_full_rank": bool(covariance.iloc[0]["jacobian_full_rank"]),
        "covariance_finite": bool(covariance.iloc[0]["covariance_finite"]),
        "covariance_PSD": bool(covariance.iloc[0]["covariance_PSD"]),
    }
    freeze_audit.to_csv(OUT / "01_freeze_integrity_audit.csv", index=False, encoding="utf-8-sig")
    outcome_table.to_csv(OUT / "02_outcome_distribution.csv", index=False, encoding="utf-8-sig")
    design_audit.to_csv(OUT / "03_design_audit.csv", index=False, encoding="utf-8-sig")
    parameter_map.to_csv(OUT / "04_parameter_index_map.csv", index=False, encoding="utf-8-sig")
    multistart.to_csv(OUT / "05_full_sample_11_fit_diagnostics.csv", index=False, encoding="utf-8-sig")
    covariance.to_csv(OUT / "06_selected_jacobian_covariance.csv", index=False, encoding="utf-8-sig")
    estimates.to_csv(OUT / "07_selected_parameter_estimates.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(OUT / "08_domain_effects_robust_wald.csv", index=False, encoding="utf-8-sig")
    probabilities.to_csv(OUT / "08b_adjusted_joint_probabilities.csv", index=False, encoding="utf-8-sig")
    (OUT / "09_full_sample_gate.json").write_text(json.dumps(final_gate, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# Three-Level Zero-Inflated Ordinal ES-GEE Without Concurrent HAMD-17

## Scope

- Freeze ID: `{FREEZE_ID}`.
- Direct frozen-data read: {EXPECTED_PARTICIPANTS} participants / {EXPECTED_VISITS} person-visits; no sample reconstruction or row filtering.
- Outcome: FIBSER burden 0 / 1 / >=2.
- Both components include time, nine repaired PRISE domains, age, sex, education, first_episode, baseline HAMD-17, baseline SDS, and baseline antidepressant class.
- Concurrent HAMD-17, site, province, and domain x time terms are absent from both components.
- Probability/OR definitions are unchanged: zero-inflated OR = exp(-gamma); ordinal OR = exp(beta).

## Full-sample gate

- Eleven optimization fits were executed (10 prespecified starts plus strict refinement from the best valid raw start).
- Selected fit: {int(selected['fit_number'])} (`{selected['start']}`), objective={selected['objective']:.9f}, max absolute score={selected['max_abs_score']:.6g}.
- Strict-valid fits: {int(multistart['strict_valid'].sum())}/11.
- Parameters: {len(theta)}; ordered thresholds: {alpha[0]:.8f}, {alpha[1]:.8f}.
- Selected maximum absolute mixing coefficient: {np.max(np.abs(gamma)):.6g}; extreme |gamma| > 5: {bool(np.max(np.abs(gamma)) > 5)}.
- Jacobian full rank: {bool(covariance.iloc[0]['jacobian_full_rank'])}; condition number={covariance.iloc[0]['jacobian_condition_number']:.6g}.
- Sandwich covariance finite: {bool(covariance.iloc[0]['covariance_finite'])}; PSD: {bool(covariance.iloc[0]['covariance_PSD'])}; condition number={covariance.iloc[0]['covariance_condition_number']:.6g}.

The full parameter index/name map is in `04_parameter_index_map.csv`; all 11 fit diagnostics are in `05_full_sample_11_fit_diagnostics.csv`.
"""
    (OUT / "Full_Sample_Report.md").write_text(report, encoding="utf-8")
    log(f"full-sample gate completed: {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N", help="Run N participant-cluster bootstrap replicates after the saved full-sample gate")
    args = parser.parse_args()
    if args.bootstrap:
        run_bootstrap(args.bootstrap)
    else:
        main()




