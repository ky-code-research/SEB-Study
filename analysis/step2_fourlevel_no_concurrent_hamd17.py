from __future__ import annotations

import argparse
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

from scipy import optimize, stats

import model_comparison_gate_fibser as legacy
import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2B3_fixed_sample_covariate_attenuation as b3
import step2_burden3_no_concurrent_HAMD_4799_rerun as engine


from public_paths import ROOT
FREEZE = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "repaired_natural_complete_case_analysis_data.csv"
VERSION_RECORD = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "data_version_record.json"
OUT = ROOT / "outputs" / "step2_fourlevel_no_concurrent_hamd17"

MODEL_ID = "FOUR_LEVEL_ZI_ORDINAL_ES_GEE_NO_CONCURRENT_HAMD17_V1"
FREEZE_ID = "PRISE_REPAIRED_NATURAL_CC_1468P_4799V_2026-10-08"
EXPECTED_FREEZE_SHA256 = "6844f512b5122beabc9905456e1a9a2111f6318c398a984232068b22c22fd16c"
EXPECTED_SOURCE_SHA256 = "d7e59303c755dbd0547b0b6c84694d038f53e775e83a6a2483057007fd3054c8"
EXPECTED_PARTICIPANTS = 1468
EXPECTED_VISITS = 4799
EXPECTED_COUNTS = {0: 3464, 1: 584, 2: 512, 3: 239}
Q = 3
N_PARAMETERS = 48
SEED = 20261008
SCORE_TOL = 5e-2
DOMAIN_VARS = [b1.DOMAIN_VAR[d] for d in b1.DOMAINS]
RHS = b3.MODEL_RHS["M3"].replace(" + concurrent_HAMD17_z", "")

# Reuse the already-audited likelihood, score, Hessian, sandwich, and optimizer
# helpers while forcing their runtime globals to the four-level model.
engine.Q = Q
engine.OUT = OUT
engine.SCORE_TOL = SCORE_TOL


def log(message: str) -> None:
    print(f"Four-level no-concurrent-HAMD17: {message}", flush=True)


engine.log = log


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


def parameter_names(Gdf: pd.DataFrame, Xdf: pd.DataFrame) -> list[str]:
    return (
        [f"mixing_gamma:{c}" for c in Gdf.columns]
        + ["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"]
        + [f"ordinal_beta:{c}" for c in Xdf.columns]
    )


def load_and_preflight() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    version = json.loads(VERSION_RECORD.read_text(encoding="utf-8"))
    freeze_hash = sha256(FREEZE)
    df = pd.read_csv(FREEZE)
    checks: list[dict] = []

    def check(name: str, observed, expected, passed: bool) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "pass": bool(passed)})

    check("freeze SHA256", freeze_hash, EXPECTED_FREEZE_SHA256, freeze_hash.lower() == EXPECTED_FREEZE_SHA256)
    check("source SHA256", version.get("source_sha256"), EXPECTED_SOURCE_SHA256, version.get("source_sha256", "").lower() == EXPECTED_SOURCE_SHA256)
    check("participants", df["participant_ID"].nunique(), EXPECTED_PARTICIPANTS, df["participant_ID"].nunique() == EXPECTED_PARTICIPANTS)
    check("person-visits", len(df), EXPECTED_VISITS, len(df) == EXPECTED_VISITS)
    duplicate_keys = int(df.duplicated(["participant_ID", "time"]).sum())
    check("duplicate participant_ID + time", duplicate_keys, 0, duplicate_keys == 0)

    if "burden_ord4" not in df.columns or df["burden_ord4"].isna().any():
        raise RuntimeError("PREFLIGHT FAIL: burden_ord4 is missing or contains missing values")
    df["burden_ord4"] = df["burden_ord4"].astype(int)
    levels = sorted(df["burden_ord4"].unique().tolist())
    counts = df["burden_ord4"].value_counts().sort_index().to_dict()
    check("outcome levels", levels, [0, 1, 2, 3], levels == [0, 1, 2, 3])
    check("outcome counts", json.dumps(counts, sort_keys=True), json.dumps(EXPECTED_COUNTS, sort_keys=True), counts == EXPECTED_COUNTS)

    Gdf, Xdf = b1.make_design(df, mix_rhs=RHS, ord_rhs=RHS)
    forbidden = [c for c in [*Gdf.columns, *Xdf.columns] if "concurrent_HAMD" in c or "site" in c.lower() or "province" in c.lower()]
    missing_mixing_domains = [c for c in DOMAIN_VARS if c not in Gdf.columns]
    missing_ordinal_domains = [c for c in DOMAIN_VARS if c not in Xdf.columns]
    names = parameter_names(Gdf, Xdf)
    check("mixing columns", Gdf.shape[1], 23, Gdf.shape[1] == 23)
    check("ordinal columns", Xdf.shape[1], 22, Xdf.shape[1] == 22)
    check("threshold count", Q, 3, Q == 3)
    check("total parameters", len(names), N_PARAMETERS, len(names) == N_PARAMETERS)
    check("unique parameter names", len(set(names)), N_PARAMETERS, len(set(names)) == N_PARAMETERS)
    check("forbidden design terms", json.dumps(forbidden), "[]", not forbidden)
    check("nine mixing domains", json.dumps(missing_mixing_domains), "[]", not missing_mixing_domains)
    check("nine ordinal domains", json.dumps(missing_ordinal_domains), "[]", not missing_ordinal_domains)
    check("threshold positions", json.dumps(names[23:26]), json.dumps(["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"]), names[23:26] == ["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"])

    audit = pd.DataFrame(checks)
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / "01_preflight_hard_checks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "category": ["0", "1", "2", ">=3"],
        "n": [counts.get(i, 0) for i in range(4)],
        "percent": [counts.get(i, 0) / len(df) * 100 for i in range(4)],
    }).to_csv(OUT / "02_outcome_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "index_zero_based": range(len(names)),
        "index_one_based": range(1, len(names) + 1),
        "block": ["mixing"] * 23 + ["threshold"] * 3 + ["ordinal"] * 22,
        "parameter": names,
    }).to_csv(OUT / "03_parameter_index_map.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"component": "mixing", "columns": Gdf.shape[1], "RHS": RHS},
        {"component": "ordered thresholds", "columns": Q, "RHS": "a0 + positive cumulative exp(log_delta) increments"},
        {"component": "ordinal", "columns": Xdf.shape[1], "RHS": RHS},
    ]).to_csv(OUT / "04_design_audit.csv", index=False, encoding="utf-8-sig")
    atomic_json(OUT / "00_preflight_manifest.json", {
        "model_id": MODEL_ID,
        "freeze_id": FREEZE_ID,
        "freeze_sha256": freeze_hash,
        "source_sha256": version.get("source_sha256"),
        "participants": int(df["participant_ID"].nunique()),
        "person_visits": int(len(df)),
        "outcome": "0 / 1 / 2 / >=3",
        "outcome_counts": {str(k): int(v) for k, v in counts.items()},
        "mixing_parameters": int(Gdf.shape[1]),
        "thresholds": Q,
        "ordinal_parameters": int(Xdf.shape[1]),
        "total_parameters": len(names),
        "all_checks_pass": bool(audit["pass"].all()),
    })
    if not audit["pass"].all():
        failed = audit.loc[~audit["pass"], "check"].tolist()
        raise RuntimeError(f"PREFLIGHT FAIL: {failed}")
    return df, Gdf, Xdf, Gdf.to_numpy(float), Xdf.to_numpy(float), audit


def adjusted_probabilities(theta: np.ndarray, Gdf: pd.DataFrame, Xdf: pd.DataFrame) -> pd.DataFrame:
    G = Gdf.to_numpy(float)
    X = Xdf.to_numpy(float)
    rows = []
    for domain in b1.DOMAINS:
        var = b1.DOMAIN_VAR[domain]
        gi, xi = list(Gdf.columns).index(var), list(Xdf.columns).index(var)
        g0, g1, x0, x1 = G.copy(), G.copy(), X.copy(), X.copy()
        g0[:, gi] = x0[:, xi] = 0.0
        g1[:, gi] = x1[:, xi] = 1.0
        p0 = engine.predict(theta, g0, x0)[0]
        p1 = engine.predict(theta, g1, x1)[0]
        row = {"Domain": domain}
        for cutoff in (1, 2, 3):
            a = p0[:, cutoff:].sum(axis=1)
            b = p1[:, cutoff:].sum(axis=1)
            row[f"P_burden_ge{cutoff}_domain0"] = float(a.mean())
            row[f"P_burden_ge{cutoff}_domain1"] = float(b.mean())
            row[f"RD_burden_ge{cutoff}_domain1_minus_domain0"] = float((b - a).mean())
        row["mean_probability_sum_domain0"] = float(p0.sum(axis=1).mean())
        row["mean_probability_sum_domain1"] = float(p1.sum(axis=1).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_full_sample() -> None:
    df, Gdf, Xdf, G, X, _ = load_and_preflight()
    y = df["burden_ord4"].to_numpy(int)
    names = parameter_names(Gdf, Xdf)
    log("preflight PASS; starting all 11 full-sample fits")
    theta, diagnostics, candidates = engine.fit_full_sample(y, G, X)
    V, covariance = engine.selected_covariance(theta, y, G, X, df["participant_ID"].to_numpy())
    effects = engine.domain_effects(theta, V, Gdf, Xdf)
    probabilities = adjusted_probabilities(theta, Gdf, Xdf)

    all_params = []
    for fit_number, candidate in enumerate(candidates, start=1):
        all_params.append(pd.DataFrame({
            "fit_number": fit_number,
            "start": candidate["row"]["start"],
            "parameter_index_zero_based": range(N_PARAMETERS),
            "parameter": names,
            "estimate": candidate["theta"],
        }))
    all_params_df = pd.concat(all_params, ignore_index=True)
    selected_row = diagnostics.loc[diagnostics["selected"]].iloc[0]
    selected_fit = int(selected_row["fit_number"])
    selected_theta = all_params_df.loc[all_params_df["fit_number"].eq(selected_fit)].sort_values("parameter_index_zero_based")["estimate"].to_numpy(float)
    comparisons = []
    selected_objective = float(selected_row["objective"])
    for fit_number, candidate in enumerate(candidates, start=1):
        comparisons.append({
            "fit_number": fit_number,
            "start": candidate["row"]["start"],
            "strict_valid": bool(candidate["strict_valid"]),
            "objective_difference_from_selected": float(candidate["assessment"]["objective"] - selected_objective),
            "max_abs_parameter_difference_from_selected": float(np.max(np.abs(candidate["theta"] - selected_theta))),
            "max_abs_mixing_coefficient": float(candidate["assessment"]["max_abs_mixing_coefficient"]),
            "min_pi": float(candidate["assessment"]["min_pi"]),
            "max_pi": float(candidate["assessment"]["max_pi"]),
            "predicted_zero": float(candidate["assessment"]["predicted_zero"]),
        })
    comparison_df = pd.DataFrame(comparisons)

    estimates = pd.DataFrame({
        "index_zero_based": range(N_PARAMETERS),
        "index_one_based": range(1, N_PARAMETERS + 1),
        "parameter": names,
        "estimate": theta,
        "robust_SE": np.sqrt(np.maximum(np.diag(V), 0.0)),
    })
    gamma, _, _, alpha, _ = engine.zi.unpack(theta, G.shape[1], X.shape[1], Q)
    full_valid = bool(
        len(diagnostics) == 11
        and bool(selected_row["strict_valid"])
        and bool(covariance.iloc[0]["jacobian_full_rank"])
        and bool(covariance.iloc[0]["covariance_finite"])
        and bool(covariance.iloc[0]["covariance_PSD"])
        and len(alpha) == 3
        and np.all(np.diff(alpha) > 0)
    )
    extreme = bool(np.max(np.abs(gamma)) > 5)
    status = "FAIL" if not full_valid else ("REVIEW REQUIRED" if extreme else "PASS")
    gate = {
        "model_id": MODEL_ID,
        "status": status,
        "full_sample_numerical_gate_pass": full_valid,
        "all_11_fits_executed": len(diagnostics) == 11,
        "strict_valid_fits": int(diagnostics["strict_valid"].sum()),
        "selected_fit_number": selected_fit,
        "selected_start": str(selected_row["start"]),
        "selected_objective": selected_objective,
        "selected_max_abs_score": float(selected_row["max_abs_score"]),
        "parameters": len(theta),
        "ordinal_thresholds": 3,
        "threshold_values": [float(x) for x in alpha],
        "max_abs_mixing_coefficient": float(np.max(np.abs(gamma))),
        "extreme_mixing_abs_gt_5": extreme,
        "jacobian_full_rank": bool(covariance.iloc[0]["jacobian_full_rank"]),
        "jacobian_condition_number": float(covariance.iloc[0]["jacobian_condition_number"]),
        "covariance_finite": bool(covariance.iloc[0]["covariance_finite"]),
        "covariance_PSD": bool(covariance.iloc[0]["covariance_PSD"]),
        "covariance_condition_number": float(covariance.iloc[0]["covariance_condition_number"]),
        "high_mixing_strict_valid_fits": int(((diagnostics["strict_valid"]) & (diagnostics["max_abs_mixing_coefficient"] > 5)).sum()),
    }
    diagnostics.to_csv(OUT / "05_full_sample_11_fit_diagnostics.csv", index=False, encoding="utf-8-sig")
    all_params_df.to_csv(OUT / "06_full_sample_all_start_named_parameters.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(OUT / "07_full_sample_multistart_agreement.csv", index=False, encoding="utf-8-sig")
    covariance.to_csv(OUT / "08_selected_jacobian_covariance.csv", index=False, encoding="utf-8-sig")
    estimates.to_csv(OUT / "09_selected_parameter_estimates.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(OUT / "10_domain_effects_robust_wald.csv", index=False, encoding="utf-8-sig")
    probabilities.to_csv(OUT / "11_adjusted_joint_probabilities.csv", index=False, encoding="utf-8-sig")
    atomic_json(OUT / "12_full_sample_gate.json", gate)
    (OUT / "Full_Sample_Report.md").write_text(
        "# Four-Level Zero-Inflated Ordinal ES-GEE Without Concurrent HAMD-17\n\n"
        f"## Full-sample status: {status}\n\n"
        f"- Frozen sample: {EXPECTED_PARTICIPANTS} participants / {EXPECTED_VISITS} person-visits.\n"
        "- Outcome: 0 / 1 / 2 / >=3; counts 3464 / 584 / 512 / 239.\n"
        "- Parameter blocks: mixing 23 + thresholds 3 + ordinal 22 = 48.\n"
        "- Concurrent HAMD-17, site/province, and domain-by-time terms are absent.\n"
        f"- All 11 fits executed; strict-valid fits: {gate['strict_valid_fits']}/11.\n"
        f"- Selected fit {selected_fit}; objective {selected_objective:.9f}; max absolute score {gate['selected_max_abs_score']:.6g}.\n"
        f"- Ordered thresholds: {', '.join(f'{x:.8f}' for x in alpha)}.\n"
        f"- Maximum absolute mixing coefficient: {gate['max_abs_mixing_coefficient']:.6g}; |gamma| > 5: {extreme}.\n"
        f"- Jacobian full rank/condition: {gate['jacobian_full_rank']} / {gate['jacobian_condition_number']:.6g}.\n"
        f"- Covariance finite/PSD/condition: {gate['covariance_finite']} / {gate['covariance_PSD']} / {gate['covariance_condition_number']:.6g}.\n",
        encoding="utf-8",
    )
    log(f"full-sample complete: status={status}, output={OUT}")
    if not full_valid:
        raise RuntimeError("Full-sample numerical gate failed; bootstrap is not permitted")


def run_bootstrap(n_boot: int) -> None:
    if n_boot != 5:
        raise ValueError("This feasibility runner is locked to exactly 5 bootstrap replicates")
    df, Gdf, Xdf, G, X, _ = load_and_preflight()
    y = df["burden_ord4"].to_numpy(int)
    names = parameter_names(Gdf, Xdf)
    gate_path = OUT / "12_full_sample_gate.json"
    estimate_path = OUT / "09_selected_parameter_estimates.csv"
    if not gate_path.exists() or not estimate_path.exists():
        raise RuntimeError("Full-sample outputs are missing")
    full_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not full_gate.get("full_sample_numerical_gate_pass", False):
        raise RuntimeError("Full-sample numerical gate did not pass")
    saved = pd.read_csv(estimate_path)
    if len(saved) != N_PARAMETERS or saved["parameter"].tolist() != names:
        raise RuntimeError("Saved 48-parameter vector does not match the live name map")
    theta0 = saved["estimate"].to_numpy(float)

    ids = df["participant_ID"].unique()
    pid = df["participant_ID"].to_numpy()
    checkpoint_dir = OUT / "bootstrap_5_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {
        "model_id": MODEL_ID,
        "freeze_sha256": EXPECTED_FREEZE_SHA256,
        "replicates_requested": 5,
        "completed_replicates": [],
        "current_replicate": None,
        "completed_starts_current_replicate": [],
        "status": "RUNNING",
    }
    if checkpoint.get("model_id") != MODEL_ID or checkpoint.get("freeze_sha256") != EXPECTED_FREEZE_SHA256 or checkpoint.get("replicates_requested") != 5:
        raise RuntimeError("Bootstrap checkpoint is incompatible with this four-level model")

    for rep in range(1, 6):
        diag_path = checkpoint_dir / f"replicate_{rep:03d}_all_start_diagnostics.csv"
        param_path = checkpoint_dir / f"replicate_{rep:03d}_all_start_named_parameters.csv"
        result_path = checkpoint_dir / f"replicate_{rep:03d}_selected_result.json"
        if result_path.exists() and rep in checkpoint.get("completed_replicates", []):
            log(f"bootstrap {rep}/5 already complete; checkpoint skip")
            continue
        rng = np.random.default_rng(SEED + rep)
        sampled = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([np.flatnonzero(pid == ident) for ident in sampled])
        yb, Gb, Xb = y[idx], G[idx], X[idx]
        generated = legacy.build_zi_starts(yb, Gb, Xb, Q + 1)
        starts = [{"start": "full_sample_selected", "theta": theta0, "method": "name_verified_full_sample_vector"}, *generated]
        if len(starts) != 11:
            raise RuntimeError(f"Bootstrap replicate {rep}: expected 11 starts, found {len(starts)}")
        old_diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
        old_params = pd.read_csv(param_path) if param_path.exists() else pd.DataFrame()
        completed = set(old_diag.get("start_number", pd.Series(dtype=int)).astype(int).tolist())
        checkpoint.update({"current_replicate": rep, "completed_starts_current_replicate": sorted(completed), "status": "RUNNING"})
        atomic_json(checkpoint_path, checkpoint)

        # No early break: all eleven starts are persisted for every replicate.
        for start_number, start in enumerate(starts, start=1):
            if start_number in completed:
                continue
            try:
                res = optimize.minimize(
                    lambda th: engine.zi.neg_obj_grad(th, yb, Gb, Xb, Q),
                    start["theta"], method="L-BFGS-B", jac=True,
                    options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 100, "maxcor": 80},
                )
                assessment = engine.basic_assessment(res.x, yb, Gb, Xb)
                basic_valid = bool(res.success and assessment["strict_score_converged"] and assessment["parameter_valid"])
                row = {"bootstrap_replicate": rep, "start_number": start_number, "start": start["start"], "start_method": start["method"], "program_success": bool(res.success), "message": str(res.message), "iterations": int(res.nit), **assessment, "basic_valid": basic_valid, "jacobian_checked": False, "jacobian_full_rank": False, "selected": False}
                param_rows = pd.DataFrame({
                    "bootstrap_replicate": rep,
                    "start_number": start_number,
                    "start": start["start"],
                    "parameter_index_zero_based": range(N_PARAMETERS),
                    "parameter": names,
                    "estimate": res.x,
                })
            except Exception as exc:
                row = {"bootstrap_replicate": rep, "start_number": start_number, "start": start["start"], "start_method": start["method"], "program_success": False, "message": repr(exc), "basic_valid": False, "jacobian_checked": False, "jacobian_full_rank": False, "selected": False}
                param_rows = pd.DataFrame()
            old_diag = pd.concat([old_diag, pd.DataFrame([row])], ignore_index=True)
            old_diag.to_csv(diag_path, index=False, encoding="utf-8-sig")
            if not param_rows.empty:
                old_params = pd.concat([old_params, param_rows], ignore_index=True)
                old_params.to_csv(param_path, index=False, encoding="utf-8-sig")
            completed.add(start_number)
            checkpoint["completed_starts_current_replicate"] = sorted(completed)
            atomic_json(checkpoint_path, checkpoint)
            log(f"bootstrap {rep}/5 start {start_number}/11 saved: success={row['program_success']}, basic_valid={row['basic_valid']}")

        if len(completed) != 11 or len(old_diag) != 11:
            raise RuntimeError(f"Bootstrap replicate {rep}: expected 11 diagnostic rows, found {len(old_diag)}")
        param_lookup = {}
        for start_number, sub in old_params.groupby("start_number", sort=False):
            ordered = sub.sort_values("parameter_index_zero_based")
            if len(ordered) == N_PARAMETERS and ordered["parameter"].tolist() == names:
                param_lookup[int(start_number)] = ordered["estimate"].to_numpy(float)
        candidates = old_diag.loc[old_diag["basic_valid"].astype(str).str.lower().eq("true")].sort_values(["objective", "max_abs_score"], ascending=[False, True])
        selected_number = None
        for _, candidate in candidates.iterrows():
            number = int(candidate["start_number"])
            candidate_theta = param_lookup.get(number)
            if candidate_theta is None:
                continue
            _, jac = engine.jacobian_assessment(candidate_theta, yb, Gb, Xb)
            for key, value in jac.items():
                old_diag.loc[old_diag["start_number"].eq(number), key] = value
            old_diag.loc[old_diag["start_number"].eq(number), "jacobian_checked"] = True
            if jac["jacobian_finite"] and jac["jacobian_full_rank"]:
                selected_number = number
                break
        result = {
            "bootstrap_replicate": rep,
            "sampled_cluster_draws": int(len(sampled)),
            "unique_original_participants": int(len(set(sampled))),
            "rows": int(len(idx)),
            "starts_executed": 11,
            "selected": selected_number is not None,
        }
        if selected_number is None:
            result["failure"] = "No strict-valid full-rank solution among all 11 starts"
        else:
            mask = old_diag["start_number"].eq(selected_number)
            old_diag.loc[mask, "selected"] = True
            selected_diag = old_diag.loc[mask].iloc[0]
            selected_theta = param_lookup[selected_number]
            gamma, _, _, alpha, _ = engine.zi.unpack(selected_theta, Gb.shape[1], Xb.shape[1], Q)
            result.update({
                "selected_start_number": selected_number,
                "selected_start": str(selected_diag["start"]),
                "objective": float(selected_diag["objective"]),
                "max_abs_score": float(selected_diag["max_abs_score"]),
                "max_abs_mixing_coefficient": float(np.max(np.abs(gamma))),
                "max_abs_ordinal_coefficient": float(selected_diag["max_abs_ordinal_coefficient"]),
                "threshold_values": [float(x) for x in alpha],
                "min_pi": float(selected_diag["min_pi"]),
                "max_pi": float(selected_diag["max_pi"]),
                "predicted_zero": float(selected_diag["predicted_zero"]),
                "jacobian_full_rank": True,
            })
        old_diag.to_csv(diag_path, index=False, encoding="utf-8-sig")
        atomic_json(result_path, result)
        checkpoint["completed_replicates"] = sorted(set(checkpoint.get("completed_replicates", [])) | {rep})
        checkpoint["current_replicate"] = None
        checkpoint["completed_starts_current_replicate"] = []
        atomic_json(checkpoint_path, checkpoint)
        log(f"bootstrap {rep}/5 complete: selected={result['selected']}")

    checkpoint["status"] = "COMPLETE"
    atomic_json(checkpoint_path, checkpoint)
    replicate_rows, diag_frames, param_frames = [], [], []
    for rep in range(1, 6):
        replicate_rows.append(json.loads((checkpoint_dir / f"replicate_{rep:03d}_selected_result.json").read_text(encoding="utf-8")))
        diag_frames.append(pd.read_csv(checkpoint_dir / f"replicate_{rep:03d}_all_start_diagnostics.csv"))
        param_frames.append(pd.read_csv(checkpoint_dir / f"replicate_{rep:03d}_all_start_named_parameters.csv"))
    replicates = pd.DataFrame(replicate_rows)
    diagnostics = pd.concat(diag_frames, ignore_index=True)
    parameters = pd.concat(param_frames, ignore_index=True)
    replicates.to_csv(OUT / "13_bootstrap_5_replicate_diagnostics.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(OUT / "14_bootstrap_5_all_start_diagnostics.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(OUT / "15_bootstrap_5_all_start_named_parameters.csv", index=False, encoding="utf-8-sig")
    selected_keys = diagnostics.loc[diagnostics["selected"].astype(str).str.lower().eq("true"), ["bootstrap_replicate", "start_number"]]
    parameters.merge(selected_keys, on=["bootstrap_replicate", "start_number"], how="inner").to_csv(OUT / "16_bootstrap_5_selected_named_parameters.csv", index=False, encoding="utf-8-sig")

    all_selected = bool(replicates["selected"].all())
    counts_ok = bool(len(diagnostics) == 55 and (replicates["starts_executed"] == 11).all())
    bootstrap_extreme = bool(replicates.loc[replicates["selected"], "max_abs_mixing_coefficient"].gt(5).any()) if all_selected else True
    if not all_selected or not counts_ok:
        status = "FAIL"
    elif full_gate["extreme_mixing_abs_gt_5"] or bootstrap_extreme:
        status = "REVIEW REQUIRED"
    else:
        status = "PASS"
    summary = {
        "model_id": MODEL_ID,
        "final_status": status,
        "replicates_requested": 5,
        "replicates_completed": int(len(replicates)),
        "replicates_with_all_11_starts": int((replicates["starts_executed"] == 11).sum()),
        "total_start_fit_rows": int(len(diagnostics)),
        "total_start_fits_expected": 55,
        "selected_strict_full_rank_solutions": int(replicates["selected"].sum()),
        "checkpoint_status": "COMPLETE",
        "early_break_used": False,
        "full_sample_extreme_mixing_abs_gt_5": bool(full_gate["extreme_mixing_abs_gt_5"]),
        "bootstrap_selected_extreme_mixing_abs_gt_5": bootstrap_extreme,
    }
    atomic_json(OUT / "17_final_summary.json", summary)
    (OUT / "Final_Model_Report.md").write_text(
        "# Four-Level Zero-Inflated Ordinal ES-GEE Without Concurrent HAMD-17\n\n"
        f"## Final status: {status}\n\n"
        "## Model and full sample\n\n"
        "- Frozen data: 1468 participants / 4799 person-visits.\n"
        "- Outcome: 0 / 1 / 2 / >=3; counts 3464 / 584 / 512 / 239.\n"
        "- Parameter blocks: mixing 23 + thresholds 3 + ordinal 22 = 48.\n"
        f"- Full-sample fits executed: 11/11; strict-valid: {full_gate['strict_valid_fits']}/11.\n"
        f"- Selected objective: {full_gate['selected_objective']:.9f}; max absolute score: {full_gate['selected_max_abs_score']:.6g}.\n"
        f"- Thresholds: {', '.join(f'{x:.8f}' for x in full_gate['threshold_values'])}.\n"
        f"- Maximum absolute mixing coefficient: {full_gate['max_abs_mixing_coefficient']:.6g}.\n\n"
        "## Five-replicate participant-cluster bootstrap feasibility run\n\n"
        f"- Replicates completed: {summary['replicates_completed']}/5.\n"
        f"- All-start fits persisted: {summary['total_start_fit_rows']}/55.\n"
        f"- Strict-valid full-rank selected solutions: {summary['selected_strict_full_rank_solutions']}/5.\n"
        f"- Selected bootstrap solutions with |gamma| > 5: {summary['bootstrap_selected_extreme_mixing_abs_gt_5']}.\n"
        "- Checkpoint status: COMPLETE; no early break.\n\n"
        "The 18 domain estimates in `10_domain_effects_robust_wald.csv` are Wald results with raw P values. This five-replicate run is a numerical stability trial, not formal bootstrap inference.\n",
        encoding="utf-8",
    )
    log(f"bootstrap complete: status={status}, output={OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap:
        run_bootstrap(args.bootstrap)
    else:
        run_full_sample()




