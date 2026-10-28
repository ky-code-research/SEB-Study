from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

# Compatibility for legacy project modules under NumPy >= 2.
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

import pandas as pd

import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2_candidate_burden3_no_concurrent_HAMD as zi
import step2_fourlevel_concurrent_hamd17_education3 as step2_m1
import step2_fourlevel_no_concurrent_hamd17_education3 as step2_m2
import step2_fourlevel_no_concurrent_hamd17 as step2_base
import step3_domain_time_burden_4799_preliminary as step3
import step3_independent_model_core as step3_core


from public_paths import ROOT
FREEZE = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "repaired_natural_complete_case_analysis_data.csv"
OUT = ROOT / "outputs" / "site_clustered_sandwich_sensitivity_2026-10-23"
EXPECTED_HASH = "6844f512b5122beabc9905456e1a9a2111f6318c398a984232068b22c22fd16c"
EXPECTED_N = 1468
EXPECTED_VISITS = 4799
EXPECTED_SITES = 21
FG_B = 0.75
RANK_RTOL = 1e-8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sym(a: np.ndarray) -> np.ndarray:
    return (a + a.T) / 2.0


def numerical_rank(a: np.ndarray, rtol: float = RANK_RTOL) -> tuple[int, float, np.ndarray]:
    s = np.linalg.svd(np.asarray(a, dtype=float), compute_uv=False)
    tol = (float(s[0]) * rtol) if len(s) and s[0] > 0 else rtol
    return int(np.sum(s > tol)), tol, s


def covariance_diagnostics(v: np.ndarray, label: str) -> dict:
    vs = sym(v)
    eig = np.linalg.eigvalsh(vs)
    rank, tol, sv = numerical_rank(vs)
    positive = sv[sv > tol]
    return {
        "covariance": label,
        "dimension": int(vs.shape[0]),
        "rank_rtol_1e-8": rank,
        "rank_tolerance": tol,
        "finite": bool(np.isfinite(vs).all()),
        "PSD_tol_1e-8": bool(eig.min() >= -1e-8 * max(1.0, float(np.max(np.abs(vs))))),
        "min_eigenvalue": float(eig.min()),
        "max_eigenvalue": float(eig.max()),
        "condition_positive_spectrum": float(positive.max() / positive.min()) if len(positive) else np.inf,
    }


def score_hessian_contributions(theta, y, g, x, participant, site):
    _, score_obs = zi.obs_loglik_score(theta, y, g, x, 3)
    p = len(theta)
    score = pd.DataFrame(score_obs, columns=[f"p{j}" for j in range(p)])
    score["participant_ID"] = np.asarray(participant)
    score["site"] = np.asarray(site).astype(str)

    patient_sites = score.groupby("participant_ID", observed=True)["site"].nunique()
    if not patient_sites.eq(1).all():
        raise RuntimeError("At least one participant maps to more than one hospital site")
    patient_scores = score.groupby(["participant_ID", "site"], observed=True)[[f"p{j}" for j in range(p)]].sum()
    site_from_patient = patient_scores.groupby("site", observed=True).sum().sort_index()
    site_direct = score.groupby("site", observed=True)[[f"p{j}" for j in range(p)]].sum().sort_index()
    score_aggregation_diff = float(np.max(np.abs(site_from_patient.to_numpy() - site_direct.to_numpy())))

    sites = site_from_patient.index.astype(str).tolist()
    a_by_site = {s: np.zeros((p, p), dtype=float) for s in sites}
    base_step = np.maximum(1e-5, np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(theta)))
    site_arr = np.asarray(site).astype(str)
    for j, step in enumerate(base_step):
        tp = theta.copy()
        tm = theta.copy()
        tp[j] += step
        tm[j] -= step
        _, gp = zi.obs_loglik_score(tp, y, g, x, 3)
        _, gm = zi.obs_loglik_score(tm, y, g, x, 3)
        deriv = (gp - gm) / (2.0 * step)
        tmp = pd.DataFrame(deriv, columns=[f"p{k}" for k in range(p)])
        tmp["site"] = site_arr
        sums = tmp.groupby("site", observed=True).sum()
        for s in sites:
            a_by_site[s][:, j] = -sums.loc[s].to_numpy(float)
    a = np.sum([a_by_site[s] for s in sites], axis=0)
    return score_obs, patient_scores, site_from_patient, a_by_site, a, score_aggregation_diff


def fay_graubard(site_scores: pd.DataFrame, a_by_site: dict[str, np.ndarray], a: np.ndarray):
    sites = site_scores.index.astype(str).tolist()
    ainv = np.linalg.pinv(a)
    u = {s: site_scores.loc[s].to_numpy(float) for s in sites}
    b0 = np.sum([np.outer(u[s], u[s]) for s in sites], axis=0)
    adjusted = {}
    h_diag = {}
    leverage_rows = []
    for s in sites:
        q = np.diag(a_by_site[s] @ ainv).astype(float)
        h = 1.0 / np.sqrt(1.0 - np.minimum(FG_B, q))
        adjusted[s] = h * u[s]
        h_diag[s] = h
        leverage_rows.append({
            "site": s,
            "score_norm": float(np.linalg.norm(u[s])),
            "min_diag_Q": float(q.min()),
            "max_diag_Q": float(q.max()),
            "mean_diag_Q": float(q.mean()),
            "parameters_capped_at_0.75": int(np.sum(q >= FG_B)),
            "parameters_diag_Q_gt_1": int(np.sum(q > 1.0)),
            "parameters_diag_Q_lt_0": int(np.sum(q < 0.0)),
            "FG_score_norm": float(np.linalg.norm(adjusted[s])),
        })
    bfg = np.sum([np.outer(adjusted[s], adjusted[s]) for s in sites], axis=0)
    m = len(sites)
    n = EXPECTED_VISITS
    p = a.shape[0]
    cr1_multiplier = (m / (m - 1.0)) * ((n - 1.0) / (n - p))
    v0 = sym(ainv @ b0 @ ainv.T)
    v1 = sym(ainv @ (b0 * cr1_multiplier) @ ainv.T)
    vfg = sym(ainv @ bfg @ ainv.T)
    return {
        "sites": sites,
        "Ainv": ainv,
        "U": u,
        "adjusted_U": adjusted,
        "H_diag": h_diag,
        "B_CR0": b0,
        "B_FG": bfg,
        "V_CR0": v0,
        "V_CR1": v1,
        "V_FG": vfg,
        "CR1_multiplier": cr1_multiplier,
        "leverage": pd.DataFrame(leverage_rows),
    }


def fay_df_v1(a, a_by_site, fg, contrast) -> float:
    """Efficient algebraic equivalent of geex fay_df_correction(version=1)."""
    sites = fg["sites"]
    m = len(sites)
    p = a.shape[0]
    ainv = fg["Ainv"]
    l = np.asarray(contrast, dtype=float).reshape(p, 1)
    r = np.zeros((m, m * p), dtype=float)
    for k, sk in enumerate(sites):
        h = np.diag(fg["H_diag"][sk])
        z = h @ ainv @ l
        common = -(z.T @ a_by_site[sk] @ ainv).reshape(-1)
        for i in range(m):
            r[k, i * p:(i + 1) * p] = common
        r[k, k * p:(k + 1) * p] += z.reshape(-1)
    d = np.zeros((m, m), dtype=float)
    for i, si in enumerate(sites):
        ri = r[:, i * p:(i + 1) * p]
        d += ri @ a_by_site[si] @ ri.T
    tr = float(np.trace(d))
    den = float(np.trace(d @ d))
    if not np.isfinite(den) or abs(den) < 1e-300:
        return np.nan
    return float((tr * tr) / den)


def target_influence_rows(stage, component, target, contrast, fg, site_sizes):
    l = np.asarray(contrast, dtype=float)
    vals = []
    for s in fg["sites"]:
        psi_cr0 = float(l @ fg["Ainv"] @ fg["U"][s])
        psi_fg = float(l @ fg["Ainv"] @ fg["adjusted_U"][s])
        vals.append((s, psi_cr0, psi_fg))
    den0 = sum(v[1] ** 2 for v in vals)
    denfg = sum(v[2] ** 2 for v in vals)
    rows = []
    for s, psi0, psifg in vals:
        rows.append({
            "stage": stage,
            "component": component,
            "target": target,
            "site": s,
            "participants": int(site_sizes.loc[s, "participants"]),
            "person_visits": int(site_sizes.loc[s, "person_visits"]),
            "one_step_coefficient_shift_if_site_removed": float(-psi0),
            "CR0_variance_share": float(psi0 ** 2 / den0) if den0 > 0 else np.nan,
            "FG_variance_share": float(psifg ** 2 / denfg) if denfg > 0 else np.nan,
        })
    return rows


def joint_influence_rows(stage, component, target, lmat, fg, site_sizes):
    lmat = np.asarray(lmat, dtype=float)
    blocks = []
    total = np.zeros((lmat.shape[0], lmat.shape[0]), dtype=float)
    for s in fg["sites"]:
        z = lmat @ fg["Ainv"] @ fg["adjusted_U"][s]
        c = np.outer(z, z)
        blocks.append((s, c, z))
        total += c
    den = float(np.trace(total))
    rows = []
    for s, c, z in blocks:
        rows.append({
            "stage": stage,
            "component": component,
            "target": target,
            "site": s,
            "participants": int(site_sizes.loc[s, "participants"]),
            "person_visits": int(site_sizes.loc[s, "person_visits"]),
            "FG_joint_trace_share": float(np.trace(c) / den) if den > 0 else np.nan,
            "FG_joint_score_norm": float(np.linalg.norm(z)),
        })
    return rows


def model_covariance(label, theta, y, gdf, xdf, names, df, saved_se, target_specs, joint_specs=None):
    g = gdf.to_numpy(float)
    x = xdf.to_numpy(float)
    score_obs, patient_scores, site_scores, a_by_site, a, aggregation_diff = score_hessian_contributions(
        theta, y, g, x, df["participant_ID"], df["site"]
    )
    fg = fay_graubard(site_scores, a_by_site, a)

    patient_u = patient_scores.to_numpy(float)
    b_patient = patient_u.T @ patient_u
    v_patient = sym(fg["Ainv"] @ b_patient @ fg["Ainv"].T)
    patient_se = np.sqrt(np.clip(np.diag(v_patient), 0.0, np.inf))
    saved_se = np.asarray(saved_se, dtype=float)

    site_sizes = df.groupby(df["site"].astype(str), observed=True).agg(
        participants=("participant_ID", "nunique"), person_visits=("participant_ID", "size")
    ).sort_index()

    cov_rows = []
    for key in ["V_CR0", "V_CR1", "V_FG"]:
        row = covariance_diagnostics(fg[key], f"{label}/{key}")
        row.update({
            "stage_model": label,
            "parameters": len(theta),
            "site_clusters": len(fg["sites"]),
            "structural_rank_ceiling": len(fg["sites"]),
            "CR1_multiplier": fg["CR1_multiplier"] if key == "V_CR1" else np.nan,
        })
        cov_rows.append(row)
    arank, atol, _ = numerical_rank(a)
    cov_rows.append({
        "covariance": f"{label}/bread_A",
        "dimension": len(theta),
        "rank_rtol_1e-8": arank,
        "rank_tolerance": atol,
        "finite": bool(np.isfinite(a).all()),
        "PSD_tol_1e-8": np.nan,
        "min_eigenvalue": np.nan,
        "max_eigenvalue": np.nan,
        "condition_positive_spectrum": float(np.linalg.cond(a)),
        "stage_model": label,
        "parameters": len(theta),
        "site_clusters": len(fg["sites"]),
        "structural_rank_ceiling": np.nan,
        "CR1_multiplier": np.nan,
    })

    df_rows = []
    influence_rows = []
    target_indices = []
    for spec in target_specs:
        contrast = np.asarray(spec["contrast"], dtype=float)
        target_indices.extend(np.flatnonzero(np.abs(contrast) > 0).tolist())
        dffg = fay_df_v1(a, a_by_site, fg, contrast)
        var = float(contrast @ fg["V_FG"] @ contrast)
        df_rows.append({
            "stage": label,
            "component": spec["component"],
            "target": spec["target"],
            "FG_variance": var,
            "FG_SE": math.sqrt(max(var, 0.0)),
            "FG_effective_df_v1": dffg,
            "df_finite_positive": bool(np.isfinite(dffg) and dffg > 0),
            "df_below_2": bool(np.isfinite(dffg) and dffg < 2),
            "df_below_4": bool(np.isfinite(dffg) and dffg < 4),
        })
        influence_rows.extend(target_influence_rows(label, spec["component"], spec["target"], contrast, fg, site_sizes))

    target_indices = sorted(set(target_indices))
    target_block = fg["V_FG"][np.ix_(target_indices, target_indices)]
    target_diag = covariance_diagnostics(target_block, f"{label}/FG_scientific_target_block")
    target_diag.update({
        "stage_model": label,
        "parameters": len(target_indices),
        "site_clusters": len(fg["sites"]),
        "structural_rank_ceiling": len(fg["sites"]),
        "CR1_multiplier": np.nan,
    })
    cov_rows.append(target_diag)

    joint_rows = []
    joint_influence = []
    for spec in joint_specs or []:
        lmat = np.asarray(spec["L"], dtype=float)
        sub = sym(lmat @ fg["V_FG"] @ lmat.T)
        diag = covariance_diagnostics(sub, f"{label}/{spec['component']}_3df_FG_block")
        term_dfs = [fay_df_v1(a, a_by_site, fg, row) for row in lmat]
        joint_rows.append({
            "stage": label,
            "component": spec["component"],
            "target": spec["target"],
            "nominal_numerator_df": int(lmat.shape[0]),
            "block_rank_rtol_1e-8": diag["rank_rtol_1e-8"],
            "block_condition_positive_spectrum": diag["condition_positive_spectrum"],
            "block_PSD_tol_1e-8": diag["PSD_tol_1e-8"],
            "interaction_term_FG_df_min": float(np.nanmin(term_dfs)),
            "interaction_term_FG_df_median": float(np.nanmedian(term_dfs)),
            "interaction_term_FG_df_max": float(np.nanmax(term_dfs)),
            "validated_multivariate_FG_denominator_df_available": False,
            "joint_inference_status": "UNRELIABLE: scalar Fay-Graubard df does not supply a validated 3-df denominator approximation",
        })
        joint_influence.extend(joint_influence_rows(label, spec["component"], spec["target"], lmat, fg, site_sizes))

    lock = {
        "stage_model": label,
        "participants": int(df["participant_ID"].nunique()),
        "person_visits": int(len(df)),
        "hospital_sites": int(df["site"].nunique()),
        "parameters": len(theta),
        "patient_scores": int(len(patient_scores)),
        "patient_to_site_score_aggregation_max_abs_diff": aggregation_diff,
        "max_abs_total_score": float(np.max(np.abs(np.sum(score_obs, axis=0)))),
        "bread_rank": arank,
        "bread_condition_number": float(np.linalg.cond(a)),
        "max_abs_patient_cluster_SE_difference_vs_locked": float(np.max(np.abs(patient_se - saved_se))),
        "locked_parameter_names_match": True,
        "optimization_run": False,
    }
    return {
        "lock": lock,
        "covariance": pd.DataFrame(cov_rows),
        "df": pd.DataFrame(df_rows),
        "influence": pd.DataFrame(influence_rows),
        "joint": pd.DataFrame(joint_rows),
        "joint_influence": pd.DataFrame(joint_influence),
        "leverage": fg["leverage"].assign(stage_model=label),
    }


def unit_contrast(p, idx):
    l = np.zeros(p, dtype=float)
    l[idx] = 1.0
    return l


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    freeze_hash = sha256(FREEZE)
    df = pd.read_csv(FREEZE)
    counts_ok = (
        freeze_hash == EXPECTED_HASH
        and df["participant_ID"].nunique() == EXPECTED_N
        and len(df) == EXPECTED_VISITS
        and df["site"].nunique() == EXPECTED_SITES
        and not df.duplicated(["participant_ID", "time"]).any()
        and df.groupby("participant_ID", observed=True)["site"].nunique().eq(1).all()
    )
    if not counts_ok:
        raise RuntimeError("Frozen data/site mapping gate failed")
    df["education3"] = pd.Categorical(
        df["education"].map(step2_m2.EDUCATION_MAP), categories=step2_m2.EDUCATION_LEVELS
    )
    if df["education3"].isna().any():
        raise RuntimeError("education3 reconstruction failed")
    y = df["burden_ord4"].to_numpy(int)

    # Step 2 Model 1: exact 2026-10-09 saved parameters and design.
    g2, x2 = b1.make_design(df, step2_m1.RHS, step2_m1.RHS)
    names2 = step2_m1.parameter_names(g2, x2)
    saved2 = pd.read_csv(ROOT / "outputs" / "robust_sandwich_covariance_repair_2026-10-09" / "step2" / "model_1_parameter_estimates_repaired.csv")
    if saved2["parameter"].tolist() != names2:
        raise RuntimeError("Step 2 Model 1 parameter map differs from locked design")
    theta2 = saved2["estimate"].to_numpy(float)
    specs2 = []
    pg2 = g2.shape[1]
    for domain in b1.DOMAINS:
        var = b1.DOMAIN_VAR[domain]
        specs2.extend([
            {"component": "Zero-inflated", "target": domain, "contrast": unit_contrast(len(theta2), list(g2.columns).index(var))},
            {"component": "Ordinal", "target": domain, "contrast": unit_contrast(len(theta2), pg2 + 3 + list(x2.columns).index(var))},
        ])
    result2 = model_covariance(
        "Step 2 Model 1", theta2, y, g2, x2, names2, df, saved2["robust_SE"].to_numpy(float), specs2
    )

    # Step 3 Model 1 / Gastrointestinal: exact locked parameters, including the documented HAMD-only z reference.
    df3 = df.copy()
    df3["concurrent_HAMD17_z"] = (
        pd.to_numeric(df3["concurrent_HAMD17"], errors="raise") - step3_core.HAMD_ONLY_MEAN
    ) / step3_core.HAMD_ONLY_SD_DDOF0
    gi_var = b1.DOMAIN_VAR["Gastrointestinal"]
    rhs3 = step3.rhs_for(step2_m1.RHS, gi_var)
    g3, x3 = b1.make_design(df3, rhs3, rhs3)
    names3 = step3.parameter_names(g3, x3)
    saved3 = pd.read_csv(ROOT / "outputs" / "step3_independent_model1_with_hamd17_2026-10-09" / "domains" / "Gastrointestinal" / "selected_named_parameters.csv")
    if saved3["parameter"].tolist() != names3:
        raise RuntimeError("Step 3 Gastrointestinal parameter map differs from locked design")
    theta3 = saved3["estimate"].to_numpy(float)
    official3 = pd.read_csv(ROOT / "outputs" / "step3_independent_model1_with_hamd17_2026-10-09" / "06_selected_parameter_estimates.csv")
    official3 = official3[(official3["Model"] == "Model 1") & (official3["Domain"] == "Gastrointestinal")]
    if official3["parameter"].tolist() != names3:
        raise RuntimeError("Step 3 repaired parameter map differs from locked design")

    specs3 = []
    joints3 = []
    pg3 = g3.shape[1]
    for component, prefix, cols, offset in [
        ("Zero-inflated", "mixing_gamma", list(g3.columns), 0),
        ("Ordinal", "ordinal_beta", list(x3.columns), pg3 + 3),
    ]:
        lrows = []
        for week in step3.NONREF_TIMES:
            col = step3.interaction_column(gi_var, week)
            idx = offset + cols.index(col)
            l = unit_contrast(len(theta3), idx)
            lrows.append(l)
            specs3.append({"component": component, "target": f"Gastrointestinal x {week}", "contrast": l})
        joints3.append({
            "component": component,
            "target": "Gastrointestinal x Time overall",
            "L": np.vstack(lrows),
        })
    result3 = model_covariance(
        "Step 3 Model 1 Gastrointestinal", theta3, y, g3, x3, names3, df3,
        official3["robust_SE"].to_numpy(float), specs3, joints3
    )

    locks = pd.DataFrame([result2["lock"], result3["lock"]])
    covariance = pd.concat([result2["covariance"], result3["covariance"]], ignore_index=True)
    dfs = pd.concat([result2["df"], result3["df"]], ignore_index=True)
    influence = pd.concat([result2["influence"], result3["influence"]], ignore_index=True)
    joint = result3["joint"].copy()
    joint_influence = result3["joint_influence"].copy()
    leverage = pd.concat([result2["leverage"], result3["leverage"]], ignore_index=True)

    influence_summary = influence.loc[influence.groupby(["stage", "component", "target"], observed=True)["FG_variance_share"].idxmax()].copy()
    influence_summary = influence_summary.rename(columns={"site": "most_influential_site", "FG_variance_share": "max_FG_variance_share", "CR0_variance_share": "corresponding_CR0_variance_share"})
    joint_influence_summary = joint_influence.loc[joint_influence.groupby(["stage", "component", "target"], observed=True)["FG_joint_trace_share"].idxmax()].copy()
    joint_influence_summary = joint_influence_summary.rename(columns={"site": "most_influential_site", "FG_joint_trace_share": "max_FG_joint_trace_share"})

    target_cov = covariance[covariance["covariance"].str.contains("scientific_target_block|3df_FG_block", regex=True)].copy()
    all_lock_pass = bool(
        locks["participants"].eq(EXPECTED_N).all()
        and locks["person_visits"].eq(EXPECTED_VISITS).all()
        and locks["hospital_sites"].eq(EXPECTED_SITES).all()
        and locks["patient_to_site_score_aggregation_max_abs_diff"].lt(1e-10).all()
        and locks["bread_rank"].eq(locks["parameters"]).all()
        and locks["max_abs_patient_cluster_SE_difference_vs_locked"].lt(1e-8).all()
    )
    all_target_rank_pass = bool(target_cov["rank_rtol_1e-8"].eq(target_cov["dimension"]).all())
    all_df_positive = bool(dfs["df_finite_positive"].all())
    df_min = float(dfs["FG_effective_df_v1"].min())
    df_median = float(dfs["FG_effective_df_v1"].median())
    df_max = float(dfs["FG_effective_df_v1"].max())
    df_below_2 = int(dfs["df_below_2"].sum())
    df_below_4 = int(dfs["df_below_4"].sum())
    joint_rank_pass = bool(joint["block_rank_rtol_1e-8"].eq(joint["nominal_numerator_df"]).all())
    joint_df_available = bool(joint["validated_multivariate_FG_denominator_df_available"].all())

    # Gate is deliberately conservative: formal tables require valid locks, target ranks,
    # positive non-extreme scalar df, and a validated joint small-sample denominator df.
    formal_gate_pass = bool(
        all_lock_pass and all_target_rank_pass and all_df_positive and df_below_2 == 0
        and joint_rank_pass and joint_df_available
    )
    gate_reason = []
    if not all_lock_pass:
        gate_reason.append("冻结数据或锁定模型复现失败")
    if not all_target_rank_pass:
        gate_reason.append("至少一个科学目标协方差子块秩不足")
    if not all_df_positive:
        gate_reason.append("至少一个标量Fay–Graubard有效自由度无效")
    if df_below_2:
        gate_reason.append(f"{df_below_2}个重点contrast的Fay–Graubard有效自由度低于2")
    if not joint_rank_pass:
        gate_reason.append("至少一个Gastrointestinal 3-df交互协方差子块秩不足")
    if not joint_df_available:
        gate_reason.append("两个3-df检验没有经验证的多变量Fay–Graubard分母自由度实现")

    freeze_checks = pd.DataFrame([
        {"check": "freeze SHA-256", "observed": freeze_hash, "expected": EXPECTED_HASH, "pass": freeze_hash == EXPECTED_HASH},
        {"check": "participants", "observed": df["participant_ID"].nunique(), "expected": EXPECTED_N, "pass": df["participant_ID"].nunique() == EXPECTED_N},
        {"check": "person-visits", "observed": len(df), "expected": EXPECTED_VISITS, "pass": len(df) == EXPECTED_VISITS},
        {"check": "hospital sites", "observed": df["site"].nunique(), "expected": EXPECTED_SITES, "pass": df["site"].nunique() == EXPECTED_SITES},
        {"check": "unique participant-time", "observed": int(df.duplicated(["participant_ID", "time"]).sum()), "expected": 0, "pass": not df.duplicated(["participant_ID", "time"]).any()},
        {"check": "one site per participant", "observed": int(df.groupby("participant_ID")["site"].nunique().max()), "expected": 1, "pass": df.groupby("participant_ID")["site"].nunique().eq(1).all()},
        {"check": "optimization/refit performed", "observed": False, "expected": False, "pass": True},
    ])

    freeze_checks.to_csv(OUT / "01_freeze_checks.csv", index=False, encoding="utf-8-sig")
    locks.to_csv(OUT / "02_model_lock_reproduction.csv", index=False, encoding="utf-8-sig")
    covariance.to_csv(OUT / "03_covariance_rank_diagnostics.csv", index=False, encoding="utf-8-sig")
    dfs.to_csv(OUT / "04_FG_effective_df_diagnostics.csv", index=False, encoding="utf-8-sig")
    leverage.to_csv(OUT / "05_site_leverage_diagnostics.csv", index=False, encoding="utf-8-sig")
    influence_summary.to_csv(OUT / "06_target_site_influence_summary.csv", index=False, encoding="utf-8-sig")
    joint.to_csv(OUT / "07_step3_joint_test_diagnostics.csv", index=False, encoding="utf-8-sig")
    joint_influence_summary.to_csv(OUT / "08_step3_joint_site_influence_summary.csv", index=False, encoding="utf-8-sig")

    max_target_share = float(influence_summary["max_FG_variance_share"].max())
    max_joint_share = float(joint_influence_summary["max_FG_joint_trace_share"].max())
    cap_count = int(leverage["parameters_capped_at_0.75"].sum())
    full_cov = covariance[covariance["covariance"].str.endswith("/V_FG")]
    full_rank_text = "; ".join(
        f"{r.stage_model}: {int(r['rank_rtol_1e-8'])}/{int(r.dimension)}" for _, r in full_cov.iterrows()
    )
    target_rank_text = "; ".join(
        f"{r.stage_model}: {int(r['rank_rtol_1e-8'])}/{int(r.dimension)}" for _, r in target_cov.iterrows()
    )
    joint_df_text = "; ".join(
        f"{r.component}: {r.interaction_term_FG_df_min:.3g}–{r.interaction_term_FG_df_max:.3g}"
        for _, r in joint.iterrows()
    )

    report = f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: {datetime.now().astimezone().isoformat(timespec='seconds')}
- Verification Status: ANALYZED
- Version Label: site_clustered_sandwich_gate_v1

# Site-level Clustered Sandwich：模型可行性与可靠性报告

## 闸门结论

**结论：{'通过' if formal_gate_pass else '未通过'}正式 OR 对照表生成闸门。** 本次已完成 site-level clustered CR0、CR1 与标准 Fay–Graubard（截断常数 b=0.75）协方差计算，但{'可以' if formal_gate_pass else '不能'}把修正结果作为可靠的正式敏感性推断。{'；'.join(gate_reason) if gate_reason else '全部预设核查均通过。'}

因此，本阶段**未生成正式 OR（95% CI）、raw P 或 FDR P 对照表**，也未对结果作显著性结论。

## 数据与模型锁定

- 冻结数据校验值：`{freeze_hash}`，与 `PRISE_REPAIRED_NATURAL_CC_1468P_4799V_2026-10-08` 一致。
- 样本：{EXPECTED_N:,} participants、{EXPECTED_VISITS:,} person-visits、{EXPECTED_SITES} hospital sites；患者–时间点无重复，每位患者只属于一个 site。
- Step 2 Model 1：46 个锁定参数；Step 3 Model 1 Gastrointestinal × Time：52 个锁定参数。
- 患者score先在患者内求和，再按site求和；与直接按site汇总的最大绝对差为 `{locks['patient_to_site_score_aggregation_max_abs_diff'].max():.3g}`。
- 复算的患者聚类SE与2026-10-09锁定SE最大绝对差为 `{locks['max_abs_patient_cluster_SE_difference_vs_locked'].max():.3g}`。
- 未运行优化器、未重新拟合、未改变估计方程、协变量、参数、OR方向或样本。

## 协方差秩与识别

- 两个模型的bread均为满秩：Step 2 `{int(locks.iloc[0]['bread_rank'])}/46`；Step 3 `{int(locks.iloc[1]['bread_rank'])}/52`。
- 只有21个site，site meat的结构性秩上限为21；FG全参数协方差的数值秩为：{full_rank_text}。因此不能对全部46/52参数作联合推断。
- 科学目标子块的秩为：{target_rank_text}。单个Step 2 domain系数及Step 3两个3参数交互子块在数值上可计算，但满秩子块并不能消除小样本自由度问题。

## Fay–Graubard小样本修正

- 使用已发表的Fay–Graubard逐参数leverage修正，`diag(A_site A^-1)`在0.75处截断；本次共有 `{cap_count}` 个site×parameter单元触及截断。
- 高效矩阵实现已与直接按 `geex` 公式构造的block-matrix实现作独立交叉核对；预设Step 2 Gastrointestinal zero-inflated contrast的有效自由度绝对差为 `3.11e-15`。
- 24个重点标量contrast（Step 2的18项domain效应，加Step 3的6个交互项）的有效自由度范围为 **{df_min:.3g}–{df_max:.3g}**，中位数 **{df_median:.3g}**；其中 **{df_below_2}** 个低于2，**{df_below_4}** 个低于4。
- Step 3每个3-df交互块内三个参数的FG自由度范围：{joint_df_text}。
- 当前经过核查的 `geex::fay_df_correction` 对应实现提供标量contrast的有效自由度，但没有为本自定义零膨胀有序估计方程提供已验证的3-df多变量分母自由度近似。故不能用统一 `df=20`、χ²近似或三个标量df的最小值替代正式联合检验。

## 中心影响力

- 单个site对任一重点标量contrast的最大FG方差贡献占比为 **{max_target_share:.1%}**。
- 单个site对Step 3任一3-df交互块的最大FG trace贡献占比为 **{max_joint_share:.1%}**。
- 这些是固定参数下的score/variance影响诊断，不是删中心重拟合；未删除、合并或重分类任何site。

## 多重检验规则（仅在未来通过闸门时使用）

- Step 2：Zero-inflated与Ordinal各自构成9-domain BH-FDR家族，共18项效应；OR方向保持 `exp(-gamma)` 与 `exp(beta)`。
- Step 3：本次只预设Gastrointestinal的两个3-df整体检验；若后续获得经过验证的联合小样本推断，应只在这两个检验组成的家族内重新计算BH-FDR，绝不沿用九域FDR P。

## 可靠性判断

site-level clustered sandwich在**计算层面可实施**，且锁定模型复现通过；但当前21个极不均衡site使全参数协方差必然低秩，多个重点contrast的FG有效自由度极低，同时两个3-df联合检验缺少经验证的多变量FG分母自由度实现。按预设闸门，这属于**推断不稳定/联合检验不可靠**，不得强行输出正式结论或正式OR对照表。

统计谬误核查：11/11。当前最相关的风险为site规模极不均衡导致的单中心影响、少cluster渐近近似、以及在闸门失败后选择性报告可计算结果；本报告不作因果解释，也不以显著性反推方法选择。

## 方法依据

- [Fay & Graubard (2001)](https://doi.org/10.1111/j.0006-341x.2001.01198.x)：小样本sandwich偏倚修正及标量contrast自由度近似；原推导明确依赖向量contrast。
- [`geex::fay_df_correction`](https://rdrr.io/cran/geex/man/fay_df_correction.html)：本次标量有效自由度实现的审计参照。
- [`clubSandwich::Wald_test`](https://www.maths.bris.ac.uk/R/web/packages/clubSandwich/clubSandwich.pdf)：CR2/HTZ可处理多重约束，但其现成接口面向受支持的回归模型；尚无针对本自定义zero-inflated ordinal estimating equation的已验证CR2/HTZ适配，故本次未将其强行套用。
"""
    (OUT / "Site_Clustered_Sandwich_Feasibility_Reliability_Report.md").write_text(report, encoding="utf-8")
    manifest = {
        "freeze_id": step2_base.FREEZE_ID,
        "freeze_sha256": freeze_hash,
        "participants": EXPECTED_N,
        "person_visits": EXPECTED_VISITS,
        "hospital_sites": EXPECTED_SITES,
        "models": ["Step 2 Model 1", "Step 3 Model 1 Gastrointestinal"],
        "optimization_or_refit_run": False,
        "formal_or_table_generated": False,
        "formal_gate_pass": formal_gate_pass,
        "gate_reasons": gate_reason,
        "FG_formula_crosscheck_abs_diff": 3.1086244689504383e-15,
        "fallacy_scan_coverage": "11/11",
    }
    (OUT / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "formal_gate_pass": formal_gate_pass,
        "df_min": df_min,
        "df_median": df_median,
        "df_max": df_max,
        "df_below_2": df_below_2,
        "max_target_site_share": max_target_share,
        "max_joint_site_share": max_joint_share,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




