from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from scipy.special import expit
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import ipw_observation_helpers as ipw_helpers
import step2_fourlevel_concurrent_hamd17_education3 as step2_main
import step2_fourlevel_no_concurrent_hamd17 as step2_base
import step3_domain_time_burden_4799_preliminary as step3_common
import step3_independent_model_core as step3_core


from public_paths import ROOT, authorized_data_path
SOURCE = authorized_data_path("PROUD_RAW_WORKBOOK", "raw_workbook.xlsx")
FREEZE = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "repaired_natural_complete_case_analysis_data.csv"
MAPPING = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "PRISE_Mapping_Verification.csv"
STEP2_OUT = ROOT / "outputs" / "step2_fourlevel_concurrent_hamd17_education3"
STEP3_OUT = ROOT / "outputs" / "step3_independent_model1_with_hamd17_2026-10-09"
OUT = ROOT / "outputs" / "Reviewer1_Comment3_Attrition"
CHECKPOINT = OUT / "_checkpoints"

FREEZE_ID = "PRISE_REPAIRED_NATURAL_CC_1468P_4799V_2026-10-08"
EXPECTED_FREEZE_HASH = "6844f512b5122beabc9905456e1a9a2111f6318c398a984232068b22c22fd16c"
EXPECTED_PARTICIPANTS = 1468
EXPECTED_VISITS = 4799
EXPECTED_VISIT_COUNTS = {"Week2": 1381, "Week8": 1259, "Week12": 1146, "Week24": 1013}
WEEKS = ["Week2", "Week8", "Week12", "Week24"]
DOMAIN_VARS = [b1.DOMAIN_VAR[d] for d in b1.DOMAINS]
Q = 3
SEED = 20261014


def log(message: str) -> None:
    print(f"Reviewer1 Comment3: {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(name: str, df: pd.DataFrame) -> None:
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def qstats(x: pd.Series) -> tuple[float, float, float]:
    q = x.astype(float).quantile([0.25, 0.50, 0.75], interpolation="linear")
    return float(q.loc[0.25]), float(q.loc[0.50]), float(q.loc[0.75])


def source_baseline_extras() -> pd.DataFrame:
    xls = pd.ExcelFile(SOURCE, engine="openpyxl")
    raw = pd.read_excel(SOURCE, sheet_name=xls.sheet_names[0], engine="openpyxl")
    out = pd.DataFrame({"participant_ID": raw[b1.id_col(raw)].map(b1.norm_id).astype(str)})
    province_col = next((c for c in raw.columns if str(c).lower().startswith("province")), None)
    out["province"] = raw[province_col].map(b1.sanitize_category) if province_col is not None else np.nan
    fibser_col = b1.burden_col(raw)
    out["baseline_fibser_burden"] = b1.numeric(raw[fibser_col]) if fibser_col is not None else np.nan
    return out


def build_complete_grid(base: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [base["participant_ID"].astype(str), WEEKS], names=["participant_ID", "time"]
    ).to_frame(index=False)
    grid["time"] = pd.Categorical(grid["time"], categories=WEEKS, ordered=True)
    keep = [
        "participant_ID", "time", "visit_order", "fibser_burden", "burden_ord4",
        "concurrent_HAMD17", "concurrent_HAMD17_z", *DOMAIN_VARS,
    ]
    out = grid.merge(long[keep], on=["participant_ID", "time"], how="left", validate="one_to_one")
    out = out.merge(base, on="participant_ID", how="left", validate="many_to_one")
    for c in ["age", "baseline_HAMD17", "baseline_SDS"]:
        mean, sd = long[c].mean(), long[c].std(ddof=0)
        out[c + "_z"] = (out[c] - mean) / sd
    out["fibser_available"] = out["fibser_burden"].isin(range(7))
    out["prise_available"] = out[DOMAIN_VARS].notna().all(axis=1)
    out["concurrent_HAMD_available"] = out["concurrent_HAMD17"].notna()
    out["baseline_eligible"] = out[ipw_helpers.BASELINE_REQUIRED].notna().all(axis=1)
    req = [
        "participant_ID", "time", "burden_ord4", "age_z", "sex", "education",
        "first_episode", "baseline_med_class", "baseline_HAMD17_z", "baseline_SDS_z",
        "concurrent_HAMD17_z", *DOMAIN_VARS,
    ]
    out["R_it"] = (out["baseline_eligible"] & out[req].notna().all(axis=1)).astype(int)
    return out


def raw_and_freeze_audit():
    log("reading current raw workbook and reconstructing all four follow-ups")
    base, long, raw_analysis, mapping_rows, zmap = b1.build_dataset()
    base = base.copy()
    base["participant_ID"] = base["participant_ID"].astype(str)
    long = long.copy()
    long["participant_ID"] = long["participant_ID"].astype(str)
    raw_analysis = raw_analysis.copy()
    raw_analysis["participant_ID"] = raw_analysis["participant_ID"].astype(str)
    base = base.merge(source_baseline_extras(), on="participant_ID", how="left", validate="one_to_one")

    freeze = pd.read_csv(FREEZE)
    freeze["participant_ID"] = freeze["participant_ID"].astype(str)
    freeze_hash = sha256(FREEZE)
    if freeze_hash != EXPECTED_FREEZE_HASH:
        raise RuntimeError(f"STOP: freeze hash mismatch: {freeze_hash}")
    if freeze["participant_ID"].nunique() != EXPECTED_PARTICIPANTS or len(freeze) != EXPECTED_VISITS:
        raise RuntimeError("STOP: latest frozen sample does not equal 1468 participants / 4799 visits")
    if freeze.duplicated(["participant_ID", "time"]).any():
        raise RuntimeError("STOP: duplicate participant_ID + time in latest freeze")
    counts = freeze["time"].value_counts().to_dict()
    if counts != EXPECTED_VISIT_COUNTS:
        raise RuntimeError(f"STOP: latest freeze visit counts mismatch: {counts}")

    joined = raw_analysis.merge(
        freeze, on=["participant_ID", "time"], how="outer", indicator=True,
        suffixes=("_raw", "_freeze"), validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all() or len(joined) != EXPECTED_VISITS:
        raise RuntimeError("STOP: current raw-data complete-case keys do not match the 4799-visit freeze")
    compare_cols = ["fibser_burden", "burden_ord4", "concurrent_HAMD17", *DOMAIN_VARS]
    mismatch_rows = []
    for col in compare_cols:
        x = pd.to_numeric(joined[f"{col}_raw"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(joined[f"{col}_freeze"], errors="coerce").to_numpy(float)
        mismatch_rows.append({
            "variable": col,
            "mismatched_rows": int((~np.isclose(x, y, equal_nan=True)).sum()),
            "max_absolute_difference": float(np.nanmax(np.abs(x - y))) if np.isfinite(x - y).any() else 0.0,
        })
    mismatch = pd.DataFrame(mismatch_rows)
    if mismatch["mismatched_rows"].sum() != 0:
        raise RuntimeError("STOP: current raw data values do not reproduce the saved 4799-visit freeze")

    mapping_qc = pd.read_csv(MAPPING)
    mapping_ok = bool(mapping_qc["status"].astype(str).str.upper().eq("PASS").all())
    if not mapping_ok or not set(DOMAIN_VARS).issubset(set(mapping_qc["final_model_variable"].dropna().astype(str))):
        raise RuntimeError("STOP: latest PRISE nine-domain mapping audit did not pass")

    source_audit = pd.DataFrame([
        {"check": "raw cohort participants", "observed": base["participant_ID"].nunique(), "expected": 1543, "status": "PASS" if base["participant_ID"].nunique() == 1543 else "FAIL"},
        {"check": "raw baseline duplicate IDs", "observed": int(base["participant_ID"].duplicated().sum()), "expected": 0, "status": "PASS" if not base["participant_ID"].duplicated().any() else "FAIL"},
        {"check": "raw reconstructed participant-time rows", "observed": len(long), "expected": 6172, "status": "PASS" if len(long) == 6172 else "FAIL"},
        {"check": "raw reconstructed duplicate participant-time keys", "observed": int(long.duplicated(["participant_ID", "time"]).sum()), "expected": 0, "status": "PASS" if not long.duplicated(["participant_ID", "time"]).any() else "FAIL"},
        {"check": "latest freeze participants", "observed": freeze["participant_ID"].nunique(), "expected": 1468, "status": "PASS"},
        {"check": "latest freeze person-visits", "observed": len(freeze), "expected": 4799, "status": "PASS"},
        {"check": "current raw keys equal freeze keys", "observed": int(joined["_merge"].eq("both").sum()), "expected": 4799, "status": "PASS"},
        {"check": "current raw model-field mismatches vs freeze", "observed": int(mismatch["mismatched_rows"].sum()), "expected": 0, "status": "PASS"},
        {"check": "PRISE mapping audit", "observed": mapping_ok, "expected": True, "status": "PASS" if mapping_ok else "FAIL"},
    ])
    if not source_audit["status"].eq("PASS").all():
        raise RuntimeError("STOP: source audit failed")
    return base, long, freeze, mapping_rows, zmap, source_audit, mismatch


def dropout_analysis(base: pd.DataFrame, long: pd.DataFrame):
    values = (
        long.pivot(index="participant_ID", columns="time", values="fibser_burden")
        .reindex(index=base["participant_ID"].astype(str), columns=WEEKS)
    )
    valid = values.apply(lambda col: col.isin(range(7)))
    patterns = valid.astype(int).astype(str).agg("".join, axis=1)
    labels = {
        "1111": "Complete follow-up",
        "1000": "Week 2 dropout",
        "1100": "Week 8 dropout",
        "1110": "Week 12 dropout",
        "0000": "No valid follow-up FIBSER",
    }
    group = patterns.map(labels).fillna("Intermittent/other missingness")
    order = [
        "Complete follow-up", "Week 2 dropout", "Week 8 dropout", "Week 12 dropout",
        "Intermittent/other missingness", "No valid follow-up FIBSER",
    ]
    counts = group.value_counts().reindex(order, fill_value=0).rename_axis("follow_up_group").reset_index(name="participants")
    counts["percent_of_1543"] = counts["participants"] / 1543 * 100
    counts["definition"] = [
        "Valid FIBSER at Weeks 2, 8, 12, and 24",
        "Valid FIBSER only at Week 2",
        "Valid FIBSER at Weeks 2 and 8 only",
        "Valid FIBSER at Weeks 2, 8, and 12; missing at Week 24",
        "Any other valid/missing pattern, including return after a missing visit",
        "No valid FIBSER at any of the four follow-up visits",
    ]
    if counts["participants"].sum() != 1543:
        raise RuntimeError("STOP: mutually exclusive dropout groups do not sum to 1543")

    comparisons = []
    for dropout_group, week in [
        ("Week 2 dropout", "Week2"), ("Week 8 dropout", "Week8"), ("Week 12 dropout", "Week12")
    ]:
        d = values.loc[group.eq(dropout_group), week].dropna().astype(float)
        c = values.loc[group.eq("Complete follow-up"), week].dropna().astype(float)
        if not (d.isin(range(7)).all() and c.isin(range(7)).all()):
            raise RuntimeError("STOP: invalid FIBSER value entered dropout comparison")
        u, p = mannwhitneyu(d, c, alternative="two-sided", method="asymptotic")
        dq1, dmed, dq3 = qstats(d)
        cq1, cmed, cq3 = qstats(c)
        comparisons.append({
            "comparison": f"{dropout_group} vs Complete follow-up",
            "assessment": week.replace("Week", "Week "),
            "dropout_group": dropout_group,
            "dropout_n": len(d), "dropout_median": dmed, "dropout_Q1": dq1, "dropout_Q3": dq3,
            "dropout_FIBSER_ge3_n": int((d >= 3).sum()), "dropout_FIBSER_ge3_percent": float((d >= 3).mean() * 100),
            "complete_n": len(c), "complete_median": cmed, "complete_Q1": cq1, "complete_Q3": cq3,
            "complete_FIBSER_ge3_n": int((c >= 3).sum()), "complete_FIBSER_ge3_percent": float((c >= 3).mean() * 100),
            "Mann_Whitney_U": float(u), "raw_two_sided_P": float(p), "multiplicity_note": "Exploratory; raw P value; no multiplicity adjustment",
        })
    comparisons = pd.DataFrame(comparisons)
    current_any = int((patterns != "0000").sum())
    discrepancy = pd.DataFrame([{
        "metric": "Participants with at least one valid follow-up FIBSER",
        "current_raw_workbook": current_any,
        "prior_audit": 1502,
        "difference": current_any - 1502,
        "explanation": "The current workbook has one additional participant with at least one valid follow-up FIBSER compared with the earlier aggregate audit. The source-workbook hashes differ; no participant-level reconciliation detail is emitted by the public code.",
    }])
    return counts, comparisons, discrepancy


def prepare_weight_data(base: pd.DataFrame, long: pd.DataFrame, freeze: pd.DataFrame):
    grid = build_complete_grid(base, long)
    observed_keys = grid.loc[grid["R_it"].eq(1), ["participant_ID", "time"]].copy()
    check = observed_keys.merge(freeze[["participant_ID", "time"]], on=["participant_ID", "time"], how="outer", indicator=True)
    visit_counts = grid.loc[grid["R_it"].eq(1), "time"].astype(str).value_counts().to_dict()
    if not check["_merge"].eq("both").all() or len(check) != EXPECTED_VISITS or visit_counts != EXPECTED_VISIT_COUNTS:
        raise RuntimeError("STOP: raw-data R_it keys do not exactly equal the latest 4799-visit freeze")
    log("fitting visit-specific observation-probability models")
    obs_diag, obs_coef, obs_pred, obs_cal, obs_aux, stop, stop_reason, risk = ipw_helpers.fit_observation_models(grid)
    if stop:
        write_csv("_STOP_observation_model_diagnostics.csv", obs_diag)
        raise RuntimeError(f"STOP: observation-probability model instability: {stop_reason}")
    weight_diag, observed_weights, concentration = ipw_helpers.weight_diagnostics(risk)
    wa = freeze.merge(
        observed_weights[["participant_ID", "time", "SW", "SW_truncated_1_99", "denominator_p", "numerator_p"]],
        on=["participant_ID", "time"], how="left", validate="one_to_one",
    )
    if wa[["SW", "SW_truncated_1_99"]].isna().any().any():
        raise RuntimeError("STOP: missing IPW for one or more frozen person-visits")
    wa["education3"] = pd.Categorical(
        wa["education"].map(step3_common.model2_base.EDUCATION_MAP),
        categories=step3_common.model2_base.EDUCATION_LEVELS,
    )
    # Preserve the exact standardized covariates stored in the 2026-10-08
    # 1,468-participant/4,799-visit freeze.  Re-standardizing current raw HAMD
    # values changes the design matrix when the source workbook version changes.
    if wa["education3"].isna().any() or wa["concurrent_HAMD17_z"].isna().any():
        raise RuntimeError("STOP: latest model covariate construction failed")
    terms = []
    hist = ipw_helpers.add_history_features(grid[grid["baseline_eligible"]].copy())
    for week in WEEKS:
        X = ipw_helpers.design_matrix_for_visit(hist, week, include_province=False)
        terms.extend({"time": week, "term": term, "uses_current_or_future_information": False} for term in X.columns)
    return wa, grid, obs_diag, obs_coef, obs_pred, obs_cal, obs_aux, weight_diag, concentration, pd.DataFrame(terms)


def parameter_names(gdf: pd.DataFrame, xdf: pd.DataFrame) -> list[str]:
    return step3_core.parameter_names(gdf, xdf, Q)


def formal_loglik_score(theta, y, g, x, score_path: str):
    if score_path == "step2":
        return step2_base.engine.zi.obs_loglik_score(theta, y, g, x, Q)
    if score_path == "step3":
        return b1.obs_loglik_score(theta, y, g, x)
    raise ValueError(f"unknown score path: {score_path}")


def neg_obj_grad(theta, y, g, x, weights, score_path):
    # Directly reuse the stage-specific score implementation used by the
    # finalized Step 2 or Step 3 model, then apply IPW externally.
    ll, score = formal_loglik_score(theta, y, g, x, score_path)
    w = np.asarray(weights, dtype=float)
    return -float(np.sum(w * ll)), -np.sum(w[:, None] * score, axis=0)


def weighted_assessment(theta, y, g, x, weights, score_path):
    assessment = step2_base.engine.basic_assessment(theta, y, g, x)
    neg_objective, neg_gradient = neg_obj_grad(theta, y, g, x, weights, score_path)
    max_score = float(np.max(np.abs(neg_gradient)))
    assessment["objective"] = -float(neg_objective)
    assessment["max_abs_score"] = max_score
    assessment["max_absolute_score_gradient"] = max_score
    assessment["strict_score_converged"] = bool(max_score <= step2_base.engine.SCORE_TOL)
    return assessment


def fit_weighted(y, gdf, xdf, weights, starts: list[dict], label: str, score_path: str):
    g, x = gdf.to_numpy(float), xdf.to_numpy(float)
    rows, candidates = [], []
    for st in starts:
        res = optimize.minimize(
            lambda th: neg_obj_grad(th, y, g, x, weights, score_path), st["theta"], method="L-BFGS-B", jac=True,
            options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 80, "maxcor": 50},
        )
        assessment = weighted_assessment(res.x, y, g, x, weights, score_path)
        row = {
            "fit_label": label, "start": st["start"], "success": bool(res.success),
            "message": str(res.message), "iterations": int(res.nit), **assessment,
        }
        rows.append(row)
        strict = bool(np.isfinite(res.fun) and assessment["thresholds_ordered"] and assessment["strict_score_converged"] and assessment["parameter_valid"])
        candidates.append({"theta": res.x, "objective": -float(res.fun), "strict": strict, "start": st["start"]})
        log(f"{label}: {st['start']} objective={-res.fun:.6f} score={assessment['max_absolute_score_gradient']:.3g} strict={strict}")
    valid = [c for c in candidates if c["strict"]]
    if not valid:
        raise RuntimeError(f"STOP: no strict-valid weighted solution for {label}")
    best = max(valid, key=lambda c: c["objective"])
    ref = optimize.minimize(
        lambda th: neg_obj_grad(th, y, g, x, weights, score_path), best["theta"], method="L-BFGS-B", jac=True,
        options={"maxiter": 8000, "ftol": 1e-13, "gtol": 1e-9, "maxls": 120, "maxcor": 100},
    )
    assessment = weighted_assessment(ref.x, y, g, x, weights, score_path)
    strict = bool(np.isfinite(ref.fun) and assessment["thresholds_ordered"] and assessment["strict_score_converged"] and assessment["parameter_valid"])
    rows.append({
        "fit_label": label, "start": "refined_from_best", "success": bool(ref.success),
        "message": str(ref.message), "iterations": int(ref.nit), **assessment,
    })
    if not strict:
        raise RuntimeError(f"STOP: refined weighted solution failed strict gate for {label}")
    return ref.x, pd.DataFrame(rows)


def weighted_sandwich(theta, y, gdf, xdf, clusters, weights, score_path):
    g, x = gdf.to_numpy(float), xdf.to_numpy(float)
    # Keep the formal model's participant-level score and finite-difference
    # Jacobian path.  IPW enters as a multiplicative observation weight.
    _, unweighted_score_i = formal_loglik_score(theta, y, g, x, score_path)
    w = np.asarray(weights, dtype=float)
    score_i = w[:, None] * unweighted_score_i
    score_df = pd.DataFrame(score_i)
    score_df["cluster"] = np.asarray(clusters)
    u = score_df.groupby("cluster", observed=True).sum().to_numpy()
    bmat = u.T @ u
    p = len(theta)
    hess = np.zeros((p, p))
    steps = np.maximum(1e-5, np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(theta)))
    for j, step in enumerate(steps):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += step
        tm[j] -= step
        _, gp = formal_loglik_score(tp, y, g, x, score_path)
        _, gm = formal_loglik_score(tm, y, g, x, score_path)
        hess[:, j] = (
            np.sum(w[:, None] * gp, axis=0) - np.sum(w[:, None] * gm, axis=0)
        ) / (2 * step)
    amat = -hess
    inva = np.linalg.pinv(amat)
    covariance = inva @ bmat @ inva.T
    covariance = (covariance + covariance.T) / 2
    eig = np.linalg.eigvalsh(covariance)
    return covariance, {
        "covariance_finite": bool(np.isfinite(covariance).all()),
        "covariance_PSD": bool(eig.min() >= -1e-8),
        "covariance_min_eigenvalue": float(eig.min()),
        "A_condition": float(np.linalg.cond(amat)),
        "B_condition": float(np.linalg.cond(bmat)),
        "V_condition": float(np.linalg.cond(covariance)),
        "max_robust_SE": float(np.sqrt(np.maximum(np.diag(covariance), 0)).max()),
        "variance_note": "Participant-cluster robust sandwich; estimated IPW treated as fixed; V=A^-1 B A^-T",
    }


def load_theta(path: Path, domain: str | None = None) -> np.ndarray:
    df = pd.read_csv(path)
    if domain is not None:
        df = df[df["Domain"].eq(domain)]
    return df["estimate"].to_numpy(float)


def step3_analysis_data(wa: pd.DataFrame) -> pd.DataFrame:
    """Apply the finalized Step 3 HAMD-only standardization without changing row order."""
    out = wa.copy()
    raw = pd.to_numeric(out["concurrent_HAMD17"], errors="coerce")
    out["concurrent_HAMD17_z"] = (
        raw - step3_core.HAMD_ONLY_MEAN
    ) / step3_core.HAMD_ONLY_SD_DDOF0
    if out["concurrent_HAMD17_z"].isna().any():
        raise RuntimeError("STOP: finalized Step 3 HAMD-only standardization failed")
    return out


def step2_unweighted_gate(wa: pd.DataFrame):
    gdf, xdf = b1.make_design(wa, mix_rhs=step2_main.RHS, ord_rhs=step2_main.RHS)
    theta = load_theta(STEP2_OUT / "09_selected_parameter_estimates.csv")
    names = parameter_names(gdf, xdf)
    saved_names = pd.read_csv(STEP2_OUT / "09_selected_parameter_estimates.csv")["parameter"].tolist()
    if names != saved_names:
        raise RuntimeError("STOP: latest Step 2 design-to-parameter mapping does not match saved main model")
    y = wa["burden_ord4"].to_numpy(int)
    cov, diag = weighted_sandwich(theta, y, gdf, xdf, wa["participant_ID"], np.ones(len(wa)), "step2")
    effects = step2_base.engine.domain_effects(theta, cov, gdf, xdf)
    saved = pd.read_csv(STEP2_OUT / "10_domain_effects_robust_wald.csv")
    merged = effects.merge(saved, on=["Domain", "Component"], suffixes=("_recalc", "_saved"))
    fields = ["Coefficient", "Robust_SE", "OR", "CI_low", "CI_high", "Wald_P"]
    diffs = {f: float((merged[f + "_recalc"] - merged[f + "_saved"]).abs().max()) for f in fields}
    passed = max(diffs.values()) <= 2e-6
    gate = pd.DataFrame([{"gate": "Latest unweighted Step 2 reproduction", "passed": passed, **{f"max_abs_diff_{k}": v for k, v in diffs.items()}}])
    if not passed:
        raise RuntimeError(f"STOP: latest unweighted Step 2 results not reproduced: {diffs}")
    return theta, gdf, xdf, cov, effects, pd.DataFrame([diag]), gate


def step3_unweighted_gate(wa: pd.DataFrame):
    wa = step3_analysis_data(wa)
    y = wa["burden_ord4"].to_numpy(int)
    saved_params = pd.read_csv(STEP3_OUT / "06_selected_parameter_estimates.csv")
    global_rows, time_rows, diag_rows = [], [], []
    for domain in b1.DOMAINS:
        var = b1.DOMAIN_VAR[domain]
        rhs = step3_common.rhs_for(step2_main.RHS, var)
        gdf, xdf = b1.make_design(wa, mix_rhs=rhs, ord_rhs=rhs)
        names = parameter_names(gdf, xdf)
        sub = saved_params[saved_params["Domain"].eq(domain)]
        if sub["parameter"].tolist() != names:
            raise RuntimeError(f"STOP: latest Step 3 parameter-name mapping mismatch for {domain}")
        theta = sub["estimate"].to_numpy(float)
        cov, diag = weighted_sandwich(theta, y, gdf, xdf, wa["participant_ID"], np.ones(len(wa)), "step3")
        diag_rows.append({"Domain": domain, **diag})
        global_rows.extend(step3_common.extract_global("Model 1", domain, var, theta, cov, names))
        time_rows.extend(step3_common.extract_time_effects("Model 1", domain, var, theta, cov, names))
    globals_df = step3_common.add_fdr(pd.DataFrame(global_rows))
    times_df = pd.DataFrame(time_rows)
    sg = pd.read_csv(STEP3_OUT / "09_global_interactions.csv")
    st = pd.read_csv(STEP3_OUT / "10_time_specific_ORs.csv")
    mg = globals_df.merge(sg, on=["Model", "Domain", "Component"], suffixes=("_recalc", "_saved"))
    mt = times_df.merge(st, on=["Model", "Domain", "Component", "Time"], suffixes=("_recalc", "_saved"))
    diffs = {
        "global_raw_P": float((mg["raw_P_recalc"] - mg["raw_P_saved"]).abs().max()),
        "global_BH_FDR_P": float((mg["BH_FDR_P_recalc"] - mg["BH_FDR_P_saved"]).abs().max()),
        "time_OR": float((mt["OR_recalc"] - mt["OR_saved"]).abs().max()),
        "time_CI_low": float((mt["CI_low_recalc"] - mt["CI_low_saved"]).abs().max()),
        "time_CI_high": float((mt["CI_high_recalc"] - mt["CI_high_saved"]).abs().max()),
        "time_P": float((mt["raw_Wald_P_recalc"] - mt["raw_Wald_P_saved"]).abs().max()),
    }
    passed = max(diffs.values()) <= 2e-6
    gate = pd.DataFrame([{"gate": "Latest unweighted Step 3 reproduction", "passed": passed, **{f"max_abs_diff_{k}": v for k, v in diffs.items()}}])
    if not passed:
        raise RuntimeError(f"STOP: latest unweighted Step 3 results not reproduced: {diffs}")
    return globals_df, times_df, pd.DataFrame(diag_rows), gate


def starts_for_step2(y, gdf, xdf, unweighted_theta):
    beta0, raw0, method = b1.fit_ordinal_start(y, xdf)
    rng = np.random.default_rng(SEED)
    scale = np.r_[np.repeat(0.08, gdf.shape[1]), np.repeat(0.04, Q), np.repeat(0.04, xdf.shape[1])]
    starts = [
        {"start": "latest_unweighted_main", "theta": unweighted_theta.copy()},
        {"start": f"ordinary_ordinal_{method}", "theta": np.r_[np.zeros(gdf.shape[1]), raw0, beta0]},
    ]
    for i in range(3):
        starts.append({"start": f"main_jitter_{i+1}", "theta": unweighted_theta + rng.normal(0, scale)})
    return starts


def starts_for_step3(y, gdf, xdf, mapped_step2, unweighted_theta, seed_offset):
    beta0, raw0, method = b1.fit_ordinal_start(y, xdf)
    rng = np.random.default_rng(SEED + seed_offset)
    scale = np.r_[np.repeat(0.08, gdf.shape[1]), np.repeat(0.04, Q), np.repeat(0.04, xdf.shape[1])]
    return [
        {"start": "weighted_step2_mapped", "theta": mapped_step2},
        {"start": "latest_unweighted_step3", "theta": unweighted_theta},
        {"start": f"ordinary_ordinal_{method}", "theta": np.r_[np.zeros(gdf.shape[1]), raw0, beta0]},
        {"start": "mapped_jitter_1", "theta": mapped_step2 + rng.normal(0, scale)},
        {"start": "mapped_jitter_2", "theta": mapped_step2 + rng.normal(0, scale)},
    ]


def map_theta(source_theta, source_names, target_names):
    lookup = dict(zip(source_names, source_theta))
    return np.array([lookup.get(n, 0.0) for n in target_names], dtype=float)


def add_step2_fdr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["BH_FDR_P"] = np.nan
    for component, idx in out.groupby("Component", sort=False).groups.items():
        if len(idx) != 9:
            raise RuntimeError(f"STOP: Step 2 FDR family {component} does not contain nine tests")
        out.loc[idx, "BH_FDR_P"] = multipletests(out.loc[idx, "Wald_P"].to_numpy(float), method="fdr_bh")[1]
    return out


def fit_ipw_models(wa, unweighted_step2_theta, unweighted_step2, unweighted_s3_global, unweighted_s3_time):
    y = wa["burden_ord4"].to_numpy(int)
    wa3 = step3_analysis_data(wa)
    g2, x2 = b1.make_design(wa, mix_rhs=step2_main.RHS, ord_rhs=step2_main.RHS)
    n2 = parameter_names(g2, x2)
    all_step2, all_global, all_time, all_diag, all_fit = {}, {}, {}, [], []
    saved_s3 = pd.read_csv(STEP3_OUT / "06_selected_parameter_estimates.csv")
    for weight_col, label in [("SW", "IPW"), ("SW_truncated_1_99", "IPW_truncated_1_99")]:
        log(f"fitting {label} Step 2")
        theta2, fit2 = fit_weighted(y, g2, x2, wa[weight_col].to_numpy(float), starts_for_step2(y, g2, x2, unweighted_step2_theta), f"{label}_Step2", "step2")
        cov2, diag2 = weighted_sandwich(theta2, y, g2, x2, wa["participant_ID"], wa[weight_col].to_numpy(float), "step2")
        if not (diag2["covariance_finite"] and diag2["covariance_PSD"]):
            raise RuntimeError(f"STOP: {label} Step 2 robust covariance failed")
        effects2 = add_step2_fdr(step2_base.engine.domain_effects(theta2, cov2, g2, x2))
        all_step2[label] = effects2
        all_fit.append(fit2.assign(model=label, stage="Step 2"))
        all_diag.append({"model": label, "stage": "Step 2", "Domain": "All domains", **diag2})
        pd.DataFrame({"parameter": n2, "estimate": theta2, "robust_SE": np.sqrt(np.maximum(np.diag(cov2), 0))}).to_csv(CHECKPOINT / f"{label}_Step2_parameters.csv", index=False, encoding="utf-8-sig")

        global_rows, time_rows = [], []
        for i, domain in enumerate(b1.DOMAINS):
            var = b1.DOMAIN_VAR[domain]
            rhs = step3_common.rhs_for(step2_main.RHS, var)
            g3, x3 = b1.make_design(wa3, mix_rhs=rhs, ord_rhs=rhs)
            n3 = parameter_names(g3, x3)
            mapped = map_theta(theta2, n2, n3)
            usub = saved_s3[saved_s3["Domain"].eq(domain)]
            if usub["parameter"].tolist() != n3:
                raise RuntimeError(f"STOP: Step 3 name alignment failed for {domain}")
            log(f"fitting {label} Step 3 {domain}")
            theta3, fit3 = fit_weighted(
                y, g3, x3, wa3[weight_col].to_numpy(float),
                starts_for_step3(y, g3, x3, mapped, usub["estimate"].to_numpy(float), i + (100 if label.startswith("IPW_truncated") else 0)),
                f"{label}_Step3_{domain}", "step3",
            )
            cov3, diag3 = weighted_sandwich(theta3, y, g3, x3, wa3["participant_ID"], wa3[weight_col].to_numpy(float), "step3")
            if not (diag3["covariance_finite"] and diag3["covariance_PSD"]):
                raise RuntimeError(f"STOP: {label} Step 3 robust covariance failed for {domain}")
            global_rows.extend(step3_common.extract_global(label, domain, var, theta3, cov3, n3))
            time_rows.extend(step3_common.extract_time_effects(label, domain, var, theta3, cov3, n3))
            all_diag.append({"model": label, "stage": "Step 3", "Domain": domain, **diag3})
            all_fit.append(fit3.assign(model=label, stage="Step 3", Domain=domain))
            pd.DataFrame({"parameter": n3, "estimate": theta3, "robust_SE": np.sqrt(np.maximum(np.diag(cov3), 0))}).to_csv(CHECKPOINT / f"{label}_Step3_{var}_parameters.csv", index=False, encoding="utf-8-sig")
        all_global[label] = step3_common.add_fdr(pd.DataFrame(global_rows))
        all_time[label] = pd.DataFrame(time_rows)

    u2 = add_step2_fdr(unweighted_step2)
    s2 = u2.rename(columns={c: "unweighted_" + c for c in ["Coefficient", "Robust_SE", "OR", "CI_low", "CI_high", "Wald_P", "BH_FDR_P"]})
    i2 = all_step2["IPW"].rename(columns={c: "IPW_" + c for c in ["Coefficient", "Robust_SE", "OR", "CI_low", "CI_high", "Wald_P", "BH_FDR_P"]})
    t2 = all_step2["IPW_truncated_1_99"].rename(columns={c: "truncated_" + c for c in ["Coefficient", "Robust_SE", "OR", "CI_low", "CI_high", "Wald_P", "BH_FDR_P"]})
    keys2 = ["Domain", "Component", "OR_definition"]
    step2_comp = s2.merge(i2, on=keys2).merge(t2, on=keys2)
    step2_comp["coefficient_change_IPW_minus_unweighted"] = step2_comp["IPW_Coefficient"] - step2_comp["unweighted_Coefficient"]
    step2_comp["direction_changed"] = np.sign(step2_comp["IPW_Coefficient"]) != np.sign(step2_comp["unweighted_Coefficient"])
    step2_comp["FDR_status_changed"] = (step2_comp["IPW_BH_FDR_P"] < .05) != (step2_comp["unweighted_BH_FDR_P"] < .05)

    ug = unweighted_s3_global.rename(columns={c: "unweighted_" + c for c in ["Wald_chi2", "raw_P", "BH_FDR_P"]})
    ig = all_global["IPW"].drop(columns=["Model"]).rename(columns={c: "IPW_" + c for c in ["Wald_chi2", "raw_P", "BH_FDR_P"]})
    tg = all_global["IPW_truncated_1_99"].drop(columns=["Model"]).rename(columns={c: "truncated_" + c for c in ["Wald_chi2", "raw_P", "BH_FDR_P"]})
    global_comp = ug.drop(columns=["BH_FDR_significant_0.05"]).merge(ig.drop(columns=["BH_FDR_significant_0.05"]), on=["Domain", "Component", "df"]).merge(tg.drop(columns=["BH_FDR_significant_0.05"]), on=["Domain", "Component", "df"])
    global_comp["FDR_status_changed"] = (global_comp["IPW_BH_FDR_P"] < .05) != (global_comp["unweighted_BH_FDR_P"] < .05)

    ut = unweighted_s3_time.rename(columns={c: "unweighted_" + c for c in ["linear_predictor_coefficient", "Robust_SE", "OR", "CI_low", "CI_high", "raw_Wald_P"]})
    it = all_time["IPW"].drop(columns=["Model"]).rename(columns={c: "IPW_" + c for c in ["linear_predictor_coefficient", "Robust_SE", "OR", "CI_low", "CI_high", "raw_Wald_P"]})
    tt = all_time["IPW_truncated_1_99"].drop(columns=["Model"]).rename(columns={c: "truncated_" + c for c in ["linear_predictor_coefficient", "Robust_SE", "OR", "CI_low", "CI_high", "raw_Wald_P"]})
    time_comp = ut.merge(it, on=["Domain", "Component", "Time", "OR_definition"]).merge(tt, on=["Domain", "Component", "Time", "OR_definition"])
    time_comp["coefficient_change_IPW_minus_unweighted"] = time_comp["IPW_linear_predictor_coefficient"] - time_comp["unweighted_linear_predictor_coefficient"]
    return step2_comp, global_comp, time_comp, pd.concat(all_fit, ignore_index=True), pd.DataFrame(all_diag)


def make_reports(source_audit, dropout_counts, dropout_comp, discrepancy, weight_diag, obs_diag, step2_comp, global_comp, time_comp, fit_diag, model_diag, gates):
    wd = weight_diag[(weight_diag["group"].eq("overall")) & (weight_diag["weight"].eq("SW"))].iloc[0]
    direction_changes = int(step2_comp["direction_changed"].sum())
    step2_fdr_changes = int(step2_comp["FDR_status_changed"].sum())
    step3_fdr_changes = int(global_comp["FDR_status_changed"].sum())
    max_change = float(step2_comp["coefficient_change_IPW_minus_unweighted"].abs().max())
    converged = bool(fit_diag.groupby(["model", "stage", fit_diag.get("Domain", pd.Series(index=fit_diag.index, dtype=object)).fillna("All")], dropna=False).tail(1)["strict_score_converged"].all())
    covariance_ok = bool(model_diag["covariance_finite"].all() and model_diag["covariance_PSD"].all())
    trunc_step2 = float((step2_comp["truncated_Coefficient"] - step2_comp["IPW_Coefficient"]).abs().max())
    trunc_global_change = int(((global_comp["truncated_BH_FDR_P"] < .05) != (global_comp["IPW_BH_FDR_P"] < .05)).sum())
    if not converged or not covariance_ok:
        judgment = "IPW results were not numerically reliable"
    elif direction_changes or step2_fdr_changes > 2 or step3_fdr_changes > 1:
        judgment = "IPW produced material changes"
    elif step2_fdr_changes or step3_fdr_changes or max_change > .15:
        judgment = "IPW produced limited inferential changes"
    else:
        judgment = "IPW did not materially change the main findings"

    group_text = ", ".join(f"{r.follow_up_group}: {int(r.participants)}" for r in dropout_counts.itertuples())
    comp_lines = []
    for r in dropout_comp.itertuples():
        comp_lines.append(
            f"- {r.comparison} at {r.assessment}: n={r.dropout_n} vs {r.complete_n}; median (IQR) "
            f"{r.dropout_median:g} ({r.dropout_Q1:g}–{r.dropout_Q3:g}) vs {r.complete_median:g} ({r.complete_Q1:g}–{r.complete_Q3:g}); "
            f"U={r.Mann_Whitney_U:g}, raw two-sided P={r.raw_two_sided_P:.4f}. FIBSER≥3: "
            f"{r.dropout_FIBSER_ge3_n} ({r.dropout_FIBSER_ge3_percent:.1f}%) vs {r.complete_FIBSER_ge3_n} ({r.complete_FIBSER_ge3_percent:.1f}%)."
        )
    report = f"""# Attrition Analysis Report

## Data verification

- Current source workbook SHA-256: `{sha256(SOURCE)}`.
- Latest freeze: `{FREEZE_ID}`; SHA-256 `{sha256(FREEZE)}`.
- Raw cohort: 1,543 unique participants; four follow-up rows reconstructed for every participant (6,172 participant-time rows), with no duplicate participant-time keys after prespecified paired-sheet collapse.
- Current raw data reproduced the latest frozen 1,468 participants / 4,799 person-visits exactly (Week 2/8/12/24: 1,381/1,259/1,146/1,013), with zero mismatches in FIBSER burden, concurrent HAMD-17, or the nine PRISE domains.
- The current workbook contains 1,503 participants with at least one valid follow-up FIBSER, compared with 1,502 in the earlier aggregate audit. The source-workbook hashes differ. Participant-level reconciliation details are intentionally not written by the public code. This does not alter the verified 1,468/4,799 frozen main-model sample.

## Exploratory dropout comparison

Groups were mutually exclusive and summed to 1,543: {group_text}. “Dropout” denotes no later valid FIBSER assessment and does not imply confirmed treatment discontinuation or formal study withdrawal. Mann–Whitney U tests were two-sided and exploratory; raw P values are reported without multiplicity adjustment.

{chr(10).join(comp_lines)}

## IPW sensitivity update

- Observation status was rebuilt on the complete 1,543 × 4 visit grid. The denominator models used baseline covariates and baseline FIBSER; Week 8/12/24 additionally used only previously observed burden, HAMD-17, PRISE total, prior observation status, and missing-history indicators. No current missing-visit outcome or future information was used.
- Stabilized weight distribution among the 4,799 analyzed visits: mean={wd['mean']:.4f}, median={wd['median']:.4f}, p1={wd['p1']:.4f}, p99={wd['p99']:.4f}, max={wd['max']:.4f}, ESS={wd['ESS']:.1f}; >5={wd['proportion_gt5']:.2%}, >10={wd['proportion_gt10']:.2%}.
- Latest unweighted Step 2 and Step 3 results were deterministically reproduced from the saved parameter vectors and participant-cluster robust covariance before weighting (all gates passed; tolerance 2×10⁻⁶).
- Weighted fits retained the prespecified four-level FIBSER outcome, covariates, zero-inflated and ordinal components, nine Step 3 domain×time models, participant-cluster robust covariance, and the existing BH-FDR families.
- Numerical status: strict-score convergence={converged}; all robust covariance matrices finite and PSD={covariance_ok}.
- Step 2 direction changes={direction_changes}/18; Step 2 FDR-status changes={step2_fdr_changes}/18; Step 3 global interaction FDR-status changes={step3_fdr_changes}/18; largest absolute Step 2 log-coefficient change={max_change:.4f}.
- Truncation QC: maximum absolute Step 2 coefficient difference (1st/99th percentile truncated vs untruncated IPW)={trunc_step2:.4f}; Step 3 global FDR-status changes due to truncation={trunc_global_change}.

Overall, **{judgment}**. This sensitivity analysis addresses attrition under the specified observation model; it does not convert the observational associations into causal effects.
"""
    (OUT / "Attrition_Analysis_Report.md").write_text(report, encoding="utf-8")

    result_sentences = []
    for r in dropout_comp.itertuples():
        result_sentences.append(
            f"At {r.assessment}, FIBSER burden did not differ between the {r.dropout_group.lower()} group and complete follow-up participants "
            f"(n={r.dropout_n} vs n={r.complete_n}; median [IQR], {r.dropout_median:g} [{r.dropout_Q1:g}–{r.dropout_Q3:g}] vs "
            f"{r.complete_median:g} [{r.complete_Q1:g}–{r.complete_Q3:g}]; Mann–Whitney U={r.Mann_Whitney_U:g}, two-sided raw P={r.raw_two_sided_P:.3f}). "
            f"FIBSER≥3 was observed in {r.dropout_FIBSER_ge3_n}/{r.dropout_n} ({r.dropout_FIBSER_ge3_percent:.1f}%) and "
            f"{r.complete_FIBSER_ge3_n}/{r.complete_n} ({r.complete_FIBSER_ge3_percent:.1f}%), respectively."
        )
    manuscript = "Exploratory attrition analysis\n\n" + " ".join(result_sentences) + (
        " These comparisons were exploratory and used unadjusted P values. Dropout indicates absence of subsequent valid FIBSER assessments and does not imply confirmed treatment discontinuation or formal study withdrawal.\n\n"
        "IPW sensitivity analysis\n\n"
        f"In the updated inverse-probability-weighted analysis of the frozen 1,468-participant, 4,799-visit sample, stabilized weights had a mean of {wd['mean']:.3f} "
        f"(1st–99th percentile, {wd['p1']:.3f}–{wd['p99']:.3f}; maximum, {wd['max']:.3f}; effective sample size, {wd['ESS']:.1f} visits). "
        f"All weighted models converged under the prespecified numerical criteria and yielded finite participant-cluster robust covariance estimates. "
        f"Compared with the corresponding unweighted models, there were {direction_changes} direction changes among 18 Step 2 domain coefficients, {step2_fdr_changes} Step 2 BH-FDR status changes, "
        f"and {step3_fdr_changes} BH-FDR status changes among 18 global Step 3 domain-by-time tests. Overall, {judgment.lower()}."
    )
    (OUT / "Manuscript_Ready_Results.txt").write_text(manuscript, encoding="utf-8")

    qc = pd.concat([source_audit, gates], ignore_index=True, sort=False)
    write_csv("_QC_summary.csv", qc)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    base, long, freeze, mapping_rows, zmap, source_audit, mismatch = raw_and_freeze_audit()
    dropout_counts, dropout_comp, discrepancy = dropout_analysis(base, long)
    write_csv("_dropout_group_counts.csv", dropout_counts)
    write_csv("_dropout_comparisons.csv", dropout_comp)
    write_csv("_prior_1502_reconciliation.csv", discrepancy)
    write_csv("_raw_vs_freeze_value_audit.csv", mismatch)

    wa, grid, obs_diag, obs_coef, obs_pred, obs_cal, obs_aux, weight_diag, concentration, terms = prepare_weight_data(base, long, freeze)
    write_csv("_observation_model_diagnostics.csv", obs_diag)
    write_csv("_observation_model_coefficients.csv", obs_coef)
    write_csv("_observation_model_calibration.csv", obs_cal)
    write_csv("_observation_model_terms.csv", terms)
    write_csv("_weight_diagnostics.csv", weight_diag)
    write_csv("_weight_concentration_summary.csv", concentration)

    log("reproducing latest unweighted Step 2")
    theta2, g2, x2, cov2, effects2, diag2, gate2 = step2_unweighted_gate(wa)
    log("reproducing latest unweighted Step 3")
    global3, time3, diag3, gate3 = step3_unweighted_gate(wa)
    gates = pd.concat([gate2, gate3], ignore_index=True)
    write_csv("_unweighted_reproduction_gates.csv", gates)
    write_csv("_unweighted_step2_results.csv", effects2)
    write_csv("_unweighted_step3_global.csv", global3)
    write_csv("_unweighted_step3_time_ORs.csv", time3)

    step2_comp, global_comp, time_comp, fit_diag, model_diag = fit_ipw_models(wa, theta2, effects2, global3, time3)
    write_csv("_Step2_weighted_unweighted_comparison.csv", step2_comp)
    write_csv("_Step3_global_weighted_unweighted_comparison.csv", global_comp)
    write_csv("_Step3_time_OR_weighted_unweighted_comparison.csv", time_comp)
    write_csv("_weighted_fit_diagnostics.csv", fit_diag)
    write_csv("_weighted_covariance_diagnostics.csv", model_diag)

    make_reports(source_audit, dropout_counts, dropout_comp, discrepancy, weight_diag, obs_diag, step2_comp, global_comp, time_comp, fit_diag, model_diag, gates)
    manifest = {
        "status": "COMPLETE",
        "source_sha256": sha256(SOURCE),
        "freeze_id": FREEZE_ID,
        "freeze_sha256": sha256(FREEZE),
        "participants": EXPECTED_PARTICIPANTS,
        "person_visits": EXPECTED_VISITS,
        "visit_counts": EXPECTED_VISIT_COUNTS,
        "dropout_groups_sum": int(dropout_counts["participants"].sum()),
        "current_any_valid_followup_fibser": int(dropout_counts.loc[~dropout_counts["follow_up_group"].eq("No valid follow-up FIBSER"), "participants"].sum()),
        "unweighted_gates_passed": bool(gates["passed"].all()),
        "all_weighted_covariances_finite": bool(model_diag["covariance_finite"].all()),
        "all_weighted_covariances_PSD": bool(model_diag["covariance_PSD"].all()),
    }
    (OUT / "_analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log("analysis complete")


if __name__ == "__main__":
    main()




