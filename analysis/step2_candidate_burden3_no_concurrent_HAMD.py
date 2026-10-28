from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit
from statsmodels.stats.multitest import multipletests

import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2B3_fixed_sample_covariate_attenuation as b3


ROOT = b1.ROOT
SOURCE = b1.SOURCE
OUT = ROOT / "outputs" / "step2_candidate_burden3_no_concurrent_HAMD"
LABEL = "Step 2 primary candidate - burden 0/1/>=2, without concurrent HAMD"
RHS = b3.MODEL_RHS["M3"].replace(" + concurrent_HAMD17_z", "")
DOMAIN_VARS = [b1.DOMAIN_VAR[d] for d in b1.DOMAINS]
EPS = np.finfo(float).tiny


def log(msg: str) -> None:
    print(f"{LABEL}: {msg}", flush=True)


def write_book(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    b1.write_book(path, sheets)


def thresholds_from_raw(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = len(raw)
    alpha = np.empty(q)
    alpha[0] = raw[0]
    for j in range(1, q):
        alpha[j] = alpha[j - 1] + np.exp(raw[j])
    jac = np.zeros((q, q))
    jac[:, 0] = 1.0
    for r in range(q):
        for c in range(1, r + 1):
            jac[r, c] = np.exp(raw[c])
    return alpha, jac


def unpack(theta: np.ndarray, pg: int, px: int, q: int):
    gamma = theta[:pg]
    raw = theta[pg:pg + q]
    beta = theta[pg + q:pg + q + px]
    alpha, jac = thresholds_from_raw(raw)
    return gamma, raw, beta, alpha, jac


def ordinal_probs_and_deriv(theta: np.ndarray, G, X, q: int):
    pg, px = G.shape[1], X.shape[1]
    _, _, beta, alpha, jac_alpha = unpack(theta, pg, px, q)
    Xv = np.asarray(X, dtype=float)
    eta = alpha[None, :] - Xv @ beta[:, None]
    cdf = expit(eta)
    d_cdf = cdf * (1.0 - cdf)
    kcat = q + 1
    p = np.empty((len(Xv), kcat))
    p[:, 0] = cdf[:, 0]
    for k in range(1, q):
        p[:, k] = cdf[:, k] - cdf[:, k - 1]
    p[:, q] = 1.0 - cdf[:, q - 1]
    p = np.maximum(p, EPS)

    dp_dalpha = np.zeros((len(Xv), kcat, q))
    dp_dalpha[:, 0, 0] = d_cdf[:, 0]
    for k in range(1, q):
        dp_dalpha[:, k, k - 1] = -d_cdf[:, k - 1]
        dp_dalpha[:, k, k] = d_cdf[:, k]
    dp_dalpha[:, q, q - 1] = -d_cdf[:, q - 1]
    dp_draw = np.einsum("nka,ar->nkr", dp_dalpha, jac_alpha)
    dp_dbeta = -np.einsum("nka,nx->nkx", dp_dalpha, Xv)
    return p, dp_draw, dp_dbeta


def obs_loglik_score(theta: np.ndarray, y, G, X, q: int):
    y = np.asarray(y, dtype=int)
    Gv = np.asarray(G, dtype=float)
    Xv = np.asarray(X, dtype=float)
    n, pg, px = len(y), Gv.shape[1], Xv.shape[1]
    gamma = theta[:pg]
    pi = expit(Gv @ gamma)
    dpi = pi * (1.0 - pi)
    p, dp_draw, dp_dbeta = ordinal_probs_and_deriv(theta, Gv, Xv, q)
    py = p[np.arange(n), y]
    zero = y == 0
    m = pi * py
    m[zero] = (1.0 - pi[zero]) + pi[zero] * p[zero, 0]
    m = np.maximum(m, EPS)
    loglik_i = np.log(m)
    grad = np.zeros((n, pg + q + px))
    gamma_factor = dpi * py
    gamma_factor[zero] = dpi[zero] * (p[zero, 0] - 1.0)
    grad[:, :pg] = (gamma_factor / m)[:, None] * Gv
    dord_raw = dp_draw[np.arange(n), y, :].copy()
    dord_beta = dp_dbeta[np.arange(n), y, :].copy()
    dord_raw[zero, :] = pi[zero, None] * dp_draw[zero, 0, :]
    dord_beta[zero, :] = pi[zero, None] * dp_dbeta[zero, 0, :]
    nonzero = ~zero
    dord_raw[nonzero, :] = pi[nonzero, None] * dord_raw[nonzero, :]
    dord_beta[nonzero, :] = pi[nonzero, None] * dord_beta[nonzero, :]
    grad[:, pg:pg + q] = dord_raw / m[:, None]
    grad[:, pg + q:] = dord_beta / m[:, None]
    return loglik_i, grad


def neg_obj_grad(theta: np.ndarray, y, G, X, q: int):
    ll, grad = obs_loglik_score(theta, y, G, X, q)
    return -float(ll.sum()), -grad.sum(axis=0)


def fit_ordinal_start(y, X, q: int):
    try:
        parts_y = []
        parts_x = []
        parts_cut = []
        for j, k in enumerate(range(q)):
            parts_y.append((y <= k).astype(float))
            parts_x.append(np.asarray(X, dtype=float))
            parts_cut.append(np.full(len(y), j))
        yy = np.concatenate(parts_y)
        xx = np.vstack(parts_x)
        cut = np.concatenate(parts_cut)
        cut_mat = np.column_stack([(cut == j).astype(float) for j in range(q)])
        z = np.column_stack([cut_mat, xx])
        import statsmodels.api as sm
        res = sm.GLM(yy, z, family=sm.families.Binomial()).fit(maxiter=200)
        alpha = np.asarray(res.params[:q])
        beta = -np.asarray(res.params[q:])
        alpha = np.maximum.accumulate(alpha + np.arange(q) * 1e-4)
        raw = np.empty(q)
        raw[0] = alpha[0]
        for j in range(1, q):
            raw[j] = np.log(max(alpha[j] - alpha[j - 1], 1e-3))
        return beta, raw, "stacked_PO_GLM"
    except Exception:
        prop = np.array([(y <= k).mean() for k in range(q)])
        prop = np.clip(prop, 1e-4, 1 - 1e-4)
        alpha = np.log(prop / (1 - prop))
        alpha = np.maximum.accumulate(alpha + np.arange(q) * 1e-3)
        raw = np.empty(q)
        raw[0] = alpha[0]
        for j in range(1, q):
            raw[j] = np.log(max(alpha[j] - alpha[j - 1], 1e-3))
        return np.zeros(X.shape[1]), raw, "empirical_cuts"


def make_starts(y, G, X, q: int, seed: int):
    beta0, raw0, method = fit_ordinal_start(y, X, q)
    starts = []
    for extra in [0.01, 0.03, 0.05, 0.10, 0.15, 0.30, 0.50]:
        gamma = np.zeros(G.shape[1])
        gamma[0] = math.log((1.0 - extra) / extra)
        starts.append({"start": f"ordinary_start_extra_zero_{extra:.2f}", "theta": np.r_[gamma, raw0, beta0], "start_method": method})
    rng = np.random.default_rng(seed)
    scale = np.r_[np.repeat(0.12, G.shape[1]), np.repeat(0.06, q), np.repeat(0.06, X.shape[1])]
    for i in range(1, 6):
        base = starts[i % len(starts)]["theta"]
        starts.append({"start": f"random_perturbation_{i}", "theta": base + rng.normal(0, scale), "start_method": method})
    return starts


def theta_summary(start, theta, y, G, X, q: int, success, message, nit, start_method=None):
    ll, score = obs_loglik_score(theta, y, G, X, q)
    gamma, _, _, alpha, _ = unpack(theta, G.shape[1], X.shape[1], q)
    pi = expit(np.asarray(G) @ gamma)
    out = {
        "start": start,
        "success": bool(success),
        "message": str(message),
        "iterations": int(nit),
        "objective": float(ll.sum()),
        "objective_per_observation": float(ll.mean()),
        "max_absolute_score_gradient": float(np.max(np.abs(score.sum(axis=0)))),
        "thresholds_ordered": bool(np.all(np.diff(alpha) > 0)),
        "median_pi": float(np.median(pi)),
        "min_pi": float(pi.min()),
        "max_pi": float(pi.max()),
        "start_method": start_method,
    }
    for j, a in enumerate(alpha):
        out[f"threshold_alpha{j}"] = float(a)
    return out


def fit_model(label: str, y, G, X, q: int, seed: int):
    rows = []
    fits = []
    for st in make_starts(y, G, X, q, seed):
        res = optimize.minimize(
            lambda th: neg_obj_grad(th, y, G, X, q),
            st["theta"],
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 3000, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50},
        )
        row = theta_summary(st["start"], res.x, y, G, X, q, res.success, res.message, res.nit, st["start_method"])
        row["model"] = label
        rows.append(row)
        fits.append({"theta": res.x, "summary": row})
        log(f"{label} {st['start']} success={res.success} objective={row['objective']:.6f} score={row['max_absolute_score_gradient']:.3g}")
    best = max(fits, key=lambda z: z["summary"]["objective"])
    ref = optimize.minimize(
        lambda th: neg_obj_grad(th, y, G, X, q),
        best["theta"],
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 80, "maxcor": 50},
    )
    ref_row = theta_summary("refined_from_best_start", ref.x, y, G, X, q, ref.success, ref.message, ref.nit, "refined_BFGS")
    ref_row["model"] = label
    rows.append(ref_row)
    log(f"{label} refined success={ref.success} objective={ref_row['objective']:.6f} score={ref_row['max_absolute_score_gradient']:.3g}")
    return ref.x, pd.DataFrame(rows)


def finite_hessian_of_score(theta, y, G, X, q: int):
    p = len(theta)
    hess_score = np.zeros((p, p))
    base_step = np.maximum(1e-5, np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(theta)))
    for j in range(p):
        step = base_step[j]
        tp = theta.copy()
        tm = theta.copy()
        tp[j] += step
        tm[j] -= step
        _, gp = obs_loglik_score(tp, y, G, X, q)
        _, gm = obs_loglik_score(tm, y, G, X, q)
        hess_score[:, j] = (gp.sum(axis=0) - gm.sum(axis=0)) / (2.0 * step)
    return hess_score


def sandwich(theta, y, G, X, q: int, clusters):
    _, score_i = obs_loglik_score(theta, y, G, X, q)
    score_df = pd.DataFrame(score_i)
    score_df["cluster"] = np.asarray(clusters)
    U = score_df.groupby("cluster", observed=True).sum().drop(columns=[], errors="ignore").to_numpy()
    B = U.T @ U
    A = -finite_hessian_of_score(theta, y, G, X, q)
    invA = np.linalg.pinv(A)
    # Cluster-robust sandwich covariance: A^{-1} B A^{-T}.
    # Retain the existing Jacobian and participant-cluster score definitions.
    V = invA @ B @ invA.T
    Vs = (V + V.T) / 2.0
    eig = np.linalg.eigvalsh(Vs)
    return {
        "A": A,
        "B": B,
        "V": V,
        "se": np.sqrt(np.maximum(np.diag(V), 0.0)),
        "A_condition": float(np.linalg.cond(A)),
        "B_condition": float(np.linalg.cond(B)),
        "V_condition": float(np.linalg.cond(V)),
        "V_min_eigenvalue": float(eig.min()),
        "V_positive_semidefinite": bool(eig.min() >= -1e-8),
        "finite": bool(np.isfinite(V).all()),
    }


def names_for(G_cols, X_cols, q: int):
    return [f"mixing_gamma:{c}" for c in G_cols] + [f"threshold_raw:{j}" for j in range(q)] + [f"ordinal_beta:{c}" for c in X_cols]


def covariance_conditioning(V, names):
    wanted = []
    for v in DOMAIN_VARS:
        wanted.append(f"mixing_gamma:{v}")
        wanted.append(f"ordinal_beta:{v}")
    rows = []
    for block, block_names in [("full_covariance", names), ("18-domain scientific covariance block", wanted)]:
        idx = [names.index(n) for n in block_names if n in names]
        sub = ((V + V.T) / 2.0)[np.ix_(idx, idx)]
        eig = np.linalg.eigvalsh(sub)
        rows.append({
            "block": block,
            "parameters": len(idx),
            "finite": bool(np.isfinite(sub).all()),
            "PSD": bool(eig.min() >= -1e-8),
            "min_eigenvalue": float(eig.min()),
            "max_eigenvalue": float(eig.max()),
            "condition_number": float(np.linalg.cond(sub)),
        })
    idx = [names.index(n) for n in wanted if n in names]
    sub = ((V + V.T) / 2.0)[np.ix_(idx, idx)]
    d = np.sqrt(np.maximum(np.diag(sub), 0.0))
    corr = sub / np.outer(d, d)
    corr[~np.isfinite(corr)] = np.nan
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    rows.append({
        "block": "18-domain scientific covariance block correlations",
        "parameters": len(idx),
        "finite": bool(np.isfinite(corr).all()),
        "PSD": np.nan,
        "condition_number": np.nan,
        "max_abs_correlation": float(np.nanmax(np.abs(off))),
    })
    return pd.DataFrame(rows)


def domain_inference(model_label, theta, se, G_cols, X_cols, q: int):
    rows = []
    pg = len(G_cols)
    for domain in b1.DOMAINS:
        var = b1.DOMAIN_VAR[domain]
        gi = G_cols.index(var)
        gamma = float(theta[gi])
        gamma_se = float(se[gi])
        z = gamma / gamma_se if gamma_se > 0 else np.nan
        rows.append({
            "model": model_label,
            "Panel": "Panel A. Zero-inflation component",
            "Domain": domain,
            "component": "mixing",
            "parameter": var,
            "Coefficient": gamma,
            "Robust SE": gamma_se,
            "OR": math.exp(-gamma),
            "95% CI": f"{math.exp(-(gamma + 1.96 * gamma_se)):.3f}, {math.exp(-(gamma - 1.96 * gamma_se)):.3f}",
            "CI low": math.exp(-(gamma + 1.96 * gamma_se)),
            "CI high": math.exp(-(gamma - 1.96 * gamma_se)),
            "Raw P": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
        })
        xi = pg + q + X_cols.index(var)
        beta = float(theta[xi])
        beta_se = float(se[xi])
        z = beta / beta_se if beta_se > 0 else np.nan
        rows.append({
            "model": model_label,
            "Panel": "Panel B. Ordinary ordinal burden component",
            "Domain": domain,
            "component": "full_ordinal",
            "parameter": var,
            "Coefficient": beta,
            "Robust SE": beta_se,
            "OR": math.exp(beta),
            "95% CI": f"{math.exp(beta - 1.96 * beta_se):.3f}, {math.exp(beta + 1.96 * beta_se):.3f}",
            "CI low": math.exp(beta - 1.96 * beta_se),
            "CI high": math.exp(beta + 1.96 * beta_se),
            "Raw P": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
        })
    out = pd.DataFrame(rows)
    out["BH-FDR P"] = np.nan
    for component, idx in out.groupby("component").groups.items():
        idx = list(idx)
        p = out.loc[idx, "Raw P"].to_numpy(float)
        mask = np.isfinite(p)
        qvals = np.full(len(p), np.nan)
        if mask.any():
            qvals[mask] = multipletests(p[mask], method="fdr_bh")[1]
        out.loc[idx, "BH-FDR P"] = qvals
    return out


def build_new_sample():
    base, long, _, mapping, zmap = b1.build_dataset()
    valid = long["fibser_burden"].isin(range(7))
    long = long.copy()
    long["burden_ord3"] = np.nan
    long.loc[valid & long["fibser_burden"].eq(0), "burden_ord3"] = 0
    long.loc[valid & long["fibser_burden"].eq(1), "burden_ord3"] = 1
    long.loc[valid & long["fibser_burden"].ge(2), "burden_ord3"] = 2
    required = [
        "participant_ID", "time", "visit_order", "burden_ord3", "burden_ord4",
        "age_z", "sex", "education", "first_episode", "baseline_med_class",
        "baseline_HAMD17_z", "baseline_SDS_z",
    ] + DOMAIN_VARS
    analysis = long.dropna(subset=required).copy()
    analysis["burden_ord3"] = analysis["burden_ord3"].astype(int)
    analysis["burden_ord4"] = analysis["burden_ord4"].astype(int)
    return base, long, analysis, mapping, zmap


def outcome_counts(long, analysis):
    rows = []
    valid = long["burden_ord3"].isin([0, 1, 2])
    for week in b1.WEEKS:
        sub = long.loc[long["time"].astype(str).eq(week) & valid]
        for k, lab in [(0, "0"), (1, "1"), (2, ">=2")]:
            rows.append({
                "dataset": "raw_valid_FIBSER",
                "week": week,
                "category": lab,
                "n": int(sub["burden_ord3"].eq(k).sum()),
                "percent": float(sub["burden_ord3"].eq(k).mean() * 100) if len(sub) else np.nan,
                "denominator": int(len(sub)),
            })
        suba = analysis.loc[analysis["time"].astype(str).eq(week)]
        for k, lab in [(0, "0"), (1, "1"), (2, ">=2")]:
            rows.append({
                "dataset": "analysis_sample",
                "week": week,
                "category": lab,
                "n": int(suba["burden_ord3"].eq(k).sum()),
                "percent": float(suba["burden_ord3"].eq(k).mean() * 100) if len(suba) else np.nan,
                "denominator": int(len(suba)),
            })
    return pd.DataFrame(rows)


def fit_and_summarize(model_label, y_col, q, analysis, G, X, seed):
    y = analysis[y_col].to_numpy(int)
    theta, fit_qc = fit_model(model_label, y, G, X, q, seed)
    sw = sandwich(theta, y, G, X, q, analysis["participant_ID"])
    names = names_for(list(G.columns), list(X.columns), q)
    cov = covariance_conditioning(sw["V"], names)
    dom = domain_inference(model_label, theta, sw["se"], list(G.columns), list(X.columns), q)
    best = fit_qc.tail(1).copy()
    gamma, raw, beta, alpha, _ = unpack(theta, G.shape[1], X.shape[1], q)
    param = pd.DataFrame({
        "parameter": names,
        "estimate": theta,
        "robust_SE": sw["se"],
    })
    param["abs_coef_gt_5"] = param["estimate"].abs() > 5
    param["SE_gt_5"] = param["robust_SE"] > 5
    diag = pd.DataFrame([{
        "model": model_label,
        "outcome": y_col,
        "categories": q + 1,
        "ordinal_thresholds_estimated": q,
        "thresholds": ", ".join(f"{a:.6f}" for a in alpha),
        "thresholds_ordered": bool(np.all(np.diff(alpha) > 0)),
        "success": bool(best["success"].iloc[0]),
        "objective": float(best["objective"].iloc[0]),
        "objective_per_observation": float(best["objective_per_observation"].iloc[0]),
        "max_absolute_score_gradient": float(best["max_absolute_score_gradient"].iloc[0]),
        "median_pi": float(best["median_pi"].iloc[0]),
        "min_pi": float(best["min_pi"].iloc[0]),
        "max_pi": float(best["max_pi"].iloc[0]),
        "covariance_finite": bool(sw["finite"]),
        "covariance_PSD": bool(sw["V_positive_semidefinite"]),
        "full_covariance_condition_number": float(cov.loc[cov["block"].eq("full_covariance"), "condition_number"].iloc[0]),
        "domain_block_condition_number": float(cov.loc[cov["block"].eq("18-domain scientific covariance block"), "condition_number"].iloc[0]),
        "max_correlation_within_domain_block": float(cov.loc[cov["block"].eq("18-domain scientific covariance block correlations"), "max_abs_correlation"].iloc[0]),
        "max_abs_coefficient": float(np.max(np.abs(theta))),
        "max_robust_SE": float(np.max(sw["se"])),
        "extreme_abs_coef_gt_5_count": int(param["abs_coef_gt_5"].sum()),
        "extreme_SE_gt_5_count": int(param["SE_gt_5"].sum()),
    }])
    return theta, fit_qc, cov, dom, diag, param


def compare_domains(main, old):
    keep = ["Domain", "component", "parameter", "Coefficient", "Robust SE", "OR", "Raw P", "BH-FDR P"]
    comp = main[keep].merge(old[keep], on=["Domain", "component", "parameter"], suffixes=("_burden3", "_burden4_same_sample"))
    comp["coefficient_change_burden3_minus_burden4"] = comp["Coefficient_burden3"] - comp["Coefficient_burden4_same_sample"]
    comp["OR_change_burden3_minus_burden4"] = comp["OR_burden3"] - comp["OR_burden4_same_sample"]
    comp["SE_change_burden3_minus_burden4"] = comp["Robust SE_burden3"] - comp["Robust SE_burden4_same_sample"]
    comp["raw_P_lt_0.05_burden3"] = comp["Raw P_burden3"] < 0.05
    comp["raw_P_lt_0.05_burden4_same_sample"] = comp["Raw P_burden4_same_sample"] < 0.05
    comp["raw_significance_status_change"] = comp["raw_P_lt_0.05_burden3"] != comp["raw_P_lt_0.05_burden4_same_sample"]
    comp["BH_FDR_lt_0.05_burden3"] = comp["BH-FDR P_burden3"] < 0.05
    comp["BH_FDR_lt_0.05_burden4_same_sample"] = comp["BH-FDR P_burden4_same_sample"] < 0.05
    comp["FDR_status_change"] = comp["BH_FDR_lt_0.05_burden3"] != comp["BH_FDR_lt_0.05_burden4_same_sample"]
    comp["direction_change"] = np.sign(comp["Coefficient_burden3"]) != np.sign(comp["Coefficient_burden4_same_sample"])
    rows = []
    for component, sub in comp.groupby("component", sort=False):
        rows.append({
            "component": component,
            "domains": len(sub),
            "raw_P_lt_0.05_burden3": int(sub["raw_P_lt_0.05_burden3"].sum()),
            "raw_P_lt_0.05_burden4_same_sample": int(sub["raw_P_lt_0.05_burden4_same_sample"].sum()),
            "raw_significance_status_changes": int(sub["raw_significance_status_change"].sum()),
            "BH_FDR_lt_0.05_burden3": int(sub["BH_FDR_lt_0.05_burden3"].sum()),
            "BH_FDR_lt_0.05_burden4_same_sample": int(sub["BH_FDR_lt_0.05_burden4_same_sample"].sum()),
            "FDR_status_changes": int(sub["FDR_status_change"].sum()),
            "direction_changes": int(sub["direction_change"].sum()),
            "max_abs_coefficient_change": float(sub["coefficient_change_burden3_minus_burden4"].abs().max()),
            "max_abs_OR_change": float(sub["OR_change_burden3_minus_burden4"].abs().max()),
            "max_abs_SE_change": float(sub["SE_change_burden3_minus_burden4"].abs().max()),
        })
    return pd.DataFrame(rows), comp


def make_report(sample_qc, outcome_qc, diag, main_domains, old_domains, comp_summary, comp):
    pub_cols = ["Domain", "Coefficient", "Robust SE", "OR", "95% CI", "Raw P", "BH-FDR P"]
    panel_a = main_domains.loc[main_domains["component"].eq("mixing"), pub_cols]
    panel_b = main_domains.loc[main_domains["component"].eq("full_ordinal"), pub_cols]
    changes = comp.loc[comp["raw_significance_status_change"] | comp["FDR_status_change"] | comp["direction_change"]]
    return f"""# {LABEL}

## Scope

Two prespecified changes: concurrent HAMD-17 removed from both components, and FIBSER burden recoded to `0 / 1 / >=2`.

Sample was rebuilt from the frozen raw workbook using only variables required by this new model. Site/province were not included in the primary model. Step 3 was not run.

## Sample QC

{sample_qc.to_markdown(index=False)}

## Raw Burden 0 / 1 / >=2 Counts by Week

{outcome_qc.loc[outcome_qc['dataset'].eq('raw_valid_FIBSER')].to_markdown(index=False)}

## Model Diagnostics

{diag.to_markdown(index=False)}

## Main Candidate: Panel A. Zero-inflation Component

OR is `OR_extra_zero=exp(-gamma)`.

{panel_a.to_markdown(index=False)}

## Main Candidate: Panel B. Ordinary Ordinal Burden Component

OR is `OR_higher_burden=exp(beta)`.

{panel_b.to_markdown(index=False)}

## Same-sample Comparator Summary

{comp_summary.to_markdown(index=False)}

## Status Changes Due to Outcome Recoding

{changes.to_markdown(index=False) if len(changes) else "No raw-significance, FDR-significance, or direction status changes."}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sha = b1.sha256(SOURCE)
    log("rebuilding eligible sample from raw workbook")
    base, long, analysis, mapping, zmap = build_new_sample()
    sample_qc = pd.DataFrame([
        {"stage": "baseline workbook sheet", "participants": base["participant_ID"].nunique(), "person_visits": np.nan},
        {"stage": "valid FIBSER burden rows", "participants": long.loc[long["burden_ord3"].isin([0, 1, 2]), "participant_ID"].nunique(), "person_visits": int(long["burden_ord3"].isin([0, 1, 2]).sum())},
        {"stage": "new model eligible analysis sample", "participants": analysis["participant_ID"].nunique(), "person_visits": len(analysis)},
    ])
    source_qc = pd.DataFrame([{
        "label": LABEL,
        "source_file": str(SOURCE),
        "SHA256": sha,
        "model_RHS_both_components": RHS,
        "sample_rule": "drop rows missing new outcome, time, 9 domains, age, sex, education, first_episode, baseline_HAMD17, baseline_SDS, or baseline_med_class; concurrent_HAMD17 not required",
    }])
    outcome_qc = outcome_counts(long, analysis)
    G, X = b1.make_design(analysis, mix_rhs=RHS, ord_rhs=RHS)
    forbidden = [c for c in list(G.columns) + list(X.columns) if "concurrent_HAMD" in c or "site" in c.lower() or "province" in c.lower()]
    if forbidden:
        raise SystemExit(f"STOP: forbidden model terms: {forbidden}")
    log(f"analysis sample participants={analysis['participant_ID'].nunique()} visits={len(analysis)}")
    log(f"design columns mixing={G.shape[1]} ordinal={X.shape[1]}")

    main_label = "burden3_0_1_ge2_without_concurrent_HAMD"
    old_label = "burden4_0_1_2_ge3_without_concurrent_HAMD_same_sample"
    theta3, fit3, cov3, dom3, diag3, param3 = fit_and_summarize(main_label, "burden_ord3", 2, analysis, G, X, 20261001)
    theta4, fit4, cov4, dom4, diag4, param4 = fit_and_summarize(old_label, "burden_ord4", 3, analysis, G, X, 20261002)
    comp_summary, comp = compare_domains(dom3, dom4)
    diag = pd.concat([diag3, diag4], ignore_index=True)

    pub_cols = ["Domain", "Coefficient", "Robust SE", "OR", "95% CI", "Raw P", "BH-FDR P"]
    write_book(OUT / "01_sample_and_outcome_QC.xlsx", {
        "source": source_qc,
        "sample": sample_qc,
        "outcome_counts": outcome_qc,
        "z_standardization": zmap,
        "variable_mapping": mapping,
    })
    write_book(OUT / "02_model_diagnostics.xlsx", {
        "diagnostics": diag,
        "main_fit_multistart": fit3,
        "same_sample_old_outcome_fit": fit4,
        "main_covariance_conditioning": cov3,
        "same_sample_old_covariance": cov4,
        "main_parameters": param3,
        "same_sample_old_parameters": param4,
    })
    write_book(OUT / "03_main_candidate_domain_inference.xlsx", {
        "Panel A Zero-inflation": dom3.loc[dom3["component"].eq("mixing"), pub_cols],
        "Panel B Ordinary ordinal": dom3.loc[dom3["component"].eq("full_ordinal"), pub_cols],
        "all_domain_inference": dom3,
    })
    write_book(OUT / "04_same_sample_outcome_recoding_comparison.xlsx", {
        "summary": comp_summary,
        "domain_comparison": comp,
        "old_outcome_same_sample_domain_inference": dom4,
    })
    report = make_report(sample_qc, outcome_qc, diag, dom3, dom4, comp_summary, comp)
    (OUT / "Step2_primary_candidate_burden3_no_concurrent_HAMD_report.md").write_text(report, encoding="utf-8")
    log(f"completed outputs: {OUT}")


if __name__ == "__main__":
    main()


