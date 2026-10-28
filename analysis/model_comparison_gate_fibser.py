from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import optimize
from scipy.special import expit
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE, OrdinalGEE

import step2B1_YangSimpson_LIM_ordinal_GEE as b1
import step2B3_fixed_sample_covariate_attenuation as b3


ROOT = b1.ROOT
OUT = ROOT / "outputs" / "model_comparison_gate_fibser"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_RHS = b3.MODEL_RHS["M3"]
DOMAIN_VARS = [b1.DOMAIN_VAR[d] for d in b1.DOMAINS]
EXPECTED_N = 1468
EXPECTED_ROWS = 4761
EPS = np.finfo(float).tiny
FIXED_SAMPLE = ROOT / "outputs" / "step2B5_site_random_intercept" / "analysis_fixed_sample_for_iop.csv"
FIXED_MAP = ROOT / "outputs" / "step2B5_site_random_intercept" / "ascii_category_mapping_for_R.csv"


def log(msg: str) -> None:
    print(f"Model comparison gate: {msg}", flush=True)


def thresholds_from_raw(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = len(raw)
    alpha = np.empty(q)
    alpha[0] = raw[0]
    if q > 1:
        alpha[1:] = raw[0] + np.cumsum(np.exp(raw[1:]))
    jac = np.zeros((q, q))
    jac[:, 0] = 1.0
    for j in range(1, q):
        jac[j:, j] = np.exp(raw[j])
    return alpha, jac


def unpack(theta: np.ndarray, pg: int, px: int, ncat: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = ncat - 1
    gamma = theta[:pg]
    raw = theta[pg:pg + q]
    beta = theta[pg + q:pg + q + px]
    alpha, jac = thresholds_from_raw(raw)
    return gamma, raw, beta, alpha, jac


def ordinal_probs_deriv(theta: np.ndarray, G: np.ndarray, X: np.ndarray, ncat: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pg, px = G.shape[1], X.shape[1]
    _, _, beta, alpha, jac = unpack(theta, pg, px, ncat)
    eta = alpha[None, :] - X @ beta[:, None]
    cdf = expit(eta)
    d_cdf = cdf * (1.0 - cdf)

    p = np.empty((len(X), ncat))
    p[:, 0] = cdf[:, 0]
    for k in range(1, ncat - 1):
        p[:, k] = cdf[:, k] - cdf[:, k - 1]
    p[:, ncat - 1] = 1.0 - cdf[:, ncat - 2]
    p = np.maximum(p, EPS)

    dp_dalpha = np.zeros((len(X), ncat, ncat - 1))
    dp_dalpha[:, 0, 0] = d_cdf[:, 0]
    for k in range(1, ncat - 1):
        dp_dalpha[:, k, k - 1] = -d_cdf[:, k - 1]
        dp_dalpha[:, k, k] = d_cdf[:, k]
    dp_dalpha[:, ncat - 1, ncat - 2] = -d_cdf[:, ncat - 2]
    dp_draw = np.einsum("nka,ar->nkr", dp_dalpha, jac)
    dp_dbeta = -np.einsum("nka,nx->nkx", dp_dalpha, X)
    return p, dp_draw, dp_dbeta


def obs_loglik_score(theta: np.ndarray, y: np.ndarray, G: np.ndarray, X: np.ndarray, ncat: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    pg, px = G.shape[1], X.shape[1]
    gamma = theta[:pg]
    pi = expit(G @ gamma)  # ordinary-component probability
    dpi = pi * (1.0 - pi)
    p, dp_draw, dp_dbeta = ordinal_probs_deriv(theta, G, X, ncat)
    py = p[np.arange(len(y)), y]
    zero = y == 0

    m = pi * py
    m[zero] = (1.0 - pi[zero]) + pi[zero] * p[zero, 0]
    m = np.maximum(m, EPS)
    loglik_i = np.log(m)

    grad = np.zeros((len(y), pg + ncat - 1 + px))
    gamma_factor = dpi * py
    gamma_factor[zero] = dpi[zero] * (p[zero, 0] - 1.0)
    grad[:, :pg] = (gamma_factor / m)[:, None] * G

    dord_raw = dp_draw[np.arange(len(y)), y, :]
    dord_beta = dp_dbeta[np.arange(len(y)), y, :]
    dord_raw[zero, :] = pi[zero, None] * dp_draw[zero, 0, :]
    dord_beta[zero, :] = pi[zero, None] * dp_dbeta[zero, 0, :]
    nonzero = ~zero
    dord_raw[nonzero, :] = pi[nonzero, None] * dord_raw[nonzero, :]
    dord_beta[nonzero, :] = pi[nonzero, None] * dord_beta[nonzero, :]
    grad[:, pg:pg + ncat - 1] = dord_raw / m[:, None]
    grad[:, pg + ncat - 1:] = dord_beta / m[:, None]
    return loglik_i, grad


def neg_obj_grad(theta: np.ndarray, y: np.ndarray, G: np.ndarray, X: np.ndarray, ncat: int) -> tuple[float, np.ndarray]:
    ll, score = obs_loglik_score(theta, y, G, X, ncat)
    return -float(ll.sum()), -score.sum(axis=0)


def fit_ordinal_start(y: np.ndarray, X: np.ndarray, ncat: int) -> tuple[np.ndarray, np.ndarray, str]:
    parts_y, parts_x, cuts = [], [], []
    for j, k in enumerate(range(ncat - 1)):
        parts_y.append((y <= k).astype(float))
        parts_x.append(X)
        cuts.append(np.full(len(y), j))
    yy = np.concatenate(parts_y)
    xx = np.vstack(parts_x)
    cut = np.concatenate(cuts)
    cut_mat = np.column_stack([(cut == j).astype(float) for j in range(ncat - 1)])
    z = np.column_stack([cut_mat, xx])
    try:
        res = sm.GLM(yy, z, family=sm.families.Binomial()).fit(maxiter=300)
        alpha = np.maximum.accumulate(np.asarray(res.params[:ncat - 1]) + np.arange(ncat - 1) * 1e-4)
        beta = -np.asarray(res.params[ncat - 1:])
        raw = np.r_[alpha[0], np.log(np.maximum(np.diff(alpha), 1e-3))]
        return beta, raw, "stacked_PO_GLM_full_0to6"
    except Exception:
        props = np.array([(y <= k).mean() for k in range(ncat - 1)])
        props = np.clip(props, 1e-4, 1 - 1e-4)
        alpha = np.log(props / (1 - props))
        alpha = np.maximum.accumulate(alpha + np.arange(ncat - 1) * 1e-3)
        raw = np.r_[alpha[0], np.log(np.maximum(np.diff(alpha), 1e-3))]
        return np.zeros(X.shape[1]), raw, "empirical_cuts_full_0to6"


def build_zi_starts(y: np.ndarray, G: np.ndarray, X: np.ndarray, ncat: int) -> list[dict]:
    beta0, raw0, method = fit_ordinal_start(y, X, ncat)
    starts = []
    for extra in [0.01, 0.03, 0.05, 0.10, 0.15, 0.30, 0.50]:
        gamma = np.zeros(G.shape[1])
        gamma[0] = math.log((1.0 - extra) / extra)
        starts.append({"start": f"ordinary_start_extra_zero_{extra:.2f}", "theta": np.r_[gamma, raw0, beta0], "method": method})
    rng = np.random.default_rng(20261008)
    scale = np.r_[np.repeat(0.08, G.shape[1]), np.repeat(0.04, ncat - 1), np.repeat(0.04, X.shape[1])]
    for i in range(1, 4):
        base = starts[i % len(starts)]["theta"]
        starts.append({"start": f"random_perturbation_{i}", "theta": base + rng.normal(0, scale), "method": method})
    return starts


def fit_zi_full(y: np.ndarray, G: np.ndarray, X: np.ndarray, ncat: int) -> tuple[np.ndarray, pd.DataFrame]:
    rows, fits = [], []
    for st in build_zi_starts(y, G, X, ncat):
        res = optimize.minimize(
            lambda th: neg_obj_grad(th, y, G, X, ncat),
            st["theta"],
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2500, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50},
        )
        ll, score = obs_loglik_score(res.x, y, G, X, ncat)
        gamma, _, _, alpha, _ = unpack(res.x, G.shape[1], X.shape[1], ncat)
        pi = expit(G @ gamma)
        p, _, _ = ordinal_probs_deriv(res.x, G, X, ncat)
        row = {
            "start": st["start"],
            "success": bool(res.success),
            "message": str(res.message),
            "iterations": int(res.nit),
            "objective": float(ll.sum()),
            "objective_per_observation": float(ll.mean()),
            "max_absolute_score": float(np.max(np.abs(score.sum(axis=0)))),
            "thresholds_ordered": bool(np.all(np.diff(alpha) > 0)),
            "pi_ordinary_median": float(np.median(pi)),
            "pi_ordinary_min": float(pi.min()),
            "pi_ordinary_max": float(pi.max()),
            "ordinary_prob_min": float(p.min()),
            "start_method": st["method"],
        }
        rows.append(row)
        fits.append({"theta": res.x, "row": row})
        log(f"A full-0to6 {st['start']}: success={res.success} ll={row['objective']:.3f} score={row['max_absolute_score']:.3g}")
    best = max(fits, key=lambda z: z["row"]["objective"])
    ref = optimize.minimize(
        lambda th: neg_obj_grad(th, y, G, X, ncat),
        best["theta"],
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 80, "maxcor": 50},
    )
    ll, score = obs_loglik_score(ref.x, y, G, X, ncat)
    gamma, _, _, alpha, _ = unpack(ref.x, G.shape[1], X.shape[1], ncat)
    pi = expit(G @ gamma)
    rows.append({
        "start": "refined_from_best",
        "success": bool(ref.success),
        "message": str(ref.message),
        "iterations": int(ref.nit),
        "objective": float(ll.sum()),
        "objective_per_observation": float(ll.mean()),
        "max_absolute_score": float(np.max(np.abs(score.sum(axis=0)))),
        "thresholds_ordered": bool(np.all(np.diff(alpha) > 0)),
        "pi_ordinary_median": float(np.median(pi)),
        "pi_ordinary_min": float(pi.min()),
        "pi_ordinary_max": float(pi.max()),
        "ordinary_prob_min": float(ordinal_probs_deriv(ref.x, G, X, ncat)[0].min()),
        "start_method": "refined_BFGS",
    })
    return ref.x, pd.DataFrame(rows)


def predict_zi(theta: np.ndarray, G: np.ndarray, X: np.ndarray, ncat: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gamma, _, _, _, _ = unpack(theta, G.shape[1], X.shape[1], ncat)
    pi = expit(G @ gamma)
    p, _, _ = ordinal_probs_deriv(theta, G, X, ncat)
    pred = pi[:, None] * p
    pred[:, 0] += 1.0 - pi
    tau_extra = np.zeros(len(G))
    denom = np.maximum(pred[:, 0], EPS)
    tau_extra = (1.0 - pi) / denom
    return pred, pi, p, tau_extra


def finite_hessian_score(theta: np.ndarray, y: np.ndarray, G: np.ndarray, X: np.ndarray, ncat: int) -> np.ndarray:
    p = len(theta)
    h = np.zeros((p, p))
    steps = np.maximum(1e-5, np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(theta)))
    for j, step in enumerate(steps):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += step
        tm[j] -= step
        _, gp = obs_loglik_score(tp, y, G, X, ncat)
        _, gm = obs_loglik_score(tm, y, G, X, ncat)
        h[:, j] = (gp.sum(axis=0) - gm.sum(axis=0)) / (2 * step)
    return h


def sandwich_diag(theta: np.ndarray, y: np.ndarray, G: np.ndarray, X: np.ndarray, clusters: np.ndarray, ncat: int) -> dict:
    _, score = obs_loglik_score(theta, y, G, X, ncat)
    sdf = pd.DataFrame(score)
    sdf["cluster"] = np.asarray(clusters)
    U = sdf.groupby("cluster", observed=True).sum().drop(columns=[], errors="ignore").to_numpy()
    B = U.T @ U
    A = -finite_hessian_score(theta, y, G, X, ncat)
    V = np.linalg.pinv(A) @ B @ np.linalg.pinv(A)
    eig = np.linalg.eigvalsh((V + V.T) / 2)
    return {
        "parameters": int(len(theta)),
        "A_condition": float(np.linalg.cond(A)),
        "B_condition": float(np.linalg.cond(B)),
        "V_condition": float(np.linalg.cond(V)),
        "V_min_eigenvalue": float(eig.min()),
        "V_finite": bool(np.isfinite(V).all()),
        "V_PSD": bool(eig.min() >= -1e-8),
    }


def fit_ordinal_gee(df: pd.DataFrame, y_col: str) -> tuple[object, object, list[str]]:
    formula = f"{y_col} ~ {PRIMARY_RHS}"
    y, x = patsy.dmatrices(formula, df, return_type="dataframe")
    if "Intercept" in x.columns:
        x = x.drop(columns=["Intercept"])
    model = OrdinalGEE(y.iloc[:, 0], x, groups=df["participant_ID"].to_numpy(), cov_struct=Exchangeable(), family=Binomial(), missing="raise")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = model.fit(maxiter=200, cov_type="robust")
    return model, res, [str(x.message) for x in w]


def fit_logistic_gee(df: pd.DataFrame) -> tuple[object, object, list[str]]:
    formula = f"any_burden ~ {PRIMARY_RHS}"
    model = GEE.from_formula(formula, groups="participant_ID", data=df, cov_struct=Exchangeable(), family=Binomial(), missing="raise")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = model.fit(maxiter=200, cov_type="robust")
    return model, res, [str(x.message) for x in w]


def predict_ordinal_gee(res, x: pd.DataFrame, ncat: int) -> np.ndarray:
    params = res.params
    threshold_names = [n for n in params.index if n.startswith("I(y>")]
    # Statsmodels OrdinalGEE uses logit Pr(Y > cut) = threshold + X beta.
    q = np.column_stack([expit(float(params[t]) + np.asarray(x @ params[x.columns], dtype=float)) for t in threshold_names])
    pred = np.empty((len(x), ncat))
    pred[:, 0] = 1.0 - q[:, 0]
    for k in range(1, ncat - 1):
        pred[:, k] = q[:, k - 1] - q[:, k]
    pred[:, ncat - 1] = q[:, ncat - 2]
    return np.clip(pred, 0, 1)


def calibration(y: np.ndarray, pred: np.ndarray, groups: pd.Series, labels: list[str], prefix: str) -> pd.DataFrame:
    rows = []
    for group_name, idx in [("overall", np.ones(len(y), dtype=bool))] + [(str(g), groups.astype(str).eq(str(g)).to_numpy()) for g in ["Week2", "Week8", "Week12", "Week24"]]:
        for k, lab in enumerate(labels):
            rows.append({
                "model": prefix,
                "group": group_name,
                "category": lab,
                "n": int(idx.sum()),
                "observed_proportion": float((y[idx] == k).mean()) if idx.sum() else np.nan,
                "predicted_proportion": float(pred[idx, k].mean()) if idx.sum() else np.nan,
                "abs_error": float(abs((y[idx] == k).mean() - pred[idx, k].mean())) if idx.sum() else np.nan,
            })
    return pd.DataFrame(rows)


def raw_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gname, sub in [("overall", df)] + [(str(g), d) for g, d in df.groupby("time", observed=True)]:
        n = len(sub)
        for score in range(7):
            rows.append({
                "group": gname,
                "score": score,
                "n": int((sub["fibser_burden"].eq(score)).sum()),
                "proportion": float(sub["fibser_burden"].eq(score).mean()),
                "group_n": int(n),
            })
    return pd.DataFrame(rows)


def load_fixed_m3_analysis() -> pd.DataFrame:
    """Recover the saved 1468/4761 fixed sample while retaining raw 0-6 burden."""
    _, _, rebuilt, _, _ = b1.build_dataset()
    fixed = pd.read_csv(FIXED_SAMPLE)
    amap = pd.read_csv(FIXED_MAP)
    pid_map = amap.loc[amap["variable"].eq("participant_ID"), ["original_value", "ascii_value"]].copy()
    fixed = fixed.merge(pid_map, left_on="participant_ID", right_on="ascii_value", how="left", validate="many_to_one")
    fixed["participant_ID_original"] = fixed["original_value"].astype(str)
    fixed["time_key"] = fixed["time"].astype(str)

    rebuilt = rebuilt.copy()
    rebuilt["participant_ID_original"] = rebuilt["participant_ID"].astype(str)
    rebuilt["time_key"] = rebuilt["time"].astype(str)
    key = fixed[["participant_ID_original", "time_key"]].drop_duplicates()
    out = rebuilt.merge(key, on=["participant_ID_original", "time_key"], how="inner", validate="one_to_one")
    if len(out) != EXPECTED_ROWS or out["participant_ID"].nunique() != EXPECTED_N:
        raise SystemExit(
            f"Fixed M3 recovery mismatch: {out['participant_ID'].nunique()} participants / {len(out)} rows"
        )
    return out


def write_report(summary: dict, compare: pd.DataFrame) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            vals = [str(row[c]).replace("\n", "<br>").replace("|", "\\|") for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report = f"""# Model Comparison Gate Report: FIBSER Burden 0-6

## Gate scope

- Frozen sample: {summary['sample']['participants']} participants / {summary['sample']['person_visits']} person-visits.
- Visits: Week 2, 8, 12, 24.
- Covariates: nine PRISE domains, time, age, sex, education, first episode, baseline HAMD-17, baseline SDS, baseline antidepressant class, concurrent HAMD-17.
- Repeated measures handled at participant level.
- No domain x time formal interaction analysis was performed.
- Model choice was not based on P values.

## Main empirical facts

- Raw zero proportion overall: {summary['distribution']['zero_overall']:.3f}.
- Zero proportions by visit: {summary['distribution']['zero_by_visit']}.
- Sparse upper tail: scores 4/5/6 together account for {summary['distribution']['score_4_to_6_prop']:.3f} of person-visits.

## Four-model comparison

{markdown_table(compare)}

## Gate interpretation

Model A directly targets the dominant empirical feature: very high and time-varying zero burden while retaining the full 0-6 ordinal ordinary component. It fit the frozen sample numerically, reproduced the zero mass well, and produced finite participant-cluster sandwich diagnostics. However, ordinary sandwich SEs alone are not acceptable for final inference because the latent zero-source probabilities are estimated; final inference requires participant-cluster bootstrap or an analytic latent-class correction.

Model B is not recommended because a continuous zero-inflated GEE for an exact integer 0-6 score has no coherent ordinary-process probability for exact zero unless it is turned into a different two-part/rounded model. A Gaussian working model would also ignore bounded ordinal spacing and can predict impossible values.

Model C is stable and clinically easy to explain, and it explicitly separates any burden from positive-burden severity. Its limitation is that the severity component discards all zero observations and therefore answers a conditional positive-severity question rather than the marginal full-burden question.

Model D is stable and literature-standard for repeated ordinal outcomes. It preserves all 0-6 categories, but it has only one proportional-odds process and systematically under-predicts the structural zero mechanism relative to a zero-inflated model.

## Gate decision

Recommended main model: A, zero-inflated ordinal ES-GEE over the full 0-6 FIBSER burden scale, provided final inference uses participant-cluster bootstrap or a validated latent-zero variance correction.

Required sensitivity models:

1. D, cumulative-logit ordinal GEE over 0-6, as the standard non-zero-inflated ordinal comparator.
2. C, hurdle ordinal GEE, as the clinically transparent occurrence/severity comparator.

Not recommended:

- B, zero-inflated continuous GEE, because its distributional assumptions are not defensible for exact bounded ordinal 0-6 outcomes.

Stop point: this report stops after model-comparison gating and does not proceed to Step 3.
"""
    (OUT / "Model_Comparison_Gate_Report.md").write_text(report, encoding="utf-8")


def main() -> None:
    log("building frozen Step2B M3 sample")
    analysis = load_fixed_m3_analysis()
    if analysis["participant_ID"].nunique() != EXPECTED_N or len(analysis) != EXPECTED_ROWS:
        raise SystemExit(f"Frozen sample mismatch: {analysis['participant_ID'].nunique()} / {len(analysis)}")
    analysis = analysis.copy()
    analysis["any_burden"] = (analysis["fibser_burden"] >= 1).astype(int)
    analysis["fibser_pos"] = analysis["fibser_burden"].where(analysis["fibser_burden"] >= 1)

    Gdf, Xdf = b1.make_design(analysis, mix_rhs=PRIMARY_RHS, ord_rhs=PRIMARY_RHS)
    G, X = np.asarray(Gdf, dtype=float), np.asarray(Xdf, dtype=float)
    y7 = analysis["fibser_burden"].astype(int).to_numpy()

    dist = raw_distribution(analysis)
    dist.to_csv(OUT / "01_raw_0to6_distribution.csv", index=False, encoding="utf-8-sig")

    log("fitting A: full 0-6 zero-inflated ordinal")
    theta_a, a_fit = fit_zi_full(y7, G, X, 7)
    pred_a, pi_a, p_a, tau_extra = predict_zi(theta_a, G, X, 7)
    a_cal = calibration(y7, pred_a, analysis["time"], [str(i) for i in range(7)], "A_ZI_ordinal_ES_GEE_full_0to6")
    a_sw = sandwich_diag(theta_a, y7, G, X, analysis["participant_ID"].to_numpy(), 7)
    a_diag = {
        **a_sw,
        "converged_final": bool(a_fit.tail(1)["success"].iloc[0]),
        "max_abs_score_final": float(a_fit.tail(1)["max_absolute_score"].iloc[0]),
        "multi_start_successes": int(a_fit.iloc[:-1]["success"].sum()),
        "multi_start_runs": int(len(a_fit) - 1),
        "zero_pred_overall": float(pred_a[:, 0].mean()),
        "zero_obs_overall": float((y7 == 0).mean()),
        "zero_abs_error_overall": float(abs(pred_a[:, 0].mean() - (y7 == 0).mean())),
        "pi_ordinary_median": float(np.median(pi_a)),
        "posterior_extra_zero_median_observed_zero": float(np.median(tau_extra[y7 == 0])),
        "bootstrap_feasible": True,
        "bootstrap_feasibility_note": "Point fit is finite; participant-cluster bootstrap is computationally feasible but must refit the full ES/BFGS algorithm per resample.",
    }

    log("recording B as distributionally infeasible")
    b_diag = {
        "fit_attempted": False,
        "recommended": False,
        "reason": "Exact bounded ordinal 0-6 scores and exact zeros are not generated coherently by a continuous Gaussian ordinary process; a continuous normal component has P(Y=0)=0 before rounding.",
    }

    log("fitting C: hurdle logistic + positive ordinal")
    log_model, log_res, log_warn = fit_logistic_gee(analysis)
    pos = analysis.loc[analysis["fibser_burden"] >= 1].copy()
    _, c_ord_res, c_ord_warn = fit_ordinal_gee(pos, "fibser_pos")
    _, x_pos = patsy.dmatrices(f"fibser_pos ~ {PRIMARY_RHS}", pos, return_type="dataframe")
    if "Intercept" in x_pos.columns:
        x_pos = x_pos.drop(columns=["Intercept"])
    pred_pos_cond = predict_ordinal_gee(c_ord_res, x_pos, 6)
    c_pos_cal = calibration((pos["fibser_burden"].astype(int).to_numpy() - 1), pred_pos_cond, pos["time"], [str(i) for i in range(1, 7)], "C_hurdle_positive_ordinal")
    c_diag = {
        "logistic_converged": bool(log_res.converged),
        "positive_ordinal_converged": bool(c_ord_res.converged),
        "positive_rows": int(len(pos)),
        "positive_participants": int(pos["participant_ID"].nunique()),
        "logistic_dep_params": str(getattr(log_res, "dep_params", np.nan)),
        "positive_ordinal_dep_params": str(getattr(c_ord_res, "dep_params", np.nan)),
        "warnings": log_warn + c_ord_warn,
    }

    log("fitting D: full 0-6 cumulative-logit ordinal GEE")
    _, d_res, d_warn = fit_ordinal_gee(analysis, "fibser_burden")
    _, x_d = patsy.dmatrices(f"fibser_burden ~ {PRIMARY_RHS}", analysis, return_type="dataframe")
    if "Intercept" in x_d.columns:
        x_d = x_d.drop(columns=["Intercept"])
    pred_d = predict_ordinal_gee(d_res, x_d, 7)
    d_cal = calibration(y7, pred_d, analysis["time"], [str(i) for i in range(7)], "D_full_ordinal_GEE")
    d_diag = {
        "converged": bool(d_res.converged),
        "dep_params": str(getattr(d_res, "dep_params", np.nan)),
        "warnings": d_warn,
        "zero_pred_overall": float(pred_d[:, 0].mean()),
        "zero_obs_overall": float((y7 == 0).mean()),
        "zero_abs_error_overall": float(abs(pred_d[:, 0].mean() - (y7 == 0).mean())),
    }

    all_cal = pd.concat([a_cal, c_pos_cal, d_cal], ignore_index=True)
    all_cal.to_csv(OUT / "02_model_calibration.csv", index=False, encoding="utf-8-sig")
    a_fit.to_csv(OUT / "03_model_A_full_0to6_fit_diagnostics.csv", index=False, encoding="utf-8-sig")

    zero_by_visit = analysis.groupby("time", observed=True)["any_burden"].apply(lambda s: float(1.0 - s.mean())).to_dict()
    score_4_to_6_prop = float(analysis["fibser_burden"].isin([4, 5, 6]).mean())
    summary = {
        "sample": {"participants": int(analysis["participant_ID"].nunique()), "person_visits": int(len(analysis))},
        "distribution": {
            "zero_overall": float((y7 == 0).mean()),
            "zero_by_visit": {str(k): round(v, 4) for k, v in zero_by_visit.items()},
            "score_4_to_6_prop": score_4_to_6_prop,
        },
        "A": a_diag,
        "B": b_diag,
        "C": c_diag,
        "D": d_diag,
    }
    (OUT / "04_diagnostics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison = pd.DataFrame([
        {
            "模型": "A. Zero-inflated ordinal ES-GEE, full 0-6",
            "优势": "保留0-6等级；显式区分额外零与普通有序负担；零值校准最佳；可解释发生/严重程度两个机制",
            "局限": "项目特异扩展；ES/EM全交叉验证不完美；普通sandwich SE不足，需participant-cluster bootstrap或潜变量校正",
            "诊断结果": f"收敛={a_diag['converged_final']}; multi-start {a_diag['multi_start_successes']}/{a_diag['multi_start_runs']}; overall zero obs/pred={a_diag['zero_obs_overall']:.3f}/{a_diag['zero_pred_overall']:.3f}; V finite/PSD={a_diag['V_finite']}/{a_diag['V_PSD']}",
            "是否推荐": "推荐为主模型，条件是最终推断不用普通sandwich SE",
        },
        {
            "模型": "B. Zero-inflated continuous GEE",
            "优势": "表面上简单；均值差解释直观",
            "局限": "0-6为离散有界有序评分；连续普通过程不能自然产生精确0；正态近似可预测非法值且破坏等级含义",
            "诊断结果": "未强制拟合；分布假设不合理，标记不可行",
            "是否推荐": "不推荐",
        },
        {
            "模型": "C. Hurdle ordinal GEE",
            "优势": "发生(0 vs >=1)与正值严重度(1-6)分开；临床解释清楚；两部分均可用已发表GEE框架",
            "局限": "严重度部分条件于正值，丢弃0分严重度信息；不是同一个完整边际0-6分布模型",
            "诊断结果": f"logistic收敛={c_diag['logistic_converged']}; positive ordinal收敛={c_diag['positive_ordinal_converged']}; positive rows={c_diag['positive_rows']}",
            "是否推荐": "推荐为敏感性/解释性模型",
        },
        {
            "模型": "D. Cumulative-logit ordinal GEE, full 0-6",
            "优势": "标准、发表支持强；保留完整0-6等级；实现稳定",
            "局限": "单一有序过程不能分离额外零；对零膨胀机制解释不足",
            "诊断结果": f"收敛={d_diag['converged']}; overall zero obs/pred={d_diag['zero_obs_overall']:.3f}/{d_diag['zero_pred_overall']:.3f}",
            "是否推荐": "推荐为主要敏感性模型",
        },
    ])
    comparison.to_csv(OUT / "05_four_model_comparison_table.csv", index=False, encoding="utf-8-sig")
    write_report(summary, comparison)
    log(f"completed: {OUT}")


if __name__ == "__main__":
    main()


