from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit
from statsmodels.stats.multitest import multipletests

import step2B1_YangSimpson_LIM_ordinal_GEE as b1


ROOT = b1.ROOT
OUT2 = ROOT / "outputs" / "step2B2_numerical_validation"
OUT = ROOT / "outputs" / "step2B3_fixed_sample_covariate_attenuation"

DOMAIN_TERMS = b1.DOMAIN_TERMS
DOMAINS = b1.DOMAINS
DOMAIN_VAR = b1.DOMAIN_VAR
TIME = "C(time, Treatment(reference='Week2'))"

MODEL_RHS = {
    "M0": f"1 + {DOMAIN_TERMS} + {TIME}",
    "M1": f"1 + {DOMAIN_TERMS} + {TIME} + age_z + C(sex) + C(education) + C(first_episode)",
    "M2": f"1 + {DOMAIN_TERMS} + {TIME} + age_z + C(sex) + C(education) + C(first_episode) + baseline_HAMD17_z + baseline_SDS_z + C(baseline_med_class)",
    "M3": f"1 + {DOMAIN_TERMS} + {TIME} + age_z + C(sex) + C(education) + C(first_episode) + baseline_HAMD17_z + baseline_SDS_z + C(baseline_med_class) + concurrent_HAMD17_z",
    "M4": b1.MIX_RHS_FULL,
}


def log(msg: str):
    print(f"Step2B.3: {msg}", flush=True)


def names_for(G, X):
    return [f"mixing_gamma:{c}" for c in G.columns] + ["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"] + [f"ordinal_beta:{c}" for c in X.columns]


def write_book(path: Path, sheets: dict[str, pd.DataFrame]):
    b1.write_book(path, sheets)


def load_refined_m4(names):
    ref = pd.read_excel(OUT2 / "04_refined_BFGS_results.xlsx", sheet_name="parameters")
    theta = ref.set_index("parameter").reindex(names)["refined_BFGS"].to_numpy(float)
    return theta


def fit_ordinal_start_fast(y, X):
    return b1.fit_ordinal_start(y, X)


def make_starts(y, G, X, rng, warm_theta=None, n_random=2):
    beta0, raw0, method = fit_ordinal_start_fast(y, X)
    starts = []
    for extra in [0.01, 0.05, 0.15, 0.30, 0.50]:
        gamma = np.zeros(G.shape[1])
        gamma[0] = math.log((1.0 - extra) / extra)
        starts.append({"start": f"ordinary_start_extra_zero_{extra:.2f}", "theta": np.r_[gamma, raw0, beta0], "start_method": method})
    scale = np.r_[np.repeat(0.12, G.shape[1]), np.repeat(0.06, 3), np.repeat(0.06, X.shape[1])]
    for i in range(1, n_random + 1):
        base = starts[i % 5]["theta"]
        starts.append({"start": f"random_perturbation_{i}", "theta": base + rng.normal(0, scale), "start_method": method})
    if warm_theta is not None:
        starts.insert(0, {"start": "warm_start_previous_layer", "theta": warm_theta, "start_method": "mapped_previous_solution"})
    return starts


def map_theta(prev_theta, prev_G_cols, prev_X_cols, G_cols, X_cols):
    prev_pg = len(prev_G_cols)
    out = np.zeros(len(G_cols) + 3 + len(X_cols))
    for j, c in enumerate(G_cols):
        if c in prev_G_cols:
            out[j] = prev_theta[prev_G_cols.index(c)]
    out[len(G_cols):len(G_cols) + 3] = prev_theta[prev_pg:prev_pg + 3]
    prev_beta0 = prev_pg + 3
    beta0 = len(G_cols) + 3
    for j, c in enumerate(X_cols):
        if c in prev_X_cols:
            out[beta0 + j] = prev_theta[prev_beta0 + prev_X_cols.index(c)]
    return out


def theta_summary(model, start, theta, y, G, X, success, message, nit):
    ll, score = b1.obs_loglik_score(theta, y, G, X)
    gamma, _, _, alpha, _ = b1.unpack(theta, G.shape[1], X.shape[1])
    pi = expit(np.asarray(G) @ gamma)
    return {
        "model": model,
        "start": start,
        "success": bool(success),
        "message": str(message),
        "iterations": int(nit),
        "objective": float(ll.sum()),
        "objective_per_observation": float(ll.mean()),
        "max_absolute_score": float(np.max(np.abs(score.sum(axis=0)))),
        "threshold_alpha0": float(alpha[0]),
        "threshold_alpha1": float(alpha[1]),
        "threshold_alpha2": float(alpha[2]),
        "thresholds_ordered": bool(alpha[0] < alpha[1] < alpha[2]),
        "median_pi": float(np.median(pi)),
        "min_pi": float(pi.min()),
        "max_pi": float(pi.max()),
    }


def fit_nested_model(model, y, G, X, starts):
    rows = []
    fits = []
    for st in starts:
        res = optimize.minimize(
            lambda th: b1.neg_obj_grad(th, y, G, X),
            st["theta"],
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 3000, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50},
        )
        row = theta_summary(model, st["start"], res.x, y, G, X, res.success, res.message, res.nit)
        row["start_method"] = st["start_method"]
        rows.append(row)
        fits.append({"theta": res.x, "summary": row})
        log(f"{model} {st['start']} success={res.success} ll={row['objective']:.6f} score={row['max_absolute_score']:.3g}")
    best = max(fits, key=lambda z: z["summary"]["objective"])
    ref = optimize.minimize(
        lambda th: b1.neg_obj_grad(th, y, G, X),
        best["theta"],
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 80, "maxcor": 50},
    )
    ref_row = theta_summary(model, "refined_from_best_start", ref.x, y, G, X, ref.success, ref.message, ref.nit)
    rows.append(ref_row)
    return ref.x, pd.DataFrame(rows)


def domain_table(model, theta, se, G_cols, X_cols):
    rows = []
    pg = len(G_cols)
    for domain in DOMAINS:
        var = DOMAIN_VAR[domain]
        if var in G_cols:
            idx = G_cols.index(var)
            est = float(theta[idx])
            s = float(se[idx])
            z = est / s if np.isfinite(s) and s > 0 else np.nan
            p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
            rows.append({
                "model": model,
                "Domain": domain,
                "component": "mixing",
                "parameter": var,
                "coefficient": est,
                "robust_SE": s,
                "OR": math.exp(-est),
                "OR_type": "OR_extra_zero=exp(-gamma)",
                "OR_95CI_low": math.exp(-(est + 1.96 * s)),
                "OR_95CI_high": math.exp(-(est - 1.96 * s)),
                "raw_P": p,
            })
        if var in X_cols:
            idx = pg + 3 + X_cols.index(var)
            est = float(theta[idx])
            s = float(se[idx])
            z = est / s if np.isfinite(s) and s > 0 else np.nan
            p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
            rows.append({
                "model": model,
                "Domain": domain,
                "component": "full_ordinal",
                "parameter": var,
                "coefficient": est,
                "robust_SE": s,
                "OR": math.exp(est),
                "OR_type": "OR_higher_burden=exp(beta)",
                "OR_95CI_low": math.exp(est - 1.96 * s),
                "OR_95CI_high": math.exp(est + 1.96 * s),
                "raw_P": p,
            })
    out = pd.DataFrame(rows)
    out["BH_FDR"] = np.nan
    for comp, idx in out.groupby("component").groups.items():
        idx = list(idx)
        pvals = out.loc[idx, "raw_P"].to_numpy(float)
        mask = np.isfinite(pvals)
        q = np.full(len(pvals), np.nan)
        if mask.any():
            q[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
        out.loc[idx, "BH_FDR"] = q
    return out


def pct_change(new, old):
    return np.nan if not np.isfinite(old) or abs(old) < 1e-12 else 100.0 * (new - old) / abs(old)


def attenuation_tables(all_domains):
    key = ["Domain", "component"]
    wide = all_domains.pivot(index=key, columns="model", values=["coefficient", "robust_SE", "OR"]).reset_index()
    wide.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in wide.columns]
    rows = []
    transitions = [("M0", "M1"), ("M1", "M2"), ("M2", "M3"), ("M3", "M4")]
    for _, r in wide.iterrows():
        base = {"Domain": r["Domain"], "component": r["component"]}
        for a, b in transitions:
            ca, cb = r[f"coefficient_{a}"], r[f"coefficient_{b}"]
            sea, seb = r[f"robust_SE_{a}"], r[f"robust_SE_{b}"]
            rows.append({
                **base,
                "transition": f"{a}_to_{b}",
                "coef_abs_change": cb - ca,
                "coef_pct_change_vs_previous": pct_change(cb, ca),
                "coef_pct_change_vs_M0": pct_change(cb, r["coefficient_M0"]),
                "OR_change": r[f"OR_{b}"] - r[f"OR_{a}"],
                "SE_ratio_vs_previous": seb / sea if sea > 0 else np.nan,
                "SE_ratio_vs_M0": seb / r["robust_SE_M0"] if r["robust_SE_M0"] > 0 else np.nan,
                "direction_changed": "YES" if np.sign(ca) != np.sign(cb) else "NO",
                "attenuation_vs_precision_note": classify_change(ca, cb, sea, seb),
            })
    change = pd.DataFrame(rows)
    summary_rows = []
    for _, r in wide.iterrows():
        summary_rows.append({
            "Domain": r["Domain"],
            "Component": r["component"],
            "M0_OR": r["OR_M0"],
            "M1_OR": r["OR_M1"],
            "M2_OR": r["OR_M2"],
            "M3_OR": r["OR_M3"],
            "M4_OR": r["OR_M4"],
            "M0_SE": r["robust_SE_M0"],
            "M1_SE": r["robust_SE_M1"],
            "M2_SE": r["robust_SE_M2"],
            "M3_SE": r["robust_SE_M3"],
            "M4_SE": r["robust_SE_M4"],
            "M2_to_M3_coef_change_pct": pct_change(r["coefficient_M3"], r["coefficient_M2"]),
            "M2_to_M3_SE_change_pct": pct_change(r["robust_SE_M3"], r["robust_SE_M2"]),
            "M3_to_M4_coef_change_pct": pct_change(r["coefficient_M4"], r["coefficient_M3"]),
            "M3_to_M4_SE_change_pct": pct_change(r["robust_SE_M4"], r["robust_SE_M3"]),
            "direction_stable": "YES" if len(set(np.sign([r[f"coefficient_M{i}"] for i in range(5)]))) <= 1 else "NO",
        })
    summary = pd.DataFrame(summary_rows)
    return wide, change, summary


def classify_change(ca, cb, sea, seb):
    mag_drop = abs(cb) < 0.8 * abs(ca)
    se_infl = seb > 1.2 * sea
    coef_stable = abs(cb - ca) < 0.05
    if mag_drop and se_infl:
        return "effect attenuation plus SE inflation"
    if mag_drop:
        return "effect attenuation"
    if coef_stable and se_infl:
        return "precision loss"
    return "no major attenuation/precision signal"


def overlap_diagnostic(df):
    rows = []
    h = df["concurrent_HAMD17_z"].to_numpy(float)
    for domain in DOMAINS:
        var = DOMAIN_VAR[domain]
        x = df[var].to_numpy(float)
        absent = h[x == 0]
        present = h[x == 1]
        r = np.corrcoef(x, h)[0, 1] if np.std(x) > 0 and np.std(h) > 0 else np.nan
        t, p = stats.ttest_ind(present, absent, equal_var=False, nan_policy="omit") if len(present) > 1 and len(absent) > 1 else (np.nan, np.nan)
        rows.append({
            "Domain": domain,
            "domain_variable": var,
            "priority_marker": domain in {"Sleep", "Gastrointestinal", "Nervous system", "Genital/Urinary", "Other psychiatric"},
            "present_n": int((x == 1).sum()),
            "absent_n": int((x == 0).sum()),
            "concurrent_HAMD_z_mean_present": float(np.mean(present)) if len(present) else np.nan,
            "concurrent_HAMD_z_mean_absent": float(np.mean(absent)) if len(absent) else np.nan,
            "mean_difference_present_minus_absent": float(np.mean(present) - np.mean(absent)) if len(present) and len(absent) else np.nan,
            "point_biserial_correlation": float(r),
            "welch_t_raw_P": float(p) if np.isfinite(p) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mean_difference_present_minus_absent", key=lambda s: s.abs(), ascending=False)


def make_report(qc, model_qc, main_summary, change, overlap, m4_compare):
    m2m3 = change.loc[change["transition"].eq("M2_to_M3")].copy()
    m3m4 = change.loc[change["transition"].eq("M3_to_M4")].copy()
    top_m2m3 = m2m3.reindex(m2m3["coef_pct_change_vs_previous"].abs().sort_values(ascending=False).index).head(8)
    top_m3m4 = m3m4.reindex(m3m4["coef_pct_change_vs_previous"].abs().sort_values(ascending=False).index).head(8)
    dir_rev = change.loc[change["direction_changed"].eq("YES")]
    return f"""# Step2B.3 Fixed-Sample Covariate Attenuation Audit

## Fixed Sample QC

{qc.to_markdown(index=False)}

## M4 QC Against Step2B.2

{m4_compare.to_markdown(index=False)}

M4 used the same full scientific specification as Step2B.2. No sample expansion, outcome recoding, domain recoding, hurdle model, or ordinary ordinal comparator was used.

## Nested Model Fit QC

{model_qc.to_markdown(index=False)}

## Main Attenuation Summary

See `03_attenuation_summary.xlsx` for the requested compact table. The first rows are:

{main_summary.head(18).to_markdown(index=False)}

## Largest Changes After Adding Concurrent HAMD: M2 to M3

{top_m2m3.to_markdown(index=False)}

## Largest Changes After Adding Site: M3 to M4

{top_m3m4.to_markdown(index=False)}

## Direction Reversal

{dir_rev.to_markdown(index=False) if len(dir_rev) else "No domain/component showed a coefficient sign reversal across adjacent model transitions."}

## Concurrent-HAMD Overlap Diagnostic

{overlap.to_markdown(index=False)}

## Interpretation Boundary

This is a fixed-sample numerical attenuation audit only. Loss of raw statistical significance is not interpreted as attenuation. Attenuation is judged by coefficient magnitude change; precision loss is judged by SE inflation with relatively stable coefficients.

Completed at Step2B.3. Step2C was not run.
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    actual_hash = b1.sha256(b1.SOURCE)
    hash_match = actual_hash.lower() == b1.EXPECTED_SHA256
    if not hash_match:
        log(f"WARNING source hash differs from Step2B.1/2 frozen hash: {actual_hash}")
    log("rebuilding fixed full-model sample")
    base, long, analysis, mapping, zmap = b1.build_dataset()
    y = analysis["burden_ord4"].to_numpy()
    qc = pd.DataFrame([{
        "source_hash": actual_hash,
        "expected_step2B_hash": b1.EXPECTED_SHA256,
        "hash_match": hash_match,
        "fixed_sample_rule": "Step2B.2 full complete-case rows used for every nested model",
        "participants": analysis["participant_ID"].nunique(),
        "person_visits": len(analysis),
        "sites": analysis["site"].nunique(),
        "row_id_checksum": pd.util.hash_pandas_object(analysis[["participant_ID", "time", "burden_ord4"]].astype(str), index=False).sum(),
    }])

    rng = np.random.default_rng(20260928)
    prev_theta = None
    prev_G_cols = None
    prev_X_cols = None
    all_domains = []
    all_model_qc = []
    theta_store = {}
    design_store = {}
    for model in ["M0", "M1", "M2", "M3", "M4"]:
        log(f"fitting/evaluating {model}")
        G, X = b1.make_design(analysis, mix_rhs=MODEL_RHS[model], ord_rhs=MODEL_RHS[model])
        G_cols, X_cols = list(G.columns), list(X.columns)
        design_store[model] = (G, X)
        if model == "M4":
            names = names_for(G, X)
            theta = load_refined_m4(names)
            summary = pd.DataFrame([theta_summary("M4", "Step2B2_refined_BFGS_frozen", theta, y, G, X, True, "frozen Step2B.2 refined_BFGS solution; no refit", 0)])
        else:
            warm = map_theta(prev_theta, prev_G_cols, prev_X_cols, G_cols, X_cols) if prev_theta is not None else None
            starts = make_starts(y, G, X, rng, warm_theta=warm, n_random=2)
            theta, summary = fit_nested_model(model, y, G, X, starts)
        theta_store[model] = theta
        all_model_qc.append(summary)
        log(f"{model} sandwich")
        sw = b1.sandwich(theta, y, G, X, analysis["participant_ID"])
        dom = domain_table(model, theta, sw["se"], G_cols, X_cols)
        dom["A_condition"] = sw["A_condition"]
        dom["B_condition"] = sw["B_condition"]
        dom["V_condition"] = sw["V_condition"]
        all_domains.append(dom)
        prev_theta, prev_G_cols, prev_X_cols = theta, G_cols, X_cols

    domain_results = pd.concat(all_domains, ignore_index=True)
    model_qc = pd.concat(all_model_qc, ignore_index=True)
    _, change, main_summary = attenuation_tables(domain_results)
    overlap = overlap_diagnostic(analysis)

    final_inf = pd.read_excel(OUT2 / "10_final_domain_inference.xlsx", sheet_name="domain_inference")
    m4 = domain_results.loc[domain_results["model"].eq("M4")].copy()
    merged = m4.merge(final_inf, on=["Domain", "component", "parameter"], suffixes=("_M4", "_Step2B2_final"))
    m4_compare = pd.DataFrame([{
        "max_abs_coefficient_difference": float((merged["coefficient_M4"] - merged["coefficient_Step2B2_final"]).abs().max()),
        "max_abs_SE_difference": float((merged["robust_SE_M4"] - merged["robust_SE_Step2B2_final"]).abs().max()),
        "max_abs_OR_difference": float((merged["OR"] - merged["OR_extra_zero"].where(merged["component"].eq("mixing"), merged["OR_higher_burden"])).abs().max()),
        "rows_compared": len(merged),
        "QC": "M4 reproduces Step2B.2 final domain inference" if (merged["coefficient_M4"] - merged["coefficient_Step2B2_final"]).abs().max() < 1e-8 else "CHECK",
    }])

    write_book(OUT / "01_fixed_sample_QC.xlsx", {
        "fixed_sample_QC": qc,
        "model_fit_QC": model_qc,
        "M4_vs_Step2B2_final": m4_compare,
    })
    write_book(OUT / "02_nested_model_domain_results.xlsx", {
        "domain_results": domain_results,
        "model_fit_QC": model_qc,
    })
    write_book(OUT / "03_attenuation_summary.xlsx", {
        "main_summary": main_summary,
        "all_transitions": change,
        "concurrent_HAMD_M2_to_M3": change.loc[change["transition"].eq("M2_to_M3")],
        "site_M3_to_M4": change.loc[change["transition"].eq("M3_to_M4")],
    })
    write_book(OUT / "04_concurrent_HAMD_overlap_diagnostic.xlsx", {"overlap_diagnostic": overlap})
    report = make_report(qc, model_qc, main_summary, change, overlap, m4_compare)
    (OUT / "Step2B3_adjustment_attenuation_report.md").write_text(report, encoding="utf-8")
    log(f"completed outputs: {OUT}")


if __name__ == "__main__":
    main()


