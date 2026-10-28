from __future__ import annotations

import hashlib
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import optimize, stats
from scipy.special import expit, logsumexp
from statsmodels.miscmodels.ordinal_model import OrderedModel


from public_paths import ROOT, authorized_data_path
SOURCE = authorized_data_path("PROUD_RAW_WORKBOOK", "raw_workbook.xlsx")
OUT = ROOT / "outputs" / "step2B1_YangSimpson_LIM_ordinal_GEE"
EXPECTED_SHA256 = "b4be917deeb7870a7b25ae243b454a430c29c7c9fe5092d5595416ac3892564b"
EXPECTED_N = 1543
EXPECTED_SITES = 21
EPS = np.finfo(float).tiny

WEEKS = ["Week2", "Week8", "Week12", "Week24"]
WEEK_LABELS = {"Week2": "Week 2", "Week8": "Week 8", "Week12": "Week 12", "Week24": "Week 24"}
FIBSER_BURDEN_CODE = "JH10_00_080_13_4038_1"

DOMAINS = [
    "Gastrointestinal",
    "Cardiac",
    "Skin",
    "Nervous system",
    "Eye/Ear",
    "Genital/Urinary",
    "Sleep",
    "Sexual",
    "Other psychiatric",
]
OCC_CODES = {
    "Sleep": "JH05_10_012_510_4040_1",
    "Gastrointestinal": "JH03_01_000708_4040_1",
    "Other psychiatric": "JH03_01_000749_4040_1",
    "Nervous system": "JH03_01_000745_4040_1",
    "Cardiac": "JH03_01_000743_4040_1",
    "Eye/Ear": "JH03_01_000746_4040_1",
    "Skin": "JH03_01_000744_4040_1",
    "Sexual": "JH03_01_000748_4040_1",
    "Genital/Urinary": "JH03_01_000747_4040_1",
}
PRISE_OCCURRENCE_LABELS = {
    "Gastrointestinal": ["胃肠"],
    "Cardiac": ["心脏"],
    "Skin": ["皮肤"],
    "Nervous system": ["神经系统"],
    "Eye/Ear": ["眼/耳", "眼耳"],
    "Genital/Urinary": ["生殖/泌尿", "生殖泌尿"],
    "Sleep": ["睡眠"],
    "Sexual": ["性功能"],
    "Other psychiatric": ["其他精神症状"],
}
DOMAIN_VAR = {
    "Gastrointestinal": "Gastrointestinal",
    "Cardiac": "Cardiac",
    "Skin": "Skin",
    "Nervous system": "Nervous",
    "Eye/Ear": "EyeEar",
    "Genital/Urinary": "Genitourinary",
    "Sleep": "Sleep",
    "Sexual": "Sexual",
    "Other psychiatric": "OtherPsych",
}

DOMAIN_TERMS = " + ".join(DOMAIN_VAR[d] for d in DOMAINS)
COMMON_RHS = (
    f"{DOMAIN_TERMS} + C(time, Treatment(reference='Week2')) + age_z + C(sex) + "
    "C(education) + C(first_episode) + C(baseline_med_class) + baseline_HAMD17_z + "
    "baseline_SDS_z + concurrent_HAMD17_z + C(site)"
)
MIX_RHS_FULL = "1 + " + COMMON_RHS
ORD_RHS_FULL = "1 + " + COMMON_RHS
MIX_RHS_NO_SITE = MIX_RHS_FULL.replace(" + C(site)", "")
ORD_RHS_NO_SITE = ORD_RHS_FULL.replace(" + C(site)", "")
MIX_RHS_SIMPLE = f"1 + {DOMAIN_TERMS} + C(time, Treatment(reference='Week2'))"


def clean(x):
    if x is None or pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip()
        return np.nan if s == "" or s.lower() in {"nan", "na", "n/a", "none", "null"} else s
    return x


def norm_id(x):
    x = clean(x)
    if pd.isna(x):
        return pd.NA
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)) and np.isfinite(x) and float(x).is_integer():
        return str(int(x))
    s = str(x).strip()
    return s[:-2] if re.fullmatch(r"\d+\.0", s) else s


def numeric(s):
    raw = s.map(clean)
    direct = pd.to_numeric(raw, errors="coerce")
    extracted = pd.to_numeric(raw.astype("string").str.extract(r"^\s*\[?(-?\d+(?:\.\d+)?)\]?.*$")[0], errors="coerce")
    return direct.where(direct.notna(), extracted)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def id_col(df):
    for col in df.columns:
        s = str(col)
        if s.startswith("ID") or "入组编码" in s:
            return col
    return df.columns[0]


def code_col(df, code):
    return next((col for col in df.columns if code and code in str(col)), None)


def first_nonmissing(x):
    y = x.dropna()
    return np.nan if y.empty else y.iloc[0]


def burden_col(df):
    hit = code_col(df, FIBSER_BURDEN_CODE)
    if hit is not None:
        return hit
    anchor = next((i for i, col in enumerate(df.columns) if any(t in str(col) for t in ("FISER", "FIBSER", "GRSEB"))), None)
    return None if anchor is None else df.columns[anchor + 3]


def occurrence_col(df, domain):
    hit = code_col(df, OCC_CODES[domain])
    if hit is not None:
        return hit
    anchor = next((i for i, col in enumerate(df.columns) if "PRISE" in str(col)), None)
    if anchor is None:
        raise KeyError(f"PRISE anchor not found while locating occurrence column for {domain}")
    cols = list(df.columns)
    # Match occurrence items by their visible PRISE domain labels. Do not fall back
    # to historical positional order, because workbook versions may reorder domains.
    candidates = []
    for i, col in enumerate(cols[anchor + 1:], start=anchor + 1):
        text = str(col).replace(" ", "")
        if any(label.replace(" ", "") in text for label in PRISE_OCCURRENCE_LABELS[domain]):
            candidates.append(col)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Prefer the actual occurrence item over symptom-type or severity columns.
        occurrence_like = [c for c in candidates if "1=是" in str(c) or "1=有" in str(c)]
        if len(occurrence_like) == 1:
            return occurrence_like[0]
    raise KeyError(
        f"Could not uniquely locate PRISE occurrence column for {domain}; "
        f"candidates={[str(c) for c in candidates]}"
    )


def concurrent_hamd_col(df):
    candidates = [c for c in df.columns if str(c) == "HAMD总分" or str(c).startswith("HAMD总分.")]
    if candidates:
        return candidates[0]
    cols = list(df.columns)
    hamd_anchor = next((i for i, c in enumerate(cols) if "HAMD-17" in str(c)), None)
    hama_anchor = next((i for i, c in enumerate(cols) if "HAMA" in str(c) and "量表" in str(c)), None)
    if hamd_anchor is not None and hama_anchor is not None:
        totals = [c for c in cols[hamd_anchor:hama_anchor] if "总分" in str(c)]
        if totals:
            return totals[-1]
    return next((c for c in df.columns if "HAMD" in str(c) and "总分" in str(c)), None)


def sanitize_category(x):
    x = clean(x)
    if pd.isna(x):
        return np.nan
    return str(x).strip()


def write_book(path: Path, sheets: dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = re.sub(r"[\[\]\*:/\\?]", "_", name)[:31]
            out = df.copy()
            out.to_excel(writer, sheet_name=safe, index=False)
            ws = writer.sheets[safe]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for j, col in enumerate(out.columns, start=1):
                vals = out.iloc[:, j - 1].astype(str).head(250)
                width = min(max(len(str(col)) + 2, int(vals.map(len).max()) + 2 if len(vals) else 10, 10), 80)
                ws.column_dimensions[ws.cell(1, j).column_letter].width = width


def read_baseline(xls):
    df = pd.read_excel(SOURCE, sheet_name=xls.sheet_names[0], engine="openpyxl")
    base = pd.DataFrame({"participant_ID": df[id_col(df)].map(norm_id)})
    base["age"] = pd.to_numeric(df["年龄"], errors="coerce")
    base["site"] = df["分中心"].map(sanitize_category)
    base["baseline_med_class"] = df["药物类别"].map(sanitize_category)
    base["sex"] = pd.to_numeric(df["性别：1=男，2=女"], errors="coerce").map({1.0: "Male", 2.0: "Female"})
    base["education"] = df["患者文化程度："].map(sanitize_category)
    base["first_episode"] = pd.to_numeric(df["本次抑郁是否首次发作：1=首发，2=复发"], errors="coerce").map({1.0: "FirstEpisode", 2.0: "RecurrentEpisode"})
    base["baseline_HAMD17"] = pd.to_numeric(df["HAMD总分"], errors="coerce")
    sds_cols = ["席汉残疾量表（SDS） 工作/学习得分", "席汉残疾量表（SDS） 社交生活得分", "席汉残疾量表（SDS） 家庭生活/家庭责任得分"]
    base["baseline_SDS"] = df[sds_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=3)
    return base


def read_followup(xls):
    positions = {"Week2": (1, 5), "Week8": (2, 6), "Week12": (3, 7), "Week24": (4, 8)}
    frames = []
    mapping_rows = []
    for visit_order, (week, poss) in enumerate(positions.items(), start=1):
        for pos in poss:
            sheet = xls.sheet_names[pos]
            df = pd.read_excel(SOURCE, sheet_name=sheet, engine="openpyxl")
            bcol = burden_col(df)
            hcol = concurrent_hamd_col(df)
            out = pd.DataFrame({"participant_ID": df[id_col(df)].map(norm_id), "time": week, "visit_order": visit_order, "source_sheet": sheet})
            out["fibser_burden"] = numeric(df[bcol])
            out["concurrent_HAMD17"] = pd.to_numeric(df[hcol], errors="coerce") if hcol is not None else np.nan
            mapping_rows.extend([
                {"source_sheet": sheet, "source_column": str(id_col(df)), "analysis_variable": "participant_ID", "role": "ID", "coding": "normalized string"},
                {"source_sheet": sheet, "source_column": str(bcol), "analysis_variable": "fibser_burden", "role": "outcome", "coding": "FIBSER burden raw 0-6; Step2B.1 collapses to 0/1/2/ge3"},
                {"source_sheet": sheet, "source_column": str(hcol), "analysis_variable": "concurrent_HAMD17", "role": "covariate", "coding": "z-standardized using rebuilt long-data mean/SD"},
            ])
            for domain in DOMAINS:
                col = occurrence_col(df, domain)
                raw = numeric(df[col])
                out[DOMAIN_VAR[domain]] = raw.map({1.0: 1.0, 2.0: 0.0, 0.0: 0.0})
                mapping_rows.append({"source_sheet": sheet, "source_column": str(col), "analysis_variable": DOMAIN_VAR[domain], "role": "PRISE occurrence domain", "coding": "1=present; 2=no recoded 0; source 0 retained as 0"})
            frames.append(out)
    raw = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in raw.columns if c not in {"participant_ID", "time", "source_sheet"}]
    collapsed = raw.groupby(["participant_ID", "time"], dropna=False)[value_cols].agg(first_nonmissing).reset_index()
    return collapsed, pd.DataFrame(mapping_rows)


def build_dataset():
    xls = pd.ExcelFile(SOURCE, engine="openpyxl")
    base = read_baseline(xls)
    follow, follow_mapping = read_followup(xls)
    if len(base) != EXPECTED_N or base["participant_ID"].nunique() != EXPECTED_N or base["site"].nunique() != EXPECTED_SITES:
        raise SystemExit(f"STOP: cohort gate failed N={len(base)}, ID={base['participant_ID'].nunique()}, sites={base['site'].nunique()}")
    long = follow.merge(base, on="participant_ID", how="left")
    valid = long["fibser_burden"].isin(range(7))
    long["burden_ord4"] = np.nan
    long.loc[valid & long["fibser_burden"].eq(0), "burden_ord4"] = 0
    long.loc[valid & long["fibser_burden"].eq(1), "burden_ord4"] = 1
    long.loc[valid & long["fibser_burden"].eq(2), "burden_ord4"] = 2
    long.loc[valid & long["fibser_burden"].ge(3), "burden_ord4"] = 3
    long["time"] = pd.Categorical(long["time"], categories=WEEKS, ordered=True)
    for c in ["participant_ID", "site", "sex", "education", "first_episode", "baseline_med_class"]:
        long[c] = long[c].astype("category")
    z_rows = []
    for c in ["age", "baseline_HAMD17", "baseline_SDS", "concurrent_HAMD17"]:
        mean = long[c].mean()
        sd = long[c].std(ddof=0)
        long[c + "_z"] = (long[c] - mean) / sd
        z_rows.append({"variable": c, "analysis_variable": c + "_z", "mean_used": mean, "sd_used_ddof0": sd})
    required = [
        "participant_ID", "site", "time", "visit_order", "burden_ord4", "age_z", "sex",
        "education", "first_episode", "baseline_med_class", "baseline_HAMD17_z",
        "baseline_SDS_z", "concurrent_HAMD17_z",
    ] + list(DOMAIN_VAR.values())
    analysis = long.dropna(subset=required).copy()
    analysis["burden_ord4"] = analysis["burden_ord4"].astype(int)
    return base, long, analysis, follow_mapping, pd.DataFrame(z_rows)


def make_design(df, mix_rhs=MIX_RHS_FULL, ord_rhs=ORD_RHS_FULL):
    g = patsy.dmatrix(mix_rhs, df, return_type="dataframe")
    x = patsy.dmatrix(ord_rhs, df, return_type="dataframe")
    if "Intercept" in x.columns:
        x = x.drop(columns=["Intercept"])
    return g, x


def unpack(theta, pg, px):
    gamma = theta[:pg]
    raw = theta[pg:pg + 3]
    beta = theta[pg + 3:pg + 3 + px]
    a0, ld1, ld2 = raw
    thresholds = np.array([a0, a0 + np.exp(ld1), a0 + np.exp(ld1) + np.exp(ld2)])
    jac_alpha = np.array([
        [1.0, 0.0, 0.0],
        [1.0, np.exp(ld1), 0.0],
        [1.0, np.exp(ld1), np.exp(ld2)],
    ])
    return gamma, raw, beta, thresholds, jac_alpha


def ordinal_probs_and_deriv(theta, G, X):
    pg, px = G.shape[1], X.shape[1]
    _, _, beta, alpha, jac_alpha = unpack(theta, pg, px)
    eta = alpha[None, :] - X @ beta[:, None]
    cdf = expit(eta)
    d_cdf = cdf * (1.0 - cdf)
    p = np.column_stack([cdf[:, 0], cdf[:, 1] - cdf[:, 0], cdf[:, 2] - cdf[:, 1], 1.0 - cdf[:, 2]])
    # Machine-level floor only for denominators/logs; no substantive clipping.
    p = np.maximum(p, EPS)
    dp_dalpha = np.zeros((len(X), 4, 3))
    dp_dalpha[:, 0, 0] = d_cdf[:, 0]
    dp_dalpha[:, 1, 0] = -d_cdf[:, 0]
    dp_dalpha[:, 1, 1] = d_cdf[:, 1]
    dp_dalpha[:, 2, 1] = -d_cdf[:, 1]
    dp_dalpha[:, 2, 2] = d_cdf[:, 2]
    dp_dalpha[:, 3, 2] = -d_cdf[:, 2]
    dp_draw = np.einsum("nka,ar->nkr", dp_dalpha, jac_alpha)
    dp_dbeta = -np.einsum("nka,nx->nkx", dp_dalpha, np.asarray(X))
    return p, dp_draw, dp_dbeta


def obs_loglik_score(theta, y, G, X, obs_weights=None):
    y = np.asarray(y, dtype=int)
    Gv = np.asarray(G, dtype=float)
    Xv = np.asarray(X, dtype=float)
    n, pg, px = len(y), Gv.shape[1], Xv.shape[1]
    gamma = theta[:pg]
    pi = expit(Gv @ gamma)
    dpi = pi * (1.0 - pi)
    p, dp_draw, dp_dbeta = ordinal_probs_and_deriv(theta, Gv, Xv)
    py = p[np.arange(n), y]
    zero = y == 0
    m = pi * py
    m[zero] = (1.0 - pi[zero]) + pi[zero] * p[zero, 0]
    m = np.maximum(m, EPS)
    loglik_i = np.log(m)
    grad = np.zeros((n, pg + 3 + px))
    gamma_factor = dpi * py
    gamma_factor[zero] = dpi[zero] * (p[zero, 0] - 1.0)
    grad[:, :pg] = (gamma_factor / m)[:, None] * Gv
    dord_raw = dp_draw[np.arange(n), y, :]
    dord_beta = dp_dbeta[np.arange(n), y, :]
    dord_raw[zero, :] = pi[zero, None] * dp_draw[zero, 0, :]
    dord_beta[zero, :] = pi[zero, None] * dp_dbeta[zero, 0, :]
    nonzero = ~zero
    dord_raw[nonzero, :] = pi[nonzero, None] * dord_raw[nonzero, :]
    dord_beta[nonzero, :] = pi[nonzero, None] * dord_beta[nonzero, :]
    grad[:, pg:pg + 3] = dord_raw / m[:, None]
    grad[:, pg + 3:] = dord_beta / m[:, None]
    if obs_weights is not None:
        w = np.asarray(obs_weights, dtype=float)
        loglik_i = loglik_i * w
        grad = grad * w[:, None]
    return loglik_i, grad


def neg_obj_grad(theta, y, G, X):
    ll, grad = obs_loglik_score(theta, y, G, X)
    return -float(ll.sum()), -grad.sum(axis=0)


def complete_neg_obj_grad(theta, y, G, X, w_ord):
    y = np.asarray(y, dtype=int)
    Gv = np.asarray(G, dtype=float)
    n, pg, px = len(y), Gv.shape[1], X.shape[1]
    gamma = theta[:pg]
    pi = expit(Gv @ gamma)
    p, dp_draw, dp_dbeta = ordinal_probs_and_deriv(theta, Gv, np.asarray(X, dtype=float))
    w_ord = np.asarray(w_ord, dtype=float)
    ll = w_ord * (np.log(np.maximum(pi, EPS)) + np.log(np.maximum(p[np.arange(n), y], EPS))) + (1.0 - w_ord) * np.log(np.maximum(1.0 - pi, EPS))
    grad = np.zeros((n, pg + 3 + px))
    grad[:, :pg] = (w_ord - pi)[:, None] * Gv
    py = np.maximum(p[np.arange(n), y], EPS)
    grad[:, pg:pg + 3] = w_ord[:, None] * dp_draw[np.arange(n), y, :] / py[:, None]
    grad[:, pg + 3:] = w_ord[:, None] * dp_dbeta[np.arange(n), y, :] / py[:, None]
    return -float(ll.sum()), -grad.sum(axis=0)


def ordinal_only_neg_obj_grad(phi, y, X, w_ord):
    theta = np.r_[0.0, phi]
    G_dummy = np.ones((len(y), 1))
    val, grad = complete_neg_obj_grad(theta, y, G_dummy, X, w_ord)
    return val, grad[1:]


def em_mstep(theta, y, G, X, w_ord):
    pg, px = G.shape[1], X.shape[1]
    # Mixing part: fractional-response binomial GLM for ordinary-component probability.
    mix_model = sm.GLM(w_ord, np.asarray(G, dtype=float), family=sm.families.Binomial())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mix_res = mix_model.fit(start_params=theta[:pg], maxiter=200)
    gamma = np.asarray(mix_res.params)
    phi0 = theta[pg:pg + 3 + px]
    ord_res = optimize.minimize(
        lambda ph: ordinal_only_neg_obj_grad(ph, y, np.asarray(X, dtype=float), w_ord),
        phi0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 25, "ftol": 1e-8, "gtol": 1e-5, "maxls": 20},
    )
    return np.r_[gamma, ord_res.x], bool(mix_res.converged and ord_res.success), f"mixing_converged={mix_res.converged}; ordinal={ord_res.message}"


def fit_ordinal_start(y, X):
    try:
        parts_y = []
        parts_x = []
        parts_cut = []
        for j, k in enumerate([0, 1, 2]):
            parts_y.append((y <= k).astype(float))
            parts_x.append(np.asarray(X, dtype=float))
            parts_cut.append(np.full(len(y), j))
        yy = np.concatenate(parts_y)
        xx = np.vstack(parts_x)
        cut = np.concatenate(parts_cut)
        cut_mat = np.column_stack([(cut == j).astype(float) for j in [0, 1, 2]])
        z = np.column_stack([cut_mat, xx])
        res = sm.GLM(yy, z, family=sm.families.Binomial()).fit(maxiter=200)
        alpha = np.asarray(res.params[:3])
        # Stacked model has logit[P(Y<=k)]=alpha_k + X*b; target uses alpha_k - X*beta.
        beta = -np.asarray(res.params[3:])
        alpha = np.maximum.accumulate(alpha + np.array([0.0, 1e-4, 2e-4]))
        raw = np.array([alpha[0], np.log(max(alpha[1] - alpha[0], 1e-3)), np.log(max(alpha[2] - alpha[1], 1e-3))])
        return beta, raw, "stacked_PO_GLM"
    except Exception:
        prop = np.array([(y <= k).mean() for k in [0, 1, 2]])
        prop = np.clip(prop, 1e-4, 1 - 1e-4)
        alpha = np.log(prop / (1 - prop))
        alpha = np.maximum.accumulate(alpha + np.array([0, 1e-3, 2e-3]))
        return np.zeros(X.shape[1]), np.array([alpha[0], np.log(alpha[1] - alpha[0]), np.log(alpha[2] - alpha[1])]), "empirical_cuts"


def build_starts(y, G, X, rng):
    beta0, raw0, start_method = fit_ordinal_start(y, X)
    starts = []
    for extra in [0.01, 0.05, 0.15, 0.30, 0.50]:
        gamma = np.zeros(G.shape[1])
        gamma[0] = math.log((1.0 - extra) / extra)
        starts.append({"start": f"ordinary_start_extra_zero_{extra:.2f}", "theta": np.r_[gamma, raw0, beta0], "start_method": start_method})
    scale = np.r_[np.repeat(0.15, G.shape[1]), np.repeat(0.08, 3), np.repeat(0.08, X.shape[1])]
    for i in range(1, 6):
        base = starts[i % 5]["theta"]
        starts.append({"start": f"random_perturbation_{i}", "theta": base + rng.normal(0, scale), "start_method": start_method})
    return starts


def fit_bfgs_multistart(y, G, X, starts, label):
    rows = []
    fits = []
    for st in starts:
        res = optimize.minimize(
            lambda th: neg_obj_grad(th, y, G, X),
            st["theta"],
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 3000, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50},
        )
        ll, grad = obs_loglik_score(res.x, y, G, X)
        p, _, _ = ordinal_probs_and_deriv(res.x, np.asarray(G), np.asarray(X))
        gamma, raw, beta, alpha, _ = unpack(res.x, G.shape[1], X.shape[1])
        pi = expit(np.asarray(G) @ gamma)
        row = {
            "model": label,
            "start": st["start"],
            "start_method": st["start_method"],
            "converged": bool(res.success),
            "message": str(res.message),
            "iterations": int(res.nit),
            "final_objective_loglik": float(ll.sum()),
            "objective_per_observation": float(ll.mean()),
            "max_absolute_gradient": float(np.max(np.abs(grad.sum(axis=0)))),
            "threshold_alpha0": alpha[0],
            "threshold_alpha1": alpha[1],
            "threshold_alpha2": alpha[2],
            "thresholds_ordered": bool(alpha[0] < alpha[1] < alpha[2]),
            "pi_min": float(pi.min()),
            "pi_p05": float(np.quantile(pi, 0.05)),
            "pi_median": float(np.median(pi)),
            "pi_p95": float(np.quantile(pi, 0.95)),
            "pi_max": float(pi.max()),
            "ordinary_prob_min": float(p.min()),
            "ordinary_prob_rowsum_max_abs_error": float(np.max(np.abs(p.sum(axis=1) - 1.0))),
        }
        rows.append(row)
        fits.append({"start": st["start"], "result": res, "theta": res.x, "summary": row})
        print(f"BFGS {label} {st['start']}: success={res.success} nit={res.nit} ll={ll.sum():.6f} grad={row['max_absolute_gradient']:.3g}", flush=True)
    return pd.DataFrame(rows), fits


def e_step_tau(theta, y, G, X):
    gamma, _, _, _, _ = unpack(theta, G.shape[1], X.shape[1])
    pi = expit(np.asarray(G) @ gamma)
    p, _, _ = ordinal_probs_and_deriv(theta, np.asarray(G), np.asarray(X))
    tau = np.ones(len(y), dtype=float)
    zero = np.asarray(y) == 0
    denom = (1.0 - pi[zero]) + pi[zero] * p[zero, 0]
    tau[zero] = (pi[zero] * p[zero, 0]) / np.maximum(denom, EPS)
    return tau, pi, p


def fit_em_multistart(y, G, X, starts):
    rows = []
    fits = []
    traces = []
    for st in starts:
        theta = st["theta"].copy()
        prev_ll = -np.inf
        converged = False
        message = "max_iterations_reached"
        for it in range(1, 1001):
            tau, pi, p = e_step_tau(theta, y, G, X)
            new_theta, mstep_ok, mstep_msg = em_mstep(theta, y, G, X, tau)
            ll, grad = obs_loglik_score(new_theta, y, G, X)
            rel = abs(ll.sum() - prev_ll) / max(1.0, abs(prev_ll)) if np.isfinite(prev_ll) else np.inf
            max_change = float(np.max(np.abs(new_theta - theta)))
            tau_new, pi_new, _ = e_step_tau(new_theta, y, G, X)
            trace = {
                "start": st["start"],
                "iteration": it,
                "objective_loglik": float(ll.sum()),
                "relative_objective_change": rel,
                "max_parameter_change": max_change,
                "median_pi_ordinary": float(np.median(pi_new)),
                "median_posterior_ordinary_component": float(np.median(tau_new)),
                "mstep_success": bool(mstep_ok),
                "mstep_message": mstep_msg,
            }
            gamma, raw, beta, alpha, _ = unpack(new_theta, G.shape[1], X.shape[1])
            trace.update({f"gamma_{i}": v for i, v in enumerate(gamma[: min(5, len(gamma))])})
            trace.update({"alpha0": alpha[0], "alpha1": alpha[1], "alpha2": alpha[2]})
            trace.update({f"beta_{i}": v for i, v in enumerate(beta[: min(5, len(beta))])})
            traces.append(trace)
            if it == 1 or it % 25 == 0:
                print(f"  EM {st['start']} iter={it} ll={ll.sum():.6f} rel={rel:.3g} max_change={max_change:.3g}", flush=True)
            theta = new_theta
            if rel < 1e-8 and max_change < 1e-6:
                converged = True
                message = "relative_objective_change<1e-8 and max_parameter_change<1e-6"
                break
            if it >= 100 and rel < 1e-8 and max_change >= 1e-3:
                converged = False
                message = "objective plateau without parameter stabilization; stopped as non-converged diagnostic failure"
                break
            if it >= 150 and rel < 1e-5 and max_change >= 1.0:
                converged = False
                message = "large parameter drift despite small objective change; stopped as non-converged diagnostic failure"
                break
            if it >= 300 and rel < 1e-4 and max_change >= 1e-3:
                converged = False
                message = "slow EM drift without satisfying parameter stabilization; stopped as non-converged diagnostic failure"
                break
            prev_ll = ll.sum()
        ll, grad = obs_loglik_score(theta, y, G, X)
        tau, pi, p = e_step_tau(theta, y, G, X)
        gamma, raw, beta, alpha, _ = unpack(theta, G.shape[1], X.shape[1])
        row = {
            "start": st["start"],
            "converged": converged,
            "message": message,
            "iterations": it,
            "final_objective_loglik": float(ll.sum()),
            "objective_per_observation": float(ll.mean()),
            "max_absolute_gradient": float(np.max(np.abs(grad.sum(axis=0)))),
            "threshold_alpha0": alpha[0],
            "threshold_alpha1": alpha[1],
            "threshold_alpha2": alpha[2],
            "thresholds_ordered": bool(alpha[0] < alpha[1] < alpha[2]),
            "pi_min": float(pi.min()),
            "pi_median": float(np.median(pi)),
            "pi_max": float(pi.max()),
            "median_posterior_ordinary_zero": float(np.median(tau[np.asarray(y) == 0])),
        }
        rows.append(row)
        fits.append({"start": st["start"], "theta": theta, "summary": row})
        print(f"EM {st['start']}: converged={converged} it={it} ll={ll.sum():.6f} grad={row['max_absolute_gradient']:.3g}", flush=True)
    return pd.DataFrame(rows), fits, pd.DataFrame(traces)


def finite_hessian_of_score(theta, y, G, X):
    p = len(theta)
    hess_score = np.zeros((p, p))
    base_step = np.maximum(1e-5, np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(theta)))
    for j in range(p):
        step = base_step[j]
        tp = theta.copy()
        tm = theta.copy()
        tp[j] += step
        tm[j] -= step
        _, gp = obs_loglik_score(tp, y, G, X)
        _, gm = obs_loglik_score(tm, y, G, X)
        hess_score[:, j] = (gp.sum(axis=0) - gm.sum(axis=0)) / (2.0 * step)
    return hess_score


def sandwich(theta, y, G, X, clusters):
    _, score_i = obs_loglik_score(theta, y, G, X)
    score_df = pd.DataFrame(score_i)
    score_df["cluster"] = np.asarray(clusters)
    U = score_df.groupby("cluster", observed=True).sum().drop(columns=[], errors="ignore").to_numpy()
    B = U.T @ U
    hess_score = finite_hessian_of_score(theta, y, G, X)
    A = -hess_score
    invA = np.linalg.pinv(A)
    # Cluster-robust sandwich covariance: A^{-1} B A^{-T}.
    # The transpose is required even when A is theoretically a symmetric
    # likelihood Hessian because the finite-difference Jacobian is not exactly
    # symmetric in floating-point arithmetic.
    V = invA @ B @ invA.T
    eig = np.linalg.eigvalsh((V + V.T) / 2.0)
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


def param_table(theta, se, G_cols, X_cols):
    pg, px = len(G_cols), len(X_cols)
    gamma, raw, beta, alpha, _ = unpack(theta, pg, px)
    names = [f"mixing_gamma:{c}" for c in G_cols] + ["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"] + [f"ordinal_beta:{c}" for c in X_cols]
    estimates = np.r_[gamma, raw, beta]
    out = pd.DataFrame({"parameter": names, "estimate": estimates, "robust_SE": se[: len(estimates)]})
    out["z"] = out["estimate"] / out["robust_SE"]
    out["raw_p"] = 2 * stats.norm.sf(np.abs(out["z"]))
    return out


def domain_results(theta, se, G_cols, X_cols):
    rows = []
    pg = len(G_cols)
    for domain in DOMAINS:
        var = DOMAIN_VAR[domain]
        if var in G_cols:
            idx = G_cols.index(var)
            est, s = theta[idx], se[idx]
            z = est / s if s > 0 else np.nan
            p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
            rows.append({
                "Domain": domain,
                "component": "mixing",
                "parameter": var,
                "estimate": est,
                "robust_SE": s,
                "OR_ordinary": math.exp(est),
                "OR_ordinary_95CI_low": math.exp(est - 1.96 * s),
                "OR_ordinary_95CI_high": math.exp(est + 1.96 * s),
                "OR_extra_zero": math.exp(-est),
                "OR_extra_zero_95CI_low": math.exp(-(est + 1.96 * s)),
                "OR_extra_zero_95CI_high": math.exp(-(est - 1.96 * s)),
                "OR_higher_burden": np.nan,
                "raw_p": p,
                "direction_note": "pi is ordinary-component probability; exp(-gamma) is extra-zero odds ratio",
            })
        if var in X_cols:
            idx = pg + 3 + X_cols.index(var)
            est, s = theta[idx], se[idx]
            z = est / s if s > 0 else np.nan
            p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
            rows.append({
                "Domain": domain,
                "component": "ordinal",
                "parameter": var,
                "estimate": est,
                "robust_SE": s,
                "OR_ordinary": np.nan,
                "OR_ordinary_95CI_low": np.nan,
                "OR_ordinary_95CI_high": np.nan,
                "OR_extra_zero": np.nan,
                "OR_extra_zero_95CI_low": np.nan,
                "OR_extra_zero_95CI_high": np.nan,
                "OR_higher_burden": math.exp(est),
                "OR_higher_burden_95CI_low": math.exp(est - 1.96 * s),
                "OR_higher_burden_95CI_high": math.exp(est + 1.96 * s),
                "raw_p": p,
                "direction_note": "ordinary ordinal component includes zero; exp(beta) shifts toward higher 0/1/2/ge3 burden",
            })
    return pd.DataFrame(rows)


def calibration(df, theta, G, X):
    y = df["burden_ord4"].to_numpy()
    gamma, _, _, _, _ = unpack(theta, G.shape[1], X.shape[1])
    pi = expit(np.asarray(G) @ gamma)
    p, _, _ = ordinal_probs_and_deriv(theta, np.asarray(G), np.asarray(X))
    pred = pi[:, None] * p
    pred[:, 0] += 1.0 - pi
    rows = []
    for group_name, subidx in [("overall", np.ones(len(df), dtype=bool))] + [(str(w), df["time"].astype(str).eq(w).to_numpy()) for w in WEEKS]:
        for k, lab in [(0, "0"), (1, "1"), (2, "2"), (3, "ge3")]:
            idx = subidx
            rows.append({
                "group": group_name,
                "category": lab,
                "n": int(idx.sum()),
                "observed_proportion": float((y[idx] == k).mean()) if idx.sum() else np.nan,
                "predicted_marginal_proportion": float(pred[idx, k].mean()) if idx.sum() else np.nan,
                "predicted_zero_additional_component": float((1.0 - pi[idx]).mean()) if k == 0 and idx.sum() else np.nan,
                "predicted_zero_ordinary_component": float((pi[idx] * p[idx, 0]).mean()) if k == 0 and idx.sum() else np.nan,
            })
    domain_rows = []
    for domain in DOMAINS:
        var = DOMAIN_VAR[domain]
        for present in [0.0, 1.0]:
            idx = df[var].eq(present).to_numpy()
            domain_rows.append({
                "domain": domain,
                "domain_state": "present" if present == 1.0 else "absent",
                "n": int(idx.sum()),
                "observed_zero_proportion": float((y[idx] == 0).mean()) if idx.sum() else np.nan,
                "predicted_zero_probability": float(pred[idx, 0].mean()) if idx.sum() else np.nan,
                "predicted_zero_additional_component": float((1.0 - pi[idx]).mean()) if idx.sum() else np.nan,
                "predicted_zero_ordinary_component": float((pi[idx] * p[idx, 0]).mean()) if idx.sum() else np.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(domain_rows)


def ordinary_comparator(df, X, X_cols):
    y = df["burden_ord4"].to_numpy()
    beta0, raw0, start_method = fit_ordinal_start(y, X)
    theta = np.r_[np.zeros(1), raw0, beta0]
    G_dummy = np.ones((len(df), 1))
    res = optimize.minimize(
        lambda th: complete_neg_obj_grad(th, y, G_dummy, X, np.ones(len(y))),
        theta,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7},
    )
    # Build sandwich only for threshold/beta block by retaining dummy gamma then dropping it.
    sw = sandwich(res.x, y, G_dummy, X, df["participant_ID"])
    se = sw["se"]
    rows = []
    for domain in DOMAINS:
        var = DOMAIN_VAR[domain]
        if var in X_cols:
            idx = 1 + 3 + X_cols.index(var)
            est, s = res.x[idx], se[idx]
            z = est / s if s > 0 else np.nan
            rows.append({
                "Domain": domain,
                "parameter": var,
                "beta": est,
                "robust_SE": s,
                "OR_higher_burden": math.exp(est),
                "raw_p": 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan,
            })
    p, _, _ = ordinal_probs_and_deriv(res.x, G_dummy, np.asarray(X))
    props = []
    for k, lab in [(0, "0"), (1, "1"), (2, "2"), (3, "ge3")]:
        props.append({"category": lab, "observed_proportion": float((y == k).mean()), "ordinary_ordinal_predicted": float(p[:, k].mean())})
    qc = pd.DataFrame([{
        "converged": bool(res.success),
        "message": str(res.message),
        "iterations": int(res.nit),
        "objective_loglik": -float(res.fun),
        "max_absolute_gradient": float(np.max(np.abs(res.jac))),
        "A_condition": sw["A_condition"],
        "B_condition": sw["B_condition"],
        "V_condition": sw["V_condition"],
        "V_positive_semidefinite": sw["V_positive_semidefinite"],
        "start_method": start_method,
    }])
    return pd.DataFrame(rows), pd.DataFrame(props), qc


def audit_tables(base, long, analysis, mapping, zmap, actual_hash):
    source = pd.DataFrame([{
        "source_file": str(SOURCE),
        "sha256": actual_hash,
        "expected_sha256": EXPECTED_SHA256,
        "hash_match": actual_hash.lower() == EXPECTED_SHA256,
        "baseline_rows": len(base),
        "baseline_unique_participants": base["participant_ID"].nunique(),
        "baseline_sites": base["site"].nunique(),
    }])
    outcome_rows = []
    for week, sub in long.groupby("time", observed=False):
        valid = sub[sub["burden_ord4"].isin([0, 1, 2, 3])]
        row = {"week": WEEK_LABELS.get(str(week), str(week)), "available_FIBSER_N": len(valid)}
        for k, lab in [(0, "0"), (1, "1"), (2, "2"), (3, "ge3")]:
            row[f"burden_{lab}_n"] = int(valid["burden_ord4"].eq(k).sum())
            row[f"burden_{lab}_pct"] = float(valid["burden_ord4"].eq(k).mean() * 100) if len(valid) else np.nan
        outcome_rows.append(row)
    sample = pd.DataFrame([
        {"sample": "baseline", "participants": base["participant_ID"].nunique(), "person_visits": np.nan, "sites": base["site"].nunique()},
        {"sample": "long_valid_FIBSER", "participants": long.loc[long["burden_ord4"].isin([0, 1, 2, 3]), "participant_ID"].nunique(), "person_visits": int(long["burden_ord4"].isin([0, 1, 2, 3]).sum()), "sites": long["site"].nunique()},
        {"sample": "step2B1_complete_case", "participants": analysis["participant_ID"].nunique(), "person_visits": len(analysis), "sites": analysis["site"].nunique()},
    ])
    sites = analysis.groupby("site", observed=True).agg(
        participants=("participant_ID", "nunique"),
        visits=("participant_ID", "size"),
        zero_n=("burden_ord4", lambda x: int((x == 0).sum())),
        nonzero_n=("burden_ord4", lambda x: int((x > 0).sum())),
    ).reset_index()
    return source, mapping, pd.DataFrame(outcome_rows), sample, sites, zmap


def literature_boundary_md():
    return """# Step 2B.1 Model Definition and Literature Boundary

## DIRECTLY FROM YANG & SIMPSON 2010

- Left-inflated mixture structure.
- An additional point mass at the lower boundary.
- A general ordinary distribution `F` that is itself capable of generating the boundary value.
- Working-independence marginal/GEE analysis for correlated data.
- Participant/cluster sandwich covariance concept.
- Quasi-Newton and EM computational routes.

## PROJECT-SPECIFIC ORDINAL SPECIALIZATION

- The ordinary distribution `F` is chosen here as a proportional-odds ordinal distribution over `0 / 1 / 2 / >=3`.
- The ordinary component uses cumulative logits `logit[P_F(Y <= k | X)] = alpha_k - X beta`, `k=0,1,2`.
- Ordered thresholds are enforced by `alpha_0=a0`, `alpha_1=a0+exp(d1)`, `alpha_2=a0+exp(d1)+exp(d2)`.
- Category probabilities are computed directly from the four-category ordinal probability vector.

This is an ordinal specialization of the Yang-Simpson LIM-GEE framework. It is not claimed that Yang & Simpson proposed or validated a proportional-odds ordinal LIM-GEE estimator.

## Direction of the Mixing Model

`logit(pi_it)=G_it gamma`, where `pi_it` is the probability of belonging to the ordinary ordinal component. Therefore:

- `OR_ordinary = exp(gamma)`.
- `OR_extra_zero = exp(-gamma)`.

The ordinary ordinal component includes zero. This is not a hurdle model.
"""


def decide_status(bfgs_summary, em_summary, comparison, swdiag):
    conv_rate = bfgs_summary["converged"].mean()
    stable_bfgs = conv_rate >= 0.8 and bfgs_summary.loc[bfgs_summary["converged"], "objective_per_observation"].max() - bfgs_summary.loc[bfgs_summary["converged"], "objective_per_observation"].min() < 1e-6
    em_stable = em_summary["converged"].mean() >= 0.8 and em_summary.loc[em_summary["converged"], "objective_per_observation"].max() - em_summary.loc[em_summary["converged"], "objective_per_observation"].min() < 1e-6
    cross = bool(comparison["same_objective_tolerance_1e-6_per_obs"].iloc[0]) if len(comparison) else False
    sw_ok = bool(swdiag["V_finite"].iloc[0]) and bool(swdiag["V_positive_semidefinite"].iloc[0]) and np.isfinite(swdiag["A_condition"].iloc[0])
    if stable_bfgs and em_stable and cross and sw_ok:
        return "A. YANG-SIMPSON LIM ORDINAL SPECIALIZATION NUMERICALLY STABLE ACROSS ALGORITHMS AND STARTS"
    if stable_bfgs and sw_ok:
        return "B. STABLE UNDER ONE ALGORITHM BUT NOT CROSS-ALGORITHM VALIDATED"
    if conv_rate > 0 or em_summary["converged"].mean() > 0:
        return "C. PARTIALLY STABLE BUT SENSITIVE TO STARTING VALUES / SPECIFICATION"
    return "D. NUMERICALLY / PRACTICALLY NON-IDENTIFIABLE"


def make_report(actual_hash, sample, bfgs_summary, em_summary, comparison, swdiag, cal, diag, status):
    return f"""# Step2B.1 Yang-Simpson LIM Ordinal GEE Report

## 1. Hash/sample QC

Source: `{SOURCE}`

SHA256 recalculated: `{actual_hash}`; matched prespecified frozen value: **YES**.

{sample.to_markdown(index=False)}

## 2. Exact LIM ordinal probability structure

`pi_it` is the ordinary-component probability. For `L=0`:

- `P(Y=0)=(1-pi_it)+pi_it*P_F(Y=0)`
- `P(Y=k)=pi_it*P_F(Y=k)` for `k>0`

`F` is the project-specific proportional-odds ordinary ordinal component over `0/1/2/>=3`.

## 3. Ordinary component includes zero?

**YES.** The ordinary component includes all four categories, including `0`. This is not a hurdle model.

## 4. BFGS

Starts run: {len(bfgs_summary)}. Converged: {int(bfgs_summary['converged'].sum())}/{len(bfgs_summary)}.

{bfgs_summary[['start','converged','iterations','final_objective_loglik','objective_per_observation','max_absolute_gradient','pi_median','thresholds_ordered']].to_markdown(index=False)}

## 5. EM

Starts run: {len(em_summary)}. Converged: {int(em_summary['converged'].sum())}/{len(em_summary)}.

{em_summary[['start','converged','iterations','final_objective_loglik','objective_per_observation','max_absolute_gradient','pi_median','thresholds_ordered']].to_markdown(index=False)}

## 6. Do BFGS and EM converge to the same solution?

{comparison.to_markdown(index=False)}

## 7. Gradient / parameter / threshold diagnostics

Threshold ordering was enforced by exponential threshold increments. Final maximum absolute BFGS score and EM score are shown above. Machine-level epsilon (`np.finfo(float).tiny`) was used only to avoid `log(0)` and division by zero in diagnostic calculations.

## 8. Sandwich covariance valid/stable?

{swdiag.to_markdown(index=False)}

## 9. Site instability diagnostics

{diag.to_markdown(index=False)}

## 10. Extra-zero component numerically identifiable?

This is judged from multi-start agreement, cross-algorithm agreement, pi distribution, posterior ordinary-zero allocation, and sandwich conditioning. See `03_BFGS_multistart_results.xlsx`, `05_EM_multistart_results.xlsx`, `07_BFGS_vs_EM_solution_comparison.xlsx`, and `08_sandwich_covariance_diagnostics.xlsx`.

## 11. Observed vs fitted zero/category calibration

{cal.to_markdown(index=False)}

## 12. 9-domain estimates

Domain estimates were exported only if the predefined stability gate reached status A or B; otherwise `10_domain_preliminary_results.xlsx` records that they should not be interpreted.

## 13. Difference from ordinary ordinal comparator

The comparator is an ordinary proportional-odds ordinal marginal model over the same `0/1/2/>=3` outcome. It has no additional boundary point mass, so it is used only to compare direction, magnitude, SEs, and calibration, not for likelihood-ratio testing against the LIM working criterion.

## 14. Remaining methodological limitations

- The Yang-Simpson general LIM-GEE framework is literature-supported; the proportional-odds ordinal ordinary component is project-specific.
- Working independence is a marginal criterion, not a full longitudinal joint likelihood.
- Numerical Hessian for the sandwich bread used central differences of analytic total scores with parameter-adaptive steps.
- Identifiability remains empirical/practical here because both mixture and ordinary components use rich overlapping covariates.

Final status:

**{status}**
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Step2B.1: checking source hash", flush=True)
    actual_hash = sha256(SOURCE)
    if actual_hash.lower() != EXPECTED_SHA256:
        raise SystemExit(f"STOP: hash mismatch. Actual SHA256={actual_hash}")

    print("Step2B.1: rebuilding analysis data from frozen workbook", flush=True)
    base, long, analysis, mapping, zmap = build_dataset()
    source_audit, variable_audit, outcome_audit, sample_audit, site_audit, zmap = audit_tables(base, long, analysis, mapping, zmap, actual_hash)
    write_book(OUT / "01_data_and_outcome_audit.xlsx", {
        "source_hash": source_audit,
        "variable_mapping": variable_audit,
        "z_standardization": zmap,
        "analysis_sample": sample_audit,
        "outcome_by_week": outcome_audit,
        "site_counts": site_audit,
    })
    (OUT / "02_model_definition_and_literature_boundary.md").write_text(literature_boundary_md(), encoding="utf-8")

    print(f"Step2B.1: complete-case analysis rows={len(analysis)}, participants={analysis['participant_ID'].nunique()}, sites={analysis['site'].nunique()}", flush=True)
    G, X = make_design(analysis)
    y = analysis["burden_ord4"].to_numpy()
    rng = np.random.default_rng(20260927)
    print(f"Step2B.1: design built, mixing columns={G.shape[1]}, ordinal columns={X.shape[1]}", flush=True)
    print("Step2B.1: constructing multistart values", flush=True)
    starts = build_starts(y, G, X, rng)

    names = [f"mixing_gamma:{c}" for c in G.columns] + ["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"] + [f"ordinal_beta:{c}" for c in X.columns]
    bfgs_path = OUT / "03_BFGS_multistart_results.xlsx"
    sol_path = OUT / "04_BFGS_parameter_solutions.xlsx"
    if bfgs_path.exists() and sol_path.exists():
        print("Step2B.1: reusing completed BFGS multistart artifacts", flush=True)
        bfgs_summary = pd.read_excel(bfgs_path, sheet_name="BFGS_multistart")
        sol_df = pd.read_excel(sol_path, sheet_name="parameter_solutions")
        bfgs_fits = []
        for st_name, sub in sol_df.groupby("start", sort=False):
            theta = sub.set_index("parameter").reindex(names)["estimate"].to_numpy(dtype=float)
            row = bfgs_summary.loc[bfgs_summary["start"].eq(st_name)].iloc[0].to_dict()
            bfgs_fits.append({"start": st_name, "theta": theta, "summary": row})
    else:
        print("Step2B.1: running primary BFGS multistart", flush=True)
        bfgs_summary, bfgs_fits = fit_bfgs_multistart(y, G, X, starts, "primary_full")
        solution_rows = []
        for fit in bfgs_fits:
            for name, val in zip(names, fit["theta"]):
                solution_rows.append({"start": fit["start"], "parameter": name, "estimate": val})
        write_book(bfgs_path, {"BFGS_multistart": bfgs_summary})
        write_book(sol_path, {"parameter_solutions": pd.DataFrame(solution_rows)})
    best_fit = max(bfgs_fits, key=lambda z: z["summary"]["final_objective_loglik"])
    best_theta = best_fit["theta"]

    em_summary, em_fits, em_traces = fit_em_multistart(y, G, X, starts)
    write_book(OUT / "05_EM_multistart_results.xlsx", {"EM_multistart": em_summary})
    write_book(OUT / "06_EM_iteration_traces.xlsx", {"EM_traces": em_traces})

    best_em = max(em_fits, key=lambda z: z["summary"]["final_objective_loglik"])
    ll_b, _ = obs_loglik_score(best_theta, y, G, X)
    ll_e, _ = obs_loglik_score(best_em["theta"], y, G, X)
    rel_param = np.abs(best_theta - best_em["theta"]) / np.maximum(1.0, np.abs(best_theta))
    comparison = pd.DataFrame([{
        "best_BFGS_start": best_fit["start"],
        "best_EM_start": best_em["start"],
        "BFGS_objective": float(ll_b.sum()),
        "EM_objective": float(ll_e.sum()),
        "absolute_objective_difference": float(abs(ll_b.sum() - ll_e.sum())),
        "objective_difference_per_observation": float(abs(ll_b.sum() - ll_e.sum()) / len(y)),
        "same_objective_tolerance_1e-6_per_obs": bool(abs(ll_b.sum() - ll_e.sum()) / len(y) < 1e-6),
        "max_relative_parameter_difference": float(rel_param.max()),
        "median_relative_parameter_difference": float(np.median(rel_param)),
    }])
    param_comp = pd.DataFrame({"parameter": names, "BFGS": best_theta, "EM": best_em["theta"], "absolute_difference": np.abs(best_theta - best_em["theta"]), "relative_difference": rel_param})
    write_book(OUT / "07_BFGS_vs_EM_solution_comparison.xlsx", {"summary": comparison, "parameter_comparison": param_comp})

    sw = sandwich(best_theta, y, G, X, analysis["participant_ID"])
    swdiag = pd.DataFrame([{
        "cluster_variable": "participant_ID",
        "clusters": analysis["participant_ID"].nunique(),
        "parameters": len(best_theta),
        "A_condition": sw["A_condition"],
        "B_condition": sw["B_condition"],
        "V_condition": sw["V_condition"],
        "V_min_eigenvalue": sw["V_min_eigenvalue"],
        "V_positive_semidefinite": sw["V_positive_semidefinite"],
        "V_finite": sw["finite"],
        "hessian_method": "central finite difference of analytic total score",
        "step_strategy": "max(1e-5, sqrt(machine_epsilon)*max(1, abs(parameter)))",
    }])
    param = param_table(best_theta, sw["se"], list(G.columns), list(X.columns))
    write_book(OUT / "08_sandwich_covariance_diagnostics.xlsx", {"diagnostics": swdiag, "all_parameters": param})

    cal, domain_cal = calibration(analysis, best_theta, G, X)
    write_book(OUT / "09_zero_and_category_calibration.xlsx", {"overall_and_time": cal, "domain_zero": domain_cal})

    status_probe = decide_status(bfgs_summary, em_summary, comparison, swdiag)
    domains = domain_results(best_theta, sw["se"], list(G.columns), list(X.columns))
    if not status_probe.startswith(("A.", "B.")):
        domains["interpretation_gate"] = "DO NOT INTERPRET: predefined stability criteria not met"
    else:
        domains["interpretation_gate"] = "preliminary only; no clinical conclusion"
    write_book(OUT / "10_domain_preliminary_results.xlsx", {"domain_results": domains})

    ordinary_domains, ordinary_props, ordinary_qc = ordinary_comparator(analysis, np.asarray(X), list(X.columns))
    write_book(OUT / "11_ordinary_ordinal_comparator.xlsx", {"domain_coefficients": ordinary_domains, "calibration": ordinary_props, "qc": ordinary_qc})

    diag_rows = []
    for label, mix_rhs, ord_rhs in [
        ("D1_no_site_both_components", MIX_RHS_NO_SITE, ORD_RHS_NO_SITE),
        ("D2_site_retained_simplified_mixing", MIX_RHS_SIMPLE, ORD_RHS_FULL),
    ]:
        Gd, Xd = make_design(analysis, mix_rhs=mix_rhs, ord_rhs=ord_rhs)
        starts_d = build_starts(y, Gd, Xd, rng)[:5]
        summ_d, _ = fit_bfgs_multistart(y, Gd, Xd, starts_d, label)
        diag_rows.append({
            "diagnostic_model": label,
            "starts": len(summ_d),
            "converged": int(summ_d["converged"].sum()),
            "best_objective_loglik": float(summ_d["final_objective_loglik"].max()),
            "best_objective_per_observation": float(summ_d["objective_per_observation"].max()),
            "max_gradient_best": float(summ_d.sort_values("final_objective_loglik", ascending=False).iloc[0]["max_absolute_gradient"]),
            "note": "Diagnostic only; primary specification not replaced.",
        })
    diag = pd.DataFrame(diag_rows)
    write_book(OUT / "12_diagnostic_models_if_needed.xlsx", {"diagnostic_models": diag})

    status = decide_status(bfgs_summary, em_summary, comparison, swdiag)
    report = make_report(actual_hash, sample_audit, bfgs_summary, em_summary, comparison, swdiag, cal, diag, status)
    (OUT / "Step2B1_YangSimpson_LIM_ordinal_GEE_report.md").write_text(report, encoding="utf-8")
    print(f"Completed Step2B.1 outputs: {OUT}")
    print(status)


if __name__ == "__main__":
    main()


