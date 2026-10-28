from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable


from public_paths import ROOT, authorized_data_path
SOURCE = authorized_data_path("PROUD_RAW_WORKBOOK", "raw_workbook.xlsx")
OUT = Path(os.environ.get(
    "BINARY_GEE_OUTPUT_DIR",
    str(ROOT / "outputs" / "binary_logistic_GEE_sensitivity_S4_S5_2026-10-18"),
))
OUT.mkdir(parents=True, exist_ok=False)

FREEZE_ID = "PRISE_REPAIRED_NATURAL_CC_1468P_4799V_2026-10-08"
EXPECTED_SOURCE_SHA256 = "876b1e3f1b55574984bd105336c1e706d4fad7eaaa7626e62318732f69324f84"
EXPECTED_PARTICIPANTS = 1468
EXPECTED_VISITS = 4799
EXPECTED_WEEK_COUNTS = {"Week2": 1381, "Week8": 1259, "Week12": 1146, "Week24": 1013}
EXPECTED_BURDEN4 = {0: 3464, 1: 584, 2: 512, 3: 239}
EXPECTED_ANY_EVENTS = 1335
EXPECTED_MODERATE_EVENTS = 239
EXPECTED_BASELINE_N = 1543
EXPECTED_SITES = 21

WEEKS = ["Week2", "Week8", "Week12", "Week24"]
FIBSER_BURDEN_CODE = "JH10_00_080_13_4038_1"
DOMAINS = [
    ("Sleep", "Sleep"),
    ("Gastrointestinal", "Gastrointestinal"),
    ("Other symptom", "OtherPsych"),
    ("Nervous system", "Nervous"),
    ("Heart", "Cardiac"),
    ("Eye/Ear", "EyeEar"),
    ("Skin", "Skin"),
    ("Sexual functioning", "Sexual"),
    ("Genital/Urinary", "Genitourinary"),
]
OCC_CODES = {
    "Sleep": "JH05_10_012_510_4040_1",
    "Gastrointestinal": "JH03_01_000708_4040_1",
    "Other symptom": "JH03_01_000749_4040_1",
    "Nervous system": "JH03_01_000745_4040_1",
    "Heart": "JH03_01_000743_4040_1",
    "Eye/Ear": "JH03_01_000746_4040_1",
    "Skin": "JH03_01_000744_4040_1",
    "Sexual functioning": "JH03_01_000748_4040_1",
    "Genital/Urinary": "JH03_01_000747_4040_1",
}
LABELS = {
    "Sleep": ["睡眠"],
    "Gastrointestinal": ["胃肠"],
    "Other symptom": ["其他精神症状"],
    "Nervous system": ["神经系统"],
    "Heart": ["心脏"],
    "Eye/Ear": ["眼/耳", "眼耳"],
    "Skin": ["皮肤"],
    "Sexual functioning": ["性功能"],
    "Genital/Urinary": ["生殖/泌尿", "生殖泌尿"],
}
EDUCATION_MAP = {
    "小学": "Junior high or below",
    "初中": "Junior high or below",
    "高中/中专/职高": "High school/vocational",
    "大学/大专": "College or above",
    "研究生及以上": "College or above",
}
EDUCATION_LEVELS = ["Junior high or below", "High school/vocational", "College or above"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def numeric(s: pd.Series) -> pd.Series:
    raw = s.map(clean)
    direct = pd.to_numeric(raw, errors="coerce")
    extracted = pd.to_numeric(raw.astype("string").str.extract(r"^\s*\[?(-?\d+(?:\.\d+)?)\]?.*$")[0], errors="coerce")
    return direct.where(direct.notna(), extracted)


def sanitize_category(x):
    x = clean(x)
    return np.nan if pd.isna(x) else str(x).strip()


def id_col(df: pd.DataFrame):
    for col in df.columns:
        s = str(col)
        if s.startswith("ID") or "入组编码" in s:
            return col
    return df.columns[0]


def code_col(df: pd.DataFrame, code: str):
    return next((col for col in df.columns if code and code in str(col)), None)


def burden_col(df: pd.DataFrame):
    hit = code_col(df, FIBSER_BURDEN_CODE)
    if hit is not None:
        return hit, "code"
    anchor = next((i for i, col in enumerate(df.columns) if any(t in str(col) for t in ("FISER", "FIBSER", "GRSEB"))), None)
    if anchor is None or anchor + 3 >= len(df.columns):
        raise KeyError("FIBSER burden column not found")
    return df.columns[anchor + 3], "validated_block_offset"


def occurrence_col(df: pd.DataFrame, domain: str):
    hit = code_col(df, OCC_CODES[domain])
    if hit is not None:
        return hit, "code"
    anchor = next((i for i, col in enumerate(df.columns) if "PRISE" in str(col)), None)
    if anchor is None:
        raise KeyError(f"PRISE anchor not found for {domain}")
    candidates = []
    for col in list(df.columns)[anchor + 1 :]:
        text = str(col).replace(" ", "")
        if any(label.replace(" ", "") in text for label in LABELS[domain]):
            candidates.append(col)
    if len(candidates) == 1:
        return candidates[0], "unique_visible_label"
    occurrence_like = [c for c in candidates if "1=是" in str(c) or "1=有" in str(c)]
    if len(occurrence_like) == 1:
        return occurrence_like[0], "visible_label_occurrence_item"
    raise KeyError(f"Could not uniquely locate {domain}; candidates={[str(c) for c in candidates]}")


def concurrent_hamd_col(df: pd.DataFrame):
    candidates = [c for c in df.columns if str(c) == "HAMD总分" or str(c).startswith("HAMD总分.")]
    if candidates:
        return candidates[0], "exact_header"
    cols = list(df.columns)
    hamd_anchor = next((i for i, c in enumerate(cols) if "HAMD-17" in str(c)), None)
    hama_anchor = next((i for i, c in enumerate(cols) if "HAMA" in str(c) and "量表" in str(c)), None)
    if hamd_anchor is not None and hama_anchor is not None:
        totals = [c for c in cols[hamd_anchor:hama_anchor] if "总分" in str(c)]
        if totals:
            return totals[-1], "between_HAMD_and_HAMA_anchors"
    hit = next((c for c in df.columns if "HAMD" in str(c) and "总分" in str(c)), None)
    return hit, "generic_HAMD_total" if hit is not None else "not_found"


def first_nonmissing(x: pd.Series):
    y = x.dropna()
    return np.nan if y.empty else y.iloc[0]


def read_baseline(xls: pd.ExcelFile) -> pd.DataFrame:
    df = pd.read_excel(SOURCE, sheet_name=xls.sheet_names[0], engine="openpyxl")
    base = pd.DataFrame({"participant_ID": df[id_col(df)].map(norm_id)})
    base["age"] = pd.to_numeric(df["年龄"], errors="coerce")
    base["site"] = df["分中心"].map(sanitize_category)
    base["baseline_med_class"] = df["药物类别"].map(sanitize_category)
    base["sex"] = pd.to_numeric(df["性别：1=男，2=女"], errors="coerce").map({1.0: "Male", 2.0: "Female"})
    base["education"] = df["患者文化程度："].map(sanitize_category)
    base["first_episode"] = pd.to_numeric(df["本次抑郁是否首次发作：1=首发，2=复发"], errors="coerce").map({1.0: "FirstEpisode", 2.0: "RecurrentEpisode"})
    base["baseline_HAMD17"] = pd.to_numeric(df["HAMD总分"], errors="coerce")
    sds_cols = [
        "席汉残疾量表（SDS） 工作/学习得分",
        "席汉残疾量表（SDS） 社交生活得分",
        "席汉残疾量表（SDS） 家庭生活/家庭责任得分",
    ]
    base["baseline_SDS"] = df[sds_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=3)
    if len(base) != EXPECTED_BASELINE_N or base["participant_ID"].nunique() != EXPECTED_BASELINE_N:
        raise RuntimeError(f"STOP: baseline cohort mismatch {len(base)}/{base['participant_ID'].nunique()}")
    if base["site"].nunique() != EXPECTED_SITES:
        raise RuntimeError(f"STOP: site-count mismatch {base['site'].nunique()}")
    if base["participant_ID"].duplicated().any():
        raise RuntimeError("STOP: duplicate participant IDs in baseline sheet")
    return base


def read_followup(xls: pd.ExcelFile):
    positions = {"Week2": (1, 5), "Week8": (2, 6), "Week12": (3, 7), "Week24": (4, 8)}
    frames, mapping_rows, sheet_rows = [], [], []
    for visit_order, (week, poss) in enumerate(positions.items(), start=1):
        for source_priority, pos in enumerate(poss, start=1):
            sheet = xls.sheet_names[pos]
            df = pd.read_excel(SOURCE, sheet_name=sheet, engine="openpyxl")
            bcol, bmethod = burden_col(df)
            hcol, hmethod = concurrent_hamd_col(df)
            ids = df[id_col(df)].map(norm_id)
            sheet_rows.append({
                "week": week, "source_sheet": sheet, "rows": len(df), "nonmissing_ID": int(ids.notna().sum()),
                "duplicate_ID_rows": int(ids[ids.notna()].duplicated().sum()), "burden_column": str(bcol),
                "burden_match_method": bmethod, "concurrent_HAMD17_column": str(hcol), "HAMD_match_method": hmethod,
            })
            out = pd.DataFrame({
                "participant_ID": ids, "time": week, "visit_order": visit_order,
                "source_sheet": sheet, "source_priority": source_priority,
            })
            out["fibser_burden"] = numeric(df[bcol])
            out["concurrent_HAMD17"] = pd.to_numeric(df[hcol], errors="coerce") if hcol is not None else np.nan
            for final_domain, var in DOMAINS:
                col, method = occurrence_col(df, final_domain)
                raw = numeric(df[col])
                invalid = sorted(set(raw.dropna().unique()) - {0.0, 1.0, 2.0})
                label_match = any(label.replace(" ", "") in str(col).replace(" ", "") for label in LABELS[final_domain])
                if invalid or not label_match:
                    raise RuntimeError(f"STOP: mapping/value gate failed {sheet}/{final_domain}: invalid={invalid}, column={col}")
                out[var] = raw.map({1.0: 1.0, 2.0: 0.0, 0.0: 0.0})
                mapping_rows.append({
                    "week": week, "source_sheet": sheet, "final_domain": final_domain,
                    "analysis_variable": var, "source_column": str(col), "match_method": method,
                    "label_verified": True, "raw_1_present": int((raw == 1).sum()),
                    "raw_2_absent": int((raw == 2).sum()), "raw_0_absent": int((raw == 0).sum()),
                    "raw_missing": int(raw.isna().sum()), "status": "PASS",
                })
            frames.append(out)
    raw = pd.concat(frames, ignore_index=True)
    nonkey = ["fibser_burden", "concurrent_HAMD17"] + [v for _, v in DOMAINS]
    conflict_rows = []
    for (pid, week), g in raw.groupby(["participant_ID", "time"], dropna=False):
        for col in nonkey:
            vals = sorted(pd.unique(g[col].dropna()).tolist())
            if len(vals) > 1:
                conflict_rows.append({"participant_ID": pid, "time": week, "variable": col, "values": "; ".join(map(str, vals))})
    raw = raw.sort_values(["participant_ID", "visit_order", "source_priority"], kind="stable")
    collapsed = raw.groupby(["participant_ID", "time"], dropna=False)[["visit_order"] + nonkey].agg(first_nonmissing).reset_index()
    return collapsed, pd.DataFrame(mapping_rows), pd.DataFrame(sheet_rows), pd.DataFrame(conflict_rows)


def rebuild_analysis():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    source_hash = sha256(SOURCE)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"STOP: raw workbook SHA-256 changed: {source_hash}")
    xls = pd.ExcelFile(SOURCE, engine="openpyxl")
    if len(xls.sheet_names) < 9:
        raise RuntimeError(f"STOP: expected at least 9 worksheets; observed {len(xls.sheet_names)}")
    base = read_baseline(xls)
    follow, mapping, sheet_audit, conflicts = read_followup(xls)
    if len(mapping) != 72 or not mapping["status"].eq("PASS").all():
        raise RuntimeError("STOP: nine-domain mapping audit did not produce 72 PASS rows")
    long = follow.merge(base, on="participant_ID", how="left", validate="many_to_one", indicator=True)
    if not long["_merge"].eq("both").all():
        raise RuntimeError("STOP: at least one follow-up ID failed baseline matching")
    long = long.drop(columns="_merge")
    valid_burden = long["fibser_burden"].isin(range(7))
    long["burden_ord4"] = np.nan
    long.loc[valid_burden & long["fibser_burden"].eq(0), "burden_ord4"] = 0
    long.loc[valid_burden & long["fibser_burden"].eq(1), "burden_ord4"] = 1
    long.loc[valid_burden & long["fibser_burden"].eq(2), "burden_ord4"] = 2
    long.loc[valid_burden & long["fibser_burden"].ge(3), "burden_ord4"] = 3
    long["time"] = pd.Categorical(long["time"], categories=WEEKS, ordered=True)
    z_rows = []
    for col in ["age", "baseline_HAMD17", "baseline_SDS", "concurrent_HAMD17"]:
        mean = float(long[col].mean())
        sd = float(long[col].std(ddof=0))
        long[col + "_z"] = (long[col] - mean) / sd
        z_rows.append({"variable": col, "analysis_variable": col + "_z", "mean": mean, "sd_ddof0": sd, "reference_rows": int(long[col].notna().sum())})
    required = [
        "participant_ID", "time", "visit_order", "burden_ord4", "age_z", "sex", "education",
        "first_episode", "baseline_med_class", "baseline_HAMD17_z", "baseline_SDS_z", "concurrent_HAMD17_z",
    ] + [v for _, v in DOMAINS]
    analysis = long.dropna(subset=required).copy()
    analysis["burden_ord4"] = analysis["burden_ord4"].astype(int)
    analysis["education3"] = analysis["education"].map(EDUCATION_MAP)
    analysis["education3"] = pd.Categorical(analysis["education3"], categories=EDUCATION_LEVELS)
    analysis["any_burden"] = (analysis["fibser_burden"] >= 1).astype(int)
    analysis["moderate_burden"] = (analysis["fibser_burden"] >= 3).astype(int)
    return analysis, long, mapping, sheet_audit, conflicts, pd.DataFrame(z_rows), xls.sheet_names, source_hash


def stop_gate(analysis: pd.DataFrame):
    observed_week = analysis["time"].astype(str).value_counts().to_dict()
    observed_burden = analysis["burden_ord4"].value_counts().sort_index().to_dict()
    checks = [
        ("participants", int(analysis["participant_ID"].nunique()), EXPECTED_PARTICIPANTS),
        ("person-visits", int(len(analysis)), EXPECTED_VISITS),
        ("duplicate participant-time keys", int(analysis.duplicated(["participant_ID", "time"]).sum()), 0),
        ("Week 2 visits", int(observed_week.get("Week2", 0)), EXPECTED_WEEK_COUNTS["Week2"]),
        ("Week 8 visits", int(observed_week.get("Week8", 0)), EXPECTED_WEEK_COUNTS["Week8"]),
        ("Week 12 visits", int(observed_week.get("Week12", 0)), EXPECTED_WEEK_COUNTS["Week12"]),
        ("Week 24 visits", int(observed_week.get("Week24", 0)), EXPECTED_WEEK_COUNTS["Week24"]),
        ("FIBSER burden 0", int(observed_burden.get(0, 0)), EXPECTED_BURDEN4[0]),
        ("FIBSER burden 1", int(observed_burden.get(1, 0)), EXPECTED_BURDEN4[1]),
        ("FIBSER burden 2", int(observed_burden.get(2, 0)), EXPECTED_BURDEN4[2]),
        ("FIBSER burden 3-6", int(observed_burden.get(3, 0)), EXPECTED_BURDEN4[3]),
        ("Any-burden events", int(analysis["any_burden"].sum()), EXPECTED_ANY_EVENTS),
        ("Moderate-or-greater events", int(analysis["moderate_burden"].sum()), EXPECTED_MODERATE_EVENTS),
        ("missing education3", int(analysis["education3"].isna().sum()), 0),
    ]
    table = pd.DataFrame(checks, columns=["check", "observed", "expected"])
    table["status"] = np.where(table["observed"].eq(table["expected"]), "PASS", "FAIL")
    if not table["status"].eq("PASS").all():
        table.to_csv(OUT / "STOP_Gate_Failures.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError("STOP: frozen-sample/event-count gate failed; models were not fit")
    return table


def model_formula(outcome: str, domain_var: str) -> str:
    return (
        f"{outcome} ~ {domain_var} + C(time, Treatment(reference='Week2')) + age_z + "
        "C(sex, Treatment(reference='Female')) + "
        "C(education3, Treatment(reference='College or above')) + "
        "C(first_episode, Treatment(reference='FirstEpisode')) + baseline_HAMD17_z + baseline_SDS_z + "
        "C(baseline_med_class, Treatment(reference='SNRI')) + concurrent_HAMD17_z"
    )


def fit_one(analysis: pd.DataFrame, outcome: str, outcome_label: str, final_domain: str, domain_var: str):
    formula = model_formula(outcome, domain_var)
    ydf, xdf = patsy.dmatrices(formula, analysis, return_type="dataframe")
    if len(xdf) != len(analysis) or ydf.isna().any().any() or xdf.isna().any().any():
        raise RuntimeError(f"STOP: model-matrix row/missing mismatch for {outcome_label}/{final_domain}")
    if domain_var not in xdf.columns:
        raise RuntimeError(f"STOP: domain coefficient missing from matrix for {outcome_label}/{final_domain}")
    rank = int(np.linalg.matrix_rank(np.asarray(xdf, dtype=float)))
    if rank != xdf.shape[1]:
        raise RuntimeError(f"STOP: rank-deficient model matrix for {outcome_label}/{final_domain}: {rank}/{xdf.shape[1]}")
    model = sm.GEE.from_formula(
        formula, groups="participant_ID", data=analysis, family=sm.families.Binomial(), cov_struct=Exchangeable()
    )
    caught = []
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        result = model.fit(maxiter=200, ctol=1e-8, cov_type="robust")
        caught = [str(w.message) for w in ws]
    beta = float(result.params[domain_var])
    se = float(result.bse[domain_var])
    p = float(result.pvalues[domain_var])
    zcrit = stats.norm.ppf(0.975)
    low, high = beta - zcrit * se, beta + zcrit * se
    cov = np.asarray(result.cov_params(), dtype=float)
    cov_sym = bool(np.allclose(cov, cov.T, rtol=1e-8, atol=1e-10))
    eig_min = float(np.linalg.eigvalsh((cov + cov.T) / 2).min()) if np.isfinite(cov).all() else np.nan
    finite = bool(np.isfinite([beta, se, p, low, high]).all() and np.isfinite(cov).all())
    ctab = pd.crosstab(analysis[domain_var].astype(int), analysis[outcome].astype(int)).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    c00, c01, c10, c11 = (int(ctab.loc[0, 0]), int(ctab.loc[0, 1]), int(ctab.loc[1, 0]), int(ctab.loc[1, 1]))
    zero_cell = any(v == 0 for v in [c00, c01, c10, c11])
    converged = bool(getattr(result, "converged", False))
    covariance_valid = bool(finite and cov_sym and eig_min >= -1e-8 and se > 0)
    flags = []
    if not converged:
        flags.append("model did not converge")
    if not covariance_valid:
        flags.append("robust covariance invalid or non-finite")
    if zero_cell:
        flags.append("zero cell in outcome-by-domain table; possible separation")
    if se > 2:
        flags.append("large robust SE (>2)")
    if abs(beta) > math.log(20):
        flags.append("extreme adjusted OR (<0.05 or >20)")
    if caught:
        flags.append("fit warning(s): " + " | ".join(caught))
    stable = bool(converged and covariance_valid and not zero_cell and se <= 2 and abs(beta) <= math.log(20))
    result_row = {
        "Domain": final_domain, "Adjusted OR": float(np.exp(beta)), "CI low": float(np.exp(low)),
        "CI high": float(np.exp(high)), "P value": p, "Participants": int(analysis["participant_ID"].nunique()),
        "Person-visits": int(len(analysis)), "Events": int(analysis[outcome].sum()), "Stable": stable,
    }
    diag_row = {
        "Outcome": outcome_label, "Domain": final_domain, "Analysis variable": domain_var,
        "Converged": converged, "Stable": stable, "Participants": int(analysis["participant_ID"].nunique()),
        "Person-visits": int(len(analysis)), "Events": int(analysis[outcome].sum()), "Non-events": int((1-analysis[outcome]).sum()),
        "Domain absent/outcome 0": c00, "Domain absent/outcome 1": c01,
        "Domain present/outcome 0": c10, "Domain present/outcome 1": c11,
        "Zero cell": zero_cell, "Coefficient": beta, "Robust SE": se, "Adjusted OR": float(np.exp(beta)),
        "CI low": float(np.exp(low)), "CI high": float(np.exp(high)), "P value": p,
        "Working correlation": float(np.asarray(result.cov_struct.dep_params).reshape(-1)[0]),
        "Covariance finite": bool(np.isfinite(cov).all()), "Covariance symmetric": cov_sym,
        "Covariance minimum eigenvalue": eig_min, "Design rows": int(xdf.shape[0]),
        "Design columns": int(xdf.shape[1]), "Design rank": rank,
        "Domain parameter name": domain_var, "Iterations": int(len(result.fit_history.get("params", []))),
        "Warnings/flags": "None" if not flags else "; ".join(flags), "Formula": formula,
    }
    params = pd.DataFrame({
        "Outcome": outcome_label, "Domain model": final_domain, "Parameter": result.params.index,
        "Coefficient": result.params.to_numpy(float), "Robust SE": result.bse.to_numpy(float),
        "P value": result.pvalues.to_numpy(float),
    })
    return result_row, diag_row, params


def write_validation_text(analysis, long, mapping, sheet_audit, conflicts, zmap, sheet_names, source_hash, gate, diagnostics):
    burden_raw = analysis["fibser_burden"].value_counts().sort_index().to_dict()
    parallel_note = "No discordant nonmissing values were found across paired follow-up sheets." if conflicts.empty else f"{len(conflicts)} discordant nonmissing paired-sheet values were retained for review; first nonmissing followed source-sheet order."
    stable_count = int(diagnostics["Stable"].sum())
    lines = [
        "Binary Logistic GEE sensitivity analyses: data validation record",
        "=" * 68,
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        f"Freeze ID target: {FREEZE_ID}",
        f"Raw source: {SOURCE}",
        f"Raw source SHA-256: {source_hash}",
        "Analysis inputs: the raw Excel workbook only. Existing scripts/mapping records were consulted for rules, not used as data or model-result inputs.",
        "",
        "Reconstruction",
        "--------------",
        f"Workbook sheets in source order: {' | '.join(sheet_names)}",
        "Baseline: first worksheet; ID normalization preserves the observed identifier apart from removing numeric .0 suffixes.",
        "Follow-up pairing: sheet positions (1,5), (2,6), (3,7), (4,8) correspond to Weeks 2, 8, 12, and 24.",
        "Paired-sheet aggregation: first nonmissing value in source-sheet order for each participant/time/variable; missing values were never recoded to absence.",
        f"Paired-sheet conflict check: {parallel_note}",
        "FIBSER burden: raw 0-6; four-level validation distribution 0/1/2/3-6.",
        "Natural complete case: nonmissing valid burden, time, all nine PRISE domains, age, sex, education, first-episode status, baseline HAMD-17, baseline SDS, baseline antidepressant class, and concurrent HAMD-17.",
        "Continuous covariates: z-standardized with population SD (ddof=0) on the rebuilt longitudinal data before complete-case restriction, matching the latest main analysis.",
        "Education: 小学/初中 -> Junior high or below; 高中/中专/职高 -> High school/vocational; 大学/大专/研究生及以上 -> College or above.",
        "",
        "Frozen-sample checks",
        "--------------------",
    ]
    lines.extend(f"{r['check']}: observed={r['observed']}; expected={r['expected']}; {r['status']}" for _, r in gate.iterrows())
    lines += [
        f"Raw FIBSER 0-6 counts: {json.dumps({int(k): int(v) for k,v in burden_raw.items()}, ensure_ascii=False, sort_keys=True)}",
        f"Long-data rows before complete-case restriction: {len(long)}",
        f"Analysis rows after natural complete-case restriction: {len(analysis)}",
        "",
        "PRISE occurrence-domain mapping verification",
        "--------------------------------------------",
        "Required final order: Sleep; Gastrointestinal; Other symptom; Nervous system; Heart; Eye/Ear; Skin; Sexual functioning; Genital/Urinary.",
        "Rule: code match first; otherwise unique visible Chinese PRISE domain label; where several labeled columns exist, select the actual occurrence item (1=是 or 1=有). No positional domain substitution.",
        "Coding: raw 1=present; raw 2=absent; explicit raw 0 retained as absent; blank remains missing.",
    ]
    for _, r in mapping.iterrows():
        lines.append(f"{r['week']} | {r['source_sheet']} | {r['final_domain']} -> {r['analysis_variable']} | {r['source_column']} | {r['match_method']} | {r['status']}")
    lines += [
        "",
        "Model settings and verification",
        "-------------------------------",
        "Binary logistic GEE; binomial family; logit link; participant ID clusters; exchangeable working correlation; robust sandwich covariance; two-sided Wald P values and 95% Wald confidence intervals.",
        "Each model contains one target PRISE domain only; no other PRISE domains and no domain-by-time interaction.",
        "References: Week 2, Female, College or above, First episode, and SNRI.",
        "Parameter trace: raw Excel -> participant/time collapse -> mapped binary domain -> complete-case matrix -> named domain coefficient -> exp(coefficient), exp(coefficient +/- 1.959964*robust SE), two-sided Wald P -> output table.",
        f"Stable models: {stable_count}/{len(diagnostics)}.",
        "",
        "Software",
        "--------",
        f"Python {platform.python_version()}; pandas {pd.__version__}; NumPy {np.__version__}; SciPy {scipy.__version__}; statsmodels {statsmodels.__version__}; patsy {patsy.__version__}",
    ]
    (OUT / "Data_Validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    mapping.to_csv(OUT / "PRISE_Mapping_Audit.csv", index=False, encoding="utf-8-sig")
    sheet_audit.to_csv(OUT / "Source_Sheet_Audit.csv", index=False, encoding="utf-8-sig")
    zmap.to_csv(OUT / "Standardization_Constants.csv", index=False, encoding="utf-8-sig")
    conflicts.to_csv(OUT / "Paired_Sheet_Conflicts.csv", index=False, encoding="utf-8-sig")


def main():
    analysis, long, mapping, sheet_audit, conflicts, zmap, sheet_names, source_hash = rebuild_analysis()
    gate = stop_gate(analysis)
    all_results, all_diag, all_params = [], [], []
    specifications = [
        ("any_burden", "Any burden (FIBSER burden 1-6)"),
        ("moderate_burden", "Moderate or greater burden (FIBSER burden 3-6)"),
    ]
    for outcome, label in specifications:
        for final_domain, domain_var in DOMAINS:
            rr, dd, pp = fit_one(analysis, outcome, label, final_domain, domain_var)
            rr["Outcome"] = label
            all_results.append(rr)
            all_diag.append(dd)
            all_params.append(pp)
            print(f"FIT {label} | {final_domain} | OR={rr['Adjusted OR']:.6g} | p={rr['P value']:.6g} | stable={rr['Stable']}", flush=True)
    results = pd.DataFrame(all_results)
    diagnostics = pd.DataFrame(all_diag)
    parameters = pd.concat(all_params, ignore_index=True)
    results.to_csv(OUT / "Model_Results.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(OUT / "Model_Diagnostics.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(OUT / "All_Model_Parameters.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(OUT / "Freeze_Gate.csv", index=False, encoding="utf-8-sig")
    write_validation_text(analysis, long, mapping, sheet_audit, conflicts, zmap, sheet_names, source_hash, gate, diagnostics)
    payload = {
        "metadata": {
            "freeze_id": FREEZE_ID, "source": str(SOURCE), "source_sha256": source_hash,
            "participants": int(analysis["participant_ID"].nunique()), "person_visits": int(len(analysis)),
            "week_counts": {k: int(v) for k, v in analysis["time"].astype(str).value_counts().to_dict().items()},
            "any_events": int(analysis["any_burden"].sum()), "moderate_events": int(analysis["moderate_burden"].sum()),
            "working_correlation": "exchangeable", "covariance": "robust sandwich", "link": "logit", "family": "binomial",
        },
        "results": results.to_dict(orient="records"),
        "diagnostics": diagnostics.to_dict(orient="records"),
        "freeze_gate": gate.to_dict(orient="records"),
        "mapping": mapping.to_dict(orient="records"),
    }
    (OUT / "workbook_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OUTPUT_DIR={OUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        (OUT / "RUN_FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise




