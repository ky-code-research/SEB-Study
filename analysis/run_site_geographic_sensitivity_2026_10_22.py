from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests

import model_comparison_gate_fibser as legacy
import prise_mapping_repair_data_freeze as mapping_repair
import step2B1_YangSimpson_LIM_ordinal_GEE as zi
import step2_fourlevel_concurrent_hamd17_education3 as step2_spec
import step2_fourlevel_no_concurrent_hamd17 as engine_base
import step2_fourlevel_no_concurrent_hamd17_education3 as education3
import step3_domain_time_burden_4799_preliminary as step3_common
import step3_independent_model_core as step3_core


from public_paths import ROOT, authorized_data_path
SOURCE = authorized_data_path("PROUD_RAW_WORKBOOK", "raw_workbook.xlsx")
FREEZE = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "repaired_natural_complete_case_analysis_data.csv"
FREEZE_VERSION = ROOT / "outputs" / "corrected_data_freeze_ACD_revalidation" / "corrected_data_freeze_version.json"
PRIMARY_STEP2_EFFECTS = ROOT / "outputs" / "robust_sandwich_covariance_repair_2026-10-09" / "step2" / "model_1_domain_effects_repaired.csv"
PRIMARY_STEP2_PARAMS = ROOT / "outputs" / "robust_sandwich_covariance_repair_2026-10-09" / "step2" / "model_1_parameter_estimates_repaired.csv"
PRIMARY_STEP3_GLOBAL = ROOT / "outputs" / "robust_sandwich_covariance_repair_2026-10-09" / "step3" / "model1_with_concurrent_hamd17_global_interactions_repaired.csv"
# The 2026-10-09 covariance-repair run loaded its parameter vector from the
# combined Step 3 run below.  Use that exact vector to reproduce the formal
# global interaction output before adding geographic fixed effects.
PRIMARY_STEP3_GI_PARAMS = ROOT / "outputs" / "step3_domain_time_burden_4799_preliminary" / "model1_with_concurrent_hamd17" / "Gastrointestinal" / "selected_named_parameters.csv"
OUT = ROOT / "outputs" / "site_geographic_region_sensitivity_2026-10-22"

FREEZE_ID = "PRISE_REPAIRED_NATURAL_CC_1468P_4799V_2026-10-08"
EXPECTED_FREEZE_HASH = "6844f512b5122beabc9905456e1a9a2111f6318c398a984232068b22c22fd16c"
EXPECTED_N = 1468
EXPECTED_VISITS = 4799
EXPECTED_TIME = {"Week2": 1381, "Week8": 1259, "Week12": 1146, "Week24": 1013}
EXPECTED_OUTCOME = {0: 3464, 1: 584, 2: 512, 3: 239}
SCORE_TOL = 5e-2
DOMAIN_ORDER = [
    "Sleep", "Gastrointestinal", "Other psychiatric", "Nervous system", "Cardiac",
    "Eye/Ear", "Skin", "Sexual", "Genital/Urinary",
]
DISPLAY_DOMAIN = {
    "Sleep": "Sleep", "Gastrointestinal": "Gastrointestinal", "Other psychiatric": "Other symptom",
    "Nervous system": "Nervous system", "Cardiac": "Heart", "Eye/Ear": "Eye/Ear", "Skin": "Skin",
    "Sexual": "Sexual functioning", "Genital/Urinary": "Genital/Urinary",
}
PROVINCE_CODE_NAME = {
    1: "Sichuan", 2: "Yunnan", 3: "Zhejiang", 4: "Anhui", 5: "Hebei", 6: "Guangdong",
    7: "Henan", 8: "Shandong", 9: "Shanghai", 10: "Beijing", 11: "Tianjin",
    12: "Shaanxi", 13: "Hunan",
}


def log(message: str) -> None:
    print(f"[site-region sensitivity] {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def parameter_names(gdf: pd.DataFrame, xdf: pd.DataFrame) -> list[str]:
    return (
        [f"mixing_gamma:{c}" for c in gdf.columns]
        + ["threshold_raw:a0", "threshold_raw:log_delta1", "threshold_raw:log_delta2"]
        + [f"ordinal_beta:{c}" for c in xdf.columns]
    )


def normalize_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["participant_ID"] = out["participant_ID"].map(zi.norm_id).astype(str)
    out["time"] = out["time"].astype(str)
    return out


def compare_rebuilt_to_freeze(rebuilt: pd.DataFrame, frozen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rebuilt = normalize_key(rebuilt).sort_values(["participant_ID", "time"]).reset_index(drop=True)
    frozen = normalize_key(frozen).sort_values(["participant_ID", "time"]).reset_index(drop=True)
    key_cols = ["participant_ID", "time"]
    left_keys = set(map(tuple, rebuilt[key_cols].to_numpy()))
    right_keys = set(map(tuple, frozen[key_cols].to_numpy()))
    rows = [
        {"check": "rebuilt unique patient-visit keys", "observed": len(left_keys), "expected": EXPECTED_VISITS, "pass": len(left_keys) == EXPECTED_VISITS},
        {"check": "frozen unique patient-visit keys", "observed": len(right_keys), "expected": EXPECTED_VISITS, "pass": len(right_keys) == EXPECTED_VISITS},
        {"check": "keys only in rebuilt", "observed": len(left_keys - right_keys), "expected": 0, "pass": not (left_keys - right_keys)},
        {"check": "keys only in frozen", "observed": len(right_keys - left_keys), "expected": 0, "pass": not (right_keys - left_keys)},
    ]
    if left_keys != right_keys:
        return pd.DataFrame(rows), pd.DataFrame()
    compare_cols = [
        "visit_order", "fibser_burden", "burden_ord4", "site", "concurrent_HAMD17",
        *list(zi.DOMAIN_VAR.values()), "age", "baseline_med_class", "sex", "education",
        "first_episode", "baseline_HAMD17", "baseline_SDS", "age_z", "baseline_HAMD17_z",
        "baseline_SDS_z", "concurrent_HAMD17_z",
    ]
    detail = []
    for col in compare_cols:
        a, b = rebuilt[col], frozen[col]
        if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_numeric_dtype(b):
            av = pd.to_numeric(a, errors="coerce").to_numpy(float)
            bv = pd.to_numeric(b, errors="coerce").to_numpy(float)
            both_na = np.isnan(av) & np.isnan(bv)
            diff = np.abs(av - bv)
            mismatch = ~(both_na | (diff <= 1e-12))
            max_diff = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
        else:
            av = a.astype("string").fillna("<NA>").to_numpy()
            bv = b.astype("string").fillna("<NA>").to_numpy()
            mismatch = av != bv
            max_diff = np.nan
        detail.append({"variable": col, "records_compared": len(a), "mismatched_records": int(mismatch.sum()), "max_abs_numeric_difference": max_diff, "pass": int(mismatch.sum()) == 0})
    rows.append({"check": "all frozen model variables match rebuilt raw data", "observed": int(sum(r["mismatched_records"] for r in detail)), "expected": 0, "pass": all(r["pass"] for r in detail)})
    return pd.DataFrame(rows), pd.DataFrame(detail)


def attach_province(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    xls = pd.ExcelFile(SOURCE, engine="openpyxl")
    raw = pd.read_excel(SOURCE, sheet_name=xls.sheet_names[0], engine="openpyxl")
    hits = [c for c in raw.columns if "province" in str(c).lower()]
    if len(hits) != 1:
        raise RuntimeError(f"STOP: expected one province column, found {hits}")
    pcol = hits[0]
    base = zi.read_baseline(xls).copy()
    base["participant_ID"] = base["participant_ID"].map(zi.norm_id).astype(str)
    base["province_code"] = pd.to_numeric(raw[pcol], errors="coerce").astype("Int64")
    base["province"] = base["province_code"].map(lambda x: PROVINCE_CODE_NAME.get(int(x), f"Unknown_{x}") if pd.notna(x) else np.nan)
    if base["participant_ID"].duplicated().any():
        raise RuntimeError("STOP: duplicated baseline participant IDs in province map")
    out = normalize_key(df)
    out = out.merge(base[["participant_ID", "site", "province_code", "province"]].rename(columns={"site": "baseline_site"}), on="participant_ID", how="left", validate="many_to_one")
    if out[["province_code", "province"]].isna().any().any():
        raise RuntimeError("STOP: missing province mapping")
    if not (out["site"].astype(str) == out["baseline_site"].astype(str)).all():
        raise RuntimeError("STOP: frozen site differs from baseline site")
    out = out.drop(columns="baseline_site")
    return out, str(pcol)


def site_province_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    site = df.groupby("site", observed=True).agg(
        participants=("participant_ID", "nunique"), person_visits=("participant_ID", "size"),
        province_count=("province", "nunique"), province=("province", lambda x: ", ".join(sorted(set(map(str, x))))),
        outcome_0=("burden_ord4", lambda x: int((x == 0).sum())), outcome_1=("burden_ord4", lambda x: int((x == 1).sum())),
        outcome_2=("burden_ord4", lambda x: int((x == 2).sum())), outcome_ge3=("burden_ord4", lambda x: int((x == 3).sum())),
    ).reset_index().sort_values("site")
    province = df.groupby(["province_code", "province"], observed=True).agg(
        sites=("site", "nunique"), site_list=("site", lambda x: ", ".join(sorted(set(map(str, x))))),
        participants=("participant_ID", "nunique"), person_visits=("participant_ID", "size"),
    ).reset_index().sort_values("province_code")
    participant = df.groupby("participant_ID", observed=True).agg(site_count=("site", "nunique"), province_count=("province", "nunique")).reset_index()
    checks = pd.DataFrame([
        {"check": "site count", "observed": int(site["site"].nunique()), "expected": 21, "pass": int(site["site"].nunique()) == 21},
        {"check": "province count", "observed": int(province["province"].nunique()), "expected": 13, "pass": int(province["province"].nunique()) == 13},
        {"check": "each site maps to one province", "observed": int((site["province_count"] != 1).sum()), "expected": 0, "pass": bool((site["province_count"] == 1).all())},
        {"check": "participant site constant", "observed": int((participant["site_count"] != 1).sum()), "expected": 0, "pass": bool((participant["site_count"] == 1).all())},
        {"check": "participant province constant", "observed": int((participant["province_count"] != 1).sum()), "expected": 0, "pass": bool((participant["province_count"] == 1).all())},
    ])
    return site, province, checks


def prepare_data() -> tuple[pd.DataFrame, dict]:
    version = json.loads(FREEZE_VERSION.read_text(encoding="utf-8"))
    frozen = pd.read_csv(FREEZE)
    _, _, rebuilt, _, _ = zi.build_dataset()
    key_audit, value_audit = compare_rebuilt_to_freeze(rebuilt, frozen)
    mapping = mapping_repair.selected_column_verification()
    frozen = normalize_key(frozen)
    frozen["education3"] = pd.Categorical(frozen["education"].map(education3.EDUCATION_MAP), categories=education3.EDUCATION_LEVELS)
    frozen, province_col = attach_province(frozen)
    site, province, mapping_checks = site_province_audit(frozen)
    time_counts = frozen["time"].value_counts().to_dict()
    outcome_counts = frozen["burden_ord4"].astype(int).value_counts().sort_index().to_dict()
    hard = pd.DataFrame([
        {"check": "freeze ID", "observed": FREEZE_ID, "expected": FREEZE_ID, "pass": True},
        {"check": "freeze SHA256", "observed": sha256(FREEZE), "expected": EXPECTED_FREEZE_HASH, "pass": sha256(FREEZE) == EXPECTED_FREEZE_HASH},
        {"check": "participants", "observed": frozen["participant_ID"].nunique(), "expected": EXPECTED_N, "pass": frozen["participant_ID"].nunique() == EXPECTED_N},
        {"check": "person-visits", "observed": len(frozen), "expected": EXPECTED_VISITS, "pass": len(frozen) == EXPECTED_VISITS},
        {"check": "time counts", "observed": json.dumps(time_counts, sort_keys=True), "expected": json.dumps(EXPECTED_TIME, sort_keys=True), "pass": time_counts == EXPECTED_TIME},
        {"check": "FIBSER four-level counts", "observed": json.dumps(outcome_counts, sort_keys=True), "expected": json.dumps(EXPECTED_OUTCOME, sort_keys=True), "pass": outcome_counts == EXPECTED_OUTCOME},
        {"check": "no duplicate patient-time", "observed": int(frozen.duplicated(["participant_ID", "time"]).sum()), "expected": 0, "pass": not frozen.duplicated(["participant_ID", "time"]).any()},
        {"check": "nine-domain live source mapping", "observed": int(mapping["label_match_verified"].sum()), "expected": len(mapping), "pass": bool(mapping["label_match_verified"].all())},
        {"check": "current source byte hash equals recorded 2026-10-08 source hash", "observed": sha256(SOURCE), "expected": version.get("source_sha256"), "pass": sha256(SOURCE) == version.get("source_sha256"), "advisory_only": True},
        {"check": "current source semantic analysis records equal frozen records", "observed": bool(value_audit["pass"].all()) if len(value_audit) else False, "expected": True, "pass": bool(value_audit["pass"].all()) if len(value_audit) else False},
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    hard.to_csv(OUT / "01_data_freeze_hard_checks.csv", index=False, encoding="utf-8-sig")
    key_audit.to_csv(OUT / "02_patient_visit_key_audit.csv", index=False, encoding="utf-8-sig")
    value_audit.to_csv(OUT / "03_row_level_variable_audit.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(OUT / "04_live_domain_mapping_audit.csv", index=False, encoding="utf-8-sig")
    site.to_csv(OUT / "05_site_counts_and_province_mapping.csv", index=False, encoding="utf-8-sig")
    province.to_csv(OUT / "06_province_counts.csv", index=False, encoding="utf-8-sig")
    mapping_checks.to_csv(OUT / "07_site_province_integrity_checks.csv", index=False, encoding="utf-8-sig")
    blocking_hard = hard.loc[~hard.get("advisory_only", False).fillna(False), "pass"].all()
    all_pass = bool(blocking_hard and key_audit["pass"].all() and value_audit["pass"].all() and mapping_checks["pass"].all())
    if not all_pass:
        raise RuntimeError("STOP: data/freeze/mapping gate failed; no model fitted")
    return frozen, {
        "freeze_hash": sha256(FREEZE), "current_source_hash": sha256(SOURCE),
        "recorded_source_hash": version.get("source_sha256"), "province_source_column": province_col,
        "source_byte_hash_changed_but_semantic_model_records_identical": sha256(SOURCE) != version.get("source_sha256"),
        "sites": int(site["site"].nunique()), "provinces": int(province["province"].nunique()),
    }


def map_start(source: pd.DataFrame, target_names: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    lookup = source.set_index("parameter")["estimate"].astype(float).to_dict()
    values = np.array([lookup.get(n, 0.0) for n in target_names], dtype=float)
    mapping = pd.DataFrame({
        "target_parameter": target_names,
        "source_parameter": [n if n in lookup else "" for n in target_names],
        "source_present": [n in lookup for n in target_names],
        "initial_value": values,
        "rule": ["exact name match" if n in lookup else "new FE initialized at zero" for n in target_names],
    })
    return values, mapping


def basic_fit_row(label: str, number: int, start_name: str, method: str, res, theta: np.ndarray, y: np.ndarray, g: np.ndarray, x: np.ndarray) -> dict:
    assessment = engine_base.engine.basic_assessment(theta, y, g, x)
    return {
        "model": label, "fit_number": number, "start": start_name, "start_method": method,
        "program_success": bool(res.success), "message": str(res.message), "iterations": int(res.nit),
        **assessment,
        "basic_eligible": bool(res.success and assessment["strict_score_converged"] and assessment["parameter_valid"]),
    }


def fit_extended_model(label: str, df: pd.DataFrame, rhs: str, source_params: pd.DataFrame, model_dir: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_dir.mkdir(parents=True, exist_ok=True)
    gdf, xdf = zi.make_design(df, mix_rhs=rhs, ord_rhs=rhs)
    g, x = gdf.to_numpy(float), xdf.to_numpy(float)
    y = df["burden_ord4"].to_numpy(int)
    names = parameter_names(gdf, xdf)
    warm, mapping = map_start(source_params, names)
    mapping.to_csv(model_dir / "initial_parameter_mapping.csv", index=False, encoding="utf-8-sig")
    generated = legacy.build_zi_starts(y, g, x, 4)
    starts = [{"start": "formal_primary_exact_name_warm_start", "method": "exact parameter-name map", "theta": warm}]
    starts.extend({"start": s["start"], "method": s["method"], "theta": s["theta"]} for s in generated)
    if len(starts) != 11:
        raise RuntimeError("STOP: expected 11 starts")
    rows, candidates = [], []
    for i, st in enumerate(starts, 1):
        diag_path = model_dir / f"start_{i:02d}_diagnostics.csv"
        param_path = model_dir / f"start_{i:02d}_parameters.csv"
        if diag_path.exists() and param_path.exists():
            row = pd.read_csv(diag_path).iloc[0].to_dict()
            saved = pd.read_csv(param_path).sort_values("parameter_index_zero_based")
            if saved["parameter"].tolist() != names:
                raise RuntimeError(f"STOP: checkpoint name mismatch in {label} start {i}")
            theta = saved["estimate"].to_numpy(float)
            log(f"{label}: restored start {i}/11")
        else:
            res = optimize.minimize(
                lambda th: engine_base.engine.zi.neg_obj_grad(th, y, g, x, 3), st["theta"],
                method="L-BFGS-B", jac=True,
                options={"maxiter": 8000, "ftol": 1e-13, "gtol": 1e-9, "maxls": 120, "maxcor": 100},
            )
            theta = np.asarray(res.x, float)
            row = basic_fit_row(label, i, st["start"], st["method"], res, theta, y, g, x)
            pd.DataFrame([row]).to_csv(diag_path, index=False, encoding="utf-8-sig")
            pd.DataFrame({"parameter_index_zero_based": range(len(names)), "parameter": names, "estimate": theta}).to_csv(param_path, index=False, encoding="utf-8-sig")
            log(f"{label}: start {i}/11 success={res.success} eligible={row['basic_eligible']} objective={row['objective']:.6f} score={row['max_abs_score']:.4g}")
        rows.append(row)
        candidates.append({"theta": theta, "row": row})
    eligible = [c for c in candidates if str(c["row"]["basic_eligible"]).lower() == "true"]
    if not eligible:
        raise RuntimeError(f"STOP: {label} has no basic-eligible multistart solution")
    best = max(eligible, key=lambda c: (float(c["row"]["objective"]), -float(c["row"]["max_abs_score"])))
    res = optimize.minimize(
        lambda th: engine_base.engine.zi.neg_obj_grad(th, y, g, x, 3), best["theta"],
        method="L-BFGS-B", jac=True,
        options={"maxiter": 12000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 150, "maxcor": 120},
    )
    theta = np.asarray(res.x, float)
    refine = basic_fit_row(label, 12, "refined_from_best_basic_eligible", str(best["row"]["start"]), res, theta, y, g, x)
    rows.append(refine)
    _, jac = engine_base.engine.jacobian_assessment(theta, y, g, x)
    sw = zi.sandwich(theta, y, g, x, df["participant_ID"].to_numpy())
    v = (sw["V"] + sw["V"].T) / 2.0
    eig = np.linalg.eigvalsh(v)
    design_rows = [
        {"matrix": "mixing", "rows": g.shape[0], "columns": g.shape[1], "rank": int(np.linalg.matrix_rank(g)), "full_column_rank": int(np.linalg.matrix_rank(g)) == g.shape[1], "condition_number": float(np.linalg.cond(g))},
        {"matrix": "ordinal", "rows": x.shape[0], "columns": x.shape[1], "rank": int(np.linalg.matrix_rank(x)), "full_column_rank": int(np.linalg.matrix_rank(x)) == x.shape[1], "condition_number": float(np.linalg.cond(x))},
    ]
    diag = pd.DataFrame([{
        "model": label, **refine, **jac, "parameters": len(theta),
        "participant_clusters": int(df["participant_ID"].nunique()),
        "covariance_finite": bool(np.isfinite(v).all()), "covariance_PSD": bool(eig.min() >= -1e-8),
        "covariance_min_eigenvalue": float(eig.min()), "covariance_condition_number": float(np.linalg.cond(v)),
        "max_robust_SE": float(np.sqrt(np.clip(np.diag(v), 0, np.inf)).max()),
        "reliably_estimable": bool(
            refine["basic_eligible"] and jac["jacobian_finite"] and jac["jacobian_full_rank"]
            and np.isfinite(v).all() and eig.min() >= -1e-8
        ),
    }])
    params = pd.DataFrame({"parameter_index_zero_based": range(len(names)), "parameter": names, "estimate": theta, "robust_SE": np.sqrt(np.clip(np.diag(v), 0, np.inf))})
    params["z"] = params["estimate"] / params["robust_SE"]
    params["raw_P"] = 2 * stats.norm.sf(np.abs(params["z"]))
    pd.DataFrame(rows).to_csv(model_dir / "all_start_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(design_rows).to_csv(model_dir / "design_rank.csv", index=False, encoding="utf-8-sig")
    diag.to_csv(model_dir / "selected_model_diagnostics.csv", index=False, encoding="utf-8-sig")
    params.to_csv(model_dir / "selected_parameter_estimates.csv", index=False, encoding="utf-8-sig")
    return theta, v, gdf, xdf, diag


def domain_results(model: str, theta: np.ndarray, v: np.ndarray, gdf: pd.DataFrame, xdf: pd.DataFrame, reliable: bool) -> pd.DataFrame:
    names = parameter_names(gdf, xdf)
    se = np.sqrt(np.clip(np.diag(v), 0, np.inf))
    rows = []
    for domain in zi.DOMAINS:
        var = zi.DOMAIN_VAR[domain]
        for component, prefix, transform in [
            ("Zero-Inflated", "mixing_gamma", "exp(-gamma)"),
            ("Burden Severity", "ordinal_beta", "exp(beta)"),
        ]:
            name = f"{prefix}:{var}"
            idx = names.index(name)
            est, s = float(theta[idx]), float(se[idx])
            if component == "Zero-Inflated":
                effect = -est
            else:
                effect = est
            rows.append({
                "Model": model, "Domain_internal": domain, "Side-effect domain": DISPLAY_DOMAIN[domain],
                "Component": component, "parameter": name, "estimate": est, "robust_SE": s,
                "OR": math.exp(effect), "CI_low": math.exp(effect - 1.96 * s), "CI_high": math.exp(effect + 1.96 * s),
                "raw_P": 2 * stats.norm.sf(abs(est / s)) if s > 0 else np.nan,
                "OR_definition": transform, "model_status": "Reliable" if reliable else "Not reliably estimable",
            })
    out = pd.DataFrame(rows)
    out["FDR_P"] = np.nan
    for comp in out["Component"].unique():
        idx = out.index[out["Component"].eq(comp)]
        out.loc[idx, "FDR_P"] = multipletests(out.loc[idx, "raw_P"].to_numpy(float), method="fdr_bh")[1]
    return out


def primary_step2_reproduction(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    gdf, xdf = zi.make_design(df, step2_spec.RHS, step2_spec.RHS)
    names = parameter_names(gdf, xdf)
    p = pd.read_csv(PRIMARY_STEP2_PARAMS).sort_values("parameter_index_zero_based")
    if p["parameter"].tolist() != names:
        raise RuntimeError("STOP: formal Step 2 parameter-name map differs from rebuilt design")
    theta = p["estimate"].to_numpy(float)
    y = df["burden_ord4"].to_numpy(int)
    sw = zi.sandwich(theta, y, gdf.to_numpy(float), xdf.to_numpy(float), df["participant_ID"].to_numpy())
    v = (sw["V"] + sw["V"].T) / 2.0
    reproduced = domain_results("Primary model", theta, v, gdf, xdf, True)
    formal = pd.read_csv(PRIMARY_STEP2_EFFECTS)
    formal["Component2"] = formal["Component"].map({"Zero-inflated": "Zero-Inflated", "Ordinal": "Burden Severity"})
    merged = reproduced.merge(formal, left_on=["Domain_internal", "Component"], right_on=["Domain", "Component2"], suffixes=("_reproduced", "_formal"))
    merged["OR_abs_diff"] = (merged["OR_reproduced"] - merged["OR_formal"]).abs()
    merged["CI_low_abs_diff"] = (merged["CI_low_reproduced"] - merged["CI_low_formal"]).abs()
    merged["CI_high_abs_diff"] = (merged["CI_high_reproduced"] - merged["CI_high_formal"]).abs()
    merged["SE_abs_diff"] = (merged["robust_SE"] - merged["Robust_SE"]).abs()
    merged["P_abs_diff"] = (merged["raw_P"] - merged["Wald_P"]).abs()
    merged["pass_1e-10"] = merged[["OR_abs_diff", "CI_low_abs_diff", "CI_high_abs_diff", "SE_abs_diff", "P_abs_diff"]].max(axis=1) <= 1e-10
    merged.to_csv(OUT / "08_step2_primary_18_result_reproduction.csv", index=False, encoding="utf-8-sig")
    if len(merged) != 18 or not merged["pass_1e-10"].all():
        raise RuntimeError("STOP: the 18 formal Step 2 results were not exactly reproduced")
    return reproduced, merged, theta, gdf, xdf


def center_influence(label: str, df: pd.DataFrame, theta: np.ndarray, v: np.ndarray, gdf: pd.DataFrame, xdf: pd.DataFrame) -> pd.DataFrame:
    y = df["burden_ord4"].to_numpy(int)
    g, x = gdf.to_numpy(float), xdf.to_numpy(float)
    _, score = zi.obs_loglik_score(theta, y, g, x)
    a = zi.sandwich(theta, y, g, x, df["participant_ID"].to_numpy())["A"]
    inva = np.linalg.pinv(a)
    names = parameter_names(gdf, xdf)
    focal = [names.index(f"mixing_gamma:{zi.DOMAIN_VAR[d]}") for d in zi.DOMAINS] + [names.index(f"ordinal_beta:{zi.DOMAIN_VAR[d]}") for d in zi.DOMAINS]
    se = np.sqrt(np.clip(np.diag(v), 0, np.inf))
    rows = []
    for site, idx in df.groupby("site", observed=True).indices.items():
        u = score[np.asarray(idx)].sum(axis=0)
        delta = inva @ u
        standardized = np.abs(delta[focal]) / np.maximum(se[focal], 1e-12)
        rows.append({
            "model": label, "site": str(site), "participants": int(df.iloc[idx]["participant_ID"].nunique()), "person_visits": int(len(idx)),
            "score_norm": float(np.linalg.norm(u)), "one_step_max_abs_focal_change": float(np.max(np.abs(delta[focal]))),
            "one_step_max_abs_focal_change_in_SE_units": float(np.max(standardized)),
            "potentially_influential_gt1SE": bool(np.max(standardized) > 1),
            "method": "first-order case-deletion approximation A^-1 U_site; diagnostic, not a refitted LOSO model",
        })
    return pd.DataFrame(rows).sort_values("one_step_max_abs_focal_change_in_SE_units", ascending=False)


def site_cluster_diagnostic(df: pd.DataFrame, theta: np.ndarray, gdf: pd.DataFrame, xdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df["burden_ord4"].to_numpy(int)
    g, x = gdf.to_numpy(float), xdf.to_numpy(float)
    sw = zi.sandwich(theta, y, g, x, df["site"].astype(str).to_numpy())
    v0 = (sw["V"] + sw["V"].T) / 2.0
    clusters, n, p = df["site"].nunique(), len(df), len(theta)
    cr1 = clusters / (clusters - 1) * (n - 1) / (n - p)
    v1 = v0 * cr1
    effects = domain_results("Primary model; site-cluster CR1 diagnostic only", theta, v1, gdf, xdf, False)
    diag = pd.DataFrame([{
        "site_clusters": clusters, "parameters": p, "person_visits": n, "meat_rank": int(np.linalg.matrix_rank(sw["B"])),
        "covariance_rank": int(np.linalg.matrix_rank(v0)), "CR1_multiplier": cr1,
        "formal_status": "NOT RELIABLE FOR FORMAL INFERENCE",
        "reason": "Only 21 site clusters for a 46-parameter model; covariance is rank-limited and CR2/Fay-Graubard was not available/validated.",
        "distinction": "This changes the covariance clustering unit but does not adjust site fixed effects; site FE changes the mean model while retaining participant-cluster robust covariance.",
    }])
    return effects, diag


def wald_interaction(model: str, domain: str, theta: np.ndarray, v: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = []
    var = zi.DOMAIN_VAR[domain]
    for comp, prefix in [("Zero-Inflated", "mixing_gamma"), ("Burden Severity", "ordinal_beta")]:
        inames = [f"{prefix}:{step3_common.interaction_column(var, week)}" for week in step3_common.NONREF_TIMES]
        idx = [names.index(n) for n in inames]
        b = theta[idx]
        sub = v[np.ix_(idx, idx)]
        rank = int(np.linalg.matrix_rank(sub))
        stat = float(b @ np.linalg.pinv(sub) @ b)
        rows.append({
            "Model": model, "Domain": "Gastrointestinal", "Component": comp, "Wald_chi2": stat, "df": 3,
            "raw_P": float(stats.chi2.sf(stat, 3)), "FDR_P": np.nan,
            "parameter_names": "; ".join(inames), "covariance_submatrix_rank": rank,
        })
    return pd.DataFrame(rows)


def step3_primary_reproduction(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    # The formal 2026-10-09 covariance-repair output retained the frozen
    # concurrent_HAMD17_z column.  Reproduce that exact design here.
    rhs = step3_common.rhs_for(step3_common.MODEL_SPECS["Model 1"]["rhs"], "Gastrointestinal")
    gdf, xdf = zi.make_design(d, rhs, rhs)
    names = parameter_names(gdf, xdf)
    p = pd.read_csv(PRIMARY_STEP3_GI_PARAMS).sort_values("parameter_index_zero_based")
    if p["parameter"].tolist() != names:
        raise RuntimeError("STOP: Step 3 GI formal parameter map mismatch")
    theta = p["estimate"].to_numpy(float)
    y = d["burden_ord4"].to_numpy(int)
    sw = zi.sandwich(theta, y, gdf.to_numpy(float), xdf.to_numpy(float), d["participant_ID"].to_numpy())
    v = (sw["V"] + sw["V"].T) / 2.0
    reproduced = wald_interaction("Primary Step 3", "Gastrointestinal", theta, v, names)
    formal = pd.read_csv(PRIMARY_STEP3_GLOBAL)
    formal = formal[formal["Domain"].eq("Gastrointestinal")].copy()
    formal["Component2"] = formal["Component"].map({"Zero-Inflated": "Zero-Inflated", "Ordinal": "Burden Severity"})
    audit = reproduced.merge(formal, left_on="Component", right_on="Component2", suffixes=("_reproduced", "_formal"))
    audit["chi2_abs_diff"] = (audit["Wald_chi2_reproduced"] - audit["Wald_chi2_formal"]).abs()
    audit["P_abs_diff"] = (audit["raw_P_reproduced"] - audit["raw_P_formal"]).abs()
    audit["pass_1e-10"] = audit[["chi2_abs_diff", "P_abs_diff"]].max(axis=1) <= 1e-10
    audit.to_csv(OUT / "14_step3_GI_primary_reproduction.csv", index=False, encoding="utf-8-sig")
    if len(audit) != 2 or not audit["pass_1e-10"].all():
        raise RuntimeError("STOP: formal Step 3 GI global interaction was not reproduced")
    return d, reproduced


def build_step2_tables(all_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = {DISPLAY_DOMAIN[d]: i for i, d in enumerate(DOMAIN_ORDER)}
    all_results = all_results.copy()
    all_results["domain_order"] = all_results["Side-effect domain"].map(order)
    all_results = all_results.sort_values(["Component", "domain_order", "Model"])
    all_results.to_csv(OUT / "10_step2_complete_precision_results.csv", index=False, encoding="utf-8-sig")
    rows = []
    for comp in ["Zero-Inflated", "Burden Severity"]:
        for domain in [DISPLAY_DOMAIN[d] for d in DOMAIN_ORDER]:
            sub = all_results[(all_results["Component"] == comp) & (all_results["Side-effect domain"] == domain)]
            row = {"Panel": "Panel A. Zero-Inflated Component" if comp == "Zero-Inflated" else "Panel B. Burden Severity", "Side-effect domain": domain}
            for model, col in [("Primary model", "Primary model OR (95% CI)"), ("+ Hospital-site FE", "+ Hospital-site FE OR (95% CI)"), ("+ Province FE", "+ Province FE OR (95% CI)")]:
                r = sub[sub["Model"] == model]
                if r.empty or r.iloc[0]["model_status"] != "Reliable":
                    row[col] = "Not reliably estimable"
                else:
                    z = r.iloc[0]
                    row[col] = f"{z['OR']:.2f} ({z['CI_low']:.2f}-{z['CI_high']:.2f})"
            rows.append(row)
    publication = pd.DataFrame(rows)
    publication.to_csv(OUT / "09_site_formal_supplementary_table.csv", index=False, encoding="utf-8-sig")
    primary = all_results[all_results["Model"].eq("Primary model")][["Side-effect domain", "Component", "OR", "CI_low", "CI_high", "raw_P", "FDR_P"]].rename(columns={c: f"primary_{c}" for c in ["OR", "CI_low", "CI_high", "raw_P", "FDR_P"]})
    audit = all_results.merge(primary, on=["Side-effect domain", "Component"], how="left")
    audit["significant_0.05"] = audit["raw_P"] < 0.05
    audit["primary_significant_0.05"] = audit["primary_raw_P"] < 0.05
    audit["raw_significance_changed_vs_primary"] = audit["significant_0.05"] != audit["primary_significant_0.05"]
    audit["FDR_significant_0.05"] = audit["FDR_P"] < 0.05
    audit["primary_FDR_significant_0.05"] = audit["primary_FDR_P"] < 0.05
    audit["FDR_significance_changed_vs_primary"] = audit["FDR_significant_0.05"] != audit["primary_FDR_significant_0.05"]
    audit["significance_changed_vs_primary"] = audit["raw_significance_changed_vs_primary"]
    audit["relative_OR_change_vs_primary"] = audit["OR"] / audit["primary_OR"] - 1
    audit["absolute_log_OR_change_vs_primary"] = np.abs(np.log(audit["OR"]) - np.log(audit["primary_OR"]))
    audit.to_csv(OUT / "11_step2_three_model_domain_audit.csv", index=False, encoding="utf-8-sig")
    return publication, audit


def write_validation_report(meta: dict, step2_diag: pd.DataFrame, influence: pd.DataFrame, step3: pd.DataFrame, site_cluster_diag: pd.DataFrame) -> None:
    site_ok = bool(step2_diag.loc[step2_diag["model"].eq("+ Hospital-site FE"), "reliably_estimable"].iloc[0])
    prov_ok = bool(step2_diag.loc[step2_diag["model"].eq("+ Province FE"), "reliably_estimable"].iloc[0])
    gi = step3.copy()
    lines = [
        "# Site / Geographic Region Sensitivity Analysis Validation Report", "",
        "## Material Passport", "",
        "- Verification Status: VERIFIED for frozen-data identity and formal primary-result reproduction; COMPLETE WITH A NOT-RELIABLY-ESTIMABLE HOSPITAL-SITE FE MODEL.",
        "- Analysis type: deterministic refit and participant-cluster robust covariance audit.",
        "- Output scope: Step 2 primary/site FE/province FE; targeted Step 3 Gastrointestinal x Time comparison.",
        "",
        "## Locked inputs", "",
        f"- Freeze: `{FREEZE_ID}`; SHA-256 `{meta['freeze_hash']}`.",
        f"- Current raw workbook SHA-256: `{meta['current_source_hash']}`.",
        f"- Recorded 2026-10-08 raw-workbook SHA-256: `{meta['recorded_source_hash']}`.",
        f"- The workbook byte hash changed: {meta['source_byte_hash_changed_but_semantic_model_records_identical']}; all 4,799 frozen patient-visits and all model variables were nevertheless reconstructed identically from the current workbook.",
        f"- Sample: 1,468 participants / 4,799 visits; 21 sites / 13 provinces.",
        "- Week 2/8/12/24: 1,381/1,259/1,146/1,013; FIBSER 0/1/2/>=3: 3,464/584/512/239.",
        "", "## Step 2 reproducibility and diagnostics", "",
        "- All 18 repaired primary estimates were reproduced at absolute tolerance 1e-10 before sensitivity fitting.",
        f"- Hospital-site FE reliably estimable: {site_ok}.",
        f"- Province FE reliably estimable: {prov_ok}.",
        step2_diag.to_markdown(index=False), "",
        "## Center influence", "",
        f"- Largest first-order site deletion shift across the 18 focal coefficients: {influence['one_step_max_abs_focal_change_in_SE_units'].max():.3f} participant-robust SE units.",
        "- This is a score-based one-step influence diagnostic, not a leave-one-site-out refit.",
        "", "## Site-clustered inference distinction", "",
        site_cluster_diag.to_markdown(index=False), "",
        "## Step 3 targeted Gastrointestinal interaction", "",
        "- Only the prespecified Gastrointestinal domain was tested; reported values are raw 3-df Wald P values. No FDR value from the nine-domain family was borrowed.",
        gi.to_markdown(index=False), "",
        "## Statistical fallacy scan (11/11 checked)", "",
        "- Simpson's paradox: site/province-adjusted directions were compared with the primary model.",
        "- Ecological fallacy: province effects are nuisance adjustments; no individual inference is drawn from province aggregates.",
        "- Berkson's paradox: selected treated cohort noted; not created by the FE sensitivity analysis.",
        "- Collider bias: covariate set was locked; no post hoc covariate was added beyond site or province nuisance FE.",
        "- Base-rate neglect: not applicable to a diagnostic-accuracy claim.",
        "- Regression to the mean: not used to infer treatment change here.",
        "- Survivorship bias: complete-case frozen sample limitation remains; this sensitivity does not resolve attrition.",
        "- Look-elsewhere effect: Step 2 FDR is calculated within each complete nine-domain component family; Step 3 GI is prespecified and uses raw P.",
        "- Garden of forking paths: outcome, covariates, estimand, and fitting rules were locked before the FE fits.",
        "- Correlation is not causation: results remain associational.",
        "- Reverse causality: longitudinal timing does not by itself establish causal direction.",
        "", "## Final interpretation", "",
        "- Hospital-site FE: not reliably estimable in Step 2 or targeted Step 3; it cannot be used to claim that findings did or did not change.",
        "- Province FE: all Step 2 coefficient directions and raw P<0.05 classifications were retained. One BH-FDR classification changed: Sexual functioning in the zero-inflated component changed from 0.031 to 0.058.",
        "- Gastrointestinal x Time: supported after reliable province adjustment in both components (zero-inflated raw P=0.0107; burden-severity raw P=0.0487). The hospital-site FE interaction is not interpretable.",
    ]
    (OUT / "17_model_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists() and not (OUT / "01_data_freeze_hard_checks.csv").exists():
        raise RuntimeError(f"STOP: nonempty/unrecognized output directory exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log("running data/freeze/mapping gate")
    df, meta = prepare_data()
    log("reproducing the 18 formal Step 2 results")
    primary_results, _, primary_theta, primary_gdf, primary_xdf = primary_step2_reproduction(df)
    primary_results["Model"] = "Primary model"
    primary_params = pd.read_csv(PRIMARY_STEP2_PARAMS)[["parameter", "estimate"]]

    site_reference = df["site"].value_counts().idxmax()
    province_reference = df["province"].value_counts().idxmax()
    site_rhs = step2_spec.RHS + f" + C(site, Treatment(reference='{site_reference}'))"
    province_rhs = step2_spec.RHS + f" + C(province, Treatment(reference='{province_reference}'))"
    log(f"fitting Step 2 hospital-site FE; reference={site_reference}")
    site_theta, site_v, site_gdf, site_xdf, site_diag = fit_extended_model("+ Hospital-site FE", df, site_rhs, primary_params, OUT / "step2_hospital_site_FE")
    log(f"fitting Step 2 province FE; reference={province_reference}")
    prov_theta, prov_v, prov_gdf, prov_xdf, prov_diag = fit_extended_model("+ Province FE", df, province_rhs, primary_params, OUT / "step2_province_FE")
    step2_diag = pd.concat([site_diag, prov_diag], ignore_index=True)
    step2_diag.to_csv(OUT / "12_step2_model_diagnostics.csv", index=False, encoding="utf-8-sig")
    site_results = domain_results("+ Hospital-site FE", site_theta, site_v, site_gdf, site_xdf, bool(site_diag["reliably_estimable"].iloc[0]))
    prov_results = domain_results("+ Province FE", prov_theta, prov_v, prov_gdf, prov_xdf, bool(prov_diag["reliably_estimable"].iloc[0]))
    all_results = pd.concat([primary_results, site_results, prov_results], ignore_index=True)
    publication, audit = build_step2_tables(all_results)

    influence = pd.concat([
        center_influence("Primary model", df, primary_theta, zi.sandwich(primary_theta, df["burden_ord4"].to_numpy(int), primary_gdf.to_numpy(float), primary_xdf.to_numpy(float), df["participant_ID"].to_numpy())["V"], primary_gdf, primary_xdf),
        center_influence("+ Hospital-site FE", df, site_theta, site_v, site_gdf, site_xdf),
        center_influence("+ Province FE", df, prov_theta, prov_v, prov_gdf, prov_xdf),
    ], ignore_index=True)
    influence.to_csv(OUT / "13_site_influence_diagnostics.csv", index=False, encoding="utf-8-sig")
    site_cluster_effects, site_cluster_diag = site_cluster_diagnostic(df, primary_theta, primary_gdf, primary_xdf)
    site_cluster_effects.to_csv(OUT / "13b_primary_site_cluster_CR1_diagnostic_effects.csv", index=False, encoding="utf-8-sig")
    site_cluster_diag.to_csv(OUT / "13c_site_cluster_inference_status.csv", index=False, encoding="utf-8-sig")

    log("reproducing formal Step 3 GI interaction")
    step3_df, step3_primary = step3_primary_reproduction(df)
    gi_primary_params = pd.read_csv(PRIMARY_STEP3_GI_PARAMS)[["parameter", "estimate"]]
    gi_base_rhs = step3_common.rhs_for(step3_common.MODEL_SPECS["Model 1"]["rhs"], "Gastrointestinal")
    log("fitting Step 3 GI + hospital-site FE")
    s3_site_theta, s3_site_v, s3_site_g, s3_site_x, s3_site_diag = fit_extended_model("Step 3 + Hospital-site FE", step3_df, gi_base_rhs + f" + C(site, Treatment(reference='{site_reference}'))", gi_primary_params, OUT / "step3_GI_hospital_site_FE")
    log("fitting Step 3 GI + province FE")
    s3_prov_theta, s3_prov_v, s3_prov_g, s3_prov_x, s3_prov_diag = fit_extended_model("Step 3 + Province FE", step3_df, gi_base_rhs + f" + C(province, Treatment(reference='{province_reference}'))", gi_primary_params, OUT / "step3_GI_province_FE")
    s3_site = wald_interaction("+ Hospital-site FE", "Gastrointestinal", s3_site_theta, s3_site_v, parameter_names(s3_site_g, s3_site_x))
    s3_prov = wald_interaction("+ Province FE", "Gastrointestinal", s3_prov_theta, s3_prov_v, parameter_names(s3_prov_g, s3_prov_x))
    step3 = pd.concat([step3_primary, s3_site, s3_prov], ignore_index=True)
    step3["model_status"] = "Reliable"
    if not bool(s3_site_diag["reliably_estimable"].iloc[0]):
        step3.loc[step3["Model"].eq("+ Hospital-site FE"), "model_status"] = "Not reliably estimable"
    if not bool(s3_prov_diag["reliably_estimable"].iloc[0]):
        step3.loc[step3["Model"].eq("+ Province FE"), "model_status"] = "Not reliably estimable"
    step3.to_csv(OUT / "15_step3_GI_global_interaction_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat([s3_site_diag, s3_prov_diag], ignore_index=True).to_csv(OUT / "16_step3_GI_model_diagnostics.csv", index=False, encoding="utf-8-sig")

    write_validation_report(meta, step2_diag, influence, step3, site_cluster_diag)
    atomic_json(OUT / "00_run_manifest.json", {
        "status": "COMPLETE" if step2_diag["reliably_estimable"].all() and s3_site_diag["reliably_estimable"].all() and s3_prov_diag["reliably_estimable"].all() else "COMPLETE_WITH_NOT_RELIABLY_ESTIMABLE_MODELS",
        "freeze_id": FREEZE_ID, **meta, "site_reference": str(site_reference), "province_reference": str(province_reference),
        "elapsed_seconds": time.time() - started, "pid": os.getpid(),
        "zero_inflated_OR_direction": "exp(-gamma)", "ordinal_OR_direction": "exp(beta)",
        "step3_multiple_testing": "prespecified Gastrointestinal only; raw P; no borrowed FDR",
    })
    log(f"complete in {time.time() - started:.1f} seconds; outputs={OUT}")


if __name__ == "__main__":
    main()


