from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import step2B1_YangSimpson_LIM_ordinal_GEE as b1


from public_paths import ROOT, authorized_data_path
SOURCE = authorized_data_path("PROUD_RAW_WORKBOOK", "raw_workbook.xlsx")
SAVED_FREEZE = ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "repaired_natural_complete_case_analysis_data.csv"
OUT = ROOT / "outputs" / "secondary_analysis_PRISE_tolerability_QC_2026-10-11"

WEEKS = ["Week2", "Week8", "Week12", "Week24"]
WEEK_LABELS = {"Week2": "Week 2", "Week8": "Week 8", "Week12": "Week 12", "Week24": "Week 24"}
EXPECTED_TIME_COUNTS = {"Week2": 1381, "Week8": 1259, "Week12": 1146, "Week24": 1013}
POSITIONS = {"Week2": (1, 5), "Week8": (2, 6), "Week12": (3, 7), "Week24": (4, 8)}
DOMAINS = [
    "Gastrointestinal",
    "Cardiac",
    "Skin",
    "Nervous system",
    "Eye/Ear",
    "Genital/Urinary",
    "Sleep",
    "Sexual functioning",
]
B1_DOMAIN = {d: ("Sexual" if d == "Sexual functioning" else d) for d in DOMAINS}
MODEL_VAR = {d: b1.DOMAIN_VAR[B1_DOMAIN[d]] for d in DOMAINS}
LEVELS = ["None", "Tolerable", "Distressing"]
BURDEN_LEVELS = ["0", "1", "2", "≥3"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_numeric(series: pd.Series) -> pd.Series:
    raw = series.map(b1.clean)
    direct = pd.to_numeric(raw, errors="coerce")
    extracted = pd.to_numeric(raw.astype("string").str.extract(r"^\s*\[?(-?\d+(?:\.\d+)?)\]?.*$")[0], errors="coerce")
    return direct.where(direct.notna(), extracted)


def normalized_text(x: object) -> str:
    return re.sub(r"\s+", "", str(x)).replace("<br>", "")


def first_nonmissing(series: pd.Series):
    nonmissing = series.dropna()
    return np.nan if nonmissing.empty else nonmissing.iloc[0]


def counts_text(series: pd.Series) -> str:
    values = clean_numeric(series)
    items = []
    for value in sorted(values.dropna().unique().tolist()):
        items.append(f"{value:g}:{int((values == value).sum())}")
    items.append(f"blank:{int(values.isna().sum())}")
    return "; ".join(items)


def locate_tolerability_column(df: pd.DataFrame, occurrence_col: object, all_occurrence_indices: list[int]) -> object:
    cols = list(df.columns)
    start = cols.index(occurrence_col)
    next_starts = [i for i in all_occurrence_indices if i > start]
    stop = min(next_starts) if next_starts else len(cols)
    candidates = []
    for col in cols[start + 1 : stop]:
        text = normalized_text(col)
        has_measure_label = "严重程度" in text
        has_code_1 = "1=可以忍受" in text or "1=可忍受" in text
        has_code_2 = "2=非常痛苦" in text
        if has_measure_label and has_code_1 and has_code_2:
            candidates.append(col)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Tolerability mapping conflict for occurrence column {occurrence_col!s}: "
            f"expected one label-matched column, found {len(candidates)}: {[str(x) for x in candidates]}"
        )
    return candidates[0]


def build_raw_followup() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(SOURCE, engine="openpyxl")
    frames: list[pd.DataFrame] = []
    mapping_rows: list[dict] = []
    sheet_key_rows: list[dict] = []

    for visit_order, (week, positions) in enumerate(POSITIONS.items(), start=1):
        for source_priority, pos in enumerate(positions, start=1):
            sheet = xls.sheet_names[pos]
            df = pd.read_excel(SOURCE, sheet_name=sheet, engine="openpyxl")
            id_source_col = b1.id_col(df)
            burden_source_col = b1.burden_col(df)
            if burden_source_col is None:
                raise RuntimeError(f"FIBSER burden mapping failed in {sheet}")

            all_occurrence_cols = {d: b1.occurrence_col(df, d) for d in b1.DOMAINS}
            occurrences = {d: all_occurrence_cols[B1_DOMAIN[d]] for d in DOMAINS}
            # Use every PRISE domain start, including the excluded ninth domain, only
            # to delimit each labeled block. This prevents the final included domain
            # from accidentally seeing a later domain's tolerability header.
            occurrence_indices = sorted(list(df.columns).index(c) for c in all_occurrence_cols.values())
            tolerability = {
                d: locate_tolerability_column(df, occurrences[d], occurrence_indices)
                for d in DOMAINS
            }

            out = pd.DataFrame(
                {
                    "participant_ID": df[id_source_col].map(b1.norm_id),
                    "time": week,
                    "visit_order": visit_order,
                    "source_sheet": sheet,
                    "source_priority": source_priority,
                    "source_excel_row": np.arange(2, len(df) + 2),
                    "fibser_burden_raw": clean_numeric(df[burden_source_col]),
                }
            )
            dup_n = int(out.duplicated(["participant_ID", "time"], keep=False).sum())
            sheet_key_rows.append(
                {
                    "audit_section": "source_sheet_key_check",
                    "check_name": f"{sheet}: unique participant_ID + visit key",
                    "status": "PASS" if dup_n == 0 else "FAIL",
                    "observed": dup_n,
                    "expected": 0,
                    "source_sheet": sheet,
                    "week": week,
                    "detail": f"rows={len(out)}; source_priority={source_priority}",
                }
            )

            for domain in DOMAINS:
                occ_col = occurrences[domain]
                tol_col = tolerability[domain]
                occ = clean_numeric(df[occ_col])
                tol = clean_numeric(df[tol_col])
                out[f"{MODEL_VAR[domain]}__occ_raw"] = occ
                out[f"{MODEL_VAR[domain]}__tol_raw"] = tol

                occ_unique = sorted(occ.dropna().unique().tolist())
                tol_unique = sorted(tol.dropna().unique().tolist())
                occ_bad = sorted(set(occ_unique) - {1.0, 2.0})
                tol_bad = sorted(set(tol_unique) - {1.0, 2.0})
                occ_zero_n = int((occ == 0).sum())
                status = "PASS" if not occ_bad and not tol_bad and occ_zero_n == 0 else "FAIL"
                mapping_rows.append(
                    {
                        "audit_section": "domain_source_mapping",
                        "check_name": f"{sheet}: {domain}",
                        "status": status,
                        "observed": "",
                        "expected": "unique label-based mapping; occurrence {1,2,blank}; tolerability {1,2,blank}",
                        "source_sheet": sheet,
                        "week": week,
                        "domain": domain,
                        "final_domain_name": domain,
                        "questionnaire_label": "PRISE (Patient-Rated Inventory of Side Effects)",
                        "occurrence_variable": str(occ_col),
                        "occurrence_excel_column": list(df.columns).index(occ_col) + 1,
                        "occurrence_value_labels": "1=Yes/present; 2=No/absent; raw 0 not defined in header and not observed",
                        "occurrence_values_seen": counts_text(df[occ_col]),
                        "tolerability_variable": str(tol_col),
                        "tolerability_excel_column": list(df.columns).index(tol_col) + 1,
                        "tolerability_value_labels": "1=Tolerable (can tolerate); 2=Distressing (very painful/distressing)",
                        "tolerability_values_seen": counts_text(df[tol_col]),
                        "mapping_method": "occurrence: current Step 2/3 code-or-visible-domain-label rule; tolerability: unique explicit label within the same domain block",
                        "detail": f"unexpected_occurrence_values={occ_bad}; unexpected_tolerability_values={tol_bad}; raw_occurrence_0_n={occ_zero_n}",
                    }
                )
            frames.append(out)

    raw = pd.concat(frames, ignore_index=True)
    value_cols = [
        c
        for c in raw.columns
        if c not in {"participant_ID", "time", "visit_order", "source_sheet", "source_priority", "source_excel_row"}
    ]
    collapsed = (
        raw.sort_values(["visit_order", "source_priority", "source_excel_row"])
        .groupby(["participant_ID", "time"], dropna=False)[value_cols]
        .agg(first_nonmissing)
        .reset_index()
    )
    return collapsed, pd.DataFrame(mapping_rows), pd.DataFrame(sheet_key_rows)


def build_freeze_audit(collapsed: pd.DataFrame, rebuilt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    source_hash = sha256(SOURCE)
    saved = pd.read_csv(SAVED_FREEZE)
    rebuilt = rebuilt.copy()
    for frame in (saved, rebuilt, collapsed):
        frame["participant_ID"] = frame["participant_ID"].map(b1.norm_id)
        frame["time"] = frame["time"].astype(str)

    keys = ["participant_ID", "time"]
    rebuilt_key_set = set(map(tuple, rebuilt[keys].astype("string").to_numpy()))
    saved_key_set = set(map(tuple, saved[keys].astype("string").to_numpy()))
    time_counts = rebuilt["time"].value_counts().to_dict()
    key_match = rebuilt_key_set == saved_key_set
    rows.extend(
        [
            {
                "audit_section": "freeze_reproduction",
                "check_name": "raw source SHA-256",
                "status": "PASS",
                "observed": source_hash,
                "expected": source_hash,
                "detail": str(SOURCE),
            },
            {
                "audit_section": "freeze_reproduction",
                "check_name": "unique participants",
                "status": "PASS" if rebuilt["participant_ID"].nunique() == 1468 else "FAIL",
                "observed": int(rebuilt["participant_ID"].nunique()),
                "expected": 1468,
            },
            {
                "audit_section": "freeze_reproduction",
                "check_name": "person-visits",
                "status": "PASS" if len(rebuilt) == 4799 else "FAIL",
                "observed": len(rebuilt),
                "expected": 4799,
            },
            {
                "audit_section": "freeze_reproduction",
                "check_name": "unique participant_ID + visit keys",
                "status": "PASS" if not rebuilt.duplicated(keys).any() else "FAIL",
                "observed": int(rebuilt.duplicated(keys).sum()),
                "expected": 0,
            },
            {
                "audit_section": "freeze_reproduction",
                "check_name": "visit time counts",
                "status": "PASS" if time_counts == EXPECTED_TIME_COUNTS else "FAIL",
                "observed": json.dumps(time_counts, ensure_ascii=False, sort_keys=True),
                "expected": json.dumps(EXPECTED_TIME_COUNTS, ensure_ascii=False, sort_keys=True),
            },
            {
                "audit_section": "freeze_reproduction",
                "check_name": "participant-visit keys match saved current freeze",
                "status": "PASS" if key_match else "FAIL",
                "observed": bool(key_match),
                "expected": True,
                "detail": str(SAVED_FREEZE),
            },
            {
                "audit_section": "freeze_reproduction",
                "check_name": "FIBSER raw range 0-6",
                "status": "PASS" if rebuilt["fibser_burden"].isin(range(7)).all() else "FAIL",
                "observed": "; ".join(
                    f"{int(k)}:{int(v)}" for k, v in rebuilt["fibser_burden"].value_counts().sort_index().items()
                ),
                "expected": "integer 0-6",
            },
            {
                "audit_section": "freeze_reproduction",
                "check_name": "FIBSER four-level range",
                "status": "PASS" if set(rebuilt["burden_ord4"].unique()) <= {0, 1, 2, 3} else "FAIL",
                "observed": "; ".join(
                    f"{int(k)}:{int(v)}" for k, v in rebuilt["burden_ord4"].value_counts().sort_index().items()
                ),
                "expected": "0,1,2,≥3 (stored 3)",
            },
        ]
    )

    joined = rebuilt.merge(collapsed, on=keys, how="left", validate="one_to_one", indicator=True)
    rows.append(
        {
            "audit_section": "freeze_reproduction",
            "check_name": "all frozen keys found in direct raw reconstruction",
            "status": "PASS" if joined["_merge"].eq("both").all() else "FAIL",
            "observed": int(joined["_merge"].eq("both").sum()),
            "expected": 4799,
        }
    )
    fibser_match = joined["fibser_burden"].astype(float).eq(joined["fibser_burden_raw"].astype(float))
    rows.append(
        {
            "audit_section": "freeze_reproduction",
            "check_name": "FIBSER matches direct raw reconstruction",
            "status": "PASS" if fibser_match.all() else "FAIL",
            "observed": int((~fibser_match).sum()),
            "expected": 0,
        }
    )
    for domain in DOMAINS:
        var = MODEL_VAR[domain]
        raw_occ = joined[f"{var}__occ_raw"]
        derived = raw_occ.map({1.0: 1.0, 2.0: 0.0})
        matches = joined[var].astype(float).eq(derived.astype(float))
        rows.append(
            {
                "audit_section": "freeze_reproduction",
                "check_name": f"{domain}: occurrence matches current Step 2/3 frozen variable",
                "status": "PASS" if matches.all() else "FAIL",
                "observed": int((~matches).sum()),
                "expected": 0,
                "domain": domain,
            }
        )
    return pd.DataFrame(rows), joined.drop(columns="_merge")


def classify_tolerability(occ: pd.Series, tol: pd.Series) -> tuple[pd.Series, pd.Series]:
    result = pd.Series(pd.NA, index=occ.index, dtype="string")
    issue = pd.Series("", index=occ.index, dtype="string")
    result.loc[occ.eq(2) & tol.isna()] = "None"
    result.loc[occ.eq(1) & tol.eq(1)] = "Tolerable"
    result.loc[occ.eq(1) & tol.eq(2)] = "Distressing"
    issue.loc[occ.eq(1) & tol.isna()] = "true_missing_tolerability_after_present"
    issue.loc[occ.eq(2) & tol.notna()] = "contradiction_absent_with_tolerability"
    issue.loc[occ.isna() & tol.notna()] = "contradiction_missing_occurrence_with_tolerability"
    issue.loc[~occ.isna() & ~occ.isin([1, 2])] = "invalid_occurrence_code"
    issue.loc[~tol.isna() & ~tol.isin([1, 2])] = "invalid_tolerability_code"
    issue.loc[occ.isna() & tol.isna()] = "missing_occurrence_or_not_assessed"
    return result, issue


def burden_counts(group: pd.DataFrame) -> dict:
    n = len(group)
    labels = group["burden4_label"]
    out: dict[str, object] = {}
    for level, stem in zip(BURDEN_LEVELS, ["0", "1", "2", "ge3"]):
        count = int(labels.eq(level).sum())
        out[f"fibser_{stem}_n"] = count
        out[f"fibser_{stem}_row_pct"] = round(100.0 * count / n, 2) if n else np.nan
    for cutoff, stem in [(1, "ge1"), (2, "ge2"), (3, "ge3")]:
        count = int(group["burden_ord4"].ge(cutoff).sum())
        out[f"fibser_{stem}_n"] = count
        out[f"fibser_{stem}_pct"] = round(100.0 * count / n, 2) if n else np.nan
    return out


def create_outputs(analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    long_parts = []
    domain_qc: dict[str, dict] = {}
    for domain in DOMAINS:
        var = MODEL_VAR[domain]
        part = analysis[["participant_ID", "time", "visit_order", "fibser_burden", "burden_ord4", f"{var}__occ_raw", f"{var}__tol_raw"]].copy()
        part.columns = ["participant_ID", "time", "visit_order", "fibser_burden", "burden_ord4", "occurrence_raw", "tolerability_raw"]
        part["domain"] = domain
        part["tolerability_level"], part["issue"] = classify_tolerability(part["occurrence_raw"], part["tolerability_raw"])
        part["burden4_label"] = part["burden_ord4"].map({0: "0", 1: "1", 2: "2", 3: "≥3"})
        long_parts.append(part)
        issue_counts = part.loc[part["issue"].ne(""), "issue"].value_counts().to_dict()
        domain_qc[domain] = {
            "classified": int(part["tolerability_level"].notna().sum()),
            "true_missing": int(part["issue"].eq("true_missing_tolerability_after_present").sum()),
            "missing_occurrence": int(part["issue"].eq("missing_occurrence_or_not_assessed").sum()),
            "contradictions": int(part["issue"].str.startswith("contradiction").sum()),
            "invalid_codes": int(part["issue"].str.startswith("invalid").sum()),
            "issue_counts": issue_counts,
        }
    long = pd.concat(long_parts, ignore_index=True)

    overall_rows = []
    for domain in DOMAINS:
        d = long.loc[long["domain"].eq(domain)]
        qc = domain_qc[domain]
        for level in LEVELS:
            g = d.loc[d["tolerability_level"].eq(level)]
            overall_rows.append(
                {
                    "domain": domain,
                    "tolerability_level": level,
                    "tolerability_code": {"None": 0, "Tolerable": 1, "Distressing": 2}[level],
                    "person_visits": len(g),
                    "unique_participants": int(g["participant_ID"].nunique()),
                    **burden_counts(g),
                    "domain_true_missing_tolerability_n": qc["true_missing"],
                    "domain_missing_occurrence_or_not_assessed_n": qc["missing_occurrence"],
                    "domain_contradiction_n": qc["contradictions"],
                    "domain_invalid_code_n": qc["invalid_codes"],
                }
            )
    overall = pd.DataFrame(overall_rows)

    time_rows = []
    for domain in DOMAINS:
        for week in WEEKS:
            for level in LEVELS:
                g = long.loc[
                    long["domain"].eq(domain)
                    & long["time"].eq(week)
                    & long["tolerability_level"].eq(level)
                ]
                counts = burden_counts(g)
                four_counts = [counts[f"fibser_{stem}_n"] for stem in ["0", "1", "2", "ge3"]]
                zero_levels = [BURDEN_LEVELS[i] for i, n in enumerate(four_counts) if n == 0]
                sparse_levels = [BURDEN_LEVELS[i] for i, n in enumerate(four_counts) if 0 < n < 5]
                small_distressing = level == "Distressing" and len(g) < 30
                flags = []
                if small_distressing:
                    flags.append("Distressing group n<30")
                if zero_levels:
                    flags.append("zero FIBSER cells: " + ",".join(zero_levels))
                if sparse_levels:
                    flags.append("sparse FIBSER cells (1-4): " + ",".join(sparse_levels))
                time_rows.append(
                    {
                        "domain": domain,
                        "time": WEEK_LABELS[week],
                        "time_code": week,
                        "tolerability_level": level,
                        "tolerability_code": {"None": 0, "Tolerable": 1, "Distressing": 2}[level],
                        "person_visits": len(g),
                        "unique_participants": int(g["participant_ID"].nunique()),
                        **counts,
                        "small_distressing_group_n_lt30": bool(small_distressing),
                        "any_zero_fibser_cell": bool(zero_levels),
                        "any_sparse_fibser_cell_n_1_to_4": bool(sparse_levels),
                        "sparsity_flag_detail": "; ".join(flags),
                    }
                )
    time_specific = pd.DataFrame(time_rows)

    conditional_rows = []
    for domain in DOMAINS:
        d = long.loc[long["domain"].eq(domain) & long["tolerability_level"].isin(["Tolerable", "Distressing"])].copy()
        per_person = d.groupby("participant_ID")["tolerability_level"].agg(lambda x: set(x.dropna()))
        positive_visit_counts = d.groupby("participant_ID").size()
        switchers = set(per_person.index[per_person.map(lambda x: x == {"Tolerable", "Distressing"})])
        eligible_ge2 = set(positive_visit_counts.index[positive_visit_counts.ge(2)])
        t_to_d = set()
        d_to_t = set()
        for pid, g in d.sort_values("visit_order").groupby("participant_ID"):
            seq = g["tolerability_level"].tolist()
            for i in range(len(seq) - 1):
                later = seq[i + 1 :]
                if seq[i] == "Tolerable" and "Distressing" in later:
                    t_to_d.add(pid)
                if seq[i] == "Distressing" and "Tolerable" in later:
                    d_to_t.add(pid)
        conditional_rows.append(
            {
                "section": "domain_summary_and_switching",
                "domain": domain,
                "person_visits": len(d),
                "unique_participants": int(d["participant_ID"].nunique()),
                "tolerable_person_visits": int(d["tolerability_level"].eq("Tolerable").sum()),
                "distressing_person_visits": int(d["tolerability_level"].eq("Distressing").sum()),
                "participants_with_2plus_positive_visits": len(eligible_ge2),
                "participants_tolerable_only": int((per_person.map(lambda x: x == {"Tolerable"})).sum()),
                "participants_distressing_only": int((per_person.map(lambda x: x == {"Distressing"})).sum()),
                "participants_with_both_levels_switchers": len(switchers),
                "switchers_pct_among_2plus_positive_visits": round(100.0 * len(switchers) / len(eligible_ge2), 2) if eligible_ge2 else np.nan,
                "participants_with_tolerable_to_distressing": len(t_to_d),
                "participants_with_distressing_to_tolerable": len(d_to_t),
                "participants_with_both_switch_directions": len(t_to_d & d_to_t),
            }
        )
        for week in WEEKS:
            for level in ["Tolerable", "Distressing"]:
                g = d.loc[d["time"].eq(week) & d["tolerability_level"].eq(level)]
                counts = burden_counts(g)
                four_counts = [counts[f"fibser_{stem}_n"] for stem in ["0", "1", "2", "ge3"]]
                conditional_rows.append(
                    {
                        "section": "time_group_fibser_distribution",
                        "domain": domain,
                        "time": WEEK_LABELS[week],
                        "time_code": week,
                        "tolerability_level": level,
                        "tolerability_code": {"Tolerable": 1, "Distressing": 2}[level],
                        "person_visits": len(g),
                        "unique_participants": int(g["participant_ID"].nunique()),
                        **counts,
                        "small_distressing_group_n_lt30": bool(level == "Distressing" and len(g) < 30),
                        "any_zero_fibser_cell": bool(any(n == 0 for n in four_counts)),
                        "any_sparse_fibser_cell_n_1_to_4": bool(any(0 < n < 5 for n in four_counts)),
                    }
                )
    conditional = pd.DataFrame(conditional_rows)
    conditional_count_columns = [
        "person_visits",
        "unique_participants",
        "tolerable_person_visits",
        "distressing_person_visits",
        "participants_with_2plus_positive_visits",
        "participants_tolerable_only",
        "participants_distressing_only",
        "participants_with_both_levels_switchers",
        "participants_with_tolerable_to_distressing",
        "participants_with_distressing_to_tolerable",
        "participants_with_both_switch_directions",
        "fibser_0_n",
        "fibser_1_n",
        "fibser_2_n",
        "fibser_ge3_n",
        "fibser_ge1_n",
        "fibser_ge2_n",
    ]
    for col in conditional_count_columns:
        if col in conditional.columns:
            conditional[col] = conditional[col].astype("Int64")
    return overall, time_specific, conditional, {"long": long, "domain_qc": domain_qc}


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        vals = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    source_hash: str,
    mapping_audit: pd.DataFrame,
    overall: pd.DataFrame,
    time_specific: pd.DataFrame,
    conditional: pd.DataFrame,
    qc: dict,
) -> None:
    overview = overall[
        [
            "domain",
            "tolerability_level",
            "person_visits",
            "unique_participants",
            "fibser_ge1_pct",
            "fibser_ge2_pct",
            "fibser_ge3_pct",
            "domain_true_missing_tolerability_n",
            "domain_contradiction_n",
        ]
    ]
    switch = conditional.loc[conditional["section"].eq("domain_summary_and_switching")].copy()
    small_rows = time_specific.loc[time_specific["small_distressing_group_n_lt30"]]
    zero_rows = time_specific.loc[time_specific["any_zero_fibser_cell"]]
    sparse_rows = time_specific.loc[time_specific["any_sparse_fibser_cell_n_1_to_4"]]
    all_mapping_pass = mapping_audit["status"].eq("PASS").all()
    total_true_missing = sum(v["true_missing"] for v in qc["domain_qc"].values())
    total_contradictions = sum(v["contradictions"] for v in qc["domain_qc"].values())
    any_domain_distressing = overall.loc[overall["tolerability_level"].eq("Distressing"), "person_visits"].min()
    min_switchers = int(switch["participants_with_both_levels_switchers"].min())
    max_switchers = int(switch["participants_with_both_levels_switchers"].max())
    decision_b = "basic descriptive support" if all_mapping_pass else "not supported because an audit gate failed"
    decision_c = (
        "basic descriptive support, but sparse Distressing/FIBSER cells require caution"
        if len(small_rows) or len(zero_rows) or len(sparse_rows)
        else "basic descriptive support without flagged sparse cells"
    )
    report = f"""# Secondary Analysis: PRISE tolerability descriptive QC report

## Scope and stopping rules

This run performs source mapping checks and descriptive statistics only. It fits no model, calculates no P value, and does not modify the source workbook or overwrite prior results.

- Source workbook SHA-256: `{source_hash}`
- Frozen sample reproduced: 1468 participants / 4799 person-visits.
- Visit counts: Week 2 = 1381, Week 8 = 1259, Week 12 = 1146, Week 24 = 1013.
- Frozen keys match the current saved Step 2/3 freeze exactly.
- All source mapping and sample gates: **{'PASS' if all_mapping_pass else 'FAIL'}**.

## Mapping and coding audit

The eight target domains were matched by the current Step 2/3 occurrence mapping rule (questionnaire code first, visible PRISE domain label fallback, no positional fallback). Within each matched domain block, tolerability was matched only when exactly one source header explicitly contained the measure label and both value labels: `1=can tolerate` and `2=very painful/distressing`.

The occurrence headers define `1=Yes/present` and `2=No/absent`. No raw occurrence value `0` was observed in any of the 64 sheet-domain mappings, so no raw zero was used to infer absence. Tolerability values were limited to 1, 2, or blank.

The derived categories were:

- occurrence 2 + blank tolerability = None;
- occurrence 1 + tolerability 1 = Tolerable;
- occurrence 1 + tolerability 2 = Distressing;
- occurrence 1 + blank tolerability = true missing tolerability;
- missing occurrence = missing/not assessed;
- inconsistent occurrence/tolerability combinations = contradiction, without automatic correction.

Across the eight domains there were {total_true_missing} true missing tolerability records and {total_contradictions} contradiction records. Domain-specific counts are in the overall output and source audit.

## Overall distribution (Plan B)

{md_table(overview, list(overview.columns))}

FIBSER counts are person-visit counts. Row percentages use the number of classified person-visits in the corresponding domain and tolerability level as the denominator. Unique participant counts are also provided because a participant may contribute multiple visits and may appear in more than one tolerability level over time.

## Time-specific sparsity

- Distressing rows with group size <30: {len(small_rows)} of 32 domain-time Distressing rows.
- Domain-time-level rows with at least one zero FIBSER category: {len(zero_rows)} of 96 rows.
- Domain-time-level rows with at least one nonzero sparse FIBSER category (1-4 observations): {len(sparse_rows)} of 96 rows.
- The smallest overall Distressing group across domains contains {int(any_domain_distressing)} person-visits.

Flags are explicit in `TimeSpecific_Tolerability_Burden_Distribution.csv`. A zero cell and a sparse cell are flagged separately. The descriptive sparse-cell rule is 1-4 observations; the small Distressing-group rule is fewer than 30 observations.

## Conditional subset feasibility (Plan C)

The conditional subset retains only visits classified as Tolerable or Distressing. It contains domain-specific counts, time-specific FIBSER distributions, and within-person switching summaries. Across domains, the number of participants observed with both tolerability levels ranges from {min_switchers} to {max_switchers}.

{md_table(switch, ['domain', 'person_visits', 'unique_participants', 'distressing_person_visits', 'participants_with_2plus_positive_visits', 'participants_with_both_levels_switchers', 'switchers_pct_among_2plus_positive_visits'])}

Switching is descriptive: a participant is a switcher when both Tolerable and Distressing are observed across positive-symptom visits for the same domain. Directional counts indicate whether a later positive visit changed from Tolerable to Distressing or from Distressing to Tolerable; both directions may occur in the same participant.

## Final assessment

- Data mapping: correct for all eight domains and all eight follow-up source sheets; no positional guessing was used.
- Frozen sample: exactly reproduced and matched to the current saved Step 2/3 freeze.
- Missingness: true missing tolerability after confirmed occurrence is retained as missing, not recoded as None. Contradictions are retained as flagged records and not repaired.
- Plan B: {decision_b}.
- Plan C: {decision_c}. The conditional distributions are adequate for description, but small Distressing groups, zero cells, and sparse FIBSER categories should be considered before any future modeling.

This run stops here. No GEE, regression, model fitting, model comparison, or P-value calculation was performed.
"""
    (OUT / "Secondary_Descriptive_QC_Report.md").write_text(report, encoding="utf-8")


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"STOP: source workbook not found: {SOURCE}")
    if not SAVED_FREEZE.exists():
        raise SystemExit(f"STOP: current saved freeze not found: {SAVED_FREEZE}")

    collapsed, mapping_rows, sheet_key_rows = build_raw_followup()
    _, _, rebuilt, _, _ = b1.build_dataset()
    freeze_rows, analysis = build_freeze_audit(collapsed, rebuilt)
    audit = pd.concat([freeze_rows, sheet_key_rows, mapping_rows], ignore_index=True, sort=False)

    hard_fail = audit.loc[audit["status"].ne("PASS")]
    if len(hard_fail):
        OUT.mkdir(parents=True, exist_ok=True)
        audit.to_csv(OUT / "Source_Mapping_Audit.csv", index=False, encoding="utf-8-sig")
        stop_report = (
            "# Secondary Analysis: PRISE tolerability QC\n\n"
            "**STOPPED before descriptive statistics.** A frozen-sample, FIBSER, participant/visit-key, "
            "or source-column mapping gate failed. See `Source_Mapping_Audit.csv`.\n"
        )
        (OUT / "Secondary_Descriptive_QC_Report.md").write_text(stop_report, encoding="utf-8")
        raise SystemExit("STOP: hard audit gate failed; descriptive outputs were not generated")

    overall, time_specific, conditional, qc = create_outputs(analysis)
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / "Source_Mapping_Audit.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUT / "Overall_Tolerability_Burden_Distribution.csv", index=False, encoding="utf-8-sig")
    time_specific.to_csv(OUT / "TimeSpecific_Tolerability_Burden_Distribution.csv", index=False, encoding="utf-8-sig")
    conditional.to_csv(OUT / "Conditional_Subset_Feasibility.csv", index=False, encoding="utf-8-sig")
    write_report(sha256(SOURCE), audit, overall, time_specific, conditional, qc)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_directory": str(OUT),
                "participants": int(analysis["participant_ID"].nunique()),
                "person_visits": len(analysis),
                "mapping_rows": len(mapping_rows),
                "true_missing_total_across_domains": sum(v["true_missing"] for v in qc["domain_qc"].values()),
                "contradictions_total_across_domains": sum(v["contradictions"] for v in qc["domain_qc"].values()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()




