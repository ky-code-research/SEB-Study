"""Rebuild the exact restricted staging bundle consumed by Step 2 and Step 3.

No patient-level file is included in the public repository. Run this command
only in an authorized environment; its output directory is ignored by Git and
must remain restricted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

EXPECTED_SOURCE_SHA256 = "d7e59303c755dbd0547b0b6c84694d038f53e775e83a6a2483057007fd3054c8"
EXPECTED_FREEZE_SHA256 = "6844f512b5122beabc9905456e1a9a2111f6318c398a984232068b22c22fd16c"
FREEZE_ID = "PRISE_REPAIRED_NATURAL_CC_1468P_4799V_2026-10-08"
EXPECTED_PARTICIPANTS = 1468
EXPECTED_VISITS = 4799
EXPECTED_WEEK_COUNTS = {"Week2": 1381, "Week8": 1259, "Week12": 1146, "Week24": 1013}
EXPECTED_BURDEN4 = {0: 3464, 1: 584, 2: 512, 3: 239}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def joined_counts(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False).sort_index()
    return "; ".join(
        f"{'<missing>' if pd.isna(value) else value}: {int(count)}"
        for value, count in counts.items()
    )


def mapping_verification(rebuild) -> pd.DataFrame:
    workbook = pd.ExcelFile(rebuild.SOURCE, engine="openpyxl")
    positions = {"Week2": (1, 5), "Week8": (2, 6), "Week12": (3, 7), "Week24": (4, 8)}
    rows: list[dict] = []
    for week, sheet_positions in positions.items():
        for position in sheet_positions:
            sheet = workbook.sheet_names[position]
            data = pd.read_excel(rebuild.SOURCE, sheet_name=sheet, engine="openpyxl")
            anchor = next((i for i, column in enumerate(data.columns) if "PRISE" in str(column)), None)
            for domain in rebuild.DOMAINS:
                variable = rebuild.DOMAIN_VAR[domain]
                column = rebuild.occurrence_col(data, domain)
                labels = rebuild.PRISE_OCCURRENCE_LABELS[domain]
                normalized_column = str(column).replace(" ", "")
                label_match = any(label.replace(" ", "") in normalized_column for label in labels)
                raw = rebuild.numeric(data[column])
                invalid = sorted(set(raw.dropna().unique()) - {0.0, 1.0, 2.0})
                rows.append({
                    "week": week, "source_sheet": sheet, "domain": domain,
                    "final_model_variable": variable,
                    "expected_label_tokens": "; ".join(labels),
                    "selected_original_column": str(column),
                    "prise_anchor_index": anchor,
                    "selected_column_index": list(data.columns).index(column),
                    "match_method": "code_or_label; no positional fallback",
                    "label_match_verified": bool(label_match),
                    "raw_values_seen": joined_counts(raw),
                    "coding_rule": "raw 1 -> 1 present; raw 2 -> 0 absent; raw 0 -> 0 retained as absent",
                    "status": "PASS" if label_match and not invalid else "FAIL",
                })
    verification = pd.DataFrame(rows)
    expected_variables = set(rebuild.DOMAIN_VAR.values())
    if len(verification) != 72 or not verification["status"].eq("PASS").all():
        raise RuntimeError("STOP: PRISE mapping verification did not produce 72 PASS rows")
    if not expected_variables.issubset(set(verification["final_model_variable"])):
        raise RuntimeError("STOP: PRISE mapping verification is missing a locked model variable")
    return verification


def frozen_gates(analysis: pd.DataFrame) -> pd.DataFrame:
    week_counts = analysis["time"].astype(str).value_counts().to_dict()
    burden_counts = analysis["burden_ord4"].value_counts().sort_index().astype(int).to_dict()
    checks = [
        ("participants", int(analysis["participant_ID"].nunique()), EXPECTED_PARTICIPANTS),
        ("person-visits", int(len(analysis)), EXPECTED_VISITS),
        ("duplicate participant-time keys", int(analysis.duplicated(["participant_ID", "time"]).sum()), 0),
        *[(f"{week} visits", int(week_counts.get(week, 0)), expected) for week, expected in EXPECTED_WEEK_COUNTS.items()],
        *[(f"FIBSER burden level {level}", int(burden_counts.get(level, 0)), expected) for level, expected in EXPECTED_BURDEN4.items()],
    ]
    audit = pd.DataFrame(checks, columns=["check", "observed", "expected"])
    audit["status"] = audit.apply(
        lambda row: "PASS" if row["observed"] == row["expected"] else "FAIL", axis=1
    )
    if not audit["status"].eq("PASS").all():
        raise RuntimeError("STOP: frozen participant/visit/outcome gate failed")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-workbook", required=True, type=Path)
    parser.add_argument(
        "--staging-dir", type=Path,
        default=Path("outputs") / "PRISE_mapping_repair_data_freeze",
        help="Restricted Step 2/3 staging directory, relative to the repository root by default.",
    )
    args = parser.parse_args()
    source = args.raw_workbook.resolve()
    staging = args.staging_dir.resolve()
    if not source.is_file():
        raise SystemExit(f"Input workbook not found: {source}")
    source_hash = sha256(source)
    if source_hash.lower() != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "STOP: authorized workbook SHA-256 does not match the locked source; "
            f"observed={source_hash}"
        )

    targets = {
        "freeze": staging / "repaired_natural_complete_case_analysis_data.csv",
        "mapping": staging / "PRISE_Mapping_Verification.csv",
        "version": staging / "data_version_record.json",
        "gate": staging / "Freeze_Preflight.csv",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise SystemExit("STOP: refusing to overwrite existing restricted staging files: " + "; ".join(existing))

    os.environ["PROUD_RAW_WORKBOOK"] = str(source)
    import step2B1_YangSimpson_LIM_ordinal_GEE as rebuild

    _, _, analysis, _, _ = rebuild.build_dataset()
    gate = frozen_gates(analysis)
    mapping = mapping_verification(rebuild)
    staging.mkdir(parents=True, exist_ok=True)

    temporary_freeze = staging / ".repaired_natural_complete_case_analysis_data.tmp.csv"
    analysis.to_csv(temporary_freeze, index=False, encoding="utf-8-sig")
    freeze_hash = sha256(temporary_freeze)
    if freeze_hash.lower() != EXPECTED_FREEZE_SHA256:
        temporary_freeze.unlink(missing_ok=True)
        raise SystemExit(
            "STOP: reconstructed freeze SHA-256 does not match the locked analysis matrix; "
            f"observed={freeze_hash}"
        )
    temporary_freeze.replace(targets["freeze"])
    mapping.to_csv(targets["mapping"], index=False, encoding="utf-8-sig")
    gate.to_csv(targets["gate"], index=False, encoding="utf-8-sig")

    version = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "freeze_id": FREEZE_ID,
        "source_input": "PROUD_RAW_WORKBOOK (restricted; path intentionally omitted)",
        "source_sha256": source_hash,
        "freeze_sha256": freeze_hash,
        "repair_script": "analysis/prepare_authorized_data.py",
        "patched_module": "analysis/step2B1_YangSimpson_LIM_ordinal_GEE.py",
        "decision": "PASS",
        "sample_repaired_natural_complete_case": {
            "participants": int(analysis["participant_ID"].nunique()),
            "person_visits": int(len(analysis)),
            "time_counts": {
                key: int(value)
                for key, value in analysis["time"].astype(str).value_counts().sort_index().to_dict().items()
            },
        },
        "mapping_rows": int(len(mapping)), "mapping_status": "PASS",
    }
    targets["version"].write_text(
        json.dumps(version, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote restricted Step 2/3 staging bundle: {staging}")
    print(f"freeze_sha256={freeze_hash}; mapping_rows={len(mapping)}; gates=PASS")


if __name__ == "__main__":
    main()
