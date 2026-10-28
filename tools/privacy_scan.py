"""Fail-closed privacy/sensitive-content scan for the public candidate tree."""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".xlsx", ".xls", ".xlsb", ".csv", ".tsv", ".parquet", ".feather",
    ".rds", ".rdata", ".sav", ".dta", ".sas7bdat", ".ndjson", ".ipynb",
    ".pkl", ".pickle", ".joblib", ".log",
}
PATTERNS = {
    "windows_absolute_path": re.compile(r"(?i)\b[A-Z]:\\(?:Users|Desktop|Documents)\\"),
    "unix_home_path": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "generic_secret_assignment": re.compile(r"(?i)(password|passwd|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    "original_workbook_name": re.compile("2\\.3更新" + "数据|去标识" + "化"),
    "participant_identifier_literal": re.compile(
        r"(?i)\b(?:participant|patient)\s*(?:ID|number|identifier)?\s*[`'\":=]*\s*\d{6,12}\b|"
        r"(?:参与者|患者)\s*(?:编号|ID)?\s*[`'\":=]*\s*\d{6,12}\b"
    ),
    "record_specific_outcome_literal": re.compile(
        r"(?i)\b\d{6,12}\b[^\r\n]{0,100}(?:FIBSER|burden|Week\s*\d)"
    ),
    "unsafe_patient_level_export_marker": re.compile(
        r"(?i)(?:dropout_person_patterns|participant_weight_concentration|"
        r"observed_visit_weights|patient_level_export|participant_level_export)"
    ),
}
TEXT_SUFFIXES = {
    ".py", ".r", ".md", ".txt", ".env", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".cff", "",
}

def main() -> int:
    findings = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts:
            findings.append((str(rel), "embedded_git_history"))
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append((str(rel), f"forbidden_file_type:{path.suffix.lower()}"))
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", "requirements.txt"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                findings.append((str(rel), "non_utf8_text"))
                continue
            for name, pattern in PATTERNS.items():
                if rel.as_posix() == "tools/privacy_scan.py" and name == "unsafe_patient_level_export_marker":
                    continue
                if pattern.search(text):
                    findings.append((str(rel), name))
    if findings:
        for rel, issue in findings:
            print(f"FAIL\t{rel}\t{issue}")
        return 1
    print(
        f"PASS: {sum(1 for p in ROOT.rglob('*') if p.is_file())} files scanned; "
        "no forbidden data, participant-record literals/export markers, local paths, "
        "credentials, notebooks, logs, or Git history detected"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

