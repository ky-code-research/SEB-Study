"""Data-free structural checks for public helper logic."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from analysis_spec import (DOMAIN_ORDER, TOLERABILITY_DOMAIN_ORDER, bh_adjust,
    corrected_sandwich, contrast_from_covariance, step2_or, tolerability_or)
from generate_synthetic_data import generate
from tolerability_contrasts import pairwise_tolerability_contrasts
from prepare_authorized_data import frozen_gates
from ipw_observation_helpers import weight_diagnostics

def main():
    d = generate(30)
    assert len(DOMAIN_ORDER) == 9 and len(TOLERABILITY_DOMAIN_ORDER) == 8
    assert d.participant_ID.nunique() == 30 and len(d) == 120
    assert not d.duplicated(["participant_ID", "time"]).any()
    q = bh_adjust([.001, .02, .20, .80])
    assert np.allclose(q, [.004, .04, 4/15, .8])
    a_inv = np.array([[1.0, .2], [.1, .8]])
    meat = np.array([[2.0, .3], [.3, 1.0]])
    v = corrected_sandwich(a_inv, meat)
    assert np.allclose(v, a_inv @ meat @ a_inv.T) and np.allclose(v, v.T)
    beta = np.array([.4, .9])
    cov = np.array([[.09, .02], [.02, .16]])
    c = np.array([-1., 1.])
    out = contrast_from_covariance(beta, cov, c)
    assert np.isclose(out["variance"], .09 + .16 - 2*.02)
    assert np.isclose(step2_or("Zero-inflated", .4), np.exp(.4))
    assert np.isclose(tolerability_or("Zero-inflated", .5), np.exp(-.5))
    assert np.isclose(tolerability_or("Ordinal", .5), np.exp(.5))
    gdf = pd.DataFrame(columns=[
        "Intercept", "C(target_tolerability)[T.Tolerable]",
        "C(target_tolerability)[T.Distressing]",
    ])
    xdf = pd.DataFrame(columns=[
        "C(target_tolerability)[T.Tolerable]",
        "C(target_tolerability)[T.Distressing]",
    ])
    theta = np.array([0.0, 0.2, 0.7, -1.0, 0.0, 0.0, 0.1, 1.0])
    robust = np.eye(len(theta)) * 0.04
    robust[1, 2] = robust[2, 1] = 0.01
    robust[6, 7] = robust[7, 6] = 0.01
    contrasts = pairwise_tolerability_contrasts(theta, robust, gdf, xdf, "Synthetic", "PASS")
    assert len(contrasts) == 6
    lookup = {(row["component"], row["contrast"]): row for row in contrasts}
    assert np.isclose(
        lookup[("Ordinal", "Distressing vs Tolerable")]["coefficient_contrast"],
        lookup[("Ordinal", "Distressing vs None")]["coefficient_contrast"]
        - lookup[("Ordinal", "Tolerable vs None")]["coefficient_contrast"],
    )
    assert np.isclose(
        lookup[("Ordinal", "Distressing vs Tolerable")]["robust_SE_full_covariance"] ** 2,
        0.04 + 0.04 - 2 * 0.01,
    )
    week_ids = {
        "Week2": range(0, 1381), "Week8": range(0, 1259),
        "Week12": range(322, 1468), "Week24": range(455, 1468),
    }
    staged = pd.DataFrame(
        [(f"S{pid:04d}", week) for week, ids in week_ids.items() for pid in ids],
        columns=["participant_ID", "time"],
    )
    staged["burden_ord4"] = np.repeat([0, 1, 2, 3], [3464, 584, 512, 239])
    assert frozen_gates(staged)["status"].eq("PASS").all()
    r_main = (ROOT / "R" / "reproduce_binary_gee_geepack.R").read_text(encoding="utf-8")
    r_compare = (ROOT / "R" / "compare_R_to_Python_binary_gee.R").read_text(encoding="utf-8")
    assert "py_occ_file" not in r_main and "Python_vs_R_comparison.csv" not in r_main
    assert "Python_vs_R_comparison.csv" in r_compare
    expected_path_text = 'ROOT / "outputs" / "PRISE_mapping_repair_data_freeze" / "repaired_natural_complete_case_analysis_data.csv"'
    for source_name in [
        "prepare_authorized_data.py", "step2_fourlevel_no_concurrent_hamd17.py",
        "run_secondary_tolerability_esgee.py", "secondary_prise_tolerability_qc.py",
    ]:
        source = (ROOT / "analysis" / source_name).read_text(encoding="utf-8")
        if source_name != "prepare_authorized_data.py":
            assert expected_path_text in source
    ipw_source = (ROOT / "analysis" / "reviewer1_comment3_attrition_ipw.py").read_text(encoding="utf-8")
    assert 'OUT = ROOT / "outputs" / "Reviewer1_Comment3_Attrition"' in ipw_source
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "outputs/" in gitignore and "Reviewer1_Comment3_Attrition/" in gitignore
    public_text = "\n".join(
        path.read_text(encoding="utf-8") for path in ROOT.rglob("*.md")
    )
    assert "Other symptoms" not in public_text
    synthetic_weights = pd.DataFrame({
        "participant_ID": ["S001", "S001", "S002"],
        "time": ["Week2", "Week8", "Week2"], "R_it": [1, 1, 1],
        "numerator_p": [.8, .7, .8], "denominator_p": [.75, .65, .85],
    })
    _, in_memory_weights, concentration = weight_diagnostics(synthetic_weights)
    assert "participant_ID" in in_memory_weights.columns
    assert "participant_ID" not in concentration.columns and len(concentration) == 1
    print("PASS: synthetic structure, frozen gates, unified staging paths, IPW output guards, terminology, aggregate-only IPW audit, BH, corrected sandwich, six tolerability contrasts, R/Python separation, covariance contrast, and OR directions")

if __name__ == "__main__":
    main()

