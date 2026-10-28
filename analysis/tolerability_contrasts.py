"""Data-free post-estimation contrasts for the eight tolerability models."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def pairwise_tolerability_contrasts(
    theta: np.ndarray,
    robust_covariance: np.ndarray,
    Gdf: pd.DataFrame,
    Xdf: pd.DataFrame,
    domain: str,
    model_status: str,
    ordered_thresholds: int = 3,
) -> list[dict]:
    """Compute the locked three pairwise contrasts without refitting.

    None is the design-matrix reference and every standard error uses the full
    robust covariance through c' V c. Multiplicity is applied after all domains
    are combined: only Distressing-vs-Tolerable belongs to the prespecified
    eight-test family within each model component.
    """
    component_specs = [
        ("Zero-inflated", Gdf, 0, -1.0,
         "exp[-contrast in gamma]: odds ratio for extra-zero membership"),
        ("Ordinal", Xdf, Gdf.shape[1] + ordered_thresholds, 1.0,
         "exp[contrast in beta]: odds ratio for higher burden within ordinary component"),
    ]
    contrast_specs = [
        ("Tolerable vs None", {"Tolerable": 1.0}, "Exploratory"),
        ("Distressing vs None", {"Distressing": 1.0}, "Exploratory"),
        ("Distressing vs Tolerable", {"Distressing": 1.0, "Tolerable": -1.0}, "Primary"),
    ]
    rows: list[dict] = []
    for component, design, offset, sign, definition in component_specs:
        level_columns = {
            level: next(
                column for column in design.columns
                if "target_tolerability" in column and f"[T.{level}]" in column
            )
            for level in ["Tolerable", "Distressing"]
        }
        columns = list(design.columns)
        for contrast_name, weights, role in contrast_specs:
            contrast = np.zeros(len(theta))
            for level, weight in weights.items():
                contrast[offset + columns.index(level_columns[level])] = weight
            estimate = float(contrast @ theta)
            variance = float(contrast @ robust_covariance @ contrast)
            standard_error = math.sqrt(max(variance, 0.0))
            z_value = estimate / standard_error if standard_error > 0 else np.nan
            log_or = sign * estimate
            rows.append({
                "domain": domain, "component": component, "contrast": contrast_name,
                "contrast_role": role, "preliminary": True,
                "model_status": model_status, "coefficient_contrast": estimate,
                "robust_SE_full_covariance": standard_error, "wald_z": z_value,
                "raw_P": float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan,
                "OR": math.exp(log_or),
                "CI_low": math.exp(log_or - 1.96 * standard_error),
                "CI_high": math.exp(log_or + 1.96 * standard_error),
                "OR_definition": definition,
                "contrast_variance_formula": "c' V_robust c",
                "interpretation_permitted": model_status == "PASS",
            })
    return rows
