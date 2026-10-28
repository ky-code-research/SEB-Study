"""Audited constants and small deterministic helpers shared by public checks.

This file does not replace the frozen fitting scripts. It exposes the locked
domain order, multiplicity families, OR directions, contrast algebra, and the
corrected sandwich multiplication order for lightweight review and tests.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

DOMAIN_ORDER = (
    "Gastrointestinal", "Cardiac", "Skin", "Nervous system", "Eye/Ear",
    "Genital/Urinary", "Sleep", "Sexual", "Other psychiatric",
)
DOMAIN_VARIABLES = {
    "Gastrointestinal": "Gastrointestinal", "Cardiac": "Cardiac", "Skin": "Skin",
    "Nervous system": "Nervous", "Eye/Ear": "EyeEar",
    "Genital/Urinary": "Genitourinary", "Sleep": "Sleep",
    "Sexual": "Sexual", "Other psychiatric": "OtherPsych",
}
TOLERABILITY_DOMAIN_ORDER = (
    "Gastrointestinal", "Cardiac", "Skin", "Nervous system", "Eye/Ear",
    "Genital/Urinary", "Sleep", "Sexual functioning",
)
WEEKS = ("Week2", "Week8", "Week12", "Week24")
EXPECTED_VISITS = {"Week2": 1381, "Week8": 1259, "Week12": 1146, "Week24": 1013}
EXPECTED_BURDEN4 = {0: 3464, 1: 584, 2: 512, 3: 239}
FREEZE_ID = "PRISE_REPAIRED_NATURAL_CC_1468P_4799V_2026-10-08"
EXPECTED_PARTICIPANTS = 1468
EXPECTED_PERSON_VISITS = 4799

COMPLETE_CASE_COLUMNS = (
    "burden_ord4", "time", "age_z", "sex", "education", "first_episode",
    "baseline_HAMD17_z", "baseline_SDS_z", "baseline_med_class",
    "concurrent_HAMD17_z", *DOMAIN_VARIABLES.values(),
)

def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("BH inputs must be a finite one-dimensional vector in [0,1]")
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out

def corrected_sandwich(a_inverse, cluster_score_meat):
    """Locked correction: V = A^{-1} B A^{-T}."""
    a_inverse = np.asarray(a_inverse, float)
    meat = np.asarray(cluster_score_meat, float)
    return a_inverse @ meat @ a_inverse.T

def scalar_wald(estimate, variance):
    se = float(np.sqrt(variance))
    z = float(estimate / se)
    return se, z, float(2 * stats.norm.sf(abs(z)))

def contrast_from_covariance(beta, covariance, contrast):
    beta = np.asarray(beta, float)
    covariance = np.asarray(covariance, float)
    contrast = np.asarray(contrast, float)
    estimate = float(contrast @ beta)
    variance = float(contrast @ covariance @ contrast)
    se, z, p = scalar_wald(estimate, variance)
    return {"estimate": estimate, "variance": variance, "se": se, "z": z, "p": p}

def step2_or(component, coefficient):
    if component == "Zero-inflated":
        return float(np.exp(coefficient))  # latest publication-facing direction
    if component == "Ordinal":
        return float(np.exp(coefficient))
    raise ValueError(component)

def tolerability_or(component, distressing_minus_tolerable):
    if component == "Zero-inflated":
        return float(np.exp(-distressing_minus_tolerable))
    if component == "Ordinal":
        return float(np.exp(distressing_minus_tolerable))
    raise ValueError(component)

FDR_FAMILIES = {
    "step2": "9 domains separately within each of the 2 model components",
    "step3": "9 global domain-by-time tests separately within model x component",
    "tolerability": "8 Distressing-vs-Tolerable contrasts separately within component",
    "binary_occurrence": "9 domain tests separately within each binary threshold",
    "binary_tolerability": "24 contrasts separately within each binary threshold",
    "hospital_step2": "9 domains separately within each component",
    "hospital_step3": "2 prespecified Gastrointestinal component tests in the hospital sensitivity",
}

