"""Observation-model helpers for the final attrition/IPW sensitivity analysis.

The statistical calculations are retained from the validated IPW source.  This
public helper intentionally exposes only aggregate diagnostics. Participant
identifiers remain in memory solely as longitudinal join/cluster keys and are
never written by this module.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

import step2B1_YangSimpson_LIM_ordinal_GEE as b1

WEEKS = b1.WEEKS
DOMAIN_VARS = [b1.DOMAIN_VAR[d] for d in b1.DOMAINS]
BASELINE_REQUIRED = [
    "age", "sex", "education", "first_episode", "baseline_HAMD17",
    "baseline_SDS", "baseline_med_class",
]


def add_history_features(grid: pd.DataFrame) -> pd.DataFrame:
    dat = grid.sort_values(["participant_ID", "time"]).copy()
    dat["prise_total"] = dat[DOMAIN_VARS].sum(axis=1, min_count=len(DOMAIN_VARS))
    for column in ["burden_ord4", "concurrent_HAMD17", "prise_total", *DOMAIN_VARS]:
        history = dat.groupby("participant_ID", observed=False)[column].ffill()
        dat[f"hist_last_{column}"] = history.groupby(dat["participant_ID"], observed=False).shift(1)
        dat[f"hist_last_{column}_missing"] = dat[f"hist_last_{column}"].isna().astype(int)
        dat[f"hist_last_{column}"] = pd.to_numeric(
            dat[f"hist_last_{column}"], errors="coerce"
        ).fillna(0)
    dat["prior_visit_observed"] = (
        dat.groupby("participant_ID", observed=False)["R_it"].shift(1).fillna(0)
    )
    dat["prior_visit_observed_missing"] = (
        dat.groupby("participant_ID", observed=False).cumcount().eq(0).astype(int)
    )
    return dat


def design_matrix_for_visit(
    dat: pd.DataFrame, week: str, include_province: bool = True
) -> pd.DataFrame:
    sub = dat[dat["time"].astype(str).eq(week)].copy()
    design = pd.DataFrame(index=sub.index)
    design["age"] = pd.to_numeric(sub["age"], errors="coerce")
    design["baseline_HAMD17"] = pd.to_numeric(sub["baseline_HAMD17"], errors="coerce")
    design["baseline_SDS"] = pd.to_numeric(sub["baseline_SDS"], errors="coerce")
    design["baseline_fibser_burden"] = pd.to_numeric(
        sub["baseline_fibser_burden"], errors="coerce"
    )
    design["baseline_fibser_burden_missing"] = design["baseline_fibser_burden"].isna().astype(int)
    design["baseline_fibser_burden"] = design["baseline_fibser_burden"].fillna(0)
    categorical = ["sex", "education", "first_episode", "baseline_med_class"]
    if include_province:
        categorical.append("province")
    indicators = pd.get_dummies(
        sub[categorical].astype("object").fillna("Missing"), drop_first=True, dtype=float
    )
    design = pd.concat([design, indicators], axis=1)
    if week != "Week2":
        sub["any_prior_history_missing"] = (
            sub["hist_last_burden_ord4_missing"].astype(bool)
            | sub["hist_last_concurrent_HAMD17_missing"].astype(bool)
            | sub["hist_last_prise_total_missing"].astype(bool)
        ).astype(int)
        history_columns = [
            "prior_visit_observed", "prior_visit_observed_missing",
            "hist_last_burden_ord4", "hist_last_concurrent_HAMD17",
            "hist_last_prise_total", "any_prior_history_missing",
        ]
        design = pd.concat([design, sub[history_columns].astype(float)], axis=1)
    design = design.loc[:, design.nunique(dropna=False) > 1]
    return sm.add_constant(design, has_constant="add").astype(float)


def fit_observation_models(
    grid: pd.DataFrame, formal_include_province: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, str, pd.DataFrame]:
    """Fit the locked visit-specific denominator models.

    The third return value is an aggregate probability summary, replacing the
    historical participant-level prediction table. It is not used in fitting.
    """
    dat = add_history_features(grid[grid["baseline_eligible"]].copy())
    coefficient_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    probability_rows: list[dict] = []
    calibration_rows: list[dict] = []
    comparator_rows: list[dict] = []
    stop = False
    stop_reason = ""
    for week in WEEKS:
        sub = dat[dat["time"].astype(str).eq(week)].copy()
        outcome = sub["R_it"].astype(float)
        for include_province, label in [
            (True, "province_auxiliary_audit"),
            (False, "formal_no_province"),
        ]:
            design = design_matrix_for_visit(dat, week, include_province=include_province)
            separation_warning = False
            converged = False
            message = ""
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", PerfectSeparationWarning)
                try:
                    fit = sm.GLM(outcome, design, family=sm.families.Binomial()).fit(maxiter=200)
                    converged = bool(getattr(fit, "converged", False))
                    message = "fit completed"
                    separation_warning = any(
                        issubclass(item.category, PerfectSeparationWarning) for item in caught
                    )
                    predicted = np.asarray(fit.predict(design), dtype=float)
                    parameters, standard_errors, p_values = fit.params, fit.bse, fit.pvalues
                except Exception as exc:
                    message = str(exc)
                    predicted = np.full(len(outcome), np.nan)
                    parameters = standard_errors = p_values = pd.Series(dtype=float)
            max_abs = float(np.nanmax(np.abs(parameters.values))) if len(parameters) else np.nan
            structural_terms = [
                term for term in parameters.index if term.endswith("_missing") or term == "const"
            ]
            max_structural = (
                float(np.nanmax(np.abs(parameters.reindex(structural_terms).dropna().values)))
                if structural_terms else np.nan
            )
            province_terms = [term for term in parameters.index if term.startswith("province_")]
            max_province = (
                float(np.nanmax(np.abs(parameters[province_terms].values)))
                if province_terms else np.nan
            )
            province_unstable = include_province and np.isfinite(max_province) and max_province > 25
            unstable = (
                (not converged) or separation_warning
                or (np.isfinite(max_structural) and max_structural > 20)
                or (np.nanmin(predicted) < 1e-5) or province_unstable
            )
            is_formal = include_province == formal_include_province
            target = diagnostic_rows if is_formal else comparator_rows
            target.append({
                "time": week, "model": label, "n": len(outcome),
                "observed": int(outcome.sum()), "observed_rate": float(outcome.mean()),
                "converged": converged, "message": message,
                "separation_warning": bool(separation_warning),
                "max_abs_coefficient": max_abs,
                "max_abs_structural_missing_coefficient": max_structural,
                "max_abs_province_coefficient": max_province,
                "min_predicted_probability": float(np.nanmin(predicted)),
                "median_predicted_probability": float(np.nanmedian(predicted)),
                "max_predicted_probability": float(np.nanmax(predicted)),
                "mean_predicted_probability": float(np.nanmean(predicted)),
                "unstable_flag": bool(unstable),
            })
            if is_formal:
                for term in parameters.index:
                    coefficient_rows.append({
                        "time": week, "model": label, "term": term,
                        "coefficient": float(parameters[term]),
                        "OR": float(np.exp(np.clip(parameters[term], -30, 30))),
                        "SE_model_based": float(standard_errors[term]),
                        "P_value_descriptive": float(p_values[term]),
                    })
                ranks = pd.Series(predicted).rank(method="first")
                bins = pd.qcut(ranks, q=min(5, len(predicted)), labels=False, duplicates="drop")
                for bin_number in sorted(pd.Series(bins).dropna().unique()):
                    selected = bins == bin_number
                    calibration_rows.append({
                        "time": week, "calibration_bin": int(bin_number) + 1,
                        "n": int(selected.sum()),
                        "mean_predicted": float(np.nanmean(predicted[selected])),
                        "observed_rate": float(outcome.to_numpy()[selected].mean()),
                    })
                probability_rows.append({
                    "time": week, "n": len(predicted),
                    "mean_predicted": float(np.nanmean(predicted)),
                    "observed_rate": float(outcome.mean()),
                })
                dat.loc[sub.index, "denominator_p"] = np.clip(predicted, 1e-6, 1 - 1e-6)
                dat.loc[sub.index, "numerator_p"] = float(outcome.mean())
                if unstable:
                    stop = True
                    stop_reason += f"{week} primary observation model unstable; "
    return (
        pd.DataFrame(diagnostic_rows), pd.DataFrame(coefficient_rows),
        pd.DataFrame(probability_rows), pd.DataFrame(calibration_rows),
        pd.DataFrame(comparator_rows), stop, stop_reason.strip(), dat,
    )


def weight_diagnostics(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return distribution diagnostics, in-memory weights, and aggregate concentration QC."""
    observed = weights[weights["R_it"].eq(1)].copy()
    observed["SW"] = observed["numerator_p"] / observed["denominator_p"]
    lower, upper = observed["SW"].quantile([0.01, 0.99])
    observed["SW_truncated_1_99"] = observed["SW"].clip(lower, upper)
    rows: list[dict] = []
    groups = [("overall", observed)] + [
        (week, observed[observed["time"].astype(str).eq(week)]) for week in WEEKS
    ]
    for group_name, subset in groups:
        for column in ["SW", "SW_truncated_1_99"]:
            values = subset[column].dropna().astype(float)
            squared_sum = float(np.sum(values ** 2))
            rows.append({
                "group": group_name, "weight": column, "n": len(values),
                "min": values.min(), "p1": values.quantile(0.01),
                "p5": values.quantile(0.05), "median": values.median(),
                "p95": values.quantile(0.95), "p99": values.quantile(0.99),
                "max": values.max(), "mean": values.mean(),
                "proportion_gt5": float((values > 5).mean()),
                "proportion_gt10": float((values > 10).mean()),
                "ESS": float((values.sum() ** 2) / squared_sum) if squared_sum > 0 else np.nan,
            })
    by_participant = observed.groupby("participant_ID", observed=True).agg(
        complete_visits=("SW", "size"), sum_SW=("SW", "sum"),
        max_SW=("SW", "max"), mean_SW=("SW", "mean"),
    )
    concentration = pd.DataFrame([{
        "participants": int(len(by_participant)),
        "median_complete_visits": float(by_participant["complete_visits"].median()),
        "max_sum_SW": float(by_participant["sum_SW"].max()),
        "max_single_visit_SW": float(by_participant["max_SW"].max()),
        "median_participant_mean_SW": float(by_participant["mean_SW"].median()),
    }])
    return pd.DataFrame(rows), observed, concentration
