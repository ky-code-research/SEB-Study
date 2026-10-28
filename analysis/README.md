# Analysis source map

Run commands from the repository root with `analysis` on `PYTHONPATH` (PowerShell example: `$env:PYTHONPATH="$PWD\analysis"`).

## Public entry points

| Analysis | Entry point | Notes |
|---|---|---|
| Data construction | `prepare_authorized_data.py` | Writes the exact Step 2/3 restricted staging filenames and enforces source SHA, freeze SHA, mapping, sample, visit and outcome gates. |
| Step 2 | `step2_fourlevel_concurrent_hamd17_education3.py` | Adopted four-level model; same nine domains in both components. Optional `--bootstrap` runs the retained five-replicate numerical-stability QC, not formal bootstrap inference. |
| Step 3 Model 1 | `step3_independent_model1_with_concurrent_hamd17.py` | Concurrent HAMD-17 model. |
| Step 3 Model 2 | `step3_independent_model2_without_concurrent_hamd17.py` | Prespecified sensitivity without concurrent HAMD-17. |
| Covariance correction | `repair_step2_step3_robust_covariance.py` | Uses `V=A_inv @ B @ A_inv.T`; does not refit saved parameters. |
| Tolerability | `run_secondary_tolerability_esgee.py` + `tolerability_contrasts.py` | Reads the same prepared freeze as Step 2/3; eight domains; 48 rows from three full-covariance contrasts × two components; BH only for the two eight-test Distressing-vs-Tolerable families. The data-free helper can be tested without creating an output directory. |
| IPW | `reviewer1_comment3_attrition_ipw.py` | Formal 1,468/4,799 update, including untruncated primary IPW and 1st/99th percentile QC. |
| Hospital clustering | `hospital_clustered_gee_final.py` | Locked coefficients; hospital-aggregated score sandwich, CR1, approximate small-cluster inference. |
| Geographic/center FE | `run_site_geographic_sensitivity_2026_10_22.py` | Province FE sensitivity plus hospital-site FE identifiability gate; hospital-site FE estimates must remain suppressed when the gate fails. |
| Binary occurrence | `run_binary_logistic_gee_multivariable.py` | Python cross-implementation source. |
| Binary tolerability | `run_binary_logistic_gee_tolerability_s6_s7.py` | Python cross-implementation source. |
| Binary adopted R version | `../R/reproduce_binary_gee_geepack.R` | `geepack::geeglm`, exchangeable, robust `san.se`. |
| Optional R/Python comparison | `../R/compare_R_to_Python_binary_gee.R` | Reads completed R and Python result tables; does not fit models. |

## Dependency-only source files

Several retained files have historical names because the adopted scripts import their tested score, probability, design-matrix, start-generation, or sandwich helper functions. They are not alternative public analysis entry points: `model_comparison_gate_fibser.py`, `step2_final_primary_ZI_ordinal_GEE.py`, `step2_candidate_burden3_no_concurrent_HAMD.py`, `step2_burden3_no_concurrent_HAMD_4799_rerun.py`, `step2_fourlevel_no_concurrent_hamd17.py`, `step2_fourlevel_no_concurrent_hamd17_education3.py`, `step2B1_YangSimpson_LIM_ordinal_GEE.py`, `step2B3_fixed_sample_covariate_attenuation.py`, `step3_domain_time_burden_4799_preliminary.py`, `secondary_prise_tolerability_qc.py`, and `site_clustered_sandwich_gate.py`. The final IPW imports the aggregate-output-only `ipw_observation_helpers.py`; the superseded executable IPW dependency was removed.

They are included only to make the selected source chain executable and auditable. Early Gaussian GEE, rejected candidate-model runners, temporary diagnostics, plotting, and document/table-layout scripts are excluded.

## Expected generated inputs/outputs

Run `python analysis/prepare_authorized_data.py --raw-workbook <authorized.xlsx>` from the repository root. It creates the exact staged paths consumed by Step 2, Step 3, and Tolerability only after all locked gates pass. The entire `outputs/` tree is ignored by Git. Do not copy an existing patient-level output tree into a public clone.

