# PROUD antidepressant side-effect burden: reproducible analysis code

This repository contains the analysis code supporting the main and sensitivity analyses of longitudinal antidepressant side-effect burden in PROUD. It is a code-only release prepared for method review.

## Data are not included

PROUD patient-level data are access restricted. This repository contains no raw, de-identified, frozen, intermediate, cached, logged, or example patient-level records. The code can be inspected publicly and rerun after an authorized user supplies the required local data. Without authorized data, the repository is **not expected to reproduce every numerical result in the paper**.

The example generator creates data entirely from a random-number generator. It does not sample or perturb PROUD records and cannot reproduce the paper estimates.

Reconstructing the complete restricted analysis matrix from the original workbook requires the historical locked workbook version whose SHA-256 is encoded in `analysis/prepare_authorized_data.py`. That verification remains strict: a different workbook version stops before staging files are created. The public repository does not include that workbook, the frozen analysis matrix, or any other restricted patient-level data.

## Frozen analysis target

The locked main analysis used 1,468 patients and 4,799 person-visits (Week 2/8/12/24: 1,381/1,259/1,146/1,013). The four-level FIBSER outcome counts were 3,464/584/512/239 for 0/1/2/≥3. These gates are deliberately retained in the formal scripts so that a different dataset cannot silently be treated as the paper analysis.

## Repository contents

- `analysis/`: frozen Python source chain and small public audit helpers.
- `R/reproduce_binary_gee_geepack.R`: standalone final binary GEE implementation using R `geepack`.
- `R/compare_R_to_Python_binary_gee.R`: optional post-fit consistency comparison; not required to run the R models.
- `examples/`: fully synthetic generator and data-free structural smoke test.
- `variable_dictionary.md`: variables, PRISE domains, outcome coding, references, and FDR families.
- `METHODS_CODE_CROSSWALK.md`: latest-Methods-to-code mapping with explicit audit status.
- `R_requirements.md` and `requirements.txt`: verified runtime information.
- `config.example.env`: local path variables; never commit the populated file.

The publication-readiness report and source-path audit are maintained outside the public candidate in the separate internal audit package. No figure, table-layout, Word, Excel, or manuscript-rendering code is included.

## Analysis modules and run order

1. **Restricted data reconstruction**  
   From the repository root, run `python analysis/prepare_authorized_data.py --raw-workbook <authorized.xlsx>`. The command writes the exact restricted staging bundle under `outputs/PRISE_mapping_repair_data_freeze/`: the frozen CSV, `PRISE_Mapping_Verification.csv`, `data_version_record.json`, and preflight gates. It refuses source/freeze SHA-256 mismatch and refuses to overwrite an existing bundle.
2. **Step 2 main model**  
   `step2_fourlevel_concurrent_hamd17_education3.py` is the adopted four-level, two-component model with concurrent HAMD-17 and three-level education. The corrected covariance is recomputed by `repair_step2_step3_robust_covariance.py`.
3. **Step 3 time interaction models**  
   Run the Model 1 and Model 2 wrappers around `step3_independent_model_core.py`; global 3-df Wald tests and time-specific effects are produced before the corrected covariance pass.
4. **Tolerability**  
   `run_secondary_tolerability_esgee.py` reads the same prepared freeze at `outputs/PRISE_mapping_repair_data_freeze/repaired_natural_complete_case_analysis_data.csv`, fits eight domain models, and derives all 48 post-estimation rows—Tolerable vs None, Distressing vs None, and Distressing vs Tolerable in both components—from each selected parameter vector and complete robust covariance. The locked BH rule is unchanged: only the eight Distressing-vs-Tolerable contrasts are adjusted separately within each component; the other 32 rows report raw Wald P values.
5. **Binary GEE sensitivity**  
   Run `R/reproduce_binary_gee_geepack.R` with the authorized frozen CSV and raw workbook. It does not require historical Python results. If those results are separately available, run `R/compare_R_to_Python_binary_gee.R` afterward for the locked consistency comparison.
6. **IPW sensitivity**  
   `reviewer1_comment3_attrition_ipw.py` is the 4,799-visit update; the earlier 4,761-visit IPW output is not the formal result. Public outputs are aggregate only; participant identifiers remain transient in memory as join/cluster keys.
7. **Hospital clustering sensitivity**  
   `hospital_clustered_gee_final.py` recomputes the outer sandwich aggregation at hospital level without refitting locked coefficients. The result is exploratory/approximate because there are 21 unequal hospital clusters.
8. **Geographic/center fixed-effect sensitivity**  
   `run_site_geographic_sensitivity_2026_10_22.py` implements the locked hospital-site and province fixed-effect checks. Province-adjusted results were estimable; hospital-site fixed-effect results remain explicitly non-interpretable because of rank deficiency.

See [analysis/README.md](analysis/README.md) for exact commands and dependency-only files.

## Path configuration

Copy `config.example.env` to a local, untracked environment configuration or set variables in the shell:

- `PROUD_RAW_WORKBOOK`: authorized source workbook.
- `PROUD_FROZEN_CSV`: authorized 4,799-visit frozen matrix for the R analysis.
- `PROUD_REPO_ROOT`: optional repository-root override.
- `PROUD_R_OUTPUT_DIR`: R output directory.
- `PROUD_PYTHON_OCCURRENCE_DIR` and `PROUD_PYTHON_TOLERABILITY_DIR`: optional inputs used only by the separate R-versus-Python comparison script.

All data and outputs should remain outside version control. The included `.gitignore` blocks common data, output, cache, log, notebook, credential, and local configuration files. IPW outputs are written under `outputs/Reviewer1_Comment3_Attrition/`; the historical root-level directory is also explicitly ignored as a backward-compatible safeguard.

## Installation

Python:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

R packages and verified versions are documented in [R_requirements.md](R_requirements.md).

## Synthetic smoke test

```bash
python examples/smoke_test.py
python examples/generate_synthetic_data.py --output scratch/synthetic_analysis.csv
```

The smoke test checks domain counts/order, synthetic participant-visit uniqueness, frozen aggregate gates, BH adjustment, the corrected sandwich multiplication `A^{-1} B A^{-T}`, all six per-domain tolerability contrast rows, separation of R fitting from Python comparison, full-covariance algebra, and locked OR directions. It is not a numerical reproduction of the study.

## Important interpretation details

- Zero-inflated publication-facing Step 2/3 ORs use `exp(gamma)` in the latest direction-locked outputs.
- Ordinary burden-severity ORs use `exp(beta)`.
- All three tolerability contrasts use `c'V_robust c`. For Distressing vs Tolerable, the variance includes `-2Cov(D,T)`; zero-inflated ORs use `exp(-c'gamma)` and ordinal ORs use `exp(c'beta)`.
- BH families are not pooled across unrelated tables or thresholds; see the dictionary.
- The custom Step 2/3 and tolerability estimators use working independence with participant-cluster robust sandwich covariance; the binary `geepack` sensitivity uses exchangeable working correlation with participant clusters and robust `san.se`.
- Complete-case and attrition limitations remain. The fitted estimates are longitudinal associations, not causal effects.

## Reproducibility scope

The code release supports inspection of variable construction, model equations, robust covariance, contrasts, multiplicity families, clustering, and missing-data handling. Reproducing the exact paper values additionally requires access to the authorized frozen PROUD data and the locked software environments. The analysis code is publicly available at https://github.com/kaiyangan111-sys/PROUD-SideEffectBurden.


## License and attribution

This repository is released under the MIT License. See LICENSE for details.

The `fay_df_v1()` implementation in `analysis/site_clustered_sandwich_gate.py` is a Python algebraic reimplementation informed by the Fay–Graubard method and `geex::fay_df_correction(version=1)`.

The geex package (Copyright © 2017 Bradley Saul) is licensed under the MIT License. Its source code is not bundled with this repository.

- geex source: https://github.com/bsaul/geex
- geex license: https://github.com/bsaul/geex/blob/c9fc32ce456a5affbf783e5c7bb09a9f1fb17705/LICENSE.md

Third-party Python and R packages remain subject to their respective licenses.
