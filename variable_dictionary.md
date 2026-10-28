# Variable dictionary and locked coding

## Keys and time

| Variable | Role | Coding/reference |
|---|---|---|
| `participant_ID` | Patient cluster key | Restricted identifier; never public. |
| `time` | Visit | Week2 (reference), Week8, Week12, Week24. |
| `visit_order` | Ordering | 1, 2, 3, 4. |
| `site` | Hospital | 21 sites in the frozen data; excluded from primary Step 2/3 fixed effects, used for center sensitivity. |

## Outcomes

| Variable | Construction |
|---|---|
| `fibser_burden` | Raw FIBSER burden, valid values 0–6. |
| `burden_ord4` | 0→0, 1→1, 2→2, 3–6→3. |
| `any_burden` | 1 if FIBSER ≥1, otherwise 0. |
| `higher_burden` | 1 if FIBSER ≥3, otherwise 0. |

## PRISE occurrence domains

The locked order is: Gastrointestinal, Cardiac (paper label Heart), Skin, Nervous system, Eye/Ear, Genital/Urinary, Sleep, Sexual (paper label Sexual functioning), Other psychiatric (paper label Other symptom).

| Paper label | Analysis variable | Source occurrence coding |
|---|---|---|
| Gastrointestinal | `Gastrointestinal` | raw 1=present; raw 2=absent; source 0 retained as absent; blank=missing |
| Heart | `Cardiac` | same |
| Skin | `Skin` | same |
| Nervous system | `Nervous` | same |
| Eye/Ear | `EyeEar` | same |
| Genital/Urinary | `Genitourinary` | same |
| Sleep | `Sleep` | same |
| Sexual functioning | `Sexual` | same |
| Other symptom | `OtherPsych` | same |

Source columns are located by questionnaire code first and visible PRISE domain label second, with no positional fallback. The eight tolerability models exclude Other psychiatric.

## Tolerability

For each target domain:

- None: occurrence=absent and tolerability blank.
- Tolerable: occurrence=present and tolerability code 1.
- Distressing: occurrence=present and tolerability code 2.
- Contradictory, invalid, or unclassified rows are missing for that domain-specific tolerability model only.

Reference is None. The two-component tolerability analysis reports Tolerable vs None, Distressing vs None, and Distressing vs Tolerable from the same fitted model and complete robust covariance.

## Covariates and references

| Variable | Coding/reference |
|---|---|
| `age_z` | Population-SD standardization (`ddof=0`) before complete-case restriction. |
| `sex` | Categorical; Female reference in final binary R validation. |
| `education3` | Junior high or below; High school/vocational; College or above (reference). |
| `first_episode` | Categorical; FirstEpisode reference in final binary R validation. |
| `baseline_HAMD17_z` | Population-SD standardized baseline HAMD-17. |
| `baseline_SDS_z` | Population-SD standardized baseline SDS. |
| `baseline_med_class` | Categorical; SNRI reference in final binary R validation. |
| `concurrent_HAMD17_z` | Visit-specific HAMD-17, standardized using the locked rule; included in primary Model 1. |

## Natural complete-case rule

A visit enters the main frozen analysis only when FIBSER burden is valid (0–6) and time, all nine PRISE occurrence domains, age, sex, education, first-episode status, baseline HAMD-17, baseline SDS, baseline antidepressant class, and concurrent HAMD-17 are nonmissing. This produced 1,468 patients and 4,799 visits. Tolerability models then apply domain-specific exclusion only for unclassified target-domain tolerability.

## Two-component model and OR direction

The mixing component models ordinary-component membership; the latest publication-facing Step 2/3 zero-inflated OR is `exp(gamma)`. The ordinary ordinal component uses `logit[P(Y≤k)] = alpha_k - X beta`; `exp(beta)>1` indicates higher burden.

For every tolerability contrast, zero-inflated OR is `exp(-c'gamma)` and ordinal OR is `exp(c'beta)`. The contrast variance is `c' V_robust c`; for D−T it is `Var(D)+Var(T)-2Cov(D,T)`.

## BH-FDR families

| Module | Formal family |
|---|---|
| Step 2 | Nine domain tests separately within each component. |
| Step 3 | Nine global 3-df domain×time tests separately within each model × component. |
| Tolerability two-component | Three contrasts are estimated per domain/component. Only the eight Distressing-vs-Tolerable tests are BH-adjusted separately within each component; Tolerable-vs-None and Distressing-vs-None are exploratory raw-P contrasts. |
| Binary occurrence | Nine domain tests separately within each threshold (S4 and S5). |
| Binary tolerability | 24 contrasts separately within each threshold (S6 and S7). |
| Hospital Step 2 | Nine domains separately within each component. |
| Hospital Step 3 | Two prespecified Gastrointestinal component tests; the original nine-domain FDR is not reused. |

