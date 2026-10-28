#!/usr/bin/env Rscript

# Optional post hoc consistency check. Run only after the standalone R model
# and the historical/formal Python result tables are both available.
suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
})

options(stringsAsFactors = FALSE, scipen = 999)
script_arg <- commandArgs(trailingOnly = FALSE)
script_flag <- grep("^--file=", script_arg, value = TRUE)
script_path <- if (length(script_flag)) sub("^--file=", "", script_flag[[1]]) else normalizePath(".")
script_dir <- normalizePath(dirname(script_path), winslash = "/", mustWork = TRUE)
root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = TRUE)

r_dir <- Sys.getenv("PROUD_R_OUTPUT_DIR", file.path(root, "outputs", "r_geepack"))
py_occ_dir <- Sys.getenv(
  "PROUD_PYTHON_OCCURRENCE_DIR",
  file.path(root, "outputs", "binary_logistic_GEE_multivariable_S4_S5_2026-10-18")
)
py_tol_dir <- Sys.getenv(
  "PROUD_PYTHON_TOLERABILITY_DIR",
  file.path(root, "outputs", "binary_logistic_GEE_tolerability_S6_S7_2026-10-19_final")
)
out_dir <- Sys.getenv("PROUD_R_PYTHON_COMPARISON_DIR", file.path(r_dir, "python_comparison"))

files <- list(
  r_occ = file.path(r_dir, "R_domain_occurrence_results.csv"),
  r_tol = file.path(r_dir, "R_tolerability_results.csv"),
  r_diag = file.path(r_dir, "R_model_diagnostics.csv"),
  py_occ = file.path(py_occ_dir, "Multivariable_Domain_Results.csv"),
  py_occ_summary = file.path(py_occ_dir, "Model_Summary.csv"),
  py_s6 = file.path(py_tol_dir, "S6_results.csv"),
  py_s7 = file.path(py_tol_dir, "S7_results.csv"),
  py_tol_summary = file.path(py_tol_dir, "Model_Summary.csv")
)
missing <- names(files)[!file.exists(unlist(files))]
if (length(missing)) {
  stop("Required comparison input(s) missing: ",
       paste(paste0(missing, "=", unlist(files[missing])), collapse = "; "))
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

r_occ <- read.csv(files$r_occ, check.names = FALSE)
r_tol <- read.csv(files$r_tol, check.names = FALSE)
r_diag <- read.csv(files$r_diag, check.names = FALSE)
py_occ_raw <- read.csv(files$py_occ, check.names = FALSE)
py_occ_summary <- read.csv(files$py_occ_summary, check.names = FALSE)
py_tol_summary <- read.csv(files$py_tol_summary, check.names = FALSE)

py_occ <- py_occ_raw %>%
  transmute(
    Outcome = ifelse(grepl("Any burden", Outcome),
                     "Any burden (FIBSER 1-6)", "Higher burden (FIBSER 3-6)"),
    Domain, Contrast = "Present vs Absent",
    Python_Coefficient = Coefficient, Python_Robust_SE = `Robust SE`,
    Python_OR = `Adjusted OR`, Python_CI_low = `CI low`, Python_CI_high = `CI high`,
    Python_P_raw = `Raw P`, Python_FDR_P = `FDR-adjusted P (9-test family)`
  ) %>%
  left_join(
    py_occ_summary %>% transmute(
      Outcome = ifelse(grepl("Any burden", Outcome),
                       "Any burden (FIBSER 1-6)", "Higher burden (FIBSER 3-6)"),
      Python_Participants = Participants, Python_Person_visits = `Person-visits`,
      Python_Events = Events, Python_Converged = Converged,
      Python_Working_correlation = `Working correlation`
    ), by = "Outcome"
  )

py_tol <- bind_rows(
  read.csv(files$py_s6, check.names = FALSE),
  read.csv(files$py_s7, check.names = FALSE)
) %>%
  transmute(
    Table,
    Outcome = ifelse(Table == "S6", "Any burden (FIBSER 1-6)", "Higher burden (FIBSER 3-6)"),
    Domain = `PRISE domain`, Contrast,
    Python_Coefficient = `Coefficient contrast`, Python_Robust_SE = `Robust SE`,
    Python_OR = `Adjusted OR`, Python_CI_low = `CI low`, Python_CI_high = `CI high`,
    Python_P_raw = `P value`, Python_FDR_P = `FDR-adjusted P value`
  ) %>%
  left_join(
    py_tol_summary %>% transmute(
      Table, Domain, Python_Participants = Participants,
      Python_Person_visits = `Person-visits`, Python_Events = Events,
      Python_Converged = Converged,
      Python_Working_correlation = `Working correlation`
    ), by = c("Table", "Domain")
  )

compare_rows <- function(r_rows, py_rows, keys) {
  r_rows %>%
    transmute(
      across(all_of(keys)), R_Coefficient = Coefficient, R_Robust_SE = Robust_SE,
      R_OR = OR, R_CI_low = CI_low, R_CI_high = CI_high,
      R_P_raw = P_raw, R_FDR_P = FDR_P,
      R_Participants = Participants, R_Person_visits = Person_visits, R_Events = Events,
      R_Raw_significant = Raw_significant, R_FDR_significant = FDR_significant
    ) %>%
    left_join(py_rows, by = keys) %>%
    mutate(
      Coefficient_abs_diff = abs(R_Coefficient - Python_Coefficient),
      Robust_SE_abs_diff = abs(R_Robust_SE - Python_Robust_SE),
      OR_abs_diff = abs(R_OR - Python_OR),
      CI_low_abs_diff = abs(R_CI_low - Python_CI_low),
      CI_high_abs_diff = abs(R_CI_high - Python_CI_high),
      P_raw_abs_diff = abs(R_P_raw - Python_P_raw),
      FDR_P_abs_diff = abs(R_FDR_P - Python_FDR_P),
      Participant_match = R_Participants == Python_Participants,
      Visit_match = R_Person_visits == Python_Person_visits,
      Event_match = R_Events == Python_Events,
      Direction_match = sign(R_Coefficient) == sign(Python_Coefficient),
      Raw_significance_match = R_Raw_significant == (Python_P_raw < 0.05),
      FDR_significance_match = R_FDR_significant == (Python_FDR_P < 0.05),
      Strict_numeric_match_1e_4 = Coefficient_abs_diff <= 1e-4 &
        Robust_SE_abs_diff <= 1e-4 & P_raw_abs_diff <= 1e-4 & FDR_P_abs_diff <= 1e-4,
      Cross_implementation_numeric_match = Coefficient_abs_diff <= 0.005 &
        Robust_SE_abs_diff <= 0.002 & P_raw_abs_diff <= 0.01 & FDR_P_abs_diff <= 0.02,
      Status = ifelse(
        Participant_match & Visit_match & Event_match & Direction_match &
          Raw_significance_match & FDR_significance_match & Cross_implementation_numeric_match,
        "PASS", "REVIEW"
      )
    )
}

occ_compare <- compare_rows(r_occ, py_occ, c("Outcome", "Domain", "Contrast")) %>%
  mutate(Analysis = "Domain occurrence", .before = 1)
tol_compare <- compare_rows(r_tol, py_tol, c("Table", "Outcome", "Domain", "Contrast")) %>%
  mutate(Analysis = "Tolerability", .before = 1)
comparison <- bind_rows(occ_compare, tol_compare)

py_model_diag <- bind_rows(
  py_occ_summary %>% transmute(
    Analysis = "Domain occurrence",
    Table = ifelse(grepl("Any burden", Outcome), "S4", "S5"),
    Domain = "All nine simultaneous",
    Python_Participants = Participants, Python_Person_visits = `Person-visits`,
    Python_Events = Events, Python_Converged = Converged,
    Python_Working_correlation = `Working correlation`,
    Python_Design_columns = `Design columns`, Python_Design_rank = `Design rank`
  ),
  py_tol_summary %>% transmute(
    Analysis = "Tolerability", Table, Domain,
    Python_Participants = Participants, Python_Person_visits = `Person-visits`,
    Python_Events = Events, Python_Converged = Converged,
    Python_Working_correlation = `Working correlation`,
    Python_Design_columns = `Design columns`, Python_Design_rank = `Design rank`
  )
)
model_comparison <- r_diag %>%
  left_join(py_model_diag, by = c("Analysis", "Table", "Domain")) %>%
  mutate(
    Working_correlation_abs_diff = abs(Working_correlation - Python_Working_correlation),
    Count_match = Participants == Python_Participants &
      Person_visits == Python_Person_visits & Events == Python_Events,
    Convergence_match = Converged == (tolower(as.character(Python_Converged)) == "true"),
    Design_dimension_match = Design_columns == Python_Design_columns &
      Design_rank == Python_Design_rank,
    Status = ifelse(
      Count_match & Convergence_match & Design_dimension_match & Full_rank &
        Robust_covariance_finite & Robust_covariance_symmetric &
        Robust_covariance_min_eigenvalue >= -1e-8 & Working_correlation_abs_diff <= 0.01,
      "PASS", "REVIEW"
    )
  )

write.csv(comparison, file.path(out_dir, "Python_vs_R_comparison.csv"),
          row.names = FALSE, na = "", fileEncoding = "UTF-8")
write.csv(model_comparison, file.path(out_dir, "Python_vs_R_model_diagnostics.csv"),
          row.names = FALSE, na = "", fileEncoding = "UTF-8")

summary_obj <- list(
  generated = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  comparison_rows = nrow(comparison),
  pass = sum(comparison$Status == "PASS"),
  review = sum(comparison$Status != "PASS"),
  strict_numeric_match_1e_4 = sum(comparison$Strict_numeric_match_1e_4),
  raw_significance_agreement = sum(comparison$Raw_significance_match),
  fdr_significance_agreement = sum(comparison$FDR_significance_match),
  max_abs_differences = list(
    coefficient = max(comparison$Coefficient_abs_diff, na.rm = TRUE),
    robust_se = max(comparison$Robust_SE_abs_diff, na.rm = TRUE),
    raw_p = max(comparison$P_raw_abs_diff, na.rm = TRUE),
    fdr_p = max(comparison$FDR_P_abs_diff, na.rm = TRUE)
  )
)
write_json(summary_obj, file.path(out_dir, "comparison_summary.json"),
           pretty = TRUE, auto_unbox = TRUE, na = "null")

report <- c(
  "# R versus Python binary GEE consistency comparison",
  "",
  paste0("- Generated: ", summary_obj$generated),
  paste0("- Rows: ", nrow(comparison), "; PASS: ", summary_obj$pass,
         "; REVIEW: ", summary_obj$review, "."),
  paste0("- Strict all-metric numeric match at 1e-4: ",
         summary_obj$strict_numeric_match_1e_4, "/", nrow(comparison), "."),
  paste0("- Raw P<0.05 agreement: ", summary_obj$raw_significance_agreement,
         "/", nrow(comparison), "."),
  paste0("- FDR P<0.05 agreement: ", summary_obj$fdr_significance_agreement,
         "/", nrow(comparison), "."),
  "",
  "This optional script does not fit any model. It compares already generated standalone R results with separately supplied Python results using the locked tolerances and decision rules."
)
writeLines(report, file.path(out_dir, "R_Python_consistency_report.md"), useBytes = TRUE)
cat(toJSON(list(output_dir = out_dir, rows = nrow(comparison), pass = summary_obj$pass,
                    review = summary_obj$review), auto_unbox = TRUE), "\n")
