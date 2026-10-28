#!/usr/bin/env Rscript

# Independent R/geepack reproduction of the frozen binary GEE sensitivity analyses.
# This script reads the frozen 4,799-visit analysis CSV, reconstructs target-domain
# tolerability directly from the frozen raw Excel source, fits the prespecified GEE
# models, and writes standalone R results and diagnostics. Python-result
# consistency checking is intentionally separated into compare_R_to_Python_binary_gee.R.

suppressPackageStartupMessages({
  library(geepack)
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(tibble)
  library(jsonlite)
  library(digest)
})

options(stringsAsFactors = FALSE, scipen = 999)
invisible(Sys.setlocale("LC_ALL", "Chinese (Simplified)_China.utf8"))

script_arg <- commandArgs(trailingOnly = FALSE)
script_flag <- grep("^--file=", script_arg, value = TRUE)
script_path <- if (length(script_flag)) sub("^--file=", "", script_flag[[1]]) else normalizePath(".")
script_dir <- normalizePath(dirname(script_path), winslash = "/", mustWork = TRUE)
root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = TRUE)
out_dir <- Sys.getenv("PROUD_R_OUTPUT_DIR", file.path(root, "outputs", "r_geepack"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

frozen_csv <- Sys.getenv("PROUD_FROZEN_CSV", file.path(root, "data", "frozen_analysis.csv"))
raw_xlsx <- Sys.getenv("PROUD_RAW_WORKBOOK", file.path(root, "data", "raw_workbook.xlsx"))
required_files <- c(frozen_csv, raw_xlsx)
if (!all(file.exists(required_files))) {
  stop("Required input(s) missing: ", paste(required_files[!file.exists(required_files)], collapse = "; "))
}

expected <- list(
  participants = 1468L,
  visits = 4799L,
  week_counts = c(Week2 = 1381L, Week8 = 1259L, Week12 = 1146L, Week24 = 1013L),
  any_events = 1335L,
  higher_events = 239L,
  python_run_raw_sha256 = "876b1e3f1b55574984bd105336c1e706d4fad7eaaa7626e62318732f69324f84"
)

domain_display <- c(
  Sleep = "Sleep",
  Gastrointestinal = "Gastrointestinal",
  OtherPsych = "Other symptom",
  Nervous = "Nervous system",
  Cardiac = "Heart",
  EyeEar = "Eye/Ear",
  Skin = "Skin",
  Sexual = "Sexual functioning",
  Genitourinary = "Genital/Urinary"
)
domain_vars <- names(domain_display)
target_vars <- c("Sleep", "Gastrointestinal", "Nervous", "Cardiac", "EyeEar", "Skin", "Sexual", "Genitourinary")
target_display <- unname(domain_display[target_vars])
weeks <- c("Week2", "Week8", "Week12", "Week24")

to_num <- function(x) {
  y <- trimws(as.character(x))
  y[y == "" | tolower(y) %in% c("nan", "na", "n/a", "none", "null")] <- NA_character_
  direct <- suppressWarnings(as.numeric(y))
  extracted <- suppressWarnings(as.numeric(str_match(y, "^\\s*\\[?(-?\\d+(?:\\.\\d+)?)\\]?.*$")[, 2]))
  ifelse(!is.na(direct), direct, extracted)
}

norm_id <- function(x) {
  y <- trimws(as.character(x))
  y[y == "" | tolower(y) %in% c("nan", "na", "n/a", "none", "null")] <- NA_character_
  sub("\\.0$", "", y)
}

first_nonmissing <- function(x) {
  z <- x[!is.na(x)]
  if (length(z)) z[[1]] else NA_real_
}

safe_write_csv <- function(x, name) {
  write.csv(x, file.path(out_dir, name), row.names = FALSE, na = "", fileEncoding = "UTF-8")
}

assert_equal <- function(observed, expected_value, label) {
  if (!identical(as.character(observed), as.character(expected_value))) {
    stop(label, " mismatch: observed=", paste(observed, collapse = ","),
         "; expected=", paste(expected_value, collapse = ","))
  }
}

raw_sha256 <- digest(raw_xlsx, algo = "sha256", file = TRUE)
raw_hash_matches_python_run <- identical(raw_sha256, expected$python_run_raw_sha256)

frozen <- read.csv(frozen_csv, check.names = FALSE, colClasses = c(participant_ID = "character"))
frozen$participant_ID <- norm_id(frozen$participant_ID)
frozen$time <- factor(frozen$time, levels = weeks, ordered = FALSE)
frozen$any_burden <- as.integer(frozen$fibser_burden >= 1)
frozen$higher_burden <- as.integer(frozen$fibser_burden >= 3)
frozen$education3 <- case_when(
  frozen$education %in% c("小学", "初中") ~ "Junior high or below",
  frozen$education == "高中/中专/职高" ~ "High school/vocational",
  frozen$education %in% c("大学/大专", "研究生及以上") ~ "College or above",
  TRUE ~ NA_character_
)

# Explicit reference levels matching the locked patsy/statsmodels formulas.
frozen$sex <- factor(frozen$sex, levels = c("Female", sort(setdiff(unique(frozen$sex), "Female"))))
frozen$education3 <- factor(frozen$education3,
                            levels = c("College or above", "Junior high or below", "High school/vocational"))
frozen$first_episode <- factor(frozen$first_episode,
                               levels = c("FirstEpisode", sort(setdiff(unique(frozen$first_episode), "FirstEpisode"))))
frozen$baseline_med_class <- factor(frozen$baseline_med_class,
                                    levels = c("SNRI", sort(setdiff(unique(frozen$baseline_med_class), "SNRI"))))

week_tab <- table(frozen$time)
assert_equal(nrow(frozen), expected$visits, "Frozen visit count")
assert_equal(length(unique(frozen$participant_ID)), expected$participants, "Frozen participant count")
assert_equal(as.integer(week_tab[weeks]), unname(expected$week_counts), "Frozen visit distribution")
assert_equal(sum(frozen$any_burden), expected$any_events, "Any-burden event count")
assert_equal(sum(frozen$higher_burden), expected$higher_events, "Higher-burden event count")
assert_equal(sum(duplicated(frozen[c("participant_ID", "time")])), 0L, "Duplicate patient-visit key count")
if (anyNA(frozen[c(domain_vars, "age_z", "sex", "education3", "first_episode", "baseline_HAMD17_z",
                          "baseline_SDS_z", "baseline_med_class", "concurrent_HAMD17_z")])) {
  stop("Frozen complete-case model variables unexpectedly contain missing values")
}
if (!all(vapply(frozen[domain_vars], function(x) all(x %in% c(0, 1)), logical(1)))) {
  stop("At least one frozen PRISE occurrence variable is not binary 0/1")
}

# Directly reconstruct the eight target-domain tolerability variables from the raw Excel workbook.
sheet_pairs <- list(
  Week2 = c("1448-2周末", "2周缺少副作用"),
  Week8 = c("1386-8周末", "8周缺少副作用"),
  Week12 = c("1214-12周末", "12周缺少副作用"),
  Week24 = c("1086-24周末", "24周缺少副作用")
)
domain_regex <- c(
  Gastrointestinal = "胃肠", Cardiac = "心脏", Skin = "皮肤", Nervous = "神经系统",
  EyeEar = "眼/耳|眼耳", Genitourinary = "生殖/泌尿|生殖泌尿", Sleep = "睡眠",
  Sexual = "性功能", OtherPsych = "其他精神症状"
)

locate_occurrence <- function(nms, variable) {
  idx <- grep(paste0("^[0-9]+\\..*(", domain_regex[[variable]], ")"), nms, perl = TRUE)
  prise_anchor <- grep("PRISE", nms, fixed = TRUE)
  if (length(prise_anchor)) idx <- idx[idx > prise_anchor[[1]]]
  if (length(idx) != 1L) stop("Occurrence mapping conflict for ", variable, ": ", paste(nms[idx], collapse = " | "))
  idx
}

raw_frames <- list()
mapping_rows <- list()
sheet_key_rows <- list()
frame_i <- 0L
map_i <- 0L
for (visit_index in seq_along(sheet_pairs)) {
  week <- names(sheet_pairs)[[visit_index]]
  for (priority in seq_along(sheet_pairs[[week]])) {
    sheet <- sheet_pairs[[week]][[priority]]
    dat <- suppressMessages(read_excel(raw_xlsx, sheet = sheet, .name_repair = "unique"))
    nms <- names(dat)
    id_idx <- which(startsWith(nms, "ID") | grepl("入组编码", nms))
    if (!length(id_idx)) id_idx <- 1L
    id_idx <- id_idx[[1]]
    occurrence_idx_all <- sort(vapply(names(domain_regex), function(v) locate_occurrence(nms, v), integer(1)))
    frame_i <- frame_i + 1L
    out <- tibble(
      participant_ID = norm_id(dat[[id_idx]]),
      time = week,
      visit_order = visit_index,
      source_priority = priority,
      source_row = seq_len(nrow(dat)) + 1L
    )
    dup_n <- sum(duplicated(out[c("participant_ID", "time")]) & !is.na(out$participant_ID))
    sheet_key_rows[[frame_i]] <- tibble(sheet = sheet, week = week, duplicate_keys = dup_n,
                                        status = ifelse(dup_n == 0L, "PASS", "REVIEW"))
    for (variable in target_vars) {
      occ_idx <- locate_occurrence(nms, variable)
      next_idx <- occurrence_idx_all[occurrence_idx_all > occ_idx]
      stop_idx <- if (length(next_idx)) min(next_idx) - 1L else length(nms)
      candidate_idx <- seq.int(occ_idx + 1L, stop_idx)
      candidate_text <- gsub("\\s+|<br>", "", nms[candidate_idx])
      tol_hit <- candidate_idx[grepl("严重程度", candidate_text) &
                                 grepl("1=可以忍受|1=可忍受", candidate_text) &
                                 grepl("2=非常痛苦", candidate_text)]
      if (length(tol_hit) != 1L) {
        stop("Tolerability mapping conflict for ", sheet, "/", variable, ": ",
             paste(nms[tol_hit], collapse = " | "))
      }
      occ <- to_num(dat[[occ_idx]])
      tol <- to_num(dat[[tol_hit]])
      if (!all(na.omit(unique(occ)) %in% c(1, 2)) || !all(na.omit(unique(tol)) %in% c(1, 2))) {
        stop("Unexpected tolerability source codes for ", sheet, "/", variable)
      }
      out[[paste0(variable, "__occ_raw")]] <- occ
      out[[paste0(variable, "__tol_raw")]] <- tol
      map_i <- map_i + 1L
      mapping_rows[[map_i]] <- tibble(
        sheet = sheet, week = week, domain = unname(domain_display[[variable]]),
        occurrence_column = nms[[occ_idx]], tolerability_column = nms[[tol_hit]],
        occurrence_excel_column = occ_idx, tolerability_excel_column = tol_hit,
        status = "PASS"
      )
    }
    raw_frames[[frame_i]] <- out
  }
}

raw_tol <- bind_rows(raw_frames) %>%
  filter(!is.na(participant_ID)) %>%
  arrange(visit_order, source_priority, source_row)
value_cols <- grep("__(occ|tol)_raw$", names(raw_tol), value = TRUE)
tol_collapsed <- raw_tol %>%
  group_by(participant_ID, time) %>%
  summarise(across(all_of(value_cols), first_nonmissing), .groups = "drop")

analysis <- frozen %>%
  mutate(time_chr = as.character(time)) %>%
  left_join(tol_collapsed, by = c("participant_ID", "time_chr" = "time")) %>%
  select(-time_chr)
if (nrow(analysis) != expected$visits || anyNA(analysis[paste0(target_vars, "__occ_raw")])) {
  stop("Direct raw tolerability reconstruction did not match all frozen patient-visit keys")
}

coding_rows <- list()
for (i in seq_along(target_vars)) {
  variable <- target_vars[[i]]
  occ <- analysis[[paste0(variable, "__occ_raw")]]
  tol <- analysis[[paste0(variable, "__tol_raw")]]
  expected_occ <- ifelse(occ == 1, 1, ifelse(occ == 2, 0, NA_real_))
  occurrence_match <- all(analysis[[variable]] == expected_occ, na.rm = FALSE)
  level <- rep(NA_character_, length(occ))
  level[occ == 2 & is.na(tol)] <- "None"
  level[occ == 1 & tol == 1] <- "Tolerable"
  level[occ == 1 & tol == 2] <- "Distressing"
  contradiction <- (occ == 2 & !is.na(tol)) | (is.na(occ) & !is.na(tol))
  invalid <- (!is.na(occ) & !occ %in% c(1, 2)) | (!is.na(tol) & !tol %in% c(1, 2))
  if (!occurrence_match || any(contradiction, na.rm = TRUE) || any(invalid, na.rm = TRUE)) {
    stop("Tolerability coding audit failed for ", variable)
  }
  analysis[[paste0("tol__", variable)]] <- factor(level, levels = c("None", "Tolerable", "Distressing"))
  counts <- table(factor(level, levels = c("None", "Tolerable", "Distressing")), useNA = "always")
  coding_rows[[i]] <- tibble(
    Domain = unname(domain_display[[variable]]), Analysis_variable = variable,
    Occurrence_mapping_match = occurrence_match,
    None = as.integer(counts[["None"]]), Tolerable = as.integer(counts[["Tolerable"]]),
    Distressing = as.integer(counts[["Distressing"]]), Missing = sum(is.na(level)),
    Contradictions = sum(contradiction, na.rm = TRUE), Invalid_codes = sum(invalid, na.rm = TRUE),
    Status = "PASS"
  )
}

analysis <- analysis %>% arrange(participant_ID, visit_order)

clinical_terms <- c("time", "age_z", "sex", "education3", "first_episode", "baseline_HAMD17_z",
                    "baseline_SDS_z", "baseline_med_class", "concurrent_HAMD17_z")

fit_gee <- function(dat, outcome, predictors, maxit = 400L) {
  formula <- reformulate(c(predictors, clinical_terms), response = outcome)
  fit <- geeglm(
    formula = formula,
    family = binomial(link = "logit"),
    data = dat,
    id = participant_ID,
    corstr = "exchangeable",
    std.err = "san.se",
    scale.fix = TRUE,
    control = geese.control(epsilon = 1e-9, maxit = maxit, trace = FALSE),
    na.action = na.fail
  )
  beta <- coef(fit)
  V <- fit$geese$vbeta
  dimnames(V) <- list(names(beta), names(beta))
  X <- model.matrix(fit)
  list(
    fit = fit, beta = beta, V = V, se = sqrt(diag(V)), X = X,
    converged = isTRUE(fit$geese$error == 0),
    alpha = as.numeric(fit$geese$alpha),
    design_rank = qr(X)$rank,
    design_columns = ncol(X),
    design_terms = paste(colnames(X), collapse = " | "),
    covariance_finite = all(is.finite(V)),
    covariance_symmetric = isTRUE(all.equal(V, t(V), tolerance = 1e-10)),
    covariance_min_eigenvalue = min(eigen((V + t(V)) / 2, symmetric = TRUE, only.values = TRUE)$values),
    formula = paste(deparse(formula), collapse = " ")
  )
}

wald_row <- function(model, term, outcome_label, domain_label) {
  beta <- unname(model$beta[[term]])
  se <- unname(model$se[[term]])
  z <- beta / se
  p <- 2 * pnorm(abs(z), lower.tail = FALSE)
  tibble(
    Analysis = "Domain occurrence", Outcome = outcome_label, Domain = domain_label,
    Contrast = "Present vs Absent", Coefficient = beta, Robust_SE = se,
    OR = exp(beta), CI_low = exp(beta - qnorm(0.975) * se), CI_high = exp(beta + qnorm(0.975) * se),
    P_raw = p
  )
}

occ_specs <- tribble(
  ~outcome, ~outcome_label, ~table_id,
  "any_burden", "Any burden (FIBSER 1-6)", "S4",
  "higher_burden", "Higher burden (FIBSER 3-6)", "S5"
)
occ_results <- list()
occ_diagnostics <- list()
for (i in seq_len(nrow(occ_specs))) {
  spec <- occ_specs[i, ]
  model <- fit_gee(analysis, spec$outcome, domain_vars, maxit = 300L)
  rows <- map2_dfr(domain_vars, unname(domain_display), ~ wald_row(model, .x, spec$outcome_label, .y))
  rows$Table <- spec$table_id
  rows$Participants <- n_distinct(analysis$participant_ID)
  rows$Person_visits <- nrow(analysis)
  rows$Events <- sum(analysis[[spec$outcome]])
  rows$FDR_P <- p.adjust(rows$P_raw, method = "BH")
  rows$Raw_significant <- rows$P_raw < 0.05
  rows$FDR_significant <- rows$FDR_P < 0.05
  occ_results[[i]] <- rows
  occ_diagnostics[[i]] <- tibble(
    Analysis = "Domain occurrence", Table = spec$table_id, Outcome = spec$outcome_label,
    Domain = "All nine simultaneous", Participants = n_distinct(analysis$participant_ID),
    Person_visits = nrow(analysis), Events = sum(analysis[[spec$outcome]]),
    Converged = model$converged, Working_correlation = model$alpha,
    Design_columns = model$design_columns, Design_rank = model$design_rank,
    Full_rank = model$design_columns == model$design_rank,
    Robust_covariance_finite = model$covariance_finite,
    Robust_covariance_symmetric = model$covariance_symmetric,
    Robust_covariance_min_eigenvalue = model$covariance_min_eigenvalue,
    Design_terms = model$design_terms,
    Formula = model$formula
  )
}
occ_results <- bind_rows(occ_results)

tol_specs <- tribble(
  ~outcome, ~outcome_label, ~table_id,
  "any_burden", "Any burden (FIBSER 1-6)", "S6",
  "higher_burden", "Higher burden (FIBSER 3-6)", "S7"
)
tol_results <- list()
tol_diagnostics <- list()
result_i <- 0L
diag_i <- 0L
for (s in seq_len(nrow(tol_specs))) {
  spec <- tol_specs[s, ]
  for (variable in target_vars) {
    d <- analysis %>% filter(!is.na(.data[[paste0("tol__", variable)]]))
    d$target_tolerability <- d[[paste0("tol__", variable)]]
    other_vars <- setdiff(domain_vars, variable)
    model <- fit_gee(d, spec$outcome, c("target_tolerability", other_vars), maxit = 400L)
    t_name <- "target_tolerabilityTolerable"
    d_name <- "target_tolerabilityDistressing"
    if (!all(c(t_name, d_name) %in% names(model$beta))) stop("Target tolerability coefficients absent")
    contrast_def <- list(
      "Tolerable vs None" = setNames(1, t_name),
      "Distressing vs None" = setNames(1, d_name),
      "Distressing vs Tolerable" = setNames(c(1, -1), c(d_name, t_name))
    )
    model_rows <- imap_dfr(contrast_def, function(weights, contrast_name) {
      cvec <- setNames(rep(0, length(model$beta)), names(model$beta))
      cvec[names(weights)] <- weights
      est <- sum(cvec * model$beta)
      variance <- as.numeric(t(cvec) %*% model$V %*% cvec)
      se <- sqrt(variance)
      z <- est / se
      p <- 2 * pnorm(abs(z), lower.tail = FALSE)
      tibble(
        Analysis = "Tolerability", Table = spec$table_id, Outcome = spec$outcome_label,
        Domain = unname(domain_display[[variable]]), Contrast = contrast_name,
        Coefficient = est, Contrast_variance = variance, Robust_SE = se, Wald_z = z,
        OR = exp(est), CI_low = exp(est - qnorm(0.975) * se), CI_high = exp(est + qnorm(0.975) * se),
        P_raw = p, Participants = n_distinct(d$participant_ID), Person_visits = nrow(d),
        Events = sum(d[[spec$outcome]]),
        Variance_formula = ifelse(contrast_name == "Distressing vs Tolerable",
                                  "Var(D)+Var(T)-2Cov(D,T)", "c' V_robust c")
      )
    })
    result_i <- result_i + 1L
    tol_results[[result_i]] <- model_rows
    diag_i <- diag_i + 1L
    tol_diagnostics[[diag_i]] <- tibble(
      Analysis = "Tolerability", Table = spec$table_id, Outcome = spec$outcome_label,
      Domain = unname(domain_display[[variable]]), Participants = n_distinct(d$participant_ID),
      Person_visits = nrow(d), Excluded_visits = nrow(analysis) - nrow(d), Events = sum(d[[spec$outcome]]),
      Converged = model$converged, Working_correlation = model$alpha,
      Design_columns = model$design_columns, Design_rank = model$design_rank,
      Full_rank = model$design_columns == model$design_rank,
      Robust_covariance_finite = model$covariance_finite,
      Robust_covariance_symmetric = model$covariance_symmetric,
      Robust_covariance_min_eigenvalue = model$covariance_min_eigenvalue,
      Design_terms = model$design_terms,
      Formula = model$formula
    )
  }
}
tol_results <- bind_rows(tol_results) %>%
  group_by(Table) %>%
  mutate(FDR_P = p.adjust(P_raw, method = "BH"),
         Raw_significant = P_raw < 0.05, FDR_significant = FDR_P < 0.05) %>%
  ungroup()

diagnostics <- bind_rows(occ_diagnostics, tol_diagnostics)
if (!all(diagnostics$Converged) || !all(diagnostics$Full_rank) ||
    !all(diagnostics$Robust_covariance_finite) || !all(diagnostics$Robust_covariance_symmetric) ||
    any(diagnostics$Robust_covariance_min_eigenvalue < -1e-8)) {
  warning("At least one fitted R model requires review; see R_model_diagnostics.csv")
}

freeze_audit <- tibble(
  Check = c("Raw Excel SHA-256 vs locked Python run", "Frozen participants", "Frozen visits", "Duplicate patient-visit keys",
            "Week2 visits", "Week8 visits", "Week12 visits", "Week24 visits",
            "Any-burden events", "Higher-burden events", "Nine domain variables binary",
            "Education three-level mapping complete"),
  Observed = c(raw_sha256, n_distinct(frozen$participant_ID), nrow(frozen),
               sum(duplicated(frozen[c("participant_ID", "time")])), as.integer(week_tab[weeks]),
               sum(frozen$any_burden), sum(frozen$higher_burden),
               all(vapply(frozen[domain_vars], function(x) all(x %in% c(0, 1)), logical(1))),
               sum(is.na(frozen$education3))),
  Expected = c(expected$python_run_raw_sha256, expected$participants, expected$visits, 0L,
               unname(expected$week_counts), expected$any_events, expected$higher_events, TRUE, 0L),
  Status = ifelse(as.character(Observed) == as.character(Expected), "PASS", "REVIEW_HASH_DIFF")
)

factor_audit <- tibble(
  Variable = c("time", "sex", "education3", "first_episode", "baseline_med_class", "target_tolerability"),
  Reference = c("Week2", "Female", "College or above", "FirstEpisode", "SNRI", "None"),
  Levels = c(paste(levels(frozen$time), collapse = " | "), paste(levels(frozen$sex), collapse = " | "),
             paste(levels(frozen$education3), collapse = " | "), paste(levels(frozen$first_episode), collapse = " | "),
             paste(levels(frozen$baseline_med_class), collapse = " | "), "None | Tolerable | Distressing"),
  Status = "PASS"
)

safe_write_csv(occ_results, "R_domain_occurrence_results.csv")
safe_write_csv(tol_results, "R_tolerability_results.csv")
safe_write_csv(diagnostics, "R_model_diagnostics.csv")
safe_write_csv(freeze_audit, "R_freeze_data_audit.csv")
safe_write_csv(bind_rows(mapping_rows), "R_tolerability_mapping_audit.csv")
safe_write_csv(bind_rows(sheet_key_rows), "R_raw_sheet_key_audit.csv")
safe_write_csv(bind_rows(coding_rows), "R_tolerability_coding_audit.csv")
safe_write_csv(factor_audit, "R_factor_reference_audit.csv")

env <- tibble(
  Component = c("R", "geepack", "readxl", "dplyr", "Operating system", "Raw Excel SHA-256", "Frozen CSV SHA-256"),
  Version_or_value = c(R.version.string, as.character(packageVersion("geepack")),
                       as.character(packageVersion("readxl")), as.character(packageVersion("dplyr")),
                       paste(Sys.info()[c("sysname", "release", "version")], collapse = " "),
                       raw_sha256, digest(frozen_csv, algo = "sha256", file = TRUE))
)
safe_write_csv(env, "R_environment.csv")

summary_obj <- list(
  generated = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  r_version = R.version.string,
  geepack_version = as.character(packageVersion("geepack")),
  model = "geepack::geeglm; binomial-logit; exchangeable; patient ID clusters; san.se robust covariance; scale fixed at 1",
  frozen = list(participants = n_distinct(frozen$participant_ID), visits = nrow(frozen),
                week_counts = as.list(as.integer(week_tab[weeks])), any_events = sum(frozen$any_burden),
                higher_events = sum(frozen$higher_burden)),
  models = list(total = nrow(diagnostics), converged = sum(diagnostics$Converged),
                full_rank = sum(diagnostics$Full_rank)),
  python_comparison = "Not run by this script; use compare_R_to_Python_binary_gee.R after obtaining Python result files."
)
write_json(summary_obj, file.path(out_dir, "run_summary.json"), pretty = TRUE, auto_unbox = TRUE, na = "null")

report <- c(
  "# Standalone R/geepack binary GEE results",
  "",
  paste0("- Generated: ", summary_obj$generated),
  paste0("- R: ", R.version.string),
  paste0("- geepack: ", packageVersion("geepack")),
  paste0("- Frozen sample: ", n_distinct(frozen$participant_ID), " patients; ", nrow(frozen), " visits; ",
         sum(frozen$any_burden), " any-burden events; ", sum(frozen$higher_burden), " higher-burden events."),
  paste0("- Models: ", nrow(diagnostics), "; converged: ", sum(diagnostics$Converged),
         "; full-rank: ", sum(diagnostics$Full_rank), "."),
  paste0("- Raw Excel hash matches locked Python run: ", raw_hash_matches_python_run,
         " (current ", raw_sha256, "; Python ", expected$python_run_raw_sha256, ")."),
  "- Python result files were not required or read. Cross-implementation comparison is a separate optional step.",
  "",
  "## Formal significance results",
  "",
  paste0("- S4 occurrence: raw P<0.05 ", sum(occ_results$Table == "S4" & occ_results$Raw_significant),
         "/9; FDR P<0.05 ", sum(occ_results$Table == "S4" & occ_results$FDR_significant), "/9."),
  paste0("- S5 occurrence: raw P<0.05 ", sum(occ_results$Table == "S5" & occ_results$Raw_significant),
         "/9; FDR P<0.05 ", sum(occ_results$Table == "S5" & occ_results$FDR_significant), "/9."),
  paste0("- S6 tolerability: raw P<0.05 ", sum(tol_results$Table == "S6" & tol_results$Raw_significant),
         "/24; FDR P<0.05 ", sum(tol_results$Table == "S6" & tol_results$FDR_significant), "/24."),
  paste0("- S7 tolerability: raw P<0.05 ", sum(tol_results$Table == "S7" & tol_results$Raw_significant),
         "/24; FDR P<0.05 ", sum(tol_results$Table == "S7" & tol_results$FDR_significant), "/24."),
  paste0("- FDR-significant occurrence rows: ",
         paste(paste(occ_results$Table[occ_results$FDR_significant], occ_results$Domain[occ_results$FDR_significant], sep = "/"), collapse = "; "), "."),
  paste0("- FDR-significant tolerability rows: ",
         paste(paste(tol_results$Table[tol_results$FDR_significant], tol_results$Domain[tol_results$FDR_significant],
                     tol_results$Contrast[tol_results$FDR_significant], sep = "/"), collapse = "; "), "."),
  "",
  "Formal BH families were kept separate: 9 occurrence-domain tests within each binary threshold and 24 tolerability contrasts within each threshold. No cross-threshold FDR replaced the formal analysis.",
  "The Distressing-vs-Tolerable standard error was calculated as c'V_robust c, including -2Cov(D,T).",
  "",
  "## Reproducibility boundary",
  "",
  "This script verifies its own frozen-data gates, mapping, model convergence, design rank, robust covariance, and FDR families. Agreement with the historical Python implementation is not claimed unless the separate comparison script is run. The fitted estimands are longitudinal associations, not causal effects."
)
writeLines(report, file.path(out_dir, "R_standalone_report.md"), useBytes = TRUE)

cat(toJSON(list(output_dir = out_dir, models = nrow(diagnostics),
                    converged = sum(diagnostics$Converged), r = R.version.string,
                    geepack = as.character(packageVersion("geepack"))), auto_unbox = TRUE), "\n")


