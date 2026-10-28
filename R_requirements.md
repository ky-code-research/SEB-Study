# R environment

The final independent binary GEE validation recorded:

- R 4.5.3 (2026-03-11 ucrt)
- geepack 1.3.13
- readxl 1.4.5
- dplyr 1.2.1

Final packaging check on 2026-10-28 confirmed that `D:/RR/R-4.5.3/bin/Rscript.exe` starts normally, reports R 4.5.3, executes a `--vanilla` expression, and parses both public R scripts without running them. The installed analysis packages were loadable at these versions: geepack 1.3.13, readxl 1.4.5, dplyr 1.2.1, tidyr 1.3.2, stringr 1.6.0, purrr 1.2.2, tibble 3.3.1, jsonlite 2.0.0, and digest 0.6.39.

This workstation emits warnings when attempting to set `C.UTF-8` locale categories and falls back to locale `C`; startup and script parsing still exit successfully. No GEE model was run during the packaging check. The earlier initialization failure came from a separate R 4.4.3 copy and does not describe the verified R 4.5.3 installation.

The script also loads `tidyr`, `stringr`, `purrr`, `tibble`, `jsonlite`, and `digest`. Their exact versions were not preserved in the final environment CSV and are therefore not claimed as verified here. A rerun should record `sessionInfo()` and package versions in its restricted output.

Install the verified core versions from an appropriate package snapshot or archive, then install the remaining declared packages:

```r
install.packages(c("geepack", "readxl", "dplyr", "tidyr", "stringr",
                   "purrr", "tibble", "jsonlite", "digest"))
```

The adopted model is `geepack::geeglm` with binomial-logit family, exchangeable working correlation, patient ID clusters, `std.err="san.se"`, fixed scale, and `na.action=na.fail`.

`R/reproduce_binary_gee_geepack.R` is the standalone model runner and requires only the authorized frozen CSV and raw workbook. `R/compare_R_to_Python_binary_gee.R` is an optional second-stage consistency check requiring already generated R and Python result tables; it never fits a model.

