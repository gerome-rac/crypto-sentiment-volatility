# 11b_google_combine.R -- Google Trends combination (Eichenauer, Indergand, Martinez & Sax 2022;
# R package trendecon 0.3.2), called by 11b_google_combine.py; run in env 'gtrends'.
#
# Per term and per input set (each draw alone, and all draws pooled):
#   1. daily, weekly and monthly windows are averaged by date over all windows of the set
#      (trendecon aggregate_windows / aggregate_averages: plain mean of the window values,
#      '<1' read as 0 as in trendecon ts_gtrends);
#   2. the averages are capped to the fetch range [from, to] (as ts_gtrends_mwd caps at 'to'),
#      weekly values kept only for weeks fully inside the daily span, monthly values only for
#      months fully inside it;
#   3. the daily path is bent to the weekly series, then to the monthly series, with
#      tempdisagg::td(method = "fast", conversion = "mean") -- exactly the two td calls of
#      trendecon proc_combine_freq(). The bug of ts_gtrends_mwd() (mm <- filter(ww, ...), the
#      monthly step reusing the weekly series) is not reproduced: the monthly series is used.
#   (proc_combine_freq's hourly extension of the most recent days is not used: the end date is
#    historical.)
# Parameter window_combine (7th argument):
#   mean  = trendecon as written: plain mean of the window values by date (each window keeps its own
#           0-100 scale); literal reading of 'averaged over windows' (MASTERPLAN 6 Oct (6)).
#   chain = not in trendecon: daily windows are first put on one scale, in order of window start
#           (ties: draw order), each window multiplied by the ratio of the mean of the already-chained
#           average to its own mean over their common days, then averaged by date. Weekly and monthly
#           windows are always averaged as in trendecon.
# Usage: Rscript 11b_google_combine.R <raw_root> <out_dir> <draws comma-separated> <from> <to> <terms comma-separated> <mean|chain>
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 7)
window_combine <- args[7]; stopifnot(window_combine %in% c("mean", "chain", "mean_prop"))
# 'mean_prop' (Deviation 2 of the D8 note, 7 Oct 2026): window mean as 'mean', then proportional
# Denton-Cholette (h = 1) instead of the additive 'fast' step; '<1' and daily averages below 0.5 set to 0.5
prop <- window_combine == "mean_prop"; LT1 <- if (prop) "0.5" else "0"; FLOOR <- 0.5
raw_root <- args[1]; out_dir <- args[2]
draws <- strsplit(args[3], ",")[[1]]; from <- as.Date(args[4]); to <- as.Date(args[5])
terms <- strsplit(args[6], ",")[[1]]
.libPaths(c(file.path(raw_root, "_r_lib"), .libPaths()))
suppressPackageStartupMessages({library(tempdisagg); library(tsbox); library(dplyr); library(readr)})
stopifnot(as.character(packageVersion("tempdisagg")) == "1.2.0")

read_windows <- function(term, freq, draw) {
  dir <- file.path(raw_root, sprintf("draw_%s", draw))
  fs <- list.files(dir, pattern = sprintf("^%s_%s_.*\\.csv$", term, freq), full.names = TRUE)
  if (!length(fs)) stop(sprintf("no %s windows for %s in draw %s", freq, term, draw))
  x <- bind_rows(lapply(fs, function(f) read_csv(f, col_types = cols(.default = "c"))))
  stopifnot(all(x$keyword == term), all(x$freq == freq))
  hits <- gsub("<1", LT1, x$hits, fixed = TRUE)
  v <- as.numeric(hits); stopifnot(!any(is.na(v)))
  tibble(time = as.Date(x$date), value = v, window = paste(x$window_start, x$window_end), draw = draw)
}
chain_windows <- function(x) {
  keys <- x %>% distinct(draw, window) %>% mutate(start = as.Date(substr(window, 1, 10))) %>% arrange(start, draw)
  acc <- NULL; nfac <- c()
  for (i in seq_len(nrow(keys))) {
    wv <- x %>% filter(draw == keys$draw[i], window == keys$window[i])
    if (is.null(acc)) { f <- 1 } else {
      cur <- agg(acc); ov <- inner_join(cur, wv, by = "time", suffix = c(".c", ".w"))
      stopifnot(nrow(ov) >= 30, sum(ov$value.w) > 0)
      f <- mean(ov$value.c) / mean(ov$value.w)
    }
    nfac <- c(nfac, f)
    acc <- bind_rows(acc, mutate(wv, value = value * f))
  }
  attr(acc, "factors") <- nfac
  acc
}
agg <- function(x) x %>% group_by(time) %>% summarize(value = mean(value), n = n(), .groups = "drop") %>% arrange(time)

combine <- function(term, ds, label) {
  d <- bind_rows(lapply(ds, function(k) read_windows(term, "d", k)))
  w <- bind_rows(lapply(ds, function(k) read_windows(term, "w", k)))
  m <- bind_rows(lapply(ds, function(k) read_windows(term, "m", k)))
  if (window_combine == "chain") d <- chain_windows(d)
  dd <- agg(d) %>% filter(time >= from, time <= to)
  if (prop) dd$value <- pmax(dd$value, FLOOR)
  stopifnot(all(diff(dd$time) == 1), min(dd$time) == from, max(dd$time) == to)
  ww <- agg(w) %>% filter(time >= min(dd$time), time + 6 <= max(dd$time))
  mm <- agg(m) %>% mutate(mend = seq_along(time)) %>% filter(time >= min(dd$time))
  mm$mend <- as.Date(sapply(mm$time, function(t) seq(t, length.out = 2, by = "1 month")[2] - 1), origin = "1970-01-01")
  mm <- mm %>% filter(mend <= max(dd$time)) %>% select(time, value, n)
  stopifnot(all(as.integer(format(ww$time, "%u")) == 7))   # Google weeks start on Sunday
  D <- select(dd, time, value); W <- select(ww, time, value); M <- select(mm, time, value)
  if (prop) {
    m_wd <- td(W ~ 0 + D, method = "denton-cholette", criterion = "proportional", h = 1, conversion = "mean"); wd <- predict(m_wd)
    m_mwd <- td(M ~ 0 + wd, method = "denton-cholette", criterion = "proportional", h = 1, conversion = "mean"); mwd <- predict(m_mwd)
  } else {
    m_wd <- td(W ~ D, method = "fast", conversion = "mean"); wd <- predict(m_wd)
    m_mwd <- td(M ~ wd, method = "fast", conversion = "mean"); mwd <- predict(m_mwd)
  }
  stopifnot(nrow(mwd) == nrow(dd), all(mwd$time == dd$time))
  co <- function(mod, stage) {
    if (prop) return(tibble(term = term, variant = label, stage = stage, slope = NA_real_, slope_se = NA_real_,
                            intercept = NA_real_, adj_r2 = NA_real_, rho = NA_real_, n_low = NA_integer_, n_high = NA_integer_))
    s <- summary(mod); cf <- s$coefficients
    tibble(term = term, variant = label, stage = stage, slope = cf[2, 1], slope_se = cf[2, 2],
           intercept = cf[1, 1], adj_r2 = s$adj.r.squared, rho = mod$rho, n_low = NROW(mod$residuals), n_high = NROW(mod$p))
  }
  # constraint checks: weekly / monthly means of the result equal the low-frequency inputs
  chk_w <- mwd %>% mutate(wk = time - (as.integer(format(time, "%u")) %% 7)) %>% group_by(wk) %>%
    summarize(v = mean(value), k = n(), .groups = "drop") %>% inner_join(ww, by = c("wk" = "time")) %>% filter(k == 7)
  chk_m <- mwd %>% mutate(mo = as.Date(format(time, "%Y-%m-01"))) %>% group_by(mo) %>%
    summarize(v = mean(value), .groups = "drop") %>% inner_join(mm, by = c("mo" = "time"))
  list(series = tibble(term = term, variant = label, time = dd$time, daily_avg = dd$value, n_windows = dd$n,
                       wd = wd$value, value = mwd$value),
       diag = bind_rows(co(m_wd, "daily_to_weekly"), co(m_mwd, "weekly_to_monthly")) %>%
         mutate(max_abs_month_constraint = max(abs(chk_m$v - chk_m$value)), n_months = nrow(chk_m),
                cor_week_means_after_monthly = cor(chk_w$v, chk_w$value), n_weeks = nrow(chk_w),
                n_daily_windows = n_distinct(d$window, d$draw), n_weekly_windows = n_distinct(w$window, w$draw),
                n_monthly_windows = n_distinct(m$window, m$draw)))
}

res <- list(); dia <- list()
for (term in terms) {
  sets <- c(setNames(as.list(draws), paste0("draw_", draws)), list(pooled = draws))
  for (lab in names(sets)) {
    r <- combine(term, sets[[lab]], lab)
    res[[length(res) + 1]] <- r$series; dia[[length(dia) + 1]] <- r$diag
  }
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
write_csv(bind_rows(res) %>% mutate(window_combine = window_combine), file.path(out_dir, sprintf("google_combined_long_%s.csv", window_combine)))
write_csv(bind_rows(dia) %>% mutate(window_combine = window_combine), file.path(out_dir, sprintf("google_td_diagnostics_%s.csv", window_combine)))
cat(sprintf("combined %d terms x %d variants\n", length(terms), length(draws) + 1))
