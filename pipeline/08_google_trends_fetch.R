# Step 08a: fetch one Google Trends draw (design agreed with Gerome, 6 Oct 2026).
# Procedure: Eichenauer, Indergand, Martinez & Sax (2022), R package trendecon 0.3.2,
# with option B (reduced windows), all deviations flagged:
#   daily   : 6-month windows starting 2019-07-01 every 45 days, 29 windows (trendecon: every 15 days)
#   weekly  : 5-year windows starting 2019-07-01 every 26 weeks, 7 windows (trendecon: every 11 weeks)
#   monthly : 2 windows (2019-07-01 and 2019-01-01 to today)
#   trendecon's end-anchored 3-month daily and 2-year weekly windows are dropped
#   (they stabilise the most recent days of a real-time index; our end date is historical).
# One term per request, worldwide (geo = ""), web search, all categories, tz = 0.
# Every window is saved as its own CSV as soon as it arrives; existing files are skipped,
# so a draw interrupted by rate limits resumes on the next run (same draw folder).
# Usage: Rscript 08_google_trends_fetch.R <draw_id>   e.g. 01
args <- commandArgs(trailingOnly = TRUE)
draw_id <- if (length(args)) args[1] else stop("draw id missing")
# Folder of the draws: data/raw/google_trends below the working directory (run from the repository root),
# or the folder named in the environment variable THESIS_GT_DIR.
root <- Sys.getenv("THESIS_GT_DIR", unset = file.path("data", "raw", "google_trends"))
lib <- file.path(root, "_r_lib")
.libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({library(gtrendsR); library(trendecon)})
options(trendecon.query_pause = as.numeric(Sys.getenv("GT_PAUSE", "25")))

terms <- c("Bitcoin", "Ethereum", "Dogecoin", "XRP", "Binance", "cryptocurrency", "crypto")
from <- as.Date("2019-07-01")
today_utc <- as.Date(format(Sys.time(), tz = "UTC"))
mk <- function(starts, len) {
  ends <- as.Date(sapply(starts, function(s) seq(s, length.out = 2, by = len)[2]), origin = "1970-01-01")
  ends <- pmin(ends, today_utc)
  data.frame(start = starts, end = ends)
}
win <- rbind(
  cbind(freq = "d", mk(seq(from, by = 45, length.out = 29), "6 months")),
  cbind(freq = "w", mk(seq(from, by = 26 * 7, length.out = 7), "5 years")),
  cbind(freq = "m", data.frame(start = as.Date(c("2019-07-01", "2019-01-01")), end = today_utc))
)
jobs <- merge(data.frame(term = terms), win)  # 7 x 38 = 266 requests
jobs <- jobs[order(match(jobs$term, terms), jobs$freq, jobs$start), ]

out <- file.path(root, sprintf("draw_%s", draw_id))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
logf <- file.path(out, "fetch_log.csv")
if (!file.exists(logf)) cat("utc_time,term,freq,start,end,status,n_rows,attempt_note\n", file = logf)

n_new <- 0; n_skip <- 0; n_fail <- 0; consecutive_fail <- 0
for (i in seq_len(nrow(jobs))) {
  j <- jobs[i, ]
  f <- file.path(out, sprintf("%s_%s_%s_%s.csv", j$term, j$freq, j$start, j$end))
  if (file.exists(f)) { n_skip <- n_skip + 1; next }
  tm <- sprintf("%s %s", j$start, j$end)
  r <- tryCatch(trendecon:::gtrends_with_backoff(keyword = j$term, geo = "", time = tm, gprop = "web",
                  category = "0", tz = 0, onlyInterest = TRUE, retry = 6, wait = 30, max_wait = 600, quiet = TRUE),
                error = function(e) e)
  now <- format(Sys.time(), tz = "UTC", usetz = FALSE)
  if (inherits(r, "error")) {
    n_fail <- n_fail + 1; consecutive_fail <- consecutive_fail + 1
    cat(sprintf("%s,%s,%s,%s,%s,error,0,\"%s\"\n", now, j$term, j$freq, j$start, j$end,
                gsub("\"|\n", " ", substr(conditionMessage(r), 1, 200))), file = logf, append = TRUE)
    if (consecutive_fail >= 3) { message("3 consecutive failures: stopping this draw for today"); break }
    next
  }
  consecutive_fail <- 0
  iot <- r$interest_over_time
  if (is.null(iot)) iot <- data.frame(date = as.Date(character()), hits = character())
  d <- data.frame(date = as.Date(iot$date), hits = as.character(iot$hits), keyword = j$term,
                  freq = j$freq, window_start = j$start, window_end = j$end, fetched_utc = now)
  tmpf <- paste0(f, ".tmp"); write.csv(d, tmpf, row.names = FALSE); file.rename(tmpf, f)
  cat(sprintf("%s,%s,%s,%s,%s,ok,%d,\n", now, j$term, j$freq, j$start, j$end, nrow(d)), file = logf, append = TRUE)
  n_new <- n_new + 1
}
message(sprintf("draw %s: new %d, skipped %d, failed %d, of %d", draw_id, n_new, n_skip, n_fail, nrow(jobs)))
