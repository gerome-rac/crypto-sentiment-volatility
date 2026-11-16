# Step 08b: Google Trends related queries for 'crypto' (28 Oct 2026).
# Purpose: the third piece of series-level evidence of the 'crypto' rule (D8 note section 5; MASTERPLAN_NOV15.md
# section 7, decisions of 6 Oct 2026, point (6)): "the rank of 'crypto.com' among Google's related queries".
# The draws of 08_google_trends_fetch.R request interest over time only (onlyInterest = TRUE), so the related
# queries are fetched here, once (28 Oct 2026).
# Settings as in the draws: one term per request, worldwide (geo = ""), web search, all categories,
# hl = "en-US", tz = 0. Three windows: the sample 2019-08-01..2023-01-31, the period 2021-11-01..2022-02-28
# named in the rule, and one comparison period before it, 2020-11-01..2021-02-28.
# Calls: gtrendsR 1.5.2, get_widget() then related_queries() -- the two calls gtrends(onlyInterest = FALSE)
# makes for the related queries; its interest-over-time, region and related-topics requests are not made.
# Every HTTP response body except the cookie request is stored as it arrived (raw_*.txt), with its md5 in
# the fetch log; the parsed lists are stored with their rank (order of Google's list, 1 = first).
# Retries: up to 4 attempts per window after an error, waits 30/60/120 s; 25 s pause between windows
# (GT_PAUSE of the draws). Fetched once: a window whose related-searches response is already stored in the
# output folder (an earlier run of this script) is not requested again; its stored response is parsed with
# gtrendsR's own parser (extract_top_rising), exactly as related_queries() parses a fresh response.
# Note (28 Oct 2026): run 1 fetched the first window and then stopped at the parsing step (gtrendsR names
# the top/rising column 'related_topics' also for queries); run 2 parsed that stored response and fetched
# the other two windows.
# Usage: Rscript 08b_google_related_queries.R <out_dir> <r_lib_dir>
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
out <- args[1]; lib <- args[2]
.libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({library(gtrendsR); library(jsonlite)})
stopifnot(as.character(packageVersion("gtrendsR")) == "1.5.2")
if (dir.exists(out)) stopifnot(all(grepl("^raw_", list.files(out, all.files = TRUE, no.. = TRUE))))   # only stored responses of an earlier run
dir.create(out, recursive = TRUE, showWarnings = FALSE)
self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
keyword <- "crypto"; geo <- ""; hl <- "en-US"; tz <- 0; category <- 0; gprop <- ""   # gprop "" = web
cookie_url <- "http://trends.google.com/Cookies/NID"
pause <- as.numeric(Sys.getenv("GT_PAUSE", "25"))
windows <- data.frame(label = c("sample_2019_08_2023_01", "period_2021_11_2022_02", "comparison_2020_11_2021_02"),
                      time = c("2019-08-01 2023-01-31", "2021-11-01 2022-02-28", "2020-11-01 2021-02-28"),
                      stringsAsFactors = FALSE)

# capture every HTTP response body as it arrives (gtrendsR calls curl::curl_fetch_memory)
cap <- new.env(); cap$calls <- list()
orig_fetch <- curl::curl_fetch_memory
utils::assignInNamespace("curl_fetch_memory", function(url, handle = curl::new_handle()) {
  res <- orig_fetch(url, handle = handle)
  cap$calls[[length(cap$calls) + 1]] <- list(url = url, status = res$status_code, content = res$content,
                                             utc = format(Sys.time(), tz = "UTC", usetz = FALSE))
  res
}, ns = "curl")

utc <- function() format(Sys.time(), tz = "UTC", usetz = FALSE)
log <- list(); parsed <- list()
for (k in seq_len(nrow(windows))) {
  if (k > 1 && length(cap$calls) > 0) Sys.sleep(pause)
  w <- windows[k, ]
  ci <- data.frame(keyword = keyword, geo = geo, time = w$time, stringsAsFactors = FALSE)
  n0 <- length(cap$calls); attempt <- 0; note <- ""; r <- NULL
  stored <- list.files(out, pattern = sprintf("^raw_%s_http[0-9]+_relatedsearches_status200\\.txt$", w$label))
  if (length(stored) == 1) {
    lines <- readLines(file.path(out, stored), encoding = "UTF-8", warn = FALSE)
    rq <- if (length(lines) <= 4) NULL else gtrendsR:::extract_top_rising(lines)
    if (!is.null(rq)) { rq$id <- NULL; rq$geo <- ""; rq$keyword <- keyword; rq$category <- category }
    r <- list(widget = NULL, rq = rq, stored = stored,
              fetched_utc = format(file.info(file.path(out, stored))$mtime, tz = "UTC", usetz = FALSE))
    note <- sprintf("not requested again: response stored by an earlier run of this script (file time %s UTC), parsed here; ", r$fetched_utc)
  } else repeat {
    attempt <- attempt + 1
    r <- tryCatch({
      widget <- gtrendsR:::get_widget(ci, category, gprop, hl, cookie_url, tz)
      list(widget = widget, rq = gtrendsR:::related_queries(widget, ci, tz, hl))
    }, error = function(e) e)
    if (!inherits(r, "error")) break
    note <- paste0(note, sprintf("attempt %d: %s; ", attempt, gsub("[\r\n\"]", " ", substr(conditionMessage(r), 1, 160))))
    if (attempt >= 4) break
    Sys.sleep(30 * 2^(attempt - 1))
  }
  calls <- if (length(cap$calls) > n0) cap$calls[(n0 + 1):length(cap$calls)] else list()
  raw_files <- c(); raw_md5 <- c()
  if (!is.null(r$stored)) {
    prev <- list.files(out, pattern = sprintf("^raw_%s_.*\\.txt$", w$label))
    raw_files <- prev; raw_md5 <- unname(tools::md5sum(file.path(out, prev)))
  }
  for (j in seq_along(calls)) {
    cl <- calls[[j]]
    if (grepl("Cookies/NID", cl$url, fixed = TRUE)) next
    kind <- if (grepl("/api/explore", cl$url, fixed = TRUE)) "explore" else if (grepl("relatedsearches", cl$url, fixed = TRUE)) "relatedsearches" else "other"
    f <- sprintf("raw_%s_http%02d_%s_status%d.txt", w$label, j, kind, cl$status)
    writeBin(cl$content, file.path(out, f))
    raw_files <- c(raw_files, f); raw_md5 <- c(raw_md5, unname(tools::md5sum(file.path(out, f))))
    writeLines(cl$url, file.path(out, sub("\\.txt$", ".url", f)))
  }
  status <- "ok"; n_top <- 0L; n_rising <- 0L
  if (inherits(r, "error")) {
    status <- "error"
  } else if (is.null(r$rq) || !NROW(r$rq)) {
    status <- "empty"
  } else {
    rq <- r$rq
    # gtrendsR's related_queries: 'subject' holds Google's value, 'value' the query text, and the top/rising
    # label sits in a column named 'related_topics' (extract_top_rising's timevar)
    stopifnot(all(c("subject", "related_topics", "value") %in% names(rq)), all(rq$related_topics %in% c("top", "rising")))
    df <- data.frame(window = w$label, time = w$time, keyword = keyword, list = rq$related_topics,
                     query = rq$value, google_value = rq$subject, stringsAsFactors = FALSE)
    df$rank <- ave(seq_len(nrow(df)), df$list, FUN = seq_along)
    df <- df[, c("window", "time", "keyword", "list", "rank", "query", "google_value")]
    write.csv(df, file.path(out, sprintf("related_queries_%s.csv", w$label)), row.names = FALSE)
    parsed[[w$label]] <- list(related_queries = rq, widget_ids = r$widget$id)
    n_top <- sum(df$list == "top"); n_rising <- sum(df$list == "rising")
  }
  log[[k]] <- data.frame(utc_time = if (!is.null(r$fetched_utc)) r$fetched_utc else utc(), window = w$label, keyword = keyword, time = w$time, geo = "worldwide",
                         gprop = "web", category = category, hl = hl, tz = tz, status = status, attempts = attempt,
                         n_top = n_top, n_rising = n_rising, n_http_calls = length(calls),
                         raw_files = paste(raw_files, collapse = ";"), raw_md5 = paste(raw_md5, collapse = ";"),
                         attempt_note = note, stringsAsFactors = FALSE)
  message(sprintf("%s: %s (top %d, rising %d, attempts %d)", w$label, status, n_top, n_rising, attempt))
}
fl <- do.call(rbind, log)
write.csv(fl, file.path(out, "fetch_log.csv"), row.names = FALSE)
saveRDS(parsed, file.path(out, "related_queries_parsed.rds"))
meta <- list(step = "08b_google_related_queries", written_at_utc = utc(), script_md5 = unname(tools::md5sum(self)),
             r_version = R.version.string, gtrendsR = as.character(packageVersion("gtrendsR")),
             curl = as.character(packageVersion("curl")), keyword = keyword, geo = "worldwide", hl = hl, tz = tz,
             windows = windows, n_ok = sum(fl$status == "ok"),
             files = lapply(setdiff(list.files(out), "fetch_meta.json"), function(f) list(file = f, md5 = unname(tools::md5sum(file.path(out, f))))))
writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE), file.path(out, "fetch_meta.json"))
