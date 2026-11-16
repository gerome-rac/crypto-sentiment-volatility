#!/usr/bin/env Rscript
# fig_robustness_pvalues_20261110.R
# The test of each subquestion in the main specification and in the 44 robustness runs (MedRV grid of 8 Nov 2026):
# unadjusted p-values on a log scale, one row per subquestion, one point per run.
#   SQ1-SQ3: one-sided Clark-West p of the step R0 -> R1, R1 -> R2, R2 -> R3.
#   SQ4:     two-sided Driscoll-Kraay p of fear minus anger (42 runs; two runs have no such test).
# Runs that leave the models of a test unchanged repeat the main result by construction (same p to 1e-12). They are
# not drawn as points; their number is printed at the right of the row.
# Input: robustness_pvalues_20261110.csv (columns sq, run, p, kind = main | run | repeat), extracted on 10 Nov 2026 from
#        code/data/pipeline/model_runs/_arm_results_medrv_20261108.csv (the 44 runs) and the main run's
#        confirmatory_holm.csv; the main two-sided SQ4 p is the value of the window runs, which leave the in-sample
#        fit unchanged.
# Style: thesis design system (navy #004B8C accent, grey #8C8C8C, ink #111111, Palatino), as fig_cum_loss_diff.
# Usage:  Rscript fig_robustness_pvalues_20261110.R <input.csv> <out.pdf>
#         (run with ~/.claude-science/conda/envs/r/bin/Rscript; its cairo_pdf embeds Palatino)
suppressPackageStartupMessages({ library(readr); library(dplyr); library(ggplot2) })
if (!isTRUE(l10n_info()[["UTF-8"]])) invisible(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"))

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
d <- read_csv(args[1], col_types = cols(sq = col_character(), run = col_character(), p = col_double(),
                                        kind = col_character()))
sqs <- c("SQ1", "SQ2", "SQ3", "SQ4")
stopifnot(setequal(unique(d$sq), sqs), all(d$kind %in% c("main", "run", "repeat")), !anyNA(d$p), all(d$p > 0, d$p <= 1),
          sum(d$kind == "main") == 4)
n_runs <- d |> filter(kind != "main") |> count(sq) |> arrange(sq)
stopifnot(identical(n_runs$n, c(44L, 44L, 44L, 42L)))

floor_p <- 1e-5
d <- d |> mutate(x = log10(pmax(p, floor_p)), row = 5 - match(sq, sqs))

# Deterministic swarm: points of one row are placed at the level closest to the row line that is still free.
# Where a row is too dense for the available levels (SQ3 near p = 0.9), the point goes to the level on which its
# nearest neighbour is farthest away, so that the circles overlap as little as possible.
dx <- 0.07; step <- 0.1; max_lev <- 4
swarm <- function(x, is_main) {
  lev <- rep(NA_integer_, length(x))
  cands <- c(0L, as.integer(rbind(seq_len(max_lev), -seq_len(max_lev))))
  ord <- order(!is_main, x)                       # the main specification first, on the row line
  for (i in ord) {
    done <- which(!is.na(lev))
    gap <- ifelse(is_main[done], 1.7 * dx, dx)     # keep clear of the larger diamond
    nearest <- vapply(cands, function(cand) {
      on <- done[lev[done] == cand]
      if (length(on) == 0) Inf else min(abs(x[on] - x[i]) / gap[lev[done] == cand])
    }, numeric(1))
    free <- which(nearest >= 1)
    lev[i] <- if (length(free) > 0) cands[free[1]] else cands[which.max(nearest)]
  }
  stopifnot(!anyNA(lev))
  lev
}
pts <- d |> filter(kind != "repeat") |> group_by(sq) |> mutate(lev = swarm(x, kind == "main")) |> ungroup() |>
  mutate(y = row + lev * step)

cnt <- d |> filter(kind != "main") |> group_by(sq, row) |>
  summarise(n = n(), below = sum(p < 0.05), rep = sum(kind == "repeat"), .groups = "drop") |>
  mutate(l1 = sprintf("%d of %d runs below 0.05", below, n),
         l2 = sprintf("%d repeat the main result by construction", rep))
print(cnt |> select(sq, n, below, rep))
stopifnot(identical(cnt$below, c(44L, 32L, 3L, 5L)), identical(cnt$rep, c(20L, 7L, 2L, 4L)))

fam <- "Palatino"; ink <- "#111111"; navy <- "#004B8C"; grey <- "#8C8C8C"
ylab <- c("SQ4: fear and anger", "SQ3: emotions", "SQ2: polarity", "SQ1: attention")
brk <- -5:0
xlab <- c("≤ 0.00001", "0.0001", "0.001", "0.01", "0.1", "1")
tx <- 0.16                                          # x position of the counts, right of p = 1

p <- ggplot() +
  geom_hline(yintercept = 1:4, linewidth = 0.2, colour = "grey90") +
  annotate("segment", x = log10(0.05), xend = log10(0.05), y = 0.45, yend = 4.62, linewidth = 0.3,
           linetype = "22", colour = ink) +
  annotate("text", x = log10(0.05) + 0.04, y = 4.62, label = "p = 0.05", hjust = 0, vjust = 0.3, size = 2.7,
           family = fam, colour = ink) +
  geom_point(data = pts |> filter(kind == "run"), aes(x, y, shape = "Robustness run"), size = 1.35, stroke = 0.35,
             colour = ink, fill = "white") +
  geom_point(data = pts |> filter(kind == "main"), aes(x, y, shape = "Main specification"), size = 2.9, stroke = 0.3,
             colour = navy, fill = navy) +
  geom_text(data = cnt, aes(x = tx, y = row + 0.14, label = l1), hjust = 0, size = 2.85, family = fam, colour = ink) +
  geom_text(data = cnt, aes(x = tx, y = row - 0.16, label = l2), hjust = 0, size = 2.6, family = fam, colour = grey) +
  scale_shape_manual(values = c("Main specification" = 23, "Robustness run" = 21), name = NULL) +
  guides(shape = guide_legend(override.aes = list(size = c(2.6, 1.6), fill = c(navy, "white"), colour = c(navy, ink)))) +
  scale_x_continuous(breaks = brk, labels = xlab, expand = c(0, 0)) +
  scale_y_continuous(breaks = 1:4, labels = ylab, expand = c(0, 0)) +
  coord_cartesian(xlim = c(-5.15, 0), ylim = c(0.45, 4.7), clip = "off") +   # limits here, so the counts right of p = 1 are kept
  labs(x = "p-value of the test (log scale, unadjusted)", y = NULL) +
  theme_classic(base_size = 9, base_family = fam) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        axis.line.x = element_line(linewidth = 0.3, colour = ink), axis.ticks.x = element_line(linewidth = 0.3, colour = ink),
        axis.ticks.length = unit(1.2, "mm"), axis.text.x = element_text(size = 8, colour = "grey20"),
        axis.text.y = element_text(size = 8.5, colour = ink, margin = margin(r = 2, unit = "mm")),
        axis.title.x = element_text(size = 8.5, colour = ink, margin = margin(t = 1.5, unit = "mm")),
        legend.position = "top", legend.justification = c(0, 0.5), legend.direction = "horizontal",
        legend.text = element_text(size = 8), legend.key.width = unit(5, "mm"), legend.key.height = unit(3, "mm"),
        legend.key.spacing.x = unit(4, "mm"), legend.margin = margin(0, 0, 0, 0),
        legend.box.margin = margin(0, 0, 1, 0, unit = "mm"),
        plot.margin = margin(t = 1, r = 54, b = 1, l = 1, unit = "mm"))

ggsave(args[2], p, device = cairo_pdf, width = 160, height = 68, units = "mm")
bb <- readBin(args[2], "raw", file.info(args[2])$size); s <- rawToChar(bb[bb != as.raw(0)])
fonts <- unique(regmatches(s, gregexpr("/BaseFont\\s*/[A-Za-z0-9+_-]+", s, useBytes = TRUE))[[1]])
stopifnot(length(fonts) > 0, all(grepl("Palatino", fonts)))
cat("max swarm level used:", max(abs(pts$lev)), "\n")
