#!/usr/bin/env Rscript
# fig_cum_loss_R1_compare.R
# Tether check of 14 Nov 2026: cumulative loss difference of the attention step (R0 - R1) over the 885 forecast days,
# for the main specification and the two arms (A85: Tether terms in no rung; A86: Tether terms enter with attention).
# Daily cross-coin mean losses: (a) squared error of the log forecast, (b) QLIKE. Positive and rising = R1 forecasts
# better than R0 over that stretch.
# Input: cum_loss_R1_compare.csv, written and checked by tether_check.py.
# Style: as latex/figures/src/fig_cum_loss_diff_20261110.R (thesis design system).
# Usage:  Rscript fig_cum_loss_R1_compare.R <cum_loss_R1_compare.csv> <out.pdf>
suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr); library(ggplot2) })
if (!isTRUE(l10n_info()[["UTF-8"]])) invisible(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"))

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
d <- read_csv(args[1], col_types = cols(target = col_date(), .default = col_double()))
stopifnot(nrow(d) == 885, d$target[1] == as.Date("2020-08-30"), d$target[885] == as.Date("2023-01-31"),
          all(diff(d$target) == 1))

long <- d |> pivot_longer(-target, names_to = "key", values_to = "v") |>
  mutate(loss = ifelse(grepl("^dse_", key), "(a) Squared error of the log forecast (MSE)", "(b) QLIKE"),
         run = sub("^(dse|dqlike)_", "", key))
runs <- c("main", "A85", "A86")
stopifnot(setequal(unique(long$run), runs), !anyNA(long$v))
long$run <- factor(long$run, runs)

fam <- "Palatino"; ink <- "#111111"
cols <- c(main = "#004B8C", A85 = "#8C8C8C", A86 = ink)
ltys <- c(main = "solid", A85 = "22", A86 = "13")
labs_r <- c(main = "Main: Tether terms in R0", A85 = "Tether terms in no rung",
            A86 = "Tether terms enter with attention (R1)")
brk <- seq(as.Date("2020-08-01"), as.Date("2023-02-01"), by = "6 months")
lab <- paste(month.abb[as.integer(format(brk, "%m"))], format(brk, "%Y"))

p <- ggplot(long, aes(target, v, colour = run, linetype = run)) +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey55") +
  geom_line(linewidth = 0.55) +
  facet_wrap(~loss, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = cols, labels = labs_r, name = NULL) +
  scale_linetype_manual(values = ltys, labels = labs_r, name = NULL) +
  scale_x_date(limits = as.Date(c("2020-08-01", "2023-02-01")), breaks = brk, labels = lab, expand = c(0, 0)) +
  labs(x = NULL, y = "Cumulative loss difference, R0 minus R1") +
  theme_classic(base_size = 9, base_family = fam) +
  theme(strip.background = element_blank(), strip.text = element_text(hjust = 0, size = 9, colour = ink),
        axis.line = element_line(linewidth = 0.3, colour = ink), axis.ticks = element_line(linewidth = 0.3, colour = ink),
        axis.ticks.length = unit(1.2, "mm"), axis.text = element_text(size = 8, colour = "grey20"),
        axis.title.y = element_text(size = 8.5, colour = ink, margin = margin(r = 1.5, unit = "mm")),
        panel.grid.major.y = element_line(linewidth = 0.2, colour = "grey90"),
        panel.spacing = unit(3, "mm"),
        legend.position = "top", legend.justification = c(0, 0.5), legend.direction = "horizontal",
        legend.text = element_text(size = 8), legend.key.width = unit(9, "mm"), legend.key.height = unit(3, "mm"),
        legend.key.spacing.x = unit(4, "mm"), legend.margin = margin(0, 0, 0, 0),
        legend.box.margin = margin(0, 0, 1, 0, unit = "mm"),
        plot.margin = margin(t = 1, r = 8, b = 1, l = 1, unit = "mm"))

ggsave(args[2], p, device = cairo_pdf, width = 160, height = 100, units = "mm")
bb <- readBin(args[2], "raw", file.info(args[2])$size); s <- rawToChar(bb[bb != as.raw(0)])
fonts <- unique(regmatches(s, gregexpr("/BaseFont\\s*/[A-Za-z0-9+_-]+", s, useBytes = TRUE))[[1]])
stopifnot(length(fonts) > 0, all(grepl("Palatino", fonts)))
fin <- long |> filter(target == max(target)) |> arrange(loss, run)
print(fin |> select(loss, run, v), n = 6)
