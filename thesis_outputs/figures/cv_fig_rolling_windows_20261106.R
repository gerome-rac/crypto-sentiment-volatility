#!/usr/bin/env Rscript
# cv_fig_rolling_windows_20261106.R
# Panel (a) of Figure 4.2 (cv_fig_design_20261028_v2.pdf) redrawn as a figure of its own:
# the rolling 365-day estimation windows (365 days x 5 coins = 1,825 rows), each with its one-day-ahead
# forecast target; 885 targets, 30 Aug 2020 - 31 Jan 2023; windows shown at half-year steps.
# Input: cv_windows_20261028.csv (window, first_origin, last_origin, forecast_origin, target, n_days), written by
#        cv_fold_geometry_20261028.py from the engine's own schedule (model/oos.py); copied from the Claude
#        Science artifact of 28 Oct 2026. Same input and geometry as cv_fig_design_20261028.R, panel (a).
# Style: thesis design system as in the v2 figure (training grey #8c8c8c, ink markers, Palatino).
# Usage:  Rscript cv_fig_rolling_windows_20261106.R <cv_windows.csv> <out.pdf>
suppressPackageStartupMessages({ library(readr); library(dplyr); library(ggplot2) })
if (!isTRUE(l10n_info()[["UTF-8"]])) invisible(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"))

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
W <- 365
win <- read_csv(args[1], col_types = cols(window = col_integer(), n_days = col_integer(), .default = col_date()))
stopifnot(nrow(win) == 885, all(win$n_days == W), all(diff(win$target) == 1),
          win$target[1] == as.Date("2020-08-30"), win$target[885] == as.Date("2023-01-31"),
          win$first_origin[1] == as.Date("2019-08-30"), all(win$last_origin - win$first_origin == W - 1))

fam <- "Palatino"; pt <- ggplot2::.pt
grey_tr <- "#8C8C8C"; ink <- "#111111"
show_t <- as.Date(c("2020-08-30", "2021-02-28", "2021-08-30", "2022-02-28", "2022-08-30", "2023-01-31"))
wa <- win |> filter(target %in% show_t) |> arrange(window) |> mutate(y = rev(seq_len(n())))
stopifnot(nrow(wa) == length(show_t))
fmt_d <- function(d) paste(as.integer(format(d, "%d")), month.abb[as.integer(format(d, "%m"))], format(d, "%Y"))
wa <- wa |> mutate(ylab = paste("Window", prettyNum(window, big.mark = ",")), tlab = fmt_d(target))
x_lim <- as.Date(c("2019-08-01", "2023-02-01"))
brk <- seq(as.Date("2019-08-01"), as.Date("2023-02-01"), by = "6 months")
lab <- paste(month.abb[as.integer(format(brk, "%m"))], format(brk, "%Y"))
first_t <- as.Date("2020-08-30")

p <- ggplot(wa) +
  annotate("segment", x = first_t, xend = first_t, y = 0.45, yend = max(wa$y) + 0.75,
           linetype = "22", linewidth = 0.3, colour = "grey45") +
  annotate("text", x = first_t + 4, y = max(wa$y) + 0.95, hjust = 0, vjust = 0.5, family = fam,
           size = 8 / pt, colour = "grey35", label = "885 forecast targets, 30 Aug 2020 – 31 Jan 2023") +
  geom_rect(aes(xmin = first_origin - 0.5, xmax = last_origin + 0.5, ymin = y - 0.32, ymax = y + 0.32,
                fill = "Training window")) +
  geom_point(aes(x = target, y = y, shape = "Forecast target"), size = 2.3, colour = ink) +
  annotate("text", x = wa$first_origin[1] + (W - 1) / 2, y = max(wa$y), label = "365 days × 5 coins",
           family = fam, size = 8 / pt, colour = ink) +
  scale_fill_manual(values = c("Training window" = grey_tr), name = NULL) +
  scale_shape_manual(values = c("Forecast target" = 18), name = NULL) +
  scale_x_date(limits = x_lim, breaks = brk, labels = lab, expand = c(0, 0)) +
  scale_y_continuous(breaks = wa$y, labels = wa$ylab, limits = c(0.45, max(wa$y) + 1.25), expand = c(0, 0),
                     sec.axis = dup_axis(labels = wa$tlab)) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL) +
  theme_classic(base_size = 9, base_family = fam) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        axis.line.x = element_line(linewidth = 0.3, colour = ink),
        axis.ticks.x = element_line(linewidth = 0.3, colour = ink),
        axis.ticks.length = unit(1.2, "mm"),
        axis.text = element_text(size = 8, colour = "grey20"),
        axis.text.y.right = element_text(size = 8, colour = "grey20", margin = margin(l = 1, unit = "mm")),
        legend.position = "top", legend.justification = c(0, 0.5), legend.direction = "horizontal",
        legend.text = element_text(size = 8), legend.key.width = unit(7, "mm"), legend.key.height = unit(3, "mm"),
        legend.key.spacing.x = unit(4, "mm"), legend.spacing.x = unit(4, "mm"),
        legend.margin = margin(0, 0, 0, 0), legend.box.margin = margin(0, 0, 1, 0, unit = "mm"),
        plot.margin = margin(t = 1, r = 2, b = 1, l = 1, unit = "mm")) +
  guides(fill = guide_legend(order = 1), shape = guide_legend(order = 2, override.aes = list(size = 2.6)))

ggsave(args[2], p, device = cairo_pdf, width = 160, height = 62, units = "mm")
bb <- readBin(args[2], "raw", file.info(args[2])$size); s <- rawToChar(bb[bb != as.raw(0)])
fonts <- unique(regmatches(s, gregexpr("/BaseFont\\s*/[A-Za-z0-9+_-]+", s, useBytes = TRUE))[[1]])
stopifnot(length(fonts) > 0, all(grepl("Palatino", fonts)))
cat("windows shown:", paste(wa$window, collapse = ", "), "| targets:", paste(wa$tlab, collapse = ", "), "\n")
