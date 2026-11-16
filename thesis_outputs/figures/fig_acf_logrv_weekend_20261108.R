#!/usr/bin/env Rscript
# fig_acf_logrv_weekend_20261108.R
# Sample ACF of daily log RV per coin (lags 1-100): (a) unadjusted, as in Figure A.1; (b) after removing the weekend
# effect (residuals of log RV on coin intercepts and one common Saturday/Sunday indicator, as the model uses it).
# Light vertical guides mark the weekly lags 7, 14, ..., 98.
# Input: acf_logrv_weekend_20261108.csv, written by acf_logrv_weekend_20261108.py from har_panel_final_20261028.parquet.
# Style: as Figure A.1 (cj_fig_acf_logrv_20261022.pdf): Okabe-Ito coin colours, grey white-noise band, dashed AR(1)
#        reference with the same lag-1 autocorrelation; thesis design system (ink #111111, Palatino).
# Usage:  Rscript fig_acf_logrv_weekend_20261108.R <acf.csv> <out.pdf>
#         (run with ~/.claude-science/conda/envs/r/bin/Rscript; its cairo_pdf embeds Palatino)
suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr); library(ggplot2) })
if (!isTRUE(l10n_info()[["UTF-8"]])) invisible(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"))

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
d <- read_csv(args[1], col_types = cols(coin = col_character(), .default = col_double()))
coins <- c("BTC", "ETH", "BNB", "XRP", "DOGE")
stopifnot(nrow(d) == 500, setequal(d$coin, coins), all(d$n_days == 1280), !anyNA(d))

ver <- c(acf_raw = "(a) Unadjusted", acf_adj = "(b) Weekend effect removed")
long <- d |> pivot_longer(c(acf_raw, acf_adj), names_to = "key", values_to = "r") |>
  mutate(version = factor(ver[key], levels = ver), coin = factor(coin, levels = coins))
ar1 <- long |> filter(lag == 1) |> select(coin, version, r1 = r) |>
  crossing(lag = 1:30) |> mutate(r = r1^lag)
band <- 1.96 / sqrt(1280)

fam <- "Palatino"; ink <- "#111111"
coin_col <- c(BTC = "#000000", ETH = "#0072B2", BNB = "#009E73", XRP = "#CC79A7", DOGE = "#D55E00")
lab_ar1 <- "AR(1) with the same lag-1 autocorrelation"
lab_band <- "95% band under white noise"

p <- ggplot(long, aes(lag)) +
  geom_vline(xintercept = seq(7, 98, 7), linewidth = 0.2, colour = "grey88") +
  geom_rect(data = distinct(long, coin, version), inherit.aes = FALSE,
            aes(xmin = -Inf, xmax = Inf, ymin = -band, ymax = band, fill = lab_band)) +
  geom_hline(yintercept = 0, linewidth = 0.25, colour = "grey40") +
  geom_segment(aes(xend = lag, y = 0, yend = r, colour = coin), linewidth = 0.28, show.legend = FALSE) +
  geom_line(data = ar1, aes(y = r, linetype = lab_ar1), linewidth = 0.35, colour = "grey30") +
  geom_text(data = distinct(long, coin, version), inherit.aes = FALSE, aes(x = 100, y = 0.86, label = coin),
            hjust = 1, vjust = 1, size = 8 / .pt, family = fam, colour = ink) +
  facet_grid(coin ~ version) +
  scale_colour_manual(values = coin_col) +
  scale_fill_manual(values = setNames("grey88", lab_band), name = NULL) +
  scale_linetype_manual(values = setNames("22", lab_ar1), name = NULL) +
  scale_x_continuous(limits = c(0, 101), breaks = seq(0, 100, 20), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-0.09, 0.88), breaks = c(0, 0.4, 0.8), labels = c("0.0", "0.4", "0.8"),
                     expand = c(0, 0)) +
  labs(x = "Lag (days)", y = "Autocorrelation of daily log RV") +
  guides(linetype = guide_legend(order = 1), fill = guide_legend(order = 2)) +
  theme_classic(base_size = 9, base_family = fam) +
  theme(strip.background = element_blank(), strip.text.x = element_text(hjust = 0, size = 9, colour = ink),
        strip.text.y = element_blank(),
        axis.line = element_line(linewidth = 0.3, colour = ink), axis.ticks = element_line(linewidth = 0.3, colour = ink),
        axis.ticks.length = unit(1.2, "mm"), axis.text = element_text(size = 8, colour = "grey20"),
        axis.title = element_text(size = 8.5, colour = ink),
        panel.spacing.y = unit(2.5, "mm"), panel.spacing.x = unit(5, "mm"),
        legend.position = "top", legend.justification = c(0, 0.5), legend.direction = "horizontal",
        legend.text = element_text(size = 8), legend.key.width = unit(7, "mm"), legend.key.height = unit(3, "mm"),
        legend.key.spacing.x = unit(4, "mm"), legend.margin = margin(0, 0, 0, 0),
        legend.box.margin = margin(0, 0, 1, 0, unit = "mm"),
        plot.margin = margin(t = 1, r = 3, b = 1, l = 1, unit = "mm"))

ggsave(args[2], p, device = cairo_pdf, width = 160, height = 170, units = "mm")
bb <- readBin(args[2], "raw", file.info(args[2])$size); s <- rawToChar(bb[bb != as.raw(0)])
fonts <- unique(regmatches(s, gregexpr("/BaseFont\\s*/[A-Za-z0-9+_-]+", s, useBytes = TRUE))[[1]])
stopifnot(length(fonts) > 0, all(grepl("Palatino", fonts)))
# weekly bump: mean over k = 1..14 of r(7k) - (r(7k-1) + r(7k+1)) / 2
bump <- long |> group_by(coin, version) |>
  summarise(bump = mean(sapply(1:14, function(k) r[lag == 7 * k] - (r[lag == 7 * k - 1] + r[lag == 7 * k + 1]) / 2)),
            r1 = r[lag == 1], r100 = r[lag == 100], .groups = "drop")
print(bump |> pivot_wider(names_from = version, values_from = c(bump, r1, r100)), width = 200)
