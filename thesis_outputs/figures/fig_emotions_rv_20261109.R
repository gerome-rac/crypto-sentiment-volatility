#!/usr/bin/env Rscript
# fig_emotions_rv_20261109.R
# Bitcoin, 1 Aug 2019 - 31 Jan 2023: (a) daily log realized volatility; (b) pooled share of fear and anger, 7-day means.
# Input: emotions_rv_btc_20261109.csv, written by latex/tables/src/res_tables_20261109.py from the MedRV panel.
# Colours: Okabe-Ito blue and vermillion for fear and anger (palette validated in light mode, all checks pass; both
#          lines are also labelled directly). Style: thesis design system (ink #111111, Palatino).
# Usage:  Rscript fig_emotions_rv_20261109.R <input.csv> <out.pdf>   (run with ~/.claude-science/conda/envs/r/bin/Rscript)
suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr); library(ggplot2) })
if (!isTRUE(l10n_info()[["UTF-8"]])) invisible(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"))
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
d <- read_csv(args[1], col_types = cols(day = col_date(), .default = col_double()))
stopifnot(nrow(d) == 1280, !anyNA(d$log_rv))
fam <- "Palatino"; ink <- "#111111"
pa <- "(a) Daily log realized volatility"; pb <- "(b) Fear and anger, share of posts in percent, 7-day mean"
panels <- c(pa, pb)
rv <- d |> transmute(day, value = log_rv, series = "log RV", panel = factor(pa, panels))
em <- d |> select(day, Fear = fear_7d, Anger = anger_7d) |> pivot_longer(-day, names_to = "series", values_to = "value") |>
  filter(!is.na(value)) |> mutate(value = 100 * value, panel = factor(pb, panels))
last <- em |> group_by(series) |> filter(day == max(day)) |> ungroup()
ev <- tibble(day = as.Date(c("2020-03-12", "2022-05-09", "2022-11-06")), label = c("COVID crash", "Terra", "FTX"))
evp <- crossing(ev, panel = factor(panels, panels))
top <- rv |> summarise(y = max(value)) |> pull(y)
col <- c(Fear = "#0072B2", Anger = "#D55E00")
p <- ggplot() +
  geom_vline(data = evp, aes(xintercept = day), linewidth = 0.25, linetype = "22", colour = "grey55") +
  geom_text(data = ev |> mutate(panel = factor(pa, panels)), aes(x = day + 8, y = top, label = label), hjust = 0, vjust = 1,
            size = 7.5 / .pt, family = fam, colour = ink) +
  geom_line(data = rv, aes(day, value), linewidth = 0.25, colour = ink) +
  geom_line(data = em, aes(day, value, colour = series), linewidth = 0.45) +
  geom_text(data = last, aes(day + 10, value, label = series), hjust = 0, size = 7.5 / .pt, family = fam, colour = ink) +
  facet_wrap(~panel, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = col, breaks = c("Fear", "Anger"), name = NULL) +
  scale_x_date(date_breaks = "6 months", date_labels = "%b %Y", expand = expansion(mult = c(0.005, 0.06))) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 8, base_family = fam) +
  theme(text = element_text(colour = ink), axis.text = element_text(colour = ink), strip.text = element_text(hjust = 0, colour = ink, size = 8),
        panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(), panel.grid.major.y = element_line(linewidth = 0.2, colour = "grey90"),
        legend.position = "bottom", legend.margin = margin(0, 0, 0, 0), legend.box.margin = margin(-6, 0, 0, 0),
        panel.spacing.y = unit(6, "pt"), plot.margin = margin(2, 4, 2, 2))
ggsave(args[2], p, width = 6.89, height = 3.5, device = cairo_pdf)
