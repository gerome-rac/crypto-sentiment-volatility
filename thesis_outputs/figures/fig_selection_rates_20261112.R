#!/usr/bin/env Rscript
# fig_selection_rates_20261112.R
# How often each variable is selected across the 885 rolling estimation windows of the main specification:
# share of the windows in which the adaptive LASSO keeps at least one term of the variable, in the rung in which its
# block enters (controls and attention: R1; polarity: R2; emotions: R3). One bar per variable, grouped by block and
# sorted within the block. A diamond marks variables with a term that is significant in sample at the 1% level
# (two controls and the two attention terms of Table 5.3).
# Input: selection_rates_20261112.csv (block, variable, label, freq, sig_insample), written and checked by
#        selection_rates_20261112.py from the main run's selection_freq_variables_R{1,2,3}.csv.
# Style: thesis design system (navy #004B8C accent, grey #8C8C8C, ink #111111, Palatino), as fig_cum_loss_diff.
# Usage:  Rscript fig_selection_rates_20261112.R <input.csv> <out.pdf>
#         (run with ~/.claude-science/conda/envs/r/bin/Rscript; its cairo_pdf embeds Palatino)
suppressPackageStartupMessages({ library(readr); library(dplyr); library(ggplot2) })
if (!isTRUE(l10n_info()[["UTF-8"]])) invisible(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"))

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
d <- read_csv(args[1], col_types = cols(block = col_character(), variable = col_character(), label = col_character(),
                                        freq = col_double(), sig_insample = col_integer()))
blocks <- c("control", "attention", "polarity", "emotion")
stopifnot(nrow(d) == 31, setequal(unique(d$block), blocks), !anyNA(d), all(d$freq >= 0, d$freq <= 1),
          identical(as.integer(table(factor(d$block, blocks))), c(11L, 12L, 2L, 6L)), sum(d$sig_insample) == 4)

titles <- c(control = "Controls (R1)", attention = "Attention (R1)", polarity = "Polarity (R2)",
            emotion = "Emotions (R3)")
# Rows from top to bottom: a title row per block, then its variables by falling rate, then half a row of space.
d <- d |> mutate(block = factor(block, blocks), pct = 100 * freq) |> arrange(block, desc(freq))
y <- numeric(nrow(d)); head_y <- numeric(length(blocks)); pos <- 0
for (b in seq_along(blocks)) {
  pos <- pos - 1; head_y[b] <- pos
  i <- which(d$block == blocks[b])
  y[i] <- pos - seq_along(i); pos <- min(y[i]) - 0.45
}
d$y <- y
heads <- tibble(y = head_y, label = unname(titles[blocks]))

fam <- "Palatino"; ink <- "#111111"; navy <- "#004B8C"; grey <- "#8C8C8C"
fills <- c(control = grey, attention = navy, polarity = navy, emotion = navy)
half <- 0.31

p <- ggplot(d) +
  geom_vline(xintercept = seq(25, 100, 25), linewidth = 0.2, colour = "grey90") +
  geom_rect(aes(xmin = 0, xmax = pct, ymin = y - half, ymax = y + half, fill = block)) +
  geom_text(aes(x = pct + 1.2, y = y, label = sprintf("%.1f", pct)), hjust = 0, size = 2.75, family = fam, colour = ink) +
  geom_point(data = d |> filter(sig_insample == 1), aes(x = pct + 9.5, y = y, shape = "Significant in sample (1% level)"),
             size = 1.9, stroke = 0.35, colour = ink, fill = "white") +
  geom_text(data = heads, aes(x = 0, y = y, label = label), hjust = 0, vjust = 0.35, size = 2.95, family = fam,
            fontface = "bold", colour = ink) +
  scale_fill_manual(values = fills, guide = "none") +
  scale_shape_manual(values = c("Significant in sample (1% level)" = 23), name = NULL) +
  scale_x_continuous(breaks = seq(0, 100, 25), expand = c(0, 0)) +
  scale_y_continuous(breaks = d$y, labels = d$label, expand = c(0, 0)) +
  coord_cartesian(xlim = c(0, 108), ylim = c(min(d$y) - 0.7, -0.4), clip = "off") +
  labs(x = "Estimation windows in which the variable is selected (%)", y = NULL) +
  theme_classic(base_size = 9, base_family = fam) +
  theme(axis.line.y = element_line(linewidth = 0.3, colour = ink), axis.ticks.y = element_blank(),
        axis.line.x = element_line(linewidth = 0.3, colour = ink), axis.ticks.x = element_line(linewidth = 0.3, colour = ink),
        axis.ticks.length = unit(1.2, "mm"), axis.text.x = element_text(size = 8, colour = "grey20"),
        axis.text.y = element_text(size = 8.2, colour = ink, margin = margin(r = 1.5, unit = "mm")),
        axis.title.x = element_text(size = 8.5, colour = ink, margin = margin(t = 1.5, unit = "mm")),
        legend.position = "top", legend.justification = c(1, 0.5), legend.text = element_text(size = 8),
        legend.key.height = unit(3, "mm"), legend.margin = margin(0, 0, 0, 0),
        legend.box.margin = margin(0, 0, 0, 0, unit = "mm"),
        plot.margin = margin(t = 1, r = 3, b = 1, l = 1, unit = "mm"))

ggsave(args[2], p, device = cairo_pdf, width = 160, height = 150, units = "mm")
bb <- readBin(args[2], "raw", file.info(args[2])$size); s <- rawToChar(bb[bb != as.raw(0)])
fonts <- unique(regmatches(s, gregexpr("/BaseFont\\s*/[A-Za-z0-9+_-]+", s, useBytes = TRUE))[[1]])
stopifnot(length(fonts) > 0, all(grepl("Palatino", fonts)))
cat("rows:", nrow(d), "| y range:", range(d$y), "\n")
