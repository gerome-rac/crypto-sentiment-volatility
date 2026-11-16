#!/usr/bin/env Rscript
# cv_fig_blocked_cv_20261107.R
# One figure for the blocked cross-validation, combining panel (e) of cv_fig_schemes_20261028_v2.pdf
# (blocked K-fold with h removed on each side of the validation block) and panel (b) of
# cv_fig_design_20261028_v2.pdf (the thesis geometry inside one 365-day window):
# five contiguous 73-day validation blocks, h = 30 days removed on each side, all five coins of a day together;
# numbers = days per segment; right margin = training rows (days x 5 coins). Same geometry in all 885 windows.
# Input: cv_folds_window365_20261107.csv (fold, day_in_window, date_window1, role), written on 7 Nov 2026 by
#        cv_fold_geometry_20261028.py (this folder) from the engine's own day_folds() (code/model/lasso.py) and
#        the EngineConfig defaults (code/model/config.py: blocked, 5 folds, 30-day gap).
# Style: thesis design system (training grey #8C8C8C, validation navy #004B8C, removed #E0E0E0, ink #111111,
#        Palatino), as in cv_fig_rolling_windows_20261106.R.
# Usage:  Rscript cv_fig_blocked_cv_20261107.R <cv_folds_window365.csv> <out.pdf>
#         (run with ~/.claude-science/conda/envs/r/bin/Rscript; its cairo_pdf embeds Palatino)
suppressPackageStartupMessages({ library(readr); library(dplyr); library(ggplot2) })
if (!isTRUE(l10n_info()[["UTF-8"]])) invisible(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"))

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
W <- 365; K <- 5; H <- 30; N_COINS <- 5
roles <- c("Training", "Validation", "Removed (buffer)")
fd <- read_csv(args[1], col_types = cols(fold = col_integer(), day_in_window = col_integer(),
                                         date_window1 = col_date(), role = col_character()))
stopifnot(nrow(fd) == K * W, setequal(unique(fd$role), roles),
          all(fd |> count(fold) |> pull(n) == W))

# run-length segments of each fold's roles
seg <- fd |> arrange(fold, day_in_window) |> group_by(fold) |>
  mutate(run = cumsum(role != lag(role, default = first(role)))) |>
  group_by(fold, run, role) |>
  summarise(x0 = min(day_in_window), x1 = max(day_in_window), .groups = "drop") |>
  mutate(len = x1 - x0 + 1L, xm = (x0 + x1) / 2, y = K + 1 - fold)
val <- seg |> filter(role == "Validation") |> arrange(fold)
stopifnot(nrow(val) == K, all(val$len == W / K),                         # one 73-day block per fold
          all(val$x0 == (seq_len(K) - 1) * W / K + 1),                    # blocks in time order, no overlap
          all(seg$len[seg$role == "Removed (buffer)"] == H),             # h = 30 on each side
          all(seg |> group_by(fold) |> summarise(n = sum(len)) |> pull(n) == W))
rows <- fd |> filter(role == "Training") |> count(fold, name = "days") |> arrange(fold) |>
  mutate(rows = days * N_COINS)
stopifnot(identical(rows$days, c(262L, 232L, 232L, 232L, 262L)))

fam <- "Palatino"; pt <- ggplot2::.pt; ink <- "#111111"
pal <- c("Training" = "#8C8C8C", "Validation" = "#004B8C", "Removed (buffer)" = "#E0E0E0")
# plotmath: italic(h) in Palatino-Italic, the rest as plain strings (the == operator would pull in the Symbol font)
lab <- seg |> mutate(txt = case_when(role == "Removed (buffer)" ~ 'italic(h)~"="~"30"',
                                     TRUE ~ paste0('"', len, '"')),
                     col = ifelse(role == "Validation", "white", ink))

p <- ggplot(seg) +
  geom_rect(aes(xmin = x0 - 0.5, xmax = x1 + 0.5, ymin = y - 0.36, ymax = y + 0.36, fill = role),
            colour = "white", linewidth = 0.35) +
  geom_text(data = lab, aes(x = xm, y = y, label = txt, colour = col), parse = TRUE,
            family = fam, size = 7 / pt) +
  scale_fill_manual(values = pal, limits = roles, name = NULL) +
  scale_colour_identity() +
  scale_x_continuous(breaks = c(1, (1:K) * W / K), expand = c(0, 0)) +
  scale_y_continuous(breaks = K:1, labels = paste("Fold", 1:K), expand = c(0, 0), limits = c(0.5, K + 0.5),
                     sec.axis = dup_axis(labels = paste(format(rows$rows, big.mark = ","), "rows"))) +
  coord_cartesian(xlim = c(0.5, W + 0.5), clip = "off") +
  labs(x = "Day of the estimation window", y = NULL) +
  theme_classic(base_size = 9, base_family = fam) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        axis.line.x = element_line(linewidth = 0.3, colour = ink),
        axis.ticks.x = element_line(linewidth = 0.3, colour = ink),
        axis.ticks.length = unit(1.2, "mm"),
        axis.text = element_text(size = 8, colour = "grey20"),
        axis.text.y.right = element_text(size = 8, colour = "grey20", margin = margin(l = 1, unit = "mm")),
        axis.title.x = element_text(size = 8.5, colour = ink, margin = margin(t = 1.5, unit = "mm")),
        legend.position = "top", legend.justification = c(0, 0.5), legend.direction = "horizontal",
        legend.text = element_text(size = 8), legend.key.width = unit(7, "mm"), legend.key.height = unit(3, "mm"),
        legend.key.spacing.x = unit(4, "mm"), legend.spacing.x = unit(4, "mm"),
        legend.margin = margin(0, 0, 0, 0), legend.box.margin = margin(0, 0, 1, 0, unit = "mm"),
        plot.margin = margin(t = 1, r = 2, b = 1, l = 1, unit = "mm"))

ggsave(args[2], p, device = cairo_pdf, width = 160, height = 52, units = "mm")
bb <- readBin(args[2], "raw", file.info(args[2])$size); s <- rawToChar(bb[bb != as.raw(0)])
fonts <- unique(regmatches(s, gregexpr("/BaseFont\\s*/[A-Za-z0-9+_-]+", s, useBytes = TRUE))[[1]])
stopifnot(length(fonts) > 0, all(grepl("Palatino", fonts)))
cat("training days per fold:", paste(rows$days, collapse = ", "), "| rows:", paste(rows$rows, collapse = ", "), "\n")
