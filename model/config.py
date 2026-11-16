"""Constants and named parameters of the estimation/evaluation engine.

Every construction detail that the frozen documents (planning/pre_analysis_note_D8.md,
MASTERPLAN_NOV15.md section 7, planning/final_test_plan.md) do not settle is a single
named field of ``EngineConfig``; its default is the most literal reading, and the field
is listed as an open point (OP-...) in code/data/pipeline/panel/reports/engine_report_20261007.md.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Optional

import pandas as pd

COINS = ("BTC", "ETH", "BNB", "XRP", "DOGE")
SAMPLE_FIRST_DAY = pd.Timestamp("2019-08-01")
SAMPLE_LAST_DAY = pd.Timestamp("2023-01-31")
N_DAYS = 1280                                   # UTC calendar days, 24/7 market
FIRST_ORIGIN = pd.Timestamp("2019-08-30")       # first day with a full 30-day HAR average
LAST_ORIGIN = pd.Timestamp("2023-01-30")        # last day with an observed next-day target
FIRST_TARGET = pd.Timestamp("2020-08-30")       # D8 note section 3
LAST_TARGET = pd.Timestamp("2023-01-31")
N_FORECAST_DAYS = 885                           # per coin; 4,425 forecasts per rung
PRE_FORECAST_YEAR = (pd.Timestamp("2019-08-30"), pd.Timestamp("2020-08-29"))
HAR_LAGS = (1, 7, 30)

# blocks entering each rung. Deviation 3 of the D8 note (7 Oct 2026, Gerome): the controls (penalised)
# are in every rung including R0, so each step adds only its own block. HARRV = HAR-RV reference in
# the MCS; HARCJ = pure HAR-CJ without controls, reported as a reference outside the MCS.
RUNG_BLOCKS = {
    "HARRV": ("har_rv",),
    "HARCJ": ("har",),
    "R0": ("har", "control"),
    "R1": ("har", "control", "attention"),
    "R2": ("har", "control", "attention", "polarity"),
    "R3": ("har", "control", "attention", "polarity", "emotion"),
}
# arm A71: the placement written in the D8 note at the freeze (controls enter with attention at R1)
RUNG_BLOCKS_CONTROLS_FROM_R1 = {
    "R0": ("har",),
    "R1": ("har", "attention", "control"),
    "R2": ("har", "attention", "control", "polarity"),
    "R3": ("har", "attention", "control", "polarity", "emotion"),
}
REFERENCE_MODELS = ("HARRV", "HARCJ")
MCS_EXCLUDE = ("HARCJ",)                         # reported in the loss summary, not in the MCS
LADDER = ("R0", "R1", "R2", "R3")
UNPENALISED_BLOCKS = ("har", "har_rv")          # HAR terms unpenalised; coin FE always unpenalised
# (small, big, label); the first three are the confirmatory H1-H3 tests (D8 note section 4)
NESTED_COMPARISONS = (("R0", "R1", "H1"), ("R1", "R2", "H2"), ("R2", "R3", "H3"), ("R1", "R3", "R1vsR3"))

# Register arms the engine itself implements. Arms that change a data rule (A10, A58, A59, ...)
# are built by the panel tracks and passed in as their own panel, run with --label.
REQUIRED = "<required>"     # marks an arm setting that must be passed on the command line
SPEC_ARMS = {
    "A00": dict(window=365, unit="pooled"),
    "A06": dict(window=250, unit="pooled"),
    "A55": dict(window="expanding", unit="pooled"),
    "A01": dict(window=365, unit="per_coin"),
    "A71": dict(window=365, unit="pooled", rung_blocks="controls_from_R1"),
    # 28 Oct 2026 (engine-arms track; register rows A61, A62, A53):
    # A61: QLIKE back-transformation by Duan (1983) smearing (evaluation only; the forecasts equal A00's)
    "A61": dict(window=365, unit="pooled", qlike_backtransform="duan"),
    # A62: evaluation target = subsampled 5-minute RV (Patton 2011 proxy; evaluation only). The column depends
    # on the edge rule, which is open (library choice O3: E1 | E2 | E3), so it must be passed: --eval-target.
    "A62": dict(window=365, unit="pooled", eval_target=REQUIRED),
    # A53: ABD (2007, fn 9) drop rule: the flagged outage coin-days are dropped (see har.FeatureBuilder)
    "A53": dict(window=365, unit="pooled", drop_days_flag="abd_drop_flag"),
}

RIDGE_GRID_REL = tuple(10.0 ** (-4 + 0.4 * i) for i in range(16))   # ridge alpha / n_rows


@dataclass
class EngineConfig:
    spec: str = "A00"
    label: str = ""
    window: object = 365                 # int (rolling days) or "expanding"
    unit: str = "pooled"                 # "pooled" (common slopes + coin FE) or "per_coin" (A01)
    rung_blocks: str = "main"            # "main" (Deviation 3: controls in every rung) | "controls_from_R1" (A71)
    first_target: str = "2020-08-30"     # OP-WIN: same forecast coin-days for every rung and arm
    last_target: str = "2023-01-31"
    estimator: str = "adaptive_lasso"    # "adaptive_lasso" | "enet" | "ols"; rungs without penalised columns are OLS
    # --- adaptive LASSO / CV (OP-CV, OP-GAMMA, OP-RIDGE) ---
    gamma: float = 1.0                   # adaptive weight exponent, w_j = 1/|b_ridge_j|^gamma (Zou 2006)
    cv_scheme: str = "blocked"           # 7 Oct 2026 (Gerome): blocked 5-fold with a gap, as Audrino et al. (2020); "forward" = forward-chaining
    cv_folds: int = 5
    cv_gap_days: int = 30                # days dropped on each side of a validation block (30 = longest HAR average; own choice)
    cv_rule: str = "min"                 # "min" | "1se"
    n_lambda: int = 50
    lambda_eps: float = 1e-3             # lambda_min = eps * lambda_max
    ridge_grid_rel: tuple = RIDGE_GRID_REL
    ridge_alpha_cv: bool = True          # first-stage ridge penalty by the same time-series CV
    ridge_alpha_fixed_rel: float = 1e-2  # used when ridge_alpha_cv is False
    enet_l1_ratio: float = 0.5           # elastic net robustness option (test 3.11)
    # --- in-window transforms (OP-CENTRE, OP-IMPUTE, OP-NAN, OP-A10, OP-WEIGHT) ---
    centre_scope: str = "per_coin"       # 7 Oct 2026 (Gerome): each coin's own window mean; "pooled" = over all coin-days
    impute_stat: str = "count_mean"      # zero-capture Twitter days: window mean of the count, or "log1p_mean"
    nan_policy: str = "window_mean"      # NaN regressor -> window mean (0 after z-scoring); "drop" drops the row
    composition_adjustment: Optional[str] = None   # A10 hook: None | "level_shift" (needs <var>_2p and <flag>)
    composition_flag: str = "twitter_absent"
    composition_full_flag: str = "twitter_full_capture"
    weight_col: Optional[str] = None     # regression weights (only if D16 precision weights are regression weights)
    # --- sample rules (28 Oct 2026; arm A53; None = A00) ---
    # drop_days_flag: boolean panel column; a flagged coin-day is dropped: the daily values of the variables in
    # drop_blocks (the price-based volatility measures) are missing on it, so the 7/30-day averages run over the
    # observed days, and no row is formed whose origin day or target day is flagged (training, in-sample, forecasts).
    drop_days_flag: Optional[str] = None
    drop_blocks: tuple = ("har", "har_rv")
    # --- losses ---
    qlike_backtransform: str = "lognormal"   # exp(f + s2/2); "duan" = Duan (1983) smearing (robustness)
    # eval_target (28 Oct 2026; arm A62): column whose next-day value is the EVALUATION target (losses, CW, DM, MCS);
    # the estimation target stays log rv_{t+1}. "rv" = A00 (evaluation target = estimation target).
    eval_target: str = "rv"
    # --- tests (OP-HAC, OP-DM, OP-MCS, OP-PANELAGG) ---
    panel_agg: str = "daymean"           # loss series for CW/DM/MCS/GW: cross-coin mean per day (T = 885)
    hac_lags: Optional[int] = None       # None -> floor(4 (T/100)^(2/9)) Bartlett lags (= 6 at T = 885)
    dm_variance: str = "hac"             # "hac" (Bartlett) with the HLN factor, or "hln" (autocovariances < h)
    mcs_alphas: tuple = (0.10, 0.25)
    mcs_reps: int = 10000
    mcs_method: str = "R"                # T_R ("R") or T_max ("max")
    mcs_block_size: Optional[int] = None  # None -> Politis-White automatic length
    gw_conditional: bool = True          # GW instruments (1, d_{t-1}); rolling windows only
    holm_alpha: float = 0.05
    # --- in-sample inference (OP-DK, OP-NW, OP-H4) ---
    dk_bandwidth: Optional[float] = None  # None -> linearmodels' automatic Bartlett bandwidth
    nw_lags: Optional[int] = None        # None -> floor(4 (T/100)^(2/9)) with T = days
    boot_reps: int = 1999
    wald_bootstrap_reps: int = 499       # 7 Oct 2026 (Gerome): block-bootstrap DK Wald (3.6-3.8) reported; chi2 version kept with a note
    boot_block: Optional[float] = None   # None -> Politis-White (arch optimal_block_length), stationary column
    h4_estimator: str = "ols"            # "ols" on all R3 regressors | "adaptive_lasso" (lambda fixed)
    h4_units: str = "raw"                # cumulative beta in raw (probability) units or "z" units
    h4_pvalue: str = "centred"           # bootstrap p: "centred" P*(t*-t >= t) | "percentile" P*(t* <= 0)
    # --- placebo (OP-PLACEBO) ---
    placebo_shift_days: int = 0
    placebo_mode: str = "circular"       # "circular" (wrap within coin) | "lag" (x(t-shift); first year NaN)
    placebo_blocks: tuple = ("polarity", "emotion")
    # --- runtime ---
    n_jobs: int = 6
    seed: int = 20261007

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)

    @classmethod
    def for_spec(cls, spec: str, **overrides) -> "EngineConfig":
        if spec not in SPEC_ARMS:
            raise ValueError(
                f"arm {spec} is not implemented inside the engine (implemented: {sorted(SPEC_ARMS)}); "
                "arms that change a data rule are run by passing that arm's panel with --spec A00 and --label")
        kw = dict(SPEC_ARMS[spec])
        kw.update({k: v for k, v in overrides.items() if v is not None})
        missing = sorted(k for k, v in kw.items() if isinstance(v, str) and v == REQUIRED)
        if missing:
            raise ValueError(f"arm {spec} needs {missing} on the command line (e.g. --{missing[0].replace('_', '-')}); "
                             "see SPEC_ARMS in model/config.py")
        return cls(spec=spec, **kw)


def rung_blocks_for(cfg) -> dict:
    """Blocks per rung for an engine configuration (reference models always as in RUNG_BLOCKS)."""
    rb = dict(RUNG_BLOCKS)
    if getattr(cfg, "rung_blocks", "main") == "controls_from_R1":
        rb.update(RUNG_BLOCKS_CONTROLS_FROM_R1)
    elif getattr(cfg, "rung_blocks", "main") != "main":
        raise ValueError(cfg.rung_blocks)
    return rb


def hac_default_lags(T: int) -> int:
    """Newey-West (1994) rule of thumb floor(4 (T/100)^(2/9))."""
    return int(4 * (T / 100.0) ** (2.0 / 9.0))
