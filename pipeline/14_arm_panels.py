#!/usr/bin/env python3
"""
14_arm_panels.py -- step 14: the estimation panels and variable specs of the register arms whose
difference from A00 lies in the DATA or the VARIABLES (har_specification_register.csv), built from
any assembled A00 panel (step 13, har_panel.parquet).

Written 28 Oct 2026, after real results were seen. Robustness tooling only: it changes no frozen rule,
no earlier step and no earlier output. Every arm changes one axis of A00 and is otherwise identical:
an arm panel is a copy of the A00 panel in which only the arm's columns are replaced or added, and its
spec (code/model/variable_spec_<ARM>.csv) is A00's spec with only the arm's rows changed.

Construction (as A00; MASTERPLAN_NOV15.md section 7 and planning/pre_analysis_note_D8.md section 7)
--------------------------------------------------------------------------------------------------
* Text block (pol, pol_sd, emo_<k>, emo_<k>_sd and the informational *_raw, text_n_posts, n_platforms_emo):
  the same construction and names as A00. Per platform, series and day the D16 precision adjustment of
  step 10b (clarification of 7 Oct 2026): value = level + R(n) (raw mean - level), level = mean of the
  platform's raw daily means over the available days of the previous 30 days, R(n) = rel_<v> of
  text_daily.parquet (sigma2 fixed from the pre-forecast year, text_calibration_preforecast.json). Then
  D15: the equal-weighted mean of the available platforms. Raw pooled means, pooled SDs and counts are
  the variant's stored POOLED rows of step 10 (or, for platform subsets, the same equal-weighted means of
  the platform rows). Step 10b's constants are imported read-only; its rule is re-implemented here and
  checked to reproduce text_precision_daily.parquet and the base panel bit for bit.
* Series-wide arms (an arm that redefines which posts belong to a coin series, or names the market
  measure) also rebuild the platform post counts att_<p> / mkt_<p> from the variant's post counts, as
  step 11 does (Twitter: count / captured share of the UTC day; missing on the 76 zero-capture days).
  The MARKET series is changed only where the arm's data rule names the market measure (A38, A41, A47).
* Identity (asserted before any arm is built): A00 rebuilt through this script's paths from the stored
  daily tables equals the base panel's text, attention, controls and volatility columns exactly.

Inputs (read-only): data/pipeline/panel/{text_daily, text_precision_daily, attention_daily, controls_daily,
daily_counts, volatility_daily, volatility_medrv_daily}.parquet, text_calibration_preforecast.json,
data/pipeline/ravenpack/emtract/news_row_headline_map.parquet, data/pipeline/ravenpack/polarity/finbert-L64/
polarity/headlines.parquet, the re-aggregation outputs of 14b_arm_reaggregate.py (A40, A42, A47), the
base panel and model/variable_spec_A00.csv.

Outputs: <out-dir>/<ARM>.parquet (+ .stats.json), variable specs (code/model/variable_spec_<ARM>.csv if
absent or identical, else <out-dir>/specs/), the manifest (one row per register arm) and a run list (one
row per engine run). An existing file is never overwritten: identical content is skipped, different
content stops the step.

Usage (from code/):
    python pipeline/14_arm_panels.py --base data/pipeline/panel/har_panel.parquet --arms all \\
        --out-dir data/pipeline/panel/arms/prov_20261028 \\
        --manifest data/pipeline/panel/arms/arm_manifest_20261028.csv
    python pipeline/14_arm_panels.py --list          # the arm registry
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                     # code/pipeline/
CODE = HERE.parent                                         # code/
PIPE = CODE / "data" / "pipeline"
PANEL = PIPE / "panel"
REGISTER = CODE.parent / "har_specification_register.csv"
STEP = "14_arm_panels"

COINS = ["BTC", "ETH", "BNB", "XRP", "DOGE"]
SERIES = COINS + ["MARKET"]
PLATS = ["reddit", "stocktwits", "twitter"]
EMO6 = ["happy", "sad", "anger", "disgust", "surprise", "fear"]
VARS9 = ["neutral", "happy", "sad", "anger", "disgust", "surprise", "fear", "valence", "polarity"]
FIRST, LAST = pd.Timestamp("2019-08-01"), pd.Timestamp("2023-01-31")
DAYS = pd.date_range(FIRST, LAST, freq="D")
FIRST_ORIGIN = pd.Timestamp("2019-08-30")

TEXT_COLS = (["pol_raw", "pol_sd"] + [f"emo_{k}_raw" for k in EMO6] + [f"emo_{k}_sd" for k in EMO6]
             + ["text_n_posts", "n_platforms_emo", "pol"] + [f"emo_{k}" for k in EMO6])
ATT_COIN = {"reddit": "att_reddit", "stocktwits": "att_stocktwits", "twitter": "att_twitter"}
ATT_MKT = {"reddit": "mkt_reddit", "stocktwits": "mkt_stocktwits", "twitter": "mkt_twitter"}
CTRL = {"ctrl_vix": "log_vix", "ctrl_move": "log_move", "ctrl_term": "term_spread", "ctrl_baa10y": "baa10y_lag1",
        "ctrl_dgs10_chg": "d_dgs10", "ctrl_dxy_ret": "ret_dxy", "ctrl_gold_ret": "ret_gold", "ctrl_epu": "log_epu_lag1",
        "ctrl_tether_news": "tether_log1p_headlines", "ctrl_tether_news_chg": "tether_dlog_headlines",
        "ctrl_weekend": "weekend"}          # as step 13 (ctrl_weekend there = forecast-day dummy, Deviation 3)
RP_ENTITY = {"BTC": "A25816", "ETH": "0A2CF4", "BNB": "A2BF5C", "XRP": "547BE3", "DOGE": "2070E1"}   # as step 11
RP_RELEVANCE_MIN = 75                                                                                 # as step 11
DATE_TAG = "28 Oct 2026"


def _import_10b():
    """Step 10b's constants, imported read-only (the module's main() is never called)."""
    spec = importlib.util.spec_from_file_location("step10b_text_precision", HERE / "10b_text_precision.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S10B = _import_10b()
LEVEL_DAYS = S10B.LEVEL_DAYS            # 30 (own choice of step 10b, flagged there)
assert list(S10B.VARS) == EMO6 + ["polarity"], S10B.VARS


# =====================================================================================================
# 1. Arm registry
# =====================================================================================================
# kind: text_variant | text_subset | emo_transform | attention | control | volatility | news_polarity | reagg
# scope (text arms): 'text' = text block only; 'series' = text block and the platform post counts.
ARMS: Dict[str, dict] = {}


def _arm(run_id, arm, kind, **kw):
    ARMS[run_id] = dict(arm=arm, kind=kind, **kw)


for p_drop, lab in (("reddit", "noReddit"), ("stocktwits", "noStockTwits"), ("twitter", "noTwitter")):
    _arm(f"A03_{lab}", "A03", "text_subset", platforms=[p for p in PLATS if p != p_drop], scope="text",
         what=f"pooled text index (D16, D15) over the two platforms other than {p_drop}; attention block as A00",
         reading="register text_source = leave-one-platform-out; data_rule 'primary data rules', so the platform "
                 "counts of R1 stay as in A00 (open point: drop the platform's counts too?)")
for p_keep, lab in (("reddit", "Reddit"), ("stocktwits", "StockTwits"), ("twitter", "Twitter")):
    _arm(f"A04_{lab}", "A04", "text_subset", platforms=[p_keep], scope="text",
         what=f"text index of {p_keep} alone (D16 platform values); attention block as A00",
         reading="register text_source = single platform; attention as A00 (as A03)")
_VAR = {  # run_id: (variant, scope, what, reading)
    "A13": ("a13_chunk", "text", "chunk-mean EmTract and FinBERT scores for texts over 62 wordpieces", "scoring only"),
    "A20": ("a20_unstratified", "text", "Twitter daily means without hour post-stratification", "aggregation only"),
    "A21": ("a21_with_empty", "text", "emotion means incl. empty-after-clean texts (EmTract's fixed empty score); "
            "polarity as A00", "register: 'emotion means'"),
    "A23": ("a23_textgate", "series", "Reddit coin series on text-gated rows (D12 gate)",
            "series definition (REDDIT map P15): counts and text"),
    "A26": ("a26_posts_only", "text", "Reddit submissions only (comments excluded) in the text means",
            "P33/purpose 'the averaged unit': text means only"),
    "A28": ("a28_single_ticker", "series", "coin series of single-ticker posts only (n_coins = 1)",
            "series definition (STOCKTWITS map PJ3): counts and text"),
    "A29": ("a29_bnb_strict", "series", "BNB = own cashtag or tier-1 term only (bnb_strict), all platforms",
            "series definition (PJ4): counts and text"),
    "A38": ("a38_no_equity_only", "series", "StockTwits without equity_only rows, in BTC and in the market measure",
            "PJ2; HAR_VARIANTS T6 (28 Sep): leaving them out is A38 in the attention denominator: counts and text"),
    "A39": ("a39_no_bsc_context", "series", "BNB without bsc_context rows, all platforms",
            "series definition (PJ4): counts and text"),
    "A41": ("a41_market_support", "series", "Reddit market series incl. the r/binance support posts",
            "market series: in A00 the MARKET text enters no rung, so the arm acts through mkt_reddit"),
    "A43": ("a43_with_bots", "series", "bot/spam-flagged posts kept in the emotion and attention measures",
            "register data_rule names both measures"),
    "A44": ("a44_dedup_text_day", "text", "exact-duplicate texts once per platform, series and UTC day in the text "
            "means; copies kept for the counts", "register data_rule: 'copies kept only for counts'"),
    "A45": ("a45_no_edited", "text", "Reddit text means without posts edited after posting",
            "P10/purpose (archive-time text versions): text means only"),
    "A46": ("a46_authored_only", "text", "Twitter text means of authored posts only (retweet-only posts excluded)",
            "D13 is the rule for the Twitter emotion measure: text means only"),
    "A48": ("a48_quoted", "text", "quote posts scored on the quoted text (emotions; FinBERT has the own text only)",
            "scoring only"),
    "A49": ("a49_both", "text", "quote posts scored on own + quoted text (emotions; FinBERT has the own text only)",
            "scoring only"),
}
for rid, (variant, scope, what, reading) in _VAR.items():
    _arm(rid, rid, "text_variant", variant=variant, scope=scope, what=what, reading=reading)
_arm("A59", "A59", "emo_transform", what="emotion levels and their centred squares replaced by the additive "
     "log-ratios ln(emo_k / emo_neutral) of the D16-adjusted pooled shares (same column names, csq = 0); within-day "
     "SDs unchanged; H4 is then computed on the ALR coefficients",
     reading="MASTERPLAN 6 Oct (7a); the neutral share gets the same D16 adjustment (rel_neutral) as the six classes "
             "(open point: the 7 Oct D16 clarification names the six classes and polarity only)")
_arm("A60", "A60", "emo_transform", what="R3 emotion block = one variable emo_certainty = ln((anger + happy + disgust) "
     "/ (fear + surprise)) of the D16-adjusted pooled shares (the A00 emo_* columns), in HAR form, no square, no SD",
     reading="MASTERPLAN 6 Oct (7b): 'from the pooled daily shares, in HAR form'; H4 not testable (engine needs a "
             "guard: model/inference.py asserts emo_fear_h1/h7/h30 in R3)")
_arm("A68", "A68", "emo_transform", what="R2 polarity slot = EmTract valence (happy - sad - anger - disgust - fear): pol = "
     "D16-adjusted pooled valence (its own R(n), rel_valence), pol_sd = pooled within-day SD of valence",
     reading="valence takes the polarity variable's construction (open point: valence from the six adjusted shares)")
_arm("A37", "A37", "control", what="ctrl_empty_share = empty-after-clean share of the coin-day's posts (pooled counts, "
     "step 10 POOLED empty_share) added to the controls (day-t value, every rung as all controls)",
     reading="register: 'added as a control'; controls enter every rung (Deviation 3)")
_arm("A56", "A56", "attention", what="the six platform post counts (coin and MARKET) replaced by the log within-day "
     "share of the platform's crypto posts (logshare_<p> of step 11), HAR form and day-t change as A00",
     reading="register 'instead of Audrino-style log counts' read as all six platform counts; zero coin-days are "
             "NaN (step 11 open point O1)")
_arm("A57", "A57", "attention", what="mkt_rs_total = Reddit + StockTwits CRYPTO_SET posts added to the market "
     "attention block (log(1 + x) of each day, HAR form, day-t change, as the platform counts)",
     reading="register: 'R1 market block + log total Reddit+StockTwits crypto volume'")
_arm("A58", "A58", "attention", what="Google Trends enters as the day-t daily log change only (no level, no HAR "
     "averages): att_google_dlog, mkt_google_<term>_dlog = the engine's own day-t log change",
     reading="register: 'Google enters as daily log change only'")
_arm("A66", "A66", "attention", what="RavenPack news volume per entity row instead of distinct headlines: att_news "
     "-> news rows; the Tether control -> log(1 + rows) and its daily change",
     reading="register 'news volume = log(1 + entity rows)' read for both RavenPack volume series (step 12 built the "
             "Tether rows column for this rule)")
_arm("A24", "A24", "attention", what="Reddit counts as total activity: live posts after the bot pass + dead rows "
     "(daily_counts.n_dead), coin and MARKET series",
     reading="register: 'Reddit attention as total activity incl. removed/deleted rows'")
_arm("A63", "A63", "control", what="credit control = ICE BofA HY OAS lagged one day (rob_hy_oas_lag1) instead of "
     "BAA10Y", reading="register; lag per the D8 clarification of 7 Oct 2026")
_arm("A64", "A64", "control", what="dollar control = DTWEXBGS log return, point in time (a value from its H.10 "
     "release day on; rob_dtwexbgs_logret_pit) instead of DXY",
     reading="D8 clarification of 7 Oct 2026: no value is used before it is published (open point: the register's "
             "day-t return, built as A64_dayt)")
_arm("A64_dayt", "A64", "control", what="dollar control = DTWEXBGS day-t log return (rob_dtwexbgs_logret; up to 7 days "
     "of look-ahead) instead of DXY", reading="register literal ('pseudo real time'); candidate only")
_arm("A65", "A65", "control", what="controls + Alternative.me Fear & Greed level (rob_fng, day-t value)",
     reading="register: 'controls + Fear & Greed (day-t value)'; no transform stated (step 12 open point 5)")
_arm("A67", "A67", "news_polarity", what="R2 + pol_news = daily mean FinBERT polarity of the coin's distinct RavenPack "
     "headlines (relevance >= 75, as att_news) with its centred square, and pol_news_sd = within-day SD over "
     "headlines (NaN below 2); HAR form; days without a headline NaN",
     reading="6 Oct (3): every sentiment variable gets level, centred square and within-day SD; D16 is not defined for "
             "news (no sigma2); open points listed")
_arm("A50", "A50", "volatility", what="C and J from MedRV + MedRQ ratio test (v = 0.96, alpha 0.999, outage split; "
     "library R04): c_med, j_med replace c, j", reading="as the 7 Oct benchmark panel (variable_spec_A50medcj.csv); "
     "MedRV/MedRQ constants still to be checked against ADS 2012")
_arm("A54", "A54", "volatility", what="HAR-MedRV: one MedRV term (1/7/30, log) replaces C and J; MedRV over the 5-minute "
     "intervals untouched by an outage (medrv_split = 09b medrv_obs)",
     reading="outage rule 5-6 Oct (6) applied to MedRV as to BV (open point: the 7 Oct provisional run used the "
             "carry-forward 'medrv'); constants still to be checked against ADS 2012")
_arm("A40", "A40", "reagg", reagg="A40", scope="series", what="coin series incl. posts whose only coin word is inside a "
     "link (link_<coin>), StockTwits and Twitter; Reddit unchanged (no per-coin link flag)",
     reading="series definition (PD4): counts and text")
_arm("A42", "A42", "reagg", reagg="A42", scope="series", what="StockTwits coin-less replies get the coin of the post "
     "they answer (reply_parent_<coin>, one step)", reading="series definition (PJ7 'coin panels'): counts and text")
_arm("A47", "A47", "reagg", reagg="A47", scope="series", what="Twitter and StockTwits market counts incl. the posts "
     "flagged ambiguous_extra (two or more different ambiguous terms)",
     reading="market measure: in A00 only mkt_twitter, mkt_stocktwits change (MARKET text enters no rung)")

# Register arms in this track that are not built, with the missing rule (build_status, reason)
NOT_BUILT = {
    "A02": ("open question", "market panel: the engine has no single-unit (market) mode (5-coin grid and coin FE are "
            "hard-coded); the R1 block for the market unit is not defined (no market news-volume series; which coin-level "
            "attention terms have market analogues); the value-weighted portfolio RV (3 Oct (1), previous-day Coin Metrics "
            "caps, 6 Oct (10)) is definable but not built; Coin Metrics snapshot-time check pending (6 Oct (12))"),
    "A11": ("open question", "the imputed 'neutral-emotion' value for Twitter gap days is not defined (EmTract neutral "
            "share = 1? the empty-text score? the window mean?), nor the gap dummy's block and penalty (6 Oct (2): no "
            "twitter_absent dummy in A00)"),
    "A12": ("open question", "retweet events carry a UTC day but no hour (retweet_events_daily: first_at/last_at only), so "
            "the hour post-stratification of the 87 partial Twitter days is undefined for them; unstated whether events "
            "are bot-filtered by the retweeter; post-level re-aggregation needed once settled"),
    "A14": ("open question", "the sampling cell is not fixed (D5: 3,000 per coin/platform/day; register and test 2.13: "
            "3,000 per coin-day) and no seed is set; post-level re-aggregation needed once settled"),
    "A15": ("open question", "the floor is not fixed: one n* per platform-coin or per variable (9 per cell), which "
            "variable defines thin_day, cells with sigma2_b <= 0 (n* infinite), and whether a thin platform-day leaves "
            "the pooled index or the coin-day leaves estimation and evaluation"),
    "A22": ("open question", "the platform precision weight is not fixed (HAR_VARIANTS section 3: w = n or "
            "w = 1/(sigma2_w/n + sigma2_b); D15: 'inverse-variance'); text report 7 Oct point 5"),
    "A33": ("open question", "defined against the old primary (within-day share of posts, D14 before 6 Oct): unclear "
            "whether A33 is log(1 + distinct authors) (one axis from A00) or the log share of distinct authors (one axis "
            "from A56); Twitter author counts have no coverage-scaling rule"),
    "A34": ("open question", "the label-share formula is not fixed (share of bullish, (bull - bear)/(bull + bear), or "
            "Antweiler-Frank log ratio); D16 and the Audrino transforms for a StockTwits-only label measure are not "
            "defined; labels are local (stocktwits/coarse.sentiment), post-level re-aggregation needed"),
    "A51": ("open question", "the register gives 'outage days carry-forward (or target=rv)'; the target of the arm (log RV "
            "of A00, or the corrected RV_c) is not fixed; theta 0.609 (Huang-Tauchen form) vs KR's printed 2.61 and the "
            "first-return duration are own readings in cj_methods R06; 'provisional until the online supplement' "
            "(rounding correction, Appendix C) is available"),
    "A17": ("open question", "DLT Twitter-only ladder: the R1 block is not defined (DLT counts only? with Wikipedia, "
            "Google, news, market counts?; DLT has no capture ledger, so no coverage scaling); D16 sigma2 for DLT (own "
            "pre-forecast-year calibration, sparse date-only rows in 2019-20) not computed; post-level re-aggregation of "
            "dlt_units + dlt_emotion + DLT FinBERT needed"),
    "A18": ("open question", "event design for the IA-gap episodes 'to be specified later (incl. whether gap 3 is "
            "usable)' (register notes, 6 Oct); in-sample event dummies and interactions are an engine design"),
    "A69": ("open question", "as A17, plus the sample restriction 2021-01-01..2022-06-30 with a 365-day rolling window "
            "leaves forecasts only from Jan 2022 (engine first/last target settings)"),
    "A70": ("open question", "conditional arm ('run only if full-sample calibration differs materially'); the materiality "
            "judgement is Gérome's (text report 7 Oct point 7); the full-sample hour profile needs the hourly post sums, "
            "which text_daily does not store (post-level re-aggregation)"),
}
ENGINE_TRACK = ["A01", "A05", "A06", "A07", "A08", "A09", "A10", "A16", "A52", "A53", "A55", "A61", "A62", "A71"]
ENGINE_DIRECT = {"A01": "--spec A01", "A06": "--spec A06", "A55": "--spec A55", "A71": "--spec A71",
                 "A61": "--spec A00 --qlike-backtransform duan --label A61"}


# =====================================================================================================
# 2. Small helpers
# =====================================================================================================
def md5(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def same(a: pd.Series, b: pd.Series) -> bool:
    """Element-wise equality with NaN == NaN (bit-exact for floats)."""
    a = pd.Series(np.asarray(a, dtype=float)); b = pd.Series(np.asarray(b, dtype=float))
    return bool(((a == b) | (a.isna() & b.isna())).all())


def as_day(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s.astype(str) if s.dtype == object else s)
    if getattr(s.dt, "tz", None) is not None:
        assert str(s.dt.tz) == "UTC"
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    assert (s == s.dt.normalize()).all(), "day column carries a time of day"
    return s.dt.normalize().astype("datetime64[ns]")


class Checks:
    def __init__(self):
        self.items: List[dict] = []

    def __call__(self, cond, msg, value=None):
        rec = {"check": msg, "passed": bool(cond)}
        if value is not None:
            rec["value"] = value
        self.items.append(rec)
        if not cond:
            raise AssertionError(msg + ("" if value is None else f" ({value})"))


# =====================================================================================================
# 3. Inputs
# =====================================================================================================
class Inputs:
    def __init__(self, reagg_dir: Optional[Path]):
        self.paths = {k: PANEL / f for k, f in [
            ("text_daily", "text_daily.parquet"), ("text_precision", "text_precision_daily.parquet"),
            ("attention", "attention_daily.parquet"), ("controls", "controls_daily.parquet"),
            ("daily_counts", "daily_counts.parquet"), ("volatility", "volatility_daily.parquet"),
            ("medrv", "volatility_medrv_daily.parquet"), ("calibration", "text_calibration_preforecast.json")]}
        self.paths["rp_map"] = PIPE / "ravenpack" / "emtract" / "news_row_headline_map.parquet"
        self.paths["rp_pol"] = PIPE / "ravenpack" / "polarity" / "finbert-L64" / "polarity" / "headlines.parquet"
        self.reagg_dir = reagg_dir
        self._cache = {}

    def get(self, key):
        if key in self._cache:
            return self._cache[key]
        p = self.paths[key]
        if key == "calibration":
            obj = json.load(open(p))
        elif key == "rp_map":
            obj = pd.read_parquet(p, columns=["rp_entity_id", "relevance", "ts_utc", "rpa_date_utc", "headline_md5"])
        elif key == "rp_pol":
            obj = pd.read_parquet(p, columns=["row_id", "version", "polarity"])
        else:
            obj = pd.read_parquet(p)
            for c in ("day", "day_utc"):
                if c in obj.columns:
                    obj["day"] = as_day(obj[c])
                    break
        self._cache[key] = obj
        return obj

    def used(self, keys) -> List[dict]:
        return [{"path": str(self.paths[k]), "bytes": os.path.getsize(self.paths[k]), "md5": md5(self.paths[k])}
                for k in sorted(set(keys))]


# =====================================================================================================
# 4. Text path: D16 (step 10b rule) and D15 pooling
# =====================================================================================================
def d16_platform_values(plat_rows: pd.DataFrame, v: str) -> Dict[tuple, pd.Series]:
    """Step 10b: per (series, platform), value = level + R(n) (raw - level); level = mean of the raw daily means
    of the previous LEVEL_DAYS days (available days only); where there is no level the raw mean is kept."""
    vals = {}
    for (s, p), g in plat_rows.groupby(["series", "platform"]):
        g = g.set_index("day").reindex(DAYS)
        raw = g[f"mean_{v}"].astype(float)
        R = g[f"rel_{v}"].astype(float)
        level = raw.shift(1).rolling(LEVEL_DAYS, min_periods=1).mean()
        val = np.where(level.notna(), level + R * (raw - level), raw)
        vals[(s, p)] = pd.Series(val, index=DAYS).where(raw.notna())
    return vals


def d16_pooled(plat_rows: pd.DataFrame, v: str) -> pd.DataFrame:
    """D15 equal-weighted mean of the available platforms' D16 values; index day, columns series."""
    V = pd.DataFrame(d16_platform_values(plat_rows, v))
    out = {}
    for s in sorted(set(c[0] for c in V.columns)):
        out[s] = V[[c for c in V.columns if c[0] == s]].mean(axis=1, skipna=True)
    return pd.DataFrame(out)


def pooled_raw(plat_rows: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted mean of the available platform means and SDs, counts summed: step 10's POOLED rows
    (same pandas groupby mean, platform order reddit, stocktwits, twitter)."""
    pr = plat_rows.copy()
    pr["_po"] = pr["platform"].map({p: i for i, p in enumerate(PLATS)})
    pr = pr.sort_values(["series", "day", "_po"])
    agg = {f"mean_{x}": "mean" for x in VARS9} | {f"sd_{x}": "mean" for x in VARS9} | {
        "n_posts": "sum", "n_empty": "sum", "n_emo": "sum", "n_pol": "sum"}
    g = pr.groupby(["series", "day"], sort=True)
    out = g.agg(agg)
    out["n_platforms_emo"] = g["mean_happy"].count()
    out["n_platforms_pol"] = g["mean_polarity"].count()
    out["empty_share"] = np.where(out.n_posts > 0, out.n_empty / out.n_posts.where(out.n_posts > 0), np.nan)
    return out.reset_index()


def platform_rows(I: Inputs, variant: str, platforms=PLATS, override: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    td = I.get("text_daily")
    t = td[(td.variant == variant) & td.platform.isin(platforms)].copy()
    if override is not None:   # re-aggregated platform rows replace the stored ones (A40, A42)
        key = ["platform", "series", "day"]
        ov = override[override.platform.isin(platforms)]
        t = t.merge(ov[key].assign(_ov=True), on=key, how="left")
        t = t[t._ov.isna()].drop(columns="_ov")
        t = pd.concat([t, ov[[c for c in t.columns if c in ov.columns]]], ignore_index=True)
    return t[t.platform != "POOLED"]


def text_block(I: Inputs, variant: str = "main", platforms=PLATS, stored_pool: bool = True,
               override: Optional[pd.DataFrame] = None, extra_vars=()) -> pd.DataFrame:
    """A00's text block for one variant and platform set: (coin x day) frame with TEXT_COLS (+ extras)."""
    plat = platform_rows(I, variant, platforms, override)
    if stored_pool and list(platforms) == PLATS and override is None:
        td = I.get("text_daily")
        pool = td[(td.variant == variant) & (td.platform == "POOLED")].copy()
    else:
        pool = pooled_raw(plat)
    pool = pool[pool.series.isin(COINS)].rename(columns={"series": "coin"})
    out = pd.DataFrame({"coin": pool["coin"].values, "day": pool["day"].values})
    out["pol_raw"] = pool["mean_polarity"].values
    out["pol_sd"] = pool["sd_polarity"].values
    for k in EMO6:
        out[f"emo_{k}_raw"] = pool[f"mean_{k}"].values
        out[f"emo_{k}_sd"] = pool[f"sd_{k}"].values
    out["text_n_posts"] = pool["n_posts"].values.astype("int64")
    out["n_platforms_emo"] = pool["n_platforms_emo"].values.astype("int64")
    for x in ("mean_neutral", "mean_valence", "sd_valence", "empty_share"):
        out[f"_pool_{x}"] = pool[x].values
    for v, col in [("polarity", "pol")] + [(k, f"emo_{k}") for k in EMO6] + [(x, f"_d16_{x}") for x in extra_vars]:
        st = d16_pooled(plat, v).stack()
        st.index.names = ["day", "coin"]
        out = out.merge(st.rename(col).reset_index(), on=["coin", "day"], how="left")
    return out


def counts_from_rows(I: Inputs, rows: pd.DataFrame) -> pd.DataFrame:
    """Platform post counts in step 11's form: Reddit, StockTwits raw; Twitter count / captured share of the UTC
    day (minutes_utc / 1440), NaN on the zero-capture days. Index (series, day), columns reddit/stocktwits/twitter."""
    ad = I.get("attention")
    cov = ad[ad.series == "BTC"][["day", "twitter_coverage_share", "twitter_absent"]]
    w = rows.pivot_table(index=["series", "day"], columns="platform", values="n_posts", aggfunc="sum").reset_index()
    w = w.merge(cov, on="day", how="left")
    w["twitter"] = w["twitter"] / w["twitter_coverage_share"]
    w.loc[w["twitter_absent"].astype(bool), "twitter"] = np.nan
    return w.drop(columns=["twitter_coverage_share", "twitter_absent"])


def attention_block(I: Inputs, rows: pd.DataFrame) -> pd.DataFrame:
    """(coin x day): att_<p> (coin series) and mkt_<p> (MARKET, same value on every coin)."""
    w = counts_from_rows(I, rows)
    coin = w[w.series.isin(COINS)].rename(columns={"series": "coin"}).rename(columns=ATT_COIN)
    mk = w[w.series == "MARKET"].drop(columns="series").rename(columns=ATT_MKT)
    return coin[["coin", "day"] + list(ATT_COIN.values())].merge(mk, on="day", how="left")


# =====================================================================================================
# 5. Identity of A00 through these paths
# =====================================================================================================
def identity_checks(I: Inputs, base: pd.DataFrame, ck: Checks) -> None:
    b = base.copy()
    # text: D16 values reproduce step 10b's table, and the block reproduces the base panel
    td = I.get("text_daily")
    plat_main = platform_rows(I, "main")
    tp = I.get("text_precision")
    for v in S10B.VARS:
        st = d16_pooled(plat_main, v).stack()
        st.index.names = ["day", "series"]
        mm = tp[["series", "day", f"mean_{v}_pw"]].merge(st.rename("mine").reset_index(), on=["series", "day"], how="outer")
        ck(same(mm[f"mean_{v}_pw"], mm["mine"]), f"identity: D16 {v} reproduces text_precision_daily (all series) exactly")
    cal = I.get("calibration")["d16_reliability"]
    pr = td[td.platform != "POOLED"]
    key = pr.platform.astype(str) + "|" + pr.series.astype(str)
    for x in VARS9:
        ncol = "n_pol" if x == "polarity" else "n_emo"
        s2w = key.map({f"{p}|{s}": cal[p][s][x]["sigma2_w"] for p in cal for s in cal[p]}).astype(float)
        s2b = key.map({f"{p}|{s}": cal[p][s][x]["sigma2_b"] for p in cal for s in cal[p]}).astype(float)
        n = pr[ncol].astype(float).where(pr[ncol] > 0)
        n = n.where(~((pr.platform == "twitter") & pr.twitter_absent.fillna(False).astype(bool)))
        ck(same(s2b / (s2b + s2w / n), pr[f"rel_{x}"]),
           f"identity: rel_{x} of every variant = R(n) with the fixed pre-forecast sigma2 (calibration json)")
    tb = text_block(I, "main")
    mm = b[["coin", "day"] + TEXT_COLS].merge(tb, on=["coin", "day"], how="left", suffixes=("", "_mine"))
    for c in TEXT_COLS:
        ck(same(mm[c], mm[c + "_mine"]), f"identity: base {c} = text path (main, 3 platforms)")
    sub = pooled_raw(plat_main)
    pool = td[(td.variant == "main") & (td.platform == "POOLED")].set_index(["series", "day"]).sort_index()
    sub = sub.set_index(["series", "day"]).sort_index()
    ck(all(same(pool[c], sub[c]) for c in [f"mean_{x}" for x in VARS9] + [f"sd_{x}" for x in VARS9] + ["n_posts"]),
       "identity: pooled means/SDs recomputed from platform rows = stored POOLED rows (subset path)")
    # attention: both count paths reproduce the base
    ab = attention_block(I, plat_main)
    mm = b[["coin", "day"] + list(ATT_COIN.values()) + list(ATT_MKT.values())].merge(ab, on=["coin", "day"], how="left",
                                                                                   suffixes=("", "_mine"))
    for c in list(ATT_COIN.values()) + list(ATT_MKT.values()):
        ck(same(mm[c], mm[c + "_mine"]), f"identity: base {c} = post counts of text_daily main (series-wide path)")
    ad = I.get("attention")
    ac = ad[ad.series.isin(COINS)].rename(columns={"series": "coin"})
    mm = b[["coin", "day", "att_wiki", "att_news"]].merge(ac[["coin", "day", "wiki_views", "news_headlines_distinct"]],
                                                          on=["coin", "day"], how="left")
    ck(same(mm.att_wiki, mm.wiki_views) and same(mm.att_news, mm.news_headlines_distinct),
       "identity: base att_wiki, att_news = attention_daily")
    # controls
    k = I.get("controls")
    mm = b[["coin", "day"] + list(CTRL)].merge(k[["day"] + list(CTRL.values())], on="day", how="left", suffixes=("", "_k"))
    for c, src in CTRL.items():
        if c == "ctrl_weekend":
            ok = same(mm[c], ((mm.day + pd.Timedelta(days=1)).dt.dayofweek >= 5).astype(float))
        else:
            ok = same(mm[c], mm[src])
        ck(ok, f"identity: base {c} = controls_daily.{src}" + (" (forecast-day weekend, Deviation 3)" if c == "ctrl_weekend" else ""))
    # volatility
    v = I.get("volatility")
    mm = b[["coin", "day", "rv", "c", "j"]].merge(v[["coin", "day", "rv", "c", "j"]], on=["coin", "day"], suffixes=("", "_v"))
    ck(all(same(mm[c], mm[c + "_v"]) for c in ("rv", "c", "j")), "identity: base rv, c, j = volatility_daily")


# =====================================================================================================
# 6. Spec editing
# =====================================================================================================
SPEC_COLS = ["name", "table", "block", "scope", "har", "transform", "dlog", "csq", "impute_flag", "notes"]


def read_spec(path) -> pd.DataFrame:
    s = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(s.columns) == SPEC_COLS, list(s.columns)
    return s


def spec_row(name, block, scope, har, transform, dlog=0, csq=0, impute_flag="", notes="", table="har_panel") -> dict:
    return dict(name=name, table=table, block=block, scope=scope, har=str(har), transform=transform, dlog=str(dlog),
                csq=str(csq), impute_flag=impute_flag, notes=notes)


class SpecEdit:
    def __init__(self, base_spec: pd.DataFrame, run_id: str):
        self.s = base_spec.copy()
        self.run_id = run_id
        self.log: List[str] = []

    def _note(self, txt):
        return f"{self.run_id} ({DATE_TAG}): {txt}"

    def has(self, name):
        return bool((self.s.name == name).any())

    def replace(self, old: str, rows: List[dict]):
        if not self.has(old):
            self.log.append(f"skipped: row {old} not in the base spec")
            return
        i = int(np.flatnonzero(self.s.name == old)[0])
        new = pd.DataFrame(rows, columns=SPEC_COLS)
        self.s = pd.concat([self.s.iloc[:i], new, self.s.iloc[i + 1:]], ignore_index=True)
        self.log.append(f"replaced {old} by {', '.join(r['name'] for r in rows)}")

    def set(self, name: str, note: str, **fields):
        if not self.has(name):
            self.log.append(f"skipped: row {name} not in the base spec")
            return
        i = self.s.index[self.s.name == name][0]
        for k, v in fields.items():
            self.s.at[i, k] = str(v)
        old = self.s.at[i, "notes"]
        self.s.at[i, "notes"] = (old + " | " if old else "") + self._note(note)
        self.log.append(f"changed {name}: {', '.join(f'{k}={v}' for k, v in fields.items()) or 'notes'}")

    def drop(self, names: List[str]):
        present = [n for n in names if self.has(n)]
        self.s = self.s[~self.s.name.isin(present)].reset_index(drop=True)
        self.log.append(f"dropped {', '.join(present)}")

    def insert_after(self, after: str, rows: List[dict]):
        i = int(np.flatnonzero(self.s.name == after)[0]) if self.has(after) else len(self.s) - 1
        new = pd.DataFrame(rows, columns=SPEC_COLS)
        self.s = pd.concat([self.s.iloc[:i + 1], new, self.s.iloc[i + 1:]], ignore_index=True)
        self.log.append(f"added {', '.join(r['name'] for r in rows)} after {after}")

    def row(self, name) -> dict:
        return self.s[self.s.name == name].iloc[0].to_dict()

    def note(self, txt):
        return self._note(txt)


# =====================================================================================================
# 7. Arm builders: each returns (changes: {column: Series aligned to base rows}, spec, info)
# =====================================================================================================
def _aligned(base: pd.DataFrame, frame: pd.DataFrame, cols: List[str]) -> Dict[str, pd.Series]:
    m = base[["coin", "day"]].merge(frame[["coin", "day"] + cols], on=["coin", "day"], how="left")
    assert len(m) == len(base)
    return {c: pd.Series(m[c].values, index=base.index) for c in cols}


def _market(base: pd.DataFrame, daily: pd.Series) -> pd.Series:
    """A market (day-level) series repeated on every coin row."""
    d = daily.copy()
    d.index = as_day(pd.Series(d.index))
    return pd.Series(base["day"].map(d).values, index=base.index, dtype=float)


SPEC_TEXT_ROWS = ["pol", "pol_sd"] + [f"emo_{k}" for k in EMO6] + [f"emo_{k}_sd" for k in EMO6]


def build_text_variant(I, base, spec0, rid, a, ck):
    v = a["variant"]
    tb = text_block(I, v)
    ch = _aligned(base, tb, TEXT_COLS)
    sp = SpecEdit(spec0, rid)
    for n in SPEC_TEXT_ROWS:                      # annotate only the spec rows whose column changes
        if not same(ch[n], base[n]):
            sp.set(n, f"text variant {v} ({a['what']})")
    used = ["text_daily"]
    if a["scope"] == "series":
        ab = attention_block(I, platform_rows(I, v))
        ch.update(_aligned(base, ab, list(ATT_COIN.values()) + list(ATT_MKT.values())))
        for n in list(ATT_COIN.values()) + list(ATT_MKT.values()):
            if not same(ch[n], base[n]):
                sp.set(n, f"post counts of text variant {v} ({a['what']})")
        used.append("attention")
    return ch, sp, used, {"variant": v, "scope": a["scope"]}


def build_text_subset(I, base, spec0, rid, a, ck):
    tb = text_block(I, "main", platforms=a["platforms"], stored_pool=False)
    ch = _aligned(base, tb, TEXT_COLS)
    sp = SpecEdit(spec0, rid)
    for n in ["pol", "pol_sd"] + [f"emo_{k}" for k in EMO6] + [f"emo_{k}_sd" for k in EMO6]:
        sp.set(n, f"text index over {'+'.join(a['platforms'])} only")
    nan_days = int(pd.isna(ch["pol"]).sum())
    return ch, sp, ["text_daily"], {"platforms": a["platforms"], "coin_days_without_text": nan_days}


def build_emo_transform(I, base, spec0, rid, a, ck):
    sp = SpecEdit(spec0, rid)
    td = I.get("text_daily")
    if rid == "A59":
        tb = text_block(I, "main", extra_vars=("neutral",))
        m = base[["coin", "day"] + [f"emo_{k}" for k in EMO6]].merge(tb, on=["coin", "day"], suffixes=("_base", ""))
        ck(all(same(m[f"emo_{k}_base"], m[f"emo_{k}"]) for k in EMO6), "A59: D16 shares of the text path = base emo_*")
        ck(bool((m["_d16_neutral"] > 0).all()), "A59: D16-adjusted neutral share positive on every coin-day")
        out = m[["coin", "day"]].copy()
        for k in EMO6:
            ck(bool((m[f"emo_{k}"] > 0).all()), f"A59: D16-adjusted {k} share positive on every coin-day")
            out[f"emo_{k}"] = np.log(m[f"emo_{k}"] / m["_d16_neutral"])
            out[f"emo_{k}_raw"] = np.log(m[f"emo_{k}_raw"] / m["_pool_mean_neutral"])
        pool = td[(td.variant == "main") & (td.platform == "POOLED") & td.series.isin(COINS)].rename(columns={"series": "coin"})
        chk = out.merge(pool[["coin", "day"] + [f"lr_{k}" for k in EMO6]], on=["coin", "day"])
        ck(all(np.allclose(chk[f"emo_{k}_raw"], chk[f"lr_{k}"], rtol=0, atol=1e-12) for k in EMO6),
           "A59: raw ALR = step 10's stored lr_<k> (POOLED main) within 1e-12")
        cols = [f"emo_{k}" for k in EMO6] + [f"emo_{k}_raw" for k in EMO6]
        for k in EMO6:
            sp.set(f"emo_{k}", f"additive log-ratio ln(emo_{k} / emo_neutral) of the D16-adjusted pooled shares "
                               f"(neutral adjusted by the same rule); replaces level and centred square", csq=0)
        corr = {k: round(float(np.corrcoef(chk[f"emo_{k}"], chk[f"lr_{k}"])[0, 1]), 4) for k in EMO6}
        return _aligned(base, out, cols), sp, ["text_daily"], {"corr_d16_alr_vs_raw_lr": corr}
    if rid == "A60":
        e = {k: base[f"emo_{k}"].astype(float) for k in EMO6}
        cert = np.log((e["anger"] + e["happy"] + e["disgust"]) / (e["fear"] + e["surprise"]))
        ck(bool(np.isfinite(cert).all()), "A60: certainty index finite on every coin-day")
        pool = td[(td.variant == "main") & (td.platform == "POOLED") & td.series.isin(COINS)].rename(columns={"series": "coin"})
        m = base[["coin", "day"]].merge(pool[["coin", "day", "certainty"]], on=["coin", "day"], how="left")
        corr = round(float(np.corrcoef(cert, m["certainty"])[0, 1]), 4)
        emo_rows = [f"emo_{k}" for k in EMO6] + [f"emo_{k}_sd" for k in EMO6]
        first = sp.s.index[sp.s.block == "emotion"][0]
        after = sp.s.at[first - 1, "name"]
        sp.drop(emo_rows)
        sp.insert_after(after, [spec_row("emo_certainty", "emotion", "coin", 1, "level", notes=sp.note(
            "ATF certainty index ln((anger + happy + disgust) / (fear + surprise)) of the D16-adjusted pooled shares; "
            "replaces the six emotions, their squares and SDs (MASTERPLAN 6 Oct (7b))"))])
        ch = {"emo_certainty": pd.Series(cert.values, index=base.index), "emo_certainty_raw": pd.Series(m["certainty"].values, index=base.index)}
        return ch, sp, ["text_daily"], {"corr_d16_certainty_vs_stored_raw": corr,
                                        "engine_note": "model/inference.py h4_test asserts emo_fear_h1/h7/h30 in R3"}
    if rid == "A68":
        tb = text_block(I, "main", extra_vars=("valence",))
        out = tb[["coin", "day"]].copy()
        out["pol"] = tb["_d16_valence"]
        out["pol_raw"] = tb["_pool_mean_valence"]
        out["pol_sd"] = tb["_pool_sd_valence"]
        sp.set("pol", "EmTract valence (happy - sad - anger - disgust - fear; Vamossy & Skog eq. 1) instead of FinBERT "
                      "polarity: D16-adjusted pooled valence (own R(n), rel_valence)")
        sp.set("pol_sd", "within-day SD of post valence (pooled platform SDs; NaN below 2 posts)")
        ch = _aligned(base, out, ["pol", "pol_raw", "pol_sd"])
        corr = round(float(pd.Series(ch["pol"]).corr(base["pol"])), 4)
        return ch, sp, ["text_daily"], {"corr_valence_vs_finbert_pol": corr}
    raise KeyError(rid)


def build_attention(I, base, spec0, rid, a, ck):
    sp = SpecEdit(spec0, rid)
    ad = I.get("attention")
    if rid == "A56":
        coin = ad[ad.series.isin(COINS)].rename(columns={"series": "coin"})
        mk = ad[ad.series == "MARKET"].set_index("day")
        ch = {}
        for p in PLATS:
            ch.update(_aligned(base, coin.rename(columns={f"logshare_{p}": f"att_{p}_logshare"}), [f"att_{p}_logshare"]))
            ch[f"mkt_{p}_logshare"] = _market(base, mk[f"logshare_{p}"])
        for p in PLATS:
            for lvl, old in (("att", ATT_COIN[p]), ("mkt", ATT_MKT[p])):
                r = sp.row(old)
                sp.replace(old, [spec_row(f"{lvl}_{p}_logshare", "attention", r["scope"], 1, "level", dlog=1,
                                          impute_flag=r["impute_flag"], notes=sp.note(
                                              f"log within-day share of {p}'s crypto posts (step 11 logshare_{p}), HAR form "
                                              f"of the daily log share and its day-t change; replaces {old}; zero days NaN (O1)"))])
        nz = {c: int(pd.isna(v).sum()) for c, v in ch.items()}
        return ch, sp, ["attention"], {"nan_coin_days": nz}
    if rid == "A57":
        dc = I.get("daily_counts")
        cs = dc[(dc.series == "CRYPTO_SET") & dc.platform.isin(["reddit", "stocktwits"])]
        tot = cs.pivot_table(index="day", columns="platform", values="n_posts", aggfunc="sum")
        raw = (tot["reddit"] + tot["stocktwits"]).astype(float)
        st = ad[ad.series == "MARKET"].set_index("day")["att_rs_total_crypto"]
        ck(same(np.log1p(raw.reindex(st.index)), st), "A57: log(1 + Reddit + StockTwits CRYPTO_SET posts) = step 11 "
           "att_rs_total_crypto on all 1,280 days")
        ch = {"mkt_rs_total": _market(base, raw)}
        r = sp.row("mkt_twitter") if sp.has("mkt_twitter") else None
        sp.insert_after("mkt_twitter", [spec_row("mkt_rs_total", "attention", "market", 1, "log1p_daily", dlog=1,
                                                 notes=sp.note("total Reddit + StockTwits crypto posts (CRYPTO_SET, after "
                                                               "the bot pass); log of each day, HAR form, day-t change"))])
        return ch, sp, ["daily_counts", "attention"], {}
    if rid == "A58":
        ch, edits = {}, []
        for col, scope in (("att_google", "coin"), ("mkt_google_cryptocurrency", "market"), ("mkt_google_crypto", "market")):
            if not sp.has(col):
                sp.log.append(f"skipped: {col} not in the base spec")
                continue
            x = base[["coin", "day", col]].pivot(index="day", columns="coin", values=col).reindex(DAYS)
            lx = np.log(x.where(x > 0))
            d = lx - lx.shift(1)
            st = d.stack()
            st.index.names = ["day", "coin"]
            frame = st.rename(f"{col}_dlog").reset_index()
            ch.update(_aligned(base, frame, [f"{col}_dlog"]))
            sp.replace(col, [spec_row(f"{col}_dlog", "attention", scope, 0, "none", notes=sp.note(
                f"day-t daily log change of {col} only (no level, no HAR averages); NaN where a level is <= 0"))])
        return ch, sp, [], {"engine_equivalence": "checked against model.har.FeatureBuilder in the smoke test"}
    if rid == "A66":
        coin = ad[ad.series.isin(COINS)].rename(columns={"series": "coin"})
        ck(same(np.log1p(coin["news_rows"].astype(float)), coin["att_news_rows"]), "A66: log(1 + news_rows) = step 11 att_news_rows")
        ch = _aligned(base, coin.rename(columns={"news_rows": "att_news_rows_n"}), ["att_news_rows_n"])
        ch["att_news_rows_n"] = ch["att_news_rows_n"].astype(float)
        k = I.get("controls").set_index("day").reindex(DAYS)
        ck(same((k.tether_log1p_headlines - k.tether_log1p_headlines.shift(1)), k.tether_dlog_headlines),
           "A66: Tether daily change = first difference of log(1 + headlines) (step 12 rule reproduced)")
        rows_l = k["rob_tether_log1p_rows"]
        ck(same(np.log1p(k["tether_n_rows"].astype(float)), rows_l), "A66: rob_tether_log1p_rows = log(1 + tether_n_rows)")
        ch["ctrl_tether_news_rows"] = _market(base, rows_l)
        ch["ctrl_tether_news_rows_chg"] = _market(base, rows_l - rows_l.shift(1))
        r = sp.row("att_news")
        sp.replace("att_news", [spec_row("att_news_rows_n", "attention", "coin", 1, "log1p_daily", notes=sp.note(
            "RavenPack news volume per entity row (relevance >= 75) instead of distinct headlines; log of each day, HAR"))])
        sp.replace("ctrl_tether_news", [spec_row("ctrl_tether_news_rows", "control", "market", 0, "none", notes=sp.note(
            "log(1 + Tether RavenPack entity rows) instead of distinct headlines (step 12 rob_tether_log1p_rows)"))])
        sp.replace("ctrl_tether_news_chg", [spec_row("ctrl_tether_news_rows_chg", "control", "market", 0, "none",
                                                     notes=sp.note("daily change of log(1 + Tether entity rows)"))])
        return ch, sp, ["attention", "controls"], {}
    if rid == "A24":
        dc = I.get("daily_counts")
        r = dc[dc.platform == "reddit"].copy()
        ck(bool(r.n_dead.notna().all()), "A24: n_dead present on every Reddit series-day")
        r["total"] = (r.n_posts + r.n_dead).astype(float)
        coin = r[r.series.isin(COINS)].rename(columns={"series": "coin"}).rename(columns={"total": "att_reddit_total"})
        ch = _aligned(base, coin, ["att_reddit_total"])
        ch["mkt_reddit_total"] = _market(base, r[r.series == "MARKET"].set_index("day")["total"])
        m = base[["coin", "day", "att_reddit"]].merge(coin[["coin", "day", "n_posts"]], on=["coin", "day"])
        ck(same(m.att_reddit, m.n_posts), "A24: base att_reddit = daily_counts n_posts (live posts after the bot pass)")
        for old, new, scope in (("att_reddit", "att_reddit_total", "coin"), ("mkt_reddit", "mkt_reddit_total", "market")):
            rr = sp.row(old)
            sp.replace(old, [spec_row(new, "attention", scope, 1, rr["transform"], dlog=rr["dlog"], notes=sp.note(
                "Reddit total activity: live posts after the bot pass + dead (removed/deleted) rows, daily_counts.n_dead"))])
        bnb = r[r.series == "BNB"][["n_posts", "n_dead"]].sum()
        return ch, sp, ["daily_counts"], {"reddit_bnb_live_posts": int(bnb.n_posts), "reddit_bnb_dead_rows": int(bnb.n_dead)}
    raise KeyError(rid)


def build_control(I, base, spec0, rid, a, ck):
    sp = SpecEdit(spec0, rid)
    k = I.get("controls").set_index("day")
    fw = k.index >= FIRST_ORIGIN

    def swap(old, new, src, note):
        ck(bool(k.loc[fw, src].notna().all()), f"{rid}: {src} complete from the first forecast origin")
        sp.replace(old, [spec_row(new, "control", "market", 0, "none", notes=sp.note(note))])
        return {new: _market(base, k[src])}
    if rid == "A63":
        return swap("ctrl_baa10y", "ctrl_hy_oas_lag1", "rob_hy_oas_lag1", "ICE BofA US high-yield OAS (BAMLH0A0HYM2), "
                    "level lagged one day, instead of BAA10Y (third-party copy of the FRED series; provenance caveat)"), sp, ["controls"], {}
    if rid == "A64":
        return swap("ctrl_dxy_ret", "ctrl_dtwexbgs_ret_pit", "rob_dtwexbgs_logret_pit", "Fed broad dollar index "
                    "DTWEXBGS log return, point in time (value used from its H.10 release day) instead of DXY"), sp, ["controls"], {}
    if rid == "A64_dayt":
        return swap("ctrl_dxy_ret", "ctrl_dtwexbgs_ret", "rob_dtwexbgs_logret", "Fed broad dollar index DTWEXBGS day-t "
                    "log return (look-ahead up to 7 days: weekly release) instead of DXY; candidate"), sp, ["controls"], {}
    if rid == "A65":
        ck(bool(k.loc[fw, "rob_fng"].notna().all()), "A65: rob_fng complete from the first forecast origin")
        sp.insert_after("ctrl_weekend", [spec_row("ctrl_fng", "control", "market", 0, "none", notes=sp.note(
            "Alternative.me Fear & Greed index, day-t value (level 0-100; stamped 00:00 UTC of day t)"))])
        return {"ctrl_fng": _market(base, k["rob_fng"])}, sp, ["controls"], {}
    if rid == "A37":
        tb = text_block(I, "main")
        ch = _aligned(base, tb.rename(columns={"_pool_empty_share": "ctrl_empty_share"}), ["ctrl_empty_share"])
        ck(bool(pd.Series(ch["ctrl_empty_share"]).between(0, 1).all()), "A37: empty share in [0, 1] on every coin-day")
        sp.insert_after("ctrl_weekend", [spec_row("ctrl_empty_share", "control", "coin", 0, "none", notes=sp.note(
            "share of the coin-day's posts that are empty after EmTract's cleaning (pooled counts, step 10 empty_share)"))])
        return ch, sp, ["text_daily"], {}
    raise KeyError(rid)


def build_volatility(I, base, spec0, rid, a, ck):
    sp = SpecEdit(spec0, rid)
    if rid == "A50":
        vm = I.get("medrv")
        ch = _aligned(base, vm, ["c_med", "j_med"])
        ck(np.allclose(ch["c_med"] + ch["j_med"], base["rv"], rtol=1e-12, atol=1e-12), "A50: c_med + j_med = rv")
        ck(bool((ch["c_med"] > 0).all() and (ch["j_med"] >= 0).all()), "A50: c_med > 0, j_med >= 0")
        sp.replace("c", [spec_row("c_med", "har", "coin", 1, "log", notes=sp.note(
            "log of the 1/7/30-day average of C from MedRV + MedRQ ratio test (ADS 2012; library R04, outage split)"))])
        sp.replace("j", [spec_row("j_med", "har", "coin", 1, "log1p", notes=sp.note("log(1 + average J), MedRV split"))])
        return ch, sp, ["medrv"], {"jump_days_med": int((ch["j_med"] > 0).sum())}
    if rid == "A54":
        v = I.get("volatility")
        ch = _aligned(base, v.rename(columns={"medrv_split": "medrv_split"}), ["medrv_split"])
        vm = I.get("medrv")
        chk = _aligned(base, vm, ["medrv_obs"])
        ck(same(ch["medrv_split"], chk["medrv_obs"]), "A54: medrv_split (step 09) = medrv_obs (step 09b)")
        ck(bool((ch["medrv_split"] > 0).all()), "A54: MedRV > 0 on every coin-day")
        sp.replace("c", [spec_row("medrv_split", "har", "coin", 1, "log", notes=sp.note(
            "HAR-MedRV: log of the 1/7/30-day average of MedRV (ADS 2012) over the 5-minute intervals untouched by an "
            "outage; replaces C and J"))])
        sp.drop(["j"])
        return ch, sp, ["volatility", "medrv"], {}
    raise KeyError(rid)


def build_news_polarity(I, base, spec0, rid, a, ck):
    sp = SpecEdit(spec0, rid)
    mp = I.get("rp_map")
    pol = I.get("rp_pol")
    ck(bool((pol.version == "own").all()) and pol.row_id.is_unique, "A67: one FinBERT score per unique headline")
    inv = {v: k for k, v in RP_ENTITY.items()}
    m = mp[mp.rp_entity_id.isin(inv) & (mp.relevance >= RP_RELEVANCE_MIN)].copy()
    m["coin"] = m.rp_entity_id.map(inv)
    m["day"] = as_day(pd.to_datetime(m.ts_utc, utc=True).dt.floor("D"))
    ck(bool((m["day"].dt.strftime("%Y-%m-%d") == m["rpa_date_utc"].astype(str).str[:10]).all()),
       "A67: UTC day of ts_utc = rpa_date_utc on every coin row")
    m = m[(m.day >= FIRST) & (m.day <= LAST)]
    u = m.drop_duplicates(["coin", "day", "headline_md5"])
    cnt = u.groupby(["coin", "day"]).size().rename("n").reset_index()
    ad = I.get("attention")
    chk = ad[ad.series.isin(COINS)].rename(columns={"series": "coin"})[["coin", "day", "news_headlines_distinct"]].merge(
        cnt, on=["coin", "day"], how="left").fillna({"n": 0})
    ck(same(chk.news_headlines_distinct, chk.n), "A67: distinct headlines per coin-day (md5) = step 11 news_headlines_distinct")
    u = u.merge(pol[["row_id", "polarity"]].rename(columns={"row_id": "headline_md5"}), on="headline_md5", how="left")
    ck(bool(u.polarity.notna().all()), "A67: every coin headline has a FinBERT score")
    g = u.groupby(["coin", "day"])["polarity"]
    agg = pd.DataFrame({"pol_news": g.mean(), "pol_news_sd": g.std(ddof=1), "_n": g.size()}).reset_index()
    agg.loc[agg._n < 2, "pol_news_sd"] = np.nan
    ch = _aligned(base, agg, ["pol_news", "pol_news_sd"])
    sp.insert_after("pol_sd", [
        spec_row("pol_news", "polarity", "coin", 1, "level", csq=1, notes=sp.note(
            "daily mean FinBERT P(pos) - P(neg) of the coin's distinct RavenPack headlines (relevance >= 75); centred "
            "square in window; NaN on days without a headline; no D16 (not defined for news)")),
        spec_row("pol_news_sd", "polarity", "coin", 1, "level", notes=sp.note(
            "within-day SD of headline polarity (NaN below 2 headlines)"))])
    info = {"coin_days_without_headline": int(pd.isna(ch["pol_news"]).sum()),
            "coin_days_sd_missing": int(pd.isna(ch["pol_news_sd"]).sum())}
    return ch, sp, ["rp_map", "rp_pol", "attention"], info


def build_reagg(I, base, spec0, rid, a, ck):
    sp = SpecEdit(spec0, rid)
    rd = I.reagg_dir
    if rd is None or not (rd / f"{rid}_platform_daily.parquet").exists():
        raise FileNotFoundError(f"{rid}: re-aggregation output missing in {rd} (run pipeline/14b_arm_reaggregate.py)")
    path = rd / f"{rid}_platform_daily.parquet"
    I.paths[f"reagg_{rid}"] = path
    ov = pd.read_parquet(path)
    ov["day"] = as_day(ov["day"])
    stp = json.load(open(str(path) + ".stats.json"))
    ck(all(c["passed"] for c in stp["checks"]), f"{rid}: every check of the re-aggregation passed ({len(stp['checks'])})")
    # the override rows replace text_daily main rows: they must come from the same text_daily and calibration
    rec = {Path(x["path"]).name: x["md5"] for x in stp["inputs"]}
    for key in ("text_daily", "calibration"):
        cur = I.paths[key]
        ck(rec.get(cur.name) == md5(cur), f"{rid}: re-aggregation was made against the current {cur.name} "
                                          f"(md5 {md5(cur)}); otherwise re-run pipeline/14b_arm_reaggregate.py")
    ch = {}
    if rid in ("A40", "A42"):
        tb = text_block(I, "main", override=ov)
        ch.update(_aligned(base, tb, TEXT_COLS))
        for n in SPEC_TEXT_ROWS:
            if not same(ch[n], base[n]):
                sp.set(n, a["what"])
    ab = attention_block(I, platform_rows(I, "main", override=ov))
    att_cols = list(ATT_COIN.values()) + list(ATT_MKT.values())
    ch.update(_aligned(base, ab, att_cols))
    changed = [c for c in att_cols if not same(ch[c], base[c])]
    for n in changed:
        sp.set(n, f"post counts with {a['what']}")
    return ch, sp, [f"reagg_{rid}", "text_daily", "attention"], {"reagg_stats": str(path) + ".stats.json"}


BUILDERS = {"text_variant": build_text_variant, "text_subset": build_text_subset, "emo_transform": build_emo_transform,
            "attention": build_attention, "control": build_control, "volatility": build_volatility,
            "news_polarity": build_news_polarity, "reagg": build_reagg}


# =====================================================================================================
# 8. Writers
# =====================================================================================================
def _text_writer(txt: str):
    def w(t):
        with open(t, "w", encoding="utf-8") as fh:
            fh.write(txt)
    return w


def _json_writer(obj):
    def w(t):
        with open(t, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, default=str)
    return w


def write_once(path: Path, writer, kind="parquet") -> str:
    """Write through a temporary file; an existing file is kept if identical and is never replaced otherwise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    if path.exists():
        if md5(path) == md5(tmp):
            os.remove(tmp)
            return "unchanged"
        os.remove(tmp)
        raise FileExistsError(f"{path} exists with different content; not overwritten (use a new --out-dir)")
    os.replace(tmp, path)
    return "written"


def spec_text(spec: pd.DataFrame) -> str:
    return spec[SPEC_COLS].to_csv(index=False, lineterminator="\n")


def write_spec(spec: pd.DataFrame, rid: str, spec_dir: Path, out_dir: Path) -> Path:
    txt = spec_text(spec)
    target = spec_dir / f"variable_spec_{rid}.csv"
    if target.exists() and target.read_text() != txt:
        target = out_dir / "specs" / f"variable_spec_{rid}.csv"
    write_once(target, _text_writer(txt), "csv")
    return target


def check_arm_panel(base, panel, spec, changed, rid, ck):
    ck(len(panel) == len(base) == 6400, f"{rid}: 6,400 coin-days")
    ck(bool((panel["coin"].astype(str).values == base["coin"].astype(str).values).all()
            and (as_day(panel["day"]).values == as_day(base["day"]).values).all()), f"{rid}: same coin-days in the same order as the base")
    keep = [c for c in base.columns if c not in changed]
    bad = [c for c in keep if not (same(panel[c], base[c]) if pd.api.types.is_numeric_dtype(base[c]) or pd.api.types.is_bool_dtype(base[c])
                                   else bool((panel[c].astype(str).values == base[c].astype(str).values).all()))]
    ck(not bad, f"{rid}: every column outside the arm's {len(changed)} changed/added columns equals the base", bad or None)
    missing = [n for n in spec["name"] if n not in panel.columns]
    flags = [f for f in spec["impute_flag"] if f and f not in panel.columns]
    ck(not missing and not flags, f"{rid}: every spec variable and impute flag is a panel column", (missing + flags) or None)
    ck(spec["name"].is_unique, f"{rid}: spec names unique")
    ck(bool(spec["transform"].isin(["log", "log1p", "level", "none", "log_daily", "log1p_daily"]).all()), f"{rid}: spec transforms valid")
    mk = spec[spec.scope == "market"]["name"].tolist()
    diff = [c for c in mk if not (panel.pivot(index="day", columns="coin", values=c).nunique(axis=1, dropna=False) <= 1).all()]
    ck(not diff, f"{rid}: market-scope variables equal across coins on every day", diff or None)
    for c in changed:
        ck(c in panel.columns, f"{rid}: changed column {c} present")
    num = [c for c in changed if pd.api.types.is_numeric_dtype(panel[c])]
    inf = [c for c in num if np.isinf(panel[c].astype(float)).any()]
    ck(not inf, f"{rid}: no infinite value in the arm columns", inf or None)


def build_arm(I, base, base_raw, spec0, rid, out_dir: Path, spec_dir: Path, base_path: Path, code_md5: str) -> dict:
    """base: the A00 panel with a normalised day column (joins); base_raw: the panel as read (written out)."""
    a = ARMS[rid]
    ck = Checks()
    ch, sp, used, info = BUILDERS[a["kind"]](I, base, spec0, rid, a, ck)
    panel = base_raw.copy()
    for c, s in ch.items():
        panel[c] = pd.Series(np.asarray(s), index=base_raw.index)
    changed = sorted(ch)
    really = [c for c in changed if c not in base_raw.columns or not same(panel[c], base_raw[c])]
    check_arm_panel(base_raw, panel, sp.s, changed, rid, ck)
    ck(len(really) > 0, f"{rid}: the arm changes at least one column of the base", really)
    out = out_dir / f"{rid}.parquet"
    status = write_once(out, lambda t: panel.to_parquet(t, index=False))
    back = pd.read_parquet(out)
    ck(len(back) == len(panel) and list(back.columns) == list(panel.columns), f"{rid}: written panel reads back with all rows and columns")
    ck(all(same(back[c], panel[c]) for c in changed), f"{rid}: arm columns read back identical")
    spath = write_spec(sp.s, rid, spec_dir, out_dir)
    stats = {"step": STEP, "arm": a["arm"], "run_id": rid, "written_at_utc": now_utc(), "python": sys.version.split()[0],
             "code_md5": code_md5, "code": str(Path(__file__).resolve()),
             "base": {"path": str(base_path), "md5": md5(base_path)},
             "inputs": I.used([u for u in used if u in I.paths]),
             "output": {"path": str(out), "md5": md5(out), "rows": int(len(panel)), "status": status},
             "variable_spec": {"path": str(spath), "md5": md5(spath), "base_spec_edits": sp.log},
             "construction": {"what": a.get("what"), "reading": a.get("reading"), "scope": a.get("scope")},
             "changed_columns": really, "added_columns": [c for c in changed if c not in base.columns],
             "info": info, "checks": ck.items}
    sfile = Path(str(out) + ".stats.json")
    if status == "unchanged" and sfile.exists():
        stats["stats_file"] = "kept (panel unchanged)"
    else:
        write_once(sfile, _json_writer(stats), "json")
    return {"run_id": rid, "panel": out, "spec": spath, "stats": stats}


# =====================================================================================================
# 9. Manifest
# =====================================================================================================
def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(CODE))
    except ValueError:
        return str(p)


# Engine-side notes for built runs (what the frozen engine of 28 Oct does with the arm's design).
ENGINE_NOTES = {
    "A60": "full run (with in-sample tests) stops at model.inference._h4_weights, which asserts emo_fear_h1/h7/h30 "
           "in R3; the out-of-sample ladder runs (smoke test with --no-insample passed). Needs an engine guard for H4 "
           "(engine track) or a run with --no-insample (lead's choice)",
}
RUN_STATUS = {"A60": "engine blocker", "A64": "open question", "A64_dayt": "open question"}


def engine_cmd(panel: str, spec: str, label: str, extra: str = "--spec A00") -> str:
    return f"cd code && PYTHONPATH=. python -m model.run {extra} --rung all --panel {panel} --variable-spec {spec}" + \
           (f" --label {label}" if label and "--label" not in extra else "")


def write_manifest(results: Dict[str, dict], base_path: Path, base_spec: Path, manifest: Path, runlist: Path) -> None:
    reg = pd.read_csv(REGISTER)
    rows, runs = [], []
    byarm: Dict[str, List[str]] = {}
    for rid, a in ARMS.items():
        byarm.setdefault(a["arm"], []).append(rid)
    for aid, sub in reg.groupby("arm_id", sort=False):
        f = sub.iloc[0]
        status = str(f["status"])
        rec = dict(arm_id=aid, role=f["role"], status=status, track="", build_status="", panel_path="", variable_spec="",
                   engine_args="", evaluation_set=f["evaluation_set"], notes="")
        if aid == "A00":
            rec.update(track="base", build_status="built", panel_path=rel(base_path), variable_spec=rel(base_spec),
                       engine_args=engine_cmd(rel(base_path), rel(base_spec), "", "--spec A00"),
                       notes="the A00 panel this run was built from (step 13); confirmatory")
        elif status.startswith(("withdrawn", "superseded")):
            rec.update(track="skipped", build_status=status.split()[0], notes=status)
        elif aid in ENGINE_TRACK:
            rec.update(track="engine", build_status="engine track", panel_path=rel(base_path))
            if aid in ENGINE_DIRECT:
                rec.update(variable_spec=rel(base_spec), engine_args=engine_cmd(rel(base_path), rel(base_spec), aid, ENGINE_DIRECT[aid]))
                rec["notes"] = "implemented inside the frozen engine (config.SPEC_ARMS or a CLI option); built by the engine track"
            else:
                rec["notes"] = "engine track (not built here)"
        elif aid in byarm:
            done = [r for r in byarm[aid] if r in results]
            rec["track"] = "data"
            if done:
                rec["panel_path"] = " ; ".join(rel(results[r]["panel"]) for r in done)
                rec["variable_spec"] = " ; ".join(rel(results[r]["spec"]) for r in done)
                rec["engine_args"] = " ; ".join(engine_cmd(rel(results[r]["panel"]), rel(results[r]["spec"]), r) for r in done)
                cand = [r for r in done if r.endswith("_dayt")]
                rec["build_status"] = "open question" if aid == "A64" else "built"
                notes = [f"{r}: {ARMS[r]['what']} [{ARMS[r]['reading']}]" + (f" ENGINE: {ENGINE_NOTES[r]}" if r in ENGINE_NOTES else "")
                         for r in done]
                rec["notes"] = " || ".join(notes)
            else:
                rec["build_status"] = "not built"
                rec["notes"] = "builder output missing in this run"
        elif aid in NOT_BUILT:
            rec.update(track="data", build_status=NOT_BUILT[aid][0], notes=NOT_BUILT[aid][1])
        else:
            rec.update(track="?", build_status="unassigned", notes="not in this script's registry")
        rows.append(rec)
    for rid, r in results.items():
        a = ARMS[rid]
        role = str(reg.loc[reg.arm_id == a["arm"], "role"].iloc[0])
        runs.append(dict(run_id=rid, arm_id=a["arm"], role=role, status=RUN_STATUS.get(rid, "ready"),
                         note=ENGINE_NOTES.get(rid, ""), kind=a["kind"], scope=a.get("scope", ""),
                         panel_path=rel(r["panel"]), panel_md5=r["stats"]["output"]["md5"],
                         variable_spec=rel(r["spec"]), spec_md5=r["stats"]["variable_spec"]["md5"],
                         engine_args=engine_cmd(rel(r["panel"]), rel(r["spec"]), rid),
                         candidate_only=rid in ("A64_dayt",), changed_columns=" ".join(r["stats"]["changed_columns"])))
    man = pd.DataFrame(rows)
    assert len(man) == reg.arm_id.nunique() == 72, len(man)
    rdf = pd.DataFrame(runs)
    st_m = write_once(manifest, _text_writer(man.to_csv(index=False, lineterminator="\n")), "csv")
    st_r = write_once(runlist, _text_writer(rdf.to_csv(index=False, lineterminator="\n")), "csv")
    return man, rdf, st_m, st_r


def write_list_stats(path: Path, status: str, df: pd.DataFrame, inputs: List[dict], checks: List[dict], code_md5: str) -> None:
    """<csv>.stats.json for the manifest and the run list (kept as is when the csv is unchanged and the stats exist)."""
    sfile = Path(str(path) + ".stats.json")
    if status == "unchanged" and sfile.exists():
        return
    stats = {"step": STEP, "written_at_utc": now_utc(), "code_md5": code_md5, "inputs": inputs,
             "output": {"path": str(path), "md5": md5(path), "rows": int(len(df)), "status": status}, "checks": checks}
    write_once(sfile, _json_writer(stats), "json")


# =====================================================================================================
# 10. main
# =====================================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="A00 panel parquet (step 13 output)")
    ap.add_argument("--arms", default="all", help="comma-separated run ids or register arm ids, or 'all'")
    ap.add_argument("--out-dir", help="directory for the arm panels")
    ap.add_argument("--base-spec", default=str(CODE / "model" / "variable_spec_A00.csv"))
    ap.add_argument("--spec-dir", default=str(CODE / "model"))
    ap.add_argument("--reagg-dir", default=str(PANEL / "arms" / "reagg_20261028"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--scope", action="append", default=[], help="override ARM=text|series for a text variant arm")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for rid, x in ARMS.items():
            print(f"{rid:16s} {x['arm']:4s} {x['kind']:14s} {x.get('scope', ''):7s} {x['what']}")
        for aid, (st, why) in NOT_BUILT.items():
            print(f"{aid:16s} {aid:4s} {'not built':14s} {st}: {why[:120]}")
        return 0
    for o in a.scope:
        k, v = o.split("=")
        assert ARMS[k]["kind"] == "text_variant" and v in ("text", "series"), o
        ARMS[k]["scope"] = v
        ARMS[k]["reading"] += f" [scope overridden to {v} on the command line]"
    base_path, out_dir = Path(a.base).resolve(), Path(a.out_dir).resolve()
    t0 = dt.datetime.now(dt.timezone.utc)
    base_raw = pd.read_parquet(base_path)
    base = base_raw.copy()
    base["day"] = as_day(base["day"])
    ck = Checks()
    ck(len(base) == 6400 and not base.duplicated(["coin", "day"]).any(), "base: 6,400 unique coin-days")
    need = set(TEXT_COLS) | set(ATT_COIN.values()) | set(ATT_MKT.values()) | set(CTRL) | {"rv", "c", "j", "att_wiki", "att_news",
                                                                                         "twitter_absent", "att_google"}
    ck(not (need - set(base.columns)), "base: every A00 column this step uses is present", sorted(need - set(base.columns)) or None)
    spec0 = read_spec(a.base_spec)
    I = Inputs(Path(a.reagg_dir) if a.reagg_dir else None)
    identity_checks(I, base, ck)
    print(f"identity checks passed: {len(ck.items)}", flush=True)
    req = list(ARMS) if a.arms == "all" else list(dict.fromkeys(
        r for x in a.arms.split(",") for r in ARMS if r == x or ARMS[r]["arm"] == x))
    unknown = [x for x in ([] if a.arms == "all" else a.arms.split(",")) if not any(r == x or ARMS[r]["arm"] == x for r in ARMS)]
    ck(not unknown, "every requested arm is in the registry", unknown or None)
    results, failed = {}, {}
    code_md5 = md5(Path(__file__).resolve())
    for rid in req:
        try:
            results[rid] = build_arm(I, base, base_raw, spec0, rid, out_dir, Path(a.spec_dir), base_path, code_md5)
            print(f"[{rid}] built: {rel(results[rid]['panel'])}  changed {results[rid]['stats']['changed_columns']}", flush=True)
        except FileNotFoundError as e:
            failed[rid] = str(e)
            print(f"[{rid}] NOT BUILT: {e}", flush=True)
    manifest = Path(a.manifest).resolve() if a.manifest else out_dir / "arm_manifest.csv"
    runlist = manifest.with_name(manifest.stem.replace("arm_manifest", "arm_runs") + ".csv"
                                 if "arm_manifest" in manifest.stem else manifest.stem + "_runs.csv")
    assert runlist != manifest
    man, rdf, st_m, st_r = write_manifest(results, base_path, Path(a.base_spec).resolve(), manifest, runlist)
    lck = Checks()
    reg = pd.read_csv(REGISTER)
    lck(len(man) == 72 and man.arm_id.is_unique and set(man.arm_id) == set(reg.arm_id),
        "manifest: one row per register arm (72, unique, same ids as the register)")
    lck(set(man.track) <= {"base", "data", "engine", "skipped"}, "manifest: track in base/data/engine/skipped")
    for _, r in rdf.iterrows():
        pp, ss = CODE / r.panel_path if not os.path.isabs(r.panel_path) else Path(r.panel_path), \
                 CODE / r.variable_spec if not os.path.isabs(r.variable_spec) else Path(r.variable_spec)
        lck(pp.exists() and md5(pp) == r.panel_md5 and ss.exists() and md5(ss) == r.spec_md5,
            f"run list {r.run_id}: panel and spec exist with the listed md5")
    lck((len(rdf) == 0 and not results) or (rdf.run_id.is_unique and set(rdf.run_id) == set(results)),
        "run list: one row per built run")
    ins = [{"path": str(p), "md5": md5(p)} for p in (base_path, Path(a.base_spec).resolve(), REGISTER)]
    write_list_stats(manifest, st_m, man, ins, lck.items, code_md5)
    write_list_stats(runlist, st_r, rdf, ins, lck.items, code_md5)
    run = {"step": STEP, "written_at_utc": now_utc(), "code_md5": code_md5, "base": {"path": str(base_path), "md5": md5(base_path)},
           "base_spec": {"path": a.base_spec, "md5": md5(a.base_spec)}, "inputs": I.used([k for k in I.paths if k in I._cache]),
           "arms_built": sorted(results), "arms_not_built": failed, "manifest": str(manifest), "run_list": str(runlist),
           "seconds": round((dt.datetime.now(dt.timezone.utc) - t0).total_seconds(), 1), "identity_checks": ck.items}
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_once(out_dir / f"_build_{stamp}.stats.json", _json_writer(run), "json")
    print(f"built {len(results)} panels, {len(failed)} not built; manifest {manifest} ({st_m}); run list {runlist.name} ({st_r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
