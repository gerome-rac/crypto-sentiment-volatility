# data/ (local only)

Nothing in this folder except this file is in git. Post-level data is never published; the
platforms' terms restrict the redistribution of post content.

## Inputs (`data/raw/`, `data/control_var/`)

| Input | Source | What the pipeline checks |
|---|---|---|
| Reddit raw archive, `raw/reddit_arcticshift_crypto/` | Arctic Shift dumps of the 14 study subreddits (posts and comments), 1 Aug 2019 to 31 Jan 2023 | `01_extract_reddit.py verify`: count, bytes and md5 of every object against the manifest, and `gzip -t` on every archive |
| StockTwits messages, one Parquet file per month | written by `01_extract_stocktwits.py` from the StockTwits dump (restricted, not redistributable) | `02_filter_stocktwits.py` reads a local copy |
| Twitter 1% sample, one Parquet folder per day | written by `01_extract_twitter.py` from the Internet Archive's Twitter Stream Grab | `02_filter_twitter.py coarse` checks the md5 of every file it fetches (remote name in `THESIS_DRIVE_REMOTE`) |
| Twitter capture ledgers, `raw/twitter_capture_ledgers/` | written by `01_extract_twitter.py`, one per archive | input of the coverage table |
| Reddit author salt | a random secret outside the repository (`SALT_PATH` in `01_extract_reddit.py`); the author hashes depend on it | only its fingerprint is written to stats files |
| Model files, `raw/emtract_models/`, `raw/polarity_models/` | see the main README | sha256 |
| DLT-Tweets shards, `raw/dlt/data/` | Hugging Face `ExponentialScience/DLT-Tweets`, revision c7d9793 | |
| Binance 1-minute klines, `raw/binance_spot_1m/` | Binance public data archive | |
| RavenPack news extract, Google Trends, Wikipedia page views, macro controls, `control_var/` | WRDS (licensed), Google Trends, Wikimedia REST API, FRED and the other sources named in Appendix A of the thesis | |

## Outputs (`data/pipeline/`)

```
reddit/extract/      reddit_<P>.parquet                   step 1 (P = BTC, ETH, BNB, XRP, DOGE, MARKET)
<platform>/coarse/   one file per group, month or day     step 2
twitter/units/       units, retweet_events_daily,         step 2 (Twitter)
                     text_variants, twitter_coverage_utc
<platform>/bot/      post_flags, campaign and template    step 3
<platform>/panel/    assignment.parquet                   step 4
panel/               daily_counts.parquet and the panels  steps 5 and 9-15
<platform>/emtract/  tokens, scores, emotion              step 6
<platform>/polarity/ tokens, scores, polarity             step 7
model_runs/          one folder per model run             model/run.py
```

Every output has a stats file next to it with its inputs (path, bytes, md5), the code and rules
fingerprints, the settings, the counts and every check.

The daily counts table has 0 Twitter posts on the 76 UTC days without any capture. Only
`twitter/units/twitter_coverage_utc.parquet` (`minutes_utc = 0`) marks those days, so a Twitter
measure must treat them as missing.
