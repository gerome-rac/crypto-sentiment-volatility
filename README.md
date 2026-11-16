# Multi-Dimensional Social-Media Sentiment and the Realized Volatility of Major Cryptocurrencies

Code of the bachelor's thesis of Gérome Racordon, University of St. Gallen (2026).

The code builds daily text measures for BTC, ETH, BNB, XRP and DOGE from Reddit, StockTwits and
Twitter posts (1 August 2019 to 31 January 2023), and estimates four nested panel HAR-CJ models of
next-day realized volatility: R0 (HAR-CJ with controls), R1 (+ attention), R2 (+ FinBERT polarity),
R3 (+ six EmTract emotions).

## Layout

```
pipeline/        data steps; the file number is the run order
  01-05            raw archives -> filtered posts -> bot flags -> coin assignment -> daily counts
  06, 07           EmTract emotion scores and FinBERT polarity for every post
  08-13            Google Trends, realized volatility (continuous and jump parts), daily text
                   series, attention measures, controls, the panel
  14, 15           panels of the robustness runs
  rules.py         every word list, pattern and threshold, with its source
  config.py        every path and fixed setting (each path can be moved with a THESIS_* variable)
model/           the forecasting models and tests; one variable_spec_*.csv per specification
tests/           tests on hand-made posts and texts
notebooks/       one notebook per platform and one for the scoring, on made-up examples
thesis_outputs/  scripts behind the tables, figures and additional analyses of the thesis
validation/      result tables of the hand-labeled sample (labels only, no post texts)
third_party/     EmTract's text cleaning code (MIT licence)
data/            local only; see data/README.md
```

Each script starts with a header that states what the step does, the rules it applies, what it
checks and what it writes, and lists its commands.

## Setup

```bash
conda env create -f environment.yml               # environment "thesis": everything except the next line
conda env create -f environment-emtract-ref.yml   # environment "emtract-ref": EmTract tokenization (tokenizers 0.10.3)
```

Steps 08 and 11b are R scripts and need R with the packages named in their headers.
A full run needs 16 GB of memory and about 45 GB of free disk for temporary files.

## Running

All commands run from the repository root.

**Tests (no data needed).**

```bash
conda activate thesis
python tests/test_rules.py
python tests/test_pipeline_sample.py
python tests/test_polarity_finbert.py
conda activate emtract-ref
python tests/test_emotion_emtract.py
```

**Steps 01-05, text pipeline.**

```bash
python pipeline/run_all.py --list         # prints every command in order
python pipeline/run_all.py                # runs steps 01-05
python pipeline/run_all.py --from 3       # starts at a later step
python pipeline/run_all.py --platforms reddit
```

Every step writes to a temporary name, runs its checks, and only then moves its output into place
with a stats file. A step whose inputs, code and rules are unchanged is skipped.

**Steps 06-07, emotion and polarity.** Each model runs in three stages: tokenize, score, merge.
`<platform>` is `reddit`, `stocktwits` or `twitter`.

```bash
conda activate emtract-ref
python pipeline/06_emotion_emtract.py tokenize <platform>
conda activate thesis
python pipeline/06_emotion_emtract.py pack <platform> --batch-id <id> --max-inputs 5000000
python pipeline/06_emotion_emtract.py score --device cuda --inputs <packed file> --out <score file>
python pipeline/06_emotion_emtract.py collect <platform> --batch-id <id>
python pipeline/06_emotion_emtract.py merge <platform>

python pipeline/07_polarity_finbert.py fetch-model
python pipeline/07_polarity_finbert.py selfcheck
python pipeline/07_polarity_finbert.py tokenize <platform>
python pipeline/07_polarity_finbert.py pack <platform> --batch-id <id> --max-inputs 5000000 --windows
python pipeline/07_polarity_finbert.py score --device cuda --inputs <packed file> --out <score file>
python pipeline/07_polarity_finbert.py collect <platform> --batch-id <id>
python pipeline/07_polarity_finbert.py merge <platform>
```

`score` is plain PyTorch and also runs with `--device cpu` or `--device mps`. The model files are
not in the repository. They are fetched from their sources and checked against these sha256 values:

| Model | Source | sha256 of the weights |
|---|---|---|
| EmTract `emotion-stocktwits` | EmTract's public model store (code: upstream commit 4212881) | e5d532f4cfa4590fd38d1d1ee00b8b0f216f4f001e14e3ba056de0d823f3bb0d |
| FinBERT | Hugging Face `ProsusAI/finbert`, revision 4556d13 | e15a7b5738df7f17553399b6d94c6e2ff69c89245d066e8e5d183f5803a554e3 |

**Steps 08-15, measures and panel.** Run the scripts in the order of their numbers; the header of
each lists its command and inputs.

**Models.**

```bash
python -m model.run --spec A00 --rung all --panel data/pipeline/panel
```

`--spec` selects the specification (`model/variable_spec_<spec>.csv`); the header of `model/run.py`
lists the options and the files a run writes. `python -m pytest model/tests` runs the model tests.

**Tables and figures.** The scripts in `thesis_outputs/` read the stored model runs and write the
tables and figure data (see `thesis_outputs/README.md`).

## Data

No data is in this repository. The terms of Reddit, StockTwits and Twitter restrict the
redistribution of post content, and the news data is licensed. `data/README.md` lists every input,
its source and how the pipeline checks it, so that the steps can be run on inputs obtained from
those sources.

## License

MIT for the code in this repository (`LICENSE`). `third_party/emtract/` keeps its own licence.
