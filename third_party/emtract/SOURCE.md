# EmTract, vendored subset

- Upstream: https://github.com/dvamossy/EmTract
- Commit: 4212881cbccd12c4cca08954a3bd37123812a26e (2024-02-15, "Delete doc/paper.bib")
- License: MIT, Copyright (c) 2021 dvamossy (upstream `LICENSE`, copied unchanged)
- Vendored on 29 Sep 2026 from a fresh `git clone` + `git checkout` of the commit above.

## Copied files

Layout mirrors upstream, so `from emtract.processors.cleaning import clean_text` works with this folder on `sys.path`.

| File here | Upstream path | Bytes (upstream) | sha256 (upstream) | sha256 (here) |
|---|---|---|---|---|
| LICENSE | LICENSE | 1065 | caf731ed1470b2ab819752217829c49eea96b0aca0b4233ebd1bcad006eb574c | same |
| emtract/__init__.py | emtract/__init__.py | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | same |
| emtract/processors/__init__.py | emtract/processors/__init__.py | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | same |
| emtract/processors/cleaning.py | emtract/processors/cleaning.py | 14220 | 97c6df8075625d96903167c1f24c5a7f01ba7e270c6e9b74bb85393f040d126c | db775342e77f221a127eb194d845173df9e56fe8e762d89265517c0711842931 (changed, see below) |
| emtract/model.py | emtract/model.py | 2459 | 0ac9a05ef8573f4f869cc88c1b51f0ad2747c753397c8322afeb751966262367 | same |
| emtract/data/dictionaries/misspell_df.csv | same path | 220886 | f06d737fd9ef7fc25a655216d1a5af76b0054f1d8aa6188b5b6648f47d8f32d3 | same |
| emtract/data/dictionaries/emoticons.csv | same path | 2814 | 3ae1ed7fb810e7b0f1af79e1256de940b1a8a4db174bb81f162fb95620eb4505 | same |
| emtract/data/dictionaries/blacklist_df.csv | same path | 812602 | 25fc2c3400284b7fedcc35126560d1e1af4346265d8d46d96c2de07815c66502 | same |
| emtract/data/dictionaries/symbol_df.csv | same path | 199252 | d59de58b5c0620fb7052dd8b830885aadb4e4930e1a608a468c402b45fb653c4 | same |
| emtract/data/dictionaries/company_name_df.csv | same path | 131219 | 73e26e59801a335ce5ab74ce35b65654deb594aad00347c882053e74e97f0f5e | same |
| emtract/data/dictionaries/symspell_dict.csv | same path | 1332885 | 708f1bba21b0348bf09ec2ecc7145360f2c384780a0663f8740a68ef4fee34f8 | same |

Why these files: `model.py` imports only `.processors.cleaning` from the package (other imports: `datasets`, `pandas`, `scipy`, `torch`, `transformers`, `enum`). `cleaning.py` imports no other package module; at import time it reads exactly the six CSV files above from `BASE_DIR = "emtract/data/dictionaries/"` (upstream reads symspell_dict.csv only in its fallback branch; the vendored file always reads it, see below). The two `__init__.py` files (both empty) are needed for the package import. `emotion_df.csv` and `finance_df.csv` are not read by either module and were not copied.

Not copied, deliberately: `emtract/data/tagged_sample.parquet.snappy`, `emotion_sources.parquet.snappy`, `symbols_companies.parquet.snappy`, `vocab.parquet.snappy`, `emtract/data/__init__.py`, `tweets_example.csv`, notebooks, `inference.py`, `model_inference.py`, model weights.

## Post-text check of the copied data files

The six CSV files are word lists. Checked with pandas on 29 Sep 2026 (per column: rows, maximum whitespace-separated tokens per entry): misspell_df 8,438 rows (`original` max 1 token; `changed` max 6 tokens, these are contraction and misspelling expansions; `type`; an empty unnamed column); emoticons 168 rows (max 2 tokens); blacklist_df 64,698 rows (`word` max 1 token, `counts`); symbol_df 18,021 tickers (1 token); company_name_df 9,519 titles (1 token); symspell_dict 82,834 rows (`word` max 1 token, `counts`). No entry has more than 6 tokens; the only entries containing "@" are two emoticons; one blacklist entry contains the string "http" as a single token. No copied file contains post text.

## Changes

One change, in `emtract/processors/cleaning.py` (upstream lines 67-75). Upstream fetches the SymSpell frequency list from GitHub at import time and falls back to the bundled CSV on any error:

```
-try:
-    symspell_list = pd.read_csv(
-        "https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt",
-        sep="\s",
-        names=["word", "counts"],
-    )["word"].tolist()
-
-except:
-    symspell_list = pd.read_csv(BASE_DIR + "symspell_dict.csv")["word"].tolist()
+# Vendored change (see SOURCE.md): read the bundled word list instead of the import-time GitHub fetch.
+symspell_list = pd.read_csv(BASE_DIR + "symspell_dict.csv")["word"].tolist()
```

Identity check, done before the change (29 Sep 2026, 08:09 UTC): the GitHub master file was downloaded with curl (1,332,879 bytes, sha256 68e9dc81c7e73bd7310b57e516ecaea0d8b6387ff71344a57c04174650a407a7) and parsed with the exact upstream call (`sep="\s"`, `names=["word","counts"]`). Its `word` list equals the `word` column of `symspell_dict.csv` element by element and in order (82,834 entries each, 0 differing positions; the `counts` columns are also equal). In both paths two entries parse as NaN (pandas default NA strings); the edit does not change that. After the change, the vendored `clean_text` and the upstream `clean_text` (which did fetch from GitHub in that run; the pandas regex-separator warning shows the URL branch executed) gave identical output on all 10,000 bodies of the upstream tagged sample (sha256 of the outputs joined with U+001E: 2259170cef3b091aa8268177dd7b712fee779882f9c35fc403459427c9560e60 in both).

Note: the SymSpell dictionary that symspellpy 6.7.0 bundles (loaded by `sym_spell.load_dictionary` via `pkg_resources`; not changed here) is a different file: 82,781 entries, sha256 b94cf9c7a78169e704099d26f731c55d329f9835893840547f7b0d5542612454. Word segmentation therefore depends on the installed symspellpy version (6.7.0 in the reference environment).

## Unchanged upstream behaviour to be aware of

- `BASE_DIR` is relative (`emtract/data/dictionaries/`), so `cleaning.py` must be imported with the working directory set to this folder (or the module-level reads fail).
- `model.py` loads the tokenizer from `./build/models/emotion-twitter` at import time (for both model variants) and builds `TrainingArguments(output_dir="results")`; importing it needs the model files under `build/models/` relative to the working directory. It also needs `datasets` (1.14.0 in the reference environment) and transformers < 4.51.
- Import creates `__pycache__/` next to the modules unless `PYTHONDONTWRITEBYTECODE=1` is set.
- Reference environment `emtract-ref`: Python 3.9.15, torch 1.11.0 (CPU), transformers 4.10.3, tokenizers 0.10.3, emoji 1.7.0, advertools 0.13.1, symspellpy 6.7.0, setuptools 59.8.0, datasets 1.14.0; exact list in the artifact `emtract_ref_env_versions.txt` of the 29 Sep 2026 reference checks.
