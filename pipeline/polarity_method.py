"""
polarity_method.py -- the polarity scoring's models, paths and text rules (step 07,
07_polarity_finbert.py). Importing it needs only the standard library; nothing here runs a model or
reads a table.

Written 1 Oct 2026 for decision D19 (one polarity model for hypothesis H2). FinBERT (ProsusAI/finbert,
Araci 2019) was chosen and scored for every corpus; CryptoBERT (ElKulako/cryptobert) stays in MODELS,
selectable with --model, because the D19 comparison used it. Everything that depends on the model is in
MODELS below; the step's code is model-agnostic.

    tokenize (CPU)  the ORIGINAL text of every row (the same string EmTract's stage 1 started from; no
                    EmTract cleaning, no other preprocessing), the model's own tokenizer at a pinned
                    Hugging Face commit, uncut, plus the window plan
                    -> <platform>/polarity/<tag>/tokens/<partition>/shard-NNNNN.parquet
    pack | score | collect (GPU for score)  token ids only go to the GPU; the model alone scores
                    [prefix] + the first (max_len - 2) tokens + [suffix]
                    -> <platform>/polarity/<tag>/scores/<batch>.parquet
    merge (CPU)     one row per (row_id, version), keyed like the EmTract emotion tables, with logits,
                    probabilities and the polarity score P(positive) - P(negative)
                    -> <platform>/polarity/<tag>/polarity/<partition>.parquet

<tag> = <model>-L<max_len>, e.g. finbert-L64 (the window plan depends on max_len). Sources of the model
facts: the thesis's D19 model-facts note (§2.1, §2.2, §4, §10) and the Hugging Face file trees at the
pinned commits (read 1 Oct 2026).

The text rules, with their sources
----------------------------------
1. Rows and versions. Same rows and the same original strings as EmTract's stage 1:
   reddit/stocktwits/twitter go through emtract_method.build_versions (Reddit dead rows get no
   version; StockTwits NULL bodies are missing), then only the requested versions are kept
   (default own). dlt (own = units.text, every row) and ravenpack (own = the unique headline)
   have only 'own', as in their EmTract arms (06_emotion_emtract.py, sections 5 and 6).
   No EmTract cleaning and no other preprocessing (D19 facts §10, rule 2).
2. Flags. is_missing_text: the source text is NULL (no model call). is_empty_text: the text exists
   but the model's own tokenizer gives 0 content tokens (empty or whitespace-only); no model call,
   NULL score (D19 facts §10, rule 1: the model's own empty rule, stored apart from EmTract's
   is_empty_after_clean).
3. Text key (optional, for --dedup): sha256 of the original text's UTF-8 bytes, 32 raw bytes.
   Identical text -> identical ids -> identical model input, so one score serves all its rows.
4. Tokens: the model's own fast tokenizer from the pinned files, add_special_tokens=False, no
   truncation, no padding: the full content-token sequence. The model input is
   prefix + ids[s : s + W] + suffix with W = max_len - 2; the prefix/suffix ids ([CLS]/[SEP] for
   BERT, <s>/</s> for RoBERTa) are read from the tokenizer, never hard-coded. Window 0 (s = 0) is
   the primary input and equals the tokenizer's own truncation=True, max_length=max_len output
   (checked on a sample by `tokenize`).
5. Windows (optional robustness input, as EmTract A13): k = ceil((n - 16) / (W - 16)) windows for
   n > W, consecutive windows overlapping by >= 16 tokens; ideal starts spread evenly from 0 to
   n - W (rounded half up); each interior start moves to the nearest word start within 8 tokens if
   every overlap stays >= 16 (EmTract stage 1's rule, emtract_method.window_starts, generalised
   from W = 62 to any W; identical for W = 62, tested). Word start = the first token of a word of
   the tokenizer's own pre-tokenizer (BatchEncoding.word_ids): for WordPiece this is every token
   not starting with '##' (as EmTract stage 1), for byte-level BPE the first token of each pre-token.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import OUT_DIR, RAW_DIR, out_path
from .emtract_method import CheckError, check, log  # noqa: F401  (runner's helpers, or the same three without DuckDB)


# ==================================================================================================
# 1. Models (pinned Hugging Face commits, weights sha256, label order, lengths)
# ==================================================================================================
@dataclass(frozen=True)
class ModelSpec:
    slug: str
    repo: str                     # Hugging Face repo id
    revision: str                 # full commit sha (pinned)
    files: Tuple[str, ...]        # files downloaded: config + tokenizer + one weights file
    weights: str                  # the weights file among `files`
    weights_sha256: str           # LFS sha256 of the weights file at `revision` (HF tree API)
    labels: Tuple[str, ...]       # config.id2label by index, lower-cased (asserted at load)
    pos_label: str                # polarity = P(pos_label) - P(neg_label)
    neg_label: str
    default_max_len: int          # primary length incl. the 2 special tokens
    max_positions: int            # usable positions incl. specials (RoBERTa: 514 - 2 offset)
    note: str = ""


MODELS: Dict[str, ModelSpec] = {
    # Araci (2019), arXiv 1908.10063. BertForSequenceClassification, bert-base-uncased WordPiece
    # (30,522; md5 = Google's vocab.txt). No tokenizer.json on HF: the fast tokenizer is built by
    # lossless conversion (fast = slow on 20,000/20,000 posts, D19 facts §4). Trained at 64 tokens
    # (Araci 2019, §4.5). Score P(positive) - P(negative) is the repo's own (README, finbert.py).
    "finbert": ModelSpec(
        slug="finbert", repo="ProsusAI/finbert", revision="4556d13015211d73dccd3fdd39d39232506f3e43",
        files=("config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt", "pytorch_model.bin"),
        weights="pytorch_model.bin",
        weights_sha256="e15a7b5738df7f17553399b6d94c6e2ff69c89245d066e8e5d183f5803a554e3",
        labels=("positive", "negative", "neutral"), pos_label="positive", neg_label="negative",
        default_max_len=64, max_positions=512,
        note="HF card: no preprocessing; repo predict(): sentence split + head/tail cut (not used here)"),
    # Kulakowski & Frasincar (2023). RobertaForSequenceClassification, byte-level BPE (50,265;
    # vocab/merges = roberta-base). tokenizer.json shipped. Trained at 128 (card); card example 64.
    # Labels are StockTwits self-tags (Bearish / Neutral / Bullish).
    "cryptobert": ModelSpec(
        slug="cryptobert", repo="ElKulako/cryptobert", revision="9e37c910fe87727cb842a9ac55c6388256fe0f15",
        files=("config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json",
               "vocab.json", "merges.txt", "model.safetensors"),
        weights="model.safetensors",
        weights_sha256="fe34ba09b3d701ff2b4ee69205e42c8143fcee1f89fe3a0cf10444988634a3c9",
        labels=("bearish", "neutral", "bullish"), pos_label="bullish", neg_label="bearish",
        default_max_len=128, max_positions=512,
        note="dataset-card cleaning (lowercase, drop cashtags/hashtags/users/URLs) has no code; not applied"),
}


# ==================================================================================================
# 2. Inputs, windows and padding
# ==================================================================================================
PLATFORMS = ("reddit", "stocktwits", "twitter", "dlt", "ravenpack")
VERSIONS = ("own", "quoted", "both")          # as EmTract (emtract_method.VERSIONS); default: own only
DEFAULT_VERSIONS = ("own",)
MIN_WINDOW_OVERLAP = 16                       # as EmTract stage 1 (a quarter of 64; Dai et al. 2022, p. 6)
MAX_MOVE = 8                                  # as EmTract stage 1: an interior start moves <= 8 tokens
PAD_STEP = 16                                 # padding buckets 16, 32, ..., max_len (as EmTract stage 2)
DEFAULT_BATCH_TOKENS = 65536                  # padded tokens per GPU batch (as EmTract stage 2 / the benchmark)
PROB_SUM_TOL = 1e-5                           # as EmTract stage 2 and merge


def tag(slug: str, max_len: int) -> str:
    return f"{slug}-L{int(max_len)}"


def pad_buckets(max_len: int) -> Tuple[int, ...]:
    b = list(range(PAD_STEP, int(max_len) + 1, PAD_STEP))
    if not b or b[-1] != max_len:
        b.append(int(max_len))
    return tuple(b)


# ==================================================================================================
# 3. Paths (every one can be moved with an environment variable; the tests use this)
# ==================================================================================================
POLARITY_MODEL_DIR = Path(os.environ.get("THESIS_POLARITY_MODELS", RAW_DIR / "polarity_models"))
RUN_DIR = Path(os.environ.get("THESIS_POLARITY_RUN", OUT_DIR / "_polarity_run"))
REF_DIR = Path(__file__).resolve().parent / "polarity_ref"      # self-check texts and slow-tokenizer ids
# run_dir(tag)/modal_inputs/ and modal_outputs/: folder names of the packed inputs and raw scores;
# the code depends on no provider.


def model_dir(slug: str) -> Path:
    return POLARITY_MODEL_DIR / slug


def tokens_dir(platform: str, run_tag: str) -> Path:
    return out_path(platform, "polarity", run_tag, "tokens")


def scores_dir(platform: str, run_tag: str) -> Path:
    return out_path(platform, "polarity", run_tag, "scores")


def units_dir(platform: str, run_tag: str) -> Path:
    """Per scored unit (shard, shard_row): primary, window mean, coverage-weighted mean (`merge`)."""
    return out_path(platform, "polarity", run_tag, "unit_scores")


def polarity_dir(platform: str, run_tag: str) -> Path:
    return out_path(platform, "polarity", run_tag, "polarity")


def run_dir(run_tag: str) -> Path:
    return RUN_DIR / run_tag


def inputs_dir(platform: str, run_tag: str) -> Path:
    return run_dir(run_tag) / "modal_inputs" / platform


def raw_dir(platform: str, run_tag: str) -> Path:
    return Path(os.environ.get("THESIS_POLARITY_RAW", run_dir(run_tag) / "modal_outputs")) / platform


def keys_file(platform: str, run_tag: str) -> Path:
    return run_dir(run_tag) / "keys" / f"{platform}.parquet"


# ==================================================================================================
# 4. Model files (fetch-model): plain HTTPS from huggingface.co at the pinned commit
# ==================================================================================================
HF = "https://huggingface.co"


def _sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def read_sums(path: Path) -> Dict[str, str]:
    out = {}
    for ln in Path(path).read_text().splitlines():
        if ln.strip():
            sha, name = ln.split(None, 1)
            out[name.strip().lstrip("*")] = sha
    return out


def fetch_model(slug: str, dest: Optional[Path] = None, sums: Optional[Path] = None,
                attempts: int = 4) -> Dict[str, object]:
    """Download the pinned files of MODELS[slug] into dest (default model_dir(slug)), check the
    weights' sha256 against the HF tree (LFS oid) and, when `sums` is given (a SHA256SUMS written
    by an earlier fetch), every file against it. Writes dest/SHA256SUMS and dest/SOURCE.json.
    Files already present with the right sha256 are not downloaded again."""
    spec = MODELS[slug]
    dest = Path(dest or model_dir(slug))
    dest.mkdir(parents=True, exist_ok=True)
    want = read_sums(Path(sums)) if sums else {}
    tree_url = f"{HF}/api/models/{spec.repo}/tree/{spec.revision}"
    with urllib.request.urlopen(tree_url, timeout=60) as r:
        tree = {x["path"]: x for x in json.loads(r.read())}
    got: Dict[str, str] = {}
    for name in spec.files:
        if name not in tree:
            raise RuntimeError(f"{spec.repo}@{spec.revision[:8]} has no file {name}")
        f = dest / name
        lfs = (tree[name].get("lfs") or {}).get("oid")
        target = want.get(name) or lfs
        if not (f.exists() and target and _sha256(f) == target):
            url = f"{HF}/{spec.repo}/resolve/{spec.revision}/{name}"
            tmp = f.with_name(f.name + ".part")
            for k in range(attempts):
                try:
                    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
                        while True:
                            b = r.read(1 << 20)
                            if not b:
                                break
                            fh.write(b)
                    break
                except Exception:                       # noqa: BLE001  (retried, then raised)
                    if k == attempts - 1:
                        raise
                    time.sleep(2 ** k)
            os.replace(tmp, f)
        sha = _sha256(f)
        if int(tree[name].get("size", -1)) != f.stat().st_size:
            raise RuntimeError(f"{name}: size {f.stat().st_size} != HF tree {tree[name].get('size')}")
        if lfs and sha != lfs:
            raise RuntimeError(f"{name}: sha256 {sha[:12]} != HF LFS oid {lfs[:12]}")
        if name == spec.weights and not sha.startswith(spec.weights_sha256):
            raise RuntimeError(f"{name}: sha256 {sha[:12]} does not start with the pinned {spec.weights_sha256}")
        if want and want.get(name) != sha:
            raise RuntimeError(f"{name}: sha256 {sha[:12]} != the SHA256SUMS given")
        got[name] = sha
    (dest / "SHA256SUMS").write_text("".join(f"{got[n]}  {n}\n" for n in spec.files))
    src = {"repo": spec.repo, "revision": spec.revision, "files": {n: {"sha256": got[n], "bytes": (dest / n).stat().st_size}
                                                                    for n in spec.files},
           "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (dest / "SOURCE.json").write_text(json.dumps(src, indent=1))
    return src


def check_model_files(slug: str, mdir: Path) -> Dict[str, str]:
    """sha256 of every file of MODELS[slug] in mdir against mdir/SHA256SUMS (raises on a mismatch)."""
    spec = MODELS[slug]
    sums = read_sums(Path(mdir) / "SHA256SUMS")
    out = {}
    for n in spec.files:
        sha = _sha256(Path(mdir) / n)
        if sums.get(n) != sha:
            raise RuntimeError(f"{slug}: {n} sha256 {sha[:12]} != SHA256SUMS")
        out[n] = sha
    return out


# ==================================================================================================
# 5. Rows, versions and the text key
# ==================================================================================================
TOKENIZE_BATCH = 1024                    # texts per tokenizer call (does not change ids)
SOURCES = {
    # platform: id column, text columns read, partitions rule
    "reddit": {"id": "id", "text": ("text", "own_text", "quoted_text", "has_quote_lines", "is_dead")},
    "stocktwits": {"id": "message_id", "text": ("message_body",)},
    "twitter": {"id": "unit_id", "text": ("text", "parent_text")},
    "dlt": {"id": "row_id", "text": ("text",)},
    "ravenpack": {"id": "headline_md5", "text": ("headline",)},
}


def build_rows(platform: str, cols: Dict[str, Sequence], versions: Sequence[str] = DEFAULT_VERSIONS
               ) -> Tuple[List[int], List[str], List[Optional[str]], Dict[str, int]]:
    """(source_index, version, original text or None, counts), one entry per (row, version)."""
    versions = tuple(versions)
    bad = [v for v in versions if v not in VERSIONS]
    if bad or not versions:
        raise ValueError(f"unknown versions {bad or versions}")
    if platform in ("reddit", "stocktwits", "twitter"):
        from . import emtract_method as EM          # pure functions; no tokenizer is loaded
        idx, ver, txt, counts = EM.build_versions(platform, cols)
        keep = [i for i, v in enumerate(ver) if v in versions]
        idx, ver, txt = [idx[i] for i in keep], [ver[i] for i in keep], [txt[i] for i in keep]
    elif platform in ("dlt", "ravenpack"):
        if versions != ("own",):
            raise ValueError(f"{platform} has only the version 'own'")
        (name,) = SOURCES[platform]["text"]
        n = len(cols[name])
        idx, ver, txt = list(range(n)), ["own"] * n, list(cols[name])
        counts = {"rows_in": n, "rows_dead": 0, "rows_live": n, "rows_quoted": 0}
    else:
        raise ValueError(f"unknown platform {platform!r}")
    counts = dict(counts)
    counts["versions_out"] = len(ver)
    counts["by_version"] = {v: ver.count(v) for v in versions}
    return idx, ver, txt, counts


def text_key(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


# ==================================================================================================
# 6. Window plan for texts over max_len - 2 tokens
# ==================================================================================================
class WindowRule:
    def __init__(self, max_len: int, overlap: int = MIN_WINDOW_OVERLAP, max_move: int = MAX_MOVE):
        self.max_len = int(max_len)
        self.W = self.max_len - 2
        self.M = int(overlap)
        self.step = self.W - self.M
        self.max_move = int(max_move)
        if self.step <= 0:
            raise ValueError(f"max_len {max_len}: window {self.W} must exceed the overlap {self.M}")

    def n_windows(self, n: int) -> int:
        if n <= 0:
            return 0
        if n <= self.W:
            return 1
        return math.ceil((n - self.M) / self.step)

    def ideal_starts(self, n: int) -> List[int]:
        k = self.n_windows(n)
        if k <= 1:
            return [0] if k == 1 else []
        last = n - self.W
        return [(2 * j * last + (k - 1)) // (2 * (k - 1)) for j in range(k)]

    def starts(self, n: int, word_start: Optional[Sequence[bool]] = None) -> List[int]:
        starts = self.ideal_starts(n)
        k = len(starts)
        if k <= 2 or word_start is None:
            return starts
        if len(word_start) != n:
            raise ValueError(f"word_start has {len(word_start)} entries, expected {n}")
        last = n - self.W
        ideal = list(starts)
        for j in range(1, k - 1):
            s0 = ideal[j]
            best = None
            for d in range(0, self.max_move + 1):
                for cand in ((s0,) if d == 0 else (s0 - d, s0 + d)):     # earlier candidate first
                    if 0 < cand < last and word_start[cand]:
                        best = cand
                        break
                if best is not None:
                    break
            if best is None or best == s0:
                continue
            prev, nxt = starts[j - 1], ideal[j + 1]
            if prev < best < nxt and best - prev <= self.step and nxt - best <= self.step:
                starts[j] = best
        return starts

    def plan_error(self, n: int, starts: Sequence[int]) -> Optional[str]:
        """None if `starts` is a valid plan for n tokens, else the reason."""
        starts = list(starts)
        k = self.n_windows(n)
        if len(starts) != k:
            return f"n={n}: {len(starts)} windows, expected {k}"
        if k == 0:
            return None
        if starts[0] != 0:
            return f"n={n}: first start {starts[0]}"
        if k == 1:
            return None
        if starts[-1] != n - self.W:
            return f"n={n}: last start {starts[-1]} != {n - self.W}"
        ideal = self.ideal_starts(n)
        for j in range(1, k):
            if not (0 < starts[j] - starts[j - 1] <= self.step):
                return f"n={n}: overlap below {self.M} or no progress at window {j}"
            if abs(starts[j] - ideal[j]) > self.max_move:
                return f"n={n}: start {starts[j]} more than {self.max_move} from its ideal {ideal[j]}"
        return None

    def coverage_weights(self, n: int, starts: Sequence[int]):
        """w_j = sum over the tokens of window j of 1/c_t (c_t = windows covering token t), as EmTract's merge."""
        import numpy as np
        diff = np.zeros(n + 1, dtype=np.int64)
        for s in starts:
            diff[s] += 1
            diff[min(s + self.W, n)] -= 1
        cover = np.cumsum(diff[:-1])
        if not (cover >= 1).all():
            raise ValueError(f"window plan leaves tokens uncovered (n={n})")
        cum = np.concatenate([[0.0], np.cumsum(1.0 / cover)])
        s = np.asarray(starts, dtype=np.int64)
        return cum[np.minimum(s + self.W, n)] - cum[s]


# ==================================================================================================
# 7. Tokenizer and model inputs
# ==================================================================================================
class PolarityTokenizer:
    """The model's own fast tokenizer from local pinned files (no network)."""

    def __init__(self, mdir, max_len: int):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")      # one core per worker process
        import tokenizers
        import transformers
        transformers.logging.set_verbosity_error()      # the 'longer than max_length' warning is expected
        self.dir = str(mdir)
        self.max_len = int(max_len)
        self.tok = transformers.AutoTokenizer.from_pretrained(self.dir, use_fast=True, local_files_only=True)
        self.tok.model_max_length = self.max_len        # unset in CryptoBERT's config: set explicitly
        if not getattr(self.tok, "is_fast", False):
            raise RuntimeError(f"{self.dir}: not a fast tokenizer ({type(self.tok).__name__})")
        both = self.tok("", add_special_tokens=True)["input_ids"]
        none = self.tok("", add_special_tokens=False)["input_ids"]
        if len(both) != 2 or none:
            raise RuntimeError(f"special tokens of an empty text: {both} (expected exactly prefix + suffix)")
        self.prefix_id, self.suffix_id = int(both[0]), int(both[1])
        self.pad_id = int(self.tok.pad_token_id)
        self.vocab_size = len(self.tok)
        cfg = os.path.join(self.dir, "config.json")
        if os.path.exists(cfg):                         # the ids must index the model's embedding rows
            import json
            n_emb = int(json.load(open(cfg)).get("vocab_size", self.vocab_size))
            if self.vocab_size > n_emb:
                raise RuntimeError(f"tokenizer has {self.vocab_size} ids, the model's embedding {n_emb}")
        self.id_dtype = "uint16" if self.vocab_size <= 65535 else "uint32"
        self.versions = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                         "class": type(self.tok).__name__}

    def specials(self) -> Dict[str, int]:
        return {"prefix": self.prefix_id, "suffix": self.suffix_id, "pad": self.pad_id, "vocab_size": self.vocab_size}

    def encode(self, texts: List[str], want_word_starts_above: Optional[int] = None
               ) -> Tuple[List[List[int]], List[Optional[List[bool]]]]:
        """Uncut content ids per text; word-start flags only for texts longer than the given count."""
        ids_out: List[List[int]] = []
        ws_out: List[Optional[List[bool]]] = []
        for a in range(0, len(texts), TOKENIZE_BATCH):
            enc = self.tok(texts[a:a + TOKENIZE_BATCH], add_special_tokens=False, truncation=False, padding=False)
            for i, ids in enumerate(enc["input_ids"]):
                ids_out.append(list(ids))
                if want_word_starts_above is not None and len(ids) > want_word_starts_above:
                    w = enc.word_ids(i)
                    ws_out.append([j == 0 or w[j] is None or w[j] != w[j - 1] for j in range(len(ids))])
                else:
                    ws_out.append(None)
        return ids_out, ws_out

    def truncated(self, texts: List[str]) -> List[List[int]]:
        """The tokenizer's own truncation with specials (what a HF pipeline gives the model)."""
        return [list(x) for x in self.tok(texts, truncation=True, max_length=self.max_len)["input_ids"]]

    def primary(self, ids: Sequence[int]) -> List[int]:
        return [self.prefix_id] + list(ids)[: self.max_len - 2] + [self.suffix_id]


def model_input(ids: Sequence[int], start: int, W: int, prefix_id: int, suffix_id: int) -> List[int]:
    return [prefix_id] + list(ids)[start:start + W] + [suffix_id]


# ==================================================================================================
# 8. Stage 1 for a batch of rows (used by `tokenize`)
# ==================================================================================================
def stage1_rows(platform: str, cols: Dict[str, Sequence], tok: PolarityTokenizer, rule: WindowRule,
                versions: Sequence[str] = DEFAULT_VERSIONS, with_key: bool = False,
                with_windows: bool = True) -> Tuple[Dict[str, list], Dict[str, object]]:
    """Columns per (row, version): source_index, version, is_missing_text, is_empty_text, n_tokens,
    input_ids (uncut), window_starts ([0] for 1..W tokens; the plan for longer texts, or [0] alone
    without windows), [text_key]. Identical texts are tokenized once (pure function: no output change)."""
    idx, ver, txt, counts = build_rows(platform, cols, versions)
    uniq = sorted({t for t in txt if t is not None})
    ids_l, ws_l = tok.encode(uniq, want_word_starts_above=rule.W if with_windows else None)
    by_text = dict(zip(uniq, zip(ids_l, ws_l)))
    miss, empty, n_tok, input_ids, starts, keys = [], [], [], [], [], []
    for t in txt:
        if t is None:
            ids, ws = [], None
        else:
            ids, ws = by_text[t]
        n = len(ids)
        miss.append(t is None)
        empty.append(t is not None and n == 0)
        n_tok.append(n)
        input_ids.append(ids)
        if n == 0:
            starts.append([])
        elif n <= rule.W or not with_windows:
            starts.append([0])
        else:
            starts.append(rule.starts(n, ws))
        if with_key:
            keys.append(None if t is None else text_key(t))
    out = {"source_index": idx, "version": ver, "is_missing_text": miss, "is_empty_text": empty,
           "n_tokens": n_tok, "input_ids": input_ids, "window_starts": starts}
    if with_key:
        out["text_key"] = keys
    counts.update({
        "missing_text": sum(miss), "empty_text": sum(empty), "texts_tokenized_distinct": len(uniq),
        "over_W_tokens": sum(1 for n in n_tok if n > rule.W), "model_inputs_primary": sum(1 for n in n_tok if n),
        "model_inputs_windows": sum(len(s) for s in starts), "max_n_tokens": max(n_tok, default=0),
        "sum_n_tokens": sum(n_tok), "max_token_id": max((max(i) for i in input_ids if i), default=-1),
    })
    return out, counts
