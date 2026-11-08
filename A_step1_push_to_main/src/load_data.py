"""Task 1 - load GoEmotions without committing any raw data.

We use the *filtered* ("simplified") release of GoEmotions published by Google
Research: 54,263 Reddit comments (out of the 58,009 raw ones) that survived the
rater-agreement filter, with the authors' predefined train/validation/test split.

    python -m src.load_data          # download (if needed) and print a summary

Data source:
    https://github.com/google-research/google-research/tree/master/goemotions
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/goemotions/data/"
)
SPLIT_FILES = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}
EMOTIONS_FILE = "emotions.txt"
EXPECTED_ROWS = {"train": 43410, "validation": 5426, "test": 5427}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"


def download_raw(raw_dir: Path = DEFAULT_RAW_DIR, force: bool = False) -> Path:
    """Download the split files + emotion list into ``raw_dir`` (skips existing files)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in [*SPLIT_FILES.values(), EMOTIONS_FILE]:
        target = raw_dir / name
        if target.exists() and not force:
            continue
        print(f"Downloading {name} ...")
        req = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            target.write_bytes(resp.read())
    return raw_dir


def emotion_names(raw_dir: Path = DEFAULT_RAW_DIR) -> list[str]:
    """The 28 label names (27 emotions + neutral); line i is label id i."""
    path = Path(raw_dir) / EMOTIONS_FILE
    if not path.exists():
        download_raw(raw_dir)
    return path.read_text(encoding="utf-8").split()


def _read_tsv(path: Path, split: str) -> pd.DataFrame:
    # keep_default_na=False: a comment that is literally "NA" / "null" must stay text.
    #
    # Quoting: the GoEmotions TSVs use CSV-style quoting. Any comment that contains a
    # double quote is wrapped in quotes and its inner quotes are doubled, e.g.
    #     "Three words. ""Dude. Stop."" "        ->   Three words. "Dude. Stop."
    # pandas' default quoting (QUOTE_MINIMAL) undoes this. Do NOT pass quoting=3
    # (QUOTE_NONE): it keeps the escaping as literal text in ~2.5% of the comments.
    df = pd.read_csv(
        path, sep="\t", header=None, names=["text", "label_ids", "comment_id"],
        dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8",
    )
    df.insert(1, "split", split)
    return df


def load_goemotions(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Return one DataFrame with columns:
    comment_id, split, text, label_ids ("8,20"), label_names ("desire,optimism").
    """
    raw_dir = Path(raw_dir)
    if not all((raw_dir / f).exists() for f in SPLIT_FILES.values()):
        download_raw(raw_dir)
    names = emotion_names(raw_dir)
    parts = [_read_tsv(raw_dir / fname, split) for split, fname in SPLIT_FILES.items()]
    df = pd.concat(parts, ignore_index=True)[["comment_id", "split", "text", "label_ids"]]
    df["label_names"] = df["label_ids"].map(
        lambda s: ",".join(names[int(i)] for i in s.split(","))
    )
    counts = df["split"].value_counts().to_dict()
    if counts != EXPECTED_ROWS:  # guards against a truncated/changed download
        raise ValueError(f"Unexpected split sizes {counts}; expected {EXPECTED_ROWS}")
    # Guards against the CSV-quoting regression (see _read_tsv): parsed correctly, only
    # a handful of comments contain a doubled quote; parsed with QUOTE_NONE, ~1,400 do.
    if df["text"].str.contains('""', regex=False).mean() > 0.005:
        raise ValueError("Comments still contain CSV-escaped quotes; check _read_tsv quoting.")
    return df


def multi_hot(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Multi-hot label matrix (one 0/1 column per emotion) aligned with ``df``."""
    names = names or emotion_names()
    out = pd.DataFrame(0, index=df.index, columns=names, dtype="int8")
    for idx, ids in zip(df.index, df["label_ids"]):
        for i in ids.split(","):
            out.at[idx, names[int(i)]] = 1
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args()
    data = load_goemotions(args.raw_dir)
    print(data.groupby("split").size())
    print(data.head())
