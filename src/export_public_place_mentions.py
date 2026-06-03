"""Export place mentions without internal corpus provenance fields."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dotenv import load_dotenv

from src.corpus_summary import filename_hash

PUBLIC_COLUMNS = [
    "work",
    "source_type",
    "source_ref",
    "place_text",
    "normalized_place",
    "place_type",
    "city",
    "region",
    "country",
    "confidence",
    "context",
]


def load_public_titles(corpus_summary_path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    with corpus_summary_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            source_hash = row.get("source_file_hash", "").strip()
            if not source_hash:
                continue
            title = row.get("bnf_best_title", "").strip() or row.get("work", "").strip()
            titles[source_hash] = title
    return titles


def export_public_mentions(
    input_path: Path,
    output_path: Path,
    corpus_summary_path: Path,
    hash_key: str | None,
) -> None:
    titles_by_hash = load_public_titles(corpus_summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as in_f, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=PUBLIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for row in reader:
            source_file = row.get("source_file", "")
            source_hash = filename_hash(source_file, hash_key) if source_file else ""
            public_row = dict(row)
            public_row["work"] = titles_by_hash.get(source_hash) or row.get("work", "")
            writer.writerow(public_row)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Export public place mentions without internal source filenames/chunk IDs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/20260602/place_mentions.csv"),
        help="Internal place_mentions.csv to export",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/20260602_place_mentions.csv"),
        help="Public CSV output path",
    )
    parser.add_argument(
        "--corpus-summary",
        type=Path,
        default=Path("docs/corpus_summary.csv"),
        help="Corpus summary CSV containing source_file_hash and BnF titles",
    )
    parser.add_argument(
        "--hash-key",
        default=None,
        help="Optional private key/salt for filename hashes. Defaults to CORPUS_FILENAME_HASH_KEY from .env via corpus_summary.",
    )
    args = parser.parse_args()

    # filename_hash receives None for unsalted hashes; corpus_summary.load_dotenv has loaded the same env key.
    import os

    hash_key = args.hash_key if args.hash_key is not None else os.getenv("CORPUS_FILENAME_HASH_KEY")
    export_public_mentions(args.input, args.output, args.corpus_summary, hash_key)
    print(f"Wrote public place mentions to {args.output}")


if __name__ == "__main__":
    main()
