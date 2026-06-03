"""Summarize the local Perec corpus and enrich it with BnF catalogue metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ebooklib import ITEM_DOCUMENT, epub

SUPPORTED_SUFFIXES = {".epub", ".pdf"}
SRU_URL = "https://catalogue.bnf.fr/api/SRU"
USER_AGENT = "literary-place-index/0.0.1 (BnF corpus metadata enrichment)"
MXC_NS = "{info:lc/xmlns/marcxchange-v2}"

CORPUS_COLUMNS = [
    "work",
    "source_file_hash",
    "source_type",
    "file_size_bytes",
    "file_size_mb",
    "page_count",
    "section_count",
    "character_count",
    "word_count",
]

BNF_COLUMNS = [
    "bnf_earliest_publication_year",
    "bnf_matched_records",
    "bnf_best_ark",
    "bnf_best_title",
    "bnf_best_publisher",
    "bnf_best_publication_statement",
    "bnf_best_isbn",
    "bnf_all_matched_arks",
    "bnf_match_score",
]

COLUMNS = CORPUS_COLUMNS + BNF_COLUMNS


@dataclass(frozen=True)
class SummaryRow:
    values: dict[str, Any]


@dataclass(frozen=True)
class BnfRecord:
    ark: str
    title: str
    responsibility: str
    publisher: str
    publication_statement: str
    isbn: str
    material_type: str
    years: tuple[int, ...]
    score: int = 0


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_work_name(path: Path) -> str:
    """Infer a display title from a filename without leaking provenance suffixes."""
    stem = path.stem.split(" -- ", 1)[0]
    stem = re.split(r"(?i)(?:\s*[-_]\s*)?(?:libgen\.li|z-lib\.org|anna['’]s?)", stem, maxsplit=1)[0]
    stem = re.sub(r"(?i)\s*\([^)]*(?:z-lib|libgen)[^)]*\)\s*", " ", stem)
    stem = re.sub(r"[_\s-]+(?:19|20)\d{2}.*$", "", stem)
    stem = stem.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", stem).strip()


def has_suspicious_encoding(text: str) -> bool:
    """Detect obviously corrupted metadata strings."""
    if not text:
        return False
    suspicious = 0
    for char in text:
        category = unicodedata.category(char)
        name = unicodedata.name(char, "")
        if category.startswith("C") or any(token in name for token in ("CJK", "HANGUL", "YI SYLLABLE")):
            suspicious += 1
    return suspicious / max(len(text), 1) > 0.05


def clean_metadata_title(title: str) -> str:
    title = clean_text(title)
    if title.lower() in {"unknown", "untitled", "sans titre"}:
        return ""
    return "" if has_suspicious_encoding(title) else title


def clean_work_title(title: str) -> str:
    """Remove author strings that sometimes appear in extracted titles."""
    title = clean_text(title)
    title = re.sub(r"^Georges?\s+P[ée]rec\s*[-–—:]?\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^P[ée]rec\s*,\s*Georges?\s*[-–—:]?\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+de\s+Georges?\s+P[ée]rec(?:\.\d+)?\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*[-–—:]?\s*Georges?\s+P[ée]rec(?:\.\d+)?\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"(?i)^Th[ée]âtre\s*,?\s*tome\s*1\s*[:;].*$", "Théâtre", title)
    return re.sub(r"\s+", " ", title).strip(" -–—:.")


def word_count(text: str) -> int:
    # Includes apostrophes/hyphens inside words, useful for French literary text.
    return len(re.findall(r"\b[\wÀ-ÖØ-öø-ÿ]+(?:[’'-][\wÀ-ÖØ-öø-ÿ]+)*\b", text, flags=re.UNICODE))


def filename_hash(filename: str, key: str | None = None) -> str:
    """Return a stable, non-readable identifier for a source filename."""
    data = filename.encode("utf-8")
    if key:
        digest = hmac.new(key.encode("utf-8"), data, hashlib.sha256).hexdigest()
    else:
        digest = hashlib.sha256(data).hexdigest()
    return digest[:16]


def first_metadata_value(values: list[tuple[str, dict[str, str]]]) -> str:
    return values[0][0].strip() if values and values[0][0].strip() else ""


def all_metadata_values(values: list[tuple[str, dict[str, str]]]) -> str:
    return "; ".join(value.strip() for value, _attrs in values if value and value.strip())


def preferred_bnf_isbn(work: str) -> str:
    """Manual BnF disambiguation when title search is ambiguous."""
    if normalize_title(work) == "theatre":
        return "9782213668925"
    return ""


def summarize_pdf(path: Path, hash_key: str | None = None) -> SummaryRow:
    with fitz.open(path) as doc:
        metadata = doc.metadata or {}
        texts = [clean_text(page.get_text("text")) for page in doc]
        full_text = "\n\n".join(text for text in texts if text)
        metadata_title = clean_metadata_title(metadata.get("title") or "")
        work = clean_work_title(metadata_title or infer_work_name(path))
        page_count = doc.page_count

    return SummaryRow(
        {
            "work": work,
            "source_file": path.name,
            "source_file_hash": filename_hash(path.name, hash_key),
            "source_type": "pdf",
            "file_size_bytes": path.stat().st_size,
            "file_size_mb": f"{path.stat().st_size / 1_000_000:.2f}",
            "page_count": page_count,
            "section_count": page_count,
            "character_count": len(full_text),
            "word_count": word_count(full_text),
            "preferred_publisher": "",
            "preferred_bnf_isbn": preferred_bnf_isbn(work),
        }
    )


def epub_texts(book: epub.EpubBook) -> list[str]:
    items_by_id = {item.get_id(): item for item in book.get_items()}
    document_items = [item for item in book.get_items() if item.get_type() == ITEM_DOCUMENT]

    ordered_items = []
    for item_id, _linear in book.spine:
        item = items_by_id.get(item_id)
        if item is not None and item.get_type() == ITEM_DOCUMENT:
            ordered_items.append(item)
    if not ordered_items:
        ordered_items = document_items

    texts: list[str] = []
    for item in ordered_items:
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for bad in soup(["script", "style", "nav"]):
            bad.decompose()
        text = clean_text(soup.get_text("\n"))
        if text:
            texts.append(text)
    return texts


def summarize_epub(path: Path, hash_key: str | None = None) -> SummaryRow:
    book = epub.read_epub(str(path))
    texts = epub_texts(book)
    full_text = "\n\n".join(texts)

    title = clean_metadata_title(first_metadata_value(book.get_metadata("DC", "title")))
    publisher = all_metadata_values(book.get_metadata("DC", "publisher"))

    return SummaryRow(
        {
            "work": clean_work_title(title or infer_work_name(path)),
            "source_file": path.name,
            "source_file_hash": filename_hash(path.name, hash_key),
            "source_type": "epub",
            "file_size_bytes": path.stat().st_size,
            "file_size_mb": f"{path.stat().st_size / 1_000_000:.2f}",
            "page_count": "",
            "section_count": len(texts),
            "character_count": len(full_text),
            "word_count": word_count(full_text),
            "preferred_publisher": publisher,
            "preferred_bnf_isbn": preferred_bnf_isbn(clean_work_title(title or infer_work_name(path))),
        }
    )


def summarize_file(path: Path, hash_key: str | None = None) -> SummaryRow:
    if path.suffix.lower() == ".pdf":
        return summarize_pdf(path, hash_key)
    if path.suffix.lower() == ".epub":
        return summarize_epub(path, hash_key)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.casefold()
    value = re.sub(r"^georges?\s+perec\s*[-–—:]?\s*", "", value)
    value = re.sub(r"\s+de\s+georges?\s+perec\s*$", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_publisher(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def search_query(title: str) -> str:
    # Keep only the main title for catalogue search; subtitles often vary between editions.
    main_title = re.split(r"\s*[:;]\s*", title, maxsplit=1)[0]
    main_title = main_title.replace('"', " ")
    return f'bib.author all "Georges Perec" and bib.title all "{main_title}"'


def sru_search(title: str, maximum_records: int = 50, isbn: str = "") -> ET.Element:
    query = f'bib.isbn all "{isbn}"' if isbn else search_query(title)
    params = urllib.parse.urlencode(
        {
            "version": "1.2",
            "operation": "searchRetrieve",
            "query": query,
            "recordSchema": "unimarcxchange",
            "maximumRecords": str(maximum_records),
        }
    )
    request = urllib.request.Request(f"{SRU_URL}?{params}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return ET.fromstring(response.read())


def subfields(datafield: ET.Element, code: str | None = None) -> list[str]:
    values = []
    for subfield in datafield.findall(f"{MXC_NS}subfield"):
        if code is None or subfield.attrib.get("code") == code:
            if subfield.text and subfield.text.strip():
                values.append(subfield.text.strip())
    return values


def datafields(record: ET.Element, tag: str) -> list[ET.Element]:
    return [field for field in record.findall(f"{MXC_NS}datafield") if field.attrib.get("tag") == tag]


def extract_years(*values: str) -> tuple[int, ...]:
    years: set[int] = set()
    for value in values:
        for year_text in re.findall(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)", value):
            year = int(year_text)
            if 1800 <= year <= 2035:
                years.add(year)
    return tuple(sorted(years))


def parse_bnf_record(record: ET.Element) -> BnfRecord:
    ark = record.attrib.get("id", "")
    if not ark:
        for controlfield in record.findall(f"{MXC_NS}controlfield"):
            if controlfield.attrib.get("tag") == "003" and controlfield.text:
                ark = controlfield.text.replace("http://catalogue.bnf.fr/", "")

    title_parts: list[str] = []
    responsibilities: list[str] = []
    material_types: list[str] = []
    for field in datafields(record, "200"):
        title_parts.extend(subfields(field, "a"))
        title_parts.extend(subfields(field, "e"))
        material_types.extend(subfields(field, "b"))
        responsibilities.extend(subfields(field, "f"))
        responsibilities.extend(subfields(field, "g"))

    pub_statements: list[str] = []
    publishers: list[str] = []
    for tag in ("210", "214"):
        for field in datafields(record, tag):
            parts = subfields(field)
            pub_statements.append("; ".join(parts))
            publishers.extend(subfields(field, "c"))

    isbns: list[str] = []
    for field in datafields(record, "010"):
        isbns.extend(subfields(field, "a"))

    control_100 = []
    for field in datafields(record, "100"):
        control_100.extend(subfields(field, "a"))

    return BnfRecord(
        ark=ark,
        title=" : ".join(title_parts),
        responsibility="; ".join(responsibilities),
        publisher="; ".join(dict.fromkeys(publishers)),
        publication_statement=" | ".join(pub_statements),
        isbn="; ".join(dict.fromkeys(isbns)),
        material_type="; ".join(dict.fromkeys(material_types)),
        years=extract_years(" ".join(pub_statements), " ".join(control_100)),
    )


def is_print_record(record: BnfRecord) -> bool:
    material = record.material_type.casefold()
    if "texte imprim" in material:
        return True
    if any(term in material for term in ("images anim", "enregistrement sonore", "musique", "vidéo", "video")):
        return False
    return bool(record.isbn) or not material


def title_match_score(work_title: str, record_title: str) -> int:
    wanted = normalize_title(work_title)
    candidate = normalize_title(record_title)
    if not wanted or not candidate:
        return 0
    ratio = SequenceMatcher(None, wanted, candidate).ratio()
    score = round(ratio * 100)
    if wanted == candidate:
        score = 100
    elif candidate in wanted and len(candidate) >= 6:
        score = max(score, 90)
    elif wanted in candidate and len(wanted) >= 6:
        score = max(score, 90)
    return score


def score_bnf_record(work_title: str, record: BnfRecord, preferred_publisher: str = "") -> int:
    score = title_match_score(work_title, record.title)
    if score == 0:
        return 0
    if "perec" in normalize_title(record.responsibility):
        score += 5
    if is_print_record(record):
        score += 5
    else:
        score -= 20
    if preferred_publisher:
        wanted_publisher = normalize_publisher(preferred_publisher)
        candidate_publisher = normalize_publisher(record.publisher)
        if wanted_publisher and wanted_publisher in candidate_publisher:
            score += 10
    return max(0, min(score, 100))


def matched_bnf_records(
    title: str, minimum_score: int, preferred_publisher: str = "", preferred_isbn: str = ""
) -> list[BnfRecord]:
    root = sru_search(title, isbn=preferred_isbn)
    records: list[BnfRecord] = []
    for element in root.findall(f".//{MXC_NS}record"):
        record = parse_bnf_record(element)
        score = score_bnf_record(title, record, preferred_publisher)
        if score >= minimum_score or preferred_isbn:
            records.append(BnfRecord(**{**record.__dict__, "score": score}))

    if preferred_isbn and records:
        return sorted(records, key=lambda r: (-r.score, min(r.years) if r.years else 9999, r.title))

    if preferred_publisher:
        wanted_publisher = normalize_publisher(preferred_publisher)
        publisher_records = [r for r in records if wanted_publisher and wanted_publisher in normalize_publisher(r.publisher)]
        if publisher_records:
            records = publisher_records

    print_records = [record for record in records if is_print_record(record)]
    if print_records:
        records = print_records

    return sorted(
        records,
        key=lambda r: (-title_match_score(title, r.title), -r.score, min(r.years) if r.years else 9999, r.title),
    )


def enrich_with_bnf(row: SummaryRow, minimum_score: int) -> SummaryRow:
    values = dict(row.values)
    for column in BNF_COLUMNS:
        values[column] = ""

    try:
        records = matched_bnf_records(
            str(values.get("work", "")),
            minimum_score,
            str(values.get("preferred_publisher", "")),
            str(values.get("preferred_bnf_isbn", "")),
        )
    except Exception as exc:
        values["bnf_best_publication_statement"] = f"BnF lookup failed: {exc}"
        return SummaryRow(values)

    if not records:
        return SummaryRow(values)

    all_years = [year for record in records for year in record.years]
    best = records[0]
    values.update(
        {
            "bnf_earliest_publication_year": str(min(all_years)) if all_years else "",
            "bnf_matched_records": str(len(records)),
            "bnf_best_ark": best.ark,
            "bnf_best_title": best.title,
            "bnf_best_publisher": best.publisher,
            "bnf_best_publication_statement": best.publication_statement,
            "bnf_best_isbn": best.isbn,
            "bnf_all_matched_arks": "; ".join(record.ark for record in records if record.ark),
            "bnf_match_score": str(max(record.score for record in records)),
        }
    )
    return SummaryRow(values)


def write_csv(path: Path, rows: list[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.values)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Write a CSV summary of the corpus in data/ with BnF metadata.")
    parser.add_argument("--input", type=Path, default=Path("data"), help="Directory containing PDF/EPUB files")
    parser.add_argument("--output", type=Path, default=Path("docs/corpus_summary.csv"), help="Output CSV path")
    parser.add_argument(
        "--hash-key",
        default=os.getenv("CORPUS_FILENAME_HASH_KEY"),
        help="Optional private key/salt for source_file_hash; defaults to CORPUS_FILENAME_HASH_KEY from .env",
    )
    parser.add_argument("--minimum-bnf-score", type=int, default=82, help="Minimum fuzzy title match score to accept")
    parser.add_argument("--bnf-sleep", type=float, default=0.2, help="Delay between BnF SRU calls, in seconds")
    parser.add_argument("--skip-bnf", action="store_true", help="Only compute local corpus metrics; do not query BnF")
    args = parser.parse_args()

    files = sorted(path for path in args.input.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
    rows: list[SummaryRow] = []
    failures: list[str] = []
    for path in files:
        try:
            row = summarize_file(path, args.hash_key)
            if not args.skip_bnf:
                row = enrich_with_bnf(row, args.minimum_bnf_score)
                time.sleep(args.bnf_sleep)
            rows.append(row)
        except Exception as exc:  # Keep one bad source from stopping the inventory.
            failures.append(f"{path.name}: {exc}")

    rows.sort(key=lambda row: str(row.values.get("work", "")).casefold())
    write_csv(args.output, rows)

    bnf_matches = sum(1 for row in rows if row.values.get("bnf_best_ark"))
    print(f"Wrote {len(rows)} rows to {args.output}")
    if not args.skip_bnf:
        print(f"BnF matches: {bnf_matches}")
    if not args.hash_key:
        print("Note: source_file_hash used unsalted SHA-256. Set CORPUS_FILENAME_HASH_KEY in .env for private, keyed hashes.")
    if failures:
        print(f"Skipped {len(failures)} files:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
