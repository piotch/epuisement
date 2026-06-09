"""Build a literary spatial mention index from local EPUB and PDF files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF
import llm
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ebooklib import ITEM_DOCUMENT, epub
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

DEFAULT_MODEL = "gpt-4o-mini"
CHARS_PER_TOKEN_ESTIMATE = 4
PROMPT_OVERHEAD_TOKENS_ESTIMATE = 700
OUTPUT_TOKENS_PER_CHUNK_ESTIMATE = 500
SECONDS_PER_CHUNK_ESTIMATE_RANGE = (2, 10)
MODEL_PRICE_PER_1M_TOKENS_USD = {
    # Check current provider pricing before relying on this for budgeting.
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}
MENTION_KINDS = {"named_place", "generic_spatial_entity"}
SPATIAL_TYPES = {
    "city",
    "street",
    "square",
    "district",
    "country",
    "region",
    "water",
    "natural_feature",
    "building",
    "institution",
    "room",
    "domestic_space",
    "threshold",
    "circulation",
    "public_space",
    "commercial_space",
    "work_space",
    "transport_space",
    "vehicle",
    "micro_place",
    "other",
}
USAGES = {"literal", "metaphorical", "uncertain"}
MICRO_PLACE_WHITELIST = {"lit", "table", "bureau", "chaise", "fauteuil", "bibliotheque"}
INCIDENTAL_OBJECT_BLACKLIST = {
    "objet",
    "livre",
    "cahier",
    "classeur",
    "ticket",
    "sacoche",
    "metal",
    "tocsin",
    "pylone",
    "tableau",
    "affiche",
    "poster",
    "machine",
    "verseuse",
    "miroir",
    "plateau",
    "cendrier",
    "bouteille",
    "boite",
    "lettre",
    "lampadaire",
}
GENERIC_OTHER_WHITELIST = {
    "mur",
    "fenetre",
    "cour",
    "parties communes",
    "sol",
    "plafond",
    "toit",
    "loggia",
    "cave",
    "grenier",
    "couloir",
    "palier",
    "escalier",
    "maison",
    "passage",
    "trottoir",
    "ville",
    "campagne",
    "monde",
    "espace",
    "banlieue",
    "zone libre",
}
MENTION_COLUMNS = [
    "work",
    "source_file",
    "source_type",
    "source_ref",
    "chunk_id",
    "mention_text",
    "normalized_text",
    "mention_kind",
    "spatial_type",
    "usage",
    "city",
    "region",
    "country",
    "confidence",
    "context",
]
NAMED_PLACE_INDEX_COLUMNS = [
    "normalized_text",
    "spatial_type",
    "city",
    "region",
    "country",
    "works",
    "mention_count",
    "sources",
    "example_context",
]
SPATIAL_TYPE_INDEX_COLUMNS = [
    "work",
    "mention_kind",
    "spatial_type",
    "normalized_text",
    "mention_count",
    "example_context",
]
FILTERED_MENTION_COLUMNS = MENTION_COLUMNS + ["filter_reason"]
LOG_COLUMNS = ["source_file", "chunk_id", "status", "error", "characters", "mentions_found"]


@dataclass(frozen=True)
class Section:
    work: str
    source_file: str
    source_type: str
    source_ref: str
    text: str


@dataclass(frozen=True)
class Chunk:
    work: str
    source_file: str
    source_type: str
    source_ref: str
    chunk_id: str
    text: str


def infer_work_name(path: Path) -> str:
    """Infer a readable work name from a filename."""
    return re.sub(r"\s+", " ", path.stem.replace("_", " ").replace("-", " ")).strip()


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries where possible."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_id_part(value: str) -> str:
    """Sanitize a source reference for deterministic chunk IDs."""
    value = value.strip().lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^a-z0-9_.-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    return value or "ref"


def extract_pdf(path: Path) -> list[Section]:
    """Extract one section per PDF page."""
    work = infer_work_name(path)
    sections: list[Section] = []
    with fitz.open(path) as doc:
        metadata_title = (doc.metadata or {}).get("title") or ""
        if metadata_title.strip():
            work = metadata_title.strip()
        for i, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            sections.append(Section(work, path.name, "pdf", str(i), text))
    return sections


def epub_item_title(html: bytes) -> str | None:
    """Find a likely title in an EPUB document item."""
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in ("h1", "h2", "h3", "title"):
        tag = soup.find(tag_name)
        if tag:
            title = clean_text(tag.get_text(" "))
            if 2 <= len(title) <= 160:
                return title
    return None


def extract_epub(path: Path) -> list[Section]:
    """Extract EPUB document sections in spine order when available."""
    book = epub.read_epub(str(path))
    title_meta = book.get_metadata("DC", "title")
    work = title_meta[0][0].strip() if title_meta and title_meta[0][0].strip() else infer_work_name(path)
    items_by_id = {item.get_id(): item for item in book.get_items()}
    document_items = [item for item in book.get_items() if item.get_type() == ITEM_DOCUMENT]
    sections: list[Section] = []

    ordered_items = []
    for item_id, _linear in book.spine:
        item = items_by_id.get(item_id)
        if item is not None and item.get_type() == ITEM_DOCUMENT:
            ordered_items.append(item)
    if not ordered_items:
        ordered_items = document_items

    used_refs: set[str] = set()
    for index, item in enumerate(ordered_items, start=1):
        html = item.get_content()
        soup = BeautifulSoup(html, "html.parser")
        for bad in soup(["script", "style", "nav"]):
            bad.decompose()
        text = clean_text(soup.get_text("\n"))
        title = epub_item_title(html)
        source_ref = title or item.get_name() or f"spine-{index}"
        if source_ref in used_refs:
            source_ref = f"{source_ref} ({item.get_name() or f'spine-{index}'})"
        used_refs.add(source_ref)
        sections.append(Section(work, path.name, "epub", source_ref, text))
    return sections


def extract_sections(path: Path) -> list[Section]:
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path)
    if path.suffix.lower() == ".epub":
        return extract_epub(path)
    return []


def looks_like_front_matter_or_toc(section: Section) -> bool:
    """Conservatively skip paratext likely to produce noisy spatial mentions."""
    ref = section.source_ref.lower()
    text = section.text.strip()
    compact = re.sub(r"\s+", " ", text.lower())
    squashed = re.sub(r"\s+", "", compact)

    ref_patterns = [
        "table des matières",
        "table des matieres",
        "copyright",
        "isbn",
        "la librairie du xxi",
        "ident1-",
    ]
    if section.source_type == "epub" and any(pattern in ref for pattern in ref_patterns):
        return True

    if not compact:
        return True

    if "tabledesmatières" in squashed or "tabledesmatieres" in squashed:
        return True

    front_matter_markers = [
        "isbn",
        "©",
        "copyright",
        "dépôt légal",
        "depot legal",
        "achevé d'imprimer",
        "acheve d'imprimer",
        "éditions du seuil",
        "editions du seuil",
        "christian bourgois éditeur",
        "christian bourgois editeur",
        "www.",
        "ce document numérique",
    ]
    marker_count = sum(1 for marker in front_matter_markers if marker in compact)
    if marker_count >= 2:
        return True

    # EPUB collection/catalogue pages often consist of long lists of authors/titles.
    # Skip them when the section reference already looks paratextual.
    if section.source_type == "epub" and "collection" in compact[:500] and len(compact) > 2000:
        return True

    return False


def split_at_boundary(text: str, target: int) -> int:
    """Choose a paragraph/sentence boundary at or before target when practical."""
    if len(text) <= target:
        return len(text)
    window_start = max(0, target - 1500)
    window = text[window_start:target]
    for pattern in (r"\n\s*\n", r"(?<=[.!?])\s+"):
        matches = list(re.finditer(pattern, window))
        if matches:
            return window_start + matches[-1].end()
    return target


def chunk_section(section: Section, max_chars: int, overlap: int) -> list[Chunk]:
    """Chunk a section into deterministic overlapping chunks."""
    text = section.text
    if not text:
        return []
    chunks: list[Chunk] = []
    start = 0
    number = 1
    safe_ref = sanitize_id_part(section.source_ref)
    stem = sanitize_id_part(Path(section.source_file).stem)
    while start < len(text):
        end = split_at_boundary(text[start:], max_chars) + start
        if end <= start:
            end = min(len(text), start + max_chars)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = f"{stem}:{safe_ref}:{number}"
            chunks.append(
                Chunk(section.work, section.source_file, section.source_type, section.source_ref, chunk_id, chunk_text)
            )
        if end >= len(text):
            break
        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start
        number += 1
    return chunks


def make_prompt(chunk: Chunk) -> str:
    return f"""Extract spatial mentions from the French literary text below.

Return only valid JSON matching this schema:
{{
  "mentions": [
    {{
      "mention_text": "string",
      "normalized_text": "string",
      "mention_kind": "named_place | generic_spatial_entity",
      "spatial_type": "city | street | square | district | country | region | water | natural_feature | building | institution | room | domestic_space | threshold | circulation | public_space | commercial_space | work_space | transport_space | vehicle | micro_place | other",
      "usage": "literal | metaphorical | uncertain",
      "city": "string or null",
      "region": "string or null",
      "country": "string or null",
      "confidence": number,
      "context": "string"
    }}
  ]
}}

Definitions:

1. named_place
Extract explicit named places: cities, countries, regions, streets, squares, districts, landmarks, buildings, institutions used as places, rivers, islands, seas, mountains, stations, parks, etc.

Named places may be real-world, fictional, uncertain, or constructed within the work. This is important for Perec: extract `rue Simon-Crubellier` and similar constructed named places. Do not omit them because they may not be geocodable.

Examples: Paris, rue Vilin, place Saint-Sulpice, avenue Junot, Mabillon, Ellis Island, Londres, Tunisie, la Seine, Montparnasse, rue Simon-Crubellier.

2. generic_spatial_entity
Also extract generic but spatially meaningful entities, especially those relevant to Georges Perec's attention to ordinary and infra-ordinary space.

Examples: chambre, lit, table, appartement, immeuble, escalier, palier, cour, rue, quartier, café, cuisine, bureau, cave, grenier, fenêtre, porte, mur, maison, passage, trottoir, ville, campagne, monde, espace, métro, autobus, train, pays.

Use `micro_place` only for small-scale supports where a body or activity can settle: lit, table, bureau, chaise, fauteuil, bibliothèque. This project is interested in Perec's infra-ordinary spaces, not in ordinary objects as such.

Rules:

- Extract only mentions explicitly present in the text.
- Preserve the exact surface form in mention_text.
- Use normalized_text for a conservative French/local canonical form.
- All normalized_text values should be in French or in the local original form.
- Never translate French place names into English.
- Do not try to decide definitively whether a named place is real or fictional.
- Do not invent geographic metadata.
- Fill city, region, and country only when directly supported by the text or very common knowledge.
- For Paris streets, squares, métro stations, monuments, and districts, city = Paris and country = France when confident.
- For generic_spatial_entity, usually leave city, region, and country null unless the text explicitly links the entity to a named place.
- For generic_spatial_entity, extract nouns or noun phrases that function as spatial settings, containers, routes, thresholds, infrastructures, or stable micro-spatial supports in the passage.
- Do not extract ordinary objects merely because they are physically located somewhere.
- For micro_place, extract only these small-scale supports where a body or activity can settle: lit, table, bureau, chaise, fauteuil, bibliothèque.
- Do not extract other furniture such as armoire, placard, étagère, commode, buffet, vaisselier, canapé, miroir, lavabo, or baignoire.
- Exclude food, materials, decorative objects, tools, documents, artworks, machines, signs, posters, small movable objects, storage furniture, and incidental props, unless they explicitly designate a place.
- Do not extract living beings or body parts as spatial entities.
- Use spatial_type = other rarely. Do not use other as a fallback for ordinary objects.
- Set usage = literal for concrete spatial description, metaphorical for figurative or conceptual spatial uses, and uncertain when the distinction is unclear.
- Exclude people, publishers, collection titles, book titles, section titles, and organizations unless the wording clearly refers to a physical place.
- Ignore tables of contents and front matter such as title pages, copyright pages, publishing-house addresses, collection catalogues, ISBN pages, and lists of works. Do not extract spatial-looking section titles from a table of contents.
- Exclude purely abstract uses when there is no spatial meaning.
- Extract each distinct mention occurrence in the passage, not only one example per type. If the same expression appears repeatedly in different sentences, include each occurrence with its own context. If it is repeated several times in the same short sentence or list, one mention is sufficient.
- Include a short context sentence or phrase.
- Return an empty mentions array if no spatial mentions are found.

Confidence scale:

- 0.95: exact, explicit, unambiguous mention; normalization certain.
- 0.80: clear spatial mention; minor uncertainty about type or metadata.
- 0.60: spatial mention is plausible but ambiguous or context-dependent.
- 0.40: uncertain; normally omit unless the mention is analytically important.

Provenance:
work: {chunk.work}
source_file: {chunk.source_file}
source_type: {chunk.source_type}
source_ref: {chunk.source_ref}
chunk_id: {chunk.chunk_id}

Text:
{chunk.text}
"""


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON defensively, including responses wrapped in code fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict) or not isinstance(data.get("mentions"), list):
        raise ValueError("JSON response must contain a mentions array")
    return data


def normalize_for_filter(text: str) -> str:
    """Lowercase and remove accents for simple deterministic filters."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def filter_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", normalize_for_filter(text)))


def should_keep_spatial_mention(row: dict[str, Any]) -> tuple[bool, str]:
    """Deterministically remove object noise from generic spatial mentions."""
    if row.get("mention_kind") == "named_place":
        return True, ""

    text = normalize_for_filter(f"{row.get('mention_text', '')} {row.get('normalized_text', '')}")
    tokens = filter_tokens(text)

    if tokens & INCIDENTAL_OBJECT_BLACKLIST:
        return False, "incidental_object_blacklist"

    spatial_type = row.get("spatial_type", "")
    if spatial_type == "micro_place" and not (tokens & MICRO_PLACE_WHITELIST):
        return False, "micro_place_not_whitelisted"

    if spatial_type == "other":
        if any(term in text for term in GENERIC_OTHER_WHITELIST):
            return True, ""
        return False, "generic_other_not_whitelisted"

    return True, ""


def validate_mentions(data: dict[str, Any], chunk: Chunk) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and normalize model mention objects for CSV output."""
    rows: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    for item in data.get("mentions", []):
        if not isinstance(item, dict):
            continue
        mention_text = str(item.get("mention_text") or "").strip()
        normalized = str(item.get("normalized_text") or mention_text).strip()
        if not mention_text or not normalized:
            continue

        mention_kind = str(item.get("mention_kind") or "generic_spatial_entity").strip()
        if mention_kind not in MENTION_KINDS:
            mention_kind = "generic_spatial_entity"

        spatial_type = str(item.get("spatial_type") or "other").strip()
        if spatial_type not in SPATIAL_TYPES:
            spatial_type = "other"

        usage = str(item.get("usage") or "uncertain").strip()
        if usage not in USAGES:
            usage = "uncertain"

        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        row = {
            "work": chunk.work,
            "source_file": chunk.source_file,
            "source_type": chunk.source_type,
            "source_ref": chunk.source_ref,
            "chunk_id": chunk.chunk_id,
            "mention_text": mention_text,
            "normalized_text": normalized,
            "mention_kind": mention_kind,
            "spatial_type": spatial_type,
            "usage": usage,
            "city": none_to_empty(item.get("city")),
            "region": none_to_empty(item.get("region")),
            "country": none_to_empty(item.get("country")),
            "confidence": f"{confidence:.3f}",
            "context": str(item.get("context") or "").strip(),
        }
        keep, reason = should_keep_spatial_mention(row)
        if keep:
            rows.append(row)
        else:
            filtered = dict(row)
            filtered["filter_reason"] = reason
            filtered_rows.append(filtered)
    return rows, filtered_rows


def none_to_empty(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "null" else text


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
def call_llm(model_id: str, prompt: str) -> str:
    """Call the configured llm model with retries."""
    model = llm.get_model(model_id)
    response = model.prompt(
        prompt,
        system=(
            "You extract spatial mentions from French literary texts for a scholarly project on Georges Perec. "
            "Return only valid JSON. Distinguish named places from generic infra-ordinary spatial entities, "
            "and distinguish literal from metaphorical usage. Be conservative with geographic metadata. "
            "Keep normalized names in French or local original form; never translate French place names into English."
        ),
    )
    return str(response)


def process_chunk_with_llm(model_id: str, chunk: Chunk) -> tuple[dict[str, Any], dict[str, Any]]:
    """Process one chunk and return its checkpoint record plus processing-log row."""
    try:
        raw = call_llm(model_id, make_prompt(chunk))
        data = parse_json_response(raw)
        mentions, filtered_mentions = validate_mentions(data, chunk)
        record = {
            "source_file": chunk.source_file,
            "source_type": chunk.source_type,
            "source_ref": chunk.source_ref,
            "chunk_id": chunk.chunk_id,
            "status": "success",
            "error": "",
            "characters": len(chunk.text),
            "mentions_found": len(mentions),
            "mentions": mentions,
            "filtered_mentions": filtered_mentions,
        }
        log_row = {
            "source_file": chunk.source_file,
            "chunk_id": chunk.chunk_id,
            "status": "success",
            "error": "",
            "characters": len(chunk.text),
            "mentions_found": len(mentions),
        }
        return record, log_row
    except Exception as exc:
        record = {
            "source_file": chunk.source_file,
            "source_type": chunk.source_type,
            "source_ref": chunk.source_ref,
            "chunk_id": chunk.chunk_id,
            "status": "failure",
            "error": str(exc),
            "characters": len(chunk.text),
            "mentions_found": 0,
            "mentions": [],
            "filtered_mentions": [],
        }
        log_row = {
            "source_file": chunk.source_file,
            "chunk_id": chunk.chunk_id,
            "status": "failure",
            "error": str(exc),
            "characters": len(chunk.text),
            "mentions_found": 0,
        }
        return record, log_row


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_intermediate(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def completed_chunk_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    return {str(r.get("chunk_id")) for r in records if r.get("status") == "success" and r.get("chunk_id")}


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dedupe_mentions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("work", ""),
            row.get("source_ref", ""),
            row.get("mention_text", ""),
            row.get("normalized_text", ""),
            row.get("mention_kind", ""),
            row.get("context", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_named_place_index(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("mention_kind") != "named_place":
            continue
        key = (
            row.get("normalized_text", ""),
            row.get("city", ""),
            row.get("region", ""),
            row.get("country", ""),
        )
        groups[key].append(row)

    index_rows: list[dict[str, Any]] = []
    for (normalized, city, region, country), items in sorted(groups.items(), key=lambda kv: kv[0]):
        type_counts = Counter(item.get("spatial_type", "other") for item in items)
        works = sorted({item.get("work", "") for item in items if item.get("work")})
        sources = sorted(
            {f"{item.get('source_file', '')}:{item.get('source_ref', '')}" for item in items if item.get("source_file")}
        )
        example = next((item.get("context", "") for item in items if item.get("context")), "")
        index_rows.append(
            {
                "normalized_text": normalized,
                "spatial_type": type_counts.most_common(1)[0][0] if type_counts else "other",
                "city": city,
                "region": region,
                "country": country,
                "works": "; ".join(works),
                "mention_count": len(items),
                "sources": "; ".join(sources),
                "example_context": example,
            }
        )
    return index_rows


def build_spatial_type_index(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("work", ""),
            row.get("mention_kind", ""),
            row.get("spatial_type", ""),
            row.get("normalized_text", ""),
        )
        groups[key].append(row)

    index_rows: list[dict[str, Any]] = []
    for (work, mention_kind, spatial_type, normalized), items in sorted(groups.items(), key=lambda kv: kv[0]):
        example = next((item.get("context", "") for item in items if item.get("context")), "")
        index_rows.append(
            {
                "work": work,
                "mention_kind": mention_kind,
                "spatial_type": spatial_type,
                "normalized_text": normalized,
                "mention_count": len(items),
                "example_context": example,
            }
        )
    return index_rows


def print_estimates(model_id: str, chunks: int, chunk_chars: int) -> None:
    """Print rough token, cost, and runtime estimates for a dry run."""
    input_tokens = int(chunk_chars / CHARS_PER_TOKEN_ESTIMATE) + chunks * PROMPT_OVERHEAD_TOKENS_ESTIMATE
    output_tokens = chunks * OUTPUT_TOKENS_PER_CHUNK_ESTIMATE
    total_tokens = input_tokens + output_tokens
    min_seconds = chunks * SECONDS_PER_CHUNK_ESTIMATE_RANGE[0]
    max_seconds = chunks * SECONDS_PER_CHUNK_ESTIMATE_RANGE[1]
    print("Estimate, very approximate:")
    print(f"  LLM chunks: {chunks:,}")
    print(f"  Text characters: {chunk_chars:,}")
    print(f"  Input tokens: ~{input_tokens:,}")
    print(f"  Output tokens: ~{output_tokens:,}")
    print(f"  Total tokens: ~{total_tokens:,}")
    prices = MODEL_PRICE_PER_1M_TOKENS_USD.get(model_id)
    if prices:
        cost = (input_tokens / 1_000_000 * prices["input"]) + (output_tokens / 1_000_000 * prices["output"])
        print(f"  Estimated API cost for {model_id}: ~${cost:.2f} USD")
    else:
        print(f"  Estimated API cost: unknown for model '{model_id}'")
        print("  Add/check model pricing manually: input_tokens and output_tokens are shown above.")
    print(f"  Sequential runtime: ~{min_seconds // 60:.0f}-{max_seconds // 60:.0f} minutes at 2-10 seconds/chunk")
    print("  Pricing, output size, and latency vary by provider/model and text difficulty.")


def rows_from_success_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "success":
            continue
        for mention in record.get("mentions", []):
            if isinstance(mention, dict):
                rows.append({col: mention.get(col, "") for col in MENTION_COLUMNS})
    return rows


def filtered_rows_from_success_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "success":
            continue
        for mention in record.get("filtered_mentions", []):
            if isinstance(mention, dict):
                rows.append({col: mention.get(col, "") for col in FILTERED_MENTION_COLUMNS})
    return rows


def discover_files(input_dir: Path, limit_files: int | None) -> list[Path]:
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".pdf", ".epub"}])
    return files[:limit_files] if limit_files else files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a literary spatial mention index from EPUB/PDF files.")
    parser.add_argument("--input", default="data", help="Input folder containing .epub and .pdf files")
    parser.add_argument("--output", default="output", help="Output folder for CSV and checkpoint files")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="llm model identifier")
    parser.add_argument("--max-chars", type=int, default=8000, help="Maximum characters per chunk")
    parser.add_argument("--overlap", type=int, default=800, help="Overlap characters between chunks")
    parser.add_argument("--limit-files", type=int, default=None, help="Process only first N files")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Process only first N chunks globally")
    parser.add_argument("--resume", action="store_true", help="Skip chunks already successful in checkpoint")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel LLM requests")
    parser.add_argument("--dry-run", action="store_true", help="Extract/chunk and log without calling an LLM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_dir / "intermediate_spatial_mentions.jsonl"

    prior_records = load_intermediate(intermediate_path)
    skip_ids = completed_chunk_ids(prior_records) if args.resume else set()
    log_rows: list[dict[str, Any]] = []
    chunks_to_process: list[Chunk] = []
    processed_chunks = 0
    discovered_chunks = 0
    estimated_llm_chunks = 0
    estimated_chunk_chars = 0

    files = discover_files(input_dir, args.limit_files)
    for file_path in tqdm(files, desc="Files"):
        try:
            sections = extract_sections(file_path)
        except Exception as exc:  # keep processing other files
            log_rows.append(
                {
                    "source_file": file_path.name,
                    "chunk_id": "",
                    "status": "failure",
                    "error": f"extraction failed: {exc}",
                    "characters": 0,
                    "mentions_found": 0,
                }
            )
            continue

        file_chunks: list[Chunk] = []
        for section in sections:
            if looks_like_front_matter_or_toc(section):
                log_rows.append(
                    {
                        "source_file": section.source_file,
                        "chunk_id": f"{sanitize_id_part(section.source_ref)}:skipped",
                        "status": "skipped_front_matter_or_toc",
                        "error": "",
                        "characters": len(section.text),
                        "mentions_found": 0,
                    }
                )
                continue
            file_chunks.extend(chunk_section(section, args.max_chars, args.overlap))

        if not file_chunks:
            log_rows.append(
                {
                    "source_file": file_path.name,
                    "chunk_id": "",
                    "status": "no_text",
                    "error": "",
                    "characters": 0,
                    "mentions_found": 0,
                }
            )
            continue

        for chunk in file_chunks:
            if args.limit_chunks is not None and discovered_chunks >= args.limit_chunks:
                break
            discovered_chunks += 1

            if chunk.chunk_id in skip_ids:
                log_rows.append(
                    {
                        "source_file": chunk.source_file,
                        "chunk_id": chunk.chunk_id,
                        "status": "skipped_success",
                        "error": "",
                        "characters": len(chunk.text),
                        "mentions_found": "",
                    }
                )
                continue

            estimated_llm_chunks += 1
            estimated_chunk_chars += len(chunk.text)

            if args.dry_run:
                log_rows.append(
                    {
                        "source_file": chunk.source_file,
                        "chunk_id": chunk.chunk_id,
                        "status": "dry_run",
                        "error": "",
                        "characters": len(chunk.text),
                        "mentions_found": 0,
                    }
                )
                processed_chunks += 1
                continue

            chunks_to_process.append(chunk)

        if args.limit_chunks is not None and discovered_chunks >= args.limit_chunks:
            break

    if not args.dry_run and chunks_to_process:
        workers = max(1, args.workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_chunk_with_llm, args.model, chunk): chunk for chunk in chunks_to_process
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="LLM chunks"):
                chunk = futures[future]
                try:
                    record, log_row = future.result()
                except Exception as exc:  # unexpected errors outside per-chunk handling
                    record = {
                        "source_file": chunk.source_file,
                        "source_type": chunk.source_type,
                        "source_ref": chunk.source_ref,
                        "chunk_id": chunk.chunk_id,
                        "status": "failure",
                        "error": str(exc),
                        "characters": len(chunk.text),
                        "mentions_found": 0,
                        "mentions": [],
                        "filtered_mentions": [],
                    }
                    log_row = {
                        "source_file": chunk.source_file,
                        "chunk_id": chunk.chunk_id,
                        "status": "failure",
                        "error": str(exc),
                        "characters": len(chunk.text),
                        "mentions_found": 0,
                    }
                append_jsonl(intermediate_path, record)
                log_rows.append(log_row)
                processed_chunks += 1

    all_records = [] if args.dry_run else load_intermediate(intermediate_path)
    mention_rows = dedupe_mentions(rows_from_success_records(all_records))
    filtered_mention_rows = filtered_rows_from_success_records(all_records)
    named_place_index_rows = build_named_place_index(mention_rows)
    spatial_type_index_rows = build_spatial_type_index(mention_rows)

    write_csv(output_dir / "spatial_mentions.csv", MENTION_COLUMNS, mention_rows)
    write_csv(output_dir / "filtered_spatial_mentions.csv", FILTERED_MENTION_COLUMNS, filtered_mention_rows)
    write_csv(output_dir / "named_place_index.csv", NAMED_PLACE_INDEX_COLUMNS, named_place_index_rows)
    write_csv(output_dir / "spatial_type_index.csv", SPATIAL_TYPE_INDEX_COLUMNS, spatial_type_index_rows)
    write_csv(output_dir / "processing_log.csv", LOG_COLUMNS, log_rows)

    print(
        f"Files: {len(files)} | chunks seen: {discovered_chunks} | chunks processed: {processed_chunks} | "
        f"mentions: {len(mention_rows)} | named places: {len(named_place_index_rows)} | "
        f"spatial types: {len(spatial_type_index_rows)}"
    )
    if args.dry_run:
        print("Dry run complete: no LLM API calls were made.")
        print_estimates(args.model, estimated_llm_chunks, estimated_chunk_chars)


if __name__ == "__main__":
    main()
