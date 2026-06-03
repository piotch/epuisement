<p align="center">
  <img src="docs/logo-rectangle.png" alt="Épuisement" width="200">
</p>

This project reads Georges Perec's texts to find passages that seem to point to places, then uses their context to identify them more precisely and build an index of the places — or possible places — present in the work. A first run of the analysis found more than 10,000 place references across 27 books; the results are available in [`output/20260602_place_mentions.csv`](output/20260602_place_mentions.csv).

## Background and motivation

This project originates in a contribution to **_La ville en infographies_**, a volume in the CNRS Éditions **[Homo Graphicus](https://www.cnrseditions.fr/collection/homo-graphicus/)** collection.

One of the planned sections, tentatively titled **"Perec : une géographie personnelle de Paris"**, seeks to explore the geography of Georges Perec's work through computational methods. The objective is modest: to construct a place index from a literary corpus and use it as a starting point for a spatial reading of the oeuvre.

The project benefits from access to a unique corpus of Georges Perec's works made available by the rights holders, thanks in particular to the support of **[Mathilde Moaty](https://cv.hal.science/mathilde-moaty)**.

It also follows from conversations with **[Martine Drozdz](https://www.mfo.ac.uk/people/martine-drozdz)** around *Lieux*, Georges Perec's long-term project of serial observation and memory, published posthumously in 2022. Perec's work remains a major reference for contemporary geographical practices attentive to ordinary places, repetition, and the passage of time (cf. *Nos Lieux Communs*, 2023).

In a sense, this project reverses the movement. Rather than using geography to think with Perec, it returns to Perec's texts themselves and asks what a computational geography of Perec's oeuvre might reveal.

## Technical overview

This is a a Python CLI tool that scans EPUB and PDF files, extracts real-world place mentions with an LLM, and writes CSV files for review.

Entity extraction and normalization are hard problems. Tools such as `spaCy` and `stanza` can be used for this task, but as far as I understand (cf. "Selected References" section), they require domain-specific tuning to produce high-quality results on literary texts.

This project starts from a simpler hypothesis: for a bounded corpus — here, the works of Georges Perec — the cost of using an LLM directly is manageable, and the gain in contextual understanding may outweigh the additional API cost.

The goal is to produce a reviewable place index while exploring how well LLMs handle literary geography.

## Run history

### 2 June 2026

* version 0.0.1
* input tokens: 2.971M
* output tokens: 728.148K
* cost: $0.31 (gpt-4o-mini)

10165 place mentions extracted

## Install

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

This creates/updates `.venv/` from `pyproject.toml` and `uv.lock`.

Configure an LLM provider through the [`llm`](https://llm.datasette.io/) tool/plugins. For the default OpenAI model, set:

```bash
export OPENAI_API_KEY=...
```

A local `.env` file is also loaded.

## Usage

```bash
uv run python -m src.place_index --input data --output output
```

Options:

```bash
--model gpt-4o-mini     # llm model identifier
--max-chars 8000        # chunk size
--overlap 800           # chunk overlap
--limit-files 2         # process only first N files
--limit-chunks 10       # process only first N chunks globally
--resume                # skip successful chunks in output/intermediate_mentions.jsonl
--workers 4             # run N LLM requests in parallel
--dry-run               # extract/chunk and log counts without calling an LLM
```

Example dry run:

```bash
uv run python -m src.place_index --input data --output output --dry-run --limit-files 1
```

A dry run extracts and chunks the files without calling an LLM. It prints the number of chunks plus rough token, cost, and sequential runtime estimates. Cost estimates are approximate and currently use built-in default pricing for `gpt-4o-mini`; check current provider pricing before relying on them for budgeting.

## Outputs

- `output/place_mentions.csv`: one row per deduplicated place mention.
- `output/place_index.csv`: one row per normalized place.
- `output/processing_log.csv`: one row per processed/skipped/failed file or chunk.
- `output/intermediate_mentions.jsonl`: checkpoint records for resumable processing.

## Notes

This v1 does not perform OCR, fuzzy deduplication, gazetteer lookup, geocoding, or database storage. City/region/country fields come only from the model and should be manually reviewed.

## Selected References

- van Dalen-Oskam, K. H., de Does, J., Marx, M., Sijaranamual, I., Depuydt, K., Verheij, B., & Geirnaert, V. (2014). Named Entity Recognition and Resolution for Literary Studies. Computational Linguistics in the Netherlands Journal, 4, 121–136.
  
- Ehrmann, M., Hamdi, A., Pontes, E. L., Romanello, M., & Doucet, A. (2021). Named Entity Recognition and Classification on Historical Documents: A Survey. arXiv:2109.11406.
  
- Dekker, N., Kuhn, J., van Erp, M., et al. Evaluating Named Entity Recognition Tools for Extracting Social Networks from Novels.