# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A single-page Streamlit app for exploring a corpus of press-review articles. It visualizes each newspaper's aggregate "relevance" over time as a bar chart, lets the user drill into a selected time bucket, and provides a TF-IDF keyword search over the articles in that selection.

## Running

The project uses a conda environment (see `.vscode/settings.json`). There is no build step, test suite, or linter configured.

```bash
streamlit run app.py
```

On first run, NLTK downloads the `punkt`/`punkt_tab` tokenizer models (triggered from `app.py` and `utils.py`).

## Architecture

Two source files:

- **`app.py`** — the entire UI and data pipeline. Flow:
  1. `read_data()` (cached with `@st.cache_resource`) loads `all_ratings0625.csv`, normalizes `rating` to 0–1 (raw scale is 0–5), parses `Date` into a `time_frame` datetime, and de-duplicates by `Title` (except for placeholder titles like `"No Title found"`).
  2. User picks "By year" or "By month" and one or more newspapers. A Plotly grouped bar chart shows summed relevance (`weighted_count`) per time bucket per newspaper.
  3. Clicking a bar (`on_select="rerun"`) opens a set of per-newspaper tabs listing the matching articles, each in an `st.expander`.
  4. Each tab has a per-newspaper TF-IDF search box; the two branches (year / month) contain near-identical display loops — **changes to article rendering usually need to be made in both places.**

- **`utils.py`** — search logic:
  - `TFIDF` — builds a term-frequency/inverse-document-frequency matrix over a list of article texts and ranks them against a query (`query()` returns `(index, score)` pairs). Instantiated fresh per search over the currently-selected `chunks`.
  - `search_snippet(text, query, n_sentences=2)` — returns a 2-sentence context window around the first query match with matched terms wrapped in `<mark>` tags (rendered via `st.markdown(..., unsafe_allow_html=True)`). Escapes backticks/`$` for markdown safety.

## Data

`all_ratings0625.csv` is the active dataset (~11.6k rows). Key columns consumed by the app: `Title`, `Date`, `Author`, `Newspaper`, `chunks` (the article text body), and `rating`. The other CSVs (`final_press_review.csv`, `chunked_press_review.csv`, `extra_times_guadian_data.csv`) are earlier/intermediate datasets and are not read by `app.py`.

When rendering `chunks` text through Streamlit markdown, escape backticks and `$` (`.replace("\`", r"\\\`").replace("$", r"\\$")`) to avoid code-span / LaTeX rendering artifacts — `search_snippet` already does this internally.
