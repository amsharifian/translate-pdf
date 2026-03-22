# Stormlight Translate

A Python toolkit for translating PDFs while preserving the original layout. Supports 13 languages, OpenAI and local Ollama models, a CLI, and a full-featured Streamlit web app with background job processing.

## Quick Setup

```bash
python3 -m venv prun
source prun/bin/activate
pip install -e .
```

This installs the `translate-pdf` CLI and all Python dependencies (including `cryptography` and `httpx` for API-key encryption and webhook support).

## Translation Provider

This project uses the OpenAI API by default.

```bash
export OPENAI_API_KEY="your-openai-api-key"
# Optional:
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Or pass them directly:

```bash
translate-pdf -i "./docs/*.pdf" \
  --openai-api-key "your-key" \
  --openai-model "gpt-4o-mini" \
  --font "assets/fonts/Vazirmatn/fonts/ttf/Vazirmatn-Regular.ttf"
```

## YAML Config

Defaults are loaded from `config.yml` (override with `--config`).

Example `config.yml`:

```yaml
openai:
  api_key: "your-openai-api-key"
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"
local:
  model: "qwen3:8b"
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
font:
  default_path: "assets/fonts/Vazirmatn/fonts/ttf/Vazirmatn-Regular.ttf"
```

## Local AI (Fully Offline)

Translate locally for free using [Ollama](https://ollama.com). No API key required.

1. Start Ollama and pull a model:
```bash
ollama serve
ollama run qwen3:8b
```

2. Translate:
```bash
translate-pdf -i "./docs/*.pdf" \
  --provider ollama --model qwen3:8b \
  --font "assets/fonts/Vazirmatn/fonts/ttf/Vazirmatn-Regular.ttf"
```

Or use the shortcut:
```bash
translate-pdf -i "./docs/*.pdf" --local --font Vazirmatn
```

### Recommended Local Models

| Model | Size | Notes |
|-------|------|-------|
| `qwen3:8b` | 8B | Good Persian support, fast |
| `llama3.3:70b` | 70B | High quality |
| `llama4:scout` | 109B | Very large |
| `llama4:maverick` | 400B | Very large |

### Local Setup Script

```bash
./scripts/ollama_pull.sh qwen3:8b
```

## HuggingFace Models (Open-Source)

Run open-source translation models locally using HuggingFace `transformers`. No API key needed — models are downloaded automatically on first use.

```bash
pip install -e .
```

Requires Python ≤ 3.12 (for PyTorch compatibility).

### Available Models

| Model | HuggingFace ID | Size | Notes |
|-------|---------------|------|-------|
| **NLLB-200 600M** | `facebook/nllb-200-distilled-600M` | ~600MB | Recommended — fast, good quality |
| **NLLB-200 1.3B** | `facebook/nllb-200-distilled-1.3B` | ~1.3GB | Best quality, larger model |
| **mBART-50** | `facebook/mbart-large-50-many-to-many-mmt` | ~2.5GB | Multilingual (50 languages) |

### CLI Usage

```bash
translate-pdf -i doc.pdf --provider huggingface --model facebook/nllb-200-distilled-600M --font Vazirmatn
```

In the web app, select **"HuggingFace (local)"** from the translation mode dropdown and pick a model.

## Font

You must supply a TTF/OTF font that supports the target script. Bundled fonts:

- **Vazirmatn**: `assets/fonts/Vazirmatn/fonts/ttf/Vazirmatn-Regular.ttf`
- **Noto Naskh Arabic**: `assets/fonts/NotoNaskhArabic-Regular.ttf`

```bash
translate-pdf -i "./docs/*.pdf" --font Vazirmatn
translate-pdf -i "./docs/*.pdf" --font "assets/fonts/NotoNaskhArabic-Regular.ttf"
```

## CLI Usage

```bash
# Basic
translate-pdf -i ./invoices --font Vazirmatn

# Multiple files with custom output
translate-pdf -i ./file1.pdf ./file2.pdf --out-dir ./out --font /path/to/font.ttf

# Specific provider & model
translate-pdf -i "./docs/*.pdf" --provider ollama --model qwen3:8b --font Vazirmatn

# Translate specific pages only
translate-pdf -i doc.pdf --pages "1-5,10,15-20" --font Vazirmatn

# Side-by-side output (original + translated pages interleaved)
translate-pdf -i doc.pdf --side-by-side --font Vazirmatn

# Use a glossary file (one "term=translation" per line)
translate-pdf -i doc.pdf --glossary terms.txt --font Vazirmatn

# Use a translation memory file for sentence-level context
translate-pdf -i doc.pdf --tm-path jobs/translation_memory.jsonl --font Vazirmatn

# Target a different language (default: fa)
translate-pdf -i doc.pdf --lang ar --font Vazirmatn
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `-i`, `--input` | PDF file(s), directory, or glob(s) |
| `-o`, `--out-dir` | Output directory (default: `translated`) |
| `--lang` | Target language code (default: `fa`) |
| `--provider` | `openai`, `ollama`, or `dummy` |
| `--local` | Shortcut for Ollama mode |
| `--model` | Model override |
| `--font` | Font path or installed name |
| `--font-fallback` | Fallback font if primary fails |
| `--glossary` | Glossary file path |
| `--pages` | Page range (e.g. `1-5,10`) |
| `--side-by-side` | Interleave original + translated pages |
| `--tm-path` | Translation memory JSONL file path |
| `--force` | Overwrite existing outputs |
| `--verbose` | Print progress |
| `--debug-draw` | Draw debug markers on output |

## Web App (Streamlit)

A full-featured web interface for uploading PDFs, configuring translation, and managing jobs.

### Running

```bash
pip install -e .
streamlit run app.py
```

### Features

- **Multi-language support** — 13 target languages: Farsi, Arabic, Turkish, Urdu, French, German, Spanish, Chinese, Japanese, Korean, Russian, Hindi, English
- **PDF preview** — View uploaded PDF pages before translating
- **Page range selection** — Translate only specific pages (e.g. `1-5,10,15-20`)
- **Glossary** — Per-job glossary terms to ensure consistent terminology
- **Global glossary** — Editable in the sidebar; automatically merged into every job. Stored in `jobs/glossary.json`
- **Side-by-side output** — Interleave original and translated pages in the output PDF
- **Cost estimate** — Rough token-based cost estimate before submitting a job
- **Job naming** — Give custom names to jobs for easy identification
- **Job logs viewer** — View per-page translation logs for any job
- **Output preview** — Thumbnail previews of translated PDF pages
- **Webhook notifications** — Optionally receive a POST request when a job completes or fails
- **Persistent notifications** — Toast alerts for completed/failed jobs, persisted across page refreshes
- **Auto-polling** — Job queue refreshes every 5 seconds automatically
- **Worker auto-start** — The background worker starts automatically when you submit a job. Status and restart controls are in the sidebar
- **Settings persistence** — Provider, model, base URL, and API key are remembered across sessions
- **Dark mode** — Fully styled for both light and dark Streamlit themes
- **API key encryption** — Keys are encrypted at rest using Fernet symmetric encryption (`jobs/.key`)
- **Translation Memory** — Review AI translations, save corrections, and have future translations learn from your feedback (sentence/paragraph-level few-shot examples via TF-IDF retrieval)

### Background Worker

The worker processes queued jobs in the background. It starts automatically when you submit a job from the web app. You can also start it manually:

```bash
python scripts/worker.py
```

Or run both web app and worker together:

```bash
python scripts/run_stormlight.py
```

Worker status is shown in the sidebar with options to restart if needed.

### Global Glossary

The global glossary lives at `jobs/glossary.json` and is editable from the sidebar. Every job automatically inherits these terms. Per-job glossary entries override global ones.

Example `jobs/glossary.json`:

```json
{
  "Machine Learning": "یادگیری ماشین",
  "Neural Network": "شبکه عصبی",
  "Artificial Intelligence": "هوش مصنوعی",
  "Algorithm": "الگوریتم",
  "Database": "پایگاه داده"
}
```

## Translation Memory

Translation Memory (TM) is a sentence/paragraph-level feedback system. After a job completes, open the **"Review translations"** panel on the job card to see every translated block. You can:

- **Accept** the AI translation as a positive example
- **Correct** the translation and save your version
- **Skip** blocks that don't need feedback

Saved pairs are stored in `jobs/translation_memory.jsonl`. On future translations, the system finds the most similar stored examples using TF-IDF cosine similarity and injects them as few-shot examples in the LLM prompt — teaching the AI your preferred style and phrasing.

Manage TM from the sidebar: browse entries, delete, export/import, or clear.

CLI usage:

```bash
translate-pdf -i doc.pdf --tm-path jobs/translation_memory.jsonl --font Vazirmatn
```

If `jobs/translation_memory.jsonl` exists, the CLI loads it automatically.

## Architecture

```
translator/
├── cli.py          # CLI entry point (translate-pdf command)
├── config.py       # YAML config loader
├── crypto.py       # Fernet encryption for API keys at rest
├── font_utils.py   # Font discovery and resolution
├── job_queue.py    # SQLite-backed job CRUD
├── pdf_io.py       # PDF file discovery
├── render.py       # Core PDF → translate → render pipeline
├── translate.py    # LLM translation with retry & chunking
└── translation_memory.py  # Sentence-level TM with TF-IDF search

scripts/
├── worker.py       # Background job processor
├── run_stormlight.py  # Launcher for web app + worker
└── ollama_pull.sh  # Helper to pull Ollama models

app.py              # Streamlit web app
config.yml          # Default configuration
```

### Key Implementation Details

- **Binary search font fitting** — Finds the largest font size that fits each text block's bounding box
- **Exponential backoff retry** — LLM calls retry up to 5 times with 1s/2s/4s/8s/16s delays on transient errors
- **Sentence-boundary chunking** — Long text blocks are split on sentence boundaries before translation
- **Page-level resume** — Translation progress is logged to `.log.jsonl` files, allowing interrupted jobs to resume from the last completed page
- **Translation Memory (TF-IDF)** — User-corrected translation pairs are stored in a JSONL file and the most relevant ones are injected as few-shot examples into the LLM prompt at translation time

## Notes

- The output preserves original PDF layout by redacting text blocks and inserting translations in-place.
- Complex PDFs (tables, multi-column, or sparse layouts) may need font-size tuning.

## Debugging

```bash
translate-pdf -i "./docs/*.pdf" --font Vazirmatn --verbose
translate-pdf -i "./docs/*.pdf" --font Vazirmatn --verbose --debug-draw
```

You can also specify a fallback font:

```bash
translate-pdf -i "./docs/*.pdf" \
  --font /path/to/primary.ttf \
  --font-fallback /path/to/fallback.ttf
```

To diagnose stuck or failed jobs, use the diagnostic script:

```bash
python scripts/debug_jobs.py
```
