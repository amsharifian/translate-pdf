# PDF to Farsi Translator

A small Python CLI that selects PDFs via an input switch, translates them to Farsi, and saves new PDFs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Translation Provider

This project uses the OpenAI API by default.

Set environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
# Optional:
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

You can also pass these as CLI inputs:

```bash
translate-pdf -i "./docs/*.pdf" \\
  --openai-api-key "your-openai-api-key" \\
  --openai-model "gpt-4o-mini" \\
  --openai-base-url "https://api.openai.com/v1" \\
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

You can run translations fully locally for free using Ollama. This uses Ollama's OpenAI-compatible API at `http://localhost:11434/v1/` and does not require an API key.

### Install Ollama

**macOS / Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**

Download and install Ollama for Windows, then verify `ollama` is available in your terminal. citeturn0search0

1. Start Ollama and pull a model:
```bash
ollama serve
ollama run qwen3:8b
```


2. Translate locally:
```bash
translate-pdf -i "./docs/*.pdf" \
  --provider ollama \
  --model qwen3:8b \
  --font "assets/fonts/Vazirmatn/fonts/ttf/Vazirmatn-Regular.ttf"
```

Or use the shortcut:

```bash
translate-pdf -i "./docs/*.pdf" --local --font "assets/fonts/Vazirmatn/fonts/ttf/Vazirmatn-Regular.ttf"
```

### Recommended Local Models

- `qwen3:8b` (good Persian support and size)
- `llama3.3:70b` (large, good quality)
- `llama4:scout` or `llama4:maverick` (very large; not 8B)

Note: Llama 4 models available in Ollama are very large (Scout is 109B, Maverick is 400B). There is no official 8B Llama 4 in Ollama at this time.

### Local Setup Script

```bash
./scripts/ollama_pull.sh qwen3:8b
```

## Font

You must supply a TTF/OTF font that supports Persian/Farsi (e.g., Noto Naskh Arabic, Vazirmatn, or IRANSans).

This repo includes downloaded fonts in `assets/fonts`:
- Vazirmatn: `assets/fonts/Vazirmatn/fonts/ttf/Vazirmatn-Regular.ttf`
- Noto Naskh Arabic: `assets/fonts/NotoNaskhArabic-Regular.ttf`

Example:

```bash
translate-pdf -i "./docs/*.pdf" --font "assets/fonts/Vazirmatn/fonts/ttf/Vazirmatn-Regular.ttf"
translate-pdf -i "./docs/*.pdf" --font "assets/fonts/NotoNaskhArabic-Regular.ttf"
translate-pdf -i "./docs/*.pdf" --font "Vazirmatn"
```

## Usage

```bash
translate-pdf -i ./invoices --font /path/to/Vazirmatn-Regular.ttf
translate-pdf -i ./file1.pdf ./file2.pdf --out-dir ./out --font /path/to/font.ttf
translate-pdf -i "./docs/*.pdf" --provider dummy --font /path/to/font.ttf
translate-pdf -i "./docs/*.pdf" --model gpt-4o-mini --font /path/to/font.ttf
translate-pdf -i "./docs/*.pdf" --provider ollama --model qwen3:8b --font /path/to/font.ttf
```

## Notes

- The output preserves original PDF layout by redacting text blocks and inserting translations in-place.
- Complex PDFs (tables, multi-column, or sparse layouts) may need font-size tuning.

## Debugging

```bash
translate-pdf -i "./docs/*.pdf" --font /path/to/font.ttf --verbose
translate-pdf -i "./docs/*.pdf" --font /path/to/font.ttf --verbose --debug-draw
```

You can also specify a fallback font:

```bash
translate-pdf -i "./docs/*.pdf" \
  --font /path/to/primary.ttf \
  --font-fallback /path/to/fallback.ttf
```
