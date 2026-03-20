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
```

## Notes

- The output preserves original PDF layout by redacting text blocks and inserting translations in-place.
- Complex PDFs (tables, multi-column, or sparse layouts) may need font-size tuning.
