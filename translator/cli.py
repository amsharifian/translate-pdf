from __future__ import annotations

import argparse
from pathlib import Path
import sys

from translator.pdf_io import discover_pdfs
from translator.render import translate_pdf_preserve_layout
from translator.translate import load_translator_config, translate_texts
from translator.font_utils import resolve_font_path
from translator.config import load_config, get_openai_config, get_default_font_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate PDFs to Farsi and save as PDFs."
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="PDF file(s), directory, or glob(s) to translate",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default="translated",
        help="Output directory for translated PDFs",
    )
    parser.add_argument(
        "--lang",
        default="fa",
        help="Target language code (default: fa)",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "dummy"],
        help="Translation provider",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model override (defaults to OPENAI_MODEL or gpt-4o-mini)",
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Path to YAML config (default: config.yml)",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="OpenAI base URL (overrides OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--font",
        required=False,
        help="Font path or installed font name that supports Farsi",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    inputs = discover_pdfs(args.input)
    if not inputs:
        print("No PDFs found from the provided input(s).", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    openai_cfg = get_openai_config(cfg)
    default_font = get_default_font_path(cfg)

    config = load_translator_config(
        args.provider,
        args.model or openai_cfg.get("model"),
        api_key_override=args.openai_api_key or openai_cfg.get("api_key"),
        base_url_override=args.openai_base_url or openai_cfg.get("base_url"),
    )
    if args.provider == "openai" and not config.api_key:
        print(
            "Missing OpenAI API key. Provide --openai-api-key or set OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 2
    font_arg = args.font or default_font
    font_path = resolve_font_path(font_arg) if font_arg else None
    if not font_path:
        print(
            "Font not found. Provide --font or set font.default_path in config.yml.",
            file=sys.stderr,
        )
        return 2
    print(f"Using font: {font_path}")

    for pdf_path in inputs:
        if not pdf_path.exists():
            print(f"Skipping missing file: {pdf_path}", file=sys.stderr)
            continue
        if pdf_path.suffix.lower() != ".pdf":
            print(f"Skipping non-pdf: {pdf_path}", file=sys.stderr)
            continue

        out_path = out_dir / f"{pdf_path.stem}.fa.pdf"
        if out_path.exists() and not args.force:
            print(f"Skipping existing output (use --force to overwrite): {out_path}")
            continue

        translate_pdf_preserve_layout(
            pdf_path,
            out_path,
            font_path,
            lambda texts: translate_texts(texts, args.lang, config),
        )
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
