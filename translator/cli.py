from __future__ import annotations

import argparse
from pathlib import Path
import sys

from translator.pdf_io import discover_pdfs
from translator.render import translate_pdf_preserve_layout
from translator.translate import load_translator_config, translate_texts
from translator.translation_memory import TranslationMemory
from translator.font_utils import resolve_font_path
from translator.config import load_config, get_openai_config, get_default_font_path, get_local_config
from openai import OpenAIError, RateLimitError
from tqdm import tqdm


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
        choices=["openai", "ollama", "huggingface"],
        help="Translation provider",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local Ollama mode (overrides --provider)",
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
        "--local-base-url",
        default=None,
        help="Local AI base URL (overrides local.base_url in config.yml)",
    )
    parser.add_argument(
        "--font",
        required=False,
        help="Font path or installed font name that supports Farsi",
    )
    parser.add_argument(
        "--font-fallback",
        default="assets/fonts/NotoNaskhArabic-Regular.ttf",
        help="Fallback font path to try if the primary font fails",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress while translating",
    )
    parser.add_argument(
        "--debug-draw",
        action="store_true",
        help="Draw debug markers/boxes to verify output rendering",
    )
    parser.add_argument(
        "--glossary",
        default=None,
        help="Path to a glossary file (one 'term=translation' per line)",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Page range to translate, e.g. '1-5,10,15-20'. Default: all pages.",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Interleave original and translated pages in the output PDF",
    )
    parser.add_argument(
        "--tm-path",
        default=None,
        help="Path to translation memory JSONL file (default: jobs/translation_memory.jsonl)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom instructions appended to the system prompt (e.g. 'Use formal tone')",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=None,
        help="Fixed font size in points (default: auto-match original)",
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
    local_cfg = get_local_config(cfg)
    default_font = get_default_font_path(cfg)

    provider = "ollama" if args.local else args.provider

    model_override = args.model or (local_cfg.get("model") if provider == "ollama" else openai_cfg.get("model"))
    api_key_override = args.openai_api_key or (
        local_cfg.get("api_key") if provider == "ollama" else openai_cfg.get("api_key")
    )
    if provider == "ollama":
        base_url_override = args.local_base_url or local_cfg.get("base_url")
    else:
        base_url_override = args.openai_base_url or openai_cfg.get("base_url")

    config = load_translator_config(
        provider,
        model_override,
        api_key_override=api_key_override,
        base_url_override=base_url_override,
    )
    if provider == "openai" and not config.api_key:
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
    font_fallback = resolve_font_path(args.font_fallback) if args.font_fallback else None
    print(f"Using font: {font_path}")

    # Parse glossary
    glossary: dict[str, str] = {}
    if args.glossary:
        gp = Path(args.glossary)
        if gp.exists():
            for line in gp.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    glossary[k.strip()] = v.strip()
    config.glossary = glossary
    if args.prompt:
        config.custom_prompt = args.prompt

    # Load translation memory
    if args.tm_path:
        config.translation_memory = TranslationMemory(args.tm_path)
    else:
        tm_default = Path("jobs/translation_memory.jsonl")
        if tm_default.exists():
            config.translation_memory = TranslationMemory(tm_default)

    # Parse page range
    page_range_set = None
    if args.pages:
        page_range_set = set()
        for part in args.pages.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                page_range_set.update(range(int(lo), int(hi) + 1))
            elif part.isdigit():
                page_range_set.add(int(part))

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

        if args.verbose:
            print(f"Translating: {pdf_path}")
        # total may be unknown until we open the PDF; tqdm handles dynamic totals
        pbar = tqdm(total=0, desc=f"{pdf_path.name} pages", unit="page", leave=False)
        try:
            try:
                translate_pdf_preserve_layout(
                    pdf_path,
                    out_path,
                    font_path,
                    lambda texts: translate_texts(texts, args.lang, config),
                    verbose=args.verbose,
                    debug_draw=args.debug_draw,
                    on_page=lambda current, total: (
                        pbar.reset(total=total),
                        pbar.update(current - pbar.n),
                    ),
                    page_range=page_range_set,
                    side_by_side=args.side_by_side,
                    font_size_override=args.font_size,
                )
            except Exception as exc:
                if not font_fallback:
                    raise
                if args.verbose:
                    print(
                        f"Primary font failed ({font_path}). Retrying with fallback {font_fallback}",
                        file=sys.stderr,
                    )
                pbar.reset(total=0)
                translate_pdf_preserve_layout(
                    pdf_path,
                    out_path,
                    font_fallback,
                    lambda texts: translate_texts(texts, args.lang, config),
                    verbose=args.verbose,
                    debug_draw=args.debug_draw,
                    on_page=lambda current, total: (
                        pbar.reset(total=total),
                        pbar.update(current - pbar.n),
                    ),
                    page_range=page_range_set,
                    side_by_side=args.side_by_side,
                    font_size_override=args.font_size,
                )
        except RateLimitError as exc:
            print(
                "OpenAI rate limit or quota error. Please check your plan/billing "
                "and try again.",
                file=sys.stderr,
            )
            print(str(exc), file=sys.stderr)
            pbar.close()
            return 3
        except OpenAIError as exc:
            print("OpenAI API error. Please try again or verify your API key.", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            pbar.close()
            return 3
        finally:
            pbar.close()
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
