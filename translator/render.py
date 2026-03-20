from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List

import arabic_reshaper
from bidi.algorithm import get_display
import fitz  # PyMuPDF


def _shape_rtl_text(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    shaped_lines: List[str] = []
    for line in lines:
        reshaped = arabic_reshaper.reshape(line)
        shaped_lines.append(get_display(reshaped))
    return "\n".join(shaped_lines)


def _block_text_and_size(block: dict) -> tuple[str, float]:
    lines = block.get("lines", [])
    parts: List[str] = []
    sizes: List[float] = []
    for line in lines:
        spans = line.get("spans", [])
        line_text_parts: List[str] = []
        for span in spans:
            text = span.get("text", "")
            if text:
                line_text_parts.append(text)
            size = span.get("size")
            if isinstance(size, (int, float)):
                sizes.append(float(size))
        parts.append("".join(line_text_parts))
    text = "\n".join(parts).strip()
    avg_size = sum(sizes) / len(sizes) if sizes else 11.0
    return text, avg_size


def translate_pdf_preserve_layout(
    pdf_path: Path,
    out_path: Path,
    font_path: str,
    translate_fn: Callable[[Iterable[str]], List[str]],
    font_name: str = "FarsiFont",
) -> None:
    if not font_path:
        raise ValueError("A TTF font path is required to render Farsi text.")

    doc = fitz.open(str(pdf_path))
    for page in doc:
        page_dict = page.get_text("dict")
        blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]

        texts: List[str] = []
        font_sizes: List[float] = []
        rects: List[fitz.Rect] = []

        for block in blocks:
            text, avg_size = _block_text_and_size(block)
            if not text:
                continue
            rect = fitz.Rect(block["bbox"])
            texts.append(text)
            font_sizes.append(avg_size)
            rects.append(rect)

        if not texts:
            continue

        translated = translate_fn(texts)

        # Redact original text blocks
        for rect in rects:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()

        # Insert translated text into the same rectangles
        for rect, translated_text, font_size in zip(rects, translated, font_sizes):
            shaped = _shape_rtl_text(translated_text)
            page.insert_textbox(
                rect,
                shaped,
                fontname=font_name,
                fontfile=font_path,
                fontsize=max(6.0, min(font_size, 24.0)),
                color=(0, 0, 0),
                align=fitz.TEXT_ALIGN_RIGHT,
            )

    doc.save(str(out_path))
    doc.close()
