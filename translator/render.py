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
    verbose: bool = False,
    debug_draw: bool = False,
    on_page: Callable[[int, int], None] | None = None,
) -> None:
    if not font_path:
        raise ValueError("A TTF font path is required to render Farsi text.")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    for page_index, page in enumerate(doc, start=1):
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
            if verbose:
                print(f"[{pdf_path.name}] Page {page_index}: no text blocks found")
            continue

        if debug_draw:
            # Draw a visible marker so we can confirm the page renders at all
            page.insert_text(
                (40, 40),
                "DEBUG: render check",
                fontname="helv",
                fontsize=12,
                color=(1, 0, 0),
            )

        if verbose:
            print(f"[{pdf_path.name}] Page {page_index}: translating {len(texts)} blocks")
            print(f"[{pdf_path.name}] Page {page_index}: first block bbox {rects[0]}")
            print(f"[{pdf_path.name}] Page {page_index}: first block font size {font_sizes[0]:.2f}")

        translated = translate_fn(texts)
        if verbose:
            non_empty = sum(1 for t in translated if t and t.strip())
            print(f"[{pdf_path.name}] Page {page_index}: translated non-empty blocks {non_empty}/{len(translated)}")

        # Only redact blocks that have non-empty translations
        redact_rects: List[fitz.Rect] = []
        translated_triplets: List[tuple[fitz.Rect, str, float]] = []
        for rect, translated_text, font_size in zip(rects, translated, font_sizes):
            if translated_text.strip():
                redact_rects.append(rect)
                translated_triplets.append((rect, translated_text, font_size))
            elif verbose:
                print(f"[{pdf_path.name}] Page {page_index}: empty translation, keeping original text")

        for rect in redact_rects:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        if redact_rects:
            page.apply_redactions()

        if debug_draw:
            for rect in rects:
                page.draw_rect(rect, color=(1, 0, 0), width=0.5)

        # Insert translated text into the same rectangles with auto-shrink
        for block_index, (rect, translated_text, font_size) in enumerate(
            translated_triplets, start=1
        ):
            shaped = _shape_rtl_text(translated_text)
            size = max(6.0, min(font_size, 24.0))
            inserted = None
            while size >= 6.0:
                inserted = page.insert_textbox(
                    rect,
                    shaped,
                    fontname=font_name,
                    fontfile=font_path,
                    fontsize=size,
                    color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_RIGHT,
                )
                if inserted is not None and inserted >= 0:
                    break
                size -= 1.0
            if inserted is None or inserted < 0:
                # Try expanding the box height downward
                expanded = fitz.Rect(rect.x0, rect.y0, rect.x1, page.rect.y1 - 20)
                size = max(6.0, min(font_size, 24.0))
                while size >= 6.0:
                    inserted = page.insert_textbox(
                        expanded,
                        shaped,
                        fontname=font_name,
                        fontfile=font_path,
                        fontsize=size,
                        color=(0, 0, 0),
                        align=fitz.TEXT_ALIGN_RIGHT,
                    )
                    if inserted is not None and inserted >= 0:
                        rect = expanded
                        break
                    size -= 1.0
                if verbose:
                    if inserted is not None and inserted >= 0:
                        print(
                            f"[{pdf_path.name}] Page {page_index} block {block_index}: "
                            f"inserted after expanding box with font size {size:.1f}"
                        )
                    else:
                        print(
                            f"[{pdf_path.name}] Page {page_index} block {block_index}: "
                            f"could not fit text even after expanding box"
                        )
            else:
                if verbose:
                    print(
                        f"[{pdf_path.name}] Page {page_index} block {block_index}: "
                        f"inserted with font size {size:.1f}"
                    )

        if on_page:
            on_page(page_index, total_pages)

    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()
