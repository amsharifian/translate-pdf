from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set

import arabic_reshaper
from bidi.algorithm import get_display
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


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


def _bisect_font_size(
    page: fitz.Page,
    rect: fitz.Rect,
    shaped: str,
    font_name: str,
    font_file: str,
    min_size: float,
    max_size: float,
) -> tuple[float, int]:
    """Binary‑search for the largest font size that fits *rect*. Returns (size, rc)."""
    lo, hi = min_size, max_size
    best_size, best_rc = lo, -1
    while hi - lo > 0.5:
        mid = (lo + hi) / 2
        rc = page.insert_textbox(
            rect, shaped,
            fontname=font_name, fontfile=font_file,
            fontsize=mid, color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_RIGHT,
            overlay=False,
        )
        if rc is not None and rc >= 0:
            best_size, best_rc = mid, rc
            lo = mid
        else:
            hi = mid
    return best_size, best_rc


def _completed_pages(log_path: Path) -> Set[int]:
    """Return set of 1-based page indices already translated (for resume)."""
    done: Set[int] = set()
    if not log_path.exists():
        return done
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            if entry.get("event") == "page_done":
                done.add(entry["page"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def _log_event(log_path: Path, event: dict) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def extract_page_text_blocks(
    pdf_path: Path, page_indices: Optional[Iterable[int]] = None,
) -> dict[int, List[dict]]:
    """Extract text blocks per page. Returns {1-based page: [block_info, ...]}."""
    doc = fitz.open(str(pdf_path))
    result: dict[int, List[dict]] = {}
    indices = set(page_indices) if page_indices else set(range(1, len(doc) + 1))
    for page_index in sorted(indices):
        if page_index < 1 or page_index > len(doc):
            continue
        page = doc[page_index - 1]
        page_dict = page.get_text("dict")
        blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]
        block_infos: List[dict] = []
        for block in blocks:
            text, avg_size = _block_text_and_size(block)
            if text:
                block_infos.append({
                    "text": text,
                    "font_size": avg_size,
                    "bbox": list(block["bbox"]),
                })
        result[page_index] = block_infos
    doc.close()
    return result


def translate_pdf_preserve_layout(
    pdf_path: Path,
    out_path: Path,
    font_path: str,
    translate_fn: Callable[[Iterable[str]], List[str]],
    font_name: str = "FarsiFont",
    verbose: bool = False,
    debug_draw: bool = False,
    on_page: Callable[[int, int], None] | None = None,
    on_pause: Callable[[], None] | None = None,
    page_range: Optional[Set[int]] = None,
    side_by_side: bool = False,
    log_path: Optional[Path] = None,
) -> None:
    if not font_path:
        raise ValueError("A TTF font path is required to render Farsi text.")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    # Determine which pages to translate
    pages_to_do = page_range if page_range else set(range(1, total_pages + 1))

    # Resume support: skip already-done pages
    if log_path is None:
        log_path = out_path.with_suffix(".log.jsonl")
    done_pages = _completed_pages(log_path)

    # Side-by-side: build new doc interleaving original + translated pages
    if side_by_side:
        sbs_doc = fitz.open()  # new empty doc

    for page_index in range(1, total_pages + 1):
        page = doc[page_index - 1]

        if side_by_side:
            # Copy original page first
            sbs_doc.insert_pdf(doc, from_page=page_index - 1, to_page=page_index - 1)

        if page_index not in pages_to_do:
            if side_by_side:
                # Insert blank translated placeholder
                sbs_doc.insert_pdf(doc, from_page=page_index - 1, to_page=page_index - 1)
            if on_page:
                on_page(page_index, total_pages)
            continue

        if page_index in done_pages:
            if verbose:
                logger.info("[%s] Page %d: already done (resume), skipping", pdf_path.name, page_index)
            if side_by_side:
                sbs_doc.insert_pdf(doc, from_page=page_index - 1, to_page=page_index - 1)
            if on_page:
                on_page(page_index, total_pages)
            continue

        if on_pause:
            on_pause()

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
                logger.info("[%s] Page %d: no text blocks found", pdf_path.name, page_index)
            _log_event(log_path, {"event": "page_done", "page": page_index, "blocks": 0})
            if side_by_side:
                sbs_doc.insert_pdf(doc, from_page=page_index - 1, to_page=page_index - 1)
            if on_page:
                on_page(page_index, total_pages)
            continue

        if debug_draw:
            page.insert_text(
                (40, 40), "DEBUG: render check", fontname="helv", fontsize=12, color=(1, 0, 0),
            )

        if verbose:
            logger.info("[%s] Page %d: translating %d blocks", pdf_path.name, page_index, len(texts))

        translated = translate_fn(texts)

        # Log per-block for the job logs viewer
        block_logs: List[dict] = []

        # For side-by-side, work on a fresh copy of this page
        target_page = page
        if side_by_side:
            sbs_doc.insert_pdf(doc, from_page=page_index - 1, to_page=page_index - 1)
            target_page = sbs_doc[-1]

        # Only redact blocks that have non-empty translations
        redact_rects: List[fitz.Rect] = []
        translated_triplets: List[tuple[fitz.Rect, str, float]] = []
        for rect, translated_text, font_size, orig_text in zip(rects, translated, font_sizes, texts):
            if translated_text.strip():
                redact_rects.append(rect)
                translated_triplets.append((rect, translated_text, font_size))
                block_logs.append({"original": orig_text, "translated": translated_text, "font_size": font_size})
            elif verbose:
                logger.info("[%s] Page %d: empty translation, keeping original", pdf_path.name, page_index)

        for rect in redact_rects:
            target_page.add_redact_annot(rect, fill=(1, 1, 1))
        if redact_rects:
            target_page.apply_redactions()

        if debug_draw:
            for rect in rects:
                target_page.draw_rect(rect, color=(1, 0, 0), width=0.5)

        # Insert translated text with binary-search font sizing
        for block_index, (rect, translated_text, font_size) in enumerate(
            translated_triplets, start=1
        ):
            shaped = _shape_rtl_text(translated_text)
            max_size = max(6.0, min(font_size, 24.0))

            best_size, inserted = _bisect_font_size(
                target_page, rect, shaped, font_name, font_path, 6.0, max_size,
            )

            if inserted is not None and inserted >= 0:
                # Commit the actual insert (bisect uses overlay=False for probing)
                target_page.insert_textbox(
                    rect, shaped,
                    fontname=font_name, fontfile=font_path,
                    fontsize=best_size, color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_RIGHT,
                )
            else:
                # Expand box downward and retry
                expanded = fitz.Rect(rect.x0, rect.y0, rect.x1, target_page.rect.y1 - 20)
                best_size, inserted = _bisect_font_size(
                    target_page, expanded, shaped, font_name, font_path, 6.0, max_size,
                )
                if inserted is not None and inserted >= 0:
                    target_page.insert_textbox(
                        expanded, shaped,
                        fontname=font_name, fontfile=font_path,
                        fontsize=best_size, color=(0, 0, 0),
                        align=fitz.TEXT_ALIGN_RIGHT,
                    )
                    if verbose:
                        logger.info(
                            "[%s] Page %d block %d: inserted after expanding box with size %.1f",
                            pdf_path.name, page_index, block_index, best_size,
                        )
                else:
                    if verbose:
                        logger.info(
                            "[%s] Page %d block %d: could not fit text even after expanding",
                            pdf_path.name, page_index, block_index,
                        )

        _log_event(log_path, {
            "event": "page_done",
            "page": page_index,
            "blocks": len(translated_triplets),
            "details": block_logs,
        })

        if on_page:
            on_page(page_index, total_pages)

    if side_by_side:
        sbs_doc.save(str(out_path), garbage=4, deflate=True)
        sbs_doc.close()
    else:
        doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()
