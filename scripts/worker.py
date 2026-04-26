from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import fitz
import httpx

from translator.job_queue import (
    init_db,
    fetch_next_job,
    update_job_status,
    update_progress,
    update_status_detail,
    get_job_status,
)
from translator.render import translate_pdf_preserve_layout
from translator.translate import load_translator_config, translate_texts
from translator.translation_memory import TranslationMemory

logger = logging.getLogger(__name__)


def _count_pages(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        with fitz.open(str(p)) as doc:
            total += len(doc)
    return total


def _fire_webhook(url: str, payload: dict) -> None:
    """POST a JSON payload to the webhook URL. Best-effort, no retries."""
    try:
        httpx.post(url, json=payload, timeout=10)
    except Exception as exc:
        logger.warning("Webhook delivery failed (%s): %s", url, exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_db()
    print("Stormlight worker started. Waiting for jobs...")
    while True:
        job = fetch_next_job()
        if not job:
            time.sleep(2)
            continue

        job_id = job["id"]
        try:
            class JobCancelled(Exception):
                pass

            if get_job_status(job_id) == "cancelled":
                continue
            update_job_status(job_id, "running")
            update_status_detail(job_id, "Starting…")

            input_files = [Path(p) for p in json.loads(job["input_files"])]
            output_dir = Path(job["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            total_pages = _count_pages(input_files)
            current_pages = 0
            update_progress(job_id, current_pages, total_pages)

            target_lang = job.get("target_lang") or "fa"
            glossary: dict[str, str] = {}
            if job.get("glossary"):
                try:
                    glossary = json.loads(job["glossary"]) if isinstance(job["glossary"], str) else job["glossary"]
                except (json.JSONDecodeError, TypeError):
                    pass

            page_range_raw = job.get("page_range")
            page_range_set = None
            if page_range_raw:
                try:
                    pages_list = json.loads(page_range_raw) if isinstance(page_range_raw, str) else page_range_raw
                    if isinstance(pages_list, list):
                        page_range_set = set(pages_list)
                except (json.JSONDecodeError, TypeError):
                    pass

            side_by_side = bool(job.get("side_by_side"))
            ocr_enabled = bool(job.get("ocr_enabled"))
            ocr_lang = (job.get("ocr_lang") or "eng").strip() or "eng"

            font_size_override = job.get("font_size")  # None = auto
            if font_size_override is not None:
                font_size_override = float(font_size_override)

            config = load_translator_config(
                provider=job["provider"],
                model_override=job.get("model"),
                api_key_override=job.get("api_key"),
                base_url_override=job.get("base_url"),
            )
            config.glossary = glossary
            config.custom_prompt = job.get("custom_prompt") or ""
            config.translation_memory = TranslationMemory()

            for in_path in input_files:
                if get_job_status(job_id) == "cancelled":
                    update_job_status(job_id, "cancelled")
                    break
                # Unique output name: include short job ID to avoid collisions
                short_id = job_id[:8]
                out_path = output_dir / f"{in_path.stem}.{short_id}.fa.pdf"

                # Remove stale log so resume logic doesn't skip pages
                log_path = out_path.with_suffix(".log.jsonl")
                if log_path.exists():
                    log_path.unlink()

                def on_pause() -> None:
                    while True:
                        status = get_job_status(job_id)
                        if status == "paused":
                            time.sleep(1)
                            continue
                        if status == "cancelled":
                            raise JobCancelled()
                        return

                def on_page(cur: int, total: int) -> None:
                    nonlocal current_pages
                    current_pages += 1
                    update_progress(job_id, current_pages, total_pages)

                def on_phase(cur: int, total: int, phase: str) -> None:
                    if phase == "ocr":
                        update_status_detail(job_id, f"OCR page {cur}/{total}")
                    elif phase == "translating":
                        update_status_detail(job_id, f"Translating page {cur}/{total}")
                    else:
                        update_status_detail(job_id, f"Processing page {cur}/{total}")

                translate_pdf_preserve_layout(
                    in_path,
                    out_path,
                    job["font_path"],
                    lambda texts: translate_texts(texts, target_lang, config),
                    verbose=True,
                    on_page=on_page,
                    on_pause=on_pause,
                    page_range=page_range_set,
                    side_by_side=side_by_side,
                    font_size_override=font_size_override,
                    enable_ocr=ocr_enabled,
                    ocr_lang=ocr_lang,
                    on_phase=on_phase,
                )

            # Clean up resume logs from output directory
            for logf in output_dir.glob("*.log.jsonl"):
                logf.unlink(missing_ok=True)

            final_status = get_job_status(job_id)
            if final_status != "cancelled":
                update_status_detail(job_id, None)
                update_job_status(job_id, "completed")
                final_status = "completed"

            # Fire webhook if configured
            webhook_url = job.get("webhook_url")
            if webhook_url:
                _fire_webhook(webhook_url, {
                    "job_id": job_id,
                    "job_name": job.get("job_name"),
                    "status": final_status,
                })

        except JobCancelled:
            update_status_detail(job_id, None)
            update_job_status(job_id, "cancelled")
            webhook_url = job.get("webhook_url")
            if webhook_url:
                _fire_webhook(webhook_url, {"job_id": job_id, "status": "cancelled"})
        except Exception as exc:
            update_status_detail(job_id, None)
            update_job_status(job_id, "failed", error=str(exc))
            webhook_url = job.get("webhook_url")
            if webhook_url:
                _fire_webhook(webhook_url, {"job_id": job_id, "status": "failed", "error": str(exc)})


if __name__ == "__main__":
    main()
