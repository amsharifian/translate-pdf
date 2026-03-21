from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import fitz  # PyMuPDF
import streamlit as st
import yaml

from translator.font_utils import resolve_font_path
from translator.config import load_config, get_openai_config, get_local_config
from translator.job_queue import (
    init_db, create_job, list_jobs, update_job_status, update_priority, delete_job,
)
from translator.render import extract_page_text_blocks
from translator.translation_memory import TranslationMemory, TMEntry, DEFAULT_TM_PATH


APP_TITLE = "Stormlight Translate"

BUNDLED_FONTS = {
    "Vazirmatn (bundled)": "assets/fonts/Vazirmatn/fonts/ttf/Vazirmatn-Regular.ttf",
    "Noto Naskh Arabic (bundled)": "assets/fonts/NotoNaskhArabic-Regular.ttf",
}

SUPPORTED_LANGUAGES = {
    "Farsi (Persian)": "fa",
    "Arabic": "ar",
    "Turkish": "tr",
    "Urdu": "ur",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Chinese (Simplified)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Hindi": "hi",
    "English": "en",
}

# Rough cost per 1M tokens (input) for common OpenAI models
_TOKEN_PRICING = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "gpt-4-turbo": 10.00,
    "gpt-3.5-turbo": 0.50,
}
_CHARS_PER_TOKEN = 4  # approximate

STATUS_STYLE = {
    "queued": ("🟡", "Queued", "Waiting in the queue for the worker to pick it up."),
    "running": ("🔵", "Running", "The worker is actively translating pages."),
    "paused": ("⏸️", "Paused", "Translation is paused. Resume to continue."),
    "completed": ("✅", "Completed", "All pages translated successfully."),
    "failed": ("❌", "Failed", "An error occurred during translation."),
    "cancelled": ("🚫", "Cancelled", "This job was cancelled by the user."),
}

NOTIF_DB = Path("jobs/.notifications.json")
GLOSSARY_PATH = Path("jobs/glossary.json")
WORKER_PID_FILE = Path("jobs/.worker.pid")
TM_PATH = DEFAULT_TM_PATH


# ── Helpers ─────────────────────────────────────────────────────


def _open_folder(path: Path) -> None:
    import os, platform
    system = platform.system().lower()
    if system == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif system == "windows":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def _parse_page_range(text: str, max_page: int) -> Optional[Set[int]]:
    """Parse '1-5, 10, 15-20' into a set of ints. Returns None on empty input."""
    if not text.strip():
        return None
    pages: Set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            for p in range(int(lo), int(hi) + 1):
                if 1 <= p <= max_page:
                    pages.add(p)
        elif part.isdigit():
            p = int(part)
            if 1 <= p <= max_page:
                pages.add(p)
    return pages or None


def _estimate_tokens(texts: list[str]) -> int:
    return sum(len(t) for t in texts) // _CHARS_PER_TOKEN


def _pdf_page_thumbnail(pdf_path: Path, page_no: int = 0, width: int = 300) -> bytes:
    """Render a page of a PDF as a PNG thumbnail."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_no]
    zoom = width / page.rect.width
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def _load_seen_notifications() -> set:
    if NOTIF_DB.exists():
        try:
            return set(json.loads(NOTIF_DB.read_text(encoding="utf-8")))
        except Exception:
            pass
    return set()


def _save_seen_notification(job_id: str) -> None:
    seen = _load_seen_notifications()
    seen.add(job_id)
    NOTIF_DB.parent.mkdir(parents=True, exist_ok=True)
    NOTIF_DB.write_text(json.dumps(list(seen)), encoding="utf-8")


def _load_global_glossary() -> dict[str, str]:
    if GLOSSARY_PATH.exists():
        try:
            return json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_global_glossary(glossary: dict[str, str]) -> None:
    GLOSSARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOSSARY_PATH.write_text(
        json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _is_worker_running() -> bool:
    """Check if the worker process is still alive via its PID file."""
    import os, signal
    if not WORKER_PID_FILE.exists():
        return False
    try:
        pid = int(WORKER_PID_FILE.read_text().strip())
        os.kill(pid, 0)  # signal 0 = check if alive
        return True
    except (ValueError, OSError):
        WORKER_PID_FILE.unlink(missing_ok=True)
        return False


def _start_worker() -> bool:
    """Launch the worker process and record its PID. Returns True on success."""
    try:
        proc = subprocess.Popen(
            [sys.executable, "scripts/worker.py"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        WORKER_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        WORKER_PID_FILE.write_text(str(proc.pid))
        return True
    except Exception:
        return False


def _ensure_worker() -> None:
    """Auto-start the worker if it's not already running."""
    if not _is_worker_running():
        if _start_worker():
            st.toast("Worker auto-started!", icon="⚙️")


def _save_settings(cfg_path: str, provider: str, model: str, base_url: str, api_key: str) -> None:
    """Persist last-used settings back to config.yml."""
    p = Path(cfg_path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    if not isinstance(data, dict):
        data = {}
    if provider == "ollama":
        data.setdefault("local", {})
        data["local"]["model"] = model
        data["local"]["base_url"] = base_url
        if api_key and api_key != "ollama":
            data["local"]["api_key"] = api_key
    else:
        data.setdefault("openai", {})
        data["openai"]["model"] = model
        data["openai"]["base_url"] = base_url
        # Don't persist the actual API key for security
    p.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


# ── Main ────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.image("assets/stormlight-logo.svg", width=96)
    st.title(APP_TITLE)
    st.caption("Translate PDFs with layout preserved — powered by LLMs.")

    init_db()
    cfg_path = "config.yml"
    cfg = load_config(cfg_path)
    openai_cfg = get_openai_config(cfg)
    local_cfg = get_local_config(cfg)

    # ── Sidebar ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        mode = st.radio("Translation mode", ["OpenAI API", "Local (Ollama)"], index=1)

        if mode == "OpenAI API":
            provider = "openai"
            model = st.text_input("Model", value=openai_cfg.get("model") or "gpt-4o-mini")
            api_key = st.text_input("OpenAI API Key", type="password", value=openai_cfg.get("api_key") or "")
            base_url = st.text_input("Base URL", value=openai_cfg.get("base_url") or "https://api.openai.com/v1")
        else:
            provider = "ollama"
            model = st.text_input("Model", value=local_cfg.get("model") or "qwen3:8b")
            api_key = st.text_input("API Key (optional)", value=local_cfg.get("api_key") or "ollama")
            base_url = st.text_input("Base URL", value=local_cfg.get("base_url") or "http://localhost:11434/v1")

        if api_key:
            masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else api_key
            st.caption(f"Using API key: {masked}")

        st.markdown("---")
        st.header("🔤 Font")
        font_choice = st.selectbox("Font", list(BUNDLED_FONTS.keys()))
        custom_font = st.file_uploader("Or upload a TTF/OTF font", type=["ttf", "otf", "ttc"])

        st.markdown("---")
        st.header("🌐 Language")
        lang_name = st.selectbox("Target language", list(SUPPORTED_LANGUAGES.keys()), index=0)
        target_lang = SUPPORTED_LANGUAGES[lang_name]

        st.markdown("---")
        st.header("🔧 Worker")
        worker_alive = _is_worker_running()
        if worker_alive:
            st.success("Worker is running", icon="✅")
        else:
            st.warning("Worker is not running", icon="⚠️")
        if st.button(
            "🔄 Restart Worker" if worker_alive else "▶ Start Worker",
            use_container_width=True,
        ):
            # Kill existing worker if running
            if worker_alive and WORKER_PID_FILE.exists():
                import os, signal
                try:
                    pid = int(WORKER_PID_FILE.read_text().strip())
                    os.kill(pid, signal.SIGTERM)
                except (ValueError, OSError):
                    pass
                WORKER_PID_FILE.unlink(missing_ok=True)
            if _start_worker():
                st.toast("Worker started!", icon="✅")
                st.rerun()
            else:
                st.error("Could not start worker.")

        if st.button("💾 Save Settings", use_container_width=True, help="Persist current model/URL to config.yml"):
            _save_settings(cfg_path, provider, model, base_url, api_key)
            st.toast("Settings saved to config.yml", icon="💾")

        st.markdown("---")
        st.header("📖 Global Glossary")
        st.caption("These terms are automatically included in every job.")
        global_glossary = _load_global_glossary()
        global_glossary_text = st.text_area(
            "Global glossary (term=translation, one per line)",
            value="\n".join(f"{k}={v}" for k, v in global_glossary.items()),
            height=150,
            key="global_glossary_editor",
        )
        if st.button("💾 Save Glossary", use_container_width=True):
            new_glossary: dict[str, str] = {}
            for line in global_glossary_text.strip().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k and v:
                        new_glossary[k] = v
            _save_global_glossary(new_glossary)
            st.toast(f"Global glossary saved — {len(new_glossary)} term(s)", icon="📖")
        if global_glossary:
            st.caption(f"{len(global_glossary)} term(s) active")

        st.markdown("---")
        st.header("🧠 Translation Memory")
        tm = TranslationMemory(TM_PATH)
        tm_count = tm.count()
        st.caption(f"{tm_count} sentence pair(s) stored")
        if tm_count > 0:
            with st.expander("Browse / manage entries"):
                entries = tm.load_all()
                for i, entry in enumerate(entries):
                    cols = st.columns([5, 1])
                    with cols[0]:
                        st.markdown(f"**EN:** {entry.source[:120]}")
                        st.markdown(f"**→** {entry.target[:120]}")
                        st.caption(f"Lang: {entry.target_lang} · {entry.created_at[:10]}")
                    with cols[1]:
                        if st.button("🗑", key=f"tm-del-{i}", help="Delete this entry"):
                            tm.delete(i)
                            st.toast("Entry deleted", icon="🗑")
                            st.rerun()
            col_exp, col_imp, col_clr = st.columns(3)
            with col_exp:
                st.download_button(
                    "📥 Export TM",
                    data=tm.export_json(),
                    file_name="translation_memory.json",
                    mime="application/json",
                    key="tm-export",
                    use_container_width=True,
                )
            with col_clr:
                if st.button("🗑 Clear TM", key="tm-clear", use_container_width=True):
                    tm.clear()
                    st.toast("Translation memory cleared", icon="🗑")
                    st.rerun()
        tm_upload = st.file_uploader("Import TM (JSON)", type=["json"], key="tm-import-file")
        if tm_upload is not None:
            imported = tm.import_json(tm_upload.read().decode("utf-8"))
            st.toast(f"Imported {imported} entries", icon="📥")
            st.rerun()

    # ── Upload area ─────────────────────────────────────────────
    st.markdown(
        """
        <style>
        [data-testid="stFileUploader"] {
            border: 2px dashed #888;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #4A90D9;
            background: rgba(74,144,217,0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "📎 Drag & drop PDFs here, or click to browse",
        type=["pdf"], accept_multiple_files=True,
    )

    # ── Advanced options ────────────────────────────────────────
    with st.expander("Advanced options"):
        col_a, col_b = st.columns(2)
        with col_a:
            job_name = st.text_input(
                "Job name (optional)",
                placeholder="My translation batch",
            )
            page_range_str = st.text_input(
                "Page range (optional)",
                placeholder="e.g. 1-5, 10, 15-20",
                help="Leave blank to translate all pages.",
            )
            side_by_side = st.checkbox("Side-by-side output", help="Interleave original + translated pages in the output PDF.")
        with col_b:
            glossary_text = st.text_area(
                "Job glossary overrides (one per line: term=translation)",
                placeholder="Machine Learning=یادگیری ماشین\nNeural Network=شبکه عصبی",
                height=120,
                help="These are merged with the global glossary. Per-job entries override global ones.",
            )
            webhook_url = st.text_input(
                "Webhook URL (optional)",
                placeholder="https://hooks.slack.com/...",
                help="POST a JSON payload to this URL when the job finishes.",
            )

    # ── Custom prompt (always visible) ──────────────────────────
    st.markdown("#### 💬 Instructions for the AI")
    custom_prompt = st.text_area(
        "Tell the AI how to translate this job",
        placeholder="e.g. Use formal tone. This is a legal document. Keep brand names in English. Translate numbers to Persian numerals.",
        height=100,
        help="These instructions are added to the system prompt for this job. Use this to control style, tone, terminology, or any domain-specific behavior.",
        label_visibility="collapsed",
    )

    # ── Preview ─────────────────────────────────────────────────
    if uploaded:
        with st.expander("📖 Preview — extracted text blocks", expanded=False):
            for f in uploaded[:3]:  # limit preview to first 3 files
                st.markdown(f"**{f.name}**")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(f.getbuffer())
                tmp.close()
                tmp_path = Path(tmp.name)
                try:
                    blocks_by_page = extract_page_text_blocks(tmp_path, range(1, 4))
                    for pg, blocks in blocks_by_page.items():
                        st.caption(f"Page {pg} — {len(blocks)} text block(s)")
                        for b in blocks[:5]:
                            st.text(b["text"][:200])
                    # Show thumbnail of first page
                    png = _pdf_page_thumbnail(tmp_path, 0, 350)
                    st.image(png, caption=f"{f.name} — page 1", width=350)
                except Exception as exc:
                    st.warning(f"Could not preview: {exc}")
                finally:
                    tmp_path.unlink(missing_ok=True)

        # ── Cost estimate ───────────────────────────────────────
        if provider == "openai" and model in _TOKEN_PRICING:
            total_chars = 0
            for f in uploaded:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(f.getbuffer())
                tmp.close()
                tmp_path = Path(tmp.name)
                try:
                    all_blocks = extract_page_text_blocks(tmp_path)
                    for blocks in all_blocks.values():
                        total_chars += sum(len(b["text"]) for b in blocks)
                except Exception:
                    pass
                finally:
                    tmp_path.unlink(missing_ok=True)
            tokens = total_chars // _CHARS_PER_TOKEN
            cost = (tokens / 1_000_000) * _TOKEN_PRICING[model]
            st.info(f"💰 Estimated: ~{tokens:,} tokens → **${cost:.4f}** (input only, {model} pricing)")

    # ── Submit ──────────────────────────────────────────────────
    submit = st.button("🚀 Submit Job", type="primary", use_container_width=True)

    if submit:
        if not uploaded:
            st.error("Please upload at least one PDF.")
            return
        if provider == "openai" and not api_key:
            st.error("OpenAI API key is required for OpenAI mode.")
            return

        # Resolve font
        font_path = None
        if custom_font is not None:
            tmp_font = tempfile.NamedTemporaryFile(delete=False, suffix=Path(custom_font.name).suffix)
            tmp_font.write(custom_font.getbuffer())
            tmp_font.flush()
            tmp_font.close()
            font_path = tmp_font.name
        else:
            font_path = resolve_font_path(BUNDLED_FONTS.get(font_choice, ""))
        if not font_path:
            st.error("Font not found. Choose a bundled font or upload one.")
            return

        # Parse glossary — merge global + per-job
        glossary: dict[str, str] = _load_global_glossary()
        for line in glossary_text.strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                glossary[k.strip()] = v.strip()  # per-job overrides global

        job_id = str(uuid.uuid4())
        job_dir = Path("jobs") / job_id
        upload_dir = job_dir / "uploads"
        output_dir = job_dir / "outputs"
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_paths: List[str] = []
        for f in uploaded:
            target = upload_dir / f.name
            target.write_bytes(f.getbuffer())
            input_paths.append(str(target))

        file_names = ", ".join(f.name for f in uploaded)
        auto_name = job_name.strip() if job_name.strip() else f"{len(uploaded)} file(s) — {file_names[:60]}"

        job = {
            "id": job_id,
            "job_name": auto_name,
            "status": "queued",
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "font_path": font_path,
            "input_files": input_paths,
            "output_dir": str(output_dir),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "progress_current": 0,
            "progress_total": 0,
            "target_lang": target_lang,
            "glossary": glossary,
            "page_range": list(sorted(_parse_page_range(page_range_str, 9999) or [])) or None,
            "side_by_side": side_by_side,
            "webhook_url": webhook_url.strip() or None,
            "custom_prompt": custom_prompt.strip(),
        }
        create_job(job)
        _ensure_worker()
        st.toast(f"✅ Job queued — {file_names}", icon="📄")
        st.rerun()

    # ── Job Queue ───────────────────────────────────────────────
    st.markdown("---")
    _render_job_queue()


# ── Job queue fragment (auto-refreshing) ────────────────────────


@st.fragment(run_every=5)
def _render_job_queue() -> None:
    st.header("Job Queue")

    # Load persistent notification set (survives page refresh)
    seen_notifs = _load_seen_notifications()

    jobs = list_jobs()
    if not jobs:
        st.info("No jobs yet — upload a PDF and click **Submit Job** to get started.")
        return

    active_jobs = [j for j in jobs if j["status"] in ("queued", "running", "paused")]
    finished_jobs = [j for j in jobs if j["status"] not in ("queued", "running", "paused")]

    def _render_job_card(job: dict) -> None:
        display_name = job.get("job_name") or "Translation"
        status = job["status"]
        icon, label, tooltip = STATUS_STYLE.get(status, ("❓", status.title(), ""))
        priority = job.get("priority") or 0

        with st.container(border=True):
            # ── Header row ──
            head_left, head_right = st.columns([4, 1])
            with head_left:
                st.markdown(f"### {icon} {display_name}")
            with head_right:
                st.markdown(
                    f"<div style='text-align:right;padding-top:8px'>"
                    f"<span title='{tooltip}' style='"
                    f"background:var(--secondary-background-color, #e8e8e8);"
                    f"border-radius:8px;padding:2px 10px;"
                    f"font-size:0.85em;cursor:help'>{label}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── Meta line ──
            created = job.get("created_at", "")
            meta_parts = [
                f"**Provider:** {job.get('provider', '–')}",
                f"**Model:** {job.get('model', '–')}",
                f"**Lang:** {job.get('target_lang', 'fa')}",
            ]
            if priority != 0:
                meta_parts.append(f"**Priority:** {priority}")
            if created:
                meta_parts.append(f"**Created:** {created}")
            st.caption("  ·  ".join(meta_parts))

            # ── Progress bar ──
            total = job.get("progress_total") or 0
            current = job.get("progress_current") or 0
            if total > 0:
                st.progress(min(current / total, 1.0))
                st.caption(f"Page {current} of {total}")
            elif status == "running":
                st.progress(0.0)
                st.caption("Starting… waiting for page count")
            elif status == "queued":
                st.progress(0.0)
                st.caption("Waiting in queue…")

            # ── Error message ──
            if job.get("error"):
                st.error(job["error"])

            # ── Action buttons ──
            btns: list = []
            if status in ("queued", "running"):
                btns = ["⏸ Pause", "✕ Cancel", "▲", "▼", "📂 Outputs"]
            elif status == "paused":
                btns = ["▶ Resume", "✕ Cancel", "▲", "▼", "📂 Outputs"]
            elif status == "completed":
                btns = ["📂 Outputs", "🗑 Remove"]
            else:  # failed / cancelled
                btns = ["🗑 Remove"]

            btn_cols = st.columns(len(btns))
            for idx, lbl in enumerate(btns):
                with btn_cols[idx]:
                    if lbl == "⏸ Pause":
                        if st.button(lbl, key=f"pause-{job['id']}", use_container_width=True):
                            update_job_status(job["id"], "paused")
                            st.rerun()
                    elif lbl == "▶ Resume":
                        if st.button(lbl, key=f"resume-{job['id']}", use_container_width=True):
                            update_job_status(job["id"], "running")
                            st.rerun()
                    elif lbl == "✕ Cancel":
                        if st.button(lbl, key=f"cancel-{job['id']}", use_container_width=True):
                            update_job_status(job["id"], "cancelled")
                            st.rerun()
                    elif lbl == "▲":
                        if st.button(lbl, key=f"prio-up-{job['id']}", help="Increase priority", use_container_width=True):
                            update_priority(job["id"], priority + 1)
                            st.rerun()
                    elif lbl == "▼":
                        if st.button(lbl, key=f"prio-down-{job['id']}", help="Decrease priority", use_container_width=True):
                            update_priority(job["id"], priority - 1)
                            st.rerun()
                    elif lbl == "📂 Outputs":
                        output_dir = Path(job["output_dir"])
                        if output_dir.exists():
                            if st.button(lbl, key=f"open-{job['id']}", use_container_width=True):
                                _open_folder(output_dir)
                    elif lbl == "🗑 Remove":
                        if st.button(lbl, key=f"delete-{job['id']}", use_container_width=True):
                            delete_job(job["id"])
                            st.rerun()

            # ── Downloads + output preview ──
            output_dir = Path(job["output_dir"])
            if status == "completed" and output_dir.exists():
                pdf_files = sorted(output_dir.glob("*.pdf"))
                if pdf_files:
                    st.markdown("**Downloads:**")
                    dl_cols = st.columns(min(len(pdf_files), 4))
                    for i, out_file in enumerate(pdf_files):
                        with dl_cols[i % len(dl_cols)]:
                            st.download_button(
                                label=f"📄 {out_file.name}",
                                data=out_file.read_bytes(),
                                file_name=out_file.name,
                                mime="application/pdf",
                                key=f"dl-{job['id']}-{out_file.name}",
                                use_container_width=True,
                            )
                    # Show thumbnail of first output
                    try:
                        png = _pdf_page_thumbnail(pdf_files[0], 0, 300)
                        st.image(png, caption="Output preview — page 1", width=300)
                    except Exception:
                        pass

            # ── Job logs viewer ──
            log_path = output_dir / (Path(job["output_dir"]).name + ".log.jsonl") if output_dir.exists() else None
            # Try to find any .log.jsonl in the output dir
            if output_dir.exists():
                jsonl_files = list(output_dir.parent.glob("**/*.log.jsonl"))

                # ── Review & Feedback (Translation Memory) ──
                if jsonl_files and status == "completed":
                    with st.expander("✏️ Review translations & save feedback"):
                        st.caption(
                            "Correct translations below, then click **Save to TM** to "
                            "teach the AI your preferred phrasing. Saved pairs are used "
                            "as few-shot examples in future translations."
                        )
                        tm = TranslationMemory(TM_PATH)
                        # Collect all blocks from all log files
                        review_blocks: list[dict] = []
                        for lf in jsonl_files[:1]:
                            for line in lf.read_text(encoding="utf-8").splitlines():
                                try:
                                    log_entry = json.loads(line)
                                    pg = log_entry.get("page", "?")
                                    for d in log_entry.get("details", []):
                                        orig = d.get("original", "").strip()
                                        trans = d.get("translated", "").strip()
                                        if orig and trans:
                                            review_blocks.append({
                                                "page": pg,
                                                "original": orig,
                                                "translated": trans,
                                            })
                                except (json.JSONDecodeError, KeyError):
                                    continue

                        if not review_blocks:
                            st.info("No translation blocks found in job logs.")
                        else:
                            tgt_lang = job.get("target_lang", "fa")
                            for bi, blk in enumerate(review_blocks):
                                st.markdown(f"---\n**Page {blk['page']}** — Block {bi + 1}")
                                st.text_area(
                                    "Original",
                                    value=blk["original"],
                                    height=80,
                                    disabled=True,
                                    key=f"rv-orig-{job['id']}-{bi}",
                                )
                                corrected = st.text_area(
                                    "Translation (edit to correct)",
                                    value=blk["translated"],
                                    height=80,
                                    key=f"rv-trans-{job['id']}-{bi}",
                                )
                                btn_c1, btn_c2, btn_c3 = st.columns(3)
                                with btn_c1:
                                    if st.button(
                                        "✅ Accept as-is",
                                        key=f"rv-accept-{job['id']}-{bi}",
                                        use_container_width=True,
                                        help="Save the AI output as a positive example",
                                    ):
                                        tm.add(TMEntry(
                                            source=blk["original"],
                                            target=blk["translated"],
                                            target_lang=tgt_lang,
                                            source_job=job["id"],
                                        ))
                                        st.toast(f"Saved to TM (block {bi+1})", icon="✅")
                                with btn_c2:
                                    if st.button(
                                        "💾 Save correction",
                                        key=f"rv-save-{job['id']}-{bi}",
                                        use_container_width=True,
                                        help="Save your corrected version to TM",
                                    ):
                                        tm.add(TMEntry(
                                            source=blk["original"],
                                            target=corrected,
                                            target_lang=tgt_lang,
                                            source_job=job["id"],
                                        ))
                                        st.toast(f"Correction saved to TM (block {bi+1})", icon="💾")
                                with btn_c3:
                                    st.button(
                                        "⏭ Skip",
                                        key=f"rv-skip-{job['id']}-{bi}",
                                        use_container_width=True,
                                    )

                # ── Log viewer ──
                if jsonl_files:
                    with st.expander("📋 Translation logs"):
                        for lf in jsonl_files[:1]:
                            for line in lf.read_text(encoding="utf-8").splitlines()[-20:]:
                                try:
                                    entry = json.loads(line)
                                    pg = entry.get("page", "?")
                                    blocks = entry.get("blocks", 0)
                                    st.caption(f"Page {pg}: {blocks} block(s) translated")
                                    for d in entry.get("details", [])[:3]:
                                        st.text(f"  EN: {d.get('original', '')[:80]}")
                                        st.text(f"  →  {d.get('translated', '')[:80]}")
                                except (json.JSONDecodeError, KeyError):
                                    continue

            # ── Font details (collapsed) ──
            font = job.get("font_path") or ""
            if font:
                with st.expander("Font details"):
                    st.text(font)

        # ── Notifications (persistent — no balloons) ──
        if status in ("completed", "failed", "cancelled"):
            if job["id"] not in seen_notifs:
                if status == "completed":
                    st.toast(f"✅ Job completed: {display_name}", icon="🎉")
                elif status == "failed":
                    st.toast(f"❌ Job failed: {display_name}", icon="⚠️")
                else:
                    st.toast(f"🚫 Job cancelled: {display_name}")
                _save_seen_notification(job["id"])
                seen_notifs.add(job["id"])

    # ── Active jobs ──
    if active_jobs:
        st.subheader(f"Active ({len(active_jobs)})")
        for job in active_jobs:
            _render_job_card(job)

    # ── History ──
    if finished_jobs:
        with st.expander(f"History — {len(finished_jobs)} finished job(s)", expanded=False):
            for job in finished_jobs:
                _render_job_card(job)


if __name__ == "__main__":
    main()
