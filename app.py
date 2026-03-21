from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import streamlit as st

from translator.font_utils import resolve_font_path
from translator.render import translate_pdf_preserve_layout
from translator.translate import load_translator_config, translate_texts
from translator.config import load_config, get_openai_config, get_local_config


APP_TITLE = "Stormlight Translate (PDF to Farsi)"

BUNDLED_FONTS = {
    "Vazirmatn (bundled)": "assets/fonts/Vazirmatn/fonts/ttf/Vazirmatn-Regular.ttf",
    "Noto Naskh Arabic (bundled)": "assets/fonts/NotoNaskhArabic-Regular.ttf",
}


def _save_uploads(files) -> List[Path]:
    saved: List[Path] = []
    for f in files:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
        tmp.write(f.getbuffer())
        tmp.flush()
        tmp.close()
        saved.append(Path(tmp.name))
    return saved


def _build_config(provider: str, model: str, api_key: str, base_url: str):
    api_key = api_key.strip() if api_key else None
    base_url = base_url.strip() if base_url else None
    model = model.strip() if model else None
    return load_translator_config(
        provider=provider,
        model_override=model,
        api_key_override=api_key,
        base_url_override=base_url,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.image("assets/stormlight-logo.svg", width=96)
    st.title(APP_TITLE)
    st.write("Translate PDFs to Farsi with layout preserved.")

    cfg = load_config("config.yml")
    openai_cfg = get_openai_config(cfg)
    local_cfg = get_local_config(cfg)

    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Translation mode",
            ["OpenAI API", "Local (Ollama)"],
            index=1,
        )

        st.header("Model")
        if mode == "OpenAI API":
            provider = "openai"
            model = st.text_input("Model", value=openai_cfg.get("model") or "gpt-4o-mini")
            api_key = st.text_input("OpenAI API Key", type="password", value=openai_cfg.get("api_key") or "")
            base_url = st.text_input("Base URL", value=openai_cfg.get("base_url") or "https://api.openai.com/v1")
        elif mode == "Local (Ollama)":
            provider = "ollama"
            model = st.text_input("Model", value=local_cfg.get("model") or "qwen3:8b")
            api_key = st.text_input("API Key (optional)", value=local_cfg.get("api_key") or "ollama")
            base_url = st.text_input("Base URL", value=local_cfg.get("base_url") or "http://localhost:11434/v1")
        if api_key:
            masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else api_key
            st.caption(f"Using API key: {masked}")

        st.header("Font")
        font_choice = st.selectbox("Font", list(BUNDLED_FONTS.keys()))
        custom_font = st.file_uploader("Or upload a TTF/OTF font", type=["ttf", "otf", "ttc"])

        st.header("Output")
        out_prefix = st.text_input("Output filename prefix", value="translated_")

    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    if st.button("Translate", type="primary"):
        if not uploaded:
            st.error("Please upload at least one PDF.")
            return

        if provider == "openai" and not api_key:
            st.error("OpenAI API key is required for OpenAI mode.")
            return

        # Resolve font path
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

        config = _build_config(provider, model, api_key, base_url)

        saved_inputs = _save_uploads(uploaded)
        results: List[tuple[str, Path]] = []

        for idx, in_path in enumerate(saved_inputs, start=1):
            st.write(f"Processing {idx}/{len(saved_inputs)}: {uploaded[idx-1].name}")
            progress = st.progress(0)
            status = st.empty()

            out_file = Path(tempfile.gettempdir()) / f"{out_prefix}{uploaded[idx-1].name}"

            def on_page(current: int, total: int) -> None:
                if total > 0:
                    progress.progress(min(current / total, 1.0))
                    status.write(f"Pages: {current}/{total}")

            try:
                translate_pdf_preserve_layout(
                    in_path,
                    out_file,
                    font_path,
                    lambda texts: translate_texts(texts, "fa", config),
                    on_page=on_page,
                )
            except Exception as exc:
                st.error(f"Failed on {uploaded[idx-1].name}: {exc}")
                continue

            results.append((uploaded[idx-1].name, out_file))
            progress.progress(1.0)
            status.write("Done")

        if results:
            st.success("Translation complete. Download below:")
            for original_name, path in results:
                data = path.read_bytes()
                st.download_button(
                    label=f"Download {original_name}",
                    data=data,
                    file_name=f"{out_prefix}{original_name}",
                    mime="application/pdf",
                )


if __name__ == "__main__":
    main()
