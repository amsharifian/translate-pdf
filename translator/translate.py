from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, List, TYPE_CHECKING
import os
import textwrap

from openai import OpenAI

if TYPE_CHECKING:
    from translator.translation_memory import TranslationMemory

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 1.0  # seconds

# ── Well-known HuggingFace models for English→Farsi ────────────
HUGGINGFACE_MODELS = {
    # -- NLLB (No Language Left Behind) --------------------------------
    "facebook/nllb-200-distilled-600M": {
        "label": "NLLB-200 600M (recommended)",
        "type": "nllb",
        "src_lang": "eng_Latn",
        "tgt_lang": "pes_Arab",
    },
    "facebook/nllb-200-distilled-1.3B": {
        "label": "NLLB-200 1.3B (best quality)",
        "type": "nllb",
        "src_lang": "eng_Latn",
        "tgt_lang": "pes_Arab",
    },
    "facebook/nllb-200-3.3B": {
        "label": "NLLB-200 3.3B (highest quality, slow)",
        "type": "nllb",
        "src_lang": "eng_Latn",
        "tgt_lang": "pes_Arab",
    },
    # -- Helsinki-NLP Opus-MT (MarianMT) ------------------------------
    "Helsinki-NLP/opus-mt-en-fa": {
        "label": "Opus-MT EN→FA (lightweight)",
        "type": "opus",
    },
    "Helsinki-NLP/opus-mt-tc-big-en-fa": {
        "label": "Opus-MT Big EN→FA (better quality)",
        "type": "opus",
    },
    # -- M2M-100 (Many-to-Many) ---------------------------------------
    "facebook/m2m100_418M": {
        "label": "M2M-100 418M (fast multilingual)",
        "type": "m2m100",
        "src_lang": "en",
        "tgt_lang": "fa",
    },
    "facebook/m2m100_1.2B": {
        "label": "M2M-100 1.2B (quality multilingual)",
        "type": "m2m100",
        "src_lang": "en",
        "tgt_lang": "fa",
    },
    # -- mBART-50 ------------------------------------------------------
    "facebook/mbart-large-50-many-to-many-mmt": {
        "label": "mBART-50 multilingual",
        "type": "mbart",
        "src_lang": "en_XX",
        "tgt_lang": "fa_IR",
    },
}


@dataclass
class TranslatorConfig:
    provider: str
    api_key: str | None
    base_url: str | None
    model: str
    timeout_seconds: int = 60
    target_lang: str = "fa"
    glossary: dict[str, str] = field(default_factory=dict)
    translation_memory: TranslationMemory | None = None
    custom_prompt: str = ""


def _chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """Split text on sentence/paragraph boundaries within *max_chars*."""
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    # Split into paragraphs first
    paragraphs = text.split("\n")
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        # Flush current if non-empty
        if current:
            chunks.append(current)
            current = ""
        # If a single paragraph > max_chars, split on sentence boundaries
        if len(para) > max_chars:
            sentences = re.split(r'(?<=[.!?؟،])\s+', para)
            for sent in sentences:
                if not current:
                    current = sent
                elif len(current) + 1 + len(sent) <= max_chars:
                    current = current + " " + sent
                else:
                    chunks.append(current)
                    current = sent
        else:
            current = para

    if current:
        chunks.append(current)
    return chunks if chunks else [text]


def _call_llm_with_retry(client: OpenAI, model: str, messages: list, timeout: int) -> str:
    """Call the LLM API with exponential backoff on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.create(
                model=model,
                input=messages,
                timeout=timeout,
            )
            output_text = getattr(response, "output_text", "") or ""
            if not output_text and getattr(response, "output", None):
                pieces: List[str] = []
                for item in response.output:
                    content = getattr(item, "content", None)
                    if not content:
                        continue
                    for part in content:
                        if getattr(part, "type", None) == "output_text":
                            pieces.append(getattr(part, "text", ""))
                output_text = "".join(pieces)
            return output_text
        except Exception as exc:
            is_last = attempt == MAX_RETRIES - 1
            if is_last:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
    return ""  # unreachable


def _build_system_prompt(
    target_lang: str,
    glossary: dict[str, str],
    tm_examples: list[tuple] | None = None,
    custom_prompt: str = "",
) -> str:
    prompt = (
        "You are a professional translator. Translate the user's text to "
        f"{target_lang}. Preserve line breaks and do not add commentary."
    )
    if custom_prompt:
        prompt += f"\n\nAdditional instructions from the user:\n{custom_prompt}"
    if glossary:
        entries = "\n".join(f"  {k} → {v}" for k, v in glossary.items())
        prompt += f"\n\nAlways use these term translations:\n{entries}"
    if tm_examples:
        prompt += "\n\nHere are examples of preferred translations — match their style and phrasing:"
        for i, (entry, _score) in enumerate(tm_examples, 1):
            prompt += f"\n\nExample {i}:\nSource: \"{entry.source}\"\nTranslation: \"{entry.target}\""
    return prompt


# ── HuggingFace local model support ─────────────────────────────
_hf_model_cache: dict[str, tuple] = {}


def _get_hf_model(model_name: str, token: str | None = None):
    """Return a cached (tokenizer, model, model_info) tuple. Downloads on first use."""
    if model_name in _hf_model_cache:
        return _hf_model_cache[model_name]

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_info = HUGGINGFACE_MODELS.get(model_name, {})
    model_type = model_info.get("type", "nllb")
    hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None

    logger.info("Loading HuggingFace model %s (type=%s) — first run downloads weights…", model_name, model_type)

    if model_type == "mbart":
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name, token=hf_token)
        tokenizer.src_lang = model_info.get("src_lang", "en_XX")
        model = MBartForConditionalGeneration.from_pretrained(model_name, token=hf_token)
    elif model_type == "m2m100":
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        tokenizer = M2M100Tokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.src_lang = model_info.get("src_lang", "en")
        model = M2M100ForConditionalGeneration.from_pretrained(model_name, token=hf_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=hf_token)

    _hf_model_cache[model_name] = (tokenizer, model, model_info)
    return tokenizer, model, model_info


def _translate_texts_huggingface(texts: List[str], config: TranslatorConfig) -> List[str]:
    """Translate texts locally using a HuggingFace model (no pipeline task needed)."""
    tokenizer, model, model_info = _get_hf_model(config.model, token=config.api_key)
    model_type = model_info.get("type", "nllb")
    prefix = model_info.get("prefix", "")

    # Set forced_bos_token_id for target language where required
    gen_kwargs: dict = {"max_length": 512}
    tgt_lang = model_info.get("tgt_lang")
    if tgt_lang and model_type == "m2m100":
        gen_kwargs["forced_bos_token_id"] = tokenizer.get_lang_id(tgt_lang)
    elif tgt_lang and model_type in ("nllb", "mbart"):
        tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        if tgt_id != tokenizer.unk_token_id:
            gen_kwargs["forced_bos_token_id"] = tgt_id

    results: List[str] = []
    for text in texts:
        chunks = _chunk_text(text, max_chars=400)
        translated_chunks: List[str] = []
        for chunk in chunks:
            input_text = prefix + chunk if prefix else chunk
            try:
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                output_ids = model.generate(**inputs, **gen_kwargs)
                translated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            except Exception as exc:
                logger.error("HuggingFace translation failed for chunk: %s", exc)
                raise RuntimeError(f"HuggingFace translation failed: {exc}") from exc
            translated_chunks.append(translated)
        results.append("\n".join(translated_chunks))
    return results


# ── Google Translate support (free, no API key) ─────────────────

def _translate_texts_google(texts: List[str], target_lang: str) -> List[str]:
    """Translate texts using the free Google Translate via deep-translator."""
    from deep_translator import GoogleTranslator

    results: List[str] = []
    for text in texts:
        chunks = _chunk_text(text, max_chars=4500)
        translated_chunks: List[str] = []
        for chunk in chunks:
            for attempt in range(3):
                try:
                    translated = GoogleTranslator(source="en", target=target_lang).translate(chunk)
                    translated_chunks.append(translated or chunk)
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise RuntimeError(f"Google Translate failed: {exc}") from exc
                    logger.warning("Google Translate attempt %d/3 failed: %s", attempt + 1, exc)
                    time.sleep(1.0 * (2 ** attempt))
        results.append("\n".join(translated_chunks))
    return results


def translate_texts(texts: Iterable[str], target_lang: str, config: TranslatorConfig) -> List[str]:
    if config.provider == "dummy":
        return list(texts)

    if config.provider == "huggingface":
        return _translate_texts_huggingface(list(texts), config)

    if config.provider == "google":
        return _translate_texts_google(list(texts), target_lang)

    if config.provider not in ("openai", "ollama"):
        raise ValueError(f"Unknown provider: {config.provider}")

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    tm = config.translation_memory
    results: List[str] = []
    for text in texts:
        chunks = _chunk_text(text)
        translated_chunks: List[str] = []
        for chunk in chunks:
            # Query TM for relevant examples per chunk
            tm_examples = None
            if tm is not None:
                tm_examples = tm.search(chunk, target_lang) or None
            system_prompt = _build_system_prompt(target_lang, config.glossary, tm_examples, config.custom_prompt)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ]
            output_text = _call_llm_with_retry(
                client, config.model, messages, config.timeout_seconds,
            )
            translated_chunks.append(output_text)
        results.append("\n".join(translated_chunks))
    return results


def load_translator_config(
    provider: str,
    model_override: str | None = None,
    api_key_override: str | None = None,
    base_url_override: str | None = None,
) -> TranslatorConfig:
    if provider == "huggingface":
        return TranslatorConfig(
            provider=provider,
            api_key=api_key_override or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
            base_url=None,
            model=model_override or "facebook/nllb-200-distilled-600M",
        )
    if provider == "google":
        return TranslatorConfig(
            provider=provider,
            api_key=None,
            base_url=None,
            model="google",
        )
    if provider == "ollama":
        return TranslatorConfig(
            provider=provider,
            api_key=api_key_override or os.getenv("OLLAMA_API_KEY", "ollama"),
            base_url=base_url_override or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model=model_override or os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        )
    return TranslatorConfig(
        provider=provider,
        api_key=api_key_override or os.getenv("OPENAI_API_KEY"),
        base_url=base_url_override or os.getenv("OPENAI_BASE_URL"),
        model=model_override or os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
    )
