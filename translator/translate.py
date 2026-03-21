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
) -> str:
    prompt = (
        "You are a professional translator. Translate the user's text to "
        f"{target_lang}. Preserve line breaks and do not add commentary."
    )
    if glossary:
        entries = "\n".join(f"  {k} → {v}" for k, v in glossary.items())
        prompt += f"\n\nAlways use these term translations:\n{entries}"
    if tm_examples:
        prompt += "\n\nHere are examples of preferred translations — match their style and phrasing:"
        for i, (entry, _score) in enumerate(tm_examples, 1):
            prompt += f"\n\nExample {i}:\nSource: \"{entry.source}\"\nTranslation: \"{entry.target}\""
    return prompt


def translate_texts(texts: Iterable[str], target_lang: str, config: TranslatorConfig) -> List[str]:
    if config.provider == "dummy":
        return list(texts)

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
            system_prompt = _build_system_prompt(target_lang, config.glossary, tm_examples)
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
        model=model_override or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
