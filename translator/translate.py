from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import os
import textwrap

from openai import OpenAI


@dataclass
class TranslatorConfig:
    provider: str
    api_key: str | None
    base_url: str | None
    model: str
    timeout_seconds: int = 60


def _chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    for paragraph in text.split("\n"):
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        chunks.extend(textwrap.wrap(paragraph, max_chars, break_long_words=False))
    return chunks


def translate_texts(texts: Iterable[str], target_lang: str, config: TranslatorConfig) -> List[str]:
    if config.provider == "dummy":
        return list(texts)

    if config.provider != "openai":
        raise ValueError(f"Unknown provider: {config.provider}")

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    results: List[str] = []
    for text in texts:
        chunks = _chunk_text(text)
        translated_chunks: List[str] = []
        for chunk in chunks:
            response = client.responses.create(
                model=config.model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional translator. Translate the user's text to "
                            f"{target_lang}. Preserve line breaks and do not add commentary."
                        ),
                    },
                    {"role": "user", "content": chunk},
                ],
                timeout=config.timeout_seconds,
            )
            output_text = getattr(response, "output_text", "") or ""
            if not output_text and getattr(response, "output", None):
                pieces: List[str] = []
                for item in response.output:
                    content = getattr(item, "content", None)
                    if not content:
                        continue
                    for part in content:
                        part_type = getattr(part, "type", None)
                        if part_type == "output_text":
                            pieces.append(getattr(part, "text", ""))
                output_text = "".join(pieces)
            translated_chunks.append(output_text)
        results.append("\n".join(translated_chunks))
    return results


def load_translator_config(
    provider: str,
    model_override: str | None = None,
    api_key_override: str | None = None,
    base_url_override: str | None = None,
) -> TranslatorConfig:
    return TranslatorConfig(
        provider=provider,
        api_key=api_key_override or os.getenv("OPENAI_API_KEY"),
        base_url=base_url_override or os.getenv("OPENAI_BASE_URL"),
        model=model_override or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
