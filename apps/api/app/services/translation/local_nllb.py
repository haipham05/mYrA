"""Small, offline adapter for English-to-Vietnamese NLLB translation."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

MODEL_ID = "facebook/nllb-200-distilled-600M"
SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "vie_Latn"
MAX_INPUT_TOKENS = 480

# Preserve layout/scientific markers outside the model call so they cannot be
# translated, normalized, or dropped by the tokenizer/model.
_PROTECTED_MARKER = re.compile(
    r"\[\[.*?\]\]|\{v\d+\}|\{[A-Za-z][\w.-]*\}|%[sd]|</?b\d+>|"
    r"</?style\b[^>]*>|%%.*?%%"
)
_TEXT_PARTS = re.compile(r"\S+\s*|\s+")


class TranslationInputTooLongError(ValueError):
    """Raised when one indivisible token cannot fit in the model input."""


class LocalNllbTranslator:
    """Translate text using the locally cached NLLB model on CUDA when available.

    Model and tokenizer loading are lazy, so importing this module does not
    affect normal API/RAG startup. ``revision`` should be pinned by the caller
    for reproducible jobs.
    """

    def __init__(
        self,
        *,
        revision: str | None = None,
        loader: Callable[..., tuple[Any, Any]] | None = None,
    ) -> None:
        self.revision = revision
        self._loader = loader
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._target_token_id: int | None = None
        self._device = "cpu"

    def translate(self, text: str) -> str:
        """Translate prose, retaining protected markers and original spacing."""
        if not text:
            return text

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._target_token_id is not None

        output: list[str] = []
        cursor = 0
        for marker in _PROTECTED_MARKER.finditer(text):
            output.extend(self._translate_text(text[cursor : marker.start()]))
            output.append(marker.group(0))
            cursor = marker.end()
        output.extend(self._translate_text(text[cursor:]))
        return "".join(output)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._loader is not None:
            self._tokenizer, self._model = self._loader(
                MODEL_ID,
                revision=self.revision,
                source_language=SOURCE_LANGUAGE,
            )
        else:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_source = os.getenv("MYRA_TRANSLATION_MODEL_PATH", MODEL_ID)
            kwargs: dict[str, Any] = {"local_files_only": True}
            if self.revision and model_source == MODEL_ID:
                kwargs["revision"] = self.revision
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                src_lang=SOURCE_LANGUAGE,
                **kwargs,
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_source, **kwargs)

        self._model.to(self._device)
        self._model.eval()
        self._target_token_id = self._tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE)

    def get_rich_text_left_placeholder(self, placeholder_id: int | str) -> str:
        return f"<b{placeholder_id}>"

    def get_rich_text_right_placeholder(self, placeholder_id: int | str) -> str:
        return f"</b{placeholder_id}>"

    def get_formular_placeholder(self, placeholder_id: int | str) -> str:
        return self.get_rich_text_left_placeholder(placeholder_id)

    def _translate_text(self, text: str) -> list[str]:
        if not text.strip():
            return [text] if text else []
        chunks = self._split_to_fit(text)
        return [self._generate(chunk) for chunk in chunks]

    def _input_length(self, text: str) -> int:
        encoded = self._tokenizer(text, add_special_tokens=True, truncation=False)
        ids = encoded["input_ids"]
        # Tokenizers may return a one-item batch for unusual configurations.
        return len(ids[0] if ids and isinstance(ids[0], list) else ids)

    def _split_to_fit(self, text: str) -> list[str]:
        if self._input_length(text) <= MAX_INPUT_TOKENS:
            return [text]

        chunks: list[str] = []
        current = ""
        for part in _TEXT_PARTS.findall(text):
            candidate = current + part
            if current and self._input_length(candidate) > MAX_INPUT_TOKENS:
                chunks.append(current)
                current = part
            else:
                current = candidate
            if self._input_length(current) > MAX_INPUT_TOKENS:
                raise TranslationInputTooLongError(
                    "A single whitespace-delimited token exceeds the NLLB input limit."
                )
        if current:
            chunks.append(current)
        return chunks

    def _generate(self, text: str) -> str:
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._target_token_id is not None

        inputs = self._tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[-1] > MAX_INPUT_TOKENS:
            raise TranslationInputTooLongError("NLLB input exceeds its token limit.")

        import torch

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                forced_bos_token_id=self._target_token_id,
                do_sample=False,
                num_beams=1,
                max_new_tokens=MAX_INPUT_TOKENS,
            )
        return self._tokenizer.decode(generated[0], skip_special_tokens=True)
