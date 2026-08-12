from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType
from typing import Any

import pytest

from app.services.translation.local_nllb import (
    MAX_INPUT_TOKENS,
    MODEL_ID,
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    LocalNllbTranslator,
    TranslationInputTooLongError,
)


class FakeIds(list):
    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self[0]))


class FakeTokenizer:
    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.generation_inputs: list[str] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == TARGET_LANGUAGE
        return 7

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        assert add_special_tokens is True
        assert truncation is False
        self.inputs.append(text)
        if return_tensors:
            self.generation_inputs.append(text)
        # One token per non-whitespace chunk and two special tokens.
        token_count = len(text.split()) + 2
        ids = list(range(token_count))
        return {"input_ids": FakeIds([ids]) if return_tensors else ids}

    def decode(self, _tokens: Any, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "đã dịch"


class FakeModel:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, device: str) -> FakeModel:
        assert device == "cpu"
        return self

    def eval(self) -> FakeModel:
        return self

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generate_calls.append(kwargs)
        return [[1]]


def _translator(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LocalNllbTranslator, FakeTokenizer, FakeModel, list[dict[str, Any]]]:
    torch_module = ModuleType("torch")
    torch_module.inference_mode = nullcontext  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    tokenizer = FakeTokenizer()
    model = FakeModel()
    load_calls: list[dict[str, Any]] = []

    def loader(model_id: str, **kwargs: Any) -> tuple[FakeTokenizer, FakeModel]:
        load_calls.append({"model_id": model_id, **kwargs})
        return tokenizer, model

    return (
        LocalNllbTranslator(revision="pinned-revision", loader=loader),
        tokenizer,
        model,
        load_calls,
    )


def test_nllb_loads_lazily_and_preserves_protected_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translator, tokenizer, model, load_calls = _translator(monkeypatch)
    assert load_calls == []

    result = translator.translate("A result [[MYRA_KEEP_1]] and <b2>label</b2>.")

    assert result == "đã dịch[[MYRA_KEEP_1]]đã dịch<b2>đã dịch</b2>đã dịch"
    assert len(load_calls) == 1
    assert load_calls[0] == {
        "model_id": MODEL_ID,
        "revision": "pinned-revision",
        "source_language": SOURCE_LANGUAGE,
    }
    assert len(model.generate_calls) == 4
    assert all(call["do_sample"] is False for call in model.generate_calls)
    assert all(call["num_beams"] == 1 for call in model.generate_calls)
    assert all(call["forced_bos_token_id"] == 7 for call in model.generate_calls)
    assert all(len(text.split()) + 2 <= MAX_INPUT_TOKENS for text in tokenizer.generation_inputs)


def test_nllb_splits_long_text_without_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    translator, tokenizer, model, _ = _translator(monkeypatch)

    result = translator.translate("word " * 960)

    assert result.count("đã dịch") == 3
    assert len(model.generate_calls) == 3
    assert all(len(text.split()) + 2 <= MAX_INPUT_TOKENS for text in tokenizer.generation_inputs)
    assert sum(text.split().count("word") for text in tokenizer.generation_inputs) == 960


def test_nllb_rejects_single_token_over_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    translator, _, _, _ = _translator(monkeypatch)
    translator._load()
    assert translator._input_length("x" * 500) <= MAX_INPUT_TOKENS  # fake tokenizer counts words

    # A tokenizer that maps one token to an oversized sequence is never asked
    # to truncate it silently.
    translator._tokenizer = type(
        "OversizedTokenizer",
        (),
        {"__call__": lambda self, text, **kwargs: {"input_ids": list(range(MAX_INPUT_TOKENS + 1))}},
    )()
    with pytest.raises(TranslationInputTooLongError):
        translator._split_to_fit("indivisible")
