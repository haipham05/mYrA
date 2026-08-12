"""Download the pinned NLLB model once into mYrA's ignored local cache."""

from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "facebook/nllb-200-distilled-600M"
MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
MODEL_DIR = Path(os.getenv("MYRA_TRANSLATION_MODEL_PATH", ".local/translation/nllb"))


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "pytorch_model.bin",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    )
    manifest = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": "CC-BY-NC-4.0",
        "source_language": "eng_Latn",
        "target_language": "vie_Latn",
    }
    (MODEL_DIR / "myra-model.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {MODEL_ID} at pinned revision {MODEL_REVISION} in {MODEL_DIR}")


if __name__ == "__main__":
    main()
