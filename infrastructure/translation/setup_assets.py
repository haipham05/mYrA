"""Download and verify only the pinned local BabelDOC rendering assets."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import httpx
from tiktoken import encoding_for_model
from babeldoc.assets import assets
from babeldoc.assets.embedding_assets_metadata import TIKTOKEN_CACHES

EXPECTED_LAYOUT_SHA3_256 = "60be061226930524958b5465c8c04af3d7c03bcb0beb66454f5da9f792e3cf2a"


def sha3(path: Path) -> str:
    digest = hashlib.sha3_256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def prepare_assets() -> tuple[Path, int, int]:
    layout_path = Path(await assets.get_doclayout_onnx_model_path_async())
    if sha3(layout_path) != EXPECTED_LAYOUT_SHA3_256:
        raise RuntimeError("Pinned DocLayout model checksum did not match")
    encoding_for_model("gpt-4o")
    for cache_name, expected_hash in TIKTOKEN_CACHES.items():
        cache_path = assets.get_cache_file_path(cache_name, "tiktoken")
        if not assets.verify_file(cache_path, expected_hash):
            raise RuntimeError("Pinned tokenizer asset checksum did not match")

    font_groups = assets.get_font_family("Vietnamese")
    font_names = sorted(
        {name for group in ("normal", "script", "fallback", "base") for name in font_groups[group]}
    )
    async with httpx.AsyncClient(timeout=60.0) as client:
        for name in font_names:
            path, metadata = await assets.get_font_and_metadata_async(name, client=client)
            if not assets.verify_file(path, metadata["sha3_256"]):
                raise RuntimeError(f"Pinned Vietnamese font checksum did not match: {name}")
        cmap_names = sorted(assets.CMAP_METADATA)
        for name in cmap_names:
            path = await assets.download_cmap_file_async(name, client=client)
            if not assets.verify_file(path, assets.CMAP_METADATA[name]["sha3_256"]):
                raise RuntimeError(f"Pinned CMap checksum did not match: {name}")
    return layout_path, len(font_names), len(cmap_names)


def main() -> None:
    layout, fonts, cmaps = asyncio.run(prepare_assets())
    print(f"Verified pinned layout model at {layout}")
    print(f"Verified {fonts} Vietnamese font files and {cmaps} CMap files")


if __name__ == "__main__":
    main()
