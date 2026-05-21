import io

import pytest

from app.storage.gcs import GCSStorage
from app.storage.local import LocalStorage


@pytest.mark.anyio
async def test_local_stream_round_trip(tmp_path):
    storage = LocalStorage(base_dir=str(tmp_path / "objects"))
    source = io.BytesIO(b"%PDF-" + b"a" * 100_000)
    key = await storage.put_stream("papers/test.pdf", source)
    assert key == "papers/test.pdf"
    chunks = [chunk async for chunk in storage.open_stream(key, chunk_size=8192)]
    assert len(chunks) > 1
    assert max(map(len, chunks)) <= 8192
    assert b"".join(chunks) == source.getvalue()


@pytest.mark.anyio
async def test_gcs_stream_uses_file_api_instead_of_download_as_bytes(monkeypatch):
    data = b"%PDF-" + b"b" * 100_000

    class FakeBlob:
        def __init__(self):
            self.uploaded = b""
            self.read_chunk_size = None

        def upload_from_file(self, source, *, rewind, content_type):
            assert rewind is True
            assert content_type == "application/pdf"
            self.uploaded = source.read()

        def open(self, mode, *, chunk_size):
            assert mode == "rb"
            self.read_chunk_size = chunk_size
            return io.BytesIO(self.uploaded)

        def download_as_bytes(self):
            raise AssertionError("GCS streaming must not buffer the whole object")

    blob = FakeBlob()

    class FakeBucket:
        def blob(self, key):
            assert key == "papers/test.pdf"
            return blob

    storage = GCSStorage(bucket_name="disposable-test")
    monkeypatch.setattr(storage, "_get_bucket", FakeBucket)
    uri = await storage.put_stream("papers/test.pdf", io.BytesIO(data))
    assert uri == "gs://disposable-test/papers/test.pdf"
    chunks = [chunk async for chunk in storage.open_stream("papers/test.pdf", chunk_size=4096)]
    assert blob.read_chunk_size == 4096
    assert max(map(len, chunks)) <= 4096
    assert b"".join(chunks) == data
