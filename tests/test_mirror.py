import base64
import hashlib
import json

import pytest

from landmem import mirror
from landmem.mirror import Store, _md5_from_header, chunk_keys, fetch_key

META = {
    "x/.zarray": {"shape": [10, 6, 4], "chunks": [4, 6, 4], "dimension_separator": "."},
    "x/.zattrs": {"_ARRAY_DIMENSIONS": ["time", "longitude", "latitude"]},
}


def test_chunk_keys_selects_touched_chunks():
    assert chunk_keys(META, "x", {"time": [0, 3]}) == ["x/0.0.0"]
    assert chunk_keys(META, "x", {"time": [3, 4, 9]}) == ["x/0.0.0", "x/1.0.0", "x/2.0.0"]
    assert chunk_keys(META, "x", {}) == ["x/0.0.0", "x/1.0.0", "x/2.0.0"]


def test_chunk_keys_rejects_bad_selection():
    with pytest.raises(KeyError):
        chunk_keys(META, "x", {"level": [0]})
    with pytest.raises(IndexError):
        chunk_keys(META, "x", {"time": [10]})


def test_md5_header_parsing():
    assert _md5_from_header("crc32c=AAAA==,md5=BBBB==") == "BBBB=="
    assert _md5_from_header(None) is None


def test_fetch_key_verifies_and_records(tmp_path, monkeypatch):
    body = b"chunk-bytes"
    good = base64.b64encode(hashlib.md5(body).digest()).decode()
    monkeypatch.setattr(mirror, "_download", lambda url, timeout: (body, good))
    store = Store("https://example.invalid", "datasets/test.zarr", tmp_path)
    assert fetch_key(store, "x/0.0.0") == len(body)
    assert (store.local / "x/0.0.0").read_bytes() == body
    assert json.loads(store.manifest.read_text().splitlines()[0])["md5"] == good
    assert fetch_key(store, "x/0.0.0") == 0  # already present: resumes without downloading


def test_fetch_key_rejects_corrupt_download(tmp_path, monkeypatch):
    monkeypatch.setattr(mirror, "_download", lambda url, timeout: (b"abc", "not-the-md5"))
    monkeypatch.setattr(mirror.time, "sleep", lambda s: None)
    store = Store("https://example.invalid", "datasets/test.zarr", tmp_path)
    with pytest.raises(OSError, match="giving up"):
        fetch_key(store, "x/0.0.0", retries=2)
    assert not (store.local / "x/0.0.0").exists()
