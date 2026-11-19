"""Partial, checksum-verified local mirror of public Zarr v2 stores on Google Cloud Storage.

Only the chunks an analysis needs are downloaded. Each chunk is fetched over HTTPS, checked
against the MD5 that Cloud Storage reports in the ``x-goog-hash`` header, written atomically and
recorded in a JSON-lines manifest. The local copy keeps the store's key layout, so xarray opens it
like the original. Interrupted downloads resume where they stopped.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

USER_AGENT = "land-blind/1.0 (research software; python-urllib)"
METADATA_KEYS = (".zmetadata", ".zgroup", ".zattrs")
COORD_LIKE = {"time", "latitude", "longitude", "level", "prediction_timedelta", "number",
              "sample", "lead_time_secs", "dayofyear", "hour"}


class MissingChunksError(RuntimeError):
    """Raised when a requested selection is not fully present in the local mirror."""


@dataclass(frozen=True)
class Store:
    """A remote Zarr v2 store and its local mirror."""

    bucket_url: str
    path: str
    data_dir: Path

    @property
    def local(self) -> Path:
        return self.data_dir / "mirror" / self.path

    @property
    def manifest(self) -> Path:
        return self.data_dir / "manifests" / (self.path.replace("/", "__") + ".jsonl")

    def url(self, key: str) -> str:
        return f"{self.bucket_url}/{self.path}/{key}"


# --------------------------------------------------------------------------------------------
# Low-level transfer
# --------------------------------------------------------------------------------------------

def _md5_from_header(header: str | None) -> str | None:
    if not header:
        return None
    for part in header.split(","):
        name, _, value = part.strip().partition("=")
        if name == "md5":
            return value
    return None


def _download(url: str, timeout: float) -> tuple[bytes, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        return body, _md5_from_header(response.headers.get("x-goog-hash"))


_manifest_lock = threading.Lock()


def fetch_key(store: Store, key: str, *, retries: int = 6, timeout: float = 120.0) -> int:
    """Download one object into the mirror. Returns bytes downloaded (0 if already present)."""
    target = store.local / key
    if target.exists():
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            body, md5 = _download(store.url(key), timeout)
            digest = base64.b64encode(hashlib.md5(body).digest()).decode()
            if md5 is not None and md5 != digest:
                raise OSError(f"MD5 mismatch for {key}: header {md5}, body {digest}")
            tmp = target.with_name(target.name + ".part")
            tmp.write_bytes(body)
            tmp.replace(target)
            with _manifest_lock:
                store.manifest.parent.mkdir(parents=True, exist_ok=True)
                with store.manifest.open("a") as fh:
                    fh.write(json.dumps({"key": key, "bytes": len(body), "md5": digest}) + "\n")
            return len(body)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                raise FileNotFoundError(store.url(key)) from err
            reason: Exception = err
        except (OSError, TimeoutError, http.client.HTTPException) as err:
            # URLError is an OSError; IncompleteRead (truncated body) is an HTTPException.
            reason = err
        if attempt == retries:
            raise OSError(f"giving up on {key} after {retries} attempts: {reason}") from reason
        log.warning("retry %d/%d for %s (%s)", attempt, retries, key, reason)
        time.sleep(delay)
        delay = min(delay * 2, 60.0)
    raise AssertionError("unreachable")


def fetch_keys(store: Store, keys: Sequence[str], *, workers: int = 4, label: str = "") -> int:
    """Download many objects concurrently, logging throughput. Returns total bytes downloaded."""
    pending = [k for k in keys if not (store.local / k).exists()]
    if not pending:
        log.info("%s: all %d objects already present", label or store.path, len(keys))
        return 0
    log.info("%s: %d of %d objects to download", label or store.path, len(pending), len(keys))
    total, done, start = 0, 0, time.monotonic()
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_key, store, k): k for k in pending}
        for future in as_completed(futures):
            done += 1
            try:
                total += future.result()
            except Exception as err:  # keep going; report every failure at the end
                failed.append(futures[future])
                log.error("%s: failed %s (%s)", label or store.path, futures[future], err)
            if done % 50 == 0 or done == len(pending):
                rate = total / max(time.monotonic() - start, 1e-9) / 1e6
                log.info("%s: %d/%d objects, %.2f GB, %.2f MB/s",
                         label or store.path, done, len(pending), total / 1e9, rate)
    if failed:
        raise OSError(f"{len(failed)} object(s) of {label or store.path} failed, e.g. {failed[0]}; "
                      "rerun to resume")
    return total


# --------------------------------------------------------------------------------------------
# Zarr layout
# --------------------------------------------------------------------------------------------

def fetch_metadata(store: Store) -> dict:
    """Fetch consolidated metadata, write per-array metadata files and fetch coordinate arrays."""
    for key in METADATA_KEYS:
        try:
            fetch_key(store, key)
        except FileNotFoundError:
            if key == ".zmetadata":
                raise
    meta = json.loads((store.local / ".zmetadata").read_text())["metadata"]
    for key, value in meta.items():
        path = store.local / key
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
    for name in array_names(meta):
        dims = array_dims(meta, name)
        is_static_field = len(dims) <= 2 and set(dims) <= {"longitude", "latitude"}
        if name in COORD_LIKE or is_static_field:
            fetch_keys(store, all_chunk_keys(meta, name), workers=4, label=f"{store.path}:{name}")
    return meta


def load_metadata(store: Store) -> dict:
    return json.loads((store.local / ".zmetadata").read_text())["metadata"]


def array_names(meta: Mapping) -> list[str]:
    return sorted(k[: -len("/.zarray")] for k in meta if k.endswith("/.zarray"))


def array_dims(meta: Mapping, name: str) -> list[str]:
    return list(meta[f"{name}/.zattrs"]["_ARRAY_DIMENSIONS"])


def _chunk_grid(meta: Mapping, name: str) -> tuple[list[int], list[int], str]:
    zarray = meta[f"{name}/.zarray"]
    return zarray["shape"], zarray["chunks"], zarray.get("dimension_separator", ".")


def all_chunk_keys(meta: Mapping, name: str) -> list[str]:
    return chunk_keys(meta, name, {})


def chunk_keys(meta: Mapping, name: str, selection: Mapping[str, Iterable[int]]) -> list[str]:
    """Keys of every chunk of ``name`` touched by an integer ``selection``.

    Dimensions absent from ``selection`` contribute all of their chunks.
    """
    shape, chunks, sep = _chunk_grid(meta, name)
    dims = array_dims(meta, name)
    unknown = set(selection) - set(dims)
    if unknown:
        raise KeyError(f"{name} has no dimension(s) {sorted(unknown)}; dims are {dims}")
    per_dim: list[list[int]] = []
    for dim, size, chunk in zip(dims, shape, chunks, strict=True):
        if dim in selection:
            idx = np.asarray(list(selection[dim]), dtype=np.int64)
            if idx.size and (idx.min() < 0 or idx.max() >= size):
                raise IndexError(f"{name}:{dim} index out of range [0, {size})")
            per_dim.append(sorted(set((idx // chunk).tolist())))
        else:
            per_dim.append(list(range(-(-size // chunk))))
    prefix = f"{name}/"
    return [prefix + sep.join(map(str, combo)) for combo in product(*per_dim)]


def require(store: Store, name: str, selection: Mapping[str, Iterable[int]]) -> None:
    """Raise :class:`MissingChunksError` unless every chunk of the selection is mirrored."""
    meta = load_metadata(store)
    missing = [k for k in chunk_keys(meta, name, selection) if not (store.local / k).exists()]
    if missing:
        raise MissingChunksError(f"{len(missing)} chunk(s) of {store.path}:{name} missing, "
                                 f"e.g. {missing[0]}")


def fetch_selection(store: Store, names: Sequence[str], selection: Mapping[str, Iterable[int]],
                    *, workers: int = 4) -> int:
    """Download the chunks of each array in ``names`` that the selection touches."""
    meta = load_metadata(store)
    total = 0
    for name in names:
        dims = array_dims(meta, name)
        sub = {d: v for d, v in selection.items() if d in dims}
        total += fetch_keys(store, chunk_keys(meta, name, sub), workers=workers,
                            label=f"{store.path.split('/')[-2]}:{name}")
    return total


def open_dataset(store: Store) -> xr.Dataset:
    """Open the local mirror lazily, without dask. Check selections with :func:`require` first."""
    return xr.open_zarr(store.local, consolidated=True, chunks=None, decode_timedelta=True)


def manifest_digest(store: Store) -> dict:
    """Object count, bytes and a SHA-256 over sorted ``key md5`` lines of the store's manifest.

    Two mirrors with the same digest hold byte-identical chunks.
    """
    if not store.manifest.exists():
        return {"objects": 0, "bytes": 0, "sha256": None}
    entries = {}
    for line in store.manifest.read_text().splitlines():
        rec = json.loads(line)
        entries[rec["key"]] = (rec["md5"], rec["bytes"])
    digest = hashlib.sha256()
    for key in sorted(entries):
        digest.update(f"{key} {entries[key][0]}\n".encode())
    return {"objects": len(entries), "bytes": int(sum(b for _, b in entries.values())),
            "sha256": digest.hexdigest()}
