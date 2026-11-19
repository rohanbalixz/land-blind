"""Load the registered analysis configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from landblind.mirror import Store

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "analysis.yaml"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def data_dir(self) -> Path:
        return self.root / self.raw["data_dir"]

    @property
    def processed(self) -> Path:
        path = self.data_dir / "processed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def results(self) -> Path:
        return self.root / "results"

    def store(self, name: str) -> Store:
        return Store(self.raw["bucket_url"], self.raw["stores"][name], self.data_dir)


def load_config(path: Path | str = DEFAULT_CONFIG, root: Path | None = None) -> Config:
    path = Path(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=root or REPO_ROOT)
