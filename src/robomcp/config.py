"""Typed settings loaded from `configs/default.yaml`.

Unknown keys and wrong types fail at start-up instead of surfacing later as a
half-configured server.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CorpusSettings:
    ros2_commit: str = "0ea463a2b27865c93acc928eccd2276e47c9f71c"
    exclude_globs: tuple[str, ...] = ("Releases/*Complete-Changelog*", "_internal/*")


@dataclass(frozen=True)
class EvalSettings:
    source_repo: str = "haroonkhan02/robotics-docs-rag"
    source_commit: str = "99393783595570318515d2da43fd911b6bfa9ac1"


@dataclass(frozen=True)
class WorldSettings:
    seed: int = 7
    n_robots: int = 3


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    index_path: Path = Path("artifacts/docs_sections.jsonl")
    policy_path: Path = Path("configs/policy.yaml")
    corpus: CorpusSettings = field(default_factory=CorpusSettings)
    eval: EvalSettings = field(default_factory=EvalSettings)
    world: WorldSettings = field(default_factory=WorldSettings)


def _build(cls: type, raw: Any, where: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(fields)}")
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        default = getattr(cls(), name)
        path = f"{where}.{name}" if where else name
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value, path)
        elif isinstance(default, Path):
            if not isinstance(value, str):
                raise ConfigError(f"{path}: expected a path string")
            kwargs[name] = Path(value)
        elif isinstance(default, tuple):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ConfigError(f"{path}: expected a list of strings")
            kwargs[name] = tuple(value)
        elif type(value) is not type(default):  # exact: True must not pass as an int
            raise ConfigError(
                f"{path}: expected {type(default).__name__}, got {type(value).__name__}"
            )
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_settings(path: Path | str | None = None) -> Settings:
    """Read settings from `path`; with no path, use configs/default.yaml if it exists."""
    if path is None:
        if not DEFAULT_CONFIG.exists():
            return Settings()
        path = DEFAULT_CONFIG
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = _build(Settings, raw, "")
    if settings.world.n_robots < 1:
        raise ConfigError("world.n_robots must be at least 1")
    return settings
