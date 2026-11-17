from pathlib import Path

import pytest
import yaml

from robomcp.config import ConfigError, Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_config_equals_code_defaults():
    assert load_settings(ROOT / "configs/default.yaml") == Settings()


def write(tmp_path, data):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_overrides_and_nested_sections(tmp_path):
    s = load_settings(write(tmp_path, {"data_dir": "d", "world": {"seed": 3}}))
    assert s.data_dir == Path("d") and s.world.seed == 3 and s.world.n_robots == 3


@pytest.mark.parametrize("data", [
    {"wrold": {}},
    {"world": {"sed": 1}},
    {"world": {"seed": "7"}},
    {"world": {"seed": True}},
    {"world": {"n_robots": 0}},
    {"corpus": {"exclude_globs": "x"}},
    {"world": 3},
])
def test_bad_configs_fail_early(tmp_path, data):
    with pytest.raises(ConfigError):
        load_settings(write(tmp_path, data))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "nope.yaml")
