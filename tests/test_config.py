"""config 测试：优先级、默认值、.env 解析、校验与定价表。不发起任何网络请求。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from clauseproof import config
from clauseproof.config import LLMConfig, Prices, load_config, prices_for

_ALL_VARS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUSEPROOF_MODEL",
    "CLAUSEPROOF_MAX_TOKENS",
    "CLAUSEPROOF_REASONING_EFFORT",
    "CLAUSEPROOF_TEMPERATURE",
    "CLAUSEPROOF_TIMEOUT",
    "CLAUSEPROOF_CACHE_DIR",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清空全部配置变量，隔离宿主环境（真实 Claude Code 会设置 ANTHROPIC_*）。"""
    for name in _ALL_VARS:
        monkeypatch.delenv(name, raising=False)


def _set_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://fake.test/api/anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-fake-token")


def _write_env(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_env_var_overrides_dotenv_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = _write_env(
        tmp_path / ".env",
        "ANTHROPIC_BASE_URL=https://from-file.test\n"
        "ANTHROPIC_AUTH_TOKEN=file-token\n"
        "CLAUSEPROOF_MODEL=glm-from-file\n",
    )
    monkeypatch.setenv("CLAUSEPROOF_MODEL", "glm-from-env")
    cfg = load_config(env_file)
    assert cfg.model == "glm-from-env"
    assert cfg.base_url == "https://from-file.test"
    assert cfg.api_key == "file-token"


def test_empty_env_value_falls_back_to_dotenv_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = _write_env(tmp_path / ".env", "ANTHROPIC_AUTH_TOKEN=file-token\n")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://fake.test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "")
    assert load_config(env_file).api_key == "file-token"


def test_dotenv_file_supplies_all_values(tmp_path: Path) -> None:
    env_file = _write_env(
        tmp_path / ".env",
        "ANTHROPIC_BASE_URL=https://file.test\n"
        "ANTHROPIC_AUTH_TOKEN=file-token\n"
        "CLAUSEPROOF_MODEL=file-model\n"
        "CLAUSEPROOF_MAX_TOKENS=123\n"
        "CLAUSEPROOF_REASONING_EFFORT=high\n"
        "CLAUSEPROOF_TEMPERATURE=0.5\n"
        "CLAUSEPROOF_TIMEOUT=12.5\n"
        f"CLAUSEPROOF_CACHE_DIR={tmp_path / 'file-cache'}\n",
    )
    assert load_config(env_file) == LLMConfig(
        base_url="https://file.test",
        api_key="file-token",
        model="file-model",
        max_tokens=123,
        reasoning_effort="high",
        temperature=0.5,
        timeout_s=12.5,
        cache_dir=tmp_path / "file-cache",
    )


def test_defaults_when_optional_vars_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setattr(config, "_project_root", lambda: tmp_path)
    cfg = load_config(tmp_path / "absent.env")
    assert cfg.model == "glm-5.3"
    assert cfg.max_tokens == 8192
    assert cfg.reasoning_effort == "low"
    assert cfg.temperature == 0.0
    assert cfg.timeout_s == 180.0
    assert cfg.cache_dir == tmp_path / ".cache" / "llm"


def test_cache_dir_from_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_CACHE_DIR", str(tmp_path / "custom-cache"))
    cfg = load_config(tmp_path / "absent.env")
    assert cfg.cache_dir == tmp_path / "custom-cache"


@pytest.mark.parametrize("missing", ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"])
def test_missing_required_var_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    _set_required(monkeypatch)
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.env")
    message = str(excinfo.value)
    assert missing in message
    assert "环境变量" in message
    assert ".env.example" in message


def test_missing_required_names_only_missing_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://fake.test")
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.env")
    message = str(excinfo.value)
    assert "ANTHROPIC_AUTH_TOKEN" in message
    assert "ANTHROPIC_BASE_URL" not in message


def test_empty_required_env_value_counts_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://fake.test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "")
    with pytest.raises(RuntimeError, match="ANTHROPIC_AUTH_TOKEN"):
        load_config(tmp_path / "absent.env")


def test_project_root_dotenv_used_when_env_file_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_env(
        tmp_path / ".env",
        "ANTHROPIC_BASE_URL=https://root.test\nANTHROPIC_AUTH_TOKEN=root-token\n",
    )
    monkeypatch.setattr(config, "_project_root", lambda: tmp_path)
    cfg = load_config()
    assert cfg.base_url == "https://root.test"
    assert cfg.api_key == "root-token"


def test_dotenv_parsing_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    env_file = _write_env(
        tmp_path / ".env",
        "\n"
        "# 整行注释\n"
        "  ANTHROPIC_BASE_URL = https://spaced.test  \n"
        "ANTHROPIC_AUTH_TOKEN='single-quoted'\n"
        'CLAUSEPROOF_MODEL="double-quoted"\n',
    )
    cfg = load_config(env_file)
    assert cfg.base_url == "https://spaced.test"
    assert cfg.api_key == "single-quoted"
    assert cfg.model == "double-quoted"


def test_dotenv_malformed_line_raises_without_echoing_content(tmp_path: Path) -> None:
    env_file = _write_env(
        tmp_path / ".env",
        "ANTHROPIC_BASE_URL=https://x.test\nANTHROPIC_AUTH_TOKEN=t\n这行没有等号\n",
    )
    with pytest.raises(RuntimeError) as excinfo:
        load_config(env_file)
    message = str(excinfo.value)
    assert "第 3 行" in message
    assert "这行没有等号" not in message


def test_invalid_max_tokens_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_MAX_TOKENS", "abc")
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.env")
    message = str(excinfo.value)
    assert "CLAUSEPROOF_MAX_TOKENS" in message
    assert "abc" in message


def test_nonpositive_max_tokens_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_MAX_TOKENS", "0")
    with pytest.raises(RuntimeError, match="CLAUSEPROOF_MAX_TOKENS"):
        load_config(tmp_path / "absent.env")


def test_invalid_temperature_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_TEMPERATURE", "hot")
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.env")
    message = str(excinfo.value)
    assert "CLAUSEPROOF_TEMPERATURE" in message
    assert "hot" in message


def test_nonpositive_timeout_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_TIMEOUT", "0")
    with pytest.raises(RuntimeError, match="CLAUSEPROOF_TIMEOUT"):
        load_config(tmp_path / "absent.env")


def test_invalid_reasoning_effort_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_REASONING_EFFORT", "medium")
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.env")
    message = str(excinfo.value)
    assert "CLAUSEPROOF_REASONING_EFFORT" in message
    for allowed in ("low", "high", "max"):
        assert allowed in message


def test_prices_for_glm_53() -> None:
    assert prices_for("glm-5.3") == Prices(
        input_cny_per_mtok=8.0,
        output_cny_per_mtok=28.0,
        cached_cny_per_mtok=2.0,
    )


def test_prices_for_unknown_model_raises() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        prices_for("gpt-4o")
    message = str(excinfo.value)
    assert "gpt-4o" in message
    assert "glm-5.3" in message


def test_llm_config_is_frozen(tmp_path: Path) -> None:
    cfg = LLMConfig(
        base_url="https://fake.test",
        api_key="sk-fake",
        model="glm-5.3",
        max_tokens=8,
        reasoning_effort="low",
        temperature=0.0,
        timeout_s=1.0,
        cache_dir=tmp_path / "cache",
    )
    with pytest.raises(FrozenInstanceError):
        cfg.model = "other"


def test_auth_token_never_echoed_in_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_MAX_TOKENS", "abc")
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.env")
    assert "sk-fake-token" not in str(excinfo.value)
