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
    "CLAUSEPROOF_PROTOCOL",
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
        "CLAUSEPROOF_PROTOCOL=anthropic\n"
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
        protocol="anthropic",
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
    # T-006：协议默认 openai，与 LLMConfig 字段默认一致（见构造路径一致性测试）
    assert cfg.protocol == "openai"


def test_cache_dir_from_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_CACHE_DIR", str(tmp_path / "custom-cache"))
    cfg = load_config(tmp_path / "absent.env")
    assert cfg.cache_dir == tmp_path / "custom-cache"


@pytest.mark.parametrize("missing", ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"])
def test_missing_required_var_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setenv("CLAUSEPROOF_PROTOCOL", "anthropic")
    _set_required(monkeypatch)
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.env")
    message = str(excinfo.value)
    assert missing in message
    assert "环境变量" in message
    assert ".env.example" in message


def test_openai_protocol_base_url_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """openai 协议的 URL 由代码内置（llm 模块常量），缺 ANTHROPIC_BASE_URL 不报错。"""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-fake-token")
    cfg = load_config(tmp_path / "absent.env")
    assert cfg.protocol == "openai"
    assert cfg.base_url == ""


def test_openai_protocol_auth_token_still_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://fake.test")
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.env")
    message = str(excinfo.value)
    assert "ANTHROPIC_AUTH_TOKEN" in message
    assert "ANTHROPIC_BASE_URL" not in message  # openai 下 base_url 不在必填清单


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


def test_anthropic_reasoning_effort_medium_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """anthropic 协议保留原三档校验（该端点忽略此字段，但发送无害）。"""
    monkeypatch.setenv("CLAUSEPROOF_PROTOCOL", "anthropic")
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_REASONING_EFFORT", "medium")
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.env")
    message = str(excinfo.value)
    assert "CLAUSEPROOF_REASONING_EFFORT" in message
    assert "protocol=anthropic" in message
    for allowed in ("low", "high", "max"):
        assert allowed in message


def test_anthropic_reasoning_effort_openai_only_value_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """openai 专属档位（如 none）在 anthropic 协议下非法。"""
    monkeypatch.setenv("CLAUSEPROOF_PROTOCOL", "anthropic")
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_REASONING_EFFORT", "none")
    with pytest.raises(RuntimeError, match="protocol=anthropic"):
        load_config(tmp_path / "absent.env")


def test_openai_reasoning_effort_seven_values_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """openai 协议合法档位共 7 个，包括 anthropic 下非法的 medium / none。"""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-fake-token")
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        monkeypatch.setenv("CLAUSEPROOF_REASONING_EFFORT", effort)
        assert load_config(tmp_path / "absent.env").reasoning_effort == effort


def test_openai_reasoning_effort_invalid_value_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-fake-token")
    monkeypatch.setenv("CLAUSEPROOF_REASONING_EFFORT", "banana")
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.env")
    message = str(excinfo.value)
    assert "CLAUSEPROOF_REASONING_EFFORT" in message
    assert "protocol=openai" in message
    for allowed in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        assert allowed in message


def test_protocol_from_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_PROTOCOL", "anthropic")
    assert load_config(tmp_path / "absent.env").protocol == "anthropic"


def test_unknown_protocol_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_PROTOCOL", "grpc")
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.env")
    message = str(excinfo.value)
    assert "CLAUSEPROOF_PROTOCOL" in message
    assert "anthropic/openai" in message
    assert "grpc" in message


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
        protocol="openai",
    )
    with pytest.raises(FrozenInstanceError):
        cfg.model = "other"


def test_protocol_default_consistent_across_construction_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """直接构造与 load_config 的默认协议一致：行为不得取决于构造路径。"""
    _set_required(monkeypatch)
    direct = LLMConfig(
        base_url="https://fake.test",
        api_key="sk-fake",
        model="glm-5.3",
        max_tokens=8,
        reasoning_effort="low",
        temperature=0.0,
        timeout_s=1.0,
        cache_dir=tmp_path / "cache",
    )
    assert direct.protocol == "openai"
    assert load_config(tmp_path / "absent.env").protocol == "openai"


def test_auth_token_never_echoed_in_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CLAUSEPROOF_MAX_TOKENS", "abc")
    with pytest.raises(RuntimeError) as excinfo:
        load_config(tmp_path / "absent.env")
    assert "sk-fake-token" not in str(excinfo.value)
