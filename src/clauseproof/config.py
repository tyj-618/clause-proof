"""配置加载与模型定价。

契约见 tasks/T-002.md：LLMConfig / load_config / Prices / prices_for。
错误信息遵循 docs/design/0004-error-semantics.md §8 的四要素格式。
本模块只读环境变量与 .env 文件，不发起任何网络请求。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 变量名与默认值来自 tasks/T-002.md「配置项」表；任何来源的空字符串都视为未设置。
_REQUIRED_VARS = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN")
_DEFAULT_MODEL = "glm-5.3"
_DEFAULT_MAX_TOKENS = 8192
_DEFAULT_REASONING_EFFORT = "low"
_REASONING_EFFORTS = ("low", "high", "max")
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_TIMEOUT_S = 180.0


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """模型调用所需的全部配置。api_key 禁止出现在日志与错误信息中。"""

    base_url: str
    api_key: str
    model: str
    max_tokens: int
    # "low" | "high" | "max"；保留发送：无害。
    # 实测该字段在 Anthropic 兼容端点被网关忽略，实际使用默认档（max）
    reasoning_effort: str
    temperature: float
    timeout_s: float
    cache_dir: Path


@dataclass(frozen=True, slots=True)
class Prices:
    """模型单价，单位：元 / 百万 tokens。"""

    input_cny_per_mtok: float
    output_cny_per_mtok: float
    cached_cny_per_mtok: float


# 定价来源：GLM-5.3 输入 ¥8 / 输出 ¥28 / 缓存命中 ¥2，单位均为「元 / 百万 tokens」。
# 出处：https://docs.bigmodel.cn/cn/guide/start/pricing，查询日期 2026-10-26。
_PRICES: dict[str, Prices] = {
    "glm-5.3": Prices(
        input_cny_per_mtok=8.0,
        output_cny_per_mtok=28.0,
        cached_cny_per_mtok=2.0,
    ),
}


def prices_for(model: str) -> Prices:
    """查模型单价。未知模型抛 RuntimeError：宁可失败，也不要算出错误的成本。"""
    prices = _PRICES.get(model)
    if prices is None:
        known = "、".join(_PRICES)
        raise RuntimeError(f"config: prices_for 违反 已知模型，期望 {known} 之一，实际 {model!r}")
    return prices


def _project_root() -> Path:
    """项目根目录：src/clauseproof/config.py 的上两级。"""
    return Path(__file__).resolve().parents[2]


def _parse_env_file(path: Path) -> dict[str, str]:
    """解析 .env：跳过空行与 # 注释，按第一个 = 切分，去掉成对引号。

    无法解析的行立即抛 RuntimeError；行内容不回显，避免泄漏凭据。
    """
    values: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition("=")
        key = key.strip()
        if not sep or not key:
            raise RuntimeError(
                f"config: {path} 第 {lineno} 行 违反 KEY=VALUE 格式，"
                "期望 KEY=VALUE，实际 无法解析的行（内容不回显）"
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        values[key] = value
    return values


def _to_int(name: str, raw: str, *, default: int) -> int:
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"config: {name} 违反 整数格式，期望 整数，实际 {raw!r}") from None


def _to_float(name: str, raw: str, *, default: float) -> float:
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"config: {name} 违反 数值格式，期望 数值，实际 {raw!r}") from None


def load_config(env_file: Path | None = None) -> LLMConfig:
    """加载配置：环境变量优先，其次项目根（或 env_file 指定）的 .env。

    两处都没有必填项时抛 RuntimeError，消息说明缺哪一项以及怎么补。
    """
    source = _project_root() / ".env" if env_file is None else env_file
    from_file = _parse_env_file(source) if source.is_file() else {}

    def value_of(name: str) -> str:
        env_value = os.environ.get(name, "")
        if env_value:
            return env_value
        return from_file.get(name, "")

    missing = [name for name in _REQUIRED_VARS if not value_of(name)]
    if missing:
        raise RuntimeError(
            "config: "
            + "、".join(missing)
            + " 违反 必填非空，期望 在环境变量或项目根 .env 中设置非空值"
            "（参见 .env.example），实际 空"
        )

    reasoning_effort = value_of("CLAUSEPROOF_REASONING_EFFORT") or _DEFAULT_REASONING_EFFORT
    if reasoning_effort not in _REASONING_EFFORTS:
        raise RuntimeError(
            "config: CLAUSEPROOF_REASONING_EFFORT 违反 取值范围，"
            f"期望 {'/'.join(_REASONING_EFFORTS)}，实际 {reasoning_effort!r}"
        )
    raw_max_tokens = value_of("CLAUSEPROOF_MAX_TOKENS")
    max_tokens = _to_int("CLAUSEPROOF_MAX_TOKENS", raw_max_tokens, default=_DEFAULT_MAX_TOKENS)
    if max_tokens < 1:
        raise RuntimeError(
            f"config: CLAUSEPROOF_MAX_TOKENS 违反 >=1，期望 >=1 的整数，实际 {max_tokens}"
        )
    raw_temperature = value_of("CLAUSEPROOF_TEMPERATURE")
    temperature = _to_float(
        "CLAUSEPROOF_TEMPERATURE", raw_temperature, default=_DEFAULT_TEMPERATURE
    )
    raw_timeout = value_of("CLAUSEPROOF_TIMEOUT")
    timeout_s = _to_float("CLAUSEPROOF_TIMEOUT", raw_timeout, default=_DEFAULT_TIMEOUT_S)
    if timeout_s <= 0.0:
        raise RuntimeError(f"config: CLAUSEPROOF_TIMEOUT 违反 >0，期望 >0 的秒数，实际 {timeout_s}")
    raw_cache_dir = value_of("CLAUSEPROOF_CACHE_DIR")
    # 缓存是项目不变式：不提供关闭开关，未设置时落到项目根 .cache/llm
    cache_dir = Path(raw_cache_dir) if raw_cache_dir else _project_root() / ".cache" / "llm"
    return LLMConfig(
        base_url=value_of("ANTHROPIC_BASE_URL"),
        api_key=value_of("ANTHROPIC_AUTH_TOKEN"),
        model=value_of("CLAUSEPROOF_MODEL") or _DEFAULT_MODEL,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        timeout_s=timeout_s,
        cache_dir=cache_dir,
    )
