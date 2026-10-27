"""模型调用层：可替换协议后端、可注入传输、磁盘缓存、重试、计量与成本。

契约见 tasks/T-002.md 与 tasks/T-006.md（Backend 抽象）。请求构造与响应解析按协议拆分为
AnthropicBackend / OpenAIBackend，由 LLMConfig.protocol 选择；全项目只有 HttpxTransport.post
一处发起网络请求（见 ADR-0001，不依赖官方 SDK，请求体完全可控）。错误信息遵循
docs/design/0004-error-semantics.md §8，LLMClient 对外只抛 RuntimeError。日志与异常只出现
prompt_hash，不出现 prompt 正文与 api_key。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx

from clauseproof.config import LLMConfig, prices_for

logger = logging.getLogger(__name__)

_ANTHROPIC_VERSION = "2023-06-01"
# 智谱 OpenAI 兼容端点（chat/completions）。URL 由代码内置而非环境变量：
# 切换协议只改 CLAUSEPROOF_PROTOCOL 一项，不要求为 openai 配置 base_url。
# 出处 https://docs.bigmodel.cn/cn/guide/models/text/glm-5.3，查询日期 2026-10-26
_OPENAI_CHAT_COMPLETIONS_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
_RETRY_BACKOFF_BASE_S = 0.5
_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Completion:
    """一次补全的结果与计量。prompt_hash 之外的调试信息只进日志。

    cached=True（磁盘缓存命中）时 cost_cny=0.0——本次未发生计费（NFR-2）；
    input_tokens / output_tokens / cached_input_tokens 保留首次响应的原始用量，
    便于追溯「如果没命中会花多少」，不参与成本计算。

    cache_written：结果是否已落盘缓存。写入成功或缓存命中为 True；
    写入失败（磁盘满、权限等）为 False，不影响结果返回。
    与 latency_ms / cached 一样属运行时状态，不写入缓存文件。
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    latency_ms: float
    cached: bool
    cache_written: bool
    stop_reason: str | None
    cost_cny: float


@runtime_checkable
class Transport(Protocol):
    """传输协议：POST 一个 JSON 请求体，返回解析后的 JSON dict。

    非 2xx 时抛 httpx.HTTPStatusError（携带 status_code 供重试决策）。
    """

    def post(self, url: str, payload: dict, headers: dict[str, str]) -> dict: ...


class HttpxTransport:
    """真实传输（httpx 直发）。项目中唯一允许发起网络请求的类（0001 §5 原则二）。"""

    def __init__(self, timeout_s: float) -> None:
        self._client = httpx.Client(timeout=timeout_s)

    def post(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        response = self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("llm: 模型端点返回了非 JSON 的 2xx 响应") from exc


@dataclass(frozen=True, slots=True)
class BackendResult:
    """端点响应解析后的统一结构，协议差异在此归一（字段映射见 tasks/T-006.md）。"""

    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    stop_reason: str | None


@runtime_checkable
class Backend(Protocol):
    """协议后端：构造请求 + 解析响应。具体端点的差异收敛在这两个方法里。"""

    def request(
        self, config: LLMConfig, prompt: str, system: str | None, max_tokens: int
    ) -> tuple[str, dict, dict[str, str]]:
        """返回 (url, payload, headers)。"""

    def parse(self, response: dict) -> BackendResult:
        """把端点响应解析为统一结构。"""


class AnthropicBackend:
    """Anthropic Messages 协议（智谱兼容端点）：顶层 system 字段、content 块响应。"""

    def request(
        self, config: LLMConfig, prompt: str, system: str | None, max_tokens: int
    ) -> tuple[str, dict, dict[str, str]]:
        payload: dict = {
            "model": config.model,
            "max_tokens": max_tokens,
            "temperature": config.temperature,
            # GLM-5.3 思考强制开启，reasoning_effort 为与 thinking 平级的顶层字段。
            # 出处 https://docs.bigmodel.cn/cn/guide/models/text/glm-5.3，查询日期 2026-10-26
            "thinking": {"type": "enabled"},
            "reasoning_effort": config.reasoning_effort,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            payload["system"] = system
        url = config.base_url.rstrip("/") + "/v1/messages"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        return url, payload, headers

    def parse(self, response: dict) -> BackendResult:
        usage = response.get("usage") or {}
        return BackendResult(
            text=self._extract_text(response),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cached_input_tokens=int(usage.get("cache_read_input_tokens", 0)),
            stop_reason=response.get("stop_reason"),
        )

    @staticmethod
    def _extract_text(response: dict) -> str:
        """拼接全部 text 块；thinking 等其他类型块不进入正文。"""
        blocks = response.get("content") or []
        return "".join(block.get("text", "") for block in blocks if block.get("type") == "text")


class OpenAIBackend:
    """OpenAI Chat Completions 协议（智谱兼容端点）：system 进 messages，URL 内置。"""

    def request(
        self, config: LLMConfig, prompt: str, system: str | None, max_tokens: int
    ) -> tuple[str, dict, dict[str, str]]:
        messages: list[dict] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": config.temperature,
            "thinking": {"type": "enabled"},
            "reasoning_effort": config.reasoning_effort,
        }
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "content-type": "application/json",
        }
        return _OPENAI_CHAT_COMPLETIONS_URL, payload, headers

    def parse(self, response: dict) -> BackendResult:
        choices = response.get("choices") or []
        first = choices[0] if choices else {}
        message = first.get("message") or {}
        # content 可能为 null（思考耗尽预算时的实测形态），按空串处理
        finish_reason = first.get("finish_reason")
        # 上层空正文保护按 stop_reason == "max_tokens" 判断，OpenAI 用 "length" 表同一件事，
        # 不归一化会把截断当成正常结束（tasks/T-006 字段映射的最后一条要求）
        stop_reason = "max_tokens" if finish_reason == "length" else finish_reason
        usage = response.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return BackendResult(
            text=message.get("content") or "",
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            cached_input_tokens=int(details.get("cached_tokens", 0)),
            stop_reason=stop_reason,
        )


def _default_backend(protocol: str) -> Backend:
    """按 config.protocol 选择默认后端。load_config 已在入口校验，这里是直接构造时的防线。"""
    if protocol == "openai":
        return OpenAIBackend()
    if protocol == "anthropic":
        return AnthropicBackend()
    raise RuntimeError(
        f"llm: config.protocol 违反 取值范围，期望 anthropic/openai，实际 {protocol!r}"
    )


def _cache_key(
    model: str,
    system: str | None,
    prompt: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    protocol: str,
) -> str:
    """缓存键：对请求身份的规范化 JSON 取 sha256（64 位十六进制）。

    任务书要求包含 model、system、prompt、max_tokens、temperature；
    另加 reasoning_effort（同样影响输出）与版本号 v（请求格式变更时使旧缓存失效）。
    base_url 与 api_key 刻意不进键：它们不改变同一模型的应答语义。
    protocol 进键：协议改变请求形态与响应分布，model + 参数并不唯一确定请求，
    切换协议不得命中另一端点生成的旧缓存（否则 token 与成本统计失真）。
    加入该字段后旧缓存条目因哈希变化不再命中，属预期行为，无需清理。
    """
    canonical = json.dumps(
        {
            "v": _CACHE_SCHEMA_VERSION,
            "model": model,
            "system": system,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "protocol": protocol,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


def _status_detail(exc: httpx.HTTPStatusError) -> str:
    """从 HTTPStatusError 提取安全的原因描述（不含请求头，也就不含 api_key）。"""
    body = exc.response.text.strip()
    detail = f"HTTP {exc.response.status_code}"
    if body:
        detail += f"，正文 {body[:200]!r}"
    return detail


class LLMClient:
    """模型客户端：缓存 → 请求（backend 构造）→ 重试 → 解析（backend 归一）→ 计量 → 成本。"""

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: Transport | None = None,
        backend: Backend | None = None,
        cache_dir: Path | None = None,
        max_retries: int = 3,
    ) -> None:
        self._config = config
        self._transport = transport if transport is not None else HttpxTransport(config.timeout_s)
        # backend 形参用于注入（测试或切换实验）；不传则按 config.protocol 选择默认后端
        self._backend = backend if backend is not None else _default_backend(config.protocol)
        # cache_dir 形参仅用于覆盖配置（测试注入）；不传则用 config.cache_dir
        self._cache_dir = cache_dir if cache_dir is not None else config.cache_dir
        self._max_retries = max_retries
        # 未知模型在构造期失败，而不是烧完一次请求后才失败
        self._prices = prices_for(config.model)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """补全 prompt。prompt_hash = sha256(prompt)，用于日志定位，不回显正文。"""
        started = time.monotonic()
        effective_max_tokens = self._config.max_tokens if max_tokens is None else max_tokens
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        key = _cache_key(
            self._config.model,
            system,
            prompt,
            effective_max_tokens,
            self._config.temperature,
            self._config.reasoning_effort,
            self._config.protocol,
        )
        hit = self._cache_read(key)
        if hit is not None:
            # NFR-2：命中条目费用为 0；token 字段保留原始用量，便于追溯
            return replace(
                hit, cached=True, cost_cny=0.0, latency_ms=_elapsed_ms(started), cache_written=True
            )

        url, payload, headers = self._backend.request(
            self._config, prompt, system, effective_max_tokens
        )
        response, attempts = self._post_with_retry(url, payload, headers, prompt_hash)
        result = self._backend.parse(response)
        if result.stop_reason == "max_tokens" and not result.text:
            raise RuntimeError(
                "llm: stop_reason=max_tokens 且正文为空，模型可能在思考阶段耗尽预算，"
                f"请提高 max_tokens（prompt_hash={prompt_hash}）"
            )
        completion = Completion(
            text=result.text,
            model=self._config.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_input_tokens=result.cached_input_tokens,
            latency_ms=_elapsed_ms(started),
            cached=False,
            stop_reason=result.stop_reason,
            cost_cny=self._cost_cny(
                result.input_tokens, result.output_tokens, result.cached_input_tokens
            ),
            cache_written=False,
        )
        completion = replace(completion, cache_written=self._cache_write(key, completion))
        if attempts > 1:
            logger.info("llm: 经 %d 次尝试成功，prompt_hash=%s", attempts, prompt_hash)
        logger.debug(
            "llm: 调用完成，尝试次数=%d，input=%d，output=%d，cached_input=%d，"
            "cost_cny=%.6f，latency_ms=%.1f，prompt_hash=%s",
            attempts,
            result.input_tokens,
            result.output_tokens,
            result.cached_input_tokens,
            completion.cost_cny,
            completion.latency_ms,
            prompt_hash,
        )
        return completion

    def _post_with_retry(
        self, url: str, payload: dict, headers: dict[str, str], prompt_hash: str
    ) -> tuple[dict, int]:
        """发请求；429 与 5xx 按指数退避重试，最多 max_retries 次。

        返回 (响应, 实际尝试次数)。4xx（除 429）与重试耗尽都抛 RuntimeError。
        """
        attempts = 0
        while True:
            attempts += 1
            try:
                return self._transport.post(url, payload, headers), attempts
            except httpx.HTTPStatusError as exc:
                detail = _status_detail(exc)
                status = exc.response.status_code
                if status != 429 and not 500 <= status <= 599:
                    raise RuntimeError(
                        f"llm: 模型调用失败（{detail}，不重试），prompt_hash={prompt_hash}"
                    ) from exc
                if attempts > self._max_retries:
                    raise RuntimeError(
                        f"llm: 模型调用失败：重试 {self._max_retries} 次后仍失败"
                        f"（共尝试 {attempts} 次），最后错误：{detail}，prompt_hash={prompt_hash}"
                    ) from exc
                delay = _RETRY_BACKOFF_BASE_S * (2 ** (attempts - 1))
                logger.info(
                    "llm: 第 %d 次尝试失败（%s），%.1fs 后重试，prompt_hash=%s",
                    attempts,
                    detail,
                    delay,
                    prompt_hash,
                )
                time.sleep(delay)
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"llm: 模型调用失败（传输层错误：{exc}），prompt_hash={prompt_hash}"
                ) from exc

    def _cost_cny(self, input_tokens: int, output_tokens: int, cached_input_tokens: int) -> float:
        """输入 × 单价 + 输出 × 单价，按百万 tokens 折算；缓存命中部分按 cached 单价。"""
        fresh_input = input_tokens - cached_input_tokens
        return (
            fresh_input * self._prices.input_cny_per_mtok
            + cached_input_tokens * self._prices.cached_cny_per_mtok
            + output_tokens * self._prices.output_cny_per_mtok
        ) / 1_000_000

    def _cache_read(self, key: str) -> Completion | None:
        """读缓存；文件缺失、损坏或字段不全一律视为未命中（宁可重调，不出错值）。"""
        path = self._cache_dir / f"{key}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Completion(
                text=data["text"],
                model=data["model"],
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
                cached_input_tokens=data["cached_input_tokens"],
                latency_ms=0.0,
                cached=False,
                stop_reason=data["stop_reason"],
                cost_cny=data["cost_cny"],
                cache_written=False,
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _cache_write(self, key: str, completion: Completion) -> bool:
        """原子写入缓存（先写 .tmp 再 os.replace），成功返回 True。

        写失败（目录不可创建、磁盘满、权限等）只告警并返回 False，
        不影响已拿到的结果；告警只含缓存文件名，不含 api_key 与 prompt 正文。
        """
        path = self._cache_dir / f"{key}.json"
        tmp = path.parent / (path.name + ".tmp")
        data = {
            "text": completion.text,
            "model": completion.model,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "cached_input_tokens": completion.cached_input_tokens,
            "stop_reason": completion.stop_reason,
            "cost_cny": completion.cost_cny,
        }
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            logger.warning(
                "llm: 缓存写入失败（%s），cache_written=False，本次结果不缓存", path.name
            )
            return False
        return True
