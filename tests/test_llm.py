"""llm 测试：缓存、重试、计量、输出异常与安全。全部使用 FakeTransport，不发起真实网络请求。"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from clauseproof import llm
from clauseproof.config import LLMConfig
from clauseproof.llm import LLMClient, Transport


def _config(**overrides: object) -> LLMConfig:
    values: dict = dict(
        base_url="https://fake.test/api/anthropic",
        api_key="sk-fake-secret",
        model="glm-5.3",
        max_tokens=1024,
        reasoning_effort="low",
        temperature=0.0,
        timeout_s=5.0,
        # 每次全新目录：不显式覆盖 cache_dir 的用例不会命中彼此或上一次运行的缓存
        cache_dir=Path(tempfile.mkdtemp(prefix="clauseproof-test-cache-")),
    )
    values.update(overrides)
    return LLMConfig(**values)


def _response(
    text: str = "判定结果",
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cached_input: int = 0,
    stop_reason: str | None = "end_turn",
) -> dict:
    return {
        "model": "glm-5.3",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cached_input,
        },
    }


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://fake.test/v1/messages")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


class FakeTransport:
    """测试替身：先按脚本抛异常，之后返回固定响应；记录每次调用的 url / payload / headers。"""

    def __init__(self, response: dict, failures: list[Exception] | None = None) -> None:
        self._response = response
        self._failures = list(failures or [])
        self.calls: list[tuple[str, dict, dict[str, str]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def post(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        self.calls.append((url, payload, dict(headers)))
        if self._failures:
            raise self._failures.pop(0)
        return self._response


# ---- 协议与构造 ----


def test_transports_satisfy_protocol() -> None:
    assert isinstance(FakeTransport(_response()), Transport)


def test_unknown_model_fails_at_construction() -> None:
    with pytest.raises(RuntimeError, match="gpt-4o"):
        LLMClient(_config(model="gpt-4o"), transport=FakeTransport(_response()))


# ---- 请求形态 ----


def test_request_payload_shape() -> None:
    transport = FakeTransport(_response())
    config = _config(max_tokens=777, temperature=0.25, reasoning_effort="high")
    LLMClient(config, transport=transport).complete("用户问题", system="系统提示")
    url, payload, headers = transport.calls[0]
    assert url == "https://fake.test/api/anthropic/v1/messages"
    assert payload["model"] == "glm-5.3"
    assert payload["max_tokens"] == 777
    assert payload["temperature"] == 0.25
    assert payload["reasoning_effort"] == "high"
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["system"] == "系统提示"
    assert payload["messages"] == [{"role": "user", "content": "用户问题"}]
    assert headers["Authorization"] == "Bearer sk-fake-secret"
    assert "anthropic-version" in headers


def test_max_tokens_override_changes_payload() -> None:
    transport = FakeTransport(_response())
    LLMClient(_config(max_tokens=777), transport=transport).complete("问题", max_tokens=99)
    assert transport.calls[0][1]["max_tokens"] == 99


def test_system_omitted_when_none() -> None:
    transport = FakeTransport(_response())
    LLMClient(_config(), transport=transport).complete("问题")
    assert "system" not in transport.calls[0][1]


# ---- 缓存 ----


def test_second_call_hits_cache_without_request(tmp_path: Path) -> None:
    transport = FakeTransport(_response())
    client = LLMClient(_config(), transport=transport, cache_dir=tmp_path / "cache")
    first = client.complete("合同条款……")
    second = client.complete("合同条款……")
    assert transport.call_count == 1
    assert first.cached is False
    assert first.cache_written is True
    assert second.cached is True
    assert second.cache_written is True
    assert second.text == first.text


def test_cache_hit_zero_cost_keeps_original_usage(tmp_path: Path) -> None:
    response = _response(input_tokens=2_000_000, output_tokens=1_000_000, cached_input=500_000)
    transport = FakeTransport(response)
    client = LLMClient(_config(), transport=transport, cache_dir=tmp_path / "cache")
    first = client.complete("合同条款……")
    second = client.complete("合同条款……")
    # NFR-2：命中条目费用为 0，但 token 字段保留原始用量便于追溯
    assert second.cached is True
    assert second.cost_cny == 0.0
    assert second.input_tokens == first.input_tokens == 2_000_000
    assert second.output_tokens == first.output_tokens == 1_000_000
    assert second.cached_input_tokens == first.cached_input_tokens == 500_000
    # 两次调用的累计成本 = 第一次的成本，不是两倍
    expected_first = 1_500_000 * 8 / 1e6 + 500_000 * 2 / 1e6 + 1_000_000 * 28 / 1e6
    assert first.cost_cny == pytest.approx(expected_first, abs=1e-9)
    assert first.cost_cny + second.cost_cny == pytest.approx(expected_first, abs=1e-9)


def test_cache_survives_new_client(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    first = LLMClient(
        _config(), transport=FakeTransport(_response()), cache_dir=cache_dir
    ).complete("同一问题")
    assert first.cached is False
    transport2 = FakeTransport(_response(text="不该出现"))
    second = LLMClient(_config(), transport=transport2, cache_dir=cache_dir).complete("同一问题")
    assert transport2.call_count == 0
    assert second.cached is True
    assert second.text == "判定结果"


def test_config_cache_dir_used_when_override_absent(tmp_path: Path) -> None:
    cfg_cache = tmp_path / "cfg-cache"
    transport = FakeTransport(_response())
    client = LLMClient(_config(cache_dir=cfg_cache), transport=transport)
    first = client.complete("合同条款……")
    second = client.complete("合同条款……")
    assert transport.call_count == 1
    assert first.cached is False
    assert first.cache_written is True
    assert second.cached is True
    assert second.cache_written is True
    assert len(list(cfg_cache.glob("*.json"))) == 1


def test_cache_dir_parameter_overrides_config(tmp_path: Path) -> None:
    cfg_cache = tmp_path / "cfg-cache"
    override = tmp_path / "override"
    transport = FakeTransport(_response())
    client = LLMClient(_config(cache_dir=cfg_cache), transport=transport, cache_dir=override)
    completion = client.complete("合同条款……")
    assert completion.cache_written is True
    assert len(list(override.glob("*.json"))) == 1
    assert not cfg_cache.exists()


def test_unwritable_cache_dir_still_returns_result(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("这是一个普通文件，占住缓存目录的位置", encoding="utf-8")
    transport = FakeTransport(_response())
    client = LLMClient(_config(), transport=transport, cache_dir=blocker)
    completion = client.complete("合同条款……")
    assert completion.text == "判定结果"
    assert completion.cached is False
    assert completion.cache_written is False
    assert transport.call_count == 1


def test_write_failure_warning_has_no_secrets(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="clauseproof.llm")
    blocker = tmp_path / "blocker"
    blocker.write_text("这是一个普通文件", encoding="utf-8")
    prompt = "机密合同条款XYZ789"
    config = _config(api_key="sk-fake-secret")
    completion = LLMClient(
        config, transport=FakeTransport(_response()), cache_dir=blocker
    ).complete(prompt)
    assert completion.cache_written is False
    assert "缓存写入失败" in caplog.text
    assert "sk-fake-secret" not in caplog.text
    assert prompt not in caplog.text


def test_cache_key_covers_all_components() -> None:
    base = llm._cache_key("glm-5.3", None, "prompt", 1024, 0.0, "low")
    assert base == llm._cache_key("glm-5.3", None, "prompt", 1024, 0.0, "low")
    assert base != llm._cache_key("glm-5.3", "system", "prompt", 1024, 0.0, "low")
    assert base != llm._cache_key("glm-5.3", None, "other", 1024, 0.0, "low")
    assert base != llm._cache_key("glm-5.3", None, "prompt", 2048, 0.0, "low")
    assert base != llm._cache_key("glm-5.3", None, "prompt", 1024, 0.7, "low")
    assert base != llm._cache_key("other-model", None, "prompt", 1024, 0.0, "low")
    assert base != llm._cache_key("glm-5.3", None, "prompt", 1024, 0.0, "high")


def test_any_parameter_change_misses_cache(tmp_path: Path) -> None:
    transport = FakeTransport(_response())
    client = LLMClient(_config(), transport=transport, cache_dir=tmp_path / "cache")
    client.complete("prompt", system="sys-a", max_tokens=100)
    assert transport.call_count == 1
    client.complete("prompt", system="sys-b", max_tokens=100)  # system 变化
    client.complete("prompt", system="sys-b", max_tokens=200)  # max_tokens 变化
    client.complete("changed", system="sys-b", max_tokens=200)  # prompt 变化
    assert transport.call_count == 4
    other = LLMClient(_config(temperature=0.7), transport=transport, cache_dir=tmp_path / "cache")
    result = other.complete("changed", system="sys-b", max_tokens=200)  # temperature 变化
    assert transport.call_count == 5
    assert result.cached is False


def test_corrupt_cache_entry_is_miss(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = llm._cache_key("glm-5.3", None, "问题", 1024, 0.0, "low")
    (cache_dir / f"{key}.json").write_text("{ 不是 JSON", encoding="utf-8")
    transport = FakeTransport(_response())
    completion = LLMClient(_config(), transport=transport, cache_dir=cache_dir).complete("问题")
    assert transport.call_count == 1
    assert completion.cached is False


# ---- 重试 ----


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retries_on_429_and_5xx_then_succeeds(status: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    failures = [_http_error(status), _http_error(status)]
    transport = FakeTransport(_response(), failures=failures)
    completion = LLMClient(_config(), transport=transport).complete("问题")
    assert transport.call_count == 3
    assert completion.text == "判定结果"
    assert completion.cached is False


def test_retry_records_attempt_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="clauseproof.llm")
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    transport = FakeTransport(_response(), failures=[_http_error(429)])
    LLMClient(_config(), transport=transport).complete("问题")
    assert "2 次尝试" in caplog.text


def test_retry_exhaustion_raises_with_last_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    transport = FakeTransport(_response(), failures=[_http_error(429)] * 4)
    client = LLMClient(_config(), transport=transport, max_retries=3)
    with pytest.raises(RuntimeError) as excinfo:
        client.complete("问题")
    message = str(excinfo.value)
    assert "429" in message
    assert transport.call_count == 4  # 1 次首发 + 3 次重试


@pytest.mark.parametrize("status", [400, 404])
def test_client_errors_not_retried(status: int) -> None:
    transport = FakeTransport(_response(), failures=[_http_error(status)])
    client = LLMClient(_config(), transport=transport)
    with pytest.raises(RuntimeError, match=str(status)):
        client.complete("问题")
    assert transport.call_count == 1


def test_backoff_increases_exponentially(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", delays.append)
    transport = FakeTransport(_response(), failures=[_http_error(429)] * 3)
    LLMClient(_config(), transport=transport, max_retries=3).complete("问题")
    assert delays == [0.5, 1.0, 2.0]


def test_latency_includes_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    real_sleep = time.sleep
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: real_sleep(0.05))
    transport = FakeTransport(_response(), failures=[_http_error(429), _http_error(429)])
    completion = LLMClient(_config(), transport=transport).complete("问题")
    assert completion.latency_ms >= 100.0


# ---- 计量 ----


def test_usage_extraction_and_cost() -> None:
    transport = FakeTransport(_response(input_tokens=1_000_000, output_tokens=1_000_000))
    completion = LLMClient(_config(), transport=transport).complete("问题")
    assert completion.model == "glm-5.3"
    assert completion.input_tokens == 1_000_000
    assert completion.output_tokens == 1_000_000
    assert completion.cached_input_tokens == 0
    # cached_input=0 退化为全价：100 万输入 × ¥8 + 100 万输出 × ¥28 = ¥36
    assert completion.cost_cny == pytest.approx(36.0, abs=1e-9)


def test_cost_uses_cached_price_for_cached_input() -> None:
    response = _response(input_tokens=1_000_000, output_tokens=1_000_000, cached_input=500_000)
    transport = FakeTransport(response)
    completion = LLMClient(_config(), transport=transport).complete("问题")
    assert completion.cached_input_tokens == 500_000
    # 50 万新输入 × ¥8 + 50 万缓存命中 × ¥2 + 100 万输出 × ¥28，除以 1e6 = ¥33
    assert completion.cost_cny == pytest.approx(33.0, abs=1e-9)


def test_stop_reason_passthrough_and_missing() -> None:
    transport = FakeTransport(_response(stop_reason="stop_sequence"))
    client = LLMClient(_config(), transport=transport)
    assert client.complete("问题").stop_reason == "stop_sequence"
    response = _response()
    del response["stop_reason"]
    other = LLMClient(_config(), transport=FakeTransport(response))
    assert other.complete("问题").stop_reason is None


def test_text_extraction_ignores_non_text_blocks() -> None:
    response = _response()
    response["content"] = [
        {"type": "thinking", "thinking": "推理过程"},
        {"type": "text", "text": "第一段"},
        {"type": "text", "text": "第二段"},
    ]
    transport = FakeTransport(response)
    completion = LLMClient(_config(), transport=transport).complete("问题")
    assert completion.text == "第一段第二段"


# ---- 输出异常 ----


def test_max_tokens_with_empty_text_raises() -> None:
    transport = FakeTransport(_response(text="", stop_reason="max_tokens"))
    client = LLMClient(_config(), transport=transport)
    with pytest.raises(RuntimeError) as excinfo:
        client.complete("问题")
    message = str(excinfo.value)
    assert "思考阶段耗尽预算" in message
    assert "提高 max_tokens" in message


def test_max_tokens_with_text_returns_normally() -> None:
    transport = FakeTransport(_response(text="部分输出", stop_reason="max_tokens"))
    completion = LLMClient(_config(), transport=transport).complete("问题")
    assert completion.text == "部分输出"
    assert completion.stop_reason == "max_tokens"


# ---- 安全 ----


def test_api_key_never_in_errors_or_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="clauseproof.llm")
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    config = _config(api_key="sk-fake-secret")

    exhausted = FakeTransport(_response(), failures=[_http_error(429)] * 4)
    with pytest.raises(RuntimeError):
        LLMClient(config, transport=exhausted, max_retries=3).complete("机密合同 prompt")
    forbidden = FakeTransport(_response(), failures=[_http_error(401)])
    with pytest.raises(RuntimeError):
        LLMClient(config, transport=forbidden).complete("机密合同 prompt")
    empty = FakeTransport(_response(text="", stop_reason="max_tokens"))
    with pytest.raises(RuntimeError):
        LLMClient(config, transport=empty).complete("机密合同 prompt")
    retried = FakeTransport(_response(), failures=[_http_error(429)])
    LLMClient(config, transport=retried).complete("机密合同 prompt")

    assert "sk-fake-secret" not in caplog.text
    # key 确实发给了传输层：证明上面查的是「带了 key 但没泄漏」，而不是「根本没带」
    assert retried.calls[0][2]["Authorization"] == "Bearer sk-fake-secret"


def test_prompt_text_never_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="clauseproof.llm")
    prompt = "绝密条款内容XYZ789"
    transport = FakeTransport(_response())
    completion = LLMClient(_config(), transport=transport).complete(prompt)
    assert completion.text == "判定结果"
    assert prompt not in caplog.text
    # 调试信息只带 prompt_hash（sha256，64 位十六进制）
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() in caplog.text
