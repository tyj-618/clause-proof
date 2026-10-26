"""judge 测试：提示词构造、输出解析、LLMJudge 集成。全部使用 FakeTransport，不发起真实网络请求。"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import httpx
import pytest

from clauseproof.config import LLMConfig
from clauseproof.judge import CallMeta, Judge, LLMJudge, build_prompt, parse_response
from clauseproof.llm import LLMClient
from clauseproof.models import ClauseKey, Judgment, Presence, Query

# 合成样例合同（非 CUAD 数据，测试与探针共用同一段文本）
_SAMPLE_CONTRACT = (
    'This Master Services Agreement ("Agreement") is entered into as of the Effective Date '
    'by and between Alpha Analytics, Inc., a Delaware corporation ("Provider"), and Beta '
    'Manufacturing Ltd., a company organized under the laws of England and Wales ("Customer").\n\n'
    "1. GOVERNING LAW. This Agreement shall be governed by and construed in accordance with "
    "the laws of the State of New York, without regard to its conflict of laws principles.\n\n"
    "2. LIMITATION OF LIABILITY. IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR ANY "
    "INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION "
    "WITH THIS AGREEMENT. NOTWITHSTANDING THE FOREGOING, EACH PARTY'S TOTAL AGGREGATE LIABILITY "
    "ARISING OUT OF OR RELATED TO THIS AGREEMENT SHALL NOT EXCEED THE AMOUNT OF FEES PAID OR "
    "PAYABLE BY CUSTOMER UNDER THIS AGREEMENT DURING THE TWELVE (12) MONTHS IMMEDIATELY "
    "PRECEDING THE EVENT GIVING RISE TO THE CLAIM."
)

# PRD §3.2 表格三列的期望文本，独立于实现抄录，验证提示词与原文档一致
_EXPECTED_BOUNDARIES: dict[ClauseKey, tuple[str, str, str]] = {
    ClauseKey.GOVERNING_LAW: (
        "明确写出适用某国家/州/联邦法律",
        "全文无管辖法律约定",
        "只写「适用相关法律」；约定多个法域但未定优先顺序；管辖法与仲裁地混写",
    ),
    ClauseKey.CAP_ON_LIABILITY: (
        "对赔偿总额设明确上限（金额 / 倍数 / 期间费用）",
        "无任何责任限制表述",
        "有上限但同时列出不受上限约束的例外情形",
    ),
    ClauseKey.UNCAP_LIABILITY: (
        "明确指出某些情形不受责任上限约束（故意、重大过失、保密、IP 侵权等）",
        "无任何「不受限」表述",
        "表述含糊（「在法律允许的最大范围内」）",
    ),
    ClauseKey.ANTI_ASSIGNMENT: (
        "限制转让合同或其权利（需对方同意 / 绝对禁止 / 限制控制权变更）",
        "未限制转让",
        "只限制部分权利；限制与例外并存；仅限制「控制权变更」而未提合同转让",
    ),
    ClauseKey.TERMINATION_FOR_CONVENIENCE: (
        "一方可在无违约、无需理由的情况下单方终止",
        "只能因违约终止，或仅到期终止",
        "需提前通知但未说明是否需理由；仅特定情形可终止",
    ),
    ClauseKey.NOTICE_PERIOD_TO_TERMINATE_RENEWAL: (
        "有自动续约条款且给出明确通知期",
        "无自动续约，或未约定通知期",
        "有自动续约但通知期需推算（「提前一个续约期」）",
    ),
}

_ABSENT_JSON = '{"present": "absent", "quotes": [], "confidence": 0.4}'
_PRESENT_JSON = '{"present": "present", "quotes": ["alpha beta"], "confidence": 0.85}'


def _query(
    clause: ClauseKey = ClauseKey.CAP_ON_LIABILITY, contract: str = _SAMPLE_CONTRACT
) -> Query:
    return Query(
        query_id=f"sample::{clause.name}",
        contract_id="sample",
        clause=clause,
        contract_text=contract,
    )


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
        cache_dir=Path(tempfile.mkdtemp(prefix="clauseproof-judge-cache-")),
    )
    values.update(overrides)
    return LLMConfig(**values)


def _response(
    text: str,
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> dict:
    return {
        "model": "glm-5.3",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": 0,
        },
    }


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://fake.test/v1/messages")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


class FakeTransport:
    """测试替身：返回固定响应或按脚本抛异常；记录每次调用的 url 与 payload。"""

    def __init__(self, response: dict, failures: list[Exception] | None = None) -> None:
        self._response = response
        self._failures = list(failures or [])
        self.calls: list[tuple[str, dict]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def post(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        self.calls.append((url, payload))
        if self._failures:
            raise self._failures.pop(0)
        return self._response


# ---- build_prompt：提示词 ----


def test_build_prompt_is_deterministic() -> None:
    query = _query()
    assert build_prompt(query) == build_prompt(query)


def test_system_sets_role_and_output_format() -> None:
    system, _ = build_prompt(_query())
    assert "合同初筛助手" in system
    assert "只输出一个 JSON 对象" in system
    expected_schema = (
        '{"present": "present|absent|uncertain", "quotes": ["...", "..."], "confidence": 0.0}'
    )
    assert expected_schema in system
    # system 定角色与格式，不给材料：合同全文与其片段都不得出现在 system
    assert _SAMPLE_CONTRACT not in system
    assert "SHALL NOT EXCEED" not in system


def test_system_restricts_present_to_three_values() -> None:
    system, _ = build_prompt(_query())
    assert (
        '- "present" 只能取 "present"（有）、"absent"（没有）、"uncertain"（模糊）三个值之一'
        in system
    )


def test_system_requires_verbatim_quotes() -> None:
    system, _ = build_prompt(_query())
    assert "逐字复制" in system
    assert "不得改写、不得省略、不得添加省略号、不得添加引号" in system


def test_system_quotes_rules_by_presence() -> None:
    system, _ = build_prompt(_query())
    assert '判定为 "present" 时，"quotes" 不得为空' in system
    assert "该条款在合同中多处出现时，须摘录全部出现位置" in system
    assert '判定为 "absent" 时，"quotes" 必须为空数组' in system
    assert '判定为 "uncertain" 时，"quotes" 可以为空数组' in system


def test_user_carries_contract_and_clause_name() -> None:
    query = _query()
    _, user = build_prompt(query)
    assert query.contract_text in user
    assert query.clause.value in user
    assert "<contract>" in user
    assert "</contract>" in user


@pytest.mark.parametrize("clause", list(ClauseKey))
def test_user_carries_prd_boundary_for_every_clause(clause: ClauseKey) -> None:
    _, user = build_prompt(_query(clause))
    for cell in _EXPECTED_BOUNDARIES[clause]:
        assert cell in user
    # 只嵌入当前条款的边界，不串类
    assert user.count("算「有」") == 1
    assert user.count("算「没有」") == 1
    assert user.count("算「模糊」") == 1


def test_answer_text_appears_only_inside_contract() -> None:
    # 合成样例中充当「答案」的句子只应作为合同全文的一部分进入提示词；
    # Query 结构上不含标注信息，system 与任务指令不得单独复述该句
    sentence = "SHALL NOT EXCEED THE AMOUNT OF FEES"
    system, user = build_prompt(_query())
    assert sentence not in system
    assert user.count(sentence) == 1


# ---- parse_response：解析 ----


def _assert_fallback(judgment: Judgment, raw: str) -> None:
    assert judgment.parse_ok is False
    assert judgment.present is Presence.UNCERTAIN
    assert judgment.quotes == ()
    assert judgment.confidence == 0.0
    assert judgment.raw_output == raw


def test_parses_bare_json() -> None:
    judgment = parse_response(_PRESENT_JSON)
    assert judgment.parse_ok is True
    assert judgment.present is Presence.PRESENT
    assert judgment.quotes == ("alpha beta",)
    assert judgment.confidence == 0.85


def test_parses_fenced_json() -> None:
    text = f"```json\n{_ABSENT_JSON}\n```"
    judgment = parse_response(text)
    assert judgment.parse_ok is True
    assert judgment.present is Presence.ABSENT
    assert judgment.quotes == ()
    assert judgment.confidence == 0.4
    assert judgment.raw_output == text


def test_parses_json_with_surrounding_prose() -> None:
    text = f"好的，以下是判定结果：{_ABSENT_JSON} 以上供参考。"
    judgment = parse_response(text)
    assert judgment.parse_ok is True
    assert judgment.present is Presence.ABSENT
    assert judgment.raw_output == text


def test_prose_with_extra_braces_falls_back() -> None:
    # 首个 { 到末个 } 之间混入非 JSON 文本时切片不可解析，按规则降级
    text = f'格式示例 {{"present": "..."}}，判定：{_ABSENT_JSON}'
    _assert_fallback(parse_response(text), text)


@pytest.mark.parametrize(
    "text",
    [
        # 非 JSON
        "",
        "模型拒绝了这次请求",
        "{ 没有闭合的括号",
        "}",
        # 顶层不是对象
        "[1, 2, 3]",
        "null",
        # 缺任一必需字段
        '{"quotes": [], "confidence": 0.5}',
        '{"present": "absent", "confidence": 0.5}',
        '{"present": "absent", "quotes": []}',
        # present 取值不在三值内
        '{"present": "maybe", "quotes": [], "confidence": 0.5}',
        '{"present": 1, "quotes": [], "confidence": 0.5}',
        # confidence 非数字、为布尔、越界
        '{"present": "absent", "quotes": [], "confidence": "high"}',
        '{"present": "absent", "quotes": [], "confidence": true}',
        '{"present": "absent", "quotes": [], "confidence": -0.1}',
        '{"present": "absent", "quotes": [], "confidence": 1.1}',
        # quotes 不是字符串数组
        '{"present": "absent", "quotes": "none", "confidence": 0.5}',
        '{"present": "present", "quotes": [1, 2], "confidence": 0.5}',
        '{"present": "present", "quotes": ["a", null], "confidence": 0.5}',
        # 违反 Judgment 不变式
        '{"present": "absent", "quotes": ["x"], "confidence": 0.5}',
        '{"present": "present", "quotes": [], "confidence": 0.5}',
        # JSON 语法错误
        '{"present": "absent" "quotes": []}',
    ],
)
def test_invalid_inputs_fall_back(text: str) -> None:
    _assert_fallback(parse_response(text), text)


def test_confidence_accepts_integer_bounds() -> None:
    low = parse_response('{"present": "absent", "quotes": [], "confidence": 0}')
    high = parse_response('{"present": "absent", "quotes": [], "confidence": 1}')
    assert low.parse_ok is True
    assert low.confidence == 0.0
    assert high.parse_ok is True
    assert high.confidence == 1.0


def test_uncertain_allows_empty_and_nonempty_quotes() -> None:
    # 0002 §7：UNCERTAIN 对 quotes 不设约束
    empty = parse_response('{"present": "uncertain", "quotes": [], "confidence": 0.6}')
    nonempty = parse_response('{"present": "uncertain", "quotes": ["x"], "confidence": 0.6}')
    assert empty.parse_ok is True and empty.quotes == ()
    assert nonempty.parse_ok is True and nonempty.quotes == ("x",)


def test_extra_fields_are_tolerated() -> None:
    text = '{"present": "absent", "quotes": [], "confidence": 0.5, "note": "extra"}'
    judgment = parse_response(text)
    assert judgment.parse_ok is True
    assert judgment.present is Presence.ABSENT


def test_quotes_with_escapes_and_unicode() -> None:
    text = '{"present": "present", "quotes": ["\\"quoted\\" 中文"], "confidence": 0.9}'
    judgment = parse_response(text)
    assert judgment.parse_ok is True
    assert judgment.quotes == ('"quoted" 中文',)


def test_parse_response_is_pure() -> None:
    assert parse_response(_PRESENT_JSON) == parse_response(_PRESENT_JSON)


# ---- LLMJudge：集成 ----


def test_llm_judge_satisfies_protocol() -> None:
    judge = LLMJudge(LLMClient(_config(), transport=FakeTransport(_response(_ABSENT_JSON))))
    assert isinstance(judge, Judge)


def test_judge_sends_prompt_via_client() -> None:
    transport = FakeTransport(_response(_ABSENT_JSON))
    query = _query()
    system, user = build_prompt(query)
    LLMJudge(LLMClient(_config(), transport=transport)).judge(query)
    assert transport.call_count == 1
    url, payload = transport.calls[0]
    assert url == "https://fake.test/api/anthropic/v1/messages"
    assert payload["system"] == system
    assert payload["messages"] == [{"role": "user", "content": user}]


def test_judge_maps_completion_to_call_meta() -> None:
    transport = FakeTransport(_response(_ABSENT_JSON, input_tokens=123, output_tokens=45))
    query = _query()
    system, user = build_prompt(query)
    judgment, meta = LLMJudge(LLMClient(_config(), transport=transport)).judge(query)
    assert judgment.parse_ok is True
    assert isinstance(meta, CallMeta)
    assert meta.model == "glm-5.3"
    assert meta.input_tokens == 123
    assert meta.output_tokens == 45
    assert meta.latency_ms >= 0.0
    assert meta.cached is False
    # 0002 §8：prompt_hash 对 system 与 user 以两个换行拼接后的完整 prompt 取 SHA-256
    joined = f"{system}\n\n{user}"
    expected = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    assert meta.prompt_hash == expected
    # 与「只哈希 user」区分开，确认 system 也计入
    assert meta.prompt_hash != hashlib.sha256(user.encode("utf-8")).hexdigest()


def test_judge_returns_parsed_judgment() -> None:
    raw = (
        '{"present": "present", "quotes": ["SHALL NOT EXCEED THE AMOUNT OF FEES"], '
        '"confidence": 0.9}'
    )
    judgment, _ = LLMJudge(LLMClient(_config(), transport=FakeTransport(_response(raw)))).judge(
        _query()
    )
    assert judgment.parse_ok is True
    assert judgment.present is Presence.PRESENT
    assert judgment.quotes == ("SHALL NOT EXCEED THE AMOUNT OF FEES",)
    assert judgment.raw_output == raw


def test_judge_degrades_unparseable_output() -> None:
    raw = "This contract does contain a cap on liability."
    transport = FakeTransport(_response(raw, input_tokens=77, output_tokens=9))
    judgment, meta = LLMJudge(LLMClient(_config(), transport=transport)).judge(_query())
    _assert_fallback(judgment, raw)
    # 解析失败不影响调用元信息
    assert meta.input_tokens == 77
    assert meta.output_tokens == 9


def test_judge_propagates_runtime_error() -> None:
    transport = FakeTransport(_response(_ABSENT_JSON), failures=[_http_error(400)])
    judge = LLMJudge(LLMClient(_config(), transport=transport))
    with pytest.raises(RuntimeError, match="400"):
        judge.judge(_query())
    # 4xx 不重试是 llm 层契约；此处同时证明 judge 层也没有重试
    assert transport.call_count == 1


def test_max_tokens_forwarded_to_client() -> None:
    transport = FakeTransport(_response(_ABSENT_JSON))
    LLMJudge(LLMClient(_config(), transport=transport), max_tokens=99).judge(_query())
    assert transport.calls[0][1]["max_tokens"] == 99


def test_max_tokens_defaults_to_client_config() -> None:
    transport = FakeTransport(_response(_ABSENT_JSON))
    LLMJudge(LLMClient(_config(max_tokens=2048), transport=transport)).judge(_query())
    assert transport.calls[0][1]["max_tokens"] == 2048


def test_cached_flag_maps_on_cache_hit(tmp_path: Path) -> None:
    # 缓存是 LLMClient 的职责；此处只验证 CallMeta.cached 忠实映射 Completion.cached
    transport = FakeTransport(_response(_ABSENT_JSON))
    judge = LLMJudge(LLMClient(_config(cache_dir=tmp_path / "cache"), transport=transport))
    _, first = judge.judge(_query())
    _, second = judge.judge(_query())
    assert transport.call_count == 1
    assert first.cached is False
    assert second.cached is True
