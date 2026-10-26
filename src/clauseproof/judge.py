"""判定器：提示词构造、模型输出解析与 LLMJudge 集成，兼 Judge 协议与 CallMeta 定义。

契约见 docs/design/0003-interfaces.md §5 与 tasks/T-003.md。六类条款的判定边界逐字取自
docs/prd/0001-contract-clause-review.md §3.2，不得增删语义。解析失败一律降级为
parse_ok=False 的 UNCERTAIN，不抛异常（0003 §5 后置条件）；LLMClient 抛出的
RuntimeError 不在本层捕获，由 runner 转失败记录（0004 §2 单条层）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from clauseproof.llm import LLMClient
from clauseproof.models import ClauseKey, Judgment, Presence, Query


@dataclass(frozen=True, slots=True)
class CallMeta:
    """一次模型调用的计量信息。"""

    model: str
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool


@runtime_checkable
class Judge(Protocol):
    """评审器协议：输入 Query，输出判定与调用元信息。"""

    def judge(self, query: Query) -> tuple[Judgment, CallMeta]: ...


# 六类条款的三态边界，逐字取自 PRD §3.2 表格的「算有 / 算没有 / 算模糊」三列，不得增删语义
_BOUNDARIES: dict[ClauseKey, str] = {
    ClauseKey.GOVERNING_LAW: (
        "- 算「有」：明确写出适用某国家/州/联邦法律\n"
        "- 算「没有」：全文无管辖法律约定\n"
        "- 算「模糊」：只写「适用相关法律」；约定多个法域但未定优先顺序；管辖法与仲裁地混写"
    ),
    ClauseKey.CAP_ON_LIABILITY: (
        "- 算「有」：对赔偿总额设明确上限（金额 / 倍数 / 期间费用）\n"
        "- 算「没有」：无任何责任限制表述\n"
        "- 算「模糊」：有上限但同时列出不受上限约束的例外情形"
    ),
    ClauseKey.UNCAP_LIABILITY: (
        "- 算「有」：明确指出某些情形不受责任上限约束（故意、重大过失、保密、IP 侵权等）\n"
        "- 算「没有」：无任何「不受限」表述\n"
        "- 算「模糊」：表述含糊（「在法律允许的最大范围内」）"
    ),
    ClauseKey.ANTI_ASSIGNMENT: (
        "- 算「有」：限制转让合同或其权利（需对方同意 / 绝对禁止 / 限制控制权变更）\n"
        "- 算「没有」：未限制转让\n"
        "- 算「模糊」：只限制部分权利；限制与例外并存；仅限制「控制权变更」而未提合同转让"
    ),
    ClauseKey.TERMINATION_FOR_CONVENIENCE: (
        "- 算「有」：一方可在无违约、无需理由的情况下单方终止\n"
        "- 算「没有」：只能因违约终止，或仅到期终止\n"
        "- 算「模糊」：需提前通知但未说明是否需理由；仅特定情形可终止"
    ),
    ClauseKey.NOTICE_PERIOD_TO_TERMINATE_RENEWAL: (
        "- 算「有」：有自动续约条款且给出明确通知期\n"
        "- 算「没有」：无自动续约，或未约定通知期\n"
        "- 算「模糊」：有自动续约但通知期需推算（「提前一个续约期」）"
    ),
}


def build_prompt(query: Query) -> tuple[str, str]:
    """纯函数，返回 (system, user)。不读文件、不调模型。"""
    clause_name = query.clause.value
    system = (
        "你是合同初筛助手。你的任务：针对给定的一类条款，判定它在一篇英文合同中是否存在，"
        "并从合同原文摘录引用片段。\n\n"
        "只输出一个 JSON 对象，除此之外不得输出任何文字（包括解释、说明或代码围栏）。"
        "JSON 结构：\n"
        '{"present": "present|absent|uncertain", "quotes": ["...", "..."], "confidence": 0.0}\n\n'
        "字段规则：\n"
        '- "present" 只能取 "present"（有）、"absent"（没有）、"uncertain"（模糊）三个值之一\n'
        '- "quotes" 是字符串数组，每个元素必须是从合同原文逐字复制的连续片段：'
        "不得改写、不得省略、不得添加省略号、不得添加引号\n"
        '- 判定为 "present" 时，"quotes" 不得为空；该条款在合同中多处出现时，'
        "须摘录全部出现位置\n"
        '- 判定为 "absent" 时，"quotes" 必须为空数组\n'
        '- 判定为 "uncertain" 时，"quotes" 可以为空数组\n'
        '- "confidence" 是 0 到 1 之间的小数，表示对本次判定的确信程度'
    )
    user = (
        f"请核查以下合同中的「{clause_name}」条款。\n\n"
        "【该条款的判定边界】\n"
        f"{_BOUNDARIES[query.clause]}\n\n"
        "【合同全文】\n"
        f"<contract>\n{query.contract_text}\n</contract>\n\n"
        f"请依据上述判定边界核查合同全文，判定「{clause_name}」条款的存在性，"
        "摘录全部相关原文片段，并只输出一个 JSON 对象。"
    )
    return system, user


def parse_response(text: str) -> Judgment:
    """解析模型输出。任何失败都返回 parse_ok=False 的 UNCERTAIN，不抛异常。"""
    fallback = Judgment(
        present=Presence.UNCERTAIN,
        quotes=(),
        confidence=0.0,
        raw_output=text,
        parse_ok=False,
    )
    # 提取规则（tasks/T-003）：取首个 { 到末个 }，容忍围栏与前后说明文字
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return fallback
    try:
        data = json.loads(text[start : end + 1])
    except (ValueError, RecursionError):
        return fallback
    if not isinstance(data, dict):
        return fallback
    try:
        present = Presence(data["present"])
    except (KeyError, TypeError, ValueError):
        return fallback
    confidence = data.get("confidence")
    # bool 是 int 的子类，必须先排除，true / false 一律视为非数字
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return fallback
    if not 0.0 <= confidence <= 1.0:  # NaN 与越界值在此一并拒绝
        return fallback
    quotes_raw = data.get("quotes")
    if not isinstance(quotes_raw, list) or not all(isinstance(item, str) for item in quotes_raw):
        return fallback
    quotes = tuple(quotes_raw)
    # 违反 Judgment 不变式（0002 §7）的组合一律降级，保证本函数不抛 ValueError
    if present is Presence.ABSENT and quotes:
        return fallback
    if present is Presence.PRESENT and not quotes:
        return fallback
    return Judgment(
        present=present,
        quotes=quotes,
        confidence=float(confidence),
        raw_output=text,
        parse_ok=True,
    )


def _prompt_hash(system: str, user: str) -> str:
    """0002 §8：对实际发送的完整 prompt（system 与 user 用两个换行拼接）取 SHA-256。"""
    joined = f"{system}\n\n{user}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class LLMJudge:
    """判定器实现：构造提示词 → 单次调用 LLMClient → 解析输出。

    只做单次调用与字段映射；重试、缓存、成本计算都在 LLMClient（0003 §10），
    不属于判定层。LLMClient 抛出的 RuntimeError 不捕获，直接上抛给 runner。
    """

    def __init__(self, client: LLMClient, *, max_tokens: int | None = None) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def judge(self, query: Query) -> tuple[Judgment, CallMeta]:
        system, user = build_prompt(query)
        completion = self._client.complete(user, system=system, max_tokens=self._max_tokens)
        judgment = parse_response(completion.text)
        meta = CallMeta(
            model=completion.model,
            prompt_hash=_prompt_hash(system, user),
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.latency_ms,
            cached=completion.cached,
        )
        return judgment, meta
