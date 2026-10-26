"""数据模型层：全部枚举与不可变数据结构。

契约见 docs/design/0002-data-model.md §2–§6，不变式见其 §7，
错误信息格式见 docs/design/0004-error-semantics.md §8。
只做构造期校验，不含业务逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NoReturn

# 以下常量均来自 0002 §7 的不变式定义，非可调参数
_QUERY_ID_SEP = "::"
_SEED_MAX = 2**63
_SPLITS = ("dev", "report")


def _violation(field: str, constraint: str, expected: object, actual: object) -> NoReturn:
    """按 0004 §8 的四要素格式抛出 ValueError。"""
    raise ValueError(f"models: {field} 违反 {constraint}，期望 {expected}，实际 {actual!r}")


class ClauseKey(Enum):
    """六类条款。全项目唯一允许出现 CUAD 原始名称的地方。"""

    GOVERNING_LAW = "Governing Law"
    CAP_ON_LIABILITY = "Cap On Liability"
    UNCAP_LIABILITY = "Uncapped Liability"
    ANTI_ASSIGNMENT = "Anti-Assignment"
    TERMINATION_FOR_CONVENIENCE = "Termination For Convenience"
    NOTICE_PERIOD_TO_TERMINATE_RENEWAL = "Notice Period To Terminate Renewal"


class Presence(Enum):
    """条款存在性判定。"""

    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class Route(Enum):
    """路由结果：自动通过 / 转人工。"""

    AUTO_PASS = "auto_pass"
    TO_HUMAN = "to_human"


class Status(Enum):
    """单条执行状态。"""

    OK = "ok"
    LLM_ERROR = "llm_error"
    CONTEXT_OVERFLOW = "context_overflow"


@dataclass(frozen=True, slots=True)
class Query:
    """模型看到的全部信息。刻意不含 gold。"""

    query_id: str
    contract_id: str
    clause: ClauseKey
    contract_text: str

    def __post_init__(self) -> None:
        if not self.contract_id:
            _violation("Query.contract_id", "非空", "非空字符串", self.contract_id)
        if _QUERY_ID_SEP in self.contract_id:
            _violation(
                "Query.contract_id",
                f"不得包含{_QUERY_ID_SEP}",
                f"不含{_QUERY_ID_SEP}",
                self.contract_id,
            )
        if not self.contract_text:
            _violation("Query.contract_text", "非空", "非空字符串", self.contract_text)
        expected_id = f"{self.contract_id}{_QUERY_ID_SEP}{self.clause.name}"
        if self.query_id != expected_id:
            _violation(
                "Query.query_id",
                '等于f"{contract_id}::{clause.name}"',
                expected_id,
                self.query_id,
            )


@dataclass(frozen=True, slots=True)
class Judgment:
    """模型输出的解析结果。parse_ok=False 表示解析失败。"""

    present: Presence
    quotes: tuple[str, ...]
    confidence: float
    raw_output: str
    parse_ok: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            _violation("Judgment.confidence", "0<=c<=1", "[0.0, 1.0]", self.confidence)
        if self.present is Presence.ABSENT and self.quotes:
            _violation("Judgment.quotes", "present==ABSENT时为空", (), self.quotes)
        if self.present is Presence.PRESENT and not self.quotes:
            _violation("Judgment.quotes", "present==PRESENT时至少一项", "非空tuple", self.quotes)
        if not self.parse_ok:
            if self.present is not Presence.UNCERTAIN:
                _violation(
                    "Judgment.present",
                    "parse_ok=False时为UNCERTAIN",
                    "UNCERTAIN",
                    self.present,
                )
            if self.confidence != 0.0:
                _violation("Judgment.confidence", "parse_ok=False时为0.0", 0.0, self.confidence)


@dataclass(frozen=True, slots=True)
class Span:
    """由 quote 在原文中定位得出的区间。"""

    quote: str
    start: int | None
    end: int | None
    valid: bool
    occurrences: int

    def __post_init__(self) -> None:
        if self.occurrences < 0:
            _violation("Span.occurrences", ">=0", ">=0", self.occurrences)
        if self.valid:
            if self.start is None:
                _violation("Span.start", "valid时非None", "int", self.start)
            if self.end is None:
                _violation("Span.end", "valid时非None", "int", self.end)
            if not 0 <= self.start < self.end:
                _violation("Span.start", "0<=start<end", "start<end", (self.start, self.end))
            if self.occurrences < 1:
                _violation("Span.occurrences", "valid时>=1", ">=1", self.occurrences)
        else:
            if self.start is not None:
                _violation("Span.start", "not valid时为None", None, self.start)
            if self.end is not None:
                _violation("Span.end", "not valid时为None", None, self.end)


@dataclass(frozen=True, slots=True)
class Record:
    """JSONL 的一行。每条记录自包含，可独立解析。"""

    run_id: str
    query_id: str
    contract_id: str
    clause: ClauseKey
    status: Status
    judgment: Judgment | None
    spans: tuple[Span, ...]
    route: Route | None
    model: str
    commit: str
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool
    created_at: str
    error_detail: str | None

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("model", self.model),
            ("commit", self.commit),
            ("prompt_hash", self.prompt_hash),
            ("created_at", self.created_at),
        ):
            if not value:
                _violation(f"Record.{name}", "非空", "非空字符串", value)
        if self.input_tokens < 0:
            _violation("Record.input_tokens", ">=0", ">=0", self.input_tokens)
        if self.output_tokens < 0:
            _violation("Record.output_tokens", ">=0", ">=0", self.output_tokens)
        if self.latency_ms < 0:
            _violation("Record.latency_ms", ">=0", ">=0", self.latency_ms)
        if self.status is Status.OK:
            if self.judgment is None:
                _violation("Record.judgment", "status==OK时非None", "Judgment", self.judgment)
            if self.route is None:
                _violation("Record.route", "status==OK时非None", "Route", self.route)
            if len(self.spans) != len(self.judgment.quotes):
                _violation(
                    "Record.spans",
                    "status==OK时len(spans)==len(judgment.quotes)",
                    len(self.judgment.quotes),
                    len(self.spans),
                )
        else:
            if self.judgment is not None:
                _violation("Record.judgment", "status!=OK时为None", None, self.judgment)
            if self.spans:
                _violation("Record.spans", "status!=OK时为空", (), self.spans)
            if self.route is not None:
                _violation("Record.route", "status!=OK时为None", None, self.route)


@dataclass(frozen=True, slots=True)
class GoldSpan:
    """gold 标注的原文区间。"""

    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            _violation("GoldSpan.start", "0<=start<end", "start<end", (self.start, self.end))
        if not self.text:
            _violation("GoldSpan.text", "非空", "非空字符串", self.text)


@dataclass(frozen=True, slots=True)
class GoldLabel:
    """只由 metrics 加载，不进入 Query。"""

    query_id: str
    has_answer: bool
    spans: tuple[GoldSpan, ...]

    def __post_init__(self) -> None:
        if self.has_answer != (len(self.spans) >= 1):
            _violation(
                "GoldLabel.has_answer",
                "等于len(spans)>=1",
                len(self.spans) >= 1,
                self.has_answer,
            )


@dataclass(frozen=True, slots=True)
class RunMeta:
    run_id: str
    commit: str
    model: str
    config_digest: str
    seed: int
    split: str
    n_contracts: int  # 子集合同数（替代原 subset_spec）
    strategy: str  # 抽样策略名，如 "length_stratified"
    started_at: str
    n_queries: int

    def __post_init__(self) -> None:
        if not 0 <= self.seed < _SEED_MAX:
            _violation("RunMeta.seed", "0<=seed<2**63", f"[0, {_SEED_MAX})", self.seed)
        if self.split not in _SPLITS:
            _violation("RunMeta.split", "取值dev或report", "dev或report", self.split)
        if self.n_contracts <= 0:
            _violation("RunMeta.n_contracts", ">0", ">0", self.n_contracts)
        if not self.strategy:
            _violation("RunMeta.strategy", "非空", "非空字符串", self.strategy)
