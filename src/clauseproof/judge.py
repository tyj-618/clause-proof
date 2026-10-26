"""Judge 协议与调用元信息。仅类型定义，无实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from clauseproof.models import Judgment, Query


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
