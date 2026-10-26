"""Judge 协议测试：FakeJudge 可注入，返回固定结果，不调用任何模型。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from clauseproof.judge import CallMeta, Judge
from clauseproof.models import ClauseKey, Judgment, Presence, Query


class FakeJudge:
    """测试替身：返回固定 Judgment 与 CallMeta，不调用任何模型。"""

    def judge(self, query: Query) -> tuple[Judgment, CallMeta]:
        judgment = Judgment(
            present=Presence.UNCERTAIN,
            quotes=(),
            confidence=0.0,
            raw_output="",
            parse_ok=True,
        )
        meta = CallMeta(
            model="fake-model",
            prompt_hash="0" * 64,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cached=False,
        )
        return judgment, meta


def _query() -> Query:
    return Query(
        query_id="C-1::GOVERNING_LAW",
        contract_id="C-1",
        clause=ClauseKey.GOVERNING_LAW,
        contract_text="甲方应在……",
    )


def test_fake_judge_satisfies_protocol() -> None:
    assert isinstance(FakeJudge(), Judge)


def test_fake_judge_returns_fixed_judgment() -> None:
    first = FakeJudge().judge(_query())
    second = FakeJudge().judge(_query())
    assert first == second
    judgment, meta = first
    assert isinstance(judgment, Judgment)
    assert judgment.present is Presence.UNCERTAIN
    assert judgment.parse_ok is True
    assert isinstance(meta, CallMeta)
    assert meta.model == "fake-model"


def test_call_meta_is_frozen() -> None:
    meta = CallMeta(
        model="model-a",
        prompt_hash="0" * 64,
        input_tokens=1,
        output_tokens=2,
        latency_ms=3.0,
        cached=False,
    )
    with pytest.raises(FrozenInstanceError):
        meta.model = "other"
