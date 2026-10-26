"""models 不变式、不可变性与错误信息格式测试。

契约见 docs/design/0002-data-model.md §7 与 docs/design/0004-error-semantics.md §8。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, replace

import pytest

from clauseproof.models import (
    ClauseKey,
    GoldLabel,
    GoldSpan,
    Judgment,
    Presence,
    Query,
    Record,
    Route,
    RunMeta,
    Span,
    Status,
)

# ---- 合法样例：每个结构一份，违规用例由 replace 派生 ----

_VALID_QUERY = Query(
    query_id="C-1::GOVERNING_LAW",
    contract_id="C-1",
    clause=ClauseKey.GOVERNING_LAW,
    contract_text="甲方应在……",
)

_ABSENT_JUDGMENT = Judgment(
    present=Presence.ABSENT,
    quotes=(),
    confidence=0.2,
    raw_output='{"present": "absent"}',
    parse_ok=True,
)
_PRESENT_JUDGMENT = Judgment(
    present=Presence.PRESENT,
    quotes=("甲方应在……",),
    confidence=0.9,
    raw_output='{"present": "present"}',
    parse_ok=True,
)
_UNCERTAIN_JUDGMENT = Judgment(
    present=Presence.UNCERTAIN,
    quotes=(),
    confidence=0.3,
    raw_output='{"present": "uncertain"}',
    parse_ok=True,
)

_VALID_SPAN = Span(quote="甲方应在……", start=0, end=5, valid=True, occurrences=1)
_INVALID_SPAN = Span(quote="甲方应在……", start=None, end=None, valid=False, occurrences=0)

_OK_RECORD = Record(
    run_id="run-1",
    query_id="C-1::GOVERNING_LAW",
    contract_id="C-1",
    clause=ClauseKey.GOVERNING_LAW,
    status=Status.OK,
    judgment=_PRESENT_JUDGMENT,
    spans=(_VALID_SPAN,),
    route=Route.AUTO_PASS,
    model="model-a",
    commit="9f8e7d6",
    prompt_hash="0" * 64,
    input_tokens=100,
    output_tokens=50,
    latency_ms=1234.5,
    cached=False,
    created_at="2026-10-21T12:00:00Z",
    error_detail=None,
)
_FAILED_RECORD = replace(
    _OK_RECORD,
    status=Status.LLM_ERROR,
    judgment=None,
    spans=(),
    route=None,
    error_detail="connection reset",
)

_GOLD_SPAN = GoldSpan(start=3, end=8, text="甲方应在……")
_GOLD_LABEL = GoldLabel(query_id="C-1::GOVERNING_LAW", has_answer=True, spans=(_GOLD_SPAN,))

_RUN_META = RunMeta(
    run_id="run-1",
    commit="9f8e7d6",
    model="model-a",
    config_digest="d1" * 32,
    seed=42,
    split="dev",
    n_contracts=50,
    strategy="length_stratified",
    started_at="2026-10-21T12:00:00Z",
    n_queries=3060,
)

_ALL_STRUCTURES = (Query, Judgment, Span, Record, GoldSpan, GoldLabel, RunMeta)

_MESSAGE_PATTERN = re.compile(r"^models: \S+ 违反 .+，期望 .+，实际 .+$")


def _expect_value_error(field: str, build: Callable[[], object]) -> None:
    """断言 build() 抛 ValueError，消息符合四要素格式且指向 field。"""
    with pytest.raises(ValueError) as excinfo:
        build()
    message = str(excinfo.value)
    assert _MESSAGE_PATTERN.match(message), message
    assert message.startswith(f"models: {field} 违反 "), message


# ---- 不可变性 ----


@pytest.mark.parametrize(
    ("instance", "attr"),
    [
        (_VALID_QUERY, "contract_id"),
        (_PRESENT_JUDGMENT, "confidence"),
        (_VALID_SPAN, "start"),
        (_OK_RECORD, "model"),
        (_GOLD_SPAN, "text"),
        (_GOLD_LABEL, "has_answer"),
        (_RUN_META, "seed"),
    ],
)
def test_instances_are_frozen(instance: object, attr: str) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(instance, attr, "x")


@pytest.mark.parametrize("cls", _ALL_STRUCTURES)
def test_structures_declare_frozen_and_slots(cls: type) -> None:
    assert cls.__dataclass_params__.frozen is True
    assert set(cls.__slots__) == {f.name for f in fields(cls)}


# ---- 契约对齐：枚举与字段顺序 ----


def test_enum_membership_matches_contract() -> None:
    assert [m.value for m in ClauseKey] == [
        "Governing Law",
        "Cap On Liability",
        "Uncapped Liability",
        "Anti-Assignment",
        "Termination For Convenience",
        "Notice Period To Terminate Renewal",
    ]
    assert [m.value for m in Presence] == ["present", "absent", "uncertain"]
    assert [m.value for m in Route] == ["auto_pass", "to_human"]
    assert [m.value for m in Status] == ["ok", "llm_error", "context_overflow"]


def test_field_order_matches_contract() -> None:
    assert [f.name for f in fields(Query)] == ["query_id", "contract_id", "clause", "contract_text"]
    assert [f.name for f in fields(Judgment)] == [
        "present",
        "quotes",
        "confidence",
        "raw_output",
        "parse_ok",
    ]
    assert [f.name for f in fields(Span)] == ["quote", "start", "end", "valid", "occurrences"]
    assert [f.name for f in fields(Record)] == [
        "run_id",
        "query_id",
        "contract_id",
        "clause",
        "status",
        "judgment",
        "spans",
        "route",
        "model",
        "commit",
        "prompt_hash",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "cached",
        "created_at",
        "error_detail",
    ]
    assert [f.name for f in fields(GoldSpan)] == ["start", "end", "text"]
    assert [f.name for f in fields(GoldLabel)] == ["query_id", "has_answer", "spans"]
    assert [f.name for f in fields(RunMeta)] == [
        "run_id",
        "commit",
        "model",
        "config_digest",
        "seed",
        "split",
        "n_contracts",
        "strategy",
        "started_at",
        "n_queries",
    ]


# ---- Query ----


def test_query_id_must_match_composition() -> None:
    _expect_value_error("Query.query_id", lambda: replace(_VALID_QUERY, query_id="wrong"))


def test_query_contract_id_must_be_non_empty() -> None:
    _expect_value_error("Query.contract_id", lambda: replace(_VALID_QUERY, contract_id=""))


def test_query_contract_text_must_be_non_empty() -> None:
    _expect_value_error("Query.contract_text", lambda: replace(_VALID_QUERY, contract_text=""))


def test_query_contract_id_must_not_contain_separator() -> None:
    _expect_value_error(
        "Query.contract_id",
        lambda: replace(_VALID_QUERY, contract_id="C::1", query_id="C::1::GOVERNING_LAW"),
    )


def test_error_message_renders_actual_value() -> None:
    """违规时错误信息必须渲染真实值，而不是类型名字面量。"""
    judgment = _PRESENT_JUDGMENT
    with pytest.raises(ValueError, match=r"实际 Judgment\("):
        replace(
            _OK_RECORD,
            status=Status.LLM_ERROR,
            judgment=judgment,
            spans=(),
            route=None,
        )


# ---- Judgment ----


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_judgment_confidence_out_of_range(confidence: float) -> None:
    _expect_value_error(
        "Judgment.confidence", lambda: replace(_PRESENT_JUDGMENT, confidence=confidence)
    )


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_judgment_confidence_bounds_are_inclusive(confidence: float) -> None:
    assert replace(_PRESENT_JUDGMENT, confidence=confidence).confidence == confidence


def test_judgment_absent_requires_no_quotes() -> None:
    _expect_value_error(
        "Judgment.quotes", lambda: replace(_ABSENT_JUDGMENT, quotes=("甲方应在……",))
    )


def test_judgment_present_requires_quotes() -> None:
    _expect_value_error("Judgment.quotes", lambda: replace(_PRESENT_JUDGMENT, quotes=()))


def test_judgment_uncertain_allows_any_quotes() -> None:
    with_quotes = replace(_UNCERTAIN_JUDGMENT, quotes=("甲方应在……",))
    assert with_quotes.quotes == ("甲方应在……",)
    assert _UNCERTAIN_JUDGMENT.quotes == ()


def test_judgment_parse_fail_forces_uncertain() -> None:
    _expect_value_error(
        "Judgment.present",
        lambda: replace(
            _UNCERTAIN_JUDGMENT, parse_ok=False, present=Presence.PRESENT, quotes=("甲方应在……",)
        ),
    )


def test_judgment_parse_fail_forces_zero_confidence() -> None:
    _expect_value_error(
        "Judgment.confidence", lambda: replace(_UNCERTAIN_JUDGMENT, parse_ok=False, confidence=0.5)
    )


def test_judgment_parse_fail_state_is_valid() -> None:
    judgment = replace(_UNCERTAIN_JUDGMENT, confidence=0.0, parse_ok=False)
    assert judgment.present is Presence.UNCERTAIN
    assert judgment.confidence == 0.0


def test_error_message_format_matches_design_example() -> None:
    with pytest.raises(
        ValueError,
        match=r"^models: Judgment\.confidence 违反 0<=c<=1，期望 \[0\.0, 1\.0\]，实际 1\.37$",
    ):
        replace(_PRESENT_JUDGMENT, confidence=1.37)


# ---- Span ----


def test_span_valid_requires_int_start() -> None:
    _expect_value_error("Span.start", lambda: replace(_VALID_SPAN, start=None))


def test_span_valid_requires_int_end() -> None:
    _expect_value_error("Span.end", lambda: replace(_VALID_SPAN, end=None))


@pytest.mark.parametrize(("start", "end"), [(-1, 5), (5, 3), (5, 5)])
def test_span_valid_requires_ordered_range(start: int, end: int) -> None:
    _expect_value_error("Span.start", lambda: replace(_VALID_SPAN, start=start, end=end))


def test_span_invalid_requires_none_start() -> None:
    _expect_value_error("Span.start", lambda: replace(_INVALID_SPAN, start=0))


def test_span_invalid_requires_none_end() -> None:
    _expect_value_error("Span.end", lambda: replace(_INVALID_SPAN, end=5))


def test_span_occurrences_must_be_non_negative() -> None:
    _expect_value_error("Span.occurrences", lambda: replace(_VALID_SPAN, occurrences=-1))


def test_span_valid_requires_positive_occurrences() -> None:
    _expect_value_error("Span.occurrences", lambda: replace(_VALID_SPAN, occurrences=0))


# ---- Record ----


def test_record_ok_requires_judgment() -> None:
    _expect_value_error("Record.judgment", lambda: replace(_OK_RECORD, judgment=None))


def test_record_ok_requires_route() -> None:
    _expect_value_error("Record.route", lambda: replace(_OK_RECORD, route=None))


def test_record_ok_requires_spans_matching_quotes() -> None:
    _expect_value_error("Record.spans", lambda: replace(_OK_RECORD, spans=()))


def test_record_ok_without_quotes_and_spans_is_valid() -> None:
    record = replace(_OK_RECORD, judgment=_ABSENT_JUDGMENT, spans=())
    assert record.spans == ()


def test_record_failure_requires_judgment_none() -> None:
    _expect_value_error(
        "Record.judgment", lambda: replace(_FAILED_RECORD, judgment=_PRESENT_JUDGMENT)
    )


def test_record_failure_requires_empty_spans() -> None:
    _expect_value_error("Record.spans", lambda: replace(_FAILED_RECORD, spans=(_VALID_SPAN,)))


def test_record_failure_requires_route_none() -> None:
    _expect_value_error("Record.route", lambda: replace(_FAILED_RECORD, route=Route.AUTO_PASS))


def test_failed_record_is_valid() -> None:
    assert _FAILED_RECORD.judgment is None
    assert _FAILED_RECORD.spans == ()
    assert _FAILED_RECORD.route is None


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
def test_record_token_counts_must_be_non_negative(field: str) -> None:
    _expect_value_error(f"Record.{field}", lambda: replace(_OK_RECORD, **{field: -1}))


def test_record_latency_must_be_non_negative() -> None:
    _expect_value_error("Record.latency_ms", lambda: replace(_OK_RECORD, latency_ms=-0.5))


@pytest.mark.parametrize(
    "field",
    ["run_id", "model", "commit", "prompt_hash", "created_at"],
)
def test_record_identity_fields_must_be_non_empty(field: str) -> None:
    _expect_value_error(f"Record.{field}", lambda: replace(_OK_RECORD, **{field: ""}))


# ---- GoldSpan / GoldLabel ----


@pytest.mark.parametrize(("start", "end"), [(-1, 4), (4, 2), (2, 2)])
def test_gold_span_requires_ordered_range(start: int, end: int) -> None:
    _expect_value_error("GoldSpan.start", lambda: replace(_GOLD_SPAN, start=start, end=end))


def test_gold_span_requires_non_empty_text() -> None:
    _expect_value_error("GoldSpan.text", lambda: replace(_GOLD_SPAN, text=""))


def test_gold_label_has_answer_must_be_true_with_spans() -> None:
    _expect_value_error("GoldLabel.has_answer", lambda: replace(_GOLD_LABEL, has_answer=False))


def test_gold_label_has_answer_must_be_false_without_spans() -> None:
    _expect_value_error("GoldLabel.has_answer", lambda: replace(_GOLD_LABEL, spans=()))


def test_gold_label_without_answer_is_valid() -> None:
    label = GoldLabel(query_id="C-1::GOVERNING_LAW", has_answer=False, spans=())
    assert label.has_answer is False


# ---- RunMeta ----


@pytest.mark.parametrize("seed", [-1, 2**63])
def test_run_meta_seed_out_of_range(seed: int) -> None:
    _expect_value_error("RunMeta.seed", lambda: replace(_RUN_META, seed=seed))


@pytest.mark.parametrize("seed", [0, 2**63 - 1])
def test_run_meta_seed_bounds_are_inclusive(seed: int) -> None:
    assert replace(_RUN_META, seed=seed).seed == seed


def test_run_meta_split_must_be_dev_or_report() -> None:
    _expect_value_error("RunMeta.split", lambda: replace(_RUN_META, split="train"))


@pytest.mark.parametrize("split", ["dev", "report"])
def test_run_meta_split_domain_is_valid(split: str) -> None:
    assert replace(_RUN_META, split=split).split == split


@pytest.mark.parametrize("n_contracts", [0, -1])
def test_run_meta_n_contracts_must_be_positive(n_contracts: int) -> None:
    _expect_value_error("RunMeta.n_contracts", lambda: replace(_RUN_META, n_contracts=n_contracts))


def test_run_meta_strategy_must_be_non_empty() -> None:
    _expect_value_error("RunMeta.strategy", lambda: replace(_RUN_META, strategy=""))
