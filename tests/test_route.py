"""route 的契约测试：docs/design/0003-interfaces.md §3、tasks/T-005.md、tasks/T-008.md 验收标准。"""

from __future__ import annotations

import pytest

from clauseproof.models import (
    ClauseKey,
    GoldLabel,
    GoldSpan,
    Judgment,
    Presence,
    Record,
    Route,
    Span,
    Status,
)
from clauseproof.route import apply_route, select_thresholds, sweep


def judged(present: Presence, confidence: float, *, parse_ok: bool = True) -> Judgment:
    quotes = ("some quote",) if present is Presence.PRESENT else ()
    return Judgment(
        present=present,
        quotes=quotes,
        confidence=confidence,
        raw_output="{}",
        parse_ok=parse_ok,
    )


@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.8, 1.0])
@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_uncertain_always_to_human_regardless_of_threshold(
    threshold: float, confidence: float
) -> None:
    judgment = judged(Presence.UNCERTAIN, confidence)
    assert apply_route(judgment, threshold) is Route.TO_HUMAN


def test_parse_failed_judgment_goes_to_human_even_at_zero_threshold() -> None:
    judgment = judged(Presence.UNCERTAIN, 0.0, parse_ok=False)
    assert apply_route(judgment, 0.0) is Route.TO_HUMAN


def test_confidence_at_threshold_passes() -> None:
    assert apply_route(judged(Presence.PRESENT, 0.8), 0.8) is Route.AUTO_PASS
    assert apply_route(judged(Presence.ABSENT, 0.8), 0.8) is Route.AUTO_PASS


def test_confidence_below_threshold_goes_to_human() -> None:
    assert apply_route(judged(Presence.PRESENT, 0.79), 0.8) is Route.TO_HUMAN
    assert apply_route(judged(Presence.ABSENT, 0.79), 0.8) is Route.TO_HUMAN


def test_zero_threshold_passes_all_decided() -> None:
    assert apply_route(judged(Presence.PRESENT, 0.0), 0.0) is Route.AUTO_PASS
    assert apply_route(judged(Presence.ABSENT, 0.0), 0.0) is Route.AUTO_PASS


def test_unit_threshold_passes_only_full_confidence() -> None:
    assert apply_route(judged(Presence.PRESENT, 1.0), 1.0) is Route.AUTO_PASS
    assert apply_route(judged(Presence.ABSENT, 1.0), 1.0) is Route.AUTO_PASS
    assert apply_route(judged(Presence.PRESENT, 0.999), 1.0) is Route.TO_HUMAN


@pytest.mark.parametrize("threshold", [-1.0, -0.01, 1.01, 2.0])
def test_out_of_range_threshold_raises_value_error(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        apply_route(judged(Presence.PRESENT, 0.5), threshold)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_boundary_thresholds_are_valid(threshold: float) -> None:
    assert apply_route(judged(Presence.PRESENT, threshold), threshold) is Route.AUTO_PASS


def test_same_input_same_output() -> None:
    judgment = judged(Presence.PRESENT, 0.9)
    assert apply_route(judgment, 0.8) == apply_route(judgment, 0.8)


# ---- select_thresholds / sweep（tasks/T-008.md 验收标准）----

_CONTRACT_ID = "c1"
_IRREDUCIBLE_FLOOR = 0.25  # 合成数据集：1 条 UNCERTAIN + 1 条失败，共 8 条


def _record(
    clause: ClauseKey,
    query_id: str,
    present: Presence,
    confidence: float,
    *,
    ok: bool = True,
) -> Record:
    if not ok:
        return Record(
            run_id="run",
            query_id=query_id,
            contract_id=_CONTRACT_ID,
            clause=clause,
            status=Status.LLM_ERROR,
            judgment=None,
            spans=(),
            route=None,
            model="m",
            commit="cm",
            prompt_hash="ph",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
            cached=False,
            created_at="t",
            error_detail="err",
        )
    judgment = judged(present, confidence)
    spans = tuple(Span(quote, 0, 1, True, 1) for quote in judgment.quotes)
    return Record(
        run_id="run",
        query_id=query_id,
        contract_id=_CONTRACT_ID,
        clause=clause,
        status=Status.OK,
        judgment=judgment,
        spans=spans,
        route=Route.AUTO_PASS,
        model="m",
        commit="cm",
        prompt_hash="ph",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        cached=False,
        created_at="t",
        error_detail=None,
    )


def _gold(query_id: str, has_answer: bool) -> GoldLabel:
    spans = (GoldSpan(0, 1, "x"),) if has_answer else ()
    return GoldLabel(query_id, has_answer, spans)


def _dataset() -> tuple[list[Record], dict[str, GoldLabel]]:
    """8 条合成记录，每类至少 1 条：6 对 2 错，1 条 UNCERTAIN，1 条失败。

    不可约复核率 = 2/8 = 0.25；不含 confidence == 1.0 的记录。
    """
    specs = [
        (ClauseKey.GOVERNING_LAW, Presence.PRESENT, 0.9, True, True),
        (ClauseKey.GOVERNING_LAW, Presence.PRESENT, 0.4, False, True),
        (ClauseKey.CAP_ON_LIABILITY, Presence.ABSENT, 0.8, False, True),
        (ClauseKey.CAP_ON_LIABILITY, Presence.ABSENT, 0.3, True, True),
        (ClauseKey.UNCAP_LIABILITY, Presence.ABSENT, 0.6, False, True),
        (ClauseKey.ANTI_ASSIGNMENT, Presence.UNCERTAIN, 0.7, True, True),
        (ClauseKey.TERMINATION_FOR_CONVENIENCE, Presence.PRESENT, 0.6, True, True),
        (ClauseKey.NOTICE_PERIOD_TO_TERMINATE_RENEWAL, Presence.PRESENT, 0.5, True, False),
    ]
    records: list[Record] = []
    gold: dict[str, GoldLabel] = {}
    for index, (clause, present, confidence, has_answer, ok) in enumerate(specs):
        query_id = f"{_CONTRACT_ID}::{clause.name}::{index}"
        records.append(_record(clause, query_id, present, confidence, ok=ok))
        gold[query_id] = _gold(query_id, has_answer)
    return records, gold


def _all_failed_dataset() -> tuple[list[Record], dict[str, GoldLabel]]:
    """6 条全部 LLM_ERROR：不可约复核率为 1.0，无任何 OK 记录。"""
    records = [
        _record(clause, f"{_CONTRACT_ID}::{clause.name}", Presence.PRESENT, 0.5, ok=False)
        for clause in ClauseKey
    ]
    return records, {}


def _review_rate(records: list[Record], thresholds: dict[ClauseKey, float]) -> float:
    """用 apply_route 重算转人工率（分母含失败记录，同 select_thresholds 口径）。"""
    n_human = sum(
        1
        for record in records
        if record.status is not Status.OK
        or apply_route(record.judgment, thresholds[record.clause]) is Route.TO_HUMAN
    )
    return n_human / len(records)


def _candidate_values(records: list[Record], clause: ClauseKey) -> set[float]:
    """该类别的候选集：实际出现过的 OK confidence 值 ∪ {0.0, 1.0}。"""
    return {0.0, 1.0} | {
        record.judgment.confidence
        for record in records
        if record.clause is clause and record.status is Status.OK
    }


@pytest.mark.parametrize("target", [0.25, 0.5, 0.75, 1.0])
def test_select_thresholds_returns_six_keys_with_candidate_values(target: float) -> None:
    records, gold = _dataset()
    thresholds = select_thresholds(records, gold, target_review_rate=target)
    assert set(thresholds) == set(ClauseKey)
    assert len(thresholds) == 6
    for clause, value in thresholds.items():
        assert value in _candidate_values(records, clause)


@pytest.mark.parametrize("target", [0.25, 0.3, 0.5, 0.75, 1.0])
def test_select_thresholds_review_rate_within_budget(target: float) -> None:
    records, gold = _dataset()
    thresholds = select_thresholds(records, gold, target_review_rate=target)
    assert _review_rate(records, thresholds) <= target + 1e-9


@pytest.mark.parametrize("target", [0.25, 0.5, 1.0])
def test_select_thresholds_is_deterministic(target: float) -> None:
    records, gold = _dataset()
    first = select_thresholds(records, gold, target_review_rate=target)
    second = select_thresholds(records, gold, target_review_rate=target)
    assert first == second


@pytest.mark.parametrize("target", [0.1, 0.24])
def test_select_thresholds_below_irreducible_floor_raises(target: float) -> None:
    records, gold = _dataset()
    with pytest.raises(ValueError) as excinfo:
        select_thresholds(records, gold, target_review_rate=target)
    message = str(excinfo.value)
    assert str(target) in message
    assert str(_IRREDUCIBLE_FLOOR) in message


def test_select_thresholds_max_target_caps_all_thresholds_at_one() -> None:
    records, gold = _dataset()
    thresholds = select_thresholds(records, gold, target_review_rate=1.0)
    assert set(thresholds.values()) == {1.0}
    # 数据中无 confidence == 1.0 的记录，故全部转人工
    assert _review_rate(records, thresholds) == 1.0


def test_confidence_exactly_one_still_auto_passes_at_max_threshold() -> None:
    records, gold = _dataset()
    clause = ClauseKey.UNCAP_LIABILITY
    query_id = f"{_CONTRACT_ID}::{clause.name}::one"
    records.append(_record(clause, query_id, Presence.ABSENT, 1.0))
    gold[query_id] = _gold(query_id, False)
    thresholds = select_thresholds(records, gold, target_review_rate=1.0)
    judgment = records[-1].judgment
    assert judgment is not None
    # 已知边界（0003 §3）：confidence 恰为 1.0 无法被阈值推给人工（>= 语义）
    assert apply_route(judgment, thresholds[clause]) is Route.AUTO_PASS
    assert _review_rate(records, thresholds) < 1.0


def test_select_thresholds_review_rate_monotone_in_target() -> None:
    records, gold = _dataset()
    rates = [
        _review_rate(records, select_thresholds(records, gold, target_review_rate=target))
        for target in (0.25, 0.5, 0.75, 1.0)
    ]
    assert rates == sorted(rates)


def test_zero_thresholds_when_budget_equals_irreducible_floor() -> None:
    records, gold = _dataset()
    thresholds = select_thresholds(records, gold, target_review_rate=_IRREDUCIBLE_FLOOR)
    # 预算只够不可约部分：任何新增转人工的提升都放不进预算，
    # 非模糊条目全部自动通过（no-op 爬档可越过空档，但不改变任何路由）
    for record in records:
        if record.status is Status.OK and record.judgment.present is not Presence.UNCERTAIN:
            assert apply_route(record.judgment, thresholds[record.clause]) is Route.AUTO_PASS
    assert _review_rate(records, thresholds) == pytest.approx(_IRREDUCIBLE_FLOOR)


def test_all_failed_records_are_the_irreducible_floor() -> None:
    records, gold = _all_failed_dataset()
    thresholds = select_thresholds(records, gold, target_review_rate=1.0)
    assert set(thresholds.values()) == {1.0}
    assert _review_rate(records, thresholds) == 1.0
    with pytest.raises(ValueError) as excinfo:
        select_thresholds(records, gold, target_review_rate=0.99)
    message = str(excinfo.value)
    assert "0.99" in message
    assert "1.0" in message


def test_sweep_returns_one_point_per_rate_within_budget_and_monotone() -> None:
    records, gold = _dataset()
    points = sweep(records, gold, review_rates=[0.25, 0.5, 0.75, 1.0])
    assert [point.target_review_rate for point in points] == [0.25, 0.5, 0.75, 1.0]
    for point in points:
        assert point.actual_review_rate <= point.target_review_rate + 1e-9
    rates = [point.actual_review_rate for point in points]
    assert rates == sorted(rates)


def test_sweep_point_metrics_follow_the_contract_formulas() -> None:
    records, gold = _dataset()
    points = sweep(records, gold, review_rates=[0.25, 0.5, 0.75, 1.0])
    # 0.25：6 条判定记录全部自动通过，其中 2 条判错
    assert points[0].auto_pass_accuracy == pytest.approx(4 / 6)
    assert points[0].post_review_accuracy == pytest.approx(1.0 - 2 / 8)
    # 0.5：GOVERNING_LAW 两条转人工，剩 4 条判定记录中 1 条判错
    assert points[1].auto_pass_accuracy == pytest.approx(3 / 4)
    assert points[1].post_review_accuracy == pytest.approx(1.0 - 1 / 8)
    # 0.75：仅剩 UNCAP_LIABILITY 与 TERMINATION 两条判对记录自动通过
    assert points[2].auto_pass_accuracy == pytest.approx(1.0)
    assert points[2].post_review_accuracy == pytest.approx(1.0)
    # 1.0：无自动通过条目
    assert points[3].auto_pass_accuracy is None
    assert points[3].post_review_accuracy == 1.0
    assert set(points[3].thresholds.values()) == {1.0}


def test_sweep_unsatisfiable_rate_fails_without_truncation() -> None:
    records, gold = _dataset()
    with pytest.raises(ValueError) as excinfo:
        sweep(records, gold, review_rates=[0.5, 0.1])
    message = str(excinfo.value)
    assert "0.1" in message
    assert "0.25" in message


def test_sweep_all_failed_records_have_no_auto_pass_accuracy() -> None:
    records, gold = _all_failed_dataset()
    points = sweep(records, gold, review_rates=[1.0])
    assert points[0].actual_review_rate == 1.0
    assert points[0].auto_pass_accuracy is None
    assert points[0].post_review_accuracy == 1.0
