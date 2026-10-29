"""metrics 的契约测试：docs/design/0003-interfaces.md §4、tasks/T-007.md 验收标准。

全部用合成记录，不读真实文件、不调模型。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from clauseproof.metrics import compute, iou
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

_MODEL = "glm-5.3"  # 价格表（config._PRICES）中唯一的已知模型


def make_record(
    query_id: str,
    *,
    clause: ClauseKey = ClauseKey.GOVERNING_LAW,
    present: Presence = Presence.PRESENT,
    confidence: float = 0.9,
    quotes: tuple[str, ...] = ("quote",),
    spans: tuple[Span, ...] | None = None,
    status: Status = Status.OK,
    cached: bool = False,
    input_tokens: int = 100,
    output_tokens: int = 10,
    latency_ms: float = 100.0,
) -> Record:
    """合成 Record；status==OK 时 spans 缺省由 quotes 生成全部 valid 的区间。"""
    if status is Status.OK:
        judgment = Judgment(
            present=present,
            quotes=quotes,
            confidence=confidence,
            raw_output="{}",
            parse_ok=True,
        )
        if spans is None:
            spans = tuple(
                Span(quote=q, start=0, end=len(q), valid=True, occurrences=1, matched="exact")
                for q in quotes
            )
        route = Route.AUTO_PASS
    else:
        judgment, spans, route = None, (), None
    return Record(
        run_id="run-1",
        query_id=query_id,
        contract_id=query_id.split("::")[0],
        clause=clause,
        status=status,
        judgment=judgment,
        spans=spans,
        route=route,
        model=_MODEL,
        commit="deadbeef",
        prompt_hash="abc123",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cached=cached,
        created_at="2026-10-27T00:00:00Z",
        error_detail=None if status is Status.OK else "boom",
    )


def make_gold(
    query_id: str, *, has_answer: bool, spans: tuple[tuple[int, int], ...] = ()
) -> GoldLabel:
    assert has_answer == (len(spans) >= 1)
    return GoldLabel(
        query_id=query_id,
        has_answer=has_answer,
        spans=tuple(GoldSpan(start=s, end=e, text="x" * (e - s)) for s, e in spans),
    )


def quadrant_records() -> list[Record]:
    """四象限各一条 + 一条模糊 + 一条失败，全部落在 GOVERNING_LAW。"""
    return [
        make_record("c1::GOVERNING_LAW", present=Presence.PRESENT),
        make_record("c2::GOVERNING_LAW", present=Presence.ABSENT, quotes=()),
        make_record("c3::GOVERNING_LAW", present=Presence.PRESENT),
        make_record("c4::GOVERNING_LAW", present=Presence.ABSENT, quotes=()),
        make_record("c5::GOVERNING_LAW", present=Presence.UNCERTAIN, quotes=()),
        make_record("c6::GOVERNING_LAW", status=Status.LLM_ERROR),
    ]


def quadrant_gold() -> dict[str, GoldLabel]:
    return {
        "c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=True, spans=((0, 5),)),
        "c2::GOVERNING_LAW": make_gold("c2::GOVERNING_LAW", has_answer=True, spans=((0, 5),)),
        "c3::GOVERNING_LAW": make_gold("c3::GOVERNING_LAW", has_answer=False),
        "c4::GOVERNING_LAW": make_gold("c4::GOVERNING_LAW", has_answer=False),
        "c5::GOVERNING_LAW": make_gold("c5::GOVERNING_LAW", has_answer=True, spans=((0, 5),)),
        "c6::GOVERNING_LAW": make_gold("c6::GOVERNING_LAW", has_answer=True, spans=((0, 5),)),
    }


def gold_for(records: list[Record], *, has_answer: bool = False) -> dict[str, GoldLabel]:
    return {
        record.query_id: make_gold(record.query_id, has_answer=has_answer) for record in records
    }


# ---------- iou ----------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0, 10), (0, 10), 1.0),  # 完全重叠
        ((0, 5), (10, 20), 0.0),  # 无重叠
        ((0, 10), (2, 8), 0.6),  # 包含：交 6 / 并 10
        ((0, 10), (5, 15), 1 / 3),  # 部分重叠：交 5 / 并 15
        ((0, 5), (5, 10), 0.0),  # 边界相接
        ((5, 5), (0, 10), 0.0),  # 零长度
        ((5, 5), (5, 5), 0.0),  # 双零长度
        ((10, 0), (0, 10), 0.0),  # 负长度防御
    ],
)
def test_iou(a: tuple[int, int], b: tuple[int, int], expected: float) -> None:
    assert iou(a[0], a[1], b[0], b[1]) == pytest.approx(expected)


# ---------- 基本契约 ----------


def test_by_clause_has_six_entries_in_clausekey_order() -> None:
    report = compute(quadrant_records(), quadrant_gold(), seed=42)
    assert [metrics.clause for metrics in report.by_clause] == list(ClauseKey)


def test_same_seed_yields_field_equal_report() -> None:
    records, gold = quadrant_records(), quadrant_gold()
    assert compute(records, gold, seed=42) == compute(records, gold, seed=42)


def test_different_seed_changes_only_accuracy_ci() -> None:
    records, gold = quadrant_records(), quadrant_gold()
    base = compute(records, gold, seed=42)
    other = compute(records, gold, seed=43)
    patched = tuple(
        replace(metrics, accuracy_ci=base.by_clause[i].accuracy_ci)
        for i, metrics in enumerate(other.by_clause)
    )
    assert replace(other, seed=42, by_clause=patched) == base
    # 不断言两个 seed 的 CI 必不相同：小样本下 bootstrap 分位值域有限，
    # 碰巧相同属正常；验收要求的是「其余字段不受 seed 影响」，上面已覆盖


def test_run_id_and_split_passed_through() -> None:
    report = compute(quadrant_records(), quadrant_gold(), seed=42, split="dev")
    assert report.run_id == "run-1"
    assert report.split == "dev"
    assert report.iou_threshold == 0.5


def test_empty_records_yield_zero_report() -> None:
    report = compute([], {}, seed=42)
    assert report.run_id == ""
    assert report.n_total == 0
    assert report.uncertain_rate == 0.0
    assert report.macro_accuracy is None
    assert len(report.by_clause) == 6
    assert report.total_cost_cny == 0.0
    assert report.mean_input_tokens == 0.0
    assert report.p95_latency_ms == 0.0


# ---------- 分母与恒等 ----------


def test_identity_holds_per_clause_and_globally() -> None:
    report = compute(quadrant_records(), quadrant_gold(), seed=42)
    for metrics in report.by_clause:
        assert metrics.n_total == metrics.n_decided + metrics.n_uncertain + metrics.n_failed
    decided_total = sum(metrics.n_decided for metrics in report.by_clause)
    assert report.n_total == decided_total + report.n_uncertain + report.n_failed


def test_failed_records_counted_in_total_and_failed_only() -> None:
    report = compute(quadrant_records(), quadrant_gold(), seed=42)
    metrics = report.by_clause[0]
    assert (metrics.n_total, metrics.n_ok, metrics.n_failed) == (6, 5, 1)
    assert metrics.n_decided == 4
    assert metrics.n_uncertain == 1


def test_uncertain_rate_denominator_is_n_total() -> None:
    report = compute(quadrant_records(), quadrant_gold(), seed=42)
    assert report.n_uncertain == 1
    assert report.uncertain_rate == pytest.approx(1 / 6)


# ---------- 判定正确性 ----------


def test_four_quadrant_correctness() -> None:
    metrics = compute(quadrant_records(), quadrant_gold(), seed=42).by_clause[0]
    assert metrics.n_correct == 2  # gold有+PRESENT、gold无+ABSENT
    assert metrics.n_decided == 4
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)  # TP=1 / (TP=1 + FN=1)
    assert metrics.precision == pytest.approx(0.5)  # TP=1 / (TP=1 + FP=1)


def test_uncertain_not_in_decided_or_correct() -> None:
    metrics = compute(quadrant_records(), quadrant_gold(), seed=42).by_clause[0]
    assert metrics.n_uncertain == 1
    assert metrics.n_correct == 2  # 模糊那条（gold 有 + UNCERTAIN）未计入


def test_no_decided_yields_none_metrics() -> None:
    records = [
        make_record(
            "c1::CAP_ON_LIABILITY",
            clause=ClauseKey.CAP_ON_LIABILITY,
            present=Presence.UNCERTAIN,
            quotes=(),
        )
    ]
    gold = {
        "c1::CAP_ON_LIABILITY": make_gold("c1::CAP_ON_LIABILITY", has_answer=True, spans=((0, 5),))
    }
    metrics = compute(records, gold, seed=42).by_clause[1]
    assert metrics.n_decided == 0
    assert metrics.accuracy is None
    assert metrics.accuracy_ci is None
    assert metrics.recall is None
    assert metrics.precision is None
    assert metrics.ece is None
    assert metrics.n_correct == 0


# ---------- 引用指标 ----------


def test_citation_uses_only_valid_spans() -> None:
    record = make_record(
        "c1::GOVERNING_LAW",
        present=Presence.PRESENT,
        quotes=("valid", "miss"),
        spans=(
            Span(quote="valid", start=12, end=22, valid=True, occurrences=1, matched="exact"),
            Span(quote="miss", start=None, end=None, valid=False, occurrences=0, matched="none"),
        ),
    )
    gold = {"c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=True, spans=((10, 20),))}
    metrics = compute([record], gold, seed=42).by_clause[0]
    # invalid 的 (10,20) 完美匹配不参与：只算 valid 的 iou((10,20),(12,22)) = 8/12
    assert metrics.citation_iou == pytest.approx(2 / 3)
    assert metrics.citation_contained == 0.0  # 22 > 20，未完全落在 gold 内
    assert metrics.citation_valid_rate == pytest.approx(0.5)


def test_citation_contained_when_prediction_inside_gold() -> None:
    record = make_record(
        "c1::GOVERNING_LAW",
        present=Presence.PRESENT,
        quotes=("valid",),
        spans=(Span(quote="valid", start=12, end=15, valid=True, occurrences=1, matched="exact"),),
    )
    gold = {"c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=True, spans=((10, 20),))}
    metrics = compute([record], gold, seed=42).by_clause[0]
    assert metrics.citation_contained == 1.0
    assert metrics.citation_iou == pytest.approx(0.3)


def test_citation_averages_over_all_gold_spans() -> None:
    record = make_record(
        "c1::GOVERNING_LAW",
        present=Presence.PRESENT,
        quotes=("valid",),
        spans=(Span(quote="valid", start=12, end=15, valid=True, occurrences=1, matched="exact"),),
    )
    gold = {
        "c1::GOVERNING_LAW": make_gold(
            "c1::GOVERNING_LAW", has_answer=True, spans=((10, 20), (30, 40))
        )
    }
    metrics = compute([record], gold, seed=42).by_clause[0]
    # span1：iou=0.3、contained=1；span2：无匹配预测、iou=0、contained=0
    assert metrics.citation_iou == pytest.approx(0.15)
    assert metrics.citation_contained == pytest.approx(0.5)


def test_citation_none_without_gold_spans() -> None:
    record = make_record("c1::GOVERNING_LAW", present=Presence.PRESENT)
    gold = {"c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=False)}
    metrics = compute([record], gold, seed=42).by_clause[0]
    assert metrics.citation_iou is None
    assert metrics.citation_contained is None
    assert metrics.citation_valid_rate == 1.0


def test_citation_exact_rate_counts_exact_over_all_spans() -> None:
    """exact / normalized / none 混合：精确匹配率分母为全部 span（含 invalid）。"""
    record = make_record(
        "c1::GOVERNING_LAW",
        present=Presence.PRESENT,
        quotes=("a", "b", "c"),
        spans=(
            Span(quote="a", start=0, end=1, valid=True, occurrences=1, matched="exact"),
            Span(quote="b", start=2, end=3, valid=True, occurrences=1, matched="normalized"),
            Span(quote="c", start=None, end=None, valid=False, occurrences=0, matched="none"),
        ),
    )
    gold = {"c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=True, spans=((0, 1),))}
    metrics = compute([record], gold, seed=42).by_clause[0]
    assert metrics.citation_exact_rate == pytest.approx(1 / 3)
    assert metrics.citation_valid_rate == pytest.approx(2 / 3)


def test_citation_exact_rate_is_one_when_all_exact() -> None:
    record = make_record("c1::GOVERNING_LAW", present=Presence.PRESENT, quotes=("a",))
    gold = {"c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=True, spans=((0, 1),))}
    metrics = compute([record], gold, seed=42).by_clause[0]
    assert metrics.citation_exact_rate == 1.0


def test_citation_exact_rate_none_without_spans() -> None:
    record = make_record("c1::GOVERNING_LAW", present=Presence.ABSENT, quotes=())
    gold = {"c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=False)}
    metrics = compute([record], gold, seed=42).by_clause[0]
    assert metrics.citation_exact_rate is None


def test_citation_valid_rate_none_without_any_span() -> None:
    record = make_record("c1::GOVERNING_LAW", present=Presence.ABSENT, quotes=())
    gold = {"c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=False)}
    metrics = compute([record], gold, seed=42).by_clause[0]
    assert metrics.citation_valid_rate is None


# ---------- 校准 ----------


def test_clause_ece_none_when_samples_form_single_bin() -> None:
    report = compute(quadrant_records(), quadrant_gold(), seed=42)
    # GOVERNING_LAW 4 条 decided：等频 10 桶全部合并成单桶 → 分类别 ece 为 None
    assert report.by_clause[0].ece is None
    assert report.by_clause[1].ece is None  # 无记录的类别


def _decided_batch(
    prefix: str,
    clause: ClauseKey,
    *,
    confidence: float,
    n_correct: int,
    n_wrong: int,
) -> list[Record]:
    """n_correct 条 ABSENT（gold 无 → 判对）+ n_wrong 条 PRESENT（gold 无 → 判错）。"""
    correct = [
        make_record(
            f"{prefix}-ok{i}::{clause.name}",
            clause=clause,
            present=Presence.ABSENT,
            quotes=(),
            confidence=confidence,
        )
        for i in range(n_correct)
    ]
    wrong = [
        make_record(
            f"{prefix}-bad{i}::{clause.name}",
            clause=clause,
            present=Presence.PRESENT,
            confidence=confidence,
        )
        for i in range(n_wrong)
    ]
    return correct + wrong


def test_global_ece_matches_hand_computation() -> None:
    records = _decided_batch(
        "g", ClauseKey.GOVERNING_LAW, confidence=0.2, n_correct=6, n_wrong=24
    ) + _decided_batch("c", ClauseKey.CAP_ON_LIABILITY, confidence=0.8, n_correct=15, n_wrong=15)
    report = compute(records, gold_for(records), seed=42)
    # 全局 60 条：排序后前 30 条 conf=0.2（6 对）、后 30 条 conf=0.8（15 对），
    # 等频 10 桶（每桶 6）按 min_bin_size=30 合并成两个 30 条桶：
    # ECE = (30/60)×|0.2−0.2| + (30/60)×|0.5−0.8| = 0.15
    assert report.ece == pytest.approx(0.15)
    # 每类各 30 条 decided 只能形成单桶 → 分类别 ece 均为 None
    assert all(metrics.ece is None for metrics in report.by_clause)


def test_global_ece_none_when_insufficient_samples() -> None:
    report = compute(quadrant_records(), quadrant_gold(), seed=42)
    # 全局仅 4 条 decided → 单桶 → None
    assert report.ece is None


# ---------- bootstrap 置信区间 ----------


def test_accuracy_ci_brackets_point_estimate_and_is_deterministic() -> None:
    metrics = compute(quadrant_records(), quadrant_gold(), seed=42).by_clause[0]
    assert metrics.accuracy_ci is not None
    low, high = metrics.accuracy_ci
    assert low <= 0.5 <= high
    again = compute(quadrant_records(), quadrant_gold(), seed=42).by_clause[0].accuracy_ci
    assert metrics.accuracy_ci == again


def test_accuracy_ci_none_without_decided() -> None:
    records = [make_record("c1::GOVERNING_LAW", present=Presence.UNCERTAIN, quotes=())]
    gold = {"c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=True, spans=((0, 5),))}
    metrics = compute(records, gold, seed=42).by_clause[0]
    assert metrics.accuracy_ci is None


# ---------- 数据异常 ----------


def test_duplicate_query_id_raises_with_id_and_count() -> None:
    records = [make_record("c1::GOVERNING_LAW"), make_record("c1::GOVERNING_LAW")]
    gold = {"c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=False)}
    with pytest.raises(ValueError, match=r"c1::GOVERNING_LAW×2"):
        compute(records, gold, seed=42)


def test_unknown_query_id_raises_with_id() -> None:
    records = [make_record("c1::GOVERNING_LAW"), make_record("c2::GOVERNING_LAW")]
    gold = {"c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=False)}
    with pytest.raises(ValueError, match="c2::GOVERNING_LAW"):
        compute(records, gold, seed=42)


def test_extra_gold_entries_are_allowed() -> None:
    records = [make_record("c1::GOVERNING_LAW")]
    gold = {
        "c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=False),
        "zz::ANTI_ASSIGNMENT": make_gold("zz::ANTI_ASSIGNMENT", has_answer=False),
    }
    report = compute(records, gold, seed=42)
    assert report.n_total == 1


# ---------- 宏平均 ----------


def test_macro_accuracy_skips_none_and_averages() -> None:
    records = quadrant_records() + [
        make_record(
            "c7::CAP_ON_LIABILITY",
            clause=ClauseKey.CAP_ON_LIABILITY,
            present=Presence.ABSENT,
            quotes=(),
        )
    ]
    gold = {
        **quadrant_gold(),
        "c7::CAP_ON_LIABILITY": make_gold("c7::CAP_ON_LIABILITY", has_answer=False),
    }
    report = compute(records, gold, seed=42)
    # GOVERNING_LAW acc=0.5，CAP_ON_LIABILITY acc=1.0，其余四类 None 跳过
    assert report.macro_accuracy == pytest.approx(0.75)


def test_macro_accuracy_all_none_is_none() -> None:
    records = [make_record("c1::GOVERNING_LAW", present=Presence.UNCERTAIN, quotes=())]
    gold = {"c1::GOVERNING_LAW": make_gold("c1::GOVERNING_LAW", has_answer=True, spans=((0, 5),))}
    assert compute(records, gold, seed=42).macro_accuracy is None


# ---------- 汇总字段 ----------


def test_total_cost_sums_paid_calls_with_cache_and_failure_at_zero() -> None:
    records = [
        make_record("c1::GOVERNING_LAW", input_tokens=125_000, output_tokens=0),
        make_record("c2::GOVERNING_LAW", input_tokens=125_000, output_tokens=0, cached=True),
        make_record("c3::GOVERNING_LAW", status=Status.LLM_ERROR, input_tokens=125_000),
    ]
    report = compute(records, gold_for(records), seed=42)
    # 只有第一条计费：125_000 × ¥8/百万 = ¥1；缓存命中与失败记录记 0
    assert report.total_cost_cny == pytest.approx(1.0)


def test_mean_tokens_denominator_is_n_ok() -> None:
    records = [
        make_record("c1::GOVERNING_LAW", input_tokens=300, output_tokens=30),
        make_record(
            "c2::GOVERNING_LAW", status=Status.LLM_ERROR, input_tokens=999, output_tokens=999
        ),
    ]
    report = compute(records, gold_for(records), seed=42)
    assert report.mean_input_tokens == pytest.approx(300.0)
    assert report.mean_output_tokens == pytest.approx(30.0)


def test_p95_latency_nearest_rank_over_ok_records_only() -> None:
    records = [
        make_record(
            f"c{i}::GOVERNING_LAW",
            present=Presence.ABSENT,
            quotes=(),
            latency_ms=float(i),
        )
        for i in range(1, 21)
    ]
    records.append(make_record("c99::GOVERNING_LAW", status=Status.LLM_ERROR, latency_ms=0.0))
    report = compute(records, gold_for(records), seed=42)
    # 20 条 OK：ceil(0.95×20)-1 = 18 → 第 19 小 = 19.0；失败记录的占位 0 不参与
    assert report.p95_latency_ms == 19.0


def test_p95_latency_small_n_takes_max() -> None:
    records = [
        make_record(f"c{i}::GOVERNING_LAW", present=Presence.ABSENT, quotes=(), latency_ms=float(i))
        for i in range(1, 15)
    ]
    report = compute(records, gold_for(records), seed=42)
    # 14 条：ceil(0.95×14)-1 = 13 → 最大值
    assert report.p95_latency_ms == 14.0
