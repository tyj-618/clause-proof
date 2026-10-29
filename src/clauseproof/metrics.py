"""指标计算：把 Record 序列与 gold 标注变成可对外报告的数字。

契约见 docs/design/0003-interfaces.md §4、tasks/T-007.md「口径定义」。
纯计算层：不读数据文件、不调模型、不写盘。唯一外部依赖是 config.prices_for
（按每条记录的 model 查单价，从 token 重算成本）。

口径要点（tasks/T-007.md）：
- 判定正确 = gold.has_answer 与 present（PRESENT/ABSENT）一致；UNCERTAIN 不进准确率分母
- accuracy 分母为 n_decided，uncertain_rate 分母为 n_total（含失败）
- bootstrap：对 n_decided 条判定有放回重采样 1000 次，取 2.5%/97.5% 分位
  （nearest-rank）；随机源为 random.Random(f"clauseproof:bootstrap:{seed}:{clause.value}")
- 成本：OK 且非缓存的记录按 input/output token 与单价重算；缓存命中与失败记录记 0
  （0003 §10：缓存命中未发生计费，token 字段仅用于追溯）
- p95 延迟：仅 OK 记录（失败记录的 latency_ms 为占位 0，非测量值），nearest-rank
- 校准：分类别 ece 与全局 Report.ece 同用等频 10 桶 / min_bin_size 30；不足 2 个
  非空桶时 ECE 为 None（单桶退化为 |平均准确率 − 平均置信度|，不构成校准误差），
  对外报告的校准指标用 Report.ece（全局）
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil

from clauseproof.calibration import expected_calibration_error, reliability_bins
from clauseproof.config import prices_for
from clauseproof.models import ClauseKey, GoldLabel, Presence, Record, Status

# 任务书固定口径，非可调参数
_BOOTSTRAP_ROUNDS = 1000
_CI_LOW_QUANTILE = 0.025
_CI_HIGH_QUANTILE = 0.975
_CALIBRATION_N_BINS = 10
_CALIBRATION_MIN_BIN_SIZE = 30
_P95_QUANTILE = 0.95


@dataclass(frozen=True, slots=True)
class ClauseMetrics:
    """单个条款类别的指标。分母口径见模块 docstring 与任务书「分母」表。"""

    clause: ClauseKey
    n_total: int  # 该类别全部记录
    n_ok: int  # status == OK
    n_failed: int  # status != OK
    n_decided: int  # OK 且 present ∈ {PRESENT, ABSENT}
    n_uncertain: int  # OK 且 present == UNCERTAIN
    n_correct: int
    accuracy: float | None  # n_correct / n_decided；n_decided==0 时 None
    accuracy_ci: tuple[float, float] | None
    recall: float | None  # TP / (TP + FN)；分母为 0 时 None
    precision: float | None  # TP / (TP + FP)；分母为 0 时 None
    citation_iou: float | None  # 对全部 gold span 的平均最大 IoU；无 gold span 时 None
    citation_contained: float | None  # gold span 被有效预测区间完全覆盖的比例
    citation_valid_rate: float | None  # 有效 span / 全部 span；无 span 时 None
    citation_exact_rate: float | None  # matched=="exact" 的 span / 全部 span；无 span 时 None
    ece: float | None  # 该类别 n_decided 条 (confidence, 正确) 的期望校准误差


@dataclass(frozen=True, slots=True)
class Report:
    """一次运行的汇总报告。by_clause 恰好 6 项，顺序同 ClauseKey 定义。"""

    run_id: str
    split: str
    seed: int
    iou_threshold: float
    by_clause: tuple[ClauseMetrics, ...]
    macro_accuracy: float | None
    ece: float | None  # 全局校准误差，pairs 为全部 n_decided 条 (confidence, 正确)
    n_total: int
    n_ok: int
    n_failed: int
    n_uncertain: int
    uncertain_rate: float
    total_cost_cny: float
    mean_input_tokens: float
    mean_output_tokens: float
    p95_latency_ms: float


def iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """闭开区间 [start, end) 的 IoU；零长度或负长度区间恒为 0。"""
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return intersection / union


def compute(
    records: Sequence[Record],
    gold: Mapping[str, GoldLabel],
    *,
    seed: int,
    iou_threshold: float = 0.5,
    split: str = "",
) -> Report:
    """把一次运行的记录与 gold 标注汇总为 Report。

    - records 的 query_id 必须唯一且都在 gold 中（违反抛 ValueError）；
      gold 有多余条目不报错（允许只评测子集）
    - 同一输入 + 同一 seed 两次调用返回逐字段相等的对象；seed 只影响 accuracy_ci
    - iou_threshold 本期不参与主指标计算，仅透传到 Report 备后续诊断口径使用
    - split 无法从 Record/GoldLabel 推出，由调用方从 RunMeta 传入，缺省为空串
    """
    _validate(records, gold)
    grouped: dict[ClauseKey, list[Record]] = {clause: [] for clause in ClauseKey}
    for record in records:
        grouped[record.clause].append(record)
    by_clause: list[ClauseMetrics] = []
    global_pairs: list[tuple[float, bool]] = []
    for clause, group in grouped.items():
        metrics, pairs = _clause_metrics(clause, group, gold, seed=seed)
        by_clause.append(metrics)
        global_pairs.extend(pairs)
    accuracies = [metrics.accuracy for metrics in by_clause if metrics.accuracy is not None]
    macro_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
    global_ece = expected_calibration_error(
        reliability_bins(
            global_pairs,
            n_bins=_CALIBRATION_N_BINS,
            min_bin_size=_CALIBRATION_MIN_BIN_SIZE,
        )
    )

    n_total = len(records)
    ok_records = [record for record in records if record.status is Status.OK]
    n_ok = len(ok_records)
    n_failed = n_total - n_ok
    n_uncertain = sum(metrics.n_uncertain for metrics in by_clause)
    latencies = sorted(record.latency_ms for record in ok_records)
    return Report(
        run_id=records[0].run_id if records else "",
        split=split,
        seed=seed,
        iou_threshold=iou_threshold,
        by_clause=tuple(by_clause),
        macro_accuracy=macro_accuracy,
        ece=global_ece,
        n_total=n_total,
        n_ok=n_ok,
        n_failed=n_failed,
        n_uncertain=n_uncertain,
        uncertain_rate=n_uncertain / n_total if n_total else 0.0,
        total_cost_cny=_total_cost_cny(records),
        mean_input_tokens=(
            sum(record.input_tokens for record in ok_records) / n_ok if n_ok else 0.0
        ),
        mean_output_tokens=(
            sum(record.output_tokens for record in ok_records) / n_ok if n_ok else 0.0
        ),
        p95_latency_ms=(latencies[ceil(_P95_QUANTILE * n_ok) - 1] if latencies else 0.0),
    )


def _validate(records: Sequence[Record], gold: Mapping[str, GoldLabel]) -> None:
    """query_id 唯一、且全部在 gold 中（tasks/T-007.md「数据异常」）。"""
    counts = Counter(record.query_id for record in records)
    duplicates = {query_id: count for query_id, count in counts.items() if count > 1}
    if duplicates:
        detail = "、".join(f"{query_id}×{count}" for query_id, count in duplicates.items())
        raise ValueError(f"metrics: records.query_id 违反 唯一性，期望 不重复，实际 重复 {detail}")
    missing = [record.query_id for record in records if record.query_id not in gold]
    if missing:
        raise ValueError(
            "metrics: records.query_id 违反 必须存在于 gold，"
            f"期望 gold 覆盖全部 query_id，实际 缺失 {missing[0]}"
        )


def _clause_metrics(
    clause: ClauseKey,
    records: Sequence[Record],
    gold: Mapping[str, GoldLabel],
    *,
    seed: int,
) -> tuple[ClauseMetrics, list[tuple[float, bool]]]:
    """单个条款类别的全部指标与该校准 pairs（供 Report.ece 全局聚合）。

    records 为该类别的全部记录。
    """
    ok_records = [record for record in records if record.status is Status.OK]
    # Record 不变式保证 status==OK 时 judgment 非 None（0002 §7）
    decided = [record for record in ok_records if record.judgment.present is not Presence.UNCERTAIN]
    outcomes = [
        gold[record.query_id].has_answer == (record.judgment.present is Presence.PRESENT)
        for record in decided
    ]

    true_positives = sum(
        1
        for record, correct in zip(decided, outcomes, strict=True)
        if correct and record.judgment.present is Presence.PRESENT
    )
    false_negatives = sum(
        1
        for record, correct in zip(decided, outcomes, strict=True)
        if not correct and record.judgment.present is Presence.ABSENT
    )
    false_positives = sum(
        1
        for record, correct in zip(decided, outcomes, strict=True)
        if not correct and record.judgment.present is Presence.PRESENT
    )
    n_decided = len(decided)
    citation_iou, citation_contained, valid_rate, exact_rate = _citation_metrics(records, gold)
    calibration_pairs = [
        (record.judgment.confidence, correct)
        for record, correct in zip(decided, outcomes, strict=True)
    ]
    bins = reliability_bins(
        calibration_pairs,
        n_bins=_CALIBRATION_N_BINS,
        min_bin_size=_CALIBRATION_MIN_BIN_SIZE,
    )
    return ClauseMetrics(
        clause=clause,
        n_total=len(records),
        n_ok=len(ok_records),
        n_failed=len(records) - len(ok_records),
        n_decided=n_decided,
        n_uncertain=len(ok_records) - n_decided,
        n_correct=sum(outcomes),
        accuracy=sum(outcomes) / n_decided if n_decided else None,
        accuracy_ci=_accuracy_ci(outcomes, seed=seed, clause=clause),
        recall=(
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else None
        ),
        precision=(
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else None
        ),
        citation_iou=citation_iou,
        citation_contained=citation_contained,
        citation_valid_rate=valid_rate,
        citation_exact_rate=exact_rate,
        ece=expected_calibration_error(bins),
    ), calibration_pairs


def _citation_metrics(
    records: Sequence[Record],
    gold: Mapping[str, GoldLabel],
) -> tuple[float | None, float | None, float | None, float | None]:
    """引用四指标：IoU 均值、完全包含比例、引用有效率、精确匹配率。

    只用 valid 的预测区间；对每个 gold span 取最大 IoU（无有效预测记 0），
    再对全部 gold span 求平均。无 gold span 时前两者为 None，无任何 span 时
    有效率与精确匹配率为 None。精确匹配率 = matched=="exact" 的 span 数 /
    全部 span 数（含 invalid），用于观察第 2 层空白归一化匹配的占比。
    """
    ious: list[float] = []
    contained: list[bool] = []
    n_spans = 0
    n_valid_spans = 0
    n_exact_spans = 0
    for record in records:
        valid_ranges = [(span.start, span.end) for span in record.spans if span.valid]
        n_spans += len(record.spans)
        n_valid_spans += len(valid_ranges)
        n_exact_spans += sum(1 for span in record.spans if span.matched == "exact")
        for gold_span in gold[record.query_id].spans:
            ious.append(
                max(
                    (
                        iou(gold_span.start, gold_span.end, start, end)
                        for start, end in valid_ranges
                    ),
                    default=0.0,
                )
            )
            contained.append(
                any(
                    gold_span.start <= start and end <= gold_span.end for start, end in valid_ranges
                )
            )
    return (
        sum(ious) / len(ious) if ious else None,
        sum(contained) / len(contained) if contained else None,
        n_valid_spans / n_spans if n_spans else None,
        n_exact_spans / n_spans if n_spans else None,
    )


def _accuracy_ci(
    outcomes: Sequence[bool], *, seed: int, clause: ClauseKey
) -> tuple[float, float] | None:
    """判定正确性的 95% bootstrap 置信区间。

    对 n_decided 条判定有放回重采样 _BOOTSTRAP_ROUNDS 次，每次取正确比例，
    再取 2.5% 与 97.5% 分位（nearest-rank）。随机源由 (seed, clause) 派生，
    同一 seed 两次运行结果完全相同。
    """
    n = len(outcomes)
    if n == 0:
        return None
    rng = random.Random(f"clauseproof:bootstrap:{seed}:{clause.value}")
    values: list[float] = []
    for _ in range(_BOOTSTRAP_ROUNDS):
        values.append(sum(rng.choices(outcomes, k=n)) / n)
    values.sort()
    low = values[ceil(_CI_LOW_QUANTILE * _BOOTSTRAP_ROUNDS) - 1]
    high = values[ceil(_CI_HIGH_QUANTILE * _BOOTSTRAP_ROUNDS) - 1]
    return (low, high)


def _total_cost_cny(records: Sequence[Record]) -> float:
    """按 token 重算的总成本：缓存命中与失败记录记 0（0003 §10）。"""
    total = 0.0
    for record in records:
        if record.status is not Status.OK or record.cached:
            continue
        prices = prices_for(record.model)
        total += (
            record.input_tokens * prices.input_cny_per_mtok
            + record.output_tokens * prices.output_cny_per_mtok
        ) / 1e6
    return total
