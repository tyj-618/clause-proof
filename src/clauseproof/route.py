"""路由决策：按置信度阈值决定自动通过或转人工，以及各类别阈值的自动选择。

契约见 docs/design/0003-interfaces.md §3、tasks/T-005.md、tasks/T-008.md。
纯函数模块：无 IO，不 import llm / storage / runner / metrics。
引用质量不参与路由决策（降级已在 judge 完成）。

select_thresholds 的确定性贪心（0003 §3 要求在此写明）：
- 候选阈值：每类别 = 该类别 OK 记录实际出现过的 confidence 值（去重升序）∪ {0.0, 1.0}。
  加入 1.0 是为了让 target=1.0 时能把各类别最高置信度的判定记录也推到人工；
  confidence 恰为 1.0 的记录仍会因 >= 语义自动通过（0003 §3 已知边界）
- 初始全部类别阈值为 0.0（尽可能自动）；UNCERTAIN 与失败记录必然转人工，
  故起始复核率即不可约复核率，target 低于它时抛 ValueError
- 每轮对每个尚有候选的类别取「下一档」（候选中略大于当前阈值的值），收益比 =
  该档新增转人工记录中的错误数 / 新增转人工数（新增为 0 时记 0）；
  取收益比最大的类别提升一档，并列按 ClauseKey 定义顺序取第一个
- 若最优提升会使整体复核率（转人工数 / 全部记录数，分母含失败记录）超过
  target + 1e-9，则停止（不退而求其次）；否则应用提升并继续，直至无可提升
- 「判定正确」口径与 metrics 相同（gold.has_answer 与 present 是否 PRESENT 一致），
  按任务书要求复制该纯函数，不 import metrics 以避免循环依赖

sweep 对每个目标复核率调 select_thresholds 并按 apply_route 重算路由统计，
任一目标不可满足时整体抛 ValueError（消息含该目标与下界），不跳过、不截断。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from clauseproof.models import (
    ClauseKey,
    GoldLabel,
    Judgment,
    Presence,
    Record,
    Route,
    Status,
)

# 任务书验收口径，非可调参数
_RATE_TOLERANCE = 1e-9


def apply_route(judgment: Judgment, threshold: float) -> Route:
    """模糊判定恒转人工（与阈值无关）；其余 confidence >= threshold 自动通过。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"route: threshold 违反 0.0<=threshold<=1.0，期望 [0.0, 1.0]，实际 {threshold!r}"
        )
    if judgment.present is Presence.UNCERTAIN:
        return Route.TO_HUMAN
    if judgment.confidence >= threshold:
        return Route.AUTO_PASS
    return Route.TO_HUMAN


@dataclass(frozen=True, slots=True)
class RoutingPoint:
    """复核率–准确率曲线上的一个采样点（0003 §3，由 sweep 构造）。"""

    target_review_rate: float
    actual_review_rate: float
    auto_pass_accuracy: float | None  # 自动通过条目中判对的比例；无自动通过条目时 None
    post_review_accuracy: float  # 1 − 自动通过且错误数 / 全部条目数
    thresholds: dict[ClauseKey, float]


def select_thresholds(
    records: Sequence[Record],
    gold: Mapping[str, GoldLabel],
    *,
    target_review_rate: float,
) -> dict[ClauseKey, float]:
    """在整体复核率 ≤ target_review_rate 的前提下确定性地选各类别阈值（算法见模块 docstring）。

    返回恰好 6 个键（全部 ClauseKey），值为该类别候选集中的元素；
    同一输入两次调用返回完全相同的字典。
    """
    if not 0.0 < target_review_rate <= 1.0:
        raise ValueError(
            "route: target_review_rate 违反 0.0<target_review_rate<=1.0，"
            f"期望 (0.0, 1.0]，实际 {target_review_rate!r}"
        )
    n_total = len(records)
    human_count = _human_count(records)
    if n_total and target_review_rate < human_count / n_total - _RATE_TOLERANCE:
        raise ValueError(
            "route: target_review_rate 违反 >= 不可约复核率（UNCERTAIN 与失败记录占比），"
            f"期望 >= {human_count / n_total!r}，实际 {target_review_rate!r}"
        )
    grouped: dict[ClauseKey, list[Record]] = {clause: [] for clause in ClauseKey}
    for record in records:
        grouped[record.clause].append(record)
    candidates = {clause: _candidates(group) for clause, group in grouped.items()}
    decided = {clause: _decided(group) for clause, group in grouped.items()}
    thresholds: dict[ClauseKey, float] = {clause: 0.0 for clause in ClauseKey}
    while True:
        best = _best_raise(candidates, decided, thresholds, gold)
        if best is None:
            break
        clause, next_value, added = best
        if added and (human_count + added) / n_total > target_review_rate + _RATE_TOLERANCE:
            break
        thresholds[clause] = next_value
        human_count += added
    return thresholds


def sweep(
    records: Sequence[Record],
    gold: Mapping[str, GoldLabel],
    *,
    review_rates: Sequence[float],
) -> tuple[RoutingPoint, ...]:
    """对每个目标复核率选阈值并计算曲线点，顺序与 review_rates 一致（算法见模块 docstring）。

    任一目标低于不可约复核率时整体抛 ValueError（select_thresholds 的消息
    已含该目标与下界），不跳过该点、不截断结果。
    """
    n_total = len(records)
    points: list[RoutingPoint] = []
    for rate in review_rates:
        thresholds = select_thresholds(records, gold, target_review_rate=rate)
        n_human, n_auto, n_auto_wrong = _routing_stats(records, gold, thresholds)
        points.append(
            RoutingPoint(
                target_review_rate=rate,
                actual_review_rate=n_human / n_total if n_total else 0.0,
                auto_pass_accuracy=((n_auto - n_auto_wrong) / n_auto if n_auto else None),
                post_review_accuracy=(1.0 - n_auto_wrong / n_total if n_total else 1.0),
                thresholds=thresholds,
            )
        )
    return tuple(points)


def _human_count(records: Sequence[Record]) -> int:
    """阈值全 0.0 时的转人工数（不可约部分）：失败记录无 judgment，UNCERTAIN 恒转人工。"""
    return sum(
        1
        for record in records
        if record.status is not Status.OK or record.judgment.present is Presence.UNCERTAIN
    )


def _routing_stats(
    records: Sequence[Record],
    gold: Mapping[str, GoldLabel],
    thresholds: Mapping[ClauseKey, float],
) -> tuple[int, int, int]:
    """给定阈值下的 (转人工数, 自动通过数, 自动通过且判错数)。

    失败记录恒转人工；OK 记录按 apply_route 判定（沿用路由规则）。
    """
    n_human = 0
    n_auto = 0
    n_auto_wrong = 0
    for record in records:
        if (
            record.status is Status.OK
            and apply_route(record.judgment, thresholds[record.clause]) is Route.AUTO_PASS
        ):
            n_auto += 1
            if not _is_correct(record, gold):
                n_auto_wrong += 1
        else:
            n_human += 1
    return n_human, n_auto, n_auto_wrong


def _candidates(clause_records: Sequence[Record]) -> list[float]:
    """候选阈值 = 该类别 OK 记录实际出现过的 confidence 值（去重升序）∪ {0.0, 1.0}。

    加入 1.0 是为了让 target=1.0 时能把各类别最高置信度的判定记录也推到人工；
    confidence 恰为 1.0 的记录仍会因 >= 语义自动通过（0003 §3 已知边界）。
    """
    values = {record.judgment.confidence for record in clause_records if record.status is Status.OK}
    return sorted(values | {0.0, 1.0})


def _decided(clause_records: Sequence[Record]) -> list[Record]:
    """该类别中会随阈值改判的记录：status == OK 且 present 非 UNCERTAIN。"""
    return [
        record
        for record in clause_records
        if record.status is Status.OK and record.judgment.present is not Presence.UNCERTAIN
    ]


def _is_correct(record: Record, gold: Mapping[str, GoldLabel]) -> bool:
    """判定正确 = gold.has_answer 与 present（PRESENT/ABSENT）一致（同 metrics 口径）。"""
    return gold[record.query_id].has_answer == (record.judgment.present is Presence.PRESENT)


def _best_raise(
    candidates: Mapping[ClauseKey, list[float]],
    decided: Mapping[ClauseKey, list[Record]],
    thresholds: Mapping[ClauseKey, float],
    gold: Mapping[str, GoldLabel],
) -> tuple[ClauseKey, float, int] | None:
    """收益比（减少错误数 / 增加复核数）最大的下一档提升。

    返回 (类别, 下一档阈值, 新增转人工数)；各类别均已无可提升时 None。
    并列按 ClauseKey 定义顺序取第一个（仅严格更大才替换）。
    新增转人工数沿用 apply_route 规则：confidence 落在 [当前阈值, 下一档) 的判定记录改判。
    """
    best: tuple[ClauseKey, float, int] | None = None
    best_ratio = -1.0
    for clause in ClauseKey:
        current = thresholds[clause]
        next_value = next((value for value in candidates[clause] if value > current), None)
        if next_value is None:
            continue
        flipped = [
            record
            for record in decided[clause]
            if current <= record.judgment.confidence < next_value
        ]
        added = len(flipped)
        errors = sum(1 for record in flipped if not _is_correct(record, gold))
        ratio = errors / added if added else 0.0
        if ratio > best_ratio:
            best = (clause, next_value, added)
            best_ratio = ratio
    return best
