"""校准度量：等频分桶的可靠性曲线与期望校准误差（ECE）。

契约见 docs/design/0003-interfaces.md §9、tasks/T-007.md「校准」。
纯函数模块：无 IO，不 import llm / storage / runner。

分桶规则（tasks/T-007.md）：
- 等频：pairs 按 confidence 升序排序后等分，除最后一桶可略多外每桶样本数相同；
  样本数不足 n_bins 时收缩桶数，保证每桶非空
- 合并：从头累积，累积量达到 min_bin_size 即封桶；尾部残余并入前一个桶；
  全部样本不足 min_bin_size 时返回单桶。由此除单桶情形外每桶样本数 >= min_bin_size
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """可靠性曲线的一个桶：confidence 落在 [lo, hi] 的样本统计。"""

    lo: float
    hi: float
    n: int
    mean_confidence: float
    accuracy: float


def reliability_bins(
    pairs: Sequence[tuple[float, bool]],
    *,
    n_bins: int = 10,
    min_bin_size: int = 30,
) -> tuple[CalibrationBin, ...]:
    """把 (confidence, 是否正确) 序列等频分桶。

    - confidence 越界（<0 或 >1）抛 ValueError
    - 空输入返回空元组
    """
    if n_bins < 1:
        raise ValueError(f"calibration: n_bins 违反 >=1，期望 >=1，实际 {n_bins}")
    if min_bin_size < 1:
        raise ValueError(f"calibration: min_bin_size 违反 >=1，期望 >=1，实际 {min_bin_size}")
    for confidence, _ in pairs:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"calibration: pairs.confidence 违反 0<=c<=1，期望 [0.0, 1.0]，实际 {confidence!r}"
            )
    if not pairs:
        return ()
    ordered = sorted(pairs, key=lambda pair: pair[0])
    n = len(ordered)
    # 等频切分：前 effective-1 桶各 base 个，最后一桶拿剩余（可略多）
    effective = min(n_bins, n)
    base = n // effective
    chunks: list[list[tuple[float, bool]]] = [
        list(ordered[i * base : (i + 1) * base]) for i in range(effective - 1)
    ]
    chunks.append(list(ordered[(effective - 1) * base :]))
    # 小桶合并：累积到 min_bin_size 封桶，尾部残余并入前一个桶
    merged: list[list[tuple[float, bool]]] = []
    current: list[tuple[float, bool]] = []
    for chunk in chunks:
        current.extend(chunk)
        if len(current) >= min_bin_size:
            merged.append(current)
            current = []
    if current:
        if merged:
            merged[-1].extend(current)
        else:
            merged.append(current)
    return tuple(_make_bin(items) for items in merged)


def _make_bin(items: Sequence[tuple[float, bool]]) -> CalibrationBin:
    """单个桶的统计量；items 已按 confidence 升序排列。"""
    n = len(items)
    return CalibrationBin(
        lo=items[0][0],
        hi=items[-1][0],
        n=n,
        mean_confidence=sum(confidence for confidence, _ in items) / n,
        accuracy=sum(1 for _, correct in items if correct) / n,
    )


def expected_calibration_error(bins: Sequence[CalibrationBin]) -> float | None:
    """ECE = Σ(nᵢ/N)·|accᵢ − confᵢ|，只统计非空桶。

    空输入、N==0 或非空桶数 < 2 时返回 None：单桶时该值退化为
    |平均准确率 − 平均置信度|，不构成校准误差，返回 None 以显式表示
    「样本不足以评估校准」。
    """
    non_empty = [bin_ for bin_ in bins if bin_.n > 0]
    total = sum(bin_.n for bin_ in non_empty)
    if len(non_empty) < 2 or total == 0:
        return None
    return sum(bin_.n / total * abs(bin_.accuracy - bin_.mean_confidence) for bin_ in non_empty)
