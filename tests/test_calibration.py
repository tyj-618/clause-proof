"""calibration 的契约测试：docs/design/0003-interfaces.md §9、tasks/T-007.md「校准」。"""

from __future__ import annotations

import pytest

from clauseproof.calibration import (
    CalibrationBin,
    expected_calibration_error,
    reliability_bins,
)


def _distinct_confidences(n: int, *, correct: bool = True) -> list[tuple[float, bool]]:
    """n 个互不相同的 confidence，升序覆盖 (0, 1)。"""
    return [(i / n + 1 / (2 * n), correct) for i in range(n)]


def test_empty_input_returns_empty_tuple_and_none_ece() -> None:
    assert reliability_bins([]) == ()
    assert expected_calibration_error(()) is None


def test_equal_frequency_split_with_min_bin_size_one() -> None:
    bins = reliability_bins(_distinct_confidences(100), n_bins=10, min_bin_size=1)
    assert len(bins) == 10
    assert all(bin_.n == 10 for bin_ in bins)


def test_last_bin_absorbs_remainder() -> None:
    # 25 条切 10 桶：前 9 桶各 2，最后一桶 7（任务书「最后一桶可略多」）
    bins = reliability_bins(_distinct_confidences(25), n_bins=10, min_bin_size=1)
    assert [bin_.n for bin_ in bins] == [2] * 9 + [7]


def test_fewer_samples_than_bins_shrinks_bin_count() -> None:
    bins = reliability_bins(_distinct_confidences(5), n_bins=10, min_bin_size=1)
    assert [bin_.n for bin_ in bins] == [1, 1, 1, 1, 1]


def test_small_bins_merged_up_to_min_bin_size() -> None:
    # 100 条等频 10 桶（每桶 10 < 30）：累积合并为 [30, 30, 40]
    bins = reliability_bins(_distinct_confidences(100), n_bins=10, min_bin_size=30)
    assert [bin_.n for bin_ in bins] == [30, 30, 40]
    assert all(bin_.n >= 30 for bin_ in bins)


def test_all_samples_below_min_bin_size_yield_single_bin() -> None:
    bins = reliability_bins(_distinct_confidences(25), n_bins=10, min_bin_size=30)
    assert len(bins) == 1
    assert bins[0].n == 25


def test_bin_lo_hi_mean_confidence_and_accuracy() -> None:
    pairs = [(0.3, True), (0.1, False), (0.2, True), (0.4, False)]
    bins = reliability_bins(pairs, n_bins=2, min_bin_size=1)
    assert (bins[0].lo, bins[0].hi) == (0.1, 0.2)
    assert (bins[1].lo, bins[1].hi) == (0.3, 0.4)
    assert bins[0].mean_confidence == pytest.approx(0.15)
    assert bins[0].accuracy == pytest.approx(0.5)
    assert bins[1].accuracy == pytest.approx(0.5)


def test_perfect_calibration_has_near_zero_ece() -> None:
    # 每个置信度档位内正确比例恰等于置信度本身
    pairs: list[tuple[float, bool]] = []
    for tenth in range(1, 11):
        pairs.extend((tenth / 10, i < tenth) for i in range(10))
    ece = expected_calibration_error(reliability_bins(pairs))
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_all_correct_at_half_confidence_has_ece_half() -> None:
    ece = expected_calibration_error(reliability_bins([(0.5, True)] * 100))
    assert ece == pytest.approx(0.5)


def test_single_bin_ece_is_none() -> None:
    single = CalibrationBin(lo=0.0, hi=1.0, n=25, mean_confidence=0.5, accuracy=1.0)
    assert expected_calibration_error([single]) is None


def test_single_bin_from_reliability_bins_ece_is_none() -> None:
    bins = reliability_bins(_distinct_confidences(25), n_bins=10, min_bin_size=30)
    assert len(bins) == 1
    assert expected_calibration_error(bins) is None


def test_two_bins_ece_matches_hand_computation() -> None:
    bins = [
        CalibrationBin(lo=0.1, hi=0.3, n=10, mean_confidence=0.2, accuracy=1.0),
        CalibrationBin(lo=0.5, hi=0.9, n=30, mean_confidence=0.8, accuracy=0.5),
    ]
    # (10/40)×|1.0−0.2| + (30/40)×|0.5−0.8| = 0.2 + 0.225 = 0.425
    assert expected_calibration_error(bins) == pytest.approx(0.425)


@pytest.mark.parametrize("confidence", [-0.1, -0.001, 1.0001, 2.0])
def test_out_of_range_confidence_raises_value_error(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        reliability_bins([(confidence, True)])


@pytest.mark.parametrize("kwargs", [{"n_bins": 0}, {"min_bin_size": 0}])
def test_non_positive_bin_parameters_raise_value_error(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        reliability_bins([(0.5, True)], **kwargs)
