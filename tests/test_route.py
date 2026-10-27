"""route 的契约测试：docs/design/0003-interfaces.md §3、tasks/T-005.md 验收标准。"""

from __future__ import annotations

import pytest

from clauseproof.models import Judgment, Presence, Route
from clauseproof.route import apply_route


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
