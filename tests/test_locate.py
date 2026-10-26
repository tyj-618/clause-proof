"""locate 的契约测试：docs/design/0003-interfaces.md §2、tasks/T-004 验收标准。"""

from __future__ import annotations

import pytest

from clauseproof.locate import locate
from clauseproof.models import Span


def test_output_matches_input_length_and_order() -> None:
    text = "alpha beta gamma beta"
    quotes = ["alpha", "beta", "delta", ""]
    spans = locate(quotes, text)
    assert len(spans) == len(quotes)
    assert [span.quote for span in spans] == quotes


def test_empty_input_sequence_gives_empty_output() -> None:
    assert locate([], "abc") == ()


def test_empty_quote_is_invalid_not_error() -> None:
    expected = Span(quote="", start=None, end=None, valid=False, occurrences=0)
    assert locate([""], "some text") == (expected,)
    # 混在有效 quote 中间也不抛异常
    spans = locate(["text", "", "some"], "some text")
    assert spans[1] == expected


def test_occurrences_equals_count_and_validity_iff_found() -> None:
    text = "banana apple banana cherry"
    cases = {"banana": 2, "apple": 1, "cherry": 1, "durian": 0}
    for quote, expected in cases.items():
        (span,) = locate([quote], text)
        assert span.occurrences == text.count(quote) == expected
        assert span.valid is (expected >= 1)


def test_non_overlapping_count_semantics() -> None:
    # text.count 按非重叠计数："aaa" 中 "aa" 只计 1 次
    (span,) = locate(["aa"], "aaa")
    assert span.occurrences == 1
    assert span.valid is True


@pytest.mark.parametrize(
    "quote",
    ["one", "two", "three", "one two three", "o", "e"],
)
def test_valid_span_bounds_slice_back_to_quote(quote: str) -> None:
    text = "one two three"
    (span,) = locate([quote], text)
    assert span.start == text.find(quote)
    assert span.end == span.start + len(quote)
    assert text[span.start : span.end] == quote


def test_first_occurrence_wins_with_honest_occurrences() -> None:
    text = "x ay x ay x"
    (span,) = locate(["ay"], text)
    assert span.occurrences == 2
    assert span.start == text.find("ay")
    assert text[span.start : span.end] == "ay"


def test_missing_quote_has_no_bounds() -> None:
    (span,) = locate(["nowhere"], "some text")
    assert span.valid is False
    assert span.start is None
    assert span.end is None
    assert span.occurrences == 0
    assert span.quote == "nowhere"


def test_pure_function_same_input_same_output() -> None:
    quotes = ["beta", "", "zzz"]
    text = "alpha beta beta"
    assert locate(quotes, text) == locate(quotes, text)


def test_empty_text_violates_precondition() -> None:
    with pytest.raises(ValueError, match="text"):
        locate(["a"], "")
