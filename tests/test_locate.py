"""locate 的契约测试：docs/design/0003-interfaces.md §2、tasks/T-004 验收标准、T-011a 三层匹配。"""

from __future__ import annotations

import pytest

import clauseproof.locate as locate_module
from clauseproof.locate import locate
from clauseproof.models import Span


def _normalize(text: str) -> str:
    """与 locate._normalize 同口径的空白归一化，供断言核心不变量。"""
    return " ".join(text.split())


def test_output_matches_input_length_and_order() -> None:
    text = "alpha beta gamma beta"
    quotes = ["alpha", "beta", "delta", ""]
    spans = locate(quotes, text)
    assert len(spans) == len(quotes)
    assert [span.quote for span in spans] == quotes


def test_empty_input_sequence_gives_empty_output() -> None:
    assert locate([], "abc") == ()


def test_empty_quote_is_invalid_not_error() -> None:
    expected = Span(quote="", start=None, end=None, valid=False, occurrences=0, matched="none")
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
    assert span.matched == "none"
    assert span.quote == "nowhere"


def test_pure_function_same_input_same_output() -> None:
    quotes = ["beta", "", "zzz"]
    text = "alpha beta beta"
    assert locate(quotes, text) == locate(quotes, text)


def test_empty_text_violates_precondition() -> None:
    with pytest.raises(ValueError, match="text"):
        locate(["a"], "")


# ---------- 第 1 层：精确匹配 ----------


def test_exact_quote_stays_in_tier_one() -> None:
    """原文含连续空格、quote 也原样复制了连续空格时，仍走第 1 层。"""
    text = "The   Parties shall"
    (span,) = locate(["The   Parties"], text)
    assert span.matched == "exact"
    assert span.valid is True
    assert span.start == text.find("The   Parties")
    assert span.end == span.start + len("The   Parties")
    assert text[span.start : span.end] == "The   Parties"


def test_exact_match_takes_priority_over_normalization() -> None:
    """精确命中时即使归一化也能命中，仍按第 1 层报告。"""
    text = "alpha   beta alpha beta"
    (span,) = locate(["alpha beta"], text)  # 末尾恰有一处精确出现
    assert span.matched == "exact"
    assert span.occurrences == 1
    assert text[span.start : span.end] == "alpha beta"


# ---------- 第 2 层：空白归一化匹配 ----------


def test_whitespace_only_difference_matches_normalized() -> None:
    """CUAD 原文含连续空格、模型复制时归一化为单空格 → 第 2 层。"""
    text = "The   Parties shall"
    (span,) = locate(["The Parties"], text)
    assert span.matched == "normalized"
    assert span.valid is True
    # 核心不变量：区间文本与 quote 归一化后相等
    assert _normalize(text[span.start : span.end]) == _normalize("The Parties")
    # 区间包含原文的连续空格（而非把空白剥掉后的碎片）
    assert text[span.start : span.end] == "The   Parties"
    assert "  " in text[span.start : span.end]


def test_normalized_core_invariant_on_tab_and_newline_runs() -> None:
    """空白 run 含制表符/换行时同样折叠匹配，区间映射回原文。"""
    text = "a\tb\t\tc\n\nd  e"
    (span,) = locate(["b c d"], text)
    assert span.matched == "normalized"
    assert _normalize(text[span.start : span.end]) == _normalize("b c d")
    assert text[span.start : span.end] == "b\t\tc\n\nd"


def test_normalized_occurrences_counted_in_normalized_text() -> None:
    """归一化匹配的 occurrences 按归一化文本计数（非重叠）。"""
    text = "x   y   z x    y"
    # 归一化文本为 "x y z x y"，"x y" 出现 2 次；原文无单空格的 "x y"，精确层落空
    (span,) = locate(["x y"], text)
    assert span.matched == "normalized"
    assert span.occurrences == 2
    assert span.start == 0
    assert text[span.start : span.end] == "x   y"


def test_quote_with_surrounding_whitespace_strips_to_content() -> None:
    """quote 首尾含空白：匹配内容字符，区间不含两侧原文空白。"""
    text = "  The   Parties  agree"
    (span,) = locate(["  The Parties "], text)
    assert span.matched == "normalized"
    assert _normalize(text[span.start : span.end]) == _normalize(" The Parties ")
    assert text[span.start : span.end] == "The   Parties"
    assert text[span.start] == "T"  # 不含前导空白
    assert text[span.end - 1] == "s"  # 不含尾随空白


def test_all_whitespace_quote_falls_through_to_none() -> None:
    """非空但全空白的 quote：精确层未命中、归一化后为空 → none。"""
    text = "ab"
    (span,) = locate([" "], text)
    assert span.matched == "none"
    assert span.valid is False
    assert span.occurrences == 0


# ---------- 第 3 层：不可定位 ----------


def test_rewritten_or_truncated_quote_is_none() -> None:
    """改写（词被替换）与截断（尾部带省略号等残迹）→ matched="none"。"""
    text = "The   Parties shall negotiate"
    for quote in ("The Counterparties shall", "The Parties shall…"):
        (span,) = locate([quote], text)
        assert span.matched == "none"
        assert span.valid is False
        assert span.start is None
        assert span.end is None
        assert span.occurrences == 0


# ---------- 归一化文本的懒构建 ----------


def test_normalized_text_built_at_most_once_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """一次 locate 调用内多个第 2 层 quote 只构建一次归一化文本。"""
    original = locate_module._build_normalized
    calls: list[int] = []

    def counting(text: str) -> tuple[str, tuple[int, ...]]:
        calls.append(1)
        return original(text)

    monkeypatch.setattr(locate_module, "_build_normalized", counting)
    spans = locate(["The Parties", "Parties  shall", "missing"], "The   Parties   shall")
    assert [span.matched for span in spans] == ["normalized", "normalized", "none"]
    assert len(calls) == 1


def test_normalized_text_not_built_when_all_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """全部第 1 层命中时不构建归一化文本（懒初始化）。"""
    calls: list[int] = []

    def counting(text: str) -> tuple[str, tuple[int, ...]]:
        calls.append(1)
        raise AssertionError("不应构建归一化文本")

    monkeypatch.setattr(locate_module, "_build_normalized", counting)
    (span,) = locate(["The   Parties"], "The   Parties")
    assert span.matched == "exact"
    assert calls == []


def test_long_text_normalization_is_built_once_and_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """10 万字符长文本：归一化只构建一次，且定位可完成（性能不退化到超时）。"""
    original = locate_module._build_normalized
    calls: list[int] = []

    def counting(text: str) -> tuple[str, tuple[int, ...]]:
        calls.append(1)
        return original(text)

    monkeypatch.setattr(locate_module, "_build_normalized", counting)
    # 10 万字符，每 10 个字符一组，组间是 3 个连续空格
    chunk = "abcdefghij"
    text = (chunk + "   ") * 10_000
    quotes = [chunk + " " + chunk, "zzz-not-there", chunk + "  " + chunk]
    spans = locate(quotes, text)
    assert [span.matched for span in spans] == ["normalized", "none", "normalized"]
    for span in spans:
        if span.valid:
            assert _normalize(text[span.start : span.end]) == _normalize(span.quote)
    assert len(calls) == 1
