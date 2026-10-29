"""引用定位：把模型声称的 quote 定位为原文中的字符区间。

契约见 docs/design/0003-interfaces.md §2、PRD FR-3。
纯函数模块：无 IO，不 import llm / storage / runner。

三层匹配（依次降级）：
1. 精确匹配：quote 在原文中逐字出现（现有行为，保持不变）
2. 空白归一化匹配：两侧空白 run 折叠为单空格（首尾丢弃）后子串匹配，
   用归一化文本的 pos[] 索引把区间映射回原文
3. matched="none"：不可定位
"""

from __future__ import annotations

from collections.abc import Sequence

from clauseproof.models import Span


def locate(quotes: Sequence[str], text: str) -> tuple[Span, ...]:
    """每个 quote 产出一个 Span，顺序与输入一一对应。

    - 前置：text 非空，违反抛 ValueError
    - 空字符串 quote → valid=False、occurrences=0、matched="none"，不抛错
      （模型可能输出空引用，属可统计的失败，不应中断运行）
    - 第 1 层：occurrences = text.count(quote)；出现 ≥ 1 次取首次出现为区间，
      matched="exact"
    - 第 2 层（仅当第 1 层落空）：在空白归一化文本上计数与首次定位，
      matched="normalized"；区间经 pos[] 映射回原文，满足核心不变量
      normalize(text[start:end]) == normalize(quote)，且包含原文的连续空白
    - 第 3 层：matched="none"、valid=False
    - 归一化文本在首次需要第 2 层时才构建，一次 locate 调用内最多构建一次
    """
    if not text:
        raise ValueError("locate: text 违反 非空，期望 非空字符串，实际 ''")
    spans: list[Span] = []
    normalized: tuple[str, tuple[int, ...]] | None = None
    for quote in quotes:
        if not quote:
            spans.append(
                Span(quote=quote, start=None, end=None, valid=False, occurrences=0, matched="none")
            )
            continue
        occurrences = text.count(quote)
        if occurrences > 0:
            start = text.find(quote)
            spans.append(
                Span(
                    quote=quote,
                    start=start,
                    end=start + len(quote),
                    valid=True,
                    occurrences=occurrences,
                    matched="exact",
                )
            )
            continue
        if normalized is None:
            normalized = _build_normalized(text)
        normalized_text, pos = normalized
        normalized_quote = _normalize(quote)
        normalized_occurrences = normalized_text.count(normalized_quote) if normalized_quote else 0
        if normalized_occurrences > 0:
            normalized_start = normalized_text.find(normalized_quote)
            # 归一化 quote 首尾必为非空白字符（_normalize 已去首尾空白），
            # 故两端都映射到内容字符，区间自然包含中间的原文连续空白
            start = pos[normalized_start]
            end = pos[normalized_start + len(normalized_quote) - 1] + 1
            spans.append(
                Span(
                    quote=quote,
                    start=start,
                    end=end,
                    valid=True,
                    occurrences=normalized_occurrences,
                    matched="normalized",
                )
            )
        else:
            spans.append(
                Span(quote=quote, start=None, end=None, valid=False, occurrences=0, matched="none")
            )
    return tuple(spans)


def _normalize(text: str) -> str:
    """空白归一化：任意空白 run 折叠为单个空格，首尾空白丢弃。

    与 " ".join(text.split()) 等价；全空白输入归一化为空串。
    """
    return " ".join(text.split())


def _build_normalized(text: str) -> tuple[str, tuple[int, ...]]:
    """构建归一化文本与位置索引：pos[i] 为归一化文本第 i 个字符在原文中的下标。

    规则与 _normalize 一致。折叠空格的 pos 取该空白 run 的首字符下标；由于归一化
    后的 quote 首尾必为非空白字符，该取值不影响映射出的区间端点。
    """
    chars: list[str] = []
    pos: list[int] = []
    i = 0
    length = len(text)
    while i < length:
        if text[i].isspace():
            run_start = i
            while i < length and text[i].isspace():
                i += 1
            # 首部（chars 为空）与尾部（i == length）的空白直接丢弃
            if chars and i < length:
                chars.append(" ")
                pos.append(run_start)
        else:
            chars.append(text[i])
            pos.append(i)
            i += 1
    return "".join(chars), tuple(pos)
