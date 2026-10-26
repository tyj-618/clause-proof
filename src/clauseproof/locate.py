"""引用定位：把模型声称的 quote 定位为原文中的字符区间。

契约见 docs/design/0003-interfaces.md §2、PRD FR-3。
纯函数模块：无 IO，不 import llm / storage / runner。
"""

from __future__ import annotations

from collections.abc import Sequence

from clauseproof.models import Span


def locate(quotes: Sequence[str], text: str) -> tuple[Span, ...]:
    """每个 quote 产出一个 Span，顺序与输入一一对应。

    - 前置：text 非空，违反抛 ValueError
    - 空字符串 quote → valid=False、occurrences=0，不抛错
      （模型可能输出空引用，属可统计的失败，不应中断运行）
    - 其余 quote：occurrences = text.count(quote)；出现 ≥ 1 次取首次出现为区间
    """
    if not text:
        raise ValueError("locate: text 违反 非空，期望 非空字符串，实际 ''")
    spans: list[Span] = []
    for quote in quotes:
        if not quote:
            spans.append(Span(quote=quote, start=None, end=None, valid=False, occurrences=0))
            continue
        occurrences = text.count(quote)
        if occurrences == 0:
            spans.append(Span(quote=quote, start=None, end=None, valid=False, occurrences=0))
        else:
            start = text.find(quote)
            spans.append(
                Span(
                    quote=quote,
                    start=start,
                    end=start + len(quote),
                    valid=True,
                    occurrences=occurrences,
                )
            )
    return tuple(spans)
