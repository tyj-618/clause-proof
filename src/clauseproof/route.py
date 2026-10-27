"""路由决策：按置信度阈值决定自动通过或转人工。

契约见 docs/design/0003-interfaces.md §3、tasks/T-005.md。
select_thresholds 不在本模块（留给 T-006）。
纯函数模块：无 IO，不 import llm / storage / runner。
引用质量不参与路由决策（降级已在 judge 完成）。
"""

from __future__ import annotations

from clauseproof.models import Judgment, Presence, Route


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
