"""数据加载层：CUAD 读取、确定性划分、长度分层抽样、Query 与 gold 构造。

契约见 docs/design/0003-interfaces.md §7、PRD FR-1；
数据源与条款范围的选择依据见 docs/adr/0006-dataset-and-clause-selection.md。
只做数据展开：不构造 Record、不调用模型（T-004 禁止事项）。
"""

from __future__ import annotations

import json
import random
import warnings
from hashlib import sha256
from pathlib import Path
from typing import NoReturn

from clauseproof.models import ClauseKey, GoldLabel, GoldSpan, Query

DEFAULT_CUAD_PATH: Path = Path(__file__).resolve().parents[2] / "data" / "cuad" / "CUAD_v1.json"

_SPLITS = ("dev", "report")
# 题面模板中类别名出现的固定锚点（T-004「数据准备」：question 形如 ... related to "X" ...）
_QUESTION_ANCHOR = 'related to "'
# CUAD 原始条款名只允许出现在 ClauseKey 定义里（0002 §2），这里由枚举值反查
_CLAUSE_BY_NAME = {clause.value: clause for clause in ClauseKey}

# 一条 CUAD 记录：(合同标题, 全文, 该合同的 qa 列表)
_Entry = tuple[str, str, tuple[dict[str, object], ...]]


def _violation(field: str, constraint: str, expected: object, actual: object) -> NoReturn:
    """按 docs/design/0004-error-semantics.md §8 的四要素格式抛出 ValueError。"""
    raise ValueError(f"dataset: {field} 违反 {constraint}，期望 {expected}，实际 {actual!r}")


def load_contracts(path: Path | None = None) -> tuple[tuple[str, str], ...]:
    """读取 CUAD，返回 ((contract_id, 全文), ...)，按 contract_id 升序，保证确定性。"""
    return tuple((title, context) for title, context, _ in _load_entries(path))


def assign_split(contract_id: str) -> str:
    """确定性划分：sha256(contract_id) 前 4 字节（大端）mod 100，< 60 → "dev"，否则 "report"。"""
    if not contract_id:
        _violation("assign_split.contract_id", "非空", "非空字符串", contract_id)
    digest = sha256(contract_id.encode("utf-8")).digest()
    return "dev" if int.from_bytes(digest[:4], "big") % 100 < 60 else "report"


def build_queries(
    *, split: str, n_contracts: int, seed: int, path: Path | None = None
) -> tuple[Query, ...]:
    """构造查询：入选合同 × 6 类条款，每份恰好 6 条。

    顺序：合同按入选顺序（短→中→长段，段内为打乱后的顺序），段内按 ClauseKey 定义序。
    Query 只含模型可见信息，不含任何 gold 字段。
    """
    entries = _select_entries(split=split, n_contracts=n_contracts, seed=seed, path=path)
    queries: list[Query] = []
    for contract_id, contract_text, _ in entries:
        for clause in ClauseKey:
            queries.append(
                Query(
                    query_id=f"{contract_id}::{clause.name}",
                    contract_id=contract_id,
                    clause=clause,
                    contract_text=contract_text,
                )
            )
    return tuple(queries)


def load_gold_with_stats(
    *, split: str, n_contracts: int, seed: int, path: Path | None = None
) -> tuple[dict[str, GoldLabel], int]:
    """构造 gold 标注，返回 (labels, 被跳过的 gold span 数)。

    query_id 集合与 build_queries 完全一致；同一 (split, n_contracts, seed) 的
    入选合同与 build_queries 相同。gold span 无法回原文逐字取回时跳过该 span
    并计数，不抛异常；跳过数 > 0 时输出一条 warning（含条数与 split，不含合同原文）。
    """
    entries = _select_entries(split=split, n_contracts=n_contracts, seed=seed, path=path)
    labels: dict[str, GoldLabel] = {}
    skipped = 0
    for contract_id, contract_text, qas in entries:
        answers_by_clause = _answers_by_clause(qas)
        for clause in ClauseKey:
            spans: list[GoldSpan] = []
            for answer in answers_by_clause.get(clause, ()):
                span = _gold_span(answer, contract_text)
                if span is None:
                    skipped += 1
                else:
                    spans.append(span)
            query_id = f"{contract_id}::{clause.name}"
            labels[query_id] = GoldLabel(
                query_id=query_id, has_answer=bool(spans), spans=tuple(spans)
            )
    if skipped:
        warnings.warn(
            f"dataset: load_gold 跳过 {skipped} 条无法回原文逐字取回的 gold span（split={split}）",
            stacklevel=2,
        )
    return labels, skipped


def load_gold(
    *, split: str, n_contracts: int, seed: int, path: Path | None = None
) -> dict[str, GoldLabel]:
    """构造 gold 标注，query_id 集合与 build_queries 完全一致（load_gold_with_stats 的薄包装）。"""
    return load_gold_with_stats(split=split, n_contracts=n_contracts, seed=seed, path=path)[0]


def _load_entries(path: Path | None) -> tuple[_Entry, ...]:
    """读文件并解析为 (title, context, qas)，按 title 升序；标题必须互不重复。"""
    src = path if path is not None else DEFAULT_CUAD_PATH
    try:
        raw = src.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"dataset: CUAD 文件不可读，期望存在且可读，实际 {src}（{exc}）"
        ) from exc
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        _violation("CUAD JSON", "合法 JSON", "SQuAD 格式", f"解析失败：{exc}")
    entries = _parse_entries(obj)
    titles = [title for title, _, _ in entries]
    if len(set(titles)) != len(titles):
        _violation("CUAD JSON.data[].title", "唯一", "互不重复的合同标题", "存在重复")
    return entries


def _parse_entries(obj: object) -> tuple[_Entry, ...]:
    """校验 SQuAD 结构并抽取 (title, paragraphs[0].context, qas)。"""
    data = obj.get("data") if isinstance(obj, dict) else None
    if not isinstance(data, list) or not data:
        _violation("CUAD JSON", '含非空 "data" 列表', "SQuAD 顶层结构", type(obj).__name__)
    entries: list[_Entry] = []
    for i, item in enumerate(data):
        where = f"CUAD JSON.data[{i}]"
        if not isinstance(item, dict):
            _violation(where, "类型", "dict", type(item).__name__)
        title = item.get("title")
        if not isinstance(title, str) or not title:
            _violation(f"{where}.title", "非空", "非空字符串", title)
        paragraphs = item.get("paragraphs")
        if not isinstance(paragraphs, list) or not paragraphs:
            _violation(f"{where}.paragraphs", "非空", "非空列表", paragraphs)
        para = paragraphs[0]
        if not isinstance(para, dict):
            _violation(f"{where}.paragraphs[0]", "类型", "dict", type(para).__name__)
        context = para.get("context")
        if not isinstance(context, str) or not context:
            _violation(
                f"{where}.paragraphs[0].context", "非空", "非空字符串", type(context).__name__
            )
        qas = para.get("qas", ())
        if not isinstance(qas, list):
            _violation(f"{where}.paragraphs[0].qas", "类型", "list", type(qas).__name__)
        for j, qa in enumerate(qas):
            if not isinstance(qa, dict):
                _violation(f"{where}.paragraphs[0].qas[{j}]", "类型", "dict", type(qa).__name__)
            if not isinstance(qa.get("question"), str):
                _violation(
                    f"{where}.paragraphs[0].qas[{j}].question",
                    "类型",
                    "str",
                    type(qa.get("question")).__name__,
                )
        entries.append((title, context, tuple(qas)))
    return tuple(sorted(entries, key=lambda entry: entry[0]))


def _select_entries(
    *, split: str, n_contracts: int, seed: int, path: Path | None
) -> tuple[_Entry, ...]:
    """分层抽样（T-004「分层抽样规则」），build_queries 与 load_gold 共用。

    规则：该 split 内合同按全文长度升序（同长按 contract_id）均分为短/中/长三段，
    余数归前面的段；各段取 ceil(n_contracts/3) 份，配额按剩余量封顶，使总数恰为
    n_contracts；三段共用一个 Random(seed)，按短→中→长顺序段内打乱后取。
    同一 (split, n_contracts, seed) 产出同一集合与同一顺序。
    """
    if split not in _SPLITS:
        _violation("dataset.split", "取值dev或report", "dev或report", split)
    if n_contracts < 1:
        _violation("dataset.n_contracts", ">=1", ">=1", n_contracts)
    pool = [entry for entry in _load_entries(path) if assign_split(entry[0]) == split]
    if len(pool) < n_contracts:
        _violation("dataset.n_contracts", "不超过该split合同数", f"<= {len(pool)}", n_contracts)
    pool.sort(key=lambda entry: (len(entry[1]), entry[0]))
    m = len(pool)
    first = (m + 2) // 3
    second = (m - first + 1) // 2
    segments = (pool[:first], pool[first : first + second], pool[first + second :])
    per = (n_contracts + 2) // 3
    rng = random.Random(seed)
    selected: list[_Entry] = []
    for segment in segments:
        remaining = n_contracts - len(selected)
        if remaining <= 0:
            break
        shuffled = list(segment)
        rng.shuffle(shuffled)
        selected.extend(shuffled[: min(per, remaining, len(segment))])
    if len(selected) < n_contracts:
        # 病态分段（如尾段为空）的兜底：按长度升序补齐，保证总数恰为 n_contracts
        chosen = {entry[0] for entry in selected}
        selected.extend(entry for entry in pool if entry[0] not in chosen)
    return tuple(selected[:n_contracts])


def _clause_name(question: str) -> str | None:
    """取题面 `related to "..."` 引号内的类别名；锚点或闭合引号缺失返回 None。"""
    anchor = question.find(_QUESTION_ANCHOR)
    if anchor == -1:
        return None
    lo = anchor + len(_QUESTION_ANCHOR)
    hi = question.find('"', lo)
    return None if hi == -1 else question[lo:hi]


def _answers_by_clause(
    qas: tuple[dict[str, object], ...],
) -> dict[ClauseKey, tuple[dict[str, object], ...]]:
    """qa 按六类条款归类；同合同同类多条时取第一条（真实数据每合同每类恰一条）。"""
    by_clause: dict[ClauseKey, list[dict[str, object]]] = {}
    for qa in qas:
        question = qa.get("question")
        name = _clause_name(question) if isinstance(question, str) else None
        clause = _CLAUSE_BY_NAME.get(name) if name is not None else None
        if clause is None or clause in by_clause:
            continue
        answers = qa.get("answers", [])
        if not isinstance(answers, list):
            _violation("CUAD JSON qas[].answers", "类型", "list", type(answers).__name__)
        by_clause[clause] = answers
    return {clause: tuple(items) for clause, items in by_clause.items()}


def _gold_span(answer: dict[str, object], context: str) -> GoldSpan | None:
    """把一条 answer 转成 GoldSpan；无法构造或无法回原文逐字取回时返回 None。"""
    text = answer.get("text")
    start = answer.get("answer_start")
    if not isinstance(text, str) or not text or not isinstance(start, int):
        return None
    if start < 0 or context[start : start + len(text)] != text:
        return None
    return GoldSpan(start=start, end=start + len(text), text=text)
