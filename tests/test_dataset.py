"""dataset 的契约测试：docs/design/0003-interfaces.md §7、tasks/T-004 验收标准。

单元测试全部使用内置的极小 CUAD 样本（不读取真实 38MB 文件）；
真实文件上的端到端校验见文末 @pytest.mark.slow 测试，默认被 -m "not slow" 排除。
"""

from __future__ import annotations

import json
from dataclasses import fields
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from clauseproof.dataset import (
    DEFAULT_CUAD_PATH,
    assign_split,
    build_queries,
    load_contracts,
    load_gold,
    load_gold_with_stats,
)
from clauseproof.models import ClauseKey, GoldLabel, Query

# ---- 极小样本构造 ----


def _question(clause: ClauseKey) -> str:
    """按 CUAD 题面模板构造问题；类别名取自 ClauseKey.value，不写原始名字面量。"""
    return (
        "Highlight the parts (if any) of this contract related to "
        f'"{clause.value}" that should be reviewed by a lawyer. Details: stub.'
    )


def _make_cuad(
    contracts: list[tuple[str, str, dict[ClauseKey, list[Any] | None]]],
) -> dict[str, Any]:
    """构造 CUAD 结构样本。

    contracts 每项为 (标题, 全文, {条款: 答案列表})。答案为 str 时按 text.index
    自动定位（保证可回原文取回）；None 表示该 qa 不带 answers 键。
    """
    data: list[dict[str, Any]] = []
    for title, text, answers_by_clause in contracts:
        qas: list[dict[str, Any]] = []
        for clause, answers in answers_by_clause.items():
            qa: dict[str, Any] = {"question": _question(clause), "id": f"{title}__{clause.value}"}
            if answers is not None:
                qa["answers"] = [
                    {"text": answer, "answer_start": text.index(answer)}
                    if isinstance(answer, str)
                    else answer
                    for answer in answers
                ]
            qas.append(qa)
        data.append({"title": title, "paragraphs": [{"context": text, "qas": qas}]})
    return {"version": "v1.0", "data": data}


def _write(sample: dict[str, Any], tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(sample), encoding="utf-8")
    return path


# ---- 小样本：3 份合同（dev 2 + report 1），覆盖有答案/空答案/无 answers 键/整题缺失 ----

_ACME = "ACME_01_SUPPLY_AGREEMENT"  # dev
_BRAVO = "BRAVO_02_DISTRIBUTION"  # dev
_ECHO = "ECHO_05_LICENSE"  # report

_ACME_TEXT = (
    "ACME SUPPLY AGREEMENT. "
    "This Agreement shall be governed by the laws of the State of New York. "
    "Neither party may assign this Agreement without the prior written consent. "
    "No delegation of duties is permitted."
)
_BRAVO_TEXT = (
    "BRAVO DISTRIBUTION AGREEMENT. "
    "Liability of each party shall be capped at one million dollars. "
    "Either party may terminate upon written notice."
)
_ECHO_TEXT = (
    "ECHO LICENSE. Either party may terminate this license for convenience with thirty days notice."
)


def _small_sample() -> dict[str, Any]:
    return _make_cuad(
        [
            (
                _ACME,
                _ACME_TEXT,
                {
                    ClauseKey.GOVERNING_LAW: [
                        "This Agreement shall be governed by the laws of the State of New York."
                    ],
                    ClauseKey.ANTI_ASSIGNMENT: [
                        "Neither party may assign this Agreement without "
                        "the prior written consent.",
                        "No delegation of duties is permitted.",
                    ],
                },
            ),
            (
                _BRAVO,
                _BRAVO_TEXT,
                {
                    ClauseKey.CAP_ON_LIABILITY: [
                        "Liability of each party shall be capped at one million dollars."
                    ],
                    ClauseKey.TERMINATION_FOR_CONVENIENCE: [],  # 题在、answers 为空
                    ClauseKey.UNCAP_LIABILITY: None,  # 题在、无 answers 键
                },
            ),
            (
                _ECHO,
                _ECHO_TEXT,
                {
                    ClauseKey.TERMINATION_FOR_CONVENIENCE: [
                        "Either party may terminate this license for convenience "
                        "with thirty days notice."
                    ],
                },
            ),
        ]
    )


def test_small_sample_split_makeup() -> None:
    """夹具假设：小样本 dev 2 份、report 1 份（改标题会改变 sha256 划分，需重选）。"""
    assert assign_split(_ACME) == "dev"
    assert assign_split(_BRAVO) == "dev"
    assert assign_split(_ECHO) == "report"


# ---- 大样本：9 份合同（dev 6 + report 3），长度各异；用于分层抽样与 seed 敏感性 ----

_LARGE_DEV = [
    ("CHARLIE_03_LEASE", 100),
    ("DELTA_04_SERVICES", 140),
    ("FOXTROT_06_NDA", 180),
    ("GOLF_07_CONSULTING", 220),
    ("HOTEL_08_FRANCHISE", 260),
    ("INDIA_09_JOINT_VENTURE", 300),
]
_LARGE_REPORT = [
    ("KILO_11_MARKETING", 120),
    ("OSCAR_15_LEASE", 160),
    ("QUEBEC_17_LICENSE", 200),
]


def _fill(title: str, target_len: int) -> str:
    """以条款句开头、空格填充到恰好 target_len 个字符（长度决定分层归属）。"""
    body = f"{title}. This agreement is governed by the laws of State 42."
    assert len(body) < target_len
    return body + " " * (target_len - len(body))


def _large_sample() -> dict[str, Any]:
    contracts = [
        (
            title,
            _fill(title, target_len),
            {ClauseKey.GOVERNING_LAW: ["This agreement is governed by the laws of State 42."]},
        )
        for title, target_len in _LARGE_DEV + _LARGE_REPORT
    ]
    return _make_cuad(contracts)


def _large_path(tmp_path: Path) -> Path:
    return _write(_large_sample(), tmp_path, "large.json")


def test_large_sample_makeup(tmp_path: Path) -> None:
    """夹具假设：大样本 dev 恰 6 份（三段各 2）、report 3 份。"""
    contracts = load_contracts(_large_path(tmp_path))
    dev = [cid for cid, _ in contracts if assign_split(cid) == "dev"]
    report = [cid for cid, _ in contracts if assign_split(cid) == "report"]
    assert len(dev) == 6
    assert len(report) == 3


# ---- load_contracts ----


def test_load_contracts_sorted_and_deterministic(tmp_path: Path) -> None:
    path = _write(_small_sample(), tmp_path, "small.json")
    first = load_contracts(path)
    assert first == load_contracts(path)
    ids = [cid for cid, _ in first]
    assert ids == sorted(ids) == sorted([_ACME, _BRAVO, _ECHO])
    texts = dict(first)
    assert texts[_ACME] == _ACME_TEXT
    assert texts[_ECHO] == _ECHO_TEXT


def test_load_contracts_missing_file_raises_runtime_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="不可读"):
        load_contracts(tmp_path / "nope.json")


@pytest.mark.parametrize(
    ("payload", "where"),
    [
        ("{not json", "JSON"),
        ('{"data": []}', "data"),
        ('{"data": [42]}', "0"),
        ('{"data": [{"title": "", "paragraphs": [{"context": "x", "qas": []}]}]}', "title"),
        ('{"data": [{"title": "t", "paragraphs": []}]}', "paragraphs"),
        ('{"data": [{"title": "t", "paragraphs": [{"context": "", "qas": []}]}]}', "context"),
        ('{"data": [{"title": "t", "paragraphs": [{"context": "x", "qas": 1}]}]}', "qas"),
    ],
)
def test_load_contracts_bad_structure_raises_value_error(
    tmp_path: Path, payload: str, where: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=where):
        load_contracts(path)


def test_load_contracts_duplicate_titles_raise(tmp_path: Path) -> None:
    sample = _make_cuad([(_ACME, _ACME_TEXT, {}), (_ACME, _BRAVO_TEXT, {})])
    path = _write(sample, tmp_path, "dup.json")
    with pytest.raises(ValueError, match="唯一"):
        load_contracts(path)


# ---- assign_split ----


def test_assign_split_matches_spec_formula() -> None:
    for cid in [_ACME, _BRAVO, _ECHO, "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT", "x"]:
        digest = sha256(cid.encode("utf-8")).digest()
        expected = "dev" if int.from_bytes(digest[:4], "big") % 100 < 60 else "report"
        assert assign_split(cid) == expected
        assert assign_split(cid) == assign_split(cid)


def test_assign_split_dev_and_report_disjoint() -> None:
    ids = [title for title, _ in _LARGE_DEV + _LARGE_REPORT] + [_ACME, _BRAVO, _ECHO]
    dev = {cid for cid in ids if assign_split(cid) == "dev"}
    report = {cid for cid in ids if assign_split(cid) == "report"}
    assert dev & report == set()
    assert dev | report == set(ids)


def test_assign_split_empty_id_raises() -> None:
    with pytest.raises(ValueError, match="contract_id"):
        assign_split("")


# ---- build_queries / load_gold：形状、一致性与 gold 语义 ----


def test_build_queries_shape_and_determinism(tmp_path: Path) -> None:
    path = _write(_small_sample(), tmp_path, "small.json")
    queries = build_queries(split="dev", n_contracts=2, seed=7, path=path)
    assert len(queries) == 12  # 2 份合同 × 6 类条款
    assert queries == build_queries(split="dev", n_contracts=2, seed=7, path=path)
    assert {query.contract_id for query in queries} == {_ACME, _BRAVO}
    # 每份合同恰好 6 条，同一合同的 6 条连续且按 ClauseKey 定义序
    assert [query.clause for query in queries[:6]] == list(ClauseKey)
    assert len({query.contract_id for query in queries[:6]}) == 1
    texts = dict(load_contracts(path))
    for query in queries:
        assert assign_split(query.contract_id) == "dev"
        assert query.query_id == f"{query.contract_id}::{query.clause.name}"
        assert query.contract_text == texts[query.contract_id]


def test_query_has_no_gold_fields() -> None:
    """Query 字段固定为四个，结构上不含任何 gold 字段（0002 §1 gold 隔离）。"""
    assert {field.name for field in fields(Query)} == {
        "query_id",
        "contract_id",
        "clause",
        "contract_text",
    }


def test_build_queries_and_load_gold_keys_match(tmp_path: Path) -> None:
    path = _write(_small_sample(), tmp_path, "small.json")
    queries = build_queries(split="dev", n_contracts=2, seed=7, path=path)
    gold = load_gold(split="dev", n_contracts=2, seed=7, path=path)
    assert {query.query_id for query in queries} == set(gold)
    assert len(gold) == 12


def test_gold_labels_semantics_and_verbatim_spans(tmp_path: Path) -> None:
    path = _write(_small_sample(), tmp_path, "small.json")
    queries = build_queries(split="dev", n_contracts=2, seed=7, path=path)
    gold = load_gold(split="dev", n_contracts=2, seed=7, path=path)
    by_id = {query.query_id: query for query in queries}

    def label(clause: ClauseKey, cid: str) -> GoldLabel:
        return gold[f"{cid}::{clause.name}"]

    assert label(ClauseKey.GOVERNING_LAW, _ACME).has_answer is True
    assert len(label(ClauseKey.GOVERNING_LAW, _ACME).spans) == 1
    assert len(label(ClauseKey.ANTI_ASSIGNMENT, _ACME).spans) == 2
    # 题在但 answers 为空 / 无 answers 键 → 条款不存在
    assert label(ClauseKey.TERMINATION_FOR_CONVENIENCE, _BRAVO).has_answer is False
    assert label(ClauseKey.TERMINATION_FOR_CONVENIENCE, _BRAVO).spans == ()
    assert label(ClauseKey.UNCAP_LIABILITY, _BRAVO).has_answer is False
    # 整题缺失（ACME 没有 CAP 的题）→ 同样不存在
    assert label(ClauseKey.CAP_ON_LIABILITY, _ACME).has_answer is False

    # 全部 gold span 可回原文逐字取回
    for query_id, lab in gold.items():
        text = by_id[query_id].contract_text
        for span in lab.spans:
            assert text[span.start : span.end] == span.text


def test_gold_on_report_split(tmp_path: Path) -> None:
    path = _write(_small_sample(), tmp_path, "small.json")
    queries = build_queries(split="report", n_contracts=1, seed=7, path=path)
    gold = load_gold(split="report", n_contracts=1, seed=7, path=path)
    assert {query.contract_id for query in queries} == {_ECHO}
    lab = gold[f"{_ECHO}::{ClauseKey.TERMINATION_FOR_CONVENIENCE.name}"]
    assert lab.has_answer is True
    assert len(lab.spans) == 1


def test_bad_gold_spans_skipped_and_counted(tmp_path: Path) -> None:
    sample = _small_sample()
    for entry in sample["data"]:
        for qa in entry["paragraphs"][0]["qas"]:
            if entry["title"] == _ACME and qa["question"] == _question(ClauseKey.ANTI_ASSIGNMENT):
                # 第二条答案给错误偏移：取不回原文，跳过但保留好的
                qa["answers"][1] = {"text": "ghost clause text", "answer_start": 3}
            if entry["title"] == _BRAVO and qa["question"] == _question(ClauseKey.CAP_ON_LIABILITY):
                # 唯一答案越界：该条款退化为无答案
                qa["answers"][0] = {"text": "ghost", "answer_start": 9999}
    path = _write(sample, tmp_path, "small_bad.json")
    with pytest.warns(UserWarning, match="跳过 2 条"):
        gold, skipped = load_gold_with_stats(split="dev", n_contracts=2, seed=7, path=path)
    assert skipped == 2
    anti = gold[f"{_ACME}::{ClauseKey.ANTI_ASSIGNMENT.name}"]
    assert len(anti.spans) == 1
    assert anti.has_answer is True
    cap = gold[f"{_BRAVO}::{ClauseKey.CAP_ON_LIABILITY.name}"]
    assert cap.spans == ()
    assert cap.has_answer is False


def test_load_gold_with_stats_returns_labels_and_skip_count(tmp_path: Path) -> None:
    """含 1 条错误 answer_start 的样本：返回 (labels, 1)，不抛异常。"""
    sample = _small_sample()
    for entry in sample["data"]:
        if entry["title"] == _ACME:
            for qa in entry["paragraphs"][0]["qas"]:
                if qa["question"] == _question(ClauseKey.ANTI_ASSIGNMENT):
                    qa["answers"][1] = {"text": "ghost clause text", "answer_start": 3}
    path = _write(sample, tmp_path, "one_bad.json")
    with pytest.warns(UserWarning, match="跳过 1 条"):
        gold, skipped = load_gold_with_stats(split="dev", n_contracts=2, seed=7, path=path)
    assert skipped == 1
    assert len(gold) == 12
    # 好的 span 保留，坏的跳过
    anti = gold[f"{_ACME}::{ClauseKey.ANTI_ASSIGNMENT.name}"]
    assert len(anti.spans) == 1
    assert anti.has_answer is True
    # 干净样本：跳过数为 0
    clean = _write(_small_sample(), tmp_path, "clean.json")
    clean_gold, clean_skipped = load_gold_with_stats(split="dev", n_contracts=2, seed=7, path=clean)
    assert clean_skipped == 0
    assert len(clean_gold) == 12


# ---- 参数校验 ----


@pytest.mark.parametrize("split", ["train", "DEV", "", "dev "])
def test_invalid_split_raises(tmp_path: Path, split: str) -> None:
    path = _write(_small_sample(), tmp_path, "small.json")
    with pytest.raises(ValueError, match="split"):
        build_queries(split=split, n_contracts=1, seed=1, path=path)
    with pytest.raises(ValueError, match="split"):
        load_gold(split=split, n_contracts=1, seed=1, path=path)


@pytest.mark.parametrize("n_contracts", [0, -1])
def test_invalid_n_contracts_raises(tmp_path: Path, n_contracts: int) -> None:
    path = _write(_small_sample(), tmp_path, "small.json")
    with pytest.raises(ValueError, match="n_contracts"):
        build_queries(split="dev", n_contracts=n_contracts, seed=1, path=path)
    with pytest.raises(ValueError, match="n_contracts"):
        load_gold(split="dev", n_contracts=n_contracts, seed=1, path=path)


def test_n_contracts_exceeding_pool_raises(tmp_path: Path) -> None:
    path = _write(_small_sample(), tmp_path, "small.json")
    with pytest.raises(ValueError, match="n_contracts"):
        build_queries(split="dev", n_contracts=3, seed=1, path=path)  # dev 池只有 2 份


# ---- 分层抽样与 seed 敏感性（大样本） ----


def test_stratified_selection_covers_segments(tmp_path: Path) -> None:
    path = _large_path(tmp_path)

    def selected(split: str, n_contracts: int, seed: int) -> set[str]:
        queries = build_queries(split=split, n_contracts=n_contracts, seed=seed, path=path)
        return {query.contract_id for query in queries}

    short = {"CHARLIE_03_LEASE", "DELTA_04_SERVICES"}
    mid = {"FOXTROT_06_NDA", "GOLF_07_CONSULTING"}
    long_ = {"HOTEL_08_FRANCHISE", "INDIA_09_JOINT_VENTURE"}

    # n=6：dev 全选
    assert selected("dev", 6, 0) == short | mid | long_
    # n=4：ceil(4/3)=2，短、中两段各取 2，长段兜底取 0
    assert selected("dev", 4, 0) == short | mid
    # n=3：三段各取 1
    picks = selected("dev", 3, 0)
    assert len(picks & short) == 1
    assert len(picks & mid) == 1
    assert len(picks & long_) == 1
    # n=1：只取最短段的一份
    assert selected("dev", 1, 0) <= short
    # report 池 3 份：n=2 → 最短两份
    assert selected("report", 2, 0) == {"KILO_11_MARKETING", "OSCAR_15_LEASE"}


def test_seed_changes_selection(tmp_path: Path) -> None:
    path = _large_path(tmp_path)
    seed_0 = {q.contract_id for q in build_queries(split="dev", n_contracts=3, seed=0, path=path)}
    seed_1 = {q.contract_id for q in build_queries(split="dev", n_contracts=3, seed=1, path=path)}
    assert seed_0 != seed_1  # 换 seed 必须改变抽到的合同集合，避免退化


def test_same_params_same_query_id_sequence(tmp_path: Path) -> None:
    path = _large_path(tmp_path)
    first = build_queries(split="dev", n_contracts=5, seed=3, path=path)
    second = build_queries(split="dev", n_contracts=5, seed=3, path=path)
    assert [q.query_id for q in first] == [q.query_id for q in second]
    assert len(first) == 30  # 5 × 6


# ---- 真实数据（38MB），默认不跑 ----


@pytest.mark.slow
def test_real_cuad_end_to_end() -> None:
    """真实文件端到端校验：510×6 展开全量、gold 全部可回原文、抽样确定。"""
    if not DEFAULT_CUAD_PATH.exists():
        pytest.skip(f"真实数据未下载：{DEFAULT_CUAD_PATH}")
    contracts = load_contracts()
    assert len(contracts) == 510  # ADR-0006
    n_dev = sum(1 for cid, _ in contracts if assign_split(cid) == "dev")
    n_report = len(contracts) - n_dev
    assert n_dev > 0 and n_report > 0

    # 两个 split 全量展开：510 × 6 = 3060 条查询（PRD FR-1）
    queries = build_queries(split="dev", n_contracts=n_dev, seed=42) + build_queries(
        split="report", n_contracts=n_report, seed=42
    )
    assert len(queries) == 3060
    assert len({query.query_id for query in queries}) == 3060

    gold_dev, skipped_dev = load_gold_with_stats(split="dev", n_contracts=n_dev, seed=42)
    assert skipped_dev == 0
    gold_report, skipped_report = load_gold_with_stats(
        split="report", n_contracts=n_report, seed=42
    )
    assert skipped_report == 0
    gold = {**gold_dev, **gold_report}
    assert set(gold) == {query.query_id for query in queries}

    # 全部 gold span 可回原文逐字取回（ADR-0006：标注 100% 精确匹配原文）
    by_id = {query.query_id: query for query in queries}
    n_spans = 0
    for query_id, lab in gold.items():
        text = by_id[query_id].contract_text
        for span in lab.spans:
            assert text[span.start : span.end] == span.text
            n_spans += 1
    # 6 类条款的 gold span 总数：量级断言，精确值随数据版本浮动
    assert 2000 <= n_spans <= 3000, f"实际 gold span 总数: {n_spans}"

    # 50 份抽样：同参数两次调用完全一致（含顺序）
    sampled = build_queries(split="report", n_contracts=50, seed=7)
    assert sampled == build_queries(split="report", n_contracts=50, seed=7)
    assert len(sampled) == 300
