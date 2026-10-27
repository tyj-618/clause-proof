"""runner 的契约测试：docs/design/0003-interfaces.md §6、tasks/T-005.md 验收标准。

全部用 FakeJudge（返回固定结果），不发起真实请求。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from clauseproof.judge import CallMeta
from clauseproof.models import (
    ClauseKey,
    Judgment,
    Presence,
    Query,
    Route,
    RunMeta,
    Status,
)
from clauseproof.runner import run
from clauseproof.storage import append_record, read_meta, read_records

_TEXT = "Alpha Corp is governed by the laws of the State of New York."
_ERROR_TEXT = "llm: 模型调用失败（fake）"
_CREATED_AT_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

# 两条 quote：一条在原文中，一条不在，覆盖 valid 与 invalid 两种 Span
_FIXED_JUDGMENT = Judgment(
    present=Presence.PRESENT,
    quotes=("Alpha Corp", "not in text"),
    confidence=0.9,
    raw_output='{"present": "present", "quotes": ["Alpha Corp"], "confidence": 0.9}',
    parse_ok=True,
)


class FakeJudge:
    """固定结果判定器：fail_ids 中的查询抛 RuntimeError，其余返回固定 Judgment。

    calls 记录实际调用过的 query_id，用于断言「不重复调用 judge」。
    """

    def __init__(self, *, fail_ids: frozenset[str] = frozenset()) -> None:
        self.fail_ids = fail_ids
        self.calls: list[str] = []

    def judge(self, query: Query) -> tuple[Judgment, CallMeta]:
        self.calls.append(query.query_id)
        if query.query_id in self.fail_ids:
            raise RuntimeError(_ERROR_TEXT)
        meta = CallMeta(
            model="fake-model",
            prompt_hash=f"hash-{query.query_id}",
            input_tokens=100,
            output_tokens=20,
            latency_ms=3.5,
            cached=False,
        )
        return _FIXED_JUDGMENT, meta


def make_queries(n: int) -> tuple[Query, ...]:
    return tuple(
        Query(
            query_id=f"C{i}::GOVERNING_LAW",
            contract_id=f"C{i}",
            clause=ClauseKey.GOVERNING_LAW,
            contract_text=_TEXT,
        )
        for i in range(n)
    )


def make_meta(n_queries: int) -> RunMeta:
    return RunMeta(
        run_id="t-run",
        commit="deadbeef",
        model="fake-model",
        config_digest="digest",
        seed=42,
        split="dev",
        n_contracts=1,
        strategy="test",
        started_at="2026-10-27T00:00:00Z",
        n_queries=n_queries,
    )


def _run(
    tmp_path: Path,
    *,
    queries,
    judge,
    thresholds: dict[ClauseKey, float] | None = None,
    resume=True,
    max_failures=5,
):
    if thresholds is None:
        thresholds = {c: 0.8 for c in ClauseKey}
    return run(
        queries=queries,
        judge=judge,
        run_meta=make_meta(len(queries)),
        thresholds=thresholds,
        out_dir=tmp_path,
        resume=resume,
        max_consecutive_failures=max_failures,
    )


def test_produces_n_records_and_both_files(tmp_path: Path) -> None:
    queries = make_queries(3)
    path = _run(tmp_path, queries=queries, judge=FakeJudge())
    assert path == tmp_path / "t-run.jsonl"
    records = list(read_records(path))
    assert len(records) == len(queries) == 3
    meta_path = tmp_path / "t-run.meta.json"
    assert meta_path.is_file()
    assert read_meta(meta_path) == make_meta(3)


def test_record_carries_run_meta_and_call_meta_fields(tmp_path: Path) -> None:
    queries = make_queries(2)
    path = _run(tmp_path, queries=queries, judge=FakeJudge())
    for record in read_records(path):
        assert record.run_id == "t-run"
        assert record.model == "fake-model"  # 来自 CallMeta
        assert record.commit == "deadbeef"  # 来自 RunMeta
        assert record.prompt_hash == f"hash-{record.query_id}"  # 来自 CallMeta
        assert _CREATED_AT_RE.fullmatch(record.created_at)
        assert record.input_tokens == 100
        assert record.output_tokens == 20
        assert record.latency_ms == 3.5
        assert record.cached is False


def test_every_quote_located_spans_match_quotes(tmp_path: Path) -> None:
    queries = make_queries(1)
    path = _run(tmp_path, queries=queries, judge=FakeJudge())
    (record,) = read_records(path)
    assert len(record.spans) == len(record.judgment.quotes) == 2
    assert record.spans[0].valid is True
    assert _TEXT[record.spans[0].start : record.spans[0].end] == "Alpha Corp"
    assert record.spans[1].valid is False
    assert record.spans[1].occurrences == 0


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.8, Route.AUTO_PASS), (0.95, Route.TO_HUMAN)],
)
def test_route_computed_by_apply_route(tmp_path: Path, threshold: float, expected: Route) -> None:
    queries = make_queries(1)
    path = _run(
        tmp_path, queries=queries, judge=FakeJudge(), thresholds={c: threshold for c in ClauseKey}
    )
    (record,) = read_records(path)
    assert record.route is expected


def test_missing_clause_key_in_thresholds_raises_value_error(tmp_path: Path) -> None:
    queries = make_queries(1)
    thresholds = {c: 0.8 for c in ClauseKey if c is not ClauseKey.UNCAP_LIABILITY}
    with pytest.raises(
        ValueError,
        match="thresholds 违反 必须覆盖全部条款类别，期望 6 个键，实际 缺失 UNCAP_LIABILITY",
    ):
        _run(tmp_path, queries=queries, judge=FakeJudge(), thresholds=thresholds)


def test_multiple_missing_clause_keys_listed_in_order(tmp_path: Path) -> None:
    queries = make_queries(1)
    excluded = (ClauseKey.GOVERNING_LAW, ClauseKey.NOTICE_PERIOD_TO_TERMINATE_RENEWAL)
    thresholds = {c: 0.8 for c in ClauseKey if c not in excluded}
    with pytest.raises(ValueError) as exc_info:
        _run(tmp_path, queries=queries, judge=FakeJudge(), thresholds=thresholds)
    # 缺失类别按 ClauseKey 定义序列出
    message = str(exc_info.value)
    assert "实际 缺失 GOVERNING_LAW, NOTICE_PERIOD_TO_TERMINATE_RENEWAL" in message


def test_per_clause_thresholds_drive_route_independently(tmp_path: Path) -> None:
    # FakeJudge 固定返回 confidence=0.9：A 类阈值 0.99 → TO_HUMAN，B 类阈值 0.1 → AUTO_PASS
    queries = (
        Query(
            query_id="C0::GOVERNING_LAW",
            contract_id="C0",
            clause=ClauseKey.GOVERNING_LAW,
            contract_text=_TEXT,
        ),
        Query(
            query_id="C0::CAP_ON_LIABILITY",
            contract_id="C0",
            clause=ClauseKey.CAP_ON_LIABILITY,
            contract_text=_TEXT,
        ),
    )
    thresholds = dict.fromkeys(ClauseKey, 0.5)
    thresholds[ClauseKey.GOVERNING_LAW] = 0.99
    thresholds[ClauseKey.CAP_ON_LIABILITY] = 0.1
    path = _run(tmp_path, queries=queries, judge=FakeJudge(), thresholds=thresholds)
    by_clause = {record.clause: record for record in read_records(path)}
    assert by_clause[ClauseKey.GOVERNING_LAW].route is Route.TO_HUMAN
    assert by_clause[ClauseKey.CAP_ON_LIABILITY].route is Route.AUTO_PASS


def test_judge_error_recorded_and_rest_continues(tmp_path: Path) -> None:
    queries = make_queries(3)
    failing = "C1::GOVERNING_LAW"
    judge = FakeJudge(fail_ids=frozenset({failing}))
    path = _run(tmp_path, queries=queries, judge=judge)
    records = list(read_records(path))
    assert len(records) == 3
    assert judge.calls == [q.query_id for q in queries]  # 失败后继续处理
    failed = next(r for r in records if r.query_id == failing)
    assert failed.status is Status.LLM_ERROR
    assert _ERROR_TEXT in failed.error_detail
    assert failed.judgment is None
    assert failed.spans == ()
    assert failed.route is None
    assert failed.input_tokens == 0
    for record in records:
        if record.query_id != failing:
            assert record.status is Status.OK


def test_circuit_breaker_stops_and_written_records_readable(tmp_path: Path) -> None:
    queries = make_queries(5)
    fail_ids = frozenset(f"C{i}::GOVERNING_LAW" for i in range(3))
    judge = FakeJudge(fail_ids=fail_ids)
    with pytest.raises(RuntimeError, match="熔断") as exc_info:
        _run(tmp_path, queries=queries, judge=judge, max_failures=3)
    assert "3" in str(exc_info.value)
    # 熔断即停：后面的查询不再调用 judge
    assert judge.calls == [f"C{i}::GOVERNING_LAW" for i in range(3)]
    # 已写入的 3 条失败记录完整可读
    records = list(read_records(tmp_path / "t-run.jsonl"))
    assert len(records) == 3
    assert all(r.status is Status.LLM_ERROR for r in records)


def test_success_resets_consecutive_failure_count(tmp_path: Path) -> None:
    # 失败、成功、失败、失败：成功归零后最多连续 2 条，不触发熔断（阈值 3）
    queries = make_queries(4)
    fail_ids = frozenset({"C0::GOVERNING_LAW", "C2::GOVERNING_LAW", "C3::GOVERNING_LAW"})
    path = _run(tmp_path, queries=queries, judge=FakeJudge(fail_ids=fail_ids), max_failures=3)
    records = list(read_records(path))
    assert len(records) == 4
    assert [r.status for r in records] == [
        Status.LLM_ERROR,
        Status.OK,
        Status.LLM_ERROR,
        Status.LLM_ERROR,
    ]


def test_circuit_breaker_after_reset_needs_full_streak(tmp_path: Path) -> None:
    # 失败、成功、失败×3：归零后重新累计到 3，最后一条触发熔断并已写盘
    queries = make_queries(5)
    fail_ids = frozenset(f"C{i}::GOVERNING_LAW" for i in (0, 2, 3, 4))
    judge = FakeJudge(fail_ids=fail_ids)
    with pytest.raises(RuntimeError, match="熔断"):
        _run(tmp_path, queries=queries, judge=judge, max_failures=3)
    records = list(read_records(tmp_path / "t-run.jsonl"))
    assert [r.status for r in records] == [
        Status.LLM_ERROR,
        Status.OK,
        Status.LLM_ERROR,
        Status.LLM_ERROR,
        Status.LLM_ERROR,
    ]


def test_resume_skips_existing_query_ids(tmp_path: Path) -> None:
    queries = make_queries(5)
    jsonl_path = tmp_path / "t-run.jsonl"
    # 预写前两条（模拟上次运行中断）
    for i in range(2):
        append_record(
            jsonl_path,
            _ok_record_for(f"C{i}::GOVERNING_LAW"),
        )
    judge = FakeJudge()
    path = _run(tmp_path, queries=queries, judge=judge, resume=True)
    assert path == jsonl_path
    assert judge.calls == [f"C{i}::GOVERNING_LAW" for i in (2, 3, 4)]
    records = list(read_records(jsonl_path))
    assert len(records) == 5  # 最终记录数等于查询数
    assert [r.query_id for r in records] == [q.query_id for q in queries]


def test_resume_full_file_makes_no_calls(tmp_path: Path) -> None:
    queries = make_queries(3)
    _run(tmp_path, queries=queries, judge=FakeJudge())
    before = (tmp_path / "t-run.jsonl").read_text(encoding="utf-8")
    second = FakeJudge()
    _run(tmp_path, queries=queries, judge=second, resume=True)
    assert second.calls == []  # 不重复调用 judge
    assert (tmp_path / "t-run.jsonl").read_text(encoding="utf-8") == before  # 文件未被追加


def test_no_resume_starts_fresh(tmp_path: Path) -> None:
    queries = make_queries(3)
    jsonl_path = tmp_path / "t-run.jsonl"
    # 预写一条旧记录（含本次查询集之外的 query_id）
    append_record(jsonl_path, _ok_record_for("OLDCONTRACT::GOVERNING_LAW"))
    judge = FakeJudge()
    path = _run(tmp_path, queries=queries, judge=judge, resume=False)
    assert path == jsonl_path
    assert len(judge.calls) == 3  # 全部重新调用
    records = list(read_records(jsonl_path))
    assert [r.query_id for r in records] == [q.query_id for q in queries]
    assert all(r.query_id != "OLDCONTRACT::GOVERNING_LAW" for r in records)


def test_meta_written_before_processing(tmp_path: Path) -> None:
    # 第一条就熔断时 meta 仍已写盘（运行登记先于任何记录）
    queries = make_queries(3)
    judge = FakeJudge(fail_ids=frozenset(f"C{i}::GOVERNING_LAW" for i in range(3)))
    with pytest.raises(RuntimeError):
        _run(tmp_path, queries=queries, judge=judge, max_failures=1)
    assert read_meta(tmp_path / "t-run.meta.json") == make_meta(3)


def _ok_record_for(query_id: str):
    """构造一条与 runner 产出同构的 OK 记录，用于预写 JSONL 模拟断点。"""
    from clauseproof.models import Record, Span

    return Record(
        run_id="t-run",
        query_id=query_id,
        contract_id=query_id.split("::", 1)[0],
        clause=ClauseKey.GOVERNING_LAW,
        status=Status.OK,
        judgment=_FIXED_JUDGMENT,
        spans=(
            Span(quote="Alpha Corp", start=0, end=10, valid=True, occurrences=1),
            Span(quote="not in text", start=None, end=None, valid=False, occurrences=0),
        ),
        route=Route.AUTO_PASS,
        model="fake-model",
        commit="deadbeef",
        prompt_hash=f"hash-{query_id}",
        input_tokens=100,
        output_tokens=20,
        latency_ms=3.5,
        cached=False,
        created_at="2026-10-27T00:00:00Z",
        error_detail=None,
    )
