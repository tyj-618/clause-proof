"""storage 的契约测试：docs/design/0002-data-model.md §8、tasks/T-005.md 验收标准。"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from clauseproof.models import (
    ClauseKey,
    Judgment,
    Presence,
    Record,
    Route,
    RunMeta,
    Span,
    Status,
)
from clauseproof.storage import append_record, read_meta, read_records, write_meta

_CREATED_AT = "2026-10-27T12:00:00Z"


def ok_record() -> Record:
    """带嵌套 judgment / spans 的完整 OK 记录；浮点刻意带超精度位数。"""
    return Record(
        run_id="run-1",
        query_id="C1::GOVERNING_LAW",
        contract_id="C1",
        clause=ClauseKey.GOVERNING_LAW,
        status=Status.OK,
        judgment=Judgment(
            present=Presence.PRESENT,
            quotes=("Alpha Corp", "not in text"),
            confidence=0.87654,
            raw_output='{"present": "present", "quotes": ["Alpha Corp"], "confidence": 0.87}',
            parse_ok=True,
        ),
        spans=(
            Span(quote="Alpha Corp", start=0, end=10, valid=True, occurrences=2),
            Span(quote="not in text", start=None, end=None, valid=False, occurrences=0),
        ),
        route=Route.AUTO_PASS,
        model="glm-5.3",
        commit="abc123",
        prompt_hash="deadbeef",
        input_tokens=1234,
        output_tokens=56,
        latency_ms=789.012,
        cached=False,
        created_at=_CREATED_AT,
        error_detail=None,
    )


def error_record() -> Record:
    return Record(
        run_id="run-1",
        query_id="C2::GOVERNING_LAW",
        contract_id="C2",
        clause=ClauseKey.GOVERNING_LAW,
        status=Status.LLM_ERROR,
        judgment=None,
        spans=(),
        route=None,
        model="glm-5.3",
        commit="abc123",
        prompt_hash="unknown",
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
        cached=False,
        created_at=_CREATED_AT,
        error_detail="llm: 模型调用失败",
    )


def sample_meta() -> RunMeta:
    return RunMeta(
        run_id="run-1",
        commit="abc123",
        model="glm-5.3",
        config_digest="digest",
        seed=42,
        split="dev",
        n_contracts=3,
        strategy="manual-3",
        started_at=_CREATED_AT,
        n_queries=18,
    )


def _rounded(record: Record) -> Record:
    """按序列化取整规则（confidence 4 位、latency_ms 1 位）得到的期望对象。"""
    return replace(
        record,
        judgment=replace(record.judgment, confidence=round(record.judgment.confidence, 4)),
        latency_ms=round(record.latency_ms, 1),
    )


def test_roundtrip_restores_equal_record(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    record = ok_record()
    append_record(path, record)
    (restored,) = read_records(path)
    assert restored == _rounded(record)


def test_roundtrip_exact_precision_is_identical(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    record = replace(
        ok_record(),
        judgment=replace(ok_record().judgment, confidence=0.5),
        latency_ms=100.5,
    )
    append_record(path, record)
    (restored,) = read_records(path)
    assert restored == record


def test_roundtrip_restores_enums_not_strings(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    append_record(path, ok_record())
    (restored,) = read_records(path)
    assert type(restored.clause) is ClauseKey
    assert restored.clause is ClauseKey.GOVERNING_LAW
    assert type(restored.status) is Status
    assert restored.status is Status.OK
    assert type(restored.judgment.present) is Presence
    assert restored.judgment.present is Presence.PRESENT
    assert type(restored.route) is Route
    assert restored.route is Route.AUTO_PASS


def test_roundtrip_keeps_nested_structure(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    record = ok_record()
    append_record(path, record)
    (restored,) = read_records(path)
    assert len(restored.spans) == 2
    assert restored.spans[0] == record.spans[0]
    assert restored.spans[1] == record.spans[1]
    assert restored.judgment.quotes == record.judgment.quotes
    assert restored.judgment.raw_output == record.judgment.raw_output
    assert restored.judgment.parse_ok is True


def test_error_record_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    record = error_record()
    append_record(path, record)
    (restored,) = read_records(path)
    assert restored == record
    assert restored.judgment is None
    assert restored.spans == ()
    assert restored.route is None


def test_empty_file_returns_empty_iterator(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    path.write_text("", encoding="utf-8")
    assert list(read_records(path)) == []


def test_missing_file_raises_runtime_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="读取记录失败"):
        list(read_records(tmp_path / "nope.jsonl"))


def test_corrupt_line_raises_runtime_error_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    append_record(path, ok_record())
    append_record(path, error_record())
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.insert(1, "{not valid json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"第 2 行.*not valid json|第 2 行") as exc_info:
        list(read_records(path))
    assert "2" in str(exc_info.value)


def test_structurally_invalid_line_raises_runtime_error(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    path.write_text('{"unexpected": 1}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="第 1 行"):
        list(read_records(path))


def test_records_read_back_in_order(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    append_record(path, ok_record())
    append_record(path, error_record())
    records = list(read_records(path))
    assert [r.query_id for r in records] == ["C1::GOVERNING_LAW", "C2::GOVERNING_LAW"]


def test_each_record_is_one_complete_line(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    append_record(path, ok_record())
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("{")
    assert json.loads(lines[0])["run_id"] == "run-1"


def test_field_order_is_fixed(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    append_record(path, ok_record())
    data = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert list(data) == [
        "run_id",
        "query_id",
        "contract_id",
        "clause",
        "status",
        "judgment",
        "spans",
        "route",
        "model",
        "commit",
        "prompt_hash",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "cached",
        "created_at",
        "error_detail",
    ]
    assert list(data["judgment"]) == ["present", "quotes", "confidence", "raw_output", "parse_ok"]
    assert list(data["spans"][0]) == ["quote", "start", "end", "valid", "occurrences"]


def test_meta_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "run-1.meta.json"
    write_meta(path, sample_meta())
    assert read_meta(path) == sample_meta()
    assert list(json.loads(path.read_text(encoding="utf-8"))) == [
        "run_id",
        "commit",
        "model",
        "config_digest",
        "seed",
        "split",
        "n_contracts",
        "strategy",
        "started_at",
        "n_queries",
    ]


def test_read_meta_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "run-1.meta.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(RuntimeError, match="合法 JSON"):
        read_meta(path)


def test_read_meta_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="读取元信息失败"):
        read_meta(tmp_path / "nope.meta.json")


def test_created_at_format_is_iso8601_utc(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    append_record(path, ok_record())
    data = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", data["created_at"])
