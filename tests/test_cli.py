"""CLI 的契约测试：tasks/T-009.md 验收标准。

全部通过 monkeypatch 注入假配置、假 judge 与迷你 CUAD，不发起真实请求、
不读真实数据集、不依赖 .env。eval → report 的曲线用例为手工推导的期望值。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from clauseproof import cli, dataset
from clauseproof.cli import app
from clauseproof.config import LLMConfig
from clauseproof.dataset import assign_split
from clauseproof.judge import CallMeta
from clauseproof.models import (
    ClauseKey,
    Judgment,
    Presence,
    Query,
    Record,
    Route,
    Span,
    Status,
)
from clauseproof.storage import append_record, read_meta, read_records

cli_runner = CliRunner()

_TEXT = (
    "Alpha Corp is governed by the laws of the State of New York. "
    "This Agreement may not be assigned without consent."
)
_GL_QUOTE = "governed by the laws of the State of New York"
_AA_QUOTE = "may not be assigned without consent"
_ERROR_TEXT = "llm: 模型调用失败（fake）"
_CURVE_ROW_RE = re.compile(r"^\s*(\d+\.\d+)\s+(\d+\.\d+)\s+(\S+)\s+(\d+\.\d+)$")

_FAKE_CONFIG = LLMConfig(
    base_url="https://fake.example",
    api_key="fake-key",
    model="glm-5.3",  # 已定价的模型，使成本断言可用真实单价
    max_tokens=8192,
    reasoning_effort="low",
    temperature=0.0,
    timeout_s=180.0,
    cache_dir=Path("/tmp/clauseproof-test-cache"),
    protocol="openai",
)


class FakeJudge:
    """固定结果判定器：judgments 按条款给固定 Judgment，fail_ids 抛 RuntimeError。"""

    def __init__(
        self,
        judgments: dict[ClauseKey, Judgment],
        *,
        fail_ids: frozenset[str] = frozenset(),
        input_tokens: int = 7152,
        output_tokens: int = 94,
        latency_ms: float = 6600.0,
    ) -> None:
        self.judgments = judgments
        self.fail_ids = fail_ids
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms

    def judge(self, query: Query) -> tuple[Judgment, CallMeta]:
        if query.query_id in self.fail_ids:
            raise RuntimeError(_ERROR_TEXT)
        return self.judgments[query.clause], CallMeta(
            model="glm-5.3",
            prompt_hash=f"hash-{query.query_id}",
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=self.latency_ms,
            cached=False,
        )


def _present(confidence: float, quote: str = _GL_QUOTE) -> Judgment:
    return Judgment(
        present=Presence.PRESENT,
        quotes=(quote,),
        confidence=confidence,
        raw_output='{"present": "present"}',
        parse_ok=True,
    )


def _absent(confidence: float) -> Judgment:
    return Judgment(
        present=Presence.ABSENT,
        quotes=(),
        confidence=confidence,
        raw_output='{"present": "absent"}',
        parse_ok=True,
    )


def _uncertain(*, parse_ok: bool = True, confidence: float = 0.5) -> Judgment:
    return Judgment(
        present=Presence.UNCERTAIN,
        quotes=(),
        # 解析失败的置信度按模型不变式必须为 0.0
        confidence=0.0 if not parse_ok else confidence,
        raw_output="...",
        parse_ok=parse_ok,
    )


def _uniform(judgment: Judgment) -> dict[ClauseKey, Judgment]:
    return dict.fromkeys(ClauseKey, judgment)


def _install(monkeypatch: pytest.MonkeyPatch, judgments, **kwargs) -> FakeJudge:
    """注入假配置与假 judge，屏蔽 .env 与真实模型（任务书工程项）。"""
    monkeypatch.setattr(cli, "load_config", lambda: _FAKE_CONFIG)
    monkeypatch.setattr(cli, "_git_commit", lambda: "deadbeef")
    judge = FakeJudge(judgments, **kwargs)
    monkeypatch.setattr(cli, "_build_judge", lambda config: judge)
    return judge


# 迷你 CUAD：gold 只给 Governing Law / Anti-Assignment 答案（有 gold 文本的两类）
_GOLD_BY_CLAUSE: dict[ClauseKey, str] = {
    ClauseKey.GOVERNING_LAW: _GL_QUOTE,
    ClauseKey.ANTI_ASSIGNMENT: _AA_QUOTE,
}


def _dev_titles(n: int) -> list[str]:
    """确定性地挑出 hash 到 dev split 的标题（assign_split 的 60% 侧）。"""
    titles: list[str] = []
    index = 0
    while len(titles) < n:
        title = f"CONTRACT-{index:04d}"
        if assign_split(title) == "dev":
            titles.append(title)
        index += 1
    return titles


def _install_cuad(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    n_contracts: int = 1,
    answered: frozenset[ClauseKey] | None = None,
) -> None:
    """写入迷你 CUAD 并指为默认数据集路径。"""
    answered = frozenset(_GOLD_BY_CLAUSE) if answered is None else answered
    data = []
    for title in _dev_titles(n_contracts):
        qas = []
        for clause in ClauseKey:
            answers = []
            if clause in answered:
                gold_text = _GOLD_BY_CLAUSE[clause]
                answers = [{"text": gold_text, "answer_start": _TEXT.find(gold_text)}]
            qas.append(
                {
                    "question": f'Does the contract contain a clause related to "{clause.value}"?',
                    "answers": answers,
                }
            )
        data.append({"title": title, "paragraphs": [{"context": _TEXT, "qas": qas}]})
    path = tmp_path / "cuad.json"
    path.write_text(json.dumps({"data": data}), encoding="utf-8")
    monkeypatch.setattr(dataset, "DEFAULT_CUAD_PATH", path)


def _eval_args(out_dir: Path, run_id: str = "t-run", **overrides) -> list[str]:
    args = [
        "eval",
        "--split",
        "dev",
        "--n-contracts",
        "1",
        "--seed",
        "42",
        "--run-id",
        run_id,
        "--out-dir",
        str(out_dir),
    ]
    for key, value in overrides.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    return args


def _run_eval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    judgments: dict[ClauseKey, Judgment],
    run_id: str = "t-run",
) -> Path:
    """跑一次假 eval 并返回其输出目录。"""
    _install_cuad(monkeypatch, tmp_path)
    _install(monkeypatch, judgments)
    out = tmp_path / "out"
    result = cli_runner.invoke(app, _eval_args(out, run_id=run_id))
    assert result.exit_code == 0
    return out


def _record_for(query_id: str) -> Record:
    """构造一条与 runner 产出同构的 OK 记录，用于追加异常数据。"""
    start = _TEXT.find(_GL_QUOTE)
    return Record(
        run_id="t-run",
        query_id=query_id,
        contract_id=query_id.split("::", 1)[0],
        clause=ClauseKey.GOVERNING_LAW,
        status=Status.OK,
        judgment=_present(0.9),
        spans=(
            Span(
                quote=_GL_QUOTE,
                start=start,
                end=start + len(_GL_QUOTE),
                valid=True,
                occurrences=1,
                matched="exact",
            ),
        ),
        route=Route.AUTO_PASS,
        model="glm-5.3",
        commit="deadbeef",
        prompt_hash="hash",
        input_tokens=100,
        output_tokens=20,
        latency_ms=1.0,
        cached=False,
        created_at="2026-10-29T00:00:00Z",
        error_detail=None,
    )


# ---------- 通用 ----------


def test_help_lists_three_subcommands() -> None:
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("ask", "eval", "report"):
        assert name in result.output


@pytest.mark.parametrize("command", ["ask", "eval", "report"])
def test_subcommand_help_contains_example(command: str) -> None:
    result = cli_runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert "示例" in result.output
    assert f"clauseproof {command}" in result.output


def test_no_args_prints_usage_and_exits_zero() -> None:
    result = cli_runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output
    assert result.exception is None


def test_unknown_subcommand_is_nonzero_with_readable_error() -> None:
    result = cli_runner.invoke(app, ["nope"])
    assert result.exit_code != 0
    assert "No such command" in result.output


# ---------- ask ----------


def test_ask_requires_exactly_one_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = tmp_path / "contract.txt"
    contract.write_text(_TEXT, encoding="utf-8")
    _install(monkeypatch, _uniform(_present(0.9)))
    neither = cli_runner.invoke(app, ["ask", "--clause", "Governing Law"])
    assert neither.exit_code == 1
    assert "二选一" in neither.output
    both = cli_runner.invoke(
        app,
        ["ask", "--contract-id", "X", "--file", str(contract), "--clause", "Governing Law"],
    )
    assert both.exit_code == 1
    assert "二选一" in both.output


def test_ask_invalid_clause_lists_legal_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = tmp_path / "contract.txt"
    contract.write_text(_TEXT, encoding="utf-8")
    _install(monkeypatch, _uniform(_present(0.9)))
    result = cli_runner.invoke(app, ["ask", "--file", str(contract), "--clause", "Bogus"])
    assert result.exit_code == 1
    assert "合法取值" in result.output
    for clause in ClauseKey:  # 列出全部合法取值
        assert clause.value in result.output
    assert "管辖法律" in result.output  # 中文名也在列


def test_ask_accepts_english_and_chinese_clause_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = tmp_path / "contract.txt"
    contract.write_text(_TEXT, encoding="utf-8")
    _install(monkeypatch, _uniform(_present(0.95)))
    english = cli_runner.invoke(app, ["ask", "--file", str(contract), "--clause", "governing law"])
    assert english.exit_code == 0
    assert "条款   Governing Law" in english.output
    chinese = cli_runner.invoke(app, ["ask", "--file", str(contract), "--clause", "管辖法律"])
    assert chinese.exit_code == 0
    assert "条款   Governing Law" in chinese.output


def test_ask_output_contains_all_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = tmp_path / "contract.txt"
    contract.write_text(_TEXT, encoding="utf-8")
    _install(monkeypatch, _uniform(_present(0.95)))
    result = cli_runner.invoke(
        app, ["ask", "--file", str(contract), "--clause", "Cap On Liability"]
    )
    assert result.exit_code == 0
    out = result.output
    assert "判定   有   confidence 0.950" in out
    assert "路由   自动通过" in out
    assert "引用（1 处，已校验）" in out
    start = _TEXT.find(_GL_QUOTE)
    assert f"[{start}:{start + len(_GL_QUOTE)}]" in out  # 引用区间
    assert _GL_QUOTE in out  # 引用文本
    assert "输入 7152 tokens" in out
    assert "输出 94 tokens" in out
    assert "耗时 6.6s" in out
    assert "成本 ¥0.060" in out  # (7152*8 + 94*28) / 1e6


def test_ask_line_count_le_40_with_many_quotes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = tmp_path / "contract.txt"
    contract.write_text(_TEXT, encoding="utf-8")
    judgment = Judgment(
        present=Presence.PRESENT,
        quotes=("Alpha Corp",) * 45,  # 45 处引用，超出展示上限
        confidence=0.9,
        raw_output="...",
        parse_ok=True,
    )
    _install(monkeypatch, _uniform(judgment))
    result = cli_runner.invoke(app, ["ask", "--file", str(contract), "--clause", "Governing Law"])
    assert result.exit_code == 0
    assert len(result.output.splitlines()) <= 40
    assert "……另有 15 处引用未显示" in result.output


def test_ask_truncates_long_quotes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    long_quote = "L" * 250
    contract = tmp_path / "contract.txt"
    contract.write_text(f"prefix {long_quote} suffix", encoding="utf-8")
    _install(monkeypatch, _uniform(_present(0.9, quote=long_quote)))
    result = cli_runner.invoke(app, ["ask", "--file", str(contract), "--clause", "Governing Law"])
    assert result.exit_code == 0
    assert "…（共 250 字符）" in result.output
    assert long_quote not in result.output  # 不整段输出


def test_ask_invalid_citation_forces_human_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = tmp_path / "contract.txt"
    contract.write_text(_TEXT, encoding="utf-8")
    _install(monkeypatch, _uniform(_present(0.95, quote="NOT IN TEXT")))
    result = cli_runner.invoke(app, ["ask", "--file", str(contract), "--clause", "Governing Law"])
    assert result.exit_code == 0
    assert "引用无效：该片段未在原文中找到" in result.output
    assert "路由   转人工" in result.output
    assert "引用（1 处，定位失败）" in result.output


def test_ask_absent_shows_no_citations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = tmp_path / "contract.txt"
    contract.write_text(_TEXT, encoding="utf-8")
    _install(monkeypatch, _uniform(_absent(0.9)))
    result = cli_runner.invoke(app, ["ask", "--file", str(contract), "--clause", "Governing Law"])
    assert result.exit_code == 0
    assert "判定   没有" in result.output
    assert "引用（无）" in result.output


def test_ask_parse_failure_marks_uncertain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = tmp_path / "contract.txt"
    contract.write_text(_TEXT, encoding="utf-8")
    _install(monkeypatch, _uniform(_uncertain(parse_ok=False)))
    result = cli_runner.invoke(app, ["ask", "--file", str(contract), "--clause", "Governing Law"])
    assert result.exit_code == 0
    assert "判定   模糊（模型输出解析失败）   confidence 0.000" in result.output
    assert "路由   转人工" in result.output


def test_ask_missing_file_exit_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install(monkeypatch, _uniform(_present(0.9)))
    result = cli_runner.invoke(
        app, ["ask", "--file", str(tmp_path / "nope.txt"), "--clause", "Governing Law"]
    )
    assert result.exit_code == 1
    assert "--file" in result.output


def test_ask_empty_file_exit_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = tmp_path / "empty.txt"
    contract.write_text("", encoding="utf-8")
    _install(monkeypatch, _uniform(_present(0.9)))
    result = cli_runner.invoke(app, ["ask", "--file", str(contract), "--clause", "Governing Law"])
    assert result.exit_code == 1
    assert "非空" in result.output


def test_ask_contract_id_loaded_from_cuad(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_cuad(monkeypatch, tmp_path)
    title = _dev_titles(1)[0]
    _install(monkeypatch, _uniform(_present(0.9)))
    result = cli_runner.invoke(app, ["ask", "--contract-id", title, "--clause", "Governing Law"])
    assert result.exit_code == 0
    assert f"合同   {title}" in result.output
    assert _GL_QUOTE in result.output  # 判的是迷你 CUAD 里的 _TEXT


def test_ask_unknown_contract_id_exit_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_cuad(monkeypatch, tmp_path)
    _install(monkeypatch, _uniform(_present(0.9)))
    result = cli_runner.invoke(
        app, ["ask", "--contract-id", "NO SUCH CONTRACT", "--clause", "Governing Law"]
    )
    assert result.exit_code == 1
    assert "--contract-id" in result.output


def test_ask_judge_failure_exit_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = tmp_path / "contract.txt"
    contract.write_text(_TEXT, encoding="utf-8")
    query_id = f"{contract}::GOVERNING_LAW"
    _install(monkeypatch, _uniform(_present(0.9)), fail_ids=frozenset({query_id}))
    result = cli_runner.invoke(app, ["ask", "--file", str(contract), "--clause", "Governing Law"])
    assert result.exit_code == 2
    assert _ERROR_TEXT in result.output


def test_ask_config_failure_exit_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = tmp_path / "contract.txt"
    contract.write_text(_TEXT, encoding="utf-8")

    def _boom() -> LLMConfig:
        raise RuntimeError("config: ANTHROPIC_AUTH_TOKEN 违反 必填非空，期望 非空，实际 空")

    monkeypatch.setattr(cli, "load_config", _boom)
    result = cli_runner.invoke(app, ["ask", "--file", str(contract), "--clause", "Governing Law"])
    assert result.exit_code == 2
    assert "ANTHROPIC_AUTH_TOKEN" in result.output


# ---------- eval ----------


def test_eval_produces_files_and_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_cuad(monkeypatch, tmp_path)
    _install(monkeypatch, _uniform(_present(0.9)))
    out = tmp_path / "out"
    result = cli_runner.invoke(app, _eval_args(out))
    assert result.exit_code == 0
    jsonl = out / "t-run.jsonl"
    meta_file = out / "t-run.meta.json"
    assert jsonl.is_file() and meta_file.is_file()
    records = list(read_records(jsonl))
    assert len(records) == 6  # 1 合同 × 6 条款
    meta = read_meta(meta_file)
    assert meta.run_id == "t-run"
    assert meta.commit == "deadbeef"
    assert meta.model == "glm-5.3"
    assert (meta.split, meta.n_contracts, meta.seed, meta.n_queries) == ("dev", 1, 42, 6)
    assert meta.strategy == "length_stratified"
    assert meta.config_digest
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", meta.started_at)
    out_text = result.output
    assert "运行 t-run 完成" in out_text
    assert "6 条 · OK 6 · 失败 0 · 模糊 0" in out_text
    assert "成本 ¥0.359" in out_text  # 6 × (7152*8 + 94*28) / 1e6
    assert "平均 6.6s/条" in out_text
    assert "耗时" in out_text
    assert f"JSONL  {jsonl}" in out_text
    assert f"meta   {meta_file}" in out_text


def test_eval_single_failure_does_not_fail_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_cuad(monkeypatch, tmp_path)
    title = _dev_titles(1)[0]
    fail_ids = frozenset({f"{title}::CAP_ON_LIABILITY"})
    _install(monkeypatch, _uniform(_present(0.9)), fail_ids=fail_ids)
    out = tmp_path / "out"
    result = cli_runner.invoke(app, _eval_args(out))
    assert result.exit_code == 0
    assert "6 条 · OK 5 · 失败 1 · 模糊 0" in result.output
    assert len(list(read_records(out / "t-run.jsonl"))) == 6


def test_eval_parse_failures_counted_in_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_cuad(monkeypatch, tmp_path)
    judgments = _uniform(_present(0.9))
    judgments[ClauseKey.NOTICE_PERIOD_TO_TERMINATE_RENEWAL] = _uncertain(parse_ok=False)
    _install(monkeypatch, judgments)
    out = tmp_path / "out"
    result = cli_runner.invoke(app, _eval_args(out))
    assert result.exit_code == 0
    assert "模糊 1（含解析失败 1）" in result.output


def test_eval_circuit_breaker_exit_2_with_readable_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_cuad(monkeypatch, tmp_path)
    title = _dev_titles(1)[0]
    fail_ids = frozenset(f"{title}::{clause.name}" for clause in ClauseKey)
    _install(monkeypatch, _uniform(_present(0.9)), fail_ids=fail_ids)
    out = tmp_path / "out"
    result = cli_runner.invoke(app, _eval_args(out, max_consecutive_failures=2))
    assert result.exit_code == 2
    assert "熔断" in result.output
    records = list(read_records(out / "t-run.jsonl"))
    assert len(records) == 2  # 已写入的记录完整可读
    assert all(record.status is Status.LLM_ERROR for record in records)


def test_eval_invalid_threshold_exit_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_cuad(monkeypatch, tmp_path)
    _install(monkeypatch, _uniform(_present(0.9)))
    result = cli_runner.invoke(app, _eval_args(tmp_path / "out", threshold=1.5))
    assert result.exit_code == 1
    assert "--threshold" in result.output


def test_eval_invalid_split_exit_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_cuad(monkeypatch, tmp_path)
    _install(monkeypatch, _uniform(_present(0.9)))
    args = [
        "eval",
        "--split",
        "bogus",
        "--n-contracts",
        "1",
        "--seed",
        "42",
        "--run-id",
        "t",
        "--out-dir",
        str(tmp_path),
    ]
    result = cli_runner.invoke(app, args)
    assert result.exit_code == 1
    assert "dev或report" in result.output


def test_git_commit_returns_repo_head() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", cli._git_commit()) is not None


def test_git_commit_unknown_outside_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli._git_commit() == "unknown"


# ---------- report ----------


def test_report_missing_run_file_exit_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_cuad(monkeypatch, tmp_path)
    result = cli_runner.invoke(app, ["report", "--run", str(tmp_path / "nope.jsonl")])
    assert result.exit_code == 1
    assert "--run" in result.output


def test_report_missing_meta_exit_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = _run_eval(monkeypatch, tmp_path, _uniform(_present(0.9)))
    (out / "t-run.meta.json").unlink()
    result = cli_runner.invoke(app, ["report", "--run", str(out / "t-run.jsonl")])
    assert result.exit_code == 1
    assert "meta" in result.output


def test_report_outputs_metrics_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = _run_eval(monkeypatch, tmp_path, _uniform(_present(0.9)))
    result = cli_runner.invoke(app, ["report", "--run", str(out / "t-run.jsonl")])
    assert result.exit_code == 0
    out_text = result.output
    for header in ("n_total", "n_decided", "n_correct", "accuracy", "引用准确率", "ECE"):
        assert header in out_text
    for clause in ClauseKey:
        assert clause.value in out_text
    # gold：Governing Law / Anti-Assignment 有答案，其余无 → 全判 PRESENT 时
    # 前两类正确（1.000），后四类误检（0.000）
    gl_line = next(line for line in out_text.splitlines() if line.startswith("Governing Law"))
    assert "1.000" in gl_line
    cap_line = next(line for line in out_text.splitlines() if line.startswith("Cap On Liability"))
    assert "0.000" in cap_line
    assert "宏平均 accuracy 0.333" in out_text  # (1 + 1 + 0×4) / 6
    assert "模糊率 0.0%（0/6）" in out_text
    assert "全局 ECE" in out_text


def test_report_records_not_in_gold_exit_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = _run_eval(monkeypatch, tmp_path, _uniform(_present(0.9)))
    append_record(out / "t-run.jsonl", _record_for("BOGUS::GOVERNING_LAW"))
    result = cli_runner.invoke(app, ["report", "--run", str(out / "t-run.jsonl")])
    assert result.exit_code == 1
    assert "gold" in result.output


def test_report_curve_requires_thresholds_from(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = _run_eval(monkeypatch, tmp_path, _uniform(_present(0.9)))
    result = cli_runner.invoke(app, ["report", "--run", str(out / "t-run.jsonl"), "--curve"])
    assert result.exit_code == 1
    assert "阈值必须在开发集上选择" in result.output


def test_report_thresholds_from_requires_curve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = _run_eval(monkeypatch, tmp_path, _uniform(_present(0.9)))
    result = cli_runner.invoke(
        app,
        [
            "report",
            "--run",
            str(out / "t-run.jsonl"),
            "--thresholds-from",
            str(out / "t-run.jsonl"),
        ],
    )
    assert result.exit_code == 1
    assert "--curve" in result.output


def test_report_review_rates_require_curve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = _run_eval(monkeypatch, tmp_path, _uniform(_present(0.9)))
    result = cli_runner.invoke(
        app, ["report", "--run", str(out / "t-run.jsonl"), "--review-rates", "0.5"]
    )
    assert result.exit_code == 1


def test_report_curve_matches_hand_computed_points(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """手算曲线用例：6 条记录（gold 仅 GL / ANTI 有答案），两批不同置信度。

    来源运行在 0.2 / 1.0 两个目标复核率下的贪心（route.select_thresholds；
    T-013 后 target_review_rate 为上界、零收益即停）：
    - 0.2：来源唯一收益提升（UNCAP 提到 1.0 翻出 0.95 判错记录）需复核率
      2/6 = 0.333，超出 0.2 预算 → 阈值全 0；评测侧仅 UNCERTAIN 转人工，
      实际复核率 1/6 = 0.167，自动通过 5 条中 UNCAP 错 → 0.800 / 0.833
    - 1.0：来源只做收益提升（UNCAP 提到 1.0）后零收益停止 → 评测侧
      UNCAP(0.60) 与 UNCERTAIN 转人工，实际复核率 2/6 = 0.333，
      自动通过 4 条全对 → 0.333 / 1.000 / 1.000
    """
    run_judgments = {
        ClauseKey.GOVERNING_LAW: _present(0.95),
        ClauseKey.CAP_ON_LIABILITY: _absent(0.90),
        ClauseKey.UNCAP_LIABILITY: _present(0.60),
        ClauseKey.ANTI_ASSIGNMENT: _present(0.95, quote=_AA_QUOTE),
        ClauseKey.TERMINATION_FOR_CONVENIENCE: _absent(0.70),
        ClauseKey.NOTICE_PERIOD_TO_TERMINATE_RENEWAL: _uncertain(),
    }
    source_judgments = {
        ClauseKey.GOVERNING_LAW: _present(0.99),
        ClauseKey.CAP_ON_LIABILITY: _absent(0.80),
        ClauseKey.UNCAP_LIABILITY: _present(0.95),
        ClauseKey.ANTI_ASSIGNMENT: _present(0.50, quote=_AA_QUOTE),
        ClauseKey.TERMINATION_FOR_CONVENIENCE: _absent(0.85),
        ClauseKey.NOTICE_PERIOD_TO_TERMINATE_RENEWAL: _uncertain(),
    }
    out = _run_eval(monkeypatch, tmp_path, run_judgments, run_id="curve-run")
    _install(monkeypatch, source_judgments)
    result = cli_runner.invoke(app, _eval_args(out, run_id="curve-src"))
    assert result.exit_code == 0

    result = cli_runner.invoke(
        app,
        [
            "report",
            "--run",
            str(out / "curve-run.jsonl"),
            "--thresholds-from",
            str(out / "curve-src.jsonl"),
            "--curve",
            "--review-rates",
            "0.2,1.0",
        ],
    )
    assert result.exit_code == 0
    assert "曲线：阈值拟合自 curve-src，评估于 curve-run" in result.output
    rows = [
        match
        for match in (_CURVE_ROW_RE.match(line) for line in result.output.splitlines())
        if match
    ]
    assert len(rows) == 2
    first, second = rows
    assert first.group(1) == "0.200"
    assert first.group(2) == "0.167"
    assert first.group(3) == "0.800"
    assert first.group(4) == "0.833"
    assert second.group(2) == "0.333"
    assert second.group(3) == "1.000"
    assert second.group(4) == "1.000"
