"""把一份运行按合同哈希拆成拟合组与评估组（互斥合同），不修改原件。

M4 曲线（T-012 可选部分）用：阈值在拟合组上选择、在评估组上统计，两组合同
互斥，构成留出验证。产出 <fit_id>.jsonl 与 <eval_id>.jsonl（含同名 meta）：

- 记录全部字段原样保留（run_id 等仍指原运行，文件名即分组标记）
- meta 仅 run_id 换为新值；split / n_contracts / seed / n_queries 保持原值 ——
  report 靠这几个字段重建 gold，改动会导致记录与 gold 错配
- 分组规则：sha256(contract_id) 的摘要整数取模 2，余 0 进拟合组、余 1 进评估组

用法：uv run python scripts/split_by_contract.py results/<run_id>.jsonl <fit_id> <eval_id>
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from clauseproof.models import Record
from clauseproof.storage import append_record, read_meta, read_records, write_meta


def _meta_path(run_path: Path) -> Path:
    """同名 meta 文件：foo.jsonl → foo.meta.json（同 cli 的派生规则）。"""
    return run_path.with_name(run_path.stem + ".meta.json")


def _is_fit(contract_id: str) -> bool:
    """分组：sha256(contract_id) 的摘要整数取模 2，余 0 → 拟合组，余 1 → 评估组。"""
    digest = int.from_bytes(sha256(contract_id.encode("utf-8")).digest(), "big")
    return digest % 2 == 0


def split(run_path: Path, fit_id: str, eval_id: str) -> tuple[Path, Path]:
    """按合同哈希把一份运行拆成两组，返回 (拟合组 JSONL, 评估组 JSONL) 路径。"""
    if fit_id == eval_id:
        raise ValueError(f"split_by_contract: run_id 违反 互异，期望 两个不同值，实际 {fit_id!r}")
    meta = read_meta(_meta_path(run_path))
    records = list(read_records(run_path))
    groups: dict[str, list[Record]] = {"fit": [], "eval": []}
    contracts: dict[str, set[str]] = {"fit": set(), "eval": set()}
    for record in records:
        key = "fit" if _is_fit(record.contract_id) else "eval"
        groups[key].append(record)
        contracts[key].add(record.contract_id)
    overlap = contracts["fit"] & contracts["eval"]
    if overlap:
        raise RuntimeError(
            "split_by_contract: 分组违反 合同互斥，期望 两组合同无交集，"
            f"实际 重叠 {sorted(overlap)[0]!r}"
        )
    out_paths: dict[str, Path] = {}
    for key, run_id in (("fit", fit_id), ("eval", eval_id)):
        out_path = run_path.with_name(f"{run_id}.jsonl")
        out_path.unlink(missing_ok=True)  # 重跑脚本时覆盖旧产物；原件只读
        write_meta(_meta_path(out_path), replace(meta, run_id=run_id))
        for record in groups[key]:
            append_record(out_path, record)
        out_paths[key] = out_path
    n_all_contracts = len(contracts["fit"] | contracts["eval"])
    print(f"拆分 {run_path}：{len(records)} 条 · {n_all_contracts} 份合同（两组合同互斥）")
    print(f"拟合组 {fit_id}：{len(groups['fit'])} 条 · {len(contracts['fit'])} 份合同")
    print(f"评估组 {eval_id}：{len(groups['eval'])} 条 · {len(contracts['eval'])} 份合同")
    print(f"输出 {out_paths['fit']} 与 {out_paths['eval']}")
    return out_paths["fit"], out_paths["eval"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按合同哈希把运行拆成拟合/评估两组（合同互斥），不覆盖原件"
    )
    parser.add_argument("run", type=Path, help="既有运行的 JSONL 路径（同名 .meta.json 需存在）")
    parser.add_argument("fit_id", help="拟合组 run_id，输出 <fit_id>.jsonl 与同名 meta")
    parser.add_argument("eval_id", help="评估组 run_id，输出 <eval_id>.jsonl 与同名 meta")
    args = parser.parse_args()
    split(args.run, args.fit_id, args.eval_id)


if __name__ == "__main__":
    main()
