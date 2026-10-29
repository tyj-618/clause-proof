"""从既有运行的 JSONL 重算 spans（locate 三层匹配）并应用 FR-2 降级，不修改原件。

产出 <名>.spans-v3.jsonl 与同名 .meta.json：
- 记录除 spans / judgment.present / route 外全部字段原样保留（run_id / commit /
  token 计量仍指原运行，文件名中的 .spans-v3 即重算标记）
- contract_text 由 dataset.build_queries 按 meta 的 (split, n_contracts, seed)
  重建，query_id 集合与原运行一致
- FR-2 降级（T-014）：present==PRESENT 且全部 span 无效 → present 改 UNCERTAIN，
  quotes 原样保留（失效率统计需要）；降级条目的路由改为转人工（UNCERTAIN 恒转
  人工，无需知道原运行的阈值），其余条目路由原样保留
- 结束时打印 exact / normalized / none 三类计数、重算前后的引用有效率对比与
  降级条数

用法：uv run python scripts/recompute_spans.py results/<run_id>.jsonl
（对原始运行执行；既有的 .spans-v2.jsonl 为历史产物，保留不动）
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from clauseproof.dataset import build_queries
from clauseproof.locate import locate
from clauseproof.models import Presence, Route, Status
from clauseproof.runner import downgrade_unverifiable
from clauseproof.storage import append_record, read_meta, read_records, write_meta


def _meta_path(run_path: Path) -> Path:
    """同名 meta 文件：foo.jsonl → foo.meta.json（同 cli 的派生规则）。"""
    return run_path.with_name(run_path.stem + ".meta.json")


def recompute(run_path: Path) -> Path:
    """重算一份运行的 spans 并应用 FR-2 降级，返回新 JSONL 路径。"""
    meta = read_meta(_meta_path(run_path))
    records = list(read_records(run_path))
    queries = build_queries(split=meta.split, n_contracts=meta.n_contracts, seed=meta.seed)
    text_by_query = {query.query_id: query.contract_text for query in queries}

    out_path = run_path.with_name(run_path.stem + ".spans-v3.jsonl")
    out_path.unlink(missing_ok=True)  # 重跑脚本时覆盖旧产物；原件只读
    write_meta(_meta_path(out_path), meta)

    n_total_spans = 0
    n_old_valid = 0
    n_exact = 0
    n_normalized = 0
    n_none = 0
    n_changed_records = 0
    n_downgraded = 0
    for record in records:
        if record.status is Status.OK:
            # Record 不变式：status==OK 时 judgment 非 None（0002 §7）
            if record.query_id not in text_by_query:
                raise RuntimeError(
                    f"recompute_spans: 记录不在重建的查询集中，期望 query_id 来自 "
                    f"build_queries(split={meta.split!r}, n_contracts={meta.n_contracts}, "
                    f"seed={meta.seed})，实际 {record.query_id}"
                )
            new_spans = locate(record.judgment.quotes, text_by_query[record.query_id])
            new_judgment = downgrade_unverifiable(record.judgment, new_spans)
            downgraded = (
                record.judgment.present is Presence.PRESENT
                and new_judgment.present is Presence.UNCERTAIN
            )
            # 降级后 UNCERTAIN 恒转人工（route.apply_route），其余路由原样保留
            new_route = Route.TO_HUMAN if downgraded else record.route
            if downgraded:
                n_downgraded += 1
        else:
            new_spans = record.spans
            new_judgment = record.judgment
            new_route = record.route
        n_total_spans += len(new_spans)
        n_old_valid += sum(1 for span in record.spans if span.valid)
        n_exact += sum(1 for span in new_spans if span.matched == "exact")
        n_normalized += sum(1 for span in new_spans if span.matched == "normalized")
        n_none += sum(1 for span in new_spans if span.matched == "none")
        if new_spans != record.spans or new_judgment != record.judgment:
            n_changed_records += 1
        append_record(
            out_path, replace(record, judgment=new_judgment, spans=new_spans, route=new_route)
        )

    old_rate = n_old_valid / n_total_spans if n_total_spans else 0.0
    new_rate = (n_exact + n_normalized) / n_total_spans if n_total_spans else 0.0
    print(f"重算 spans：{run_path} → {out_path}（{len(records)} 条记录）")
    print(
        f"span 总数 {n_total_spans} · exact {n_exact} · normalized {n_normalized} · none {n_none}"
    )
    print(f"有效率 {old_rate:.3f} → {new_rate:.3f}（{new_rate - old_rate:+.3f}）")
    print(f"spans 或判定有变化的记录 {n_changed_records}/{len(records)}")
    print(f"FR-2 降级 {n_downgraded} 条（PRESENT 且全部引用定位失败 → UNCERTAIN，转人工）")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从旧 JSONL 重算 spans（三层匹配）并应用 FR-2 降级，不覆盖原件"
    )
    parser.add_argument("run", type=Path, help="既有运行的 JSONL 路径（同名 .meta.json 需存在）")
    args = parser.parse_args()
    recompute(args.run)


if __name__ == "__main__":
    main()
