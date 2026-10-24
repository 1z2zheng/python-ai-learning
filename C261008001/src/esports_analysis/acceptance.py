from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import pandas as pd

from .comparison import build_parameter_comparison, merge_comparison_metrics
from .overview import calculate_summary, load_overview_data
from .preference import DEFAULT_WEIGHTS, DIMENSIONS, calculate_dimension_scores, rank_products
from .pricing import (
    build_positioning_data,
    build_price_band_summary,
    load_price_history,
)
from .promotion import build_product_promotion_metrics


@dataclass
class AcceptanceCheck:
    check_id: str
    category: str
    name: str
    target: str
    actual: str
    passed: bool
    note: str = ""


def _timed(callable_object):
    started = perf_counter()
    result = callable_object()
    return result, (perf_counter() - started) * 1000


def _check(
    check_id: str,
    category: str,
    name: str,
    target: str,
    actual: object,
    passed: bool,
    note: str = "",
) -> AcceptanceCheck:
    return AcceptanceCheck(check_id, category, name, target, str(actual), bool(passed), note)


def run_acceptance_checks(database_path: str | Path) -> tuple[list[AcceptanceCheck], dict[str, float]]:
    checks: list[AcceptanceCheck] = []
    timings: dict[str, float] = {}

    latest, timings["数据概览查询"] = _timed(lambda: load_overview_data(database_path))
    history, timings["历史价格查询"] = _timed(lambda: load_price_history(database_path))
    scored, timings["六维评分"] = _timed(lambda: calculate_dimension_scores(latest))

    demo_mask = latest["product_name"].fillna("").str.contains("演示") | latest["shop_name"].fillna("").str.contains("功能测试数据")
    demo_count = int(demo_mask.sum())
    price_research_ids = set(history.loc[
        history["price_source"].eq("manmanbuy") & history["platform"].isin(["jd", "tmall"]),
        "product_id",
    ])
    research_data = latest[~demo_mask & latest["product_id"].isin(price_research_ids)].copy()
    research_platforms = int(research_data["platform"].nunique()) if not research_data.empty else 0
    research_brands = int(research_data["brand"].nunique()) if not research_data.empty else 0

    checks.extend(
        [
            _check("DATA-001", "功能测试数据", "系统测试商品规模", ">= 80款", len(latest), len(latest) >= 80,
                   "允许包含明确标识的演示商品，仅用于功能与性能测试"),
            _check("DATA-002", "功能测试数据", "价格快照规模", ">= 商品数×3", len(history), len(history) >= len(latest) * 3),
            _check("DATA-003", "论文研究数据", "真实历史价格商品记录", ">= 80款", len(research_data), len(research_data) >= 80,
                   f"仅计京东/天猫真实历史报价记录；排除{demo_count}条演示及无研究价格目录；平台记录数不等于独立型号数"),
            _check("DATA-004", "论文研究数据", "非演示数据来源", ">= 2个平台", research_platforms, research_platforms >= 2),
            _check("DATA-005", "论文研究数据", "非演示品牌覆盖", ">= 5个品牌", research_brands, research_brands >= 5),
        ]
    )

    summary = calculate_summary(latest)
    checks.append(
        _check(
            "QUALITY-001", "数据质量", "关键字段完整率", ">= 95%",
            f"{summary['completeness_rate']:.1f}%", summary["completeness_rate"] >= 95,
        )
    )
    snapshot_counts = history.groupby("product_id").size() if not history.empty else pd.Series(dtype=int)
    history_coverage = float(snapshot_counts.reindex(latest["product_id"], fill_value=0).ge(3).mean() * 100) if len(latest) else 0.0
    checks.append(
        _check("QUALITY-002", "数据质量", "至少三期价格覆盖率", ">= 95%", f"{history_coverage:.1f}%", history_coverage >= 95)
    )

    score_columns_present = set(DIMENSIONS).issubset(scored.columns)
    scores_in_range = score_columns_present and all(scored[name].between(0, 100).all() for name in DIMENSIONS)
    checks.append(
        _check("FUNC-001", "功能", "六维偏好评分", "6个维度且分数在0至100", len(DIMENSIONS), scores_in_range)
    )

    default_category = "mouse" if (scored["category"] == "mouse").any() else scored["category"].iloc[0]
    category_scored = scored[scored["category"] == default_category]
    ranked, timings["偏好排名"] = _timed(lambda: rank_products(category_scored, DEFAULT_WEIGHTS))
    checks.append(
        _check("FUNC-002", "功能", "偏好排序", "全部候选均有排名", len(ranked), len(ranked) == len(category_scored))
    )

    band_summary, timings["价格带统计"] = _timed(lambda: build_price_band_summary(category_scored, default_category))
    positioning, timings["性能价格定位"] = _timed(lambda: build_positioning_data(category_scored))
    pricing_ready = int(band_summary["商品数"].sum()) == len(category_scored) and positioning["positioning_group"].notna().all()
    checks.append(
        _check("FUNC-003", "功能", "定价分析", "价格带与定位覆盖全部候选", len(category_scored), pricing_ready)
    )

    promotion_metrics, timings["促销指标"] = _timed(lambda: build_product_promotion_metrics(history))
    checks.append(
        _check("FUNC-004", "功能", "促销分析", "每款商品均有促销指标", len(promotion_metrics), len(promotion_metrics) == len(latest))
    )

    merged = merge_comparison_metrics(scored, history)
    comparison_candidates = merged[merged["category"] == default_category].head(5)
    if len(comparison_candidates) >= 2:
        comparison_ids = comparison_candidates["product_id"].tolist()
        comparison_table, timings["竞品对比"] = _timed(
            lambda: build_parameter_comparison(merged, comparison_ids)
        )
        comparison_ready = len(comparison_table.columns) == len(comparison_ids) + 1
    else:
        timings["竞品对比"] = 0.0
        comparison_ready = False
    checks.append(
        _check("FUNC-005", "功能", "竞品对比", "2至5款同类商品生成参数表", len(comparison_candidates), comparison_ready)
    )

    overview_time = timings["数据概览查询"]
    preference_time = timings["六维评分"] + timings["偏好排名"]
    comparison_time = timings["竞品对比"]
    checks.extend(
        [
            _check("PERF-001", "性能", "数据概览查询", "< 5000 ms", f"{overview_time:.1f} ms", overview_time < 5000),
            _check("PERF-002", "性能", "偏好评分与排名", "< 2000 ms", f"{preference_time:.1f} ms", preference_time < 2000),
            _check("PERF-003", "性能", "竞品对比生成", "< 3000 ms", f"{comparison_time:.1f} ms", comparison_time < 3000),
        ]
    )
    return checks, timings


def checks_to_frame(checks: list[AcceptanceCheck]) -> pd.DataFrame:
    frame = pd.DataFrame(asdict(item) for item in checks)
    return frame.rename(
        columns={
            "check_id": "编号",
            "category": "类别",
            "name": "检查项",
            "target": "目标",
            "actual": "实际结果",
            "passed": "通过",
            "note": "说明",
        }
    )


def generate_acceptance_markdown(
    checks: list[AcceptanceCheck],
    timings: dict[str, float],
) -> str:
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    passed = sum(item.passed for item in checks)
    failed = len(checks) - passed
    functional = [item for item in checks if item.category in {"功能", "性能", "功能测试数据", "数据质量"}]
    functional_passed = all(item.passed for item in functional)
    research = [item for item in checks if item.category == "论文研究数据"]
    research_ready = all(item.passed for item in research)
    overall = "通过" if functional_passed and research_ready else "有条件通过" if functional_passed else "未通过"

    lines = [
        "# 电竞外设竞品分析系统自动验收报告",
        "",
        f"生成时间：{now}",
        "",
        f"总体结论：**{overall}**",
        "",
        f"检查项共 {len(checks)} 项，通过 {passed} 项，未通过 {failed} 项。",
        "",
        "系统功能与性能测试和论文研究数据就绪度分别判断。演示数据可以用于验证系统，但不得用于市场结论。",
        "",
        "## 验收结果",
        "",
        "| 编号 | 类别 | 检查项 | 目标 | 实际结果 | 结果 | 说明 |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in checks:
        note = item.note.replace("|", "｜")
        result = "通过" if item.passed else "未通过"
        lines.append(
            f"| {item.check_id} | {item.category} | {item.name} | {item.target} | "
            f"{item.actual} | {result} | {note} |"
        )

    lines.extend(["", "## 模块运行耗时", "", "| 模块 | 耗时 ms |", "|---|---:|"])
    for name, milliseconds in timings.items():
        lines.append(f"| {name} | {milliseconds:.2f} |")

    lines.extend(
        [
            "",
            "## 结论说明",
            "",
            f"- 系统功能和性能验收：{'通过' if functional_passed else '未通过'}。",
            f"- 论文研究数据就绪度：{'通过' if research_ready else '未通过'}。",
            "- 未通过的论文数据项需要使用合规获取的真实数据补齐后重新运行验收。",
            "- 性能结果来自本次本机单次运行，正式论文应补充多轮测试的均值、最大值和测试环境。",
            "",
        ]
    )
    return "\n".join(lines)

