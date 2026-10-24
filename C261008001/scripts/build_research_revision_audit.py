"""Prepare evidence and independent review protocols without inventing validation."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esports_analysis.config import get_database_path
from esports_analysis.pricing import build_cross_platform_comparison, load_price_history
from esports_analysis.overview import load_overview_data
from esports_analysis.preference import calculate_dimension_scores, rank_products, DIMENSIONS, DEFAULT_WEIGHTS


def main():
    out = ROOT / "output" / "research_revision"
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = ROOT / "data" / "manmanbuy" / "cross_platform_pairs.csv"
    pairs = pd.read_csv(pairs_path, dtype=str)
    audit_path = pairs_path.parent / "cross_platform_verification.json"
    records = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    for row in pairs.to_dict("records"):
        records.setdefault(row["pair_key"], {
            **row, "status": "pending", "configuration_evidence": "", "offer_evidence": "",
            "verified_by": "", "verified_at": "", "jd_variant": "", "tmall_variant": "",
            "shop_evidence": "", "note": "需核验版本、颜色、轴体、配列、套装、店铺与优惠资格",
        })
    audit_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    history = load_price_history(get_database_path())
    history = history[history["price_source"] == "manmanbuy"]
    cross = build_cross_platform_comparison(history, pairs)
    cross.to_csv(out / "cross_platform_audit.csv", index=False, encoding="utf-8-sig")
    research_ids = set(history["product_id"])
    overview = load_overview_data(get_database_path())
    scored = calculate_dimension_scores(overview[overview["product_id"].isin(research_ids)])
    sensitivity_rows = []
    for category, part in scored.groupby("category"):
        baseline = rank_products(part, DEFAULT_WEIGHTS).set_index("product_id")
        top = set(baseline.head(5).index)
        scenarios = [("equal_weights", {d: 1 for d in DIMENSIONS})]
        scenarios += [("remove_" + d, {k: 0 if k == d else v for k, v in DEFAULT_WEIGHTS.items()}) for d in DIMENSIONS]
        rng = np.random.default_rng(20261014)
        scenarios += [(f"random_{i:04d}", dict(zip(DIMENSIONS, rng.uniform(1, 100, len(DIMENSIONS))))) for i in range(1000)]
        for scenario, weights in scenarios:
            ranked = rank_products(part, weights).set_index("product_id")
            aligned = baseline["rank"].to_frame("base").join(ranked["rank"].rename("scenario"))
            correlation = aligned["base"].rank().corr(aligned["scenario"].rank())
            sensitivity_rows.append({"category": category, "scenario": scenario,
                                     "rank_correlation": correlation,
                                     "top5_overlap": len(top & set(ranked.head(5).index)) / min(5, len(part)),
                                     "top1_product_id": int(ranked.index[0])})
        for missing in (25, 75):
            changed = calculate_dimension_scores(overview[(overview["category"] == category) & overview["product_id"].isin(research_ids)], missing_score=missing)
            ranked = rank_products(changed, DEFAULT_WEIGHTS).set_index("product_id")
            aligned = baseline["rank"].to_frame("base").join(ranked["rank"].rename("scenario"))
            sensitivity_rows.append({"category": category, "scenario": f"missing_score_{missing}",
                                     "rank_correlation": aligned["base"].rank().corr(aligned["scenario"].rank()),
                                     "top5_overlap": len(top & set(ranked.head(5).index)) / min(5, len(part)),
                                     "top1_product_id": int(ranked.index[0])})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(out / "score_sensitivity.csv", index=False, encoding="utf-8-sig")
    sensitivity.groupby("category").agg(
        mean_rank_correlation=("rank_correlation", "mean"),
        min_rank_correlation=("rank_correlation", "min"),
        mean_top5_overlap=("top5_overlap", "mean"),
    ).to_csv(out / "score_sensitivity_summary.csv", encoding="utf-8-sig")
    metrics = {
        "price_observations": len(history),
        "platform_product_records": int(history["product_id"].nunique()),
        "candidate_model_pairs": len(pairs),
        "verified_comparable_pairs": int(cross["comparison_eligible"].sum()),
        "independent_user_study": "pending",
        "independent_comment_annotation": "pending",
    }
    inputs = [get_database_path(), pairs_path, ROOT / "src" / "esports_analysis" / "review_cleaning.py", ROOT / "src" / "esports_analysis" / "preference.py"]
    metrics["sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    (out / "evidence_manifest.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol = """# 研究改进与验证方案

## 研究问题

1. 在核验配置和报价条件后，同型号商品的跨平台报价差异与历史波动如何变化。
2. 在采集方式、评分组和SKU归属分层后，评论关注属性及文本情感结构如何变化。
3. 规则代理排序能否帮助用户完成选品任务，其选择一致性和可用性如何。

## 数据边界

价格观测数、平台商品记录数和独立型号候选数分别报告。日序列有自相关，不能把每个价格点当成独立用户或独立商品。第三方报价不等于成交价，评价数只作为累积热度代理，不计算实际销量关系。

## 模型依据与消融验证

评分版本rule-v2.0-proxy。交互配置、响应规格、重量与配列、保修、灯效和相对价格为六类代理。取消磁轴优于机械轴的等级假定，手感不再重复使用重量，价格不再重复汇入性能，保修代理不再汇入热插拔，灯效代理不再汇入配列。

规则节点与默认权重属于待验证的演示设定。缺失值50是计算占位，不能解释为真实中等质量。脚本已经执行移除单维度、等权、1000组随机权重和缺失值25/50/75的敏感性分析，输出Spearman相关与前五名重合率。节点不确定性与用户偏好不能互相替代。

## 独立评论复核

使用output/review_analysis中的双人标注模板。两名实际标注者独立标注六维属性及正向、中性、混合、负向情感，禁止查看自动预测。先在小部分文本试标并统一定义，另留独立测试样本，冻结词典后再评估。记录分歧与裁决。报告Cohen kappa、各类别precision/recall/F1、macro F1和混淆矩阵，尚无标注时不报告准确率。总体评论与方面级标签分别评估。

## 用户实验

计划邀请约12至20名用户，记录实际参与人数和经验分组。使用相同候选集和预算任务，采用顺序平衡方式比较普通列表与系统。记录任务成功、时间、错误、选中型号、理由及可用性评分。参与者使用匿名编号，获得同意后收集。当前没有执行实验，不填写结果。

## 跨平台核验

逐组填写cross_platform_verification.json。核对代际、颜色、轴体、配列、连接方式、套装和店铺，记录证据及核验时间。确认优惠是否需要会员、地区补贴或优惠券。只有完整核验、观测日期相同且没有重大价格异常的记录用于平台优势统计。其他记录保留为待核验候选。

## 论文修订要求

旧版情感准确率、旧版综合排名和鼠标耐用性优劣结论全部暂停引用。论文使用最新thesis_evidence.json及六维清洗证据，注明词典版本、样本选择和观察期。既有Word初稿属于旧版，需在独立复核完成后统一更新，不能仅替换数字。未完成用户实验和独立标注如实记为待完成。
"""
    (out / "research_revision_protocol.md").write_text(protocol, encoding="utf-8")
    participant_path = out / "user_study_records.csv"
    if not participant_path.exists():
        with participant_path.open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle).writerow(["anonymous_id", "experience_group", "condition", "order", "task_id", "success", "elapsed_seconds", "error_count", "chosen_model", "choice_reason", "consent_record", "study_date"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
