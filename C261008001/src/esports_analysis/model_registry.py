"""Evidence-preserving model grouping; a candidate pair is not verified identity."""
import csv
from pathlib import Path

import pandas as pd

from .product_evidence import load_evidence_data

PAIR_PATH = Path(__file__).resolve().parents[2] / "data/manmanbuy/cross_platform_pairs.csv"
# Exact source IDs, not fuzzy title stripping. Versions remain distinct.
EXTRA_MODELS = {
    "100001691967": ("logitech_g502_hero_wired", "G502 HERO有线版"),
    "100041149519": ("logitech_g502x_wireless", "G502 X LIGHTSPEED无线版"),
    "100033870975": ("razer_deathadder_v3_pro", "炼狱蝰蛇V3专业版"),
    "100190452813": ("razer_deathadder_v4_pro", "炼狱蝰蛇V4专业版"),
    "100063306539": ("razer_blackwidow_v4", "黑寡妇蜘蛛V4"),
    "100043546994": ("razer_blackwidow_v4_pro", "黑寡妇蜘蛛V4专业版"),
    "100266391406": ("razer_blackwidow_v4_lowprofile", "黑寡妇蜘蛛V4矮轴极速版"),
    "100056847840": ("razer_cobra_pro", "眼镜蛇专业版"),
    "100108649784": ("razer_viper_v3_pro", "毒蝰V3专业版"),
    "7571300": ("logitech_g304_lightspeed", "G304 LIGHTSPEED"),
    "10098016952114": ("razer_basilisk_v3_wired", "巴塞利斯蛇V3有线版"),
}
CONFLICTS = {
    ("jd", "100066734479"): ("rog_scope2_98_nx_wireless", "游侠2 98 NX版", "两侧NX灰轴与RX光轴标题冲突，拆组"),
    ("tmall", "827716604537"): ("rog_scope2_rx98_wireless", "游侠2 RX98版", "两侧NX灰轴与RX光轴标题冲突，拆组"),
    ("jd", "100036347936"): ("dareu_a950_mecha", "A950机甲版", "机甲版与镁合金标题存在版本疑点，保守拆组"),
    ("tmall", "673074579811"): ("dareu_a950_magnesium", "A950镁合金版", "机甲版与镁合金标题存在版本疑点，保守拆组"),
}


def normalize_model_records(products, pairs, verified_ids=()):
    lookups = {}
    for pair in pairs:
        for platform, field in (("jd", "jd_sku"), ("tmall", "tmall_item_id")):
            key = (platform, str(pair[field]))
            if key in lookups and lookups[key][0] != pair["pair_key"]:
                raise ValueError("同一平台SKU存在多个型号归属")
            lookups[key] = (pair["pair_key"], pair["model"])
    rows = []
    for product in products.to_dict("records"):
        platform, sku = product["platform"], str(product["platform_product_id"])
        key = (platform, sku)
        model = lookups.get(key)
        basis = "候选配对表；不代表配置已核验"
        if platform == "jd" and sku in EXTRA_MODELS:
            model = EXTRA_MODELS[sku]
            basis = "指定SKU的源标题和版本记录"
        risk = ""
        if key in CONFLICTS:
            code, name, risk = CONFLICTS[key]
            model = (code, name)
            basis = "标题版本存在冲突，未采用旧配对关系"
        if model is None:
            model = (f"unresolved_{platform}_{sku}", "待核验型号")
            basis = "没有明确型号映射，不使用模糊名称自动合并"
        # Namespace prevents equal model strings across brands/categories collapsing.
        candidate_key = f"{product['category']}:{product['brand']}:{model[0]}"
        status = "商品页版本已建档" if product["product_id"] in verified_ids else "型号归属待复核"
        if platform == "tmall":
            status = "跨平台配置待核验"
        if risk:
            status = "版本冲突已拆组"
        rows.append({"商品ID": product["product_id"], "平台": platform, "SKU": sku,
                     "品类": product["category"], "品牌": product["brand"], "型号组编号": candidate_key,
                     "候选型号": model[1], "归属状态": status, "归属依据": basis, "版本风险": risk,
                     "严格评论数": int(product.get("strict_review_count", 0)),
                     "源商品标题": product["product_name"], "商品来源": product.get("product_url", "")})
    return pd.DataFrame(rows, columns=["商品ID", "平台", "SKU", "品类", "品牌", "型号组编号", "候选型号",
        "归属状态", "归属依据", "版本风险", "严格评论数", "源商品标题", "商品来源"])


def build_model_registry(database_path):
    bundle = load_evidence_data(database_path)
    products = bundle["products"].copy()
    products = products[products["platform"].isin(["jd", "tmall"])
                        & ~products["product_name"].fillna("").str.contains("演示", regex=False)
                        & ~products["shop_name"].fillna("").str.contains("功能测试数据", regex=False)]
    with PAIR_PATH.open(encoding="utf-8-sig", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    records = normalize_model_records(products, pairs, set(bundle["verified"]["product_id"]))
    if records.empty:
        return records, pd.DataFrame(columns=["型号组编号", "候选型号", "平台记录数", "平台数", "严格评论记录数"])
    groups = records.groupby(["型号组编号", "候选型号"], as_index=False).agg(
        平台记录数=("商品ID", "nunique"), 平台数=("平台", "nunique"), 严格评论记录数=("严格评论数", "sum"))
    return records, groups
