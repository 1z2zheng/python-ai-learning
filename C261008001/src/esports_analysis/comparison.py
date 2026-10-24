from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .preference import DIMENSIONS
from .promotion import build_product_promotion_metrics


COMMON_FIELDS = [
    ("current_price", "当前价格", "currency"),
    ("historical_average", "历史均价", "currency"),
    ("historical_minimum", "历史最低价", "currency"),
    ("connection_type", "连接方式", "text"),
    ("weight_g", "重量", "gram"),
    ("polling_rate_hz", "回报率", "hz"),
]

CATEGORY_FIELDS = {
    "mouse": [
        ("dpi_max", "最大 DPI", "integer"),
        ("sensor_model", "传感器型号", "text"),
        ("button_count", "按键数量", "integer"),
        ("handedness", "外形设计", "text"),
    ],
    "keyboard": [
        ("switch_type", "轴体类型", "text"),
        ("switch_name", "轴体名称", "text"),
        ("layout_size", "键盘配列", "text"),
        ("key_count", "按键数量", "integer"),
        ("actuation_mm", "最小触发行程", "mm"),
        ("hot_swappable", "支持热插拔", "boolean"),
    ],
}


def merge_comparison_metrics(latest_scored: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    promotion_metrics = build_product_promotion_metrics(history)
    price_columns = [
        "product_id", "historical_average", "historical_minimum", "historical_maximum",
        "discount_rate", "promotion_frequency", "is_historical_low",
    ]
    return latest_scored.merge(promotion_metrics[price_columns], on="product_id", how="left")


def validate_product_selection(data: pd.DataFrame, product_ids: Iterable[int]) -> list[int]:
    selected_ids = list(dict.fromkeys(int(value) for value in product_ids))
    if not 2 <= len(selected_ids) <= 5:
        raise ValueError("请选择2至5款商品进行对比")
    selected = data[data["product_id"].isin(selected_ids)]
    if len(selected) != len(selected_ids):
        raise ValueError("选择中包含不存在的商品")
    if selected["category"].nunique() != 1:
        raise ValueError("只能比较同一类别的商品")
    return selected_ids


def _format_value(value: object, value_type: str) -> str:
    if value is None or pd.isna(value):
        return "未提供"
    if value_type == "currency":
        return f"¥{float(value):,.2f}"
    if value_type == "gram":
        return f"{float(value):g} g"
    if value_type == "hz":
        return f"{int(float(value))} Hz"
    if value_type == "mm":
        return f"{float(value):g} mm"
    if value_type == "integer":
        return str(int(float(value)))
    if value_type == "boolean":
        return "是" if bool(value) else "否"
    if value_type == "score":
        return f"{float(value):.1f}"
    return str(value)


def build_parameter_comparison(data: pd.DataFrame, product_ids: Iterable[int]) -> pd.DataFrame:
    selected_ids = validate_product_selection(data, product_ids)
    selected = data.set_index("product_id").loc[selected_ids].reset_index()
    category = selected["category"].iloc[0]
    fields = COMMON_FIELDS + CATEGORY_FIELDS[category]
    fields += [(dimension, label, "score") for dimension, label in DIMENSIONS.items()]

    result: dict[str, list[str]] = {"对比项": [label for _, label, _ in fields]}
    for _, product in selected.iterrows():
        label = f"{product['brand']}｜{product['product_name']}｜SKU {product.get('platform_product_id', product['product_id'])}"
        result[label] = [_format_value(product.get(field), value_type) for field, _, value_type in fields]
    return pd.DataFrame(result)


def build_comparison_summaries(data: pd.DataFrame, product_ids: Iterable[int]) -> dict[int, dict[str, object]]:
    selected_ids = validate_product_selection(data, product_ids)
    selected = data.set_index("product_id").loc[selected_ids]
    price_median = float(selected["current_price"].median())
    summaries: dict[int, dict[str, object]] = {}

    for product_id, product in selected.iterrows():
        scores = {dimension: float(product[dimension]) for dimension in DIMENSIONS}
        ordered = sorted(scores, key=scores.get, reverse=True)
        strongest = ordered[:2]
        weakest = ordered[-2:]
        price = float(product["current_price"])
        price_difference = (price - price_median) / price_median * 100 if price_median else 0.0
        historical_minimum = product.get("historical_minimum")
        if pd.notna(historical_minimum) and historical_minimum > 0:
            minimum_difference = (price - float(historical_minimum)) / float(historical_minimum) * 100
        else:
            minimum_difference = None
        summaries[int(product_id)] = {
            "product_name": product["product_name"],
            "brand": product["brand"],
            "strengths": [(DIMENSIONS[name], round(scores[name], 1)) for name in strongest],
            "weaknesses": [(DIMENSIONS[name], round(scores[name], 1)) for name in weakest],
            "price_difference_from_median": round(price_difference, 1),
            "difference_from_historical_minimum": round(minimum_difference, 1) if minimum_difference is not None else None,
        }
    return summaries


def build_evidence_comparison(bundle: dict, product_ids: Iterable[int], database_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare provenance and sample coverage; do not turn missing evidence into scores."""
    from .product_evidence import product_evidence
    selected_ids = validate_product_selection(bundle['products'], product_ids)
    records = []
    dimensions = []
    required = {
        'mouse': ['connection_type', 'weight_g', 'polling_rate_hz', 'dpi_max', 'sensor_model'],
        'keyboard': ['connection_type', 'polling_rate_hz', 'switch_type', 'switch_name', 'layout_size'],
    }
    field_labels = {'connection_type':'连接方式','weight_g':'重量','polling_rate_hz':'回报率',
                    'dpi_max':'最大DPI','sensor_model':'传感器型号','switch_type':'轴体类型',
                    'switch_name':'轴体名称','layout_size':'配列'}
    for pid in selected_ids:
        evidence = product_evidence(bundle, pid, database_path)
        p = evidence['product']
        missing = [field for field in required[p['category']] if pd.isna(p.get(field))]
        summary = evidence['price_summary']
        row = summary.iloc[0] if not summary.empty else None
        records.append({
            '商品': p['product_name'], 'SKU': str(p['platform_product_id']),
            '平台': p['platform'], '严格有效评论': len(evidence['reviews']),
            '历史报价数': len(evidence['history']),
            '末次报价日期': row['date_end'].date().isoformat() if row is not None else '未提供',
            '商品页核验': '编号与版本已核验' if not evidence['verification'].empty else '详细核验未建档',
            '关键规格已知': len(required[p['category']])-len(missing),
            '关键规格总数': len(required[p['category']]),
            '待补字段': '、'.join(field_labels[field] for field in missing) or '无',
            '商品来源': p['product_url'],
        })
        part = evidence['dimensions'].copy()
        part['商品'] = p['product_name']
        part['SKU'] = str(p['platform_product_id'])
        dimensions.append(part)
    return pd.DataFrame(records), pd.concat(dimensions, ignore_index=True)

