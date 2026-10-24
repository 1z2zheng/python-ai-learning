from __future__ import annotations

import json
from collections.abc import Mapping
from contextvars import ContextVar

import numpy as np
import pandas as pd


SCORE_VERSION = "rule-v2.0-proxy"
MISSING_SCORE = ContextVar("missing_score", default=50.0)

DIMENSIONS = {
    "feel_score": "交互配置代理",
    "response_score": "响应规格代理",
    "comfort_score": "重量与配列适配代理",
    "durability_score": "保修与维护便利代理",
    "appearance_score": "灯效配置代理",
    "value_score": "相对价格优势",
}

DEFAULT_WEIGHTS = {
    "feel_score": 60.0,
    "response_score": 80.0,
    "comfort_score": 70.0,
    "durability_score": 50.0,
    "appearance_score": 30.0,
    "value_score": 80.0,
}


def _clamp(value: float) -> float:
    return float(max(0.0, min(100.0, value)))


def _known(value: object) -> bool:
    return value is not None and not pd.isna(value)


def _polling_score(value: object) -> float:
    if not _known(value):
        return MISSING_SCORE.get()
    points = [(125, 25), (500, 50), (1000, 70), (2000, 80), (4000, 90), (8000, 100)]
    return _clamp(float(np.interp(float(value), [x for x, _ in points], [y for _, y in points])))


def _warranty_score(value: object) -> float:
    if not _known(value):
        return MISSING_SCORE.get()
    return _clamp(40 + float(value) * 2.5)


def _boolean_score(value: object, yes: float, no: float, missing: float | None = None) -> float:
    if not _known(value):
        return MISSING_SCORE.get() if missing is None else missing
    return yes if bool(value) else no


def _mouse_weight_score(value: object) -> float:
    if not _known(value):
        return MISSING_SCORE.get()
    weight = float(value)
    if 50 <= weight <= 70:
        return 95.0
    if weight < 50:
        return _clamp(95 - (50 - weight) * 1.5)
    return _clamp(95 - (weight - 70) * 1.5)


def _keyboard_layout_score(value: object) -> float:
    if not _known(value):
        return MISSING_SCORE.get()
    layout = str(value).strip().lower()
    if layout in {"65%", "68键", "68"}:
        return 85.0
    if layout in {"75%", "80%", "87键", "87"}:
        return 90.0
    if layout in {"98%", "96%", "100%", "104键", "108键"}:
        return 75.0
    if layout in {"60%", "61键", "61"}:
        return 80.0
    return 70.0


def _switch_score(value: object) -> float:
    if not _known(value):
        return 50.0
    name = str(value).strip().lower()
    if "magnetic" in name or "磁" in name:
        return 95.0
    if "optical" in name or "光" in name:
        return 90.0
    if "mechanical" in name or "机械" in name:
        return 75.0
    if "membrane" in name or "薄膜" in name:
        return 50.0
    return 65.0


def _actuation_score(value: object) -> float:
    if not _known(value):
        return MISSING_SCORE.get()
    distance = float(value)
    return _clamp(100 - max(0.0, distance - 0.1) * 22)


def _dpi_score(value: object) -> float:
    if not _known(value):
        return MISSING_SCORE.get()
    points = [(4000, 40), (8000, 50), (16000, 70), (26000, 85), (35000, 95)]
    return _clamp(float(np.interp(float(value), [x for x, _ in points], [y for _, y in points])))


def _price_scores(prices: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(prices, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(MISSING_SCORE.get(), index=prices.index)
    if numeric.notna().sum() <= 1 or numeric.max() == numeric.min():
        return pd.Series(np.where(numeric.notna(), 70.0, MISSING_SCORE.get()), index=prices.index)
    percentile = numeric.rank(method="average", pct=True)
    return (100 - percentile * 60).fillna(MISSING_SCORE.get()).clip(0, 100)


def _mouse_scores(row: pd.Series, price_score: float) -> tuple[dict[str, float], list[str]]:
    weight = _mouse_weight_score(row.get("weight_g"))
    polling = _polling_score(row.get("polling_rate_hz"))
    dpi = _dpi_score(row.get("dpi_max"))
    warranty = _warranty_score(row.get("warranty_months"))
    rgb = _boolean_score(row.get("rgb"), 90, 60)
    buttons = 70.0 if _known(row.get("button_count")) else MISSING_SCORE.get()
    handedness = 80.0 if _known(row.get("handedness")) else MISSING_SCORE.get()

    scores = {
        "feel_score": buttons,
        "response_score": 0.55 * polling + 0.45 * dpi,
        "comfort_score": 0.75 * weight + 0.25 * handedness,
        "durability_score": warranty,
        "appearance_score": rgb,
    }
    scores["value_score"] = price_score
    evidence = [
        f"重量：{row.get('weight_g') if _known(row.get('weight_g')) else '未提供'} g",
        f"回报率：{row.get('polling_rate_hz') if _known(row.get('polling_rate_hz')) else '未提供'} Hz",
        f"最大 DPI：{row.get('dpi_max') if _known(row.get('dpi_max')) else '未提供'}",
        f"保修期：{row.get('warranty_months') if _known(row.get('warranty_months')) else '未提供'} 个月",
    ]
    return {key: round(_clamp(value), 2) for key, value in scores.items()}, evidence


def _keyboard_scores(row: pd.Series, price_score: float) -> tuple[dict[str, float], list[str]]:
    # A switch technology label alone does not establish experience quality.
    switch = 50.0
    polling = _polling_score(row.get("polling_rate_hz"))
    actuation = _actuation_score(row.get("actuation_mm"))
    layout = _keyboard_layout_score(row.get("layout_size"))
    warranty = _warranty_score(row.get("warranty_months"))
    hot_swap = _boolean_score(row.get("hot_swappable"), 90, 60)
    rgb = _boolean_score(row.get("rgb"), 90, 60)

    scores = {
        "feel_score": 0.70 * switch + 0.30 * hot_swap,
        "response_score": 0.60 * polling + 0.40 * actuation,
        "comfort_score": layout,
        "durability_score": warranty,
        "appearance_score": rgb,
    }
    scores["value_score"] = price_score
    evidence = [
        f"轴体类型：{row.get('switch_type') if _known(row.get('switch_type')) else '未提供'}",
        f"最小触发行程：{row.get('actuation_mm') if _known(row.get('actuation_mm')) else '未提供'} mm",
        f"回报率：{row.get('polling_rate_hz') if _known(row.get('polling_rate_hz')) else '未提供'} Hz",
        f"配列：{row.get('layout_size') if _known(row.get('layout_size')) else '未提供'}",
    ]
    return {key: round(_clamp(value), 2) for key, value in scores.items()}, evidence


def calculate_dimension_scores(data: pd.DataFrame, missing_score: float = 50.0) -> pd.DataFrame:
    if not 0 <= missing_score <= 100:
        raise ValueError("缺失值占位分须在0至100之间")
    token = MISSING_SCORE.set(float(missing_score))
    try:
        return _calculate_dimension_scores(data)
    finally:
        MISSING_SCORE.reset(token)


def _calculate_dimension_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate transparent rule-based dimension scores for each product."""
    if data.empty:
        result = data.copy()
        for dimension in DIMENSIONS:
            result[dimension] = pd.Series(dtype=float)
        result["score_version"] = pd.Series(dtype=str)
        result["score_evidence"] = pd.Series(dtype=str)
        return result

    result_parts: list[pd.DataFrame] = []
    for category, category_data in data.groupby("category", sort=False):
        part = category_data.copy()
        price_scores = _price_scores(part["current_price"])
        calculated = []
        for index, row in part.iterrows():
            if category == "mouse":
                scores, evidence = _mouse_scores(row, float(price_scores.loc[index]))
            elif category == "keyboard":
                scores, evidence = _keyboard_scores(row, float(price_scores.loc[index]))
            else:
                raise ValueError(f"不支持的产品类别: {category}")
            calculated.append({**scores, "score_evidence": json.dumps(evidence, ensure_ascii=False)})
        score_frame = pd.DataFrame(calculated, index=part.index)
        for column in score_frame.columns:
            part[column] = score_frame[column]
        part["score_version"] = SCORE_VERSION
        result_parts.append(part)
    return pd.concat(result_parts).sort_index().reset_index(drop=True)


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    missing = set(DIMENSIONS) - set(weights)
    if missing:
        raise ValueError(f"缺少偏好权重: {', '.join(sorted(missing))}")
    values = {key: max(0.0, float(weights[key])) for key in DIMENSIONS}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("至少设置一个非零偏好权重")
    return {key: value / total for key, value in values.items()}


def rank_products(data: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    normalized = normalize_weights(weights)
    ranked = data.copy()
    ranked["match_score"] = sum(ranked[column] * weight for column, weight in normalized.items())
    for column, weight in normalized.items():
        ranked[f"contribution_{column}"] = ranked[column] * weight
    ranked["match_score"] = ranked["match_score"].round(2)
    ranked = ranked.sort_values(["match_score", "current_price"], ascending=[False, True])
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked.reset_index(drop=True)

