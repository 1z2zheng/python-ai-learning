from __future__ import annotations

import pandas as pd


def filter_history_by_date(
    history: pd.DataFrame,
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("开始日期不能晚于结束日期")
    dates = pd.to_datetime(history["collected_date"], errors="coerce").dt.normalize()
    return history[dates.between(start, end, inclusive="both")].copy().reset_index(drop=True)


def build_product_promotion_metrics(history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate price and promotion metrics for each product in the observation window."""
    columns = [
        "product_id", "platform", "category", "brand", "product_name",
        "current_price", "previous_price", "historical_average", "historical_minimum",
        "historical_maximum", "list_price", "discount_rate", "below_average_rate",
        "change_from_previous", "snapshot_count", "promotion_count",
        "promotion_frequency", "is_historical_low", "latest_date",
    ]
    if history.empty:
        return pd.DataFrame(columns=columns)

    records = []
    for product_id, group in history.groupby("product_id", sort=False):
        ordered = group.sort_values("collected_at")
        valid = ordered.dropna(subset=["effective_price"])
        if valid.empty:
            continue
        latest = valid.iloc[-1]
        current = float(latest["effective_price"])
        previous = float(valid.iloc[-2]["effective_price"]) if len(valid) >= 2 else None
        average = float(valid["effective_price"].mean())
        minimum = float(valid["effective_price"].min())
        maximum = float(valid["effective_price"].max())
        list_price = float(latest["list_price"]) if pd.notna(latest["list_price"]) else None
        discount_rate = (
            max(0.0, (list_price - current) / list_price * 100)
            if list_price is not None and list_price > 0
            else 0.0
        )
        below_average_rate = (average - current) / average * 100 if average > 0 else 0.0
        change_from_previous = (
            (current - previous) / previous * 100
            if previous is not None and previous > 0
            else None
        )
        promotion_mask = valid["promotion_text"].fillna("").str.strip().ne("")
        promotion_count = int(promotion_mask.sum())
        records.append(
            {
                "product_id": product_id,
                "platform": latest["platform"],
                "category": latest["category"],
                "brand": latest["brand"],
                "product_name": latest["product_name"],
                "current_price": round(current, 2),
                "previous_price": round(previous, 2) if previous is not None else None,
                "historical_average": round(average, 2),
                "historical_minimum": round(minimum, 2),
                "historical_maximum": round(maximum, 2),
                "list_price": round(list_price, 2) if list_price is not None else None,
                "discount_rate": round(discount_rate, 2),
                "below_average_rate": round(below_average_rate, 2),
                "change_from_previous": round(change_from_previous, 2) if change_from_previous is not None else None,
                "snapshot_count": len(valid),
                "promotion_count": promotion_count,
                "promotion_frequency": round(promotion_count / len(valid) * 100, 2),
                "is_historical_low": abs(current - minimum) < 1e-9,
                "latest_date": latest["collected_date"],
            }
        )
    return pd.DataFrame(records, columns=columns)


def build_brand_promotion_summary(product_metrics: pd.DataFrame) -> pd.DataFrame:
    if product_metrics.empty:
        return pd.DataFrame(
            columns=["品牌", "商品数", "平均当前价", "平均标价降幅", "平均促销频次", "历史低价商品占比"]
        )
    summary = (
        product_metrics.groupby("brand", as_index=False)
        .agg(
            商品数=("product_id", "nunique"),
            平均当前价=("current_price", "mean"),
            平均标价降幅=("discount_rate", "mean"),
            平均促销频次=("promotion_frequency", "mean"),
            历史低价商品数=("is_historical_low", "sum"),
        )
        .rename(columns={"brand": "品牌"})
    )
    summary["历史低价商品占比"] = summary["历史低价商品数"] / summary["商品数"] * 100
    return summary.drop(columns="历史低价商品数")


def build_daily_promotion_summary(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["日期", "价格记录", "促销记录", "促销记录占比", "平均标价降幅"])
    data = history.copy()
    data["促销记录标记"] = data["promotion_text"].fillna("").str.strip().ne("").astype(int)
    valid_list = data["list_price"].notna() & (data["list_price"] > 0)
    data["标价降幅"] = 0.0
    data.loc[valid_list, "标价降幅"] = (
        (data.loc[valid_list, "list_price"] - data.loc[valid_list, "effective_price"])
        / data.loc[valid_list, "list_price"]
        * 100
    ).clip(lower=0)
    summary = (
        data.groupby("collected_date", as_index=False)
        .agg(
            价格记录=("product_id", "count"),
            促销记录=("促销记录标记", "sum"),
            平均标价降幅=("标价降幅", "mean"),
        )
        .rename(columns={"collected_date": "日期"})
    )
    summary["促销记录占比"] = summary["促销记录"] / summary["价格记录"] * 100
    return summary


def detect_raise_then_discount(
    history: pd.DataFrame,
    *,
    rise_threshold: float = 10.0,
    fall_threshold: float = 5.0,
) -> pd.DataFrame:
    """Flag a rise followed by a later fall; this is a clue, not a misconduct verdict."""
    columns = [
        "商品", "品牌", "上涨前价格", "阶段高价", "随后价格", "上涨幅度",
        "回落幅度", "高价日期", "提示",
    ]
    records = []
    for _, group in history.groupby("product_id", sort=False):
        ordered = group.sort_values("collected_at").dropna(subset=["effective_price"])
        if len(ordered) < 3:
            continue
        prices = ordered["effective_price"].astype(float).tolist()
        for peak_index in range(1, len(prices) - 1):
            before = prices[peak_index - 1]
            peak = prices[peak_index]
            after_rows = ordered.iloc[peak_index + 1 :]
            after_position = after_rows["effective_price"].astype(float).idxmin()
            after_row = ordered.loc[after_position]
            after = float(after_row["effective_price"])
            rise = (peak - before) / before * 100 if before > 0 else 0.0
            fall = (peak - after) / peak * 100 if peak > 0 else 0.0
            if rise >= rise_threshold and fall >= fall_threshold:
                peak_row = ordered.iloc[peak_index]
                records.append(
                    {
                        "商品": peak_row["product_name"],
                        "品牌": peak_row["brand"],
                        "上涨前价格": round(before, 2),
                        "阶段高价": round(peak, 2),
                        "随后价格": round(after, 2),
                        "上涨幅度": round(rise, 2),
                        "回落幅度": round(fall, 2),
                        "高价日期": peak_row["collected_date"],
                        "提示": "价格先明显上涨后回落，请结合促销规则和更长观察期核实",
                    }
                )
                break
    return pd.DataFrame(records, columns=columns)


def summarize_promotion_metrics(product_metrics: pd.DataFrame) -> dict[str, object]:
    if product_metrics.empty:
        return {
            "product_count": 0,
            "average_current_price": None,
            "average_discount_rate": 0.0,
            "historical_low_count": 0,
            "average_promotion_frequency": 0.0,
        }
    return {
        "product_count": int(product_metrics["product_id"].nunique()),
        "average_current_price": float(product_metrics["current_price"].mean()),
        "average_discount_rate": float(product_metrics["discount_rate"].mean()),
        "historical_low_count": int(product_metrics["is_historical_low"].sum()),
        "average_promotion_frequency": float(product_metrics["promotion_frequency"].mean()),
    }

