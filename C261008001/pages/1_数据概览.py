from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esports_analysis.config import get_database_path
from esports_analysis.overview import (
    CATEGORY_LABELS,
    CONNECTION_LABELS,
    PLATFORM_LABELS,
    build_quality_report,
    calculate_summary,
    filter_overview_data,
    load_overview_data,
)


st.set_page_config(page_title="数据概览", page_icon="📊", layout="wide")


@st.cache_data(ttl=300)
def get_data(database_path: str, modified_at: float) -> pd.DataFrame:
    del modified_at
    return load_overview_data(database_path)


def reset_filters() -> None:
    for key in ("overview_category", "overview_platforms", "overview_brands", "overview_price"):
        st.session_state.pop(key, None)


database_path = get_database_path()
database_mtime = database_path.stat().st_mtime if database_path.exists() else 0.0
data = get_data(str(database_path), database_mtime)

st.title("数据概览")
st.caption("查看当前数据覆盖、价格结构与字段质量。所有图表使用相同筛选结果。")

if data.empty:
    st.warning("数据库中还没有商品数据，请先使用 CSV 导入命令添加数据。")
    st.code(r".\.venv\Scripts\python.exe scripts\import_csv.py data\seed\products_sample.csv")
    st.stop()

with st.sidebar:
    st.header("筛选条件")
    if st.button("重置筛选", use_container_width=True):
        reset_filters()
        st.rerun()

    category = st.radio(
        "产品类别",
        options=list(CATEGORY_LABELS),
        format_func=CATEGORY_LABELS.get,
        key="overview_category",
        horizontal=True,
    )
    category_data = data[data["category"] == category]

    platform_options = sorted(category_data["platform"].dropna().unique().tolist())
    platforms = st.multiselect(
        "平台",
        options=platform_options,
        default=platform_options,
        format_func=lambda value: PLATFORM_LABELS.get(value, value),
        key="overview_platforms",
    )
    platform_data = category_data[category_data["platform"].isin(platforms)] if platforms else category_data.iloc[0:0]

    brand_options = sorted(platform_data["brand"].dropna().unique().tolist())
    brands = st.multiselect(
        "品牌",
        options=brand_options,
        default=brand_options,
        key="overview_brands",
    )

    valid_prices = platform_data["current_price"].dropna()
    if valid_prices.empty:
        selected_price = None
        st.info("当前条件下没有有效价格。")
    else:
        price_min = int(valid_prices.min())
        price_max = max(price_min + 1, int(valid_prices.max()) + 1)
        selected_price = st.slider(
            "价格范围 元",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, price_max),
            key="overview_price",
        )

filtered = filter_overview_data(
    data,
    category=category,
    platforms=platforms,
    brands=brands,
    price_range=selected_price,
)

if filtered.empty:
    st.warning("当前筛选条件下没有商品，请放宽条件或重置筛选。")
    st.stop()

summary = calculate_summary(filtered)
latest_update = summary["latest_update"]
latest_text = latest_update.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M") if pd.notna(latest_update) else "未提供"
st.caption(f"数据更新时间：{latest_text} ｜ 当前类别：{CATEGORY_LABELS[category]}")

metric_columns = st.columns(5)
metric_columns[0].metric("商品数", summary["product_count"])
metric_columns[1].metric("品牌数", summary["brand_count"])
metric_columns[2].metric("平台数", summary["platform_count"])
average_price_text = f"¥{summary['average_price']:,.0f}" if summary["average_price"] is not None else "未提供"
metric_columns[3].metric("平均价格", average_price_text)
metric_columns[4].metric("关键字段完整率", f"{summary['completeness_rate']:.1f}%")

if len(data) < 80:
    st.info(f"当前数据库共 {len(data)} 款商品，尚未达到开题报告规定的 80 款目标；现有数据用于功能验证。")

chart_left, chart_right = st.columns(2)
with chart_left:
    brand_counts = filtered.groupby("brand", as_index=False).agg(商品数=("product_id", "nunique"))
    brand_counts = brand_counts.sort_values("商品数", ascending=True)
    figure = px.bar(
        brand_counts,
        x="商品数",
        y="brand",
        orientation="h",
        title="品牌商品数量",
        labels={"brand": "品牌"},
        text_auto=True,
    )
    figure.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

with chart_right:
    figure = px.histogram(
        filtered,
        x="current_price",
        color="brand",
        nbins=min(10, max(3, len(filtered))),
        title="商品价格分布",
        labels={"current_price": "当前价格 元", "brand": "品牌"},
        barmode="overlay",
        opacity=0.75,
    )
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

chart_left, chart_right = st.columns(2)
with chart_left:
    connection_counts = filtered.assign(
        连接方式=filtered["connection_type"].fillna("unknown").map(
            lambda value: CONNECTION_LABELS.get(value, value)
        )
    ).groupby("连接方式", as_index=False).agg(商品数=("product_id", "nunique"))
    figure = px.pie(
        connection_counts,
        names="连接方式",
        values="商品数",
        hole=0.45,
        title="连接方式占比",
    )
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

with chart_right:
    brand_prices = filtered.groupby("brand", as_index=False).agg(平均价格=("current_price", "mean"))
    brand_prices = brand_prices.sort_values("平均价格", ascending=False)
    figure = px.bar(
        brand_prices,
        x="brand",
        y="平均价格",
        title="品牌平均价格",
        labels={"brand": "品牌", "平均价格": "平均价格 元"},
        text_auto=".0f",
    )
    figure.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

st.subheader("数据质量")
quality = build_quality_report(filtered, category)
st.dataframe(
    quality,
    use_container_width=True,
    hide_index=True,
    column_config={
        "完整率": st.column_config.ProgressColumn(
            "完整率",
            min_value=0,
            max_value=100,
            format="%.1f%%",
        )
    },
)

st.subheader("商品明细")
details = filtered[
    ["brand", "product_name", "platform", "current_price", "connection_type", "collected_at"]
].copy()
details["platform"] = details["platform"].map(lambda value: PLATFORM_LABELS.get(value, value))
details["connection_type"] = details["connection_type"].fillna("unknown").map(
    lambda value: CONNECTION_LABELS.get(value, value)
)
details["collected_at"] = details["collected_at"].dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d %H:%M")
details = details.rename(
    columns={
        "brand": "品牌",
        "product_name": "商品名称",
        "platform": "平台",
        "current_price": "当前价格",
        "connection_type": "连接方式",
        "collected_at": "数据时间",
    }
)
st.dataframe(
    details,
    use_container_width=True,
    hide_index=True,
    column_config={"当前价格": st.column_config.NumberColumn(format="¥ %.2f")},
)

