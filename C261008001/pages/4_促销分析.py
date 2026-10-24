from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esports_analysis.config import get_database_path
from esports_analysis.overview import CATEGORY_LABELS, PLATFORM_LABELS
from esports_analysis.pricing import load_price_history
from esports_analysis.promotion import (
    build_brand_promotion_summary,
    build_daily_promotion_summary,
    build_product_promotion_metrics,
    detect_raise_then_discount,
    filter_history_by_date,
    summarize_promotion_metrics,
)


st.set_page_config(page_title="促销分析", page_icon="🏷️", layout="wide")


@st.cache_data(ttl=300)
def get_history(database_path: str, modified_at: float):
    del modified_at
    return load_price_history(database_path)


def reset_promotion_filters() -> None:
    for key in ("promo_platforms", "promo_brands", "promo_dates", "promo_product"):
        st.session_state.pop(key, None)


database_path = get_database_path()
database_mtime = database_path.stat().st_mtime if database_path.exists() else 0.0
history = get_history(str(database_path), database_mtime)

st.title("促销与历史价格分析")
st.caption("比较当前价格与历史价格，观察促销频次，并识别需要进一步核实的异常价格路径。")

if history.empty:
    st.warning("数据库中还没有价格快照，请先导入包含历史日期的 CSV。")
    st.stop()

with st.sidebar:
    st.header("筛选条件")
    category = st.radio(
        "产品类别",
        options=list(CATEGORY_LABELS),
        format_func=CATEGORY_LABELS.get,
        key="promo_category",
        horizontal=True,
        on_change=reset_promotion_filters,
    )
    category_history = history[history["category"] == category].copy()

    platform_options = sorted(category_history["platform"].dropna().unique().tolist())
    platforms = st.multiselect(
        "平台",
        platform_options,
        default=platform_options,
        format_func=lambda value: PLATFORM_LABELS.get(value, value),
        key="promo_platforms",
    )
    platform_history = category_history[category_history["platform"].isin(platforms)]

    brand_options = sorted(platform_history["brand"].dropna().unique().tolist())
    brands = st.multiselect("品牌", brand_options, default=brand_options, key="promo_brands")

    min_date = platform_history["collected_date"].min().date()
    max_date = platform_history["collected_date"].max().date()
    date_range = st.date_input(
        "观察日期",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="promo_dates",
    )

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("请选择完整的开始和结束日期。")
    st.stop()

selected_history = platform_history[platform_history["brand"].isin(brands)]
selected_history = filter_history_by_date(selected_history, date_range[0], date_range[1])
if selected_history.empty:
    st.warning("当前筛选条件下没有价格记录，请调整品牌或日期范围。")
    st.stop()

product_metrics = build_product_promotion_metrics(selected_history)
summary = summarize_promotion_metrics(product_metrics)

metric_columns = st.columns(5)
metric_columns[0].metric("商品数", summary["product_count"])
metric_columns[1].metric("平均当前价", f"¥{summary['average_current_price']:,.0f}")
metric_columns[2].metric("平均标价降幅", f"{summary['average_discount_rate']:.1f}%")
metric_columns[3].metric("处于历史低价", summary["historical_low_count"])
metric_columns[4].metric("平均促销频次", f"{summary['average_promotion_frequency']:.1f}%")

chart_left, chart_right = st.columns(2)
with chart_left:
    figure = px.box(
        product_metrics,
        x="brand",
        y="discount_rate",
        points="all",
        hover_name="product_name",
        title="品牌当前标价降幅分布",
        labels={"brand": "品牌", "discount_rate": "相对标价降幅 %"},
    )
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

with chart_right:
    daily = build_daily_promotion_summary(selected_history)
    figure = px.line(
        daily,
        x="日期",
        y="促销记录占比",
        markers=True,
        title="促销记录占比趋势",
        labels={"促销记录占比": "促销记录占比 %"},
        hover_data=["价格记录", "促销记录", "平均标价降幅"],
    )
    figure.update_yaxes(range=[0, 100])
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

chart_left, chart_right = st.columns(2)
with chart_left:
    product_options = product_metrics["product_name"].tolist()
    selected_product = st.selectbox("查看单品价格", product_options, key="promo_product")
    product_history = selected_history[selected_history["product_name"] == selected_product]
    figure = px.line(
        product_history,
        x="collected_date",
        y="effective_price",
        markers=True,
        title=f"{selected_product} 历史有效价格",
        labels={"collected_date": "日期", "effective_price": "有效价格 元"},
        hover_data=["list_price", "sale_price", "coupon_price", "promotion_text"],
    )
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

with chart_right:
    comparison = product_metrics.melt(
        id_vars=["product_name", "brand"],
        value_vars=["current_price", "historical_average", "historical_minimum"],
        var_name="价格类型",
        value_name="价格",
    )
    labels = {
        "current_price": "当前价",
        "historical_average": "历史均价",
        "historical_minimum": "历史最低价",
    }
    comparison["价格类型"] = comparison["价格类型"].map(labels)
    figure = px.bar(
        comparison,
        x="product_name",
        y="价格",
        color="价格类型",
        barmode="group",
        title="当前价与历史价格对比",
        labels={"product_name": "商品", "价格": "价格 元"},
        hover_data=["brand"],
    )
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

st.subheader("商品促销指标")
display = product_metrics[
    [
        "brand", "product_name", "current_price", "historical_average",
        "historical_minimum", "discount_rate", "below_average_rate",
        "promotion_frequency", "is_historical_low",
    ]
].rename(
    columns={
        "brand": "品牌",
        "product_name": "商品",
        "current_price": "当前价",
        "historical_average": "历史均价",
        "historical_minimum": "历史最低价",
        "discount_rate": "标价降幅",
        "below_average_rate": "低于历史均价",
        "promotion_frequency": "促销频次",
        "is_historical_low": "当前为历史低价",
    }
)
st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "当前价": st.column_config.NumberColumn(format="¥ %.2f"),
        "历史均价": st.column_config.NumberColumn(format="¥ %.2f"),
        "历史最低价": st.column_config.NumberColumn(format="¥ %.2f"),
        "标价降幅": st.column_config.NumberColumn(format="%.1f%%"),
        "低于历史均价": st.column_config.NumberColumn(format="%.1f%%"),
        "促销频次": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
    },
)

st.subheader("价格路径提示")
alerts = detect_raise_then_discount(selected_history)
if alerts.empty:
    st.success("当前观察期内未发现满足规则的明显先涨后降路径。")
else:
    st.warning(f"发现 {len(alerts)} 条需要进一步核实的价格路径。")
    st.dataframe(alerts, use_container_width=True, hide_index=True)
st.caption("规则为价格较前一期上涨至少10%，之后又回落至少5%。提示仅表示价格路径异常，不构成对促销行为性质的判断。")

st.subheader("品牌促销汇总")
brand_summary = build_brand_promotion_summary(product_metrics)
st.dataframe(
    brand_summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "平均当前价": st.column_config.NumberColumn(format="¥ %.2f"),
        "平均标价降幅": st.column_config.NumberColumn(format="%.1f%%"),
        "平均促销频次": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        "历史低价商品占比": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
    },
)

