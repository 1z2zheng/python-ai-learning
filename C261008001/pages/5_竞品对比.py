from pathlib import Path
import sys

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esports_analysis.comparison import (
    build_evidence_comparison,
    build_comparison_summaries,
    build_parameter_comparison,
    merge_comparison_metrics,
    validate_product_selection,
)
from esports_analysis.config import get_database_path
from esports_analysis.overview import CATEGORY_LABELS, load_overview_data
from esports_analysis.preference import DIMENSIONS, SCORE_VERSION, calculate_dimension_scores
from esports_analysis.pricing import load_price_history
from esports_analysis.product_evidence import load_evidence_data


st.set_page_config(page_title="竞品对比", page_icon="⚖️", layout="wide")


@st.cache_data(ttl=300)
def get_comparison_data(database_path: str, modified_at: float):
    del modified_at
    bundle = load_evidence_data(database_path)
    latest = calculate_dimension_scores(bundle['products'])
    return merge_comparison_metrics(latest, bundle['history']), bundle


def reset_comparison_selection() -> None:
    st.session_state.pop("comparison_products", None)


database_path = get_database_path()
database_mtime = database_path.stat().st_mtime if database_path.exists() else 0.0
data, evidence_bundle = get_comparison_data(str(database_path), database_mtime)

st.title("竞品对比")
st.caption(f"比较2至5款有真实报价的同类商品，结合规格代理指标和来源明确的评论。评分版本：{SCORE_VERSION}")

if data.empty:
    st.warning("数据库中还没有商品数据，请先导入 CSV。")
    st.stop()

with st.sidebar:
    st.header("选择竞品")
    category = st.radio(
        "产品类别",
        options=list(CATEGORY_LABELS),
        format_func=CATEGORY_LABELS.get,
        key="comparison_category",
        horizontal=True,
        on_change=reset_comparison_selection,
    )
    category_data = data[data["category"] == category].copy()
    product_labels = {
        int(row["product_id"]): f"{row['brand']}｜{row['product_name']}｜SKU {row['platform_product_id']}｜严格评论 {row['strict_review_count']}条"
        for _, row in category_data.iterrows()
    }
    default_count = min(3, len(product_labels))
    selected_ids = st.multiselect(
        "商品",
        options=list(product_labels),
        default=list(product_labels)[:default_count],
        format_func=product_labels.get,
        max_selections=5,
        key="comparison_products",
    )
    st.caption(f"已选择 {len(selected_ids)} 款，要求2至5款。")

try:
    selected_ids = validate_product_selection(category_data, selected_ids)
except ValueError as exc:
    st.info(str(exc))
    st.stop()

selected = category_data.set_index("product_id").loc[selected_ids].reset_index()
coverage, comment_dimensions = build_evidence_comparison(evidence_bundle, selected_ids, database_path)

st.subheader('对比证据覆盖')
st.dataframe(coverage, use_container_width=True, hide_index=True,
             column_config={'商品来源': st.column_config.LinkColumn()})
if coverage['严格有效评论'].lt(20).any():
    st.warning('部分商品的严格评论少于20条。评论内容用于探索与核对，不作为商品质量排名依据。')
if coverage['末次报价日期'].nunique() > 1:
    st.info('商品末次报价日期不同。价格图展示各自末次观测，价差不是同一时点的成交价差。')
if coverage['关键规格已知'].lt(coverage['关键规格总数']).any():
    st.caption('部分关键规格缺失，规则代理分采用占位值。低分不能直接解释为真实产品短板。')

metric_columns = st.columns(4)
metric_columns[0].metric("对比商品", len(selected))
metric_columns[1].metric("最低当前价", f"¥{selected['current_price'].min():,.0f}")
metric_columns[2].metric("最高当前价", f"¥{selected['current_price'].max():,.0f}")
metric_columns[3].metric("最大价差", f"¥{selected['current_price'].max() - selected['current_price'].min():,.0f}")

chart_left, chart_right = st.columns(2)
with chart_left:
    figure = go.Figure()
    dimension_keys = list(DIMENSIONS)
    dimension_labels = list(DIMENSIONS.values())
    for _, product in selected.iterrows():
        values = [product[key] for key in dimension_keys]
        figure.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=dimension_labels + [dimension_labels[0]],
                fill="toself",
                name=f"{product['brand']}｜{product['product_name']}",
            )
        )
    figure.update_layout(
        title="六维规格代理得分对比",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        margin=dict(l=30, r=30, t=50, b=20),
    )
    st.plotly_chart(figure, use_container_width=True)

with chart_right:
    price_data = selected.melt(
        id_vars=["product_name", "brand"],
        value_vars=["current_price", "historical_average", "historical_minimum"],
        var_name="价格类型",
        value_name="价格",
    )
    price_data["价格类型"] = price_data["价格类型"].map(
        {
            "current_price": "当前价",
            "historical_average": "历史均价",
            "historical_minimum": "历史最低价",
        }
    )
    figure = px.bar(
        price_data,
        x="product_name",
        y="价格",
        color="价格类型",
        barmode="group",
        title="当前价与历史价格",
        labels={"product_name": "商品", "价格": "价格 元"},
        hover_data=["brand"],
    )
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

st.subheader("参数与评分对比")
comparison_table = build_parameter_comparison(category_data, selected_ids)
st.dataframe(comparison_table, use_container_width=True, hide_index=True)

st.subheader("规则得分构成")
st.caption("优势和短板取自每款商品自身六维得分的最高两项与最低两项，价格位置按当前所选商品的中位数计算。")
summaries = build_comparison_summaries(category_data, selected_ids)
summary_columns = st.columns(len(selected_ids))
for column, product_id in zip(summary_columns, selected_ids):
    item = summaries[product_id]
    strengths = "、".join(f"{name} {score:.1f}" for name, score in item["strengths"])
    weaknesses = "、".join(f"{name} {score:.1f}" for name, score in item["weaknesses"])
    price_difference = item["price_difference_from_median"]
    if abs(price_difference) < 0.05:
        price_text = "与所选商品价格中位数相当"
    elif price_difference < 0:
        price_text = f"比所选商品价格中位数低 {abs(price_difference):.1f}%"
    else:
        price_text = f"比所选商品价格中位数高 {price_difference:.1f}%"
    minimum_difference = item["difference_from_historical_minimum"]
    history_text = (
        "当前即观察期历史最低价"
        if minimum_difference is not None and abs(minimum_difference) < 0.05
        else f"当前比观察期最低价高 {minimum_difference:.1f}%"
        if minimum_difference is not None
        else "缺少历史最低价"
    )
    with column:
        st.markdown(f"**{item['brand']}｜{item['product_name']}**")
        st.write(f"较高代理得分：{strengths}")
        st.write(f"较低代理得分：{weaknesses}")
        st.write(price_text)
        st.write(history_text)

st.info("对比结论基于当前结构化参数、规则评分和选定观察期价格，不替代实际手感测试或长期耐用性测试。")

st.subheader('严格评论的关注结构')
if not coverage['严格有效评论'].gt(0).any():
    st.info('所选商品暂无严格SKU有效评论。')
else:
    dimension_names = comment_dimensions['维度'].drop_duplicates().tolist()
    focus = st.selectbox('选择评论属性', dimension_names)
    plot = comment_dimensions[comment_dimensions['维度'].eq(focus)].copy()
    plot['商品标签'] = plot['商品'] + '｜SKU ' + plot['SKU']
    long = plot.melt(id_vars=['商品标签'], value_vars=['正向','中性','混合','负向'], var_name='规则情感',value_name='情感提及数')
    figure = px.bar(long, x='商品标签', y='情感提及数', color='规则情感', barmode='stack', title=f'{focus}的原始提及计数',
                    color_discrete_map={'正向':'#4C956C','负向':'#D1495B','中性':'#B8B8B8','混合':'#D9A441'})
    st.plotly_chart(figure,use_container_width=True)
    st.caption('各商品样本数与评分组分布可能不同，提及计数及净情感不用于产品优劣排序。零提及不代表用户不关注。')
    st.dataframe(comment_dimensions, use_container_width=True, hide_index=True,
                 column_config={'净情感值':st.column_config.NumberColumn(format='%.3f')})

st.download_button('下载所选商品证据覆盖', coverage.to_csv(index=False).encode('utf-8-sig'),
                   file_name='竞品证据覆盖.csv',mime='text/csv')

