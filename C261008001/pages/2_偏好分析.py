from pathlib import Path
import json
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esports_analysis.config import get_database_path
from esports_analysis.overview import (
    CATEGORY_LABELS,
    CONNECTION_LABELS,
    load_overview_data,
)
from esports_analysis.preference import (
    DEFAULT_WEIGHTS,
    DIMENSIONS,
    SCORE_VERSION,
    calculate_dimension_scores,
    rank_products,
)
from esports_analysis.review_analysis import (
    build_product_feedback_summary,
    load_review_research_data,
)
from esports_analysis.review_identity import load_verified_product_catalog


st.set_page_config(page_title="偏好分析", page_icon="🎯", layout="wide")


@st.cache_data(ttl=300)
def get_scored_data(database_path: str, modified_at: float) -> pd.DataFrame:
    del modified_at
    return calculate_dimension_scores(load_overview_data(database_path))


@st.cache_data(ttl=300)
def get_review_data(database_path: str, modified_at: float):
    del modified_at
    reviews, heat = load_review_research_data(database_path)
    return reviews, build_product_feedback_summary(reviews, heat)


@st.cache_data(ttl=300)
def get_review_preference_analysis(
    summary_path: str,
    summary_mtime: float,
    cleaned_path: str,
    cleaned_mtime: float,
):
    del summary_mtime, cleaned_mtime
    summary_file = Path(summary_path)
    analysis_dir = summary_file.parent
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    cleaned = pd.read_csv(cleaned_path, encoding="utf-8-sig")
    dimensions = pd.read_csv(
        analysis_dir / "review_dimension_summary.csv", encoding="utf-8-sig"
    )
    categories = pd.read_csv(
        analysis_dir / "review_category_dimension_summary.csv", encoding="utf-8-sig"
    )
    products = pd.read_csv(
        analysis_dir / "review_product_dimension_summary_strict.csv", encoding="utf-8-sig"
    )
    return summary, cleaned, dimensions, categories, products


def reset_preferences() -> None:
    keys = ["pref_category", "pref_brands", "pref_connections", "pref_budget", "pref_count"]
    keys.extend(f"weight_{name}" for name in DIMENSIONS)
    for key in keys:
        st.session_state.pop(key, None)


def reset_category_filters() -> None:
    for key in ("pref_brands", "pref_connections", "pref_budget"):
        st.session_state.pop(key, None)


database_path = get_database_path()
database_mtime = database_path.stat().st_mtime if database_path.exists() else 0.0
data = get_scored_data(str(database_path), database_mtime)
reviews, feedback = get_review_data(str(database_path), database_mtime)

st.title("用户偏好分析")
st.caption(f"规格代理指标与自定义权重匹配，评分规则版本：{SCORE_VERSION}。规则节点尚待用户验证。")

if data.empty:
    st.warning("数据库中还没有商品数据，请先导入 CSV。")
    st.stop()

with st.sidebar:
    st.header("选购条件")
    if st.button("恢复默认设置", use_container_width=True):
        reset_preferences()
        st.rerun()

    category = st.radio(
        "产品类别",
        options=list(CATEGORY_LABELS),
        format_func=CATEGORY_LABELS.get,
        key="pref_category",
        horizontal=True,
        on_change=reset_category_filters,
    )
    category_data = data[data["category"] == category].copy()

    brand_options = sorted(category_data["brand"].dropna().unique().tolist())
    brands = st.multiselect("品牌", brand_options, default=brand_options, key="pref_brands")

    connection_options = sorted(category_data["connection_type"].dropna().unique().tolist())
    connections = st.multiselect(
        "连接方式",
        connection_options,
        default=connection_options,
        format_func=lambda value: CONNECTION_LABELS.get(value, value),
        key="pref_connections",
    )

    price_values = category_data["current_price"].dropna()
    price_min = int(price_values.min())
    price_max = max(price_min + 1, int(price_values.max()) + 1)
    budget = st.slider(
        "预算范围 元",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max),
        key="pref_budget",
    )

    st.header("偏好权重")
    weights = {}
    for dimension, label in DIMENSIONS.items():
        weights[dimension] = st.slider(
            label,
            min_value=0,
            max_value=100,
            value=int(DEFAULT_WEIGHTS[dimension]),
            key=f"weight_{dimension}",
        )
    recommendation_count = st.selectbox("推荐数量", [3, 5, 10], key="pref_count")

candidates = category_data[
    category_data["brand"].isin(brands)
    & category_data["connection_type"].isin(connections)
    & category_data["current_price"].between(budget[0], budget[1], inclusive="both")
].copy()

if candidates.empty:
    st.warning("当前条件下没有候选商品，请放宽品牌、连接方式或预算限制。")
    st.stop()

try:
    ranked = rank_products(candidates, weights)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

shown = ranked.head(min(recommendation_count, len(ranked))).copy()

metric_columns = st.columns(4)
metric_columns[0].metric("候选商品", len(candidates))
metric_columns[1].metric("推荐商品", len(shown))
metric_columns[2].metric("最高匹配度", f"{shown.iloc[0]['match_score']:.1f}")
metric_columns[3].metric("评分版本", SCORE_VERSION)

st.info("推荐分数继续使用可核验产品参数；真实评论情感和优缺点作为独立证据展示，避免少量代表性评论过度影响排名。")

chart_left, chart_right = st.columns(2)
with chart_left:
    bar_data = shown.sort_values("match_score", ascending=True)
    figure = px.bar(
        bar_data,
        x="match_score",
        y="product_name",
        orientation="h",
        color="brand",
        text="match_score",
        title="推荐商品匹配度",
        labels={"match_score": "综合匹配度", "product_name": "商品", "brand": "品牌"},
        range_x=[0, 100],
    )
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

with chart_right:
    figure = go.Figure()
    radar_dimensions = list(DIMENSIONS)
    radar_labels = list(DIMENSIONS.values())
    for _, product in shown.head(3).iterrows():
        values = [product[column] for column in radar_dimensions]
        figure.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name=product["product_name"],
            )
        )
    figure.update_layout(
        title="前三名六维得分",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        margin=dict(l=30, r=30, t=50, b=20),
    )
    st.plotly_chart(figure, use_container_width=True)

st.subheader("推荐结果")
display_columns = ["rank", "brand", "product_name", "current_price", "match_score", *DIMENSIONS]
display = shown[display_columns].rename(
    columns={
        "rank": "排名",
        "brand": "品牌",
        "product_name": "商品名称",
        "current_price": "当前价格",
        "match_score": "综合匹配度",
        **DIMENSIONS,
    }
)
st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "当前价格": st.column_config.NumberColumn(format="¥ %.2f"),
        "综合匹配度": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
    },
)

selected_name = st.selectbox("查看推荐依据", shown["product_name"].tolist())
selected = shown[shown["product_name"] == selected_name].iloc[0]
normalized_total = sum(max(0.0, float(value)) for value in weights.values())
contributions = pd.DataFrame(
    {
        "维度": list(DIMENSIONS.values()),
        "贡献值": [selected[f"contribution_{dimension}"] for dimension in DIMENSIONS],
        "原始得分": [selected[dimension] for dimension in DIMENSIONS],
        "权重占比": [max(0.0, weights[dimension]) / normalized_total * 100 for dimension in DIMENSIONS],
    }
)
figure = px.bar(
    contributions.sort_values("贡献值", ascending=True),
    x="贡献值",
    y="维度",
    orientation="h",
    text_auto=".1f",
    title=f"{selected_name} 的综合得分构成",
    hover_data=["原始得分", "权重占比"],
)
figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(figure, use_container_width=True)

evidence = json.loads(selected["score_evidence"])
st.markdown("**参数依据**")
st.markdown("\n".join(f"- {item}" for item in evidence))
st.caption("缺失参数采用中性分 50，并在依据中标注为未提供；价格得分按当前同类别候选商品的相对价格计算。")

st.subheader("真实评价热度与用户反馈")
category_feedback = feedback[feedback["category"] == category].copy()
category_reviews = reviews[reviews["category"] == category].copy() if not reviews.empty else reviews
if category_feedback.empty:
    st.info("当前类别暂无真实评价数据。")
else:
    review_metrics = st.columns(4)
    review_metrics[0].metric("有评论或热度商品", len(category_feedback))
    review_metrics[1].metric("脱敏评论样本", int(category_feedback["sample_review_count"].sum()))
    review_metrics[2].metric(
        "评价数下限合计",
        f"{int(category_feedback['review_count_lower_bound'].fillna(0).sum()):,}",
    )
    review_metrics[3].metric(
        "抽样方式",
        "非概率样本",
    )

    relationship = category_feedback.merge(
        category_data[["product_id", "current_price"]], on="product_id", how="left"
    ).dropna(subset=["current_price", "review_count_lower_bound"])
    if not relationship.empty:
        figure = px.scatter(
            relationship,
            x="current_price",
            y="review_count_lower_bound",
            color="brand",
            size="sample_review_count",
            hover_name="product_name",
            title="价格与评价热度关系",
            labels={
                "current_price": "当前价格 元",
                "review_count_lower_bound": "评价数下限（热度代理）",
                "brand": "品牌",
            },
            log_y=True,
        )
        figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(figure, use_container_width=True)

    feedback_display = category_feedback.rename(columns={
        "brand": "品牌", "product_name": "商品名称",
        "review_count_text": "平台评价数", "sample_review_count": "评论样本",
        "average_rating": "样本平均评分", "positive_rate": "正向占比",
        "top_aspects": "主要关注点", "top_positive_points": "评论优点摘录",
        "top_negative_points": "评论缺点摘录", "platform_tags": "平台评价标签",
    })
    st.dataframe(
        feedback_display[[
            "品牌", "商品名称", "平台评价数", "评论样本", "样本平均评分", "平台评价标签",
        ]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "样本平均评分": st.column_config.NumberColumn(format="%.2f"),
            "正向占比": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        },
    )
    st.caption("评价数为平台展示值的下限代理；情感与关注结构统一使用下方六维模块，星级仅作为原始采集属性。")

st.subheader("六维评论偏好")
analysis_dir = ROOT / "output" / "review_analysis"
summary_path = analysis_dir / "review_analysis_summary.json"
analysis_metadata = {}
if summary_path.exists():
    try:
        analysis_metadata = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        analysis_metadata = {}
cleaned_path = Path(
    analysis_metadata.get("outputs", {}).get(
        "cleaned_reviews", ROOT / "data" / "processed" / "jd_reviews_cleaned_latest.csv"
    )
)
dimension_mentions_path = Path(
    analysis_metadata.get("outputs", {}).get(
        "dimension_mentions", ROOT / "data" / "processed" / "jd_review_dimension_mentions_latest.csv"
    )
)

required_analysis_files = [
    summary_path,
    cleaned_path,
    analysis_dir / "review_dimension_summary.csv",
    analysis_dir / "review_category_dimension_summary.csv",
    analysis_dir / "review_product_dimension_summary_strict.csv",
]
missing_analysis_files = [path for path in required_analysis_files if not path.exists()]

if missing_analysis_files:
    st.info("六维评论偏好数据尚未生成。运行评论分析脚本后即可查看。")
    st.code("python scripts/analyze_review_preferences.py")
else:
    (
        review_analysis_summary,
        cleaned_reviews,
        review_dimension_summary,
        review_category_summary,
        review_product_summary,
    ) = get_review_preference_analysis(
        str(summary_path),
        summary_path.stat().st_mtime,
        str(cleaned_path),
        cleaned_path.stat().st_mtime,
    )

    category_cleaned = cleaned_reviews[cleaned_reviews["category"] == category].copy()
    category_eligible = category_cleaned[category_cleaned["preference_eligible"] == True].copy()  # noqa: E712
    category_dimensions = review_category_summary[
        review_category_summary["category"] == category
    ].copy()
    category_products = review_product_summary[
        review_product_summary["category"] == category
    ].copy()

    quality_metrics = st.columns(5)
    quality_metrics[0].metric("原始评论", len(category_cleaned))
    quality_metrics[1].metric("有效评论", len(category_eligible))
    eligibility_rate = len(category_eligible) / len(category_cleaned) * 100 if len(category_cleaned) else 0
    quality_metrics[2].metric("有效率", f"{eligibility_rate:.1f}%")
    quality_metrics[3].metric(
        "严格目标SKU",
        int((category_cleaned["sku_scope"] == "目标SKU").sum()),
    )
    quality_metrics[4].metric(
        "其他SKU待核验",
        int((category_cleaned["sku_scope"] == "其他SKU待核验").sum()),
    )

    st.caption(
        "有效评论排除了平台占位文本、重复文本、过短文本、仅含物流客服信息的文本和未识别产品属性的文本。"
    )
    st.subheader("严格型号评论覆盖")
    strict = category_cleaned[category_cleaned["product_level_eligible_strict"] == True].copy()
    if strict.empty:
        st.info("当前类别暂无来源SKU明确且通过清洗的评论。")
    else:
        coverage = strict.groupby(["platform_product_id", "brand", "product_name"], as_index=False).agg(
            有效评论=("review_row_id", "nunique")
        ).rename(columns={"platform_product_id": "SKU", "brand": "品牌", "product_name": "商品"})
        st.dataframe(coverage, use_container_width=True, hide_index=True)
        st.caption("已核验来源SKU的评论按实际商品归属统计；原采集入口仍保留。")
    catalog = load_verified_product_catalog(database_path)
    catalog = catalog[catalog["category"] == category]
    if not catalog.empty:
        with st.expander("新增型号与商品页核验来源"):
            display = catalog.rename(columns={"brand": "品牌", "product_name": "商品", "sku": "SKU", "history_points": "历史报价数", "lowest_price": "最低报价", "verified_at": "核验时间", "verified_url": "商品来源"})
            st.dataframe(display[["品牌", "商品", "SKU", "历史报价数", "最低报价", "核验时间", "商品来源"]],
                         use_container_width=True, hide_index=True,
                         column_config={"商品来源": st.column_config.LinkColumn(), "最低报价": st.column_config.NumberColumn(format="¥ %.2f")})
    st.warning("评分组定额抽样与历史展示样本分布不同。下图仅描述采集样本，不能推断总体好评率或键盘与鼠标质量差异。")
    stratified_path = analysis_dir / "review_stratified_summary.csv"
    if stratified_path.exists():
        strata = pd.read_csv(stratified_path, dtype={"platform_product_id": str})
        strata = strata[strata["category"] == category]
        with st.expander("按型号、采集日期与评分组查看"):
            st.dataframe(strata, use_container_width=True, hide_index=True)

    if category_dimensions.empty:
        st.info("当前类别尚无可用于六维评论偏好分析的样本。")
    else:
        sentiment_order = ["positive_count", "neutral_count", "mixed_count", "negative_count"]
        sentiment_labels = {"positive_count": "正向", "neutral_count": "中性", "mixed_count": "混合", "negative_count": "负向"}
        sentiment_long = category_dimensions.melt(
            id_vars=["dimension"],
            value_vars=sentiment_order,
            var_name="sentiment_type",
            value_name="sentiment_mentions",
        )
        sentiment_long["情感"] = sentiment_long["sentiment_type"].map(sentiment_labels)
        sentiment_long["偏好维度"] = sentiment_long["dimension"]

        chart_left, chart_right = st.columns(2)
        with chart_left:
            figure = px.bar(
                sentiment_long,
                x="偏好维度",
                y="sentiment_mentions",
                color="情感",
                barmode="stack",
                title=f"{CATEGORY_LABELS[category]}六维评价情感结构",
                labels={"sentiment_mentions": "方面级提及次数"},
                category_orders={"情感": ["正向", "中性", "混合", "负向"]},
                color_discrete_map={"正向": "#4C956C", "中性": "#B8B8B8", "混合": "#D9A441", "负向": "#D1495B"},
            )
            figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(figure, use_container_width=True)

        with chart_right:
            net_data = category_dimensions.sort_values("net_sentiment")
            figure = px.bar(
                net_data,
                x="net_sentiment",
                y="dimension",
                orientation="h",
                color="net_sentiment",
                color_continuous_scale="RdYlGn",
                range_color=[-1, 1],
                text_auto=".2f",
                title=f"{CATEGORY_LABELS[category]}六维净情感值",
                labels={"net_sentiment": "净情感值", "dimension": "偏好维度"},
            )
            figure.update_xaxes(range=[-1, 1])
            figure.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(figure, use_container_width=True)

        category_table = category_dimensions.rename(columns={
            "dimension": "偏好维度",
            "mention_count": "提及次数",
            "unique_review_count": "涉及评论",
            "positive_count": "正向",
            "neutral_count": "中性",
            "negative_count": "负向",
            "mixed_count": "混合",
            "net_sentiment": "净情感值",
        })
        st.dataframe(
            category_table[["偏好维度", "提及次数", "涉及评论", "正向", "中性", "混合", "负向", "净情感值"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "净情感值": st.column_config.ProgressColumn(
                    min_value=-1, max_value=1, format="%.2f"
                )
            },
        )

        weakest = category_dimensions.sort_values("net_sentiment").iloc[0]
        strongest = category_dimensions.sort_values("net_sentiment", ascending=False).iloc[0]
        st.info(
            f"当前{CATEGORY_LABELS[category]}样本中，{strongest['dimension']}净情感值最高"
            f"（{strongest['net_sentiment']:.2f}）；{weakest['dimension']}最低"
            f"（{weakest['net_sentiment']:.2f}）。结果描述当前样本，不代表平台全体消费者。"
        )

    with st.expander("查看严格SKU商品级结果与数据边界"):
        st.write(
            "严格商品级汇总只纳入来源SKU与目标SKU一致且通过清洗的评论。其他SKU仍待核验，"
            "历史样本未记录来源SKU。小样本结果只用于逐条复核，不据此发布型号质量排名。"
        )
        if category_products.empty:
            st.info("当前类别没有足够的严格目标SKU评论。")
        else:
            product_table = category_products.rename(columns={
                "brand": "品牌",
                "product_name": "商品名称",
                "dimension": "偏好维度",
                "mention_count": "提及次数",
                "positive_count": "正向",
                "neutral_count": "中性",
                "negative_count": "负向",
                "net_sentiment": "净情感值",
            })
            st.dataframe(
                product_table[["品牌", "商品名称", "偏好维度", "提及次数", "正向", "中性", "负向", "净情感值"]],
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "净情感值＝（正向提及数－负向提及数）÷该维度全部提及数，范围为－1至1。"
            f"分析生成时间：{review_analysis_summary['generated_at']}。"
        )

    download_left, download_right = st.columns(2)
    download_left.download_button(
        "下载清洗后评论",
        data=cleaned_path.read_bytes(),
        file_name=cleaned_path.name,
        mime="text/csv",
        use_container_width=True,
    )
    if dimension_mentions_path.exists():
        download_right.download_button(
            "下载方面级情感明细",
            data=dimension_mentions_path.read_bytes(),
            file_name=dimension_mentions_path.name,
            mime="text/csv",
            use_container_width=True,
        )

