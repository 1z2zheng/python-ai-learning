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
from esports_analysis.overview import CATEGORY_LABELS, PLATFORM_LABELS, load_overview_data
from esports_analysis.preference import SCORE_VERSION, calculate_dimension_scores
from esports_analysis.pair_verification import load_pair_verifications, save_pair_verification
from esports_analysis.pricing import (
    PARAMETER_OPTIONS,
    assign_price_bands,
    build_brand_history_summary,
    build_cross_platform_comparison,
    build_positioning_data,
    build_price_band_summary,
    build_product_history_summary,
    load_price_history,
    summarize_price_history,
)


st.set_page_config(page_title="定价分析", page_icon="💹", layout="wide")


@st.cache_data(ttl=300)
def get_pricing_data(database_path: str, modified_at: float):
    del modified_at
    latest = load_overview_data(database_path)
    priced_latest = latest.dropna(subset=["current_price"]).copy()
    return calculate_dimension_scores(priced_latest), load_price_history(database_path)


def reset_pricing_filters() -> None:
    for key in (
        "pricing_platforms",
        "pricing_brands",
        "pricing_parameter",
        "pricing_products",
        "pricing_sources",
    ):
        st.session_state.pop(key, None)


database_path = get_database_path()
database_mtime = database_path.stat().st_mtime if database_path.exists() else 0.0
latest_data, history = get_pricing_data(str(database_path), database_mtime)

st.title("竞品定价分析")
st.caption("分析品牌价格带、参数与价格关系、历史趋势以及性能价格定位。")

if latest_data.empty:
    st.warning("数据库中还没有商品数据，请先导入 CSV。")
    st.stop()

with st.sidebar:
    st.header("筛选条件")
    category = st.radio(
        "产品类别",
        options=list(CATEGORY_LABELS),
        format_func=CATEGORY_LABELS.get,
        key="pricing_category",
        horizontal=True,
        on_change=reset_pricing_filters,
    )
    category_data = latest_data[latest_data["category"] == category].copy()

    platform_options = sorted(category_data["platform"].dropna().unique().tolist())
    platforms = st.multiselect(
        "平台",
        platform_options,
        default=platform_options,
        format_func=lambda value: PLATFORM_LABELS.get(value, value),
        key="pricing_platforms",
    )
    platform_data = category_data[category_data["platform"].isin(platforms)]

    brand_options = sorted(platform_data["brand"].dropna().unique().tolist())
    brands = st.multiselect("品牌", brand_options, default=brand_options, key="pricing_brands")

    history_scope = history[
        (history["category"] == category)
        & history["platform"].isin(platforms)
        & history["brand"].isin(brands)
    ]
    source_options = sorted(history_scope["price_source"].dropna().unique().tolist())
    default_sources = ["manmanbuy"] if "manmanbuy" in source_options else source_options
    sources = st.multiselect(
        "历史价格来源",
        source_options,
        default=default_sources,
        key="pricing_sources",
    )

    parameter = st.selectbox(
        "价格关联参数",
        options=list(PARAMETER_OPTIONS[category]),
        format_func=PARAMETER_OPTIONS[category].get,
        key="pricing_parameter",
    )

filtered = platform_data[platform_data["brand"].isin(brands)].copy()
if filtered.empty:
    st.warning("当前筛选条件下没有商品，请调整平台或品牌。")
    st.stop()

filtered_history = history[
    (history["category"] == category)
    & history["platform"].isin(platforms)
    & history["brand"].isin(brands)
    & history["price_source"].isin(sources)
].copy()
history_summary = summarize_price_history(filtered_history)
product_history = build_product_history_summary(filtered_history)
brand_history = build_brand_history_summary(product_history)

pairs_path = ROOT / "data" / "manmanbuy" / "cross_platform_pairs.csv"
pairs = pd.read_csv(pairs_path, dtype=str) if pairs_path.exists() else pd.DataFrame()
cross_platform = build_cross_platform_comparison(
    history[history["price_source"] == "manmanbuy"], pairs, load_pair_verifications(database_path)
)
if not cross_platform.empty:
    cross_platform = cross_platform[cross_platform["category"] == category]

metric_columns = st.columns(6)
metric_columns[0].metric("目录商品", filtered["product_id"].nunique())
metric_columns[1].metric("有历史价商品", history_summary["product_count"])
metric_columns[2].metric("价格快照", history_summary["snapshot_count"])
metric_columns[3].metric("当前平均价", f"¥{filtered['current_price'].mean():,.0f}")
lowest_text = (
    f"¥{history_summary['lowest_price']:,.0f}"
    if history_summary["lowest_price"] is not None
    else "—"
)
metric_columns[4].metric("观察期最低价", lowest_text)
volatility_text = (
    f"{product_history['volatility_rate'].median():.1f}%"
    if not product_history.empty
    else "—"
)
metric_columns[5].metric("价格波动率中位数", volatility_text)

if history_summary["date_start"] is not None:
    date_start = history_summary["date_start"].strftime("%Y-%m-%d")
    date_end = history_summary["date_end"].strftime("%Y-%m-%d")
    st.caption(f"价格观察期：{date_start} 至 {date_end} ｜ 性能评分版本：{SCORE_VERSION}")
else:
    st.warning("型号配对只构成候选关系。配置、店铺、优惠资格和报价日期核验完成前，价格差不能解释为平台优势。")
    st.caption(f"当前筛选条件下没有历史价格 ｜ 性能评分版本：{SCORE_VERSION}")

chart_left, chart_right = st.columns(2)
with chart_left:
    banded = assign_price_bands(filtered, category)
    band_counts = (
        banded.groupby(["price_band", "brand"], observed=False)
        .agg(商品数=("product_id", "nunique"))
        .reset_index()
    )
    figure = px.bar(
        band_counts,
        x="price_band",
        y="商品数",
        color="brand",
        barmode="stack",
        title="品牌价格带分布",
        labels={"price_band": "价格带", "brand": "品牌"},
    )
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

with chart_right:
    parameter_data = filtered.dropna(subset=[parameter, "current_price"])
    if parameter_data.empty:
        st.info(f"当前商品缺少{PARAMETER_OPTIONS[category][parameter]}数据，无法生成关联图。")
    else:
        figure = px.scatter(
            parameter_data,
            x=parameter,
            y="current_price",
            color="brand",
            hover_name="product_name",
            size="current_price",
            title=f"{PARAMETER_OPTIONS[category][parameter]}与价格关系",
            labels={parameter: PARAMETER_OPTIONS[category][parameter], "current_price": "当前价格 元", "brand": "品牌"},
        )
        figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(figure, use_container_width=True)

chart_left, chart_right = st.columns(2)
with chart_left:
    product_options = sorted(filtered_history["product_name"].dropna().unique().tolist())
    selected_products = st.multiselect(
        "趋势商品 最多选择5款",
        product_options,
        default=product_options[: min(3, len(product_options))],
        max_selections=5,
        key="pricing_products",
    )
    trend_data = filtered_history[filtered_history["product_name"].isin(selected_products)]
    if trend_data.empty:
        st.info("请选择至少一款商品查看历史价格。")
    else:
        figure = px.line(
            trend_data,
            x="collected_date",
            y="effective_price",
            color="product_name",
            markers=True,
            title="商品历史价格趋势",
            labels={"collected_date": "日期", "effective_price": "有效价格 元", "product_name": "商品"},
            hover_data=["brand", "promotion_text"],
        )
        figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(figure, use_container_width=True)

with chart_right:
    positioning = build_positioning_data(filtered)
    figure = px.scatter(
        positioning,
        x="current_price",
        y="performance_score",
        color="positioning_group",
        symbol="brand",
        hover_name="product_name",
        title="性能价格定位图",
        labels={
            "current_price": "当前价格 元",
            "performance_score": "综合性能分",
            "positioning_group": "定位",
            "brand": "品牌",
        },
        range_y=[0, 100],
    )
    figure.add_vline(x=positioning["price_median"].iloc[0], line_dash="dash", line_color="#888")
    figure.add_hline(y=positioning["performance_median"].iloc[0], line_dash="dash", line_color="#888")
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)

st.subheader("真实历史价格研究指标")
if product_history.empty:
    st.info("当前筛选条件下没有可计算的历史价格数据。")
else:
    brand_chart, product_chart = st.columns(2)
    with brand_chart:
        figure = px.scatter(
            brand_history,
            x="average_latest_price",
            y="average_volatility_rate",
            size="snapshot_count",
            color="brand",
            hover_name="brand",
            title="品牌价格水平与波动率",
            labels={
                "average_latest_price": "商品最新均价 元",
                "average_volatility_rate": "平均波动率 %",
                "snapshot_count": "历史价格点",
                "brand": "品牌",
            },
        )
        figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(figure, use_container_width=True)

    with product_chart:
        saving_rank = product_history.nlargest(10, "potential_saving").sort_values(
            "potential_saving"
        )
        figure = px.bar(
            saving_rank,
            x="potential_saving",
            y="product_name",
            color="brand",
            orientation="h",
            title="历史末次价与历史最低价的差额",
            labels={
                "potential_saving": "潜在节省 元",
                "product_name": "商品",
                "brand": "品牌",
            },
        )
        figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(figure, use_container_width=True)

    display_summary = product_history.rename(
        columns={
            "brand": "品牌",
            "product_name": "商品名称",
            "snapshot_count": "价格点",
            "date_start": "开始日期",
            "date_end": "结束日期",
            "average_price": "历史均价",
            "lowest_price": "历史最低价",
            "highest_price": "历史最高价",
                "latest_price": "历史末次价",
            "volatility_rate": "波动率",
            "lowest_gap_rate": "距最低价涨幅",
            "potential_saving": "潜在节省",
        }
    )
    st.dataframe(
        display_summary[
            [
                "品牌",
                "商品名称",
                "价格点",
                "开始日期",
                "结束日期",
                "历史均价",
                "历史最低价",
                "历史最高价",
                "历史末次价",
                "波动率",
                "距最低价涨幅",
                "潜在节省",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "历史均价": st.column_config.NumberColumn(format="¥ %.2f"),
            "历史最低价": st.column_config.NumberColumn(format="¥ %.2f"),
            "历史最高价": st.column_config.NumberColumn(format="¥ %.2f"),
            "历史末次价": st.column_config.NumberColumn(format="¥ %.2f"),
            "潜在节省": st.column_config.NumberColumn(format="¥ %.2f"),
            "波动率": st.column_config.NumberColumn(format="%.2f%%"),
            "距最低价涨幅": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    st.caption("波动率为价格标准差÷历史均价；潜在节省为历史末次价格与历史最低价之差，仅作研究参考。")

st.subheader("京东与天猫同型号价格对比")
if cross_platform.empty:
    st.info("当前类别暂无已配对的京东—天猫同型号历史价格。")
else:
    st.warning("候选商品并非已确认同配置。核验记录只绑定当前SKU、报价日期与价格；新报价出现后需要重新核验。")
    with st.expander("逐组核验与保存", expanded=False):
        pair_key = st.selectbox("选择候选组", cross_platform["pair_key"].tolist(),
                                format_func=lambda key: cross_platform.set_index("pair_key").loc[key, "model"])
        chosen = cross_platform[cross_platform["pair_key"].eq(pair_key)].iloc[0]
        st.markdown(f"[京东商品页](https://item.jd.com/{chosen['jd_sku']}.html) · [天猫商品页](https://detail.tmall.com/item.htm?id={chosen['tmall_item_id']})")
        st.write(f"当前历史末次报价：京东 ¥{chosen['jd_latest_price']:.2f}，天猫 ¥{chosen['tmall_latest_price']:.2f}")
        previous = load_pair_verifications(database_path).get(pair_key, {})
        if previous:
            st.write("最近保存的核验记录", previous)
        with st.form(f"verify_{pair_key}"):
            status = st.selectbox("核验结论", ["pending", "different", "verified"],
                                  format_func={"pending": "待核验", "different": "配置不同", "verified": "已核验：配置和价格条件可比"}.get)
            reviewer = st.text_input("核验人")
            jd_variant = st.text_input("京东所选配置：型号、颜色、轴体、连接、套装")
            tm_variant = st.text_input("天猫所选配置：型号、颜色、轴体、连接、套装")
            config_evidence = st.text_area("配置核验依据：来源链接、所选SKU与原文")
            shop_evidence = st.text_area("店铺依据：两侧店铺名称及类型")
            offer_evidence = st.text_area("优惠核验依据：对应报价日期、券、会员、补贴资格、运费及价格口径")
            note = st.text_area("核验说明：一致项、差异项或待确认问题")
            submitted = st.form_submit_button("保存核验记录")
        if submitted:
            snapshot = chosen.to_dict()
            snapshot["comparison_date"] = str(chosen["comparison_date"].date()) if pd.notna(chosen["comparison_date"]) else None
            try:
                save_pair_verification(database_path, snapshot, dict(status=status, verified_by=reviewer,
                    jd_variant=jd_variant, tmall_variant=tm_variant, configuration_evidence=config_evidence,
                    shop_evidence=shop_evidence, offer_evidence=offer_evidence, note=note))
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun()
    eligible_comparison = cross_platform[cross_platform["comparison_eligible"]].copy()
    cheaper_counts = eligible_comparison["cheaper_platform"].value_counts().to_dict()
    compare_metrics = st.columns(4)
    compare_metrics[0].metric("核验可比 / 候选组", f"{len(eligible_comparison)} / {len(cross_platform)}")
    compare_metrics[1].metric("京东末次价更低", cheaper_counts.get("京东", 0))
    compare_metrics[2].metric("天猫末次价更低", cheaper_counts.get("天猫", 0))
    compare_metrics[3].metric("两平台历史价点", int(
        cross_platform["jd_snapshot_count"].sum()
        + cross_platform["tmall_snapshot_count"].sum()
    ))

    compare_chart = eligible_comparison.melt(
        id_vars=["brand", "model"],
        value_vars=["jd_latest_price", "tmall_latest_price"],
        var_name="平台",
        value_name="历史末次价",
    )
    compare_chart["平台"] = compare_chart["平台"].map(
        {"jd_latest_price": "京东", "tmall_latest_price": "天猫"}
    )
    figure = px.bar(
        compare_chart,
        x="model",
        y="历史末次价",
        color="平台",
        barmode="group",
        hover_data=["brand"],
        title="同型号商品历史末次价对比",
        labels={"model": "型号", "历史末次价": "价格 元"},
    )
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    if eligible_comparison.empty:
        st.info("尚无满足核验及报价条件的候选组，不生成平台价差排名。下表报价仅供核验参考。")
    else:
        st.plotly_chart(figure, use_container_width=True)

    comparison_table = cross_platform.rename(columns={
        "brand": "品牌",
        "model": "同型号商品",
        "jd_snapshot_count": "京东价格点",
        "tmall_snapshot_count": "天猫价格点",
        "overlap_start": "共同观察开始",
        "overlap_end": "共同观察结束",
        "jd_latest_price": "京东末次价",
        "tmall_latest_price": "天猫末次价",
        "latest_gap": "天猫减京东",
        "cheaper_platform": "末次价较低平台",
        "jd_lowest_price": "京东历史最低价",
        "tmall_lowest_price": "天猫历史最低价",
    })
    st.dataframe(
        comparison_table[[
            "品牌", "同型号商品", "京东价格点", "天猫价格点",
            "共同观察开始", "共同观察结束", "京东末次价", "天猫末次价",
            "天猫减京东", "末次价较低平台", "京东历史最低价", "天猫历史最低价",
        ]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "京东末次价": st.column_config.NumberColumn(format="¥ %.2f"),
            "天猫末次价": st.column_config.NumberColumn(format="¥ %.2f"),
            "天猫减京东": st.column_config.NumberColumn(format="¥ %.2f"),
            "京东历史最低价": st.column_config.NumberColumn(format="¥ %.2f"),
            "天猫历史最低价": st.column_config.NumberColumn(format="¥ %.2f"),
        },
    )
    st.dataframe(cross_platform[["model", "verification_status", "comparison_eligible", "risk_note"]], use_container_width=True, hide_index=True)
    st.caption("价格为第三方记录的报价；候选配对保留供核验，只有comparison_eligible为真的记录用于平台优势计数。")

st.subheader("价格带统计")
band_summary = build_price_band_summary(filtered, category)
st.dataframe(
    band_summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "平均价格": st.column_config.NumberColumn(format="¥ %.2f"),
        "商品占比": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
    },
)
st.caption("当前示例数据没有可靠销量字段，因此价格带表仅展示商品数量和平均价格，不推断利润甜区或销售额贡献。")

