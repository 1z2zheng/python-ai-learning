from pathlib import Path
import sys
import json

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esports_analysis.config import get_database_path
from esports_analysis.db import connect
from esports_analysis.overview import CATEGORY_LABELS, CONNECTION_LABELS, PLATFORM_LABELS
from esports_analysis.product_evidence import load_evidence_data, product_evidence

st.set_page_config(page_title="商品证据", page_icon="🔎", layout="wide")
st.title("商品证据")
st.caption("按具体SKU查看规格、历史报价和来源明确的评论，逐项追溯证据。")

@st.cache_data(ttl=120)
def get_bundle(path, modified_at):
    return load_evidence_data(path)

database = get_database_path()
with connect(database) as db:
    has_public_specs = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='public_spec_research_samples'").fetchone()
    public_specs = pd.read_sql_query("SELECT * FROM public_spec_research_samples ORDER BY brand,model_name", db) if has_public_specs else pd.DataFrame()
if not public_specs.empty:
    with st.expander(f"公开历史索引规格补充样本：{len(public_specs)}个型号"):
        st.warning("此分组仅补充规格研究，不属于实时价格或严格评论样本。索引存在时间滞后，不能用于当前在售判断和平台价差排名。")
        for sample in public_specs.to_dict("records"):
            st.markdown(f"**{sample['model_name']}** · [来源商品页]({sample['source_url']})")
            st.caption(f"SKU {sample['source_sku']} · 检索日期 {sample['retrieved_date']} · {sample['freshness_note']}")
            attributes = json.loads(sample['attributes_json'])
            st.dataframe(pd.DataFrame([{"参数": key, "来源原值": value} for key,value in attributes.items()]),
                         use_container_width=True, hide_index=True)
bundle = get_bundle(str(database), database.stat().st_mtime if database.exists() else 0)
products = bundle["products"]
if products.empty:
    st.info("暂无真实历史报价或来源明确的研究商品。")
    st.stop()
with st.sidebar:
    st.header("商品筛选")
    category = st.selectbox("品类", ["全部", *CATEGORY_LABELS], format_func=lambda c: CATEGORY_LABELS.get(c,c))
    platform = st.selectbox("平台", ["全部", *sorted(products["platform"].unique())], format_func=lambda p: PLATFORM_LABELS.get(p,p))
    only_strict = st.checkbox("仅看有严格SKU有效评论的商品", value=True)
    brand = st.selectbox("品牌", ["全部", *sorted(products["brand"].unique())])
    search = st.text_input("搜索型号或SKU")

filtered = products.copy()
for column, value in (("category", category), ("platform", platform), ("brand", brand)):
    if value != "全部":
        filtered = filtered[filtered[column].eq(value)]
if only_strict:
    filtered = filtered[filtered["strict_review_count"].gt(0)]
if search.strip():
    filtered = filtered[filtered["product_name"].str.contains(search.strip(), regex=False)
                        | filtered["platform_product_id"].str.contains(search.strip(), regex=False)]
if filtered.empty:
    st.info("筛选结果为空。可以关闭严格评论筛选，查看仅有报价或规格证据的商品。")
    st.stop()
names = {int(r.product_id): f"{r.product_name} ｜ SKU {r.platform_product_id} ｜ 严格评论 {r.strict_review_count}条"
         for r in filtered.itertuples()}
chosen = st.selectbox("选择商品", list(names), format_func=names.get)
result = product_evidence(bundle, chosen, database)
product = result["product"]
history, reviews = result["history"], result["reviews"]
summary = result["price_summary"]

st.subheader(product["product_name"])
st.caption(f"{PLATFORM_LABELS.get(product['platform'], product['platform'])} · {CATEGORY_LABELS[product['category']]} · SKU {product['platform_product_id']}")
st.markdown(f"[商品来源]({product['product_url']})")
metrics = st.columns(4)
metrics[0].metric("严格SKU有效评论", len(reviews))
metrics[1].metric("历史报价点", len(history))
metrics[2].metric("最新历史报价", f"¥{summary.iloc[0]['latest_price']:.2f}" if not summary.empty else "暂无")
metrics[3].metric("历史最低报价", f"¥{summary.iloc[0]['lowest_price']:.2f}" if not summary.empty else "暂无")
if len(reviews) < 20:
    st.warning(f"当前仅有{len(reviews)}条严格SKU有效评论，页面用于探索与原文核对，不形成质量排名。")
else:
    st.info("评论来自非概率抽样；样本增多仍不能直接解释为总体满意度。")

tab_specs, tab_prices, tab_reviews, tab_sources = st.tabs(["型号与规格", "历史价格", "评论证据", "来源与缺口"])
with tab_specs:
    specs = {"连接方式":"connection_type", "重量 g":"weight_g", "回报率 Hz":"polling_rate_hz",
             "RGB灯效":"rgb", "保修期 月":"warranty_months"}
    if product["category"] == "mouse":
        specs.update({"最大DPI":"dpi_max", "传感器":"sensor_model", "按键数":"button_count", "惯用手":"handedness"})
    else:
        specs.update({"轴体类型":"switch_type", "轴体名称":"switch_name", "配列":"layout_size",
                      "键数":"key_count", "触发行程 mm":"actuation_mm", "热插拔":"hot_swappable"})
    values = []
    for name, key in specs.items():
        value = product.get(key)
        if pd.isna(value):
            shown = "未提供"
        elif key == "connection_type":
            shown = CONNECTION_LABELS.get(value, value)
        elif key in ("rgb", "hot_swappable"):
            shown = "支持" if value else "不支持"
        else:
            shown = str(value)
        values.append({"参数":name,"标准化值":shown})
    st.dataframe(pd.DataFrame(values), use_container_width=True, hide_index=True)
    st.caption("区间参数保留在来源原值中，不把区间上限当作精确规格。缺失字段保持未提供。")
    with st.expander("商品页原始参数"):
        if result["raw_attributes"].empty:
            st.info("暂无结构化商品页参数，查看来源记录核对规格。")
        else:
            st.dataframe(result["raw_attributes"], use_container_width=True, hide_index=True)

with tab_prices:
    if history.empty:
        st.info("暂无可用历史报价。")
    else:
        chart = history.sort_values("collected_at").copy()
        chart["报价日期"] = chart["collected_at"].dt.tz_convert("Asia/Shanghai").dt.date
        figure = px.line(chart, x="报价日期", y="effective_price", title="历史报价趋势", labels={"effective_price":"报价 元"})
        st.plotly_chart(figure, use_container_width=True)
        row = summary.iloc[0]
        st.caption(f"观察期 {row['date_start'].date()}至{row['date_end'].date()}；报价波动率 {row['volatility_rate']:.2f}%；末次报价较历史最低价高 {row['lowest_gap_rate']:.2f}%。")
        st.caption("慢慢买记录的是第三方报价，优惠资格和库存状态可能不同；历史最低价不保证当前可购买。")
        st.download_button("下载本商品历史报价", chart.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{product['platform_product_id']}_历史报价.csv", mime="text/csv")

with tab_reviews:
    if reviews.empty:
        st.info("暂无严格SKU有效评论。其他来源或未知SKU评论未混入本商品。")
    else:
        dimension = result["dimensions"]
        long = dimension.melt(id_vars="维度", value_vars=["正向", "中性", "混合", "负向"], var_name="规则情感", value_name="提及次数")
        figure = px.bar(long, x="维度", y="提及次数", color="规则情感", barmode="stack", title="严格SKU评论的六维关注结构",
                        color_discrete_map={"正向":"#4C956C","负向":"#D1495B","中性":"#B8B8B8","混合":"#D9A441"})
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(dimension, use_container_width=True, hide_index=True,
                     column_config={"净情感值":st.column_config.NumberColumn(format="%.3f")})
        st.caption("情感由文本规则推断，未经独立标注验证。提及率与情感不转化为消费者权重或质量分数。")
        dimension_filter = st.selectbox("按关注属性查看原文", ["全部", *dimension["维度"].tolist()])
        sentiment_filter = st.selectbox("按文本情感查看", ["全部", "正向", "负向", "中性", "混合"])
        if dimension_filter == "全部":
            selected_reviews = reviews.copy()
            if sentiment_filter != "全部":
                selected_reviews = selected_reviews[selected_reviews["sentiment"].eq(sentiment_filter)]
        else:
            excerpts = result["excerpts"]
            subset = excerpts[excerpts["维度"].eq(dimension_filter)]
            if sentiment_filter != "全部":
                subset = subset[subset["规则情感"].eq(sentiment_filter)]
            selected_reviews = reviews[reviews["content"].isin(subset["正文"])]
        if selected_reviews.empty:
            st.info("该属性和情感组合暂无评论。")
        for review in selected_reviews.head(12).to_dict("records"):
            with st.expander(f"{review['reviewed_at']} · 来源SKU {review['source_product_id']} · 整体规则情感 {review['sentiment']}"):
                st.write(review["content"])
                st.caption(f"原采集入口 {review['requested_sku']} · {review['identity_resolution']}")
        st.caption("页面最多展示前12条匹配原文，下载可查看本商品全部严格评论。")
        st.download_button("下载本商品严格评论", reviews.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{product['platform_product_id']}_严格评论.csv", mime="text/csv")

with tab_sources:
    verification = result["verification"]
    if verification.empty:
        st.write("评论来源SKU与商品入口一致。该商品尚未建档详细的商品页编号和版本核验记录。")
    else:
        v = verification.iloc[0]
        st.write(f"商品页编号和版本标题已核验，核验时间 {v['verified_at']}。")
        st.markdown(f"[核验商品页]({v['verified_url']})")
    st.write(f"规格记录更新时间：{result['spec_updated_at'] or '未提供'}")
    if result["spec_source"]:
        st.code(result["spec_source"], language=None)
    missing = [r["参数"] for r in values if r["标准化值"] == "未提供"]
    if missing:
        st.write("待补规格：" + "、".join(missing))
    st.write("评论仅采用来源SKU一致并通过去重及质量筛选的文本。原采集入口保留，来源不明或其他SKU不会自动归入当前商品。")
