from pathlib import Path
import json
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.config import get_database_path
from esports_analysis.review_annotation import (
    load_annotation_data, save_annotation, annotation_history,
    DIMENSION_NAMES, SENTIMENTS, DECISIONS,
)

st.set_page_config(page_title="评论复核", page_icon="💬", layout="wide")
st.title("评论复核")
st.caption("查看AI辅助标签，逐条确认或修改，保存复核过程。")
path = ROOT / "output/review_analysis/review_ai_annotations.json"
if not path.exists():
    st.info("暂无辅助标签数据，请先生成评论标签。")
    st.stop()
database = get_database_path()
try:
    data = load_annotation_data(path, database)
except (ValueError, OSError) as exc:
    st.error(f"辅助标签读取失败：{exc}")
    st.stop()
if data.empty:
    st.info("暂无待复核评论。")
    st.stop()
if "review_save_notice" in st.session_state:
    st.success(st.session_state.pop("review_save_notice"))

metrics = st.columns(4)
metrics[0].metric("评论数", len(data))
metrics[1].metric("已确认或修改", int(data["复核状态"].isin(["已确认", "已修改"]).sum()))
metrics[2].metric("待复核", int(data["复核状态"].isin(["待人工复核", "文本已更新需复核", "需再核对"]).sum()))
metrics[3].metric("已排除", int(data["复核状态"].eq("排除").sum()))

with st.sidebar:
    st.header("评论筛选")
    reviewer = st.text_input("复核人昵称或编号", key="review_reviewer")
    status = st.selectbox("复核状态", ["全部", *sorted(data["复核状态"].unique())])
    scope = st.selectbox("SKU归属", ["全部", *sorted(data["SKU归属"].unique())])
    polarity = st.selectbox("AI整体情感", ["全部", *sorted(data["AI整体情感"].unique())])
    keyword = st.text_input("搜索正文或SKU")

filtered = data.copy()
for column, value in (("复核状态", status), ("SKU归属", scope), ("AI整体情感", polarity)):
    if value != "全部":
        filtered = filtered[filtered[column].eq(value)]
if keyword.strip():
    filtered = filtered[filtered["评论正文"].str.contains(keyword.strip(), regex=False) |
                        filtered["入口SKU"].str.contains(keyword.strip(), regex=False)]
if filtered.empty:
    st.info("当前筛选没有评论，请调整筛选条件。")
    st.stop()

tab_review, tab_summary = st.tabs(["逐条复核", "结果概览"])
with tab_review:
    chosen = st.selectbox("选择评论", filtered["评论编号"].tolist(),
                          format_func=lambda value: f"#{value}  {filtered.loc[filtered['评论编号']==value, '评论正文'].iloc[0][:42]}")
    row = filtered[filtered["评论编号"] == chosen].iloc[0].to_dict()
    st.caption(f"入口SKU {row['入口SKU']} · {row['SKU归属']} · {row['复核状态']}")
    if row.get('解析后SKU'):
        st.caption(f"解析后SKU {row['解析后SKU']} · {row.get('来源核验', '')} · {row.get('解析后商品', '')}")
    st.subheader("评论正文")
    st.write(row["评论正文"])
    st.caption("AI判断依据：" + row["AI判断依据及疑点"])
    st.dataframe(pd.DataFrame({"维度": DIMENSION_NAMES,
                              "AI情感": [row["AI" + d] for d in DIMENSION_NAMES]}),
                 use_container_width=True, hide_index=True)
    has_saved_aspects = bool(row.get("人工复核维度及情感"))
    existing = json.loads(row["人工复核维度及情感"]) if has_saved_aspects else {}
    with st.form(f"review_form_{row['入口SKU']}_{row['原评论哈希ID']}"):
        current = row.get("人工复核整体情感") or row["AI整体情感"]
        if current not in SENTIMENTS:
            current='中性'
        overall = st.selectbox("复核后的整体情感", SENTIMENTS, index=SENTIMENTS.index(current))
        aspect_values = {}
        columns = st.columns(3)
        for index, dimension in enumerate(DIMENSION_NAMES):
            options = ["未涉及", *SENTIMENTS]
            value = existing.get(dimension, "未涉及") if has_saved_aspects else row["AI"+dimension]
            aspect_values[dimension] = columns[index % 3].selectbox(dimension, options, index=options.index(value))
        decision = st.selectbox("保存状态", DECISIONS,
                               index=DECISIONS.index(row["复核状态"]) if row["复核状态"] in DECISIONS else (1 if row['AI整体情感']=='未标注' else 0),
                               help="标签与AI一致时选已确认；修改或首次填写标签时选已修改；其他情况请记录原因。")
        note = st.text_area("复核备注", value=row.get("人工复核备注", ""))
        submit = st.form_submit_button("保存本条复核", use_container_width=True)
    if submit:
        try:
            aspects = {d: s for d, s in aspect_values.items() if s != "未涉及"}
            save_annotation(database, row, reviewer, overall, aspects, decision, note)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["review_save_notice"] = f"评论#{chosen}复核已保存。"
            st.rerun()
    with st.expander("本条复核历史"):
        history = annotation_history(database, str(row["入口SKU"]), row["原评论哈希ID"])
        if history.empty:
            st.caption("尚无逐条保存记录。")
        else:
            st.dataframe(history, use_container_width=True, hide_index=True)

with tab_summary:
    st.caption("AI辅助标签与已保存人工复核标签分别展示，空白复核不视为确认。")
    source = st.radio("统计来源", ["AI辅助标签", "已确认或修改的复核标签"], horizontal=True)
    if source == "AI辅助标签":
        sample = filtered
        counts = sample["AI整体情感"].value_counts().rename_axis("情感").reset_index(name="评论数")
    else:
        sample = filtered[filtered["复核状态"].isin(["已确认", "已修改"])]
        counts = sample["人工复核整体情感"].value_counts().rename_axis("情感").reset_index(name="评论数")
    if sample.empty:
        st.info("当前筛选尚无已保存复核标签。")
    else:
        figure = px.bar(counts, x="情感", y="评论数", color="情感", title=f"{source}的整体情感分布")
        st.plotly_chart(figure, use_container_width=True)
    st.dataframe(filtered[["评论编号", "入口SKU", "复核状态", "评论正文", "AI整体情感", "人工复核整体情感", "复核人", "复核日期"]],
                 use_container_width=True, hide_index=True)
    st.download_button("下载当前筛选的标签与复核记录", filtered.to_csv(index=False).encode("utf-8-sig"),
                       file_name="评论复核记录.csv", mime="text/csv")

st.caption("页面保存的是AI辅助人工复核记录，不自动转入独立双人标注或模型准确率统计。")
