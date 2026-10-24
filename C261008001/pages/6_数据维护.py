from pathlib import Path
import sys
import tempfile

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esports_analysis.config import get_database_path
from esports_analysis.db import get_database_summary
from esports_analysis.model_registry import build_model_registry
from esports_analysis.importer import import_products_csv, validate_products_csv
from esports_analysis.maintenance import (
    build_database_quality_summary,
    build_research_collection_tasks,
    load_recent_collection_jobs,
)


st.set_page_config(page_title="数据维护", page_icon="🧰", layout="wide")


def save_upload_temporarily(content: bytes) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="esports_upload_", suffix=".csv", delete=False)
    try:
        handle.write(content)
        return Path(handle.name)
    finally:
        handle.close()


database_path = get_database_path()
template_path = ROOT / "data" / "seed" / "products_template.csv"

st.title("数据维护")
st.caption("下载标准模板、校验并导入 CSV，查看导入任务和字段质量。")

summary = get_database_summary(database_path)
metric_columns = st.columns(4)
metric_columns[0].metric("商品数", summary["product_count"])
metric_columns[1].metric("品牌数", summary["brand_count"])
metric_columns[2].metric("价格快照", summary["price_snapshot_count"])
metric_columns[3].metric("参数记录", summary["spec_count"])

st.subheader("型号归并与独立样本口径")
model_records, model_groups = build_model_registry(database_path)
model_metrics = st.columns(4)
model_metrics[0].metric("研究平台商品记录", len(model_records))
model_metrics[1].metric("候选型号组", len(model_groups))
model_metrics[2].metric("跨平台待核验记录", int(model_records["归属状态"].eq("跨平台配置待核验").sum()))
model_metrics[3].metric("版本疑点拆组记录", int(model_records["版本风险"].ne("").sum()))
st.caption("平台记录不等于独立型号。候选组按SKU映射，保留代际、有线无线、专业版和配列差异；颜色或套装不单独增加型号。跨平台候选尚未全部核验，因此候选组数不能作为最终确认的独立型号数，也不能据此宣称样本统计独立。")
with st.expander("查看型号归并明细和每款证据覆盖"):
    if model_records.empty:
        st.info("暂无符合范围的商品记录。")
    else:
        st.dataframe(model_groups, use_container_width=True, hide_index=True)
        st.dataframe(model_records, use_container_width=True, hide_index=True,
                     column_config={"商品来源": st.column_config.LinkColumn("商品来源")})
        st.download_button("导出型号归属明细", model_records.to_csv(index=False).encode("utf-8-sig"),
                           file_name="research_model_registry.csv", mime="text/csv")

st.subheader("研究数据补充清单")
st.caption("仅检查有真实报价、严格评论或商品页核验证据的京东、天猫商品；排除演示数据。清单自动计算，不发起采集。")
review_target = st.number_input("每款严格评论补采目标", min_value=1, max_value=200, value=20, step=1)
tasks = build_research_collection_tasks(database_path, int(review_target))
if tasks.empty:
    st.info("当前没有待补任务，或尚未导入符合范围的研究证据；这不代表系统验收全部达标。")
else:
    counters = st.columns(3)
    counters[0].metric("待补商品", tasks["商品ID"].nunique())
    counters[1].metric("待补任务", len(tasks))
    counters[2].metric("高优先级任务", int(tasks["优先级"].eq("高").sum()))
    filters = st.columns(3)
    category = filters[0].selectbox("任务类别", ["全部"] + sorted(tasks["类别"].unique()))
    kind = filters[1].selectbox("任务类型", ["全部"] + sorted(tasks["任务类型"].unique()))
    priority = filters[2].selectbox("任务优先级", ["全部", "高", "中"])
    selected = tasks.copy()
    for column, value in (("类别", category), ("任务类型", kind), ("优先级", priority)):
        if value != "全部":
            selected = selected[selected[column].eq(value)]
    if selected.empty:
        st.info("当前筛选没有待补任务。")
    else:
        st.dataframe(selected, use_container_width=True, hide_index=True,
                     column_config={"商品链接": st.column_config.LinkColumn("商品来源")})
        st.download_button("导出当前补充清单", selected.to_csv(index=False).encode("utf-8-sig"),
                           file_name="research_collection_tasks.csv", mime="text/csv")
    st.caption("高优先级：缺少型号核验、无严格评论或无报价日期。中优先级：参数缺失、评论未达补采目标或不足3个报价日期。数量目标不代表样本具有统计代表性；遇到频控应停止采集。")

st.subheader("CSV导入")
st.download_button(
    "下载CSV模板",
    data=template_path.read_bytes(),
    file_name="products_template.csv",
    mime="text/csv",
)
uploaded = st.file_uploader("上传商品CSV", type=["csv"], accept_multiple_files=False)

if uploaded is not None:
    temporary_path = save_upload_temporarily(uploaded.getvalue())
    try:
        try:
            validation = validate_products_csv(temporary_path)
        except (ValueError, FileNotFoundError) as exc:
            st.error(f"文件结构校验失败：{exc}")
        else:
            validation_columns = st.columns(3)
            validation_columns[0].metric("读取行数", validation.row_count)
            validation_columns[1].metric("有效行", validation.valid_count)
            validation_columns[2].metric("错误行", validation.invalid_count)

            if validation.normalized_rows:
                st.markdown("**标准化预览 前20行**")
                preview = pd.DataFrame(validation.normalized_rows)
                preview_columns = [
                    "platform", "platform_product_id", "category", "brand",
                    "product_name", "sale_price", "collected_at",
                ]
                st.dataframe(preview[preview_columns], use_container_width=True, hide_index=True)

            if validation.errors:
                st.error("文件包含错误行，必须全部修正后才能写入数据库。")
                st.code("\n".join(validation.errors[:50]))
            elif validation.row_count == 0:
                st.warning("文件没有数据行。")
            else:
                st.success("校验通过。点击确认后才会写入正式数据库。")
                if st.button("确认导入", type="primary"):
                    result = import_products_csv(temporary_path, database_path)
                    st.cache_data.clear()
                    st.success(
                        f"导入完成：读取 {result.fetched_count}，新增 {result.inserted_count}，"
                        f"更新 {result.updated_count}，失败 {result.failed_count}。"
                    )
                    st.rerun()
    finally:
        temporary_path.unlink(missing_ok=True)

st.subheader("最近导入任务")
jobs = load_recent_collection_jobs(database_path)
if jobs.empty:
    st.info("暂无导入任务记录。")
else:
    st.dataframe(jobs, use_container_width=True, hide_index=True)

st.subheader("字段质量")
quality = build_database_quality_summary(database_path)
st.dataframe(
    quality,
    use_container_width=True,
    hide_index=True,
    column_config={
        "完整率": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")
    },
)

st.info("名称中包含“演示”的商品以及标注为“功能测试数据”的记录仅用于功能验证，不应进入论文市场结论。")

