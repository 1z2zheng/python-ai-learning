from pathlib import Path
import sys
import os

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esports_analysis.config import get_database_path
from esports_analysis.db import get_database_summary, initialize_database
from esports_analysis.model_registry import build_model_registry


st.set_page_config(
    page_title="电竞外设竞品分析",
    page_icon="🎮",
    layout="wide",
)

database_path = get_database_path()
initialize_database(database_path)
summary = get_database_summary(database_path)

st.title("电竞外设竞品定价与偏好可视化系统")
st.caption("查看竞品数据、参数匹配与历史报价，管理评论复核和数据质量。")
if os.getenv("PORTFOLIO_DEMO") == "1":
    st.info("演示模式：80 个合成商品、240 条价格快照，用于体验分析功能，非真实市场数据。评论与跨平台核验功能需要另行导入真实数据。")

col1, col2, col3, col4 = st.columns(4)
col1.metric("全部目录记录（含测试数据）", summary["product_count"])
col2.metric("品牌数", summary["brand_count"])
col3.metric("价格快照", summary["price_snapshot_count"])
col4.metric("参数记录", summary["spec_count"])

records, groups = build_model_registry(database_path)
research_columns = st.columns(3)
research_columns[0].metric("研究平台商品记录", len(records))
research_columns[1].metric("候选型号组", len(groups))
research_columns[2].metric("有严格评论的平台商品", int(records["严格评论数"].gt(0).sum()))
st.caption("候选型号组按型号与版本归并，不代表跨平台配置已全部核验。最终独立型号数待复核；明细及版本冲突见数据维护。")

st.subheader("项目状态")
st.info("通过左侧导航进入分析、评论复核、数据维护和验收页面。数据覆盖情况请查看系统验收。")
st.write(f"数据库：`{database_path}`")

st.subheader("功能进度")
st.markdown(
    """
- ✅ 数据概览
- ✅ 用户偏好分析
- ✅ 定价分析
- ✅ 促销分析
- ✅ 竞品对比
- ✅ 数据维护与CSV上传
- ✅ 自动验收与测试报告
- ✅ 评论复核与标签修改
- ✅ 商品证据与来源追溯
"""
)

