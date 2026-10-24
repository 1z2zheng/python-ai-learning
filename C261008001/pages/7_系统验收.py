from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esports_analysis.acceptance import (
    checks_to_frame,
    generate_acceptance_markdown,
    run_acceptance_checks,
)
from esports_analysis.config import get_database_path


st.set_page_config(page_title="系统验收", page_icon="✅", layout="wide")


@st.cache_data(ttl=120)
def get_acceptance(database_path: str, modified_at: float):
    del modified_at
    return run_acceptance_checks(database_path)


database_path = get_database_path()
database_mtime = database_path.stat().st_mtime if database_path.exists() else 0.0
checks, timings = get_acceptance(str(database_path), database_mtime)
frame = checks_to_frame(checks)

st.title("系统自动验收")
st.caption("分别检查系统功能、性能和论文研究数据是否达到开题报告要求。")

passed = int(frame["通过"].sum())
failed = len(frame) - passed
functional = frame[frame["类别"] != "论文研究数据"]
research = frame[frame["类别"] == "论文研究数据"]
functional_ready = bool(functional["通过"].all())
research_ready = bool(research["通过"].all())

metric_columns = st.columns(4)
metric_columns[0].metric("检查项", len(frame))
metric_columns[1].metric("通过", passed)
metric_columns[2].metric("未通过", failed)
metric_columns[3].metric("系统状态", "通过" if functional_ready else "未通过")

if functional_ready:
    st.success("系统功能、数据质量和性能验收通过。")
else:
    st.error("系统验收存在未通过项，请先修复。")

if research_ready:
    st.success("论文研究数据已达到当前量化要求。")
else:
    st.warning("论文研究数据尚未就绪。演示数据不能用于市场分析结论。")

display = frame.copy()
display["结果"] = display["通过"].map({True: "通过", False: "未通过"})
display = display.drop(columns="通过")
st.dataframe(display, use_container_width=True, hide_index=True)

st.subheader("模块运行耗时")
timing_rows = [{"模块": name, "耗时 ms": value} for name, value in timings.items()]
if timing_rows:
    import pandas as pd

    st.dataframe(pd.DataFrame(timing_rows), use_container_width=True, hide_index=True)

report = generate_acceptance_markdown(checks, timings)
st.download_button(
    "下载验收报告",
    data=report.encode("utf-8"),
    file_name="system_acceptance_report.md",
    mime="text/markdown",
)

