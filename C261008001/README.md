# 电竞键鼠竞品定价与偏好分析系统

基于 Python、SQLite、Streamlit 和 Plotly 的毕业设计项目，用于分析电竞鼠标、键盘的价格走势、产品规格、评价热度、用户偏好和竞品定位。

## 已实现功能

- 京东与天猫同型号价格对比
- 慢慢买历史价格导入与价格波动分析
- DPI、重量、回报率、轴体、配列等规格对比
- 评价数热度代理、脱敏评论、情感和优缺点分析
- 六维偏好评分、价格带、促销与竞品定位
- CSV 校验导入、数据质量检查和系统验收

评价数属于热度代理，不等同于实际销量；评论样本用于探索性分析，不代表平台全部用户。

## 快速运行

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\init_db.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开 `http://127.0.0.1:8501`。

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 数据更新

更新京东历史价格：

```powershell
.\.venv\Scripts\python.exe scripts\import_manmanbuy_history.py --provider official-web
```

更新天猫配对样本：

```powershell
.\.venv\Scripts\python.exe scripts\import_manmanbuy_history.py `
  --targets data\manmanbuy\tmall_cross_platform_targets.csv `
  --output data\processed\manmanbuy_tmall_history.csv `
  --log logs\manmanbuy_tmall_history.json `
  --provider official-web
```

采集京东公开评价与商品属性后导入：

```powershell
node historical-price-spider\collect_jd_research_data.mjs `
  data\manmanbuy\esports_targets.csv `
  data\raw\jd_research_20261009.json

.\.venv\Scripts\python.exe scripts\import_jd_research_data.py
```

慢慢买和京东页面采集均使用用户手动登录的专用 Edge 会话，不保存或输出账号、密码和 Cookie。遇到验证码、登录失效或平台频控时停止采集。

## 项目结构

```text
app.py                       Streamlit 入口
pages/                       分析页面
src/esports_analysis/        数据库与分析逻辑
scripts/                     导入、采集和验收脚本
data/                        目标表、原始数据、处理数据和数据库
historical-price-spider/     浏览器采集适配器
tests/                       自动化测试
docs/项目说明.md             数据口径、架构和研究限制
output/                      答辩、数据集和验收交付物
```

默认数据库为 `data/esports_analysis.db`。可通过环境变量 `ESPORTS_DB_PATH` 指定其他数据库。
