# 电竞外设竞品定价与偏好分析系统

将键鼠商品参数、历史报价与评论证据接入 SQLite，通过 Streamlit 页面完成数据筛选、偏好匹配、定价分析和竞品比较。

![数据概览演示](../docs/assets/esports-overview.png)

图中为合成演示数据，不代表真实市场数据。正式数据分析保留来源、版本冲突和研究口径，不把评价数直接当销量，也不把规则评分包装为用户真实偏好。

## 可以体验什么

- 数据概览：类别、平台、品牌与价格区间筛选。
- 偏好匹配：可调整六维权重，查看排名、得分条形图与雷达图。
- 定价与促销分析：报价走势、价格带与优惠情况。
- 竞品比较：参数、报价和候选型号比较。
- 数据维护：CSV 校验导入、评论复核、证据追溯与系统验收。

## 快速体验

Python 3.12，进入本目录后执行：

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe demo.py
```

默认地址为 `http://localhost:8501`。也可指定 `python demo.py --port 8502`。macOS / Linux 使用 `python3 -m venv .venv`，将解释器路径替换为 `.venv/bin/python`。

`demo.py` 会在独立的 `data/demo.db` 中准备 80 个合成商品、240 条报价快照，不使用正式库；该样例没有真实评论、平台销量或经过核验的跨平台研究记录。对应模块在缺少数据时会提示补充数据。

## 使用正式数据

进入本目录，使用安装好依赖的 Python：

```powershell
python scripts/init_db.py
python scripts/import_csv.py data/seed/products_sample.csv
python -m streamlit run app.py
```

默认业务库为 `data/esports_analysis.db`，可用 `ESPORTS_DB_PATH` 覆盖。仓库没有上传个人业务数据库、原始评论下载或登录会话。采集与研究口径详见 [项目说明](docs/项目说明.md) 和 [浏览器采集适配器](historical-price-spider/README.md)。平台采集需要自行登录；遇到验证码或限流应停止采集。

`data/processed` 保留已有处理结果；正式评论分析还需要相应的业务库记录与分析输出。演示数据只用于功能体验，不能用于研究结论。

## 验证

```powershell
python demo.py --prepare-only
python -m pytest tests -q -p no:cacheprovider
```

2026-10-24 在独立源码副本上运行：**73 项测试通过**；演示数据校验和浏览器页面检查通过。[验证记录](../docs/verification.md)说明了环境与边界。

## 代码结构

| 路径 | 内容 |
| --- | --- |
| `app.py`、`pages/` | 首页与九个分析/管理页面 |
| `src/esports_analysis/` | 数据库、数据清洗、评分、价格分析和评论处理 |
| `scripts/` | 数据导入、采集、分析与验收工具 |
| `data/seed/` | CSV 模板和合成演示样例 |
| `tests/` | 数据处理与页面自动化检查 |
| `demo.py` | 独立演示库准备与启动入口 |

规则评分、评论情感与采集覆盖均有适用范围；不声称已完成商业上线、真实用户验证或自动化交易。
