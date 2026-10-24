# 验证记录

日期：2026-10-24。

## 电竞外设竞品分析

在仅恢复仓库源码和演示数据的独立工作副本中检查，依赖使用本机已有 Python 3.12 环境。没有声称进行过全新机器的 pip 安装测试。

- `python demo.py --prepare-only`：通过；独立 SQLite 演示库包含 80 个合成商品与 240 条价格快照。
- 重复执行准备命令：通过；不重复导入商品与价格快照。
- `python -m pytest tests -q -p no:cacheprovider`：73 passed。
- `python demo.py --port 8517`：已实际启动，并在浏览器确认首页统计与数据概览图表正常显示。README 截图来源于这次运行。

本次修复：补齐公开规格测试样例；偏好页面测试使用独立临时数据库，不依赖开发机业务数据；在缺少可选评论分析产物时核验明确提示和键盘排序结果；移除页面提示中的硬编码 Python 路径。

## 两个视觉项目

展示图来自已有本地推理文件，未重新训练、重新评估或将单图结果当作总体指标。权重和完整训练数据没有上传，克隆后需按 `MODEL_ASSETS.json` 补齐权重。

## 展示图来源

- `esports-overview.png`：本次启动的键鼠合成数据演示截图。
- `wheat-input.png` 与 `wheat-overlay.png`：麦穗项目既有推理记录 `7e5245514914`，同一输入与输出。
- `disease-overlay.jpg`：病害项目既有 `system_acceptance_plantdoc` 记录，番茄早疫病样例 `003_Tomato_Early_blight__tomato_early_blight_google_0163.jpg`。

数据、第三方样例及预训练组件的使用权遵循各自来源条款；本仓库未替第三方资产授予新的许可。
