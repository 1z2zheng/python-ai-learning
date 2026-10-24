# 小麦与大麦项目交付包

整理日期：2026-10-20。

本包仅保留可运行程序、小麦和大麦正式数据、9个推理检查点、必要训练与评估代码、最终实验结果及最终报告/PPT。

## 启动

本机双击“启动项目.cmd”，使用保留的 `venv-unet` 环境。浏览器访问 http://127.0.0.1:5000 。

换到另一台Windows电脑：先安装Python 3.12，再运行 `install_dependencies.cmd` 安装依赖，然后双击“启动项目.cmd”。压缩包不含虚拟环境，首次安装需要联网；依赖版本取自本机验证环境，PyTorch为CUDA 13.0构建。无可用CUDA时程序自动选择CPU。

## 功能

- 小麦麦穗分割：RGB的PNG/JPG/TIFF，三模型集成，输出像素级掩膜和叠加图。
- 大麦长势分类：蓝、绿、红、红边、近红外五通道TIFF，输出正常/旺盛或干枯/衰退。
- 大麦倒伏识别：相同五通道TIFF，输出直立/倒伏及图块级区域。不是精细像素分割；尺寸不足完整30×30图块的右下边缘不参与区域预测。
- 支持批量分析、区域图/叠加图/CSV下载和历史记录。小麦未输出分类置信度，页面以“—”显示。

## 文件

| 目录 | 内容 |
| --- | --- |
| `C260928003程序/` | 程序、配置、9个模型、核心小麦数据、最终实验结果和功能测试 |
| `数据集/` | 芬兰大麦135张五通道图像、春大麦样本矩阵及来源说明 |
| `系统测试样例/` | 可直接上传的大麦五通道TIFF |
| `最终成果/` | 最新开题报告DOCX、对应PDF及最终答辩PPT，原文件内容保留 |

小麦训练数据：`C260928003程序/wheat_head_dataset_500/`（400/50/50划分）。
小麦独立盲测：`C260928003程序/blind_holdout_30/evaluation_dataset/`。
小麦外部测试：`C260928003程序/external_test_gwfss_100/`。

## 训练、评估与测试

在 `C260928003程序` 目录运行（使用已安装依赖的Python）：

```powershell
python train.py --config configs/ablation_multiscale_seed42.json
python predict.py --image wheat_head_dataset_500/images/某张图片.jpg --output-dir predictions
python evaluate.py --checkpoint runs/ablation_multiscale_seed123/best.pt --data-root wheat_head_dataset_500 --split test --output-dir evaluation
python -m pytest tests -q -p no:cacheprovider
```

训练大麦模型见程序目录的README。三个小麦种子配置均保留；仅保留最终模型权重，旧模型的历史对比数字保存在 `reports/`，不承诺用本包复现已删除旧架构检查点的结果。

压缩包内 `文件校验清单.json` 记录每个交付文件的SHA256，可用于检查传输完整性。清理和打包结果见 `交付核验.json`。

历史归档、旧数据版本、燕麦与水稻数据及功能、非普通大麦的辅助资料、原始重复下载包、自动标注中间目录、旧训练检查点、网页预测记录、报告/PPT制作过程文件、Node开发依赖及缓存已从项目删除。程序运行后会按需新建结果目录。
