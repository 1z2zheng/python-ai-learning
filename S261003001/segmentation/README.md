# PlantVillage 病斑分割标注准备

本阶段只使用 8 个病害类别，不选健康图片。默认每类均衡抽取 75 张，共 600 张。

## 1. 准备待标注图片

```powershell
cd D:\S261003001
python segmentation\prepare_annotation.py `
  --classification-data dataset_classification `
  --output dataset_segmentation `
  --samples-per-class 75
```

输出图片位于 `dataset_segmentation\labelme`，抽样信息记录在 `annotation_manifest.csv`。

## 2. 使用 Labelme 标注

为避免 Labelme 的 GUI/ONNX 依赖影响 PyTorch 训练环境，使用独立的 Python 3.12 Conda 环境安装。不要升级或改动用于模型训练的 `base` 环境。

```powershell
cd D:\S261003001
conda env create -f segmentation\environment-labelme.yml
conda run -n labelme312 labelme D:\S261003001\dataset_segmentation\labelme
```

以后再次启动 Labelme 只需执行：

```powershell
cd D:\S261003001
conda run -n labelme312 labelme D:\S261003001\dataset_segmentation\labelme
```

Labelme 7.x 可在 `Edit > Settings > Language` 中选择简体中文。

标注规范：

- 使用 Create Polygons 沿病斑边界绘制多边形。
- 所有病斑统一使用标签 `lesion`，注意全部小写。
- 一张叶片有多个病斑时分别绘制多个 `lesion` 多边形。
- 不要标叶片正常区域、阴影和纯背景。
- JSON 直接保存在图片所在的 `labelme` 目录。

## 3. 转换为二值 mask

完成标注后执行：

```powershell
python segmentation\labelme_to_masks.py `
  --labelme-dir dataset_segmentation\labelme `
  --output dataset_segmentation
```

结果目录：

```text
dataset_segmentation/
├── labelme/              # 待标注图片及 Labelme JSON
├── images/               # 已转换的训练图片
├── masks/                # 二值 mask：0=背景，255=病斑
└── annotation_manifest.csv
```

## 4. 检查 mask

```powershell
python segmentation\validate_masks.py --data dataset_segmentation
```

检查内容包括图片与 mask 是否一一对应、尺寸是否一致、mask 是否仅包含 0 和 255，以及是否存在空 mask。检查报告写入 `dataset_segmentation\mask_report.json`。

完成并通过检查后，再进行训练集/验证集/测试集划分和 U-Net 训练。

## 5. 训练 U-Net 试验模型

当前 32 对数据均为苹果黑星病，只适合验证训练流程和生成该病害的预标注，不能代表其他病害类别。

```powershell
cd D:\S261003001
python segmentation\train_unet.py `
  --data dataset_segmentation `
  --output runs\unet_apple_scab `
  --epochs 60 `
  --batch-size 4 `
  --workers 0 `
  --patience 12
```

脚本会按固定随机种子划分 23 张训练、6 张验证、3 张测试数据，并输出：

```text
runs/unet_apple_scab/
├── best.pt                 # 验证集 Dice 最优权重
├── splits.json             # 本次数据划分
├── metrics.json            # 测试指标和训练历史
├── training_curves.png     # Loss 与 Dice 曲线
└── test_predictions.png    # 测试集预测可视化
```
