# PlantVillage 图像分类

当前分类任务包含 11 类：番茄 4 类、马铃薯 3 类、苹果 4 类。流程为数据整理、模型训练、测试集评估和单张图片预测。

## 1. 安装依赖

```powershell
cd classification
python -m pip install -r requirements.txt
```

## 2. 整理 PlantVillage

将下面的源路径替换为本机 PlantVillage 根目录：

```powershell
python prepare_dataset.py --source "D:\data\PlantVillage" --output "..\dataset_classification"
```

脚本会递归找到原始类别目录，固定随机种子后按 70% / 20% / 10% 划分，并生成 `classes.json` 和 `manifest.json`。输出目录已存在时默认停止，确实需要重建可加 `--overwrite`。

整理完成后可执行完整性检查：

```powershell
python validate_dataset.py --data "..\dataset_classification"
```

该命令检查损坏图片、跨数据集重复内容和类别目录一致性，并生成 `dataset_counts.csv` 与 `dataset_report.json`。

## 3. 训练与测试

```powershell
python train.py --data "..\dataset_classification" --output "..\runs\mobilenet_v3_small"
```

默认使用 ImageNet 预训练的 MobileNetV3-Small、224×224 输入、30 个 epoch，并依据验证集准确率保存最佳模型和提前停止。输出包括：

默认还会采用类别平衡损失，降低少数类（例如 `Potato_healthy`）被多数类淹没的风险。如需关闭可添加 `--class-weighting none`。

- `best.pt`：最佳权重及类别映射
- `metrics.json`：测试准确率、分类报告、混淆矩阵和训练历史
- `accuracy.png`：训练/验证准确率曲线

可改用 ResNet18：

```powershell
python train.py --model resnet18 --data "..\dataset_classification" --output "..\runs\resnet18"
```

Windows 若多进程读取数据报错，可加 `--workers 0`。

## 4. 单张图片预测

```powershell
python predict.py --checkpoint "..\runs\mobilenet_v3_small\best.pt" --image "D:\data\leaf.jpg"
```

程序输出概率最高的 3 个类别。实际部署时应加入“非叶片/未知图片”检测；仅用 PlantVillage 训练的模型不适合直接把任意图片都强制归入这 11 类。

## 5. 补充人工核实的田间苹果黑腐病照片

以下命令从项目根目录 `D:\S261003001` 执行。

不要把名称带有 `Apple___Black_rot` 的数据包直接当成新田间数据。ABCGMP v3 的该类别虽然有 1,987 张，但抽查是统一背景叶片；检查结论见 `external/abcgmp_v3/INSPECTION.md`。这些图片**没有**加入训练。

把已人工核实病害类别、且有明确使用许可的田间照片放在一个单独文件夹（不要放其他病害）。先预览去重结果：

```powershell
python classification\import_reviewed_blackrot.py --source "D:\田间黑腐病已核实照片" --source-url "照片来源或自采说明" --license "许可说明" --confirmed-labels --dry-run
```

确认后去掉 `--dry-run` 再运行。脚本只向 `dataset_natural_extra/train/Apple_Black_rot` 复制合格图，校验与 PlantVillage、PlantSeg 和现有额外数据的全部划分是否近重复；不会移动原件，也不会改动验证集和测试集。输出 `field_blackrot_manifest.csv` 和 `field_blackrot_report.json`。照片的真实病名仍需人工判断，去重脚本无法证明标签正确。

补图后建议先用去重后的 PlantSeg 划分重新训练混合基线，再微调；原有权重与新划分不能混用：

```powershell
python classification\train_mixed.py --plantseg-splits runs\plantseg_split_audit\splits_deduplicated.json --output runs\mobilenet_v3_large_mixed_clean
python classification\train_refined.py --plantseg-splits runs\plantseg_split_audit\splits_deduplicated.json --checkpoint runs\mobilenet_v3_large_mixed_clean\best.pt --output runs\mobilenet_v3_large_refined_clean
```

这两步是训练命令，不会由数据导入脚本自动执行。最后用同一份清洁测试划分比较基线和新模型，重点看 `Apple_Black_rot` 的 F1 和混淆矩阵，不只看整体准确率。

## 6. AppleLeaf9 田间黑腐病补充与训练

已核实 AppleLeaf9 的 `Frogeye leaf spot` 为苹果黑腐病的叶部病征，可对应 `Apple_Black_rot`。来源、论文、许可、标签依据及处理限制见 `dataset_natural_extra/SOURCE_APPLELEAF9.md`。下载与去重可复现运行：

```powershell
python classification\download_appleleaf9_blackrot.py
```

脚本只取作者原始仓库这一类别；保存原图来源和固定版本，筛查现有全部图片及同批旋转/翻转近重复后复制到额外训练集。候选数量不等于最终接收数量，实际统计见 `dataset_natural_extra/appleleaf9_report.json`。分类图片没有像素级病斑标注，因此不进入分割训练集。

数据导入完成后，由用户启动下面的训练脚本：

```powershell
cd D:\S261003001
python train_blackrot_improved.py
```

它先使用去重后的 PlantSeg 划分训练混合基线，输出到 `runs/mobilenet_v3_large_mixed_clean_appleleaf9`，成功后再读取包含 AppleLeaf9 的额外训练集进行微调，输出到 `runs/mobilenet_v3_large_refined_appleleaf9`。已存在的输出目录会触发停止，避免覆盖实验。`train_refined.py` 的 `--extra-share` 与原来的 `--plantdoc-share` 是同一采样参数的两个名称，涵盖全部额外田间训练数据。

训练后对照同一次清洁划分的基线，查看总体准确率、macro-F1、苹果黑腐病 Precision/Recall/F1 和混淆矩阵；先按验证集决定模型，再报告测试结果。旧模型的原始划分测试数值不能与新划分的数值直接当作公平提升比较。

## 7. 直接在系统原模型上保守续训

AppleLeaf9 两阶段实验没有超过系统原模型时，可使用以下独立实验入口：

```powershell
cd D:\S261003001
python classification\train_original_continued.py
```

起点固定为系统使用的 `runs/mobilenet_v3_large_plantdoc/best.pt`。默认只更新末尾四个特征模块和分类头，冻结 BatchNorm 统计量；骨干学习率为 `2e-6`，分类头为 `1e-5`，使用原模型的输出分布约束续训，减轻已有能力丢失。PlantDoc 与 AppleLeaf9 分开采样；AppleLeaf9 只占黑腐病类别采样质量的 12.5%，不会因为有三千多张而按文件数量主导训练。采用较轻的数据增强，每轮固定 200 批次，最多 8 轮、连续 3 轮无改进提前结束。

程序先重新评估原模型验证表现。只有自然场景验证准确率、实际有样本类别的 macro-F1 均不低于原模型，且二者组合分更高、PlantVillage 验证准确率下降不超过 0.5 个百分点，才保存新的最佳权重。若没有符合条件的轮次，最终 `best.pt` 保留原模型权重，`best_epoch` 为 0。这里的约束只保证验证指标满足规则，不保证独立测试集也提升。

输出目录为 `runs/mobilenet_v3_large_original_continued`，包含 `config.json`、逐轮 `history.json`、`best.pt`、`metrics.json`。测试集仅在验证选择结束后用于一次最终对比，始终与真正的原模型比较；程序不会自动替换系统权重。输入检查可单独运行：

```powershell
python classification\train_original_continued.py --check-only
```

方法说明：续训使用去重后的 PlantSeg 清单，但起点继承了原模型在原始划分上的训练历史，因此必须称为原模型续训实验，不能声称它是从零开始的完全清洁基线。现有测试集已被多次观察，最终结论仍应补充独立田间测试集验证。
