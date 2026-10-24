# 统一推理程序

该程序串联 MobileNetV3 病害分类模型和 DeepLabV3+ 病斑分割模型。

在项目根目录运行：

```powershell
python inference\predict.py --image "你的图片路径.jpg"
```

输出位于 `runs/unified_inference/图片名/`：

- `input.jpg`：自动校正方向后的输入图；
- `lesion_mask.png`：二值病斑掩膜；
- `lesion_overlay.jpg`：红色病斑叠加图；
- `result.json`：类别、置信度、候选类别、病斑比例和模型信息。

若分类结果为健康，程序默认跳过病斑分割。需要强制查看分割结果时添加：

```powershell
python inference\predict.py --image "你的图片路径.jpg" --force-segmentation
```
