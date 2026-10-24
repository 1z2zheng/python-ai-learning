# Streamlit 可视化系统

## 启动

双击项目根目录的 `start_app.bat`，或者在 PowerShell 中执行：

```powershell
cd D:\S261003001
python -m streamlit run app\streamlit_app.py
```

浏览器会自动打开 `http://localhost:8501`。

## 功能

- 上传 JPG、PNG 或 WEBP 图片；
- 使用 PlantVillage + PlantSeg + PlantDoc 微调的 MobileNetV3-Large，显示病害类别、中文名称、置信度及前三候选；
- 健康图片自动跳过分割；
- 病害图片显示二值病斑 Mask、红色叠加图和病斑比例；
- 低置信度结果提示人工复核；
- 显示 Grad-CAM 分类关注区域；热力图不等于病斑边界；
- 按病斑像素占整幅图片的比例给出轻度、中度、重度参考等级；此等级不是叶片受害率或农学诊断标准；
- 可调整病斑判定阈值、推理分辨率和最小病斑面积；
- 下载 JSON、Mask 和叠加图；
- 一键下载包含原图、结果 JSON、病斑图和 Grad-CAM 图的 ZIP；
- 将检测结果保存至 `runs/streamlit_inference/`。
