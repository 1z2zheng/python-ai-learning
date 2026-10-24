# 农作物病害识别与病斑分割系统

将病害分类、病斑区域分割、Grad-CAM 可解释图与结果导出接入同一个 Streamlit 应用。

![病斑预测效果](../docs/assets/disease-overlay.jpg)

已有本地番茄早疫病样例输出，红色区域为预测病斑；这里展示输出形式，不以单张样例推断总体准确率。

## 功能

- 上传 JPG、PNG、WEBP，显示病害中文类别、置信度和前三候选。
- 使用 MobileNetV3 分类，使用 DeepLabV3+ / MobileNetV3 分割病斑。
- 健康类别跳过分割，低置信度提示人工复核。
- 显示 Grad-CAM 关注区域，调整分割阈值、分辨率与最小区域。
- 下载 JSON、Mask、叠加图及包含全部结果的 ZIP。

Grad-CAM 是分类关注区域，不是病斑边界。病斑占整幅图片的比例仅提供参考，不等于叶片受害率或农学诊断等级。

## 运行

**仓库没有上传模型权重和完整训练集**。先恢复 [MODEL_ASSETS.json](MODEL_ASSETS.json) 中 `required_for_app: true` 的两个检查点到原相对路径，再核对 SHA256。模型没有公开下载地址，缺少检查点时无法进行真实推理。

建议 Python 3.12，在本目录执行：

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r app/requirements.txt
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
```

macOS / Linux 可使用 `python3` 和 `.venv/bin/python`。默认打开 `http://localhost:8501`；若其他应用占用端口，加 `--server.port 8502`。

## 源码导航

| 路径 | 内容 |
| --- | --- |
| `classification/` | 分类数据准备、训练、评估与弱类分析 |
| `segmentation/` | 标注转换、分割训练与结果生成 |
| `inference/predict.py` | 分类和分割统一推理 |
| `inference/explain.py` | Grad-CAM |
| `app/streamlit_app.py` | 可视化交互 |
| `evaluation/` | 评估与验收脚本 |

更多实现说明见各模块 README。模块中的旧本机绝对路径应替换为当前目录。本次整理没有重新训练、重新评估模型或部署在线服务。
