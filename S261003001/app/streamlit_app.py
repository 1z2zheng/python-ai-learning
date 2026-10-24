"""Streamlit interface for crop disease classification and lesion segmentation."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from inference.predict import (  # noqa: E402
    CHINESE_NAMES,
    DEFAULT_CLASSIFIER,
    DEFAULT_SEGMENTER,
    classify,
    load_classifier,
    load_segmenter,
    segment,
    select_device,
    severity_level,
)
from inference.explain import grad_cam  # noqa: E402
from app.result_bundle import image_bytes, result_archive  # noqa: E402


st.set_page_config(
    page_title="农作物病害智能识别系统",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1280px;}
    .hero {padding: 1.35rem 1.6rem; border-radius: 16px;
           background: linear-gradient(120deg, #123c2f, #247052); color: white;
           margin-bottom: 1.2rem; box-shadow: 0 6px 22px rgba(18,60,47,.16);}
    .hero h1 {margin: 0; font-size: 2rem;}
    .result-card {padding: 1rem 1.2rem; border: 1px solid #dce8e2;
                  border-radius: 12px; background: #f8fbf9; margin: .4rem 0 1rem;}
    div[data-testid="stMetric"] {background: #f7faf8; border: 1px solid #e0ebe5;
                                 padding: .75rem; border-radius: 10px;}
    </style>
    <div class="hero">
      <h1>🌿 农作物病害智能识别系统</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="正在加载分类与分割模型……")
def load_models(device_name: str):
    device = select_device(device_name)
    classifier, classes, classifier_transform, classifier_checkpoint = load_classifier(
        DEFAULT_CLASSIFIER, device
    )
    segmenter, segmenter_checkpoint = load_segmenter(DEFAULT_SEGMENTER, device)
    return (
        device,
        classifier,
        classes,
        classifier_transform,
        classifier_checkpoint,
        segmenter,
        segmenter_checkpoint,
    )


@st.cache_data(show_spinner=False)
def model_metrics() -> dict[str, float]:
    classification = json.loads(
        (PROJECT / "runs" / "mobilenet_v3_large_plantdoc" / "metrics.json").read_text(encoding="utf-8")
    )
    segmentation = json.loads(
        (PROJECT / "runs" / "deeplabv3plus_plantseg" / "metrics.json").read_text(encoding="utf-8")
    )
    return {
        "plantvillage_accuracy": float(classification["plantvillage_test"]["accuracy"]),
        "plantseg_accuracy": float(classification["plantseg_test"]["accuracy"]),
        "plantseg_confidence": float(classification["plantseg_test"]["mean_confidence"]),
        "segmentation_dice": float(segmentation["test"]["dice"]),
        "segmentation_iou": float(segmentation["test"]["iou"]),
    }


def run_prediction(
    image: Image.Image,
    source_name: str,
    confidence_threshold: float,
    force_segmentation: bool,
    device_name: str,
    segmentation_threshold: float,
    segmentation_size: int,
    min_component_percent: float,
) -> dict[str, object]:
    (
        device,
        classifier,
        classes,
        classifier_transform,
        classifier_checkpoint,
        segmenter,
        segmenter_checkpoint,
    ) = load_models(device_name)
    started = time.perf_counter()
    predictions = classify(
        image, classifier, classes, classifier_transform, device, top_k=3
    )
    primary = predictions[0]
    predicted_class = str(primary["class"])
    predicted_index = classes.index(predicted_class)
    confidence = float(primary["confidence"])
    healthy = predicted_class.endswith("_healthy")
    uncertain = confidence < confidence_threshold
    should_segment = force_segmentation or not healthy
    mask = overlay = None
    lesion_ratio = 0.0
    if should_segment:
        mask, overlay, lesion_ratio = segment(
            image,
            segmenter,
            segmenter_checkpoint,
            device,
            threshold_override=segmentation_threshold,
            image_size_override=segmentation_size,
            min_component_percent=min_component_percent,
        )
    classification_crop, heatmap = grad_cam(
        image,
        classifier,
        classifier_transform,
        device,
        predicted_index,
        int(classifier_checkpoint.get("image_size", 224)),
    )
    elapsed = time.perf_counter() - started
    report = {
        "input_filename": source_name,
        "device": str(device),
        "classification": {
            "predicted_class": predicted_class,
            "predicted_name_zh": primary["name_zh"],
            "confidence": confidence,
            "confidence_threshold": confidence_threshold,
            "uncertain": uncertain,
            "healthy": healthy,
            "top_k": predictions,
            "model": classifier_checkpoint["model_name"],
        },
        "segmentation": {
            "executed": should_segment,
            "model": segmenter_checkpoint["model_name"] if should_segment else None,
            "threshold": segmentation_threshold if should_segment else None,
            "inference_size": segmentation_size if should_segment else None,
            "min_component_percent": min_component_percent if should_segment else None,
            "lesion_ratio": lesion_ratio,
            "lesion_percent_of_image": lesion_ratio * 100,
            "severity_reference": severity_level(lesion_ratio, healthy, should_segment),
        },
        "explanation": {
            "method": "Grad-CAM",
            "class": predicted_class,
            "note": "分类关注区域示意，不是病斑边界或分割结果。",
        },
        "elapsed_seconds": elapsed,
    }
    return {
        "report": report,
        "mask": mask,
        "overlay": overlay,
        "classification_crop": classification_crop,
        "heatmap": heatmap,
    }


def save_result(image: Image.Image, result: dict[str, object]) -> Path:
    report = result["report"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    source_stem = Path(str(report["input_filename"])).stem
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_stem)[:60]
    output = PROJECT / "runs" / "streamlit_inference" / f"{timestamp}_{safe_stem}"
    output.mkdir(parents=True, exist_ok=False)
    image.save(output / "input.jpg", quality=95)
    if result["mask"] is not None:
        result["mask"].save(output / "lesion_mask.png")
        result["overlay"].save(output / "lesion_overlay.jpg", quality=95)
    if result["heatmap"] is not None:
        result["classification_crop"].save(output / "classification_crop.jpg", quality=95)
        result["heatmap"].save(output / "classification_gradcam.jpg", quality=95)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


metrics = model_metrics()
with st.sidebar:
    st.header("系统设置")
    device_choice = st.selectbox(
        "推理设备",
        options=("auto", "cuda", "cpu"),
        format_func=lambda value: {
            "auto": "自动选择",
            "cuda": "GPU（CUDA）",
            "cpu": "CPU",
        }[value],
    )
    confidence_threshold = st.slider(
        "低置信度阈值",
        min_value=0.30,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help="分类置信度低于该值时提示人工复核。",
    )
    force_segmentation = st.checkbox(
        "健康类别也执行分割",
        value=False,
        help="默认情况下，预测为健康时不运行病斑分割。",
    )
    st.divider()
    st.subheader("病斑精细化")
    segmentation_threshold = st.slider(
        "病斑判定阈值",
        min_value=0.35,
        max_value=0.80,
        value=0.65,
        step=0.01,
        help="数值越高，病斑区域越小、更保守；数值越低，检出区域越多。训练集最优阈值为0.43。",
    )
    st.caption("病斑块偏大时建议调至 0.65～0.75；漏检时再逐步降低。")
    segmentation_size = st.select_slider(
        "分割精细度",
        options=(384, 512, 640),
        value=512,
        format_func=lambda value: f"{value} × {value}",
        help="更高分辨率可保留更多小病斑细节，但推理稍慢。",
    )
    min_component_percent = st.slider(
        "最小病斑面积（%）",
        min_value=0.00,
        max_value=0.20,
        value=0.03,
        step=0.01,
        help="移除面积过小的孤立噪点；不会缩小主要病斑。",
    )
    st.divider()
    st.subheader("模型信息")
    st.caption("分类：MobileNetV3-Large（PlantVillage + PlantSeg + PlantDoc）")
    st.caption("分割：DeepLabV3+ + MobileNetV3-Large")
    st.metric("PlantVillage分类准确率", f"{metrics['plantvillage_accuracy']:.2%}")
    st.metric("PlantSeg野外准确率", f"{metrics['plantseg_accuracy']:.2%}")
    st.metric("PlantSeg平均置信度", f"{metrics['plantseg_confidence']:.2%}")
    st.metric("分割测试 Dice", f"{metrics['segmentation_dice']:.4f}")
    st.metric("分割测试 IoU", f"{metrics['segmentation_iou']:.4f}")
    if torch.cuda.is_available():
        st.success(f"CUDA可用：{torch.cuda.get_device_name(0)}")
    else:
        st.info("当前使用CPU推理")


st.subheader("上传待检测图片")
uploaded = st.file_uploader(
    "上传图片",
    type=("jpg", "jpeg", "png", "webp"),
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("请上传一张苹果、马铃薯或番茄的叶片/果实图片开始识别。")
    st.stop()

payload = uploaded.getvalue()
upload_hash = hashlib.sha256(payload).hexdigest()
try:
    with Image.open(io.BytesIO(payload)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
except Exception as error:
    st.error(f"无法读取图片：{error}")
    st.stop()

if image.width * image.height > 30_000_000:
    st.error("图片超过3000万像素，请压缩后重新上传。")
    st.stop()

if st.session_state.get("upload_hash") != upload_hash:
    st.session_state["upload_hash"] = upload_hash
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("saved_path", None)

settings_key = (
    upload_hash,
    confidence_threshold,
    force_segmentation,
    device_choice,
    segmentation_threshold,
    segmentation_size,
    min_component_percent,
)
if st.session_state.get("prediction_settings") != settings_key:
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("saved_path", None)

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.image(image, caption=f"原始图片：{uploaded.name}", use_container_width=True)
with right:
    st.markdown("### 检测说明")
    st.write(f"图片尺寸：{image.width} × {image.height}")
    st.write("系统先判断作物与病害类别，再对病害图片定位病斑。")
    st.write("病斑比例按整幅图像像素计算，仅作为严重程度参考。")
    analyze = st.button("开始智能识别", type="primary", use_container_width=True)

if analyze:
    try:
        with st.spinner("正在进行病害分类与病斑分割……"):
            st.session_state["prediction_result"] = run_prediction(
                image,
                uploaded.name,
                confidence_threshold,
                force_segmentation,
                device_choice,
                segmentation_threshold,
                segmentation_size,
                min_component_percent,
            )
            st.session_state["prediction_settings"] = settings_key
            st.session_state.pop("saved_path", None)
    except Exception as error:
        st.exception(error)

result = st.session_state.get("prediction_result")
if result is None:
    st.stop()

report = result["report"]
classification = report["classification"]
segmentation = report["segmentation"]
st.divider()
st.subheader("识别结果")

if classification["uncertain"]:
    st.warning("分类置信度较低，建议结合人工观察或补充真实场景数据后复核。")
elif classification["healthy"]:
    st.success("未发现已知病害类别，当前分类结果为健康。")
else:
    st.success("检测到病害，并已完成病斑区域定位。")

metric_columns = st.columns(4)
metric_columns[0].metric("预测类别", classification["predicted_name_zh"])
metric_columns[1].metric("分类置信度", f"{classification['confidence']:.2%}")
metric_columns[2].metric(
    "病斑比例",
    f"{segmentation['lesion_percent_of_image']:.2f}%" if segmentation["executed"] else "未执行",
)
metric_columns[3].metric("推理耗时", f"{report['elapsed_seconds']:.2f} 秒")

st.info(
    f"严重程度参考：{segmentation['severity_reference']}。按病斑像素占整幅图片的比例分级："
    "轻度 <10%，中度 10%～25%，重度 ≥25%。图片背景和拍摄距离会影响结果。"
)

st.markdown("#### 分类候选")
st.dataframe(
    [
        {
            "排名": item["rank"],
            "类别": item["name_zh"],
            "英文标签": item["class"],
            "置信度": f"{item['confidence']:.2%}",
        }
        for item in classification["top_k"]
    ],
    hide_index=True,
    use_container_width=True,
)

if segmentation["executed"]:
    st.markdown("#### 病斑分割")
    image_columns = st.columns(3)
    image_columns[0].image(image, caption="原始图片", use_container_width=True)
    image_columns[1].image(result["mask"], caption="病斑二值掩膜", use_container_width=True)
    image_columns[2].image(result["overlay"], caption="病斑叠加效果", use_container_width=True)

st.markdown("#### 分类关注区域（Grad-CAM）")
explain_columns = st.columns(2)
explain_columns[0].image(
    result["classification_crop"], caption="分类模型实际看到的中心裁剪图", use_container_width=True
)
explain_columns[1].image(
    result["heatmap"], caption="预测类别的关注区域", use_container_width=True
)
st.caption("热力图表示分类依据的大致位置，不代表病斑边界；病斑范围以二值掩膜为准。")

st.markdown("#### 下载与保存")
st.download_button(
    "一键下载完整结果（ZIP）",
    data=result_archive(image, result),
    file_name="crop_disease_result.zip",
    mime="application/zip",
    use_container_width=True,
)
download_columns = st.columns(4)
report_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
download_columns[0].download_button(
    "下载结果 JSON",
    data=report_bytes,
    file_name="result.json",
    mime="application/json",
    use_container_width=True,
)
if result["mask"] is not None:
    download_columns[1].download_button(
        "下载病斑 Mask",
        data=image_bytes(result["mask"], "PNG"),
        file_name="lesion_mask.png",
        mime="image/png",
        use_container_width=True,
    )
    download_columns[2].download_button(
        "下载叠加图",
        data=image_bytes(result["overlay"], "JPEG"),
        file_name="lesion_overlay.jpg",
        mime="image/jpeg",
        use_container_width=True,
    )
download_columns[3].download_button(
    "下载关注热力图",
    data=image_bytes(result["heatmap"], "JPEG"),
    file_name="classification_gradcam.jpg",
    mime="image/jpeg",
    use_container_width=True,
)
if st.button("保存到项目"):
    try:
        st.session_state["saved_path"] = str(save_result(image, result))
    except Exception as error:
        st.error(f"保存失败：{error}")
if st.session_state.get("saved_path"):
    st.success(f"结果已保存：{st.session_state['saved_path']}")

st.caption("本系统仅用于课程设计和科研演示，预测结果不能替代专业农业诊断。")
