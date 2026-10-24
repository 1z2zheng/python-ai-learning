# 小麦与大麦遥感分析程序

启动与目录说明见上一级README。网页包含小麦麦穗分割、大麦长势分类、大麦倒伏区域识别。

小麦模型：`runs/ablation_multiscale_seed{42,123,2026}/best.pt`，MultiScaleUNet，三模型logits平均集成。模型封版说明和已有指标见 `final_model/C260928003-最终模型清单.json` 与 `reports/C260928003-最终模型评估报告.md`。

大麦模型：`outputs/spring_barley_multitask_5band/{growth,lodging}/seed_{42,123,2026}/best.pt`。三模型概率平均。训练结果见相邻的 `summary.json`。

大麦重新训练命令：

```powershell
python train_spring_barley_multitask.py --tiff "../数据集/春大麦长势倒伏_Zenodo15497780/patch_matrix_v1/SG/_SG_CIR_sample_composit_crs_nan.tif" --metadata "../数据集/春大麦长势倒伏_Zenodo15497780/patch_matrix_v1/SG/_SG_CIR_sample_composit.txt" --output outputs/spring_barley_multitask_5band --tasks growth lodging --seeds 42 123 2026 --epochs 50
```

健康检查：`GET /api/health`。推理：`POST /api/predict`，表单字段mode为wheat、growth或lodging，images为1—10个文件，总大小不超过120MB。

测试包含真实模型加载、三类样例推理、输出尺寸、批次CSV、历史记录删除和异常输入，运行 `python -m pytest tests -q -p no:cacheprovider`。

小麦数据为现有已处理的分割训练/测试集。大麦倒伏为图块级区域识别，不能当作精细像素标注。`reports/基线对比` 是较早基线结果；当前正式小麦模型以最终模型清单为准。
