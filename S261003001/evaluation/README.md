# 系统验收与实验材料

重新生成全部实验图表：

```powershell
cd D:\S261003001
python evaluation\generate_report.py
```

输出目录：`runs/final_report/`。

`system_acceptance_template.csv` 用于人工记录30～60张独立测试图片的端到端结果。测试图片不能来自训练集。
# 实验评估与系统验收

生成论文实验汇总材料：

```powershell
python evaluation\generate_report.py
```

在PlantSeg独立测试集上按类别均衡抽取60张图片，执行分类与分割系统验收：

```powershell
python evaluation\system_acceptance.py
```

结果保存在 `runs\system_acceptance`。程序会自动生成逐图CSV、总体统计、混淆矩阵、典型案例图和Markdown验收报告。
