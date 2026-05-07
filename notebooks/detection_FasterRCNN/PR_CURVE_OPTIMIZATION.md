# Faster R-CNN V1 PR 曲线升级方案

**创建时间**: 2026-05-08  
**状态**: ✅ 已完成  
**实验**: Faster R-CNN V1 Baseline

---

##  优化目标

将当前的综合 PR 曲线（所有类别混合）升级为 **YOLOv8 风格的分类别 PR 曲线**，包含：
1. 每个类别的独立 PR 曲线
2. 所有类别的综合 PR 曲线（加粗显示）
3. 每个类别的 mAP@0.5 值标注

---

## ✅ 实施完成

### **Step 1: 修改 DetectionEvaluator 类** ✅

#### **新增方法：`_compute_per_class_pr_curves()`** ✅
```python
def _compute_per_class_pr_curves(predictions, ground_truths, num_classes, class_names):
    """
    计算每个类别的 PR 曲线数据
    
    实现逻辑：
    1. 定义置信度阈值数组：np.arange(0.0, 1.05, 0.05)
    2. 对每个阈值计算每个类别的 TP, FP, FN
    3. 计算 Precision 和 Recall
    4. 使用 sklearn.metrics.auc 计算 AP
    """
```

#### **修改方法：`plot_pr_curve()`** ✅
```python
def plot_pr_curve(self, output_dir):
    """
    从 pr_curves_data.json 绘制分类别 PR 曲线（YOLOv8 风格）
    
    实现：
    1. 读取 pr_curves_data.json
    2. 绘制每个类别的 PR 曲线（细线，不同颜色）
    3. 绘制综合 PR 曲线（粗线，蓝色）
    4. 在图例中显示每个类别的 AP 值
    5. 添加标题和网格
    """
```

### **Step 2: 修改 evaluate_fasterrcnn() 方法** ✅

在评估时调用新方法生成 PR 数据：
```python
# 生成 PR 曲线数据
print("\nGenerating PR curves data (per-class)...")
pr_data = self._compute_per_class_pr_curves(all_preds, all_gts, num_classes, class_names)
pr_path = output_path / 'pr_curves_data.json'
with open(pr_path, 'w') as f:
    json.dump(pr_data, f, indent=2)
print(f"✓ PR curves data saved to: {pr_path}")
```

### **Step 3: V1 Notebook** ✅

Cell 11 无需修改，[[plot_pr_curve()](file:///Users/elliott/vscode_workplace/CNN_A2/notebooks/detection_YOLOv8/YOLOv8_modules.ipynb#L909-L938)](file:///Users/elliott/vscode_workplace/CNN_A2/notebooks/detection_FasterRCNN/FasterRCNN_modules.ipynb#L1473-L1526) 方法会自动读取新的 JSON 数据并绘制。

---

##  输出文件结构

```
outputs/detection_fasterrcnn_v1/
├── training/
│   ├── training_history.csv
│   ├── loss_curve.png
│   └── map_curve.png
├── evaluation/
│   ├── evaluation_metrics.json
│   ├── pr_curves_data.json          # ✅ NEW - PR 曲线数据
│   ├── confusion_matrix.png
│   └── confusion_matrix_normalized.png
```

---

##  可视化效果

### **预期输出**
```
=== Precision-Recall Curve ===
================================================================================

[PR 曲线图]
├─ Alert: 0.85 (细线)
├─ Angry: 0.78 (细线)
├─ Frown: 0.82 (细线)
├─ Happy: 0.91 (细线)
├─ Relaxed: 0.88 (细线)
└─ all classes: 0.85 mAP@0.5 (粗线，蓝色)
```

---

##  验证清单

- [x] _compute_per_class_pr_curves() 方法实现
- [x] plot_pr_curve() 方法更新（YOLOv8 风格）
- [x] evaluate_fasterrcnn() 添加 PR 数据生成
- [x] 保存 pr_curves_data.json 到 evaluation/ 目录
- [x] 图例显示每个类别的 AP 值
- [x] 综合 PR 曲线加粗显示
- [x] 无语法错误

---

##  下一步

运行 V1 Notebook 测试新的 PR 曲线可视化效果。

**当前状态**: V1 PR 曲线优化已完成，准备测试 ✅
