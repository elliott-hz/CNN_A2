# YOLOv8 Detection Experiments - Notebook Creation Plan

**Created:** 2026-05-06  
**Student ID:** 25509225  
**Project:** CNN Assignment 2  
**Status:** Planning Phase

---

## 📋 Overview

将现有的 3 个 YOLOv8 检测实验 Python 脚本转换为自包含的 Jupyter Notebooks，采用与 ResNet50 分类实验相同的组织架构（1 个共享模块 + 3 个实验 notebooks）。

**源文件：**
- `experiments/detection_YOLOv8_v1.py`（Baseline）
- `experiments/detection_YOLOv8_v2.py`（Deeper Backbone）
- `experiments/detection_YOLOv8_v3.py`（Shallower Backbone）

---

## ️ Target File Structure

```
notebooks/detection_YOLOv8/
├── make_notebook.md                          # 本文件（方案文档）
├── YOLOv8_modules.ipynb                      # 共享模块库
├── detection_YOLOv8_v1.ipynb                 # V1: Baseline 实验
├── detection_YOLOv8_v2.ipynb                 # V2: Deeper Backbone 实验
└── detection_YOLOv8_v3.ipynb                 # V3: Shallower Backbone 实验
```

---

## 🎯 Core Requirements

### **显示要求**
✅ 所有可视化 inline 显示（bounding box 检测图、训练曲线、PR 曲线）  
✅ 所有指标打印到 notebook 输出  
❌ 不保存 CSV 文件（training_history.csv, metrics.csv）  
❌ 不保存 PNG 文件（confusion_matrix.png, training_curves.png）  
❌ 不保存 JSON/TXT/MD 报告文件  
❌ 不保存 experiment_summary.md  

### **文件保存要求**
⚠️ 根据用户偏好：**不保存任何模型检查点文件**（包括 best_model/final model）  
✅ 仅通过 inline 输出展示所有结果  

### **默认配置**
- **Pretrained:** `True`（可通过编辑 Cell 2 变量修改）
- **Dataset Config:** `/home/sagemaker-user/CNN_A2/data/25509225/Object_Detection/yolo/data.yaml`
- **Output Directory:** `outputs/detection_yolov8_[v1|v2|v3]/`（简化路径，无时间戳子目录）

---

## 📝 Architecture Design

### **YOLOv8_modules.ipynb（6 Cells）**

**Cell 1: Imports & Setup**
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules import C2f
import yaml
import cv2
import copy
from typing import Dict, List, Optional

%matplotlib inline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

**Cell 2: YOLOv8Detector Model**
```python
class YOLOv8Detector(nn.Module):
    """
    YOLOv8 detector with configurable backbone and customization support.
    """
    def __init__(self, backbone='m', input_size=640, 
                 confidence_threshold=0.5, nms_iou_threshold=0.45,
                 pretrained=True, customize_type=None):
        # 完整模型定义（从源代码迁移）
        # 包含 _add_conv_layers() 和 _reduce_conv_layers()
        pass
    
    def _add_conv_layers(self):
        """V2: Add 2 Conv layers after backbone layer2"""
        pass
    
    def _reduce_conv_layers(self):
        """V3: Reduce C2f repeats in backbone layer4"""
        pass
    
    def forward(self, x, **kwargs):
        return self.model(x, verbose=False, **kwargs)
    
    def train_model(self, data, epochs=100, imgsz=None, **kwargs):
        """Train using Ultralytics framework"""
        pass
    
    def save(self, save_path):
        """Save model"""
        pass
```

**Cell 3: Model Configurations**
```python
# Model configurations (供实验 notebooks 使用)
YOLOV8_BASELINE_CONFIG = {
    'backbone': 'm',
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
    'pretrained': True,
    'customize_type': None
}

YOLOV8_V2_CONFIG = {
    'backbone': 'm',
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
    'pretrained': True,
    'customize_type': 'deeper'
}

YOLOV8_V3_CONFIG = {
    'backbone': 'm',
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
    'pretrained': True,
    'customize_type': 'shallower'
}
```

**Cell 4: YOLOv8Trainer**
```python
class YOLOv8Trainer:
    """
    Simple trainer for YOLOv8 models using Ultralytics.
    """
    def __init__(self, learning_rate=0.001, batch_size=16,
                 epochs=100, optimizer='adam', weight_decay=1e-4,
                 use_amp=True, patience=15, cos_lr=False, close_mosaic=0):
        # 训练配置
        pass
    
    def train(self, model, train_data, val_data, output_dir, **kwargs):
        """
        Train YOLOv8 model.
        Returns training results dict.
        """
        results = model.train_model(
            data=train_data,
            epochs=self.epochs,
            batch=self.batch_size,
            lr0=self.learning_rate,
            optimizer=self.optimizer,
            weight_decay=self.weight_decay,
            amp=self.use_amp,
            patience=self.patience,
            cos_lr=self.cos_lr,
            close_mosaic=self.close_mosaic,
            project=output_dir,
            name='train',
            **kwargs
        )
        return results
```

**Cell 5: Training Configurations**
```python
# Training configurations for each experiment
YOLOV8_V1_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs': 300,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'use_amp': True,
    'patience': 50,
    'cos_lr': False,
    'close_mosaic': 0
}

YOLOV8_V2_CONFIG = {
    'learning_rate': 0.0005,
    'batch_size': 12,
    'epochs': 300,
    'optimizer': 'adam',
    'weight_decay': 5e-4,
    'use_amp': True,
    'patience': 50,
    'cos_lr': True,
    'close_mosaic': 10
}

YOLOV8_V3_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 20,
    'epochs': 300,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'use_amp': True,
    'patience': 50,
    'cos_lr': False,
    'close_mosaic': 0
}
```

**Cell 6: DetectionEvaluator**
```python
class DetectionEvaluator:
    """
    Evaluator for YOLOv8 detection models.
    """
    def __init__(self):
        self.metrics = {}
    
    def evaluate_yolov8(self, model, test_dataset, output_dir=None):
        """
        Evaluate YOLOv8 model on test dataset.
        
        Returns:
            dict: Evaluation metrics including mAP, precision, recall, F1
        """
        # Use Ultralytics built-in validation
        results = model.model.val(
            data=test_dataset,
            split='test',
            save_json=False,  # No file saving
            save_hybrid=False
        )
        
        # Extract metrics
        metrics = {
            'map50': results.box.map50,
            'map50_95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'f1': results.box.f1,
            'per_class_metrics': results.box.mean_results()
        }
        
        return metrics
    
    def plot_detection_results(self, model, test_dataset, num_samples=5):
        """
        Plot detection results with bounding boxes.
        Returns matplotlib figure for inline display.
        """
        # Load sample images and run inference
        # Plot with bounding boxes using matplotlib
        # Return fig object
        pass
    
    def plot_training_curves(self, training_results):
        """
        Plot training curves (loss, mAP).
        Returns matplotlib figure for inline display.
        """
        # Extract metrics from training results
        # Plot loss curves and mAP curves
        # Return fig object
        pass
    
    def plot_pr_curves(self, training_results):
        """
        Plot Precision-Recall curves.
        Returns matplotlib figure for inline display.
        """
        # Extract PR data from training results
        # Plot PR curves for each class
        # Return fig object
        pass
```

---

## 📝 Experiment Notebooks Structure

每个实验 notebook 包含 **11 Cells**，结构与 ResNet50 实验一致：

### **detection_YOLOv8_v1.ipynb（Baseline）**

**Cell 1:** Load Modules (`%run ./YOLOv8_modules.ipynb`)

**Cell 2:** Configuration
```python
# === Model Configuration ===
YOLOV8_BASELINE_CONFIG_LOCAL = {
    'backbone': 'm',
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
    'pretrained': True,
    'customize_type': None
}

# === Training Configuration ===
TRAINING_CONFIG_V1 = {
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs': 300,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'use_amp': True,
    'patience': 50,
    'cos_lr': False,
    'close_mosaic': 0
}

# === Experiment Settings ===
STUDENT_ID = "25509225"
DATASET_CONFIG = f"/home/sagemaker-user/CNN_A2/data/{STUDENT_ID}/Object_Detection/yolo/data.yaml"
USE_PRETRAINED = True

output_dir = Path(f'outputs/detection_yolov8_v1')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EXPERIMENT V1: YOLOv8 Baseline")
print("=" * 80)
```

**Cell 3:** Step 1 - Load Dataset Configuration
```python
print("\n[1/5] Loading dataset configuration...")
with open(DATASET_CONFIG, 'r') as f:
    dataset_config = yaml.safe_load(f)

print(f'Classes: {dataset_config["nc"]} ({dataset_config["names"]})')
print(f'Train path: {dataset_config.get("train", "N/A")}')
print(f'Val path: {dataset_config.get("val", "N/A")}')
print(f'Test path: {dataset_config.get("test", "N/A")}')
```

**Cell 4:** Step 2 - Initialize Model
```python
print("\n[2/5] Initializing YOLOv8 model...")
model_config = YOLOV8_BASELINE_CONFIG_LOCAL.copy()
model_config['pretrained'] = USE_PRETRAINED

model = YOLOv8Detector(**model_config)
print(f'Model: Standard YOLOv8{model_config["backbone"]}')
print(f'Input size: {model_config["input_size"]}')
print(f'Pretrained: {USE_PRETRAINED}')
print(f'Customization: None (Baseline)')
```

**Cell 5:** Step 3 - Train
```python
print("\n[3/5] Training model...")
trainer = YOLOv8Trainer(**TRAINING_CONFIG_V1)

training_results = trainer.train(
    model=model,
    train_data=DATASET_CONFIG,
    val_data=DATASET_CONFIG,
    output_dir=str(output_dir)
)

print(f'Best mAP50: {training_results.box.map50:.4f}')
print(f'Best mAP50-95: {training_results.box.map:.4f}')
```

**Cell 6:** Step 4 - Evaluate
```python
print("\n[4/5] Evaluating on test set...")
evaluator = DetectionEvaluator()
metrics = evaluator.evaluate_yolov8(
    model=model,
    test_dataset=DATASET_CONFIG
)

print("\n=== Test Set Metrics ===")
print(f"mAP@0.5: {metrics['map50']:.4f}")
print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

**Cell 7:** Display Detection Results
```python
print("\n=== Detection Results Visualization ===")
fig = evaluator.plot_detection_results(model, DATASET_CONFIG, num_samples=5)
plt.show()
```

**Cell 8:** Display Training Curves
```python
print("\n=== Training Curves ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig = evaluator.plot_training_curves(training_results)
plt.show()
```

**Cell 9:** Display PR Curves
```python
print("\n=== Precision-Recall Curves ===")
fig = evaluator.plot_pr_curves(training_results)
plt.show()
```

**Cell 10:** Analysis & Comparison
```python
print("\n=== Model Analysis ===")
print(f"Backbone: YOLOv8{model_config['backbone']}")
print(f"Parameters: ~{model_param_count}M")
print(f"Customization: Baseline (no modifications)")
print(f"Best mAP@0.5: {training_results.box.map50:.4f}")
```

**Cell 11:** Final Summary
```python
print("\n" + "=" * 80)
print("EXPERIMENT COMPLETED")
print("=" * 80)
print(f"\nExperiment: V1 - YOLOv8 Baseline")
print(f"Model: YOLOv8{model_config['backbone']} (Standard)")
print(f"Pretrained: {USE_PRETRAINED}")
print(f"Best mAP@0.5: {training_results.box.map50:.4f}")
print(f"Best mAP@0.5:0.95: {training_results.box.map:.4f}")
print(f"Test mAP@0.5: {metrics['map50']:.4f}")
print(f"Output directory: {output_dir}")
```

---

### **detection_YOLOv8_v2.ipynb（Deeper Backbone）**

**Cell 2:** Configuration - 修改为 V2 配置
```python
YOLOV8_V2_CONFIG_LOCAL = {
    'backbone': 'm',
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
    'pretrained': True,
    'customize_type': 'deeper'  # Add conv layers
}

TRAINING_CONFIG_V2 = {
    'learning_rate': 0.0005,  # Lower LR for stability
    'batch_size': 12,         # Smaller batch for larger model
    'epochs': 300,
    'optimizer': 'adam',
    'weight_decay': 5e-4,     # Higher weight decay
    'use_amp': True,
    'patience': 50,
    'cos_lr': True,           # Cosine LR schedule
    'close_mosaic': 10        # Close mosaic last 10 epochs
}
```

**Cell 4:** Initialize Model - 添加自定义描述
```python
print('Customization: Added convolutional layers in backbone')
print('  - Location: After backbone layer2 (C2f module)')
print('  - Added layers: 2 Conv(128, 3x3, stride=1, padding=1)')
print('  - Purpose: Deepen shallow-layer feature extraction')
```

**Cell 11:** Final Summary - 修改实验名称和描述
```python
print(f"\nExperiment: V2 - YOLOv8 Deeper Backbone")
print(f"Model: Custom YOLOv8{model_config['backbone']} (Deeper)")
print(f"Customization: Added 2 Conv layers after backbone layer2")
```

---

### **detection_YOLOv8_v3.ipynb（Shallower Backbone）**

**Cell 2:** Configuration - 修改为 V3 配置
```python
YOLOV8_V3_CONFIG_LOCAL = {
    'backbone': 'm',
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
    'pretrained': True,
    'customize_type': 'shallower'  # Reduce conv layers
}

TRAINING_CONFIG_V3 = {
    'learning_rate': 0.001,
    'batch_size': 20,         # Larger batch for lighter model
    'epochs': 300,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'use_amp': True,
    'patience': 50,
    'cos_lr': False,
    'close_mosaic': 0
}
```

**Cell 4:** Initialize Model - 添加自定义描述
```python
print('Customization: Reduced backbone depth by modifying C2f modules')
print('  - Location: Backbone layer4 (P4/16 level)')
print('  - Change: Reduced C2f repeats from 6 to 3')
print('  - Benefit: Faster training, reduced overfitting risk')
```

**Cell 11:** Final Summary - 修改实验名称和描述
```python
print(f"\nExperiment: V3 - YOLOv8 Shallower Backbone")
print(f"Model: Custom YOLOv8{model_config['backbone']} (Shallower)")
print(f"Customization: Reduced C2f repeats in backbone layer4")
```

---

## 🔧 Key Code Modifications Required

### **1. 移除文件保存逻辑**

**YOLOv8Trainer.train():**
```python
# ❌ Removed: All artifact saving
# - No CSV files
# - No PNG files
# - No model checkpoints

# ✅ Return results object only
return results  # Ultralytics Results object
```

**DetectionEvaluator.evaluate_yolov8():**
```python
# ❌ Removed: 
# - json.dump(metrics)
# - CSV file writing
# - PNG file saving (confusion matrix, PR curves)

# ✅ Return metrics dict only
return metrics
```

**Removed Methods:**
- `generate_experiment_summary()` → Replaced with print statements in notebooks
- `organize_training_artifacts()` → No longer needed
- `plot_confusion_matrix()` → Will be implemented inline in notebooks

### **2. 配置分布**

**YOLOv8_modules.ipynb:**
- ✅ 包含所有模型配置字典（YOLOV8_BASELINE_CONFIG, V2, V3）
- ✅ 包含所有训练配置字典（YOLOV8_V1_CONFIG, V2, V3）
- ❌ 不包含具体的实验设置（DATASET_CONFIG, USE_PRETRAINED）

**实验 Notebooks（Cell 2）:**
- ✅ 包含专属的模型配置字典（local copy）
- ✅ 包含专属的训练配置字典（local copy）
- ✅ 包含实验设置（DATASET_CONFIG, USE_PRETRAINED, output_dir）
- ✅ 完全自包含，可独立运行

### **3. 检测任务特殊处理**

**Bounding Box 可视化：**
```python
def plot_detection_results(self, model, test_dataset, num_samples=5):
    """
    Plot detection results with bounding boxes inline.
    """
    import cv2
    import matplotlib.pyplot as plt
    
    # Load sample images from test dataset
    # Run inference
    # Draw bounding boxes on images
    # Return matplotlib figure with detections
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    for i, ax in enumerate(axes):
        # Load image
        # Run model inference
        # Draw boxes and labels
        ax.imshow(image)
        ax.set_title(f"Detected: {class_name} ({conf:.2f})")
        ax.axis('off')
    
    plt.tight_layout()
    return fig
```

**训练曲线绘制：**
```python
def plot_training_curves(self, training_results):
    """
    Plot training curves from Ultralytics results.
    """
    # Extract metrics from results.csv or results object
    # Plot train/val loss, mAP50, mAP50-95
    # Return fig object for inline display
    pass
```

**PR 曲线绘制：**
```python
def plot_pr_curves(self, training_results):
    """
    Plot Precision-Recall curves for each class.
    """
    # Extract PR data from results
    # Plot PR curves using matplotlib
    # Return fig object for inline display
    pass
```

---

## ️ Expected Challenges & Solutions

### **Challenge 1: Ultralytics API Compatibility**

**问题：**  
Ultralytics 版本更新可能导致 API 变化（如 `results.box.map` 属性访问）。

**解决方案：**
- 使用 try-except 兼容不同版本
- 添加版本检测和回退逻辑
```python
try:
    map50 = results.box.map50
except AttributeError:
    map50 = results.metrics['map50']
```

### **Challenge 2: Bounding Box Visualization**

**问题：**  
检测任务的可视化比分类任务复杂，需要绘制 bounding boxes、类别标签和置信度。

**解决方案：**
- 使用 OpenCV 或 matplotlib 的 patches.Rectangle 绘制检测框
- 参考 Ultralytics 内置的可视化方法（如 `model.predict()` 返回的结果）
- 确保颜色编码和标签清晰可读

### **Challenge 3: Dataset Path Configuration**

**问题：**  
在 GPU 服务器上运行时，数据集路径可能不正确。

**解决方案：**
- 使用绝对路径：`/home/sagemaker-user/CNN_A2/data/25509225/Object_Detection/yolo/data.yaml`
- 在 Cell 2 中提供可编辑的 DATASET_CONFIG 变量
- 添加路径验证和错误提示

### **Challenge 4: Training Results Extraction**

**问题：**  
Ultralytics 的 training results 对象结构可能因版本而异。

**解决方案：**
- 使用 `results.results_file` 访问 CSV 文件（但不保存）
- 直接从 results 对象提取关键指标（map50, map, precision, recall）
- 添加容错处理和默认值

### **Challenge 5: Memory Management**

**问题：**  
检测任务（尤其是 YOLOv8m）比分类任务更占用显存。

**解决方案：**
- 根据实验配置调整 batch_size（V1: 16, V2: 12, V3: 20）
- 启用混合精度训练（use_amp=True）
- 在可视化时限制样本数量（num_samples=5）

---

## ✅ Implementation Checklist

- [ ] **Step 1:** Create YOLOv8_modules.ipynb
  - [ ] Cell 1: Imports & Setup
  - [ ] Cell 2: YOLOv8Detector Model
  - [ ] Cell 3: Model Configurations
  - [ ] Cell 4: YOLOv8Trainer
  - [ ] Cell 5: Training Configurations
  - [ ] Cell 6: DetectionEvaluator

- [ ] **Step 2:** Create detection_YOLOv8_v1.ipynb
  - [ ] 11 Cells following the structure
  - [ ] Test inline visualization

- [ ] **Step 3:** Create detection_YOLOv8_v2.ipynb
  - [ ] 11 Cells with V2 configuration
  - [ ] Customization descriptions

- [ ] **Step 4:** Create detection_YOLOv8_v3.ipynb
  - [ ] 11 Cells with V3 configuration
  - [ ] Customization descriptions

- [ ] **Step 5:** Verification
  - [ ] All notebooks can run independently (after loading modules)
  - [ ] All outputs display inline
  - [ ] No file saving (except as required)
  - [ ] Test one complete experiment flow

---

## 📊 Expected Output

每个实验 notebook 运行完成后，输出包括：

1. **实验配置信息**
2. **数据集统计**（类别数、路径）
3. **模型信息**（backbone、输入尺寸、预训练、自定义类型）
4. **训练过程**（每个 epoch 的 loss 和 mAP）
5. **测试集评估指标**（mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1）
6. **检测结果可视化**（inline bounding box plots）
7. **训练曲线**（inline loss/mAP curves）
8. **PR 曲线**（inline Precision-Recall curves）
9. **模型分析**（参数量、自定义描述）
10. **最终总结**（所有关键指标）

---

## 🔄 Recovery Instructions

如果执行中断：

1. 阅读本文件了解已完成的工作
2. 检查哪些 notebooks 已成功创建
3. 从下一个未完成的步骤继续
4. 每完成一步后与用户确认

---

**Last Updated:** 2026-05-06  
**Status:** Ready to execute Step 1  
**Next Action:** Create YOLOv8_modules.ipynb
