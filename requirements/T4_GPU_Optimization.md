# T4 GPU Optimization Guide

**Student ID:** 25509225  
**GPU:** NVIDIA T4 (16GB VRAM, ~10GB usable)  
**Last Updated:** 2026-04-30

---

## Overview

All experiments have been optimized for NVIDIA T4 GPU with 16GB VRAM (approximately 10GB usable during training due to system overhead).

---

## Batch Size Recommendations

| Experiment | Model | Recommended Batch Size | Notes |
|------------|-------|------------------------|-------|
| Exp01 | YOLOv8m | **16** | Reduced from 24 |
| Exp02 | Faster R-CNN | **2** | Very memory-intensive, reduced from 4 |
| Exp03 | ResNet50 Baseline | **16** | Reduced from 32 |
| Exp04 | ResNet50 Customized | **16** | Reduced from 32 |

---

## Memory Optimization Strategies

### 1. Mixed Precision Training (AMP)

All classification experiments use Automatic Mixed Precision (AMP):
```python
use_amp = True  # Enabled by default in ClassificationTrainer
```

**Benefits:**
- Reduces memory usage by ~50%
- Speeds up training by 2-3x on T4 GPU
- No accuracy loss

### 2. Reduced num_workers

DataLoaders use `num_workers=2` instead of 4:
```python
DataLoader(dataset, batch_size=16, num_workers=2, ...)
```

**Reason:** Prevents CPU memory pressure that can cause OOM on GPU.

### 3. Gradient Accumulation (if needed)

If you still encounter OOM errors, enable gradient accumulation:

```python
# In experiment script
GRADIENT_ACCUMULATION_STEPS = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
```

This allows using larger effective batch sizes without increasing memory usage.

---

## Setup Instructions

### 1. Initialize GPU Environment

```bash
chmod +x setupGPU.sh
bash setupGPU.sh
```

This will:
- Install PyTorch with CUDA 11.8 support
- Install all dependencies
- Verify GPU availability
- Display batch size recommendations

### 2. Verify GPU Setup

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA: True
GPU: Tesla T4
```

---

## Running Experiments

### Quick Start

```bash
# Detection experiments
python experiments/exp01_detection_YOLOv8.py          # ~16GB peak memory
python experiments/exp02_detection_FasterRCNN.py      # ~10GB peak memory

# Classification experiments
python experiments/exp03_classification_ResNet50_v1.py # ~8GB peak memory
python experiments/exp04_classification_ResNet50_v2.py # ~8GB peak memory
```

### Monitor GPU Usage

During training, monitor GPU memory in another terminal:

```bash
watch -n 1 nvidia-smi
```

Or use:
```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv -l 1
```

---

## Troubleshooting OOM Errors

### Error: CUDA out of memory

**Solution 1: Reduce batch size**
```python
BATCH_SIZE = 8  # Instead of 16
```

**Solution 2: Enable gradient accumulation**
```python
# Modify trainer to accumulate gradients
for i, (inputs, targets) in enumerate(train_loader):
    # ... forward pass ...
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Solution 3: Reduce image size**
```python
# For classification
transforms.Resize(128),  # Instead of 256
transforms.CenterCrop(128),  # Instead of 224
```

**Solution 4: Close other GPU processes**
```bash
# Check what's using GPU
nvidia-smi

# Kill unnecessary processes
kill -9 <PID>
```

---

## Performance Expectations

### Training Time Estimates (T4 GPU)

| Experiment | Epochs | Batch Size | Estimated Time |
|------------|--------|------------|----------------|
| Exp01 (YOLOv8) | 100 | 16 | ~2-3 hours |
| Exp02 (Faster R-CNN) | 50 | 2 | ~4-6 hours |
| Exp03 (ResNet50) | 50 | 16 | ~1-2 hours |
| Exp04 (ResNet50) | 60 | 16 | ~1.5-2.5 hours |

*Times are approximate and depend on dataset size and complexity.*

---

## Key Optimizations Applied

### 1. YOLOv8 (exp01)
- ✅ Batch size: 24 → **16**
- ✅ AMP enabled (mixed precision)
- ✅ Image size: 640px (optimal for T4)

### 2. Faster R-CNN (exp02)
- ✅ Batch size: 4 → **2** (critical for T4)
- ✅ Two-stage detector is memory-intensive
- ✅ Consider reducing epochs if training too slow

### 3. ResNet50 (exp03 & exp04)
- ✅ Batch size: 32 → **16**
- ✅ num_workers: 4 → **2**
- ✅ AMP enabled in trainer
- ✅ Pin memory enabled for faster data transfer

---

## Best Practices

1. **Always run setupGPU.sh first** to ensure proper CUDA installation
2. **Monitor GPU memory** during first few epochs
3. **Start with recommended batch sizes**, reduce if OOM occurs
4. **Use AMP** for all training (already enabled)
5. **Close browser tabs and other apps** to free up system memory
6. **Run one experiment at a time** to avoid GPU memory conflicts

---

## Comparison: T4 vs Other GPUs

| GPU | VRAM | Usable | Max Batch (ResNet50) |
|-----|------|--------|----------------------|
| T4 | 16GB | ~10GB | 16 |
| V100 | 32GB | ~28GB | 64 |
| A100 | 40GB | ~36GB | 128 |
| RTX 3090 | 24GB | ~20GB | 32 |

T4 is suitable for all experiments but requires careful batch size management.

---

## Additional Resources

- [PyTorch Mixed Precision Training](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [NVIDIA T4 Specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [CUDA Out of Memory Troubleshooting](https://pytorch.org/docs/stable/notes/cuda.html#out-of-memory-error)

---

**Note:** If you consistently hit OOM errors even with reduced batch sizes, consider using gradient checkpointing or switching to a smaller model backbone.
