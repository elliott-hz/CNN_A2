# Experiments Analysis Report

**Run:** train_history-run-1  
**Date:** 2026-05-02  
**Pretrained:** False (Training from scratch)  
**Student ID:** 25509225

---

## Common Training Configuration

All experiments use the same training configuration:

```python
learning_rate = 1e-4
weight_decay = 1e-4 (Baseline/V3) or 1e-3 (V1/V2)
optimizer = AdamW
epochs = 200
warmup_epochs = 5 (linear warmup)
scheduler = ReduceLROnPlateau (patience=7, factor=0.5)
early_stopping_patience = 50
label_smoothing = 0.1
batch_size = 16
mixed_precision = True
```

---

## Model Configurations

### Baseline
- **Architecture:** Standard ResNet50
- **FC Head:** 2048 → 10 (single layer)
- **Dropout:** 0.5
- **Backbone Modification:** None
- **Total Params:** ~25.6M

### V1 (Enhanced FC)
- **Architecture:** ResNet50 + Enhanced FC
- **FC Head:** 2048 → 512 → 256 → 10 (with BatchNorm)
- **Dropout:** 0.5
- **Weight Decay:** 1e-3 (stronger regularization)
- **Backbone Modification:** None
- **Total Params:** ~25.6M

### V2 (TRUE CNN - Add Conv)
- **Architecture:** ResNet50 + Conv blocks after layer2
- **FC Head:** 2048 → 512 → 256 → 10 (with BatchNorm)
- **Dropout:** 0.5
- **Weight Decay:** 1e-3
- **Backbone Modification:** Added conv blocks after layer2
- **Total Params:** ~26.2M

### V3 (TRUE CNN - Remove Layer)
- **Architecture:** ResNet50 - layer3 (reduced depth)
- **FC Head:** 2048 → 10 (single layer)
- **Dropout:** 0.5
- **Weight Decay:** 1e-4
- **Backbone Modification:** Removed layer3
- **Total Params:** ~16.4M

---

## Training Results Summary

| Experiment | Epochs Trained | Best Val Acc | Final Val Acc | Final Train Acc | Notes |
|------------|---------------|--------------|---------------|-----------------|-------|
| **Baseline** | 109 | 0.9264 | 0.9004 | 1.0000 | Early stopped at epoch 109 |
| **V1** | 126 | 0.9394 | 0.9091 | 1.0000 | Early stopped at epoch 126 |
| **V2** | 104 | 0.9351 | 0.9177 | 1.0000 | Early stopped at epoch 104 |
| **V3** | 72 | 0.9524 | 0.9437 | 1.0000 | Early stopped at epoch 72 |

---

## Loss & Accuracy Curve Analysis

### 1. Baseline Experiment

**Training Phases:**

**Phase 1: Warmup (Epochs 1-5)**
- Learning rate linearly increases from 2e-5 to 1e-4
- Train loss drops rapidly: 2.37 → 1.52
- Train accuracy improves: 14.9% → 56.2%
- Validation accuracy follows similar trend: 27.3% → 52.0%

**Phase 2: Rapid Learning (Epochs 6-25)**
- Stable learning rate at 1e-4
- Train loss continues decreasing: 1.41 → 0.66
- Train accuracy reaches near-perfect: 58.8% → 97.7%
- Validation accuracy peaks around epoch 26: 90.5%
- Gap between train/val starts widening (overfitting begins)

**Phase 3: Plateau with LR Reductions (Epochs 26-109)**
- Multiple LR reductions triggered by scheduler:
  - Epoch 58: 1e-4 → 5e-5
  - Epoch 74: 5e-5 → 2.5e-5
  - Epoch 82: 2.5e-5 → 1.3e-5
  - Epoch 90: 1.3e-5 → 6.5e-6
  - Epoch 101: 6.5e-6 → 3.3e-6
  - Epoch 109: 3.3e-6 → 1.6e-6
- Train accuracy reaches 100% by epoch 62
- Validation accuracy fluctuates between 88-93%
- Train loss stabilizes around 0.53, val loss around 0.74-0.76
- Early stopping triggered at epoch 109 (no improvement for 50 epochs)

**Overfitting Analysis:**
- ⚠️ **Clear overfitting pattern**: Train acc 100% vs Val acc ~90%
- Train-val gap: ~10% accuracy difference
- Train loss (0.53) significantly lower than val loss (0.74)
- Model memorizes training data but struggles to generalize

---

### 2. V1 Experiment (Enhanced FC Head)

**Training Phases:**

**Phase 1: Slow Start (Epochs 1-10)**
- Slower initial convergence compared to Baseline
- Train loss: 2.47 → 1.71
- Train accuracy: 9.8% → 47.4%
- Validation accuracy: 10.0% → 55.8%
- Enhanced FC head requires more time to learn

**Phase 2: Steady Improvement (Epochs 11-35)**
- Gradual but consistent progress
- Train accuracy reaches 90.8% by epoch 35
- Validation accuracy peaks at 87.9% (epoch 28)
- Higher weight decay (1e-3) provides stronger regularization

**Phase 3: Extended Training with LR Schedule (Epochs 36-126)**
- Multiple LR reductions:
  - Epoch 36: 1e-4 → 5e-5
  - Epoch 51: 5e-5 → 2.5e-5
  - Epoch 64: 2.5e-5 → 1.3e-5
  - Epoch 72: 1.3e-5 → 6.5e-6
  - Epoch 82: 6.5e-6 → 3.3e-6
  - Epoch 90: 3.3e-6 → 1.6e-6
  - Further reductions continue until ~1e-7
- Best validation accuracy achieved at epoch 76: **93.94%**
- Train accuracy reaches 100% by epoch 87
- Longest training duration (126 epochs) among all experiments

**Overfitting Analysis:**
- ⚠️ **Moderate overfitting**: Train acc 100% vs Val acc ~91%
- Train-val gap: ~9% accuracy difference
- Stronger regularization (weight_decay=1e-3) helps control overfitting
- Enhanced FC head adds capacity but requires careful tuning

**Key Observation:**
- Despite longer training and enhanced architecture, V1 only marginally outperforms Baseline
- Suggests that FC modifications alone are insufficient for significant gains

---

### 3. V2 Experiment (TRUE CNN - Added Conv Blocks)

**Training Phases:**

**Phase 1: Moderate Start (Epochs 1-10)**
- Similar initial pace to Baseline
- Train loss: 2.41 → 1.57
- Train accuracy: 11.4% → 51.6%
- Validation accuracy: 17.7% → 64.1%

**Phase 2: Consistent Learning (Epochs 11-35)**
- Steady improvement in both metrics
- Train accuracy reaches 92.9% by epoch 34
- Validation accuracy fluctuates around 80-88%
- Backbone modification shows stable learning behavior

**Phase 3: Refinement Phase (Epochs 36-104)**
- LR reduction schedule:
  - Epoch 36: 1e-4 → 5e-5
  - Epoch 62: 5e-5 → 2.5e-5
  - Epoch 70: 2.5e-5 → 1.3e-5
  - Epoch 78: 1.3e-5 → 6.5e-6
  - Epoch 90: 6.5e-6 → 3.3e-6
  - Epoch 99: 3.3e-6 → 1.6e-6
- Best validation accuracy at epoch 54: **93.51%**
- Train accuracy reaches 100% by epoch 79
- Early stopping at epoch 104

**Overfitting Analysis:**
- ⚠️ **Moderate overfitting**: Train acc 100% vs Val acc ~92%
- Train-val gap: ~8% accuracy difference
- Adding convolutional blocks increases model complexity
- Performance comparable to V1, suggesting architectural changes need more tuning

**Key Observation:**
- TRUE CNN customization (added conv blocks) doesn't significantly outperform simpler approaches
- May require different hyperparameters or more training data to show advantages

---

### 4. V3 Experiment (TRUE CNN - Removed Layer3) ⭐ BEST

**Training Phases:**

**Phase 1: Fast Convergence (Epochs 1-10)**
- **Fastest initial learning** among all experiments
- Train loss: 2.32 → 0.88
- Train accuracy: 20.2% → 88.5%
- Validation accuracy: 44.6% → 82.3%
- Reduced model depth accelerates feature learning

**Phase 2: Rapid Saturation (Epochs 11-25)**
- Quick approach to high accuracy
- Train accuracy reaches 99.1% by epoch 25
- Validation accuracy peaks early: 93.9% (epoch 29)
- Smaller model (16.4M params) learns efficiently

**Phase 3: Fine-tuning (Epochs 26-72)**
- LR reductions:
  - Epoch 37: 1e-4 → 5e-5
  - Epoch 47: 5e-5 → 2.5e-5
  - Epoch 69: 2.5e-5 → 1.3e-5
- Best validation accuracy at epochs 38, 48, 54, 64, 70: **95.24%**
- Train accuracy reaches 100% by epoch 35 (earliest among all)
- Shortest training duration (72 epochs)

**Overfitting Analysis:**
- ✅ **Best generalization**: Train acc 100% vs Val acc ~94-95%
- Train-val gap: Only ~5-6% (smallest among all experiments)
- Reduced model complexity prevents severe overfitting
- Lower parameter count acts as implicit regularization

**Key Observations:**
1. **Efficiency**: Achieves best performance in shortest time (72 epochs vs 104-126)
2. **Generalization**: Smallest train-val gap indicates better generalization
3. **Simplicity wins**: Removing layer3 reduces complexity while maintaining representational power
4. **Resource-friendly**: 16.4M params vs 25-26M in other models (~36% reduction)

---

## Comparative Analysis

### Performance Ranking

| Rank | Experiment | Best Val Acc | Training Efficiency | Generalization |
|------|-----------|--------------|-------------------|----------------|
| 🥇 1st | **V3** | 95.24% | ⭐⭐⭐⭐⭐ (72 epochs) | ⭐⭐⭐⭐⭐ (5-6% gap) |
| 🥈 2nd | **V1** | 93.94% | ⭐⭐ (126 epochs) | ⭐⭐⭐ (9% gap) |
| 🥉 3rd | **V2** | 93.51% | ⭐⭐⭐ (104 epochs) | ⭐⭐⭐ (8% gap) |
| 4th | **Baseline** | 92.64% | ⭐⭐⭐ (109 epochs) | ⭐⭐ (10% gap) |

### Key Insights

#### 1. **Model Complexity vs Performance**
- **Counter-intuitive finding**: Simpler model (V3) outperforms complex ones
- V3's reduced depth (removed layer3) prevents over-parameterization
- Larger models (V1, V2) suffer from overfitting despite regularization

#### 2. **Training Dynamics**
- All experiments show similar learning patterns:
  - Fast initial learning (epochs 1-25)
  - Gradual saturation (epochs 26-50)
  - Fine-tuning with LR reductions (epochs 50+)
- V3 converges fastest due to smaller search space

#### 3. **Overfitting Patterns**
- **Common issue**: All models reach 100% train accuracy
- **Severity varies**:
  - Baseline: Worst (10% gap)
  - V1/V2: Moderate (8-9% gap)
  - V3: Best (5-6% gap)
- **Root cause**: Small dataset (1,589 images) + powerful models

#### 4. **Learning Rate Schedule Effectiveness**
- ReduceLROnPlateau successfully triggers multiple reductions
- Helps escape local minima and fine-tune weights
- V3 benefits most (fewer reductions needed)

#### 5. **Architectural Modifications Impact**
- **FC enhancement (V1)**: Marginal improvement (+1.3% vs Baseline)
- **Added conv blocks (V2)**: Similar to V1 (+0.9% vs Baseline)
- **Removed layer (V3)**: Significant gain (+2.6% vs Baseline)
- **Conclusion**: Reducing complexity > Adding complexity for this dataset

---

## Recommendations for Future Work

Based on current results, here are actionable next steps:

### 1. **Enable Data Augmentation (Highest Priority)** ✅ Ready to Run

**Action:** Re-run all 4 experiments with `--dataAugmentation enhanced`
```bash
python3 experiments/classification_ResNet50_baseline.py --pretrained False --dataAugmentation enhanced
python3 experiments/classification_ResNet50_v1.py --pretrained False --dataAugmentation enhanced
python3 experiments/classification_ResNet50_v2.py --pretrained False --dataAugmentation enhanced
python3 experiments/classification_ResNet50_v3.py --pretrained False --dataAugmentation enhanced
```

**Expected Impact:** Should reduce overfitting by 3-5% and potentially reach 97-98% validation accuracy. No code changes needed.

---

### 2. **Optimize V3 Architecture**

**Try removing layer4 instead of layer3:**
- **Layer3 removal (current V3):** Removes middle-depth features, keeps high-level semantic features from layer4
- **Layer4 removal (proposed):** Removes highest-level abstract features, keeps more spatial details from layer3
- **Which is better?** Depends on dataset characteristics. For bird classification, layer4's semantic features may be more important, so layer3 removal works well. Test layer4 removal to compare.

**Implementation:** Create a new config in `ResNet50ClassifierModel.py`:
```python
CUSTOMIZED_V4_CONFIG = {
    'num_classes': 10,
    'dropout_rate': 0.5,
    'pretrained': False,
    'additional_fc_layers': False,
    'use_batch_norm': True,
    'modify_backbone': True,
    'remove_layer': 'layer4',  # Try removing layer4 instead
    'add_conv_after_layer': None
}
```

---

### 3. **Adjust Training Hyperparameters**

**Increase initial learning rate to 1e-3:**
- Current: 1e-4 (conservative for fine-tuning)
- Recommended: 1e-3 (better for training from scratch)
- Why: From-scratch training needs higher LR to learn features quickly

**Extend warmup from 5 to 10 epochs:**
- Current: 5 epochs warmup
- Recommended: 10 epochs warmup
- Why: Smoother transition prevents early instability with higher LR

**Keep ReduceLROnPlateau (don't switch to cosine annealing):**
- Current scheduler works well
- Cosine annealing is alternative but not necessary
- ReduceLROnPlateau adapts to validation performance automatically

**Implementation:** Modify training configs in `classification_trainer.py` or pass custom parameters when creating trainer.

---

### 4. **Focus on ResNet50 Variants Only**

**Stay within current experimental framework:**
- ✅ Baseline: Standard ResNet50
- ✅ V1: Enhanced FC head
- ✅ V2: Add conv blocks (backbone expansion)
- ✅ V3: Remove layer3 (backbone reduction)
- ✅ Proposed V4: Remove layer4 (alternative reduction)

**Do NOT explore:**
- ❌ Other architectures (MobileNet, EfficientNet, etc.)
- ❌ Knowledge distillation
- ❌ Attention mechanisms
- ❌ Complex augmentation strategies (mixup/cutmix)

**Rationale:** Assignment requires demonstrating understanding of CNN architecture customization through ResNet50 modifications, not exploring entirely different models.

---

### 5. **Key Insight: "Less is More"**

**What it means:**
- For small datasets (1,589 images), simpler models generalize better
- V3 (16.4M params) outperforms larger models (25-26M params)
- Reducing complexity acts as implicit regularization
- Prevents model from memorizing noise in limited data

**Practical implication:**
- When data is scarce, prefer architectural simplification over enhancement
- Focus on preventing overfitting rather than increasing capacity
- Data augmentation becomes critical to artificially expand effective dataset size

---

## Immediate Next Steps

1. **Run all 4 experiments with enhanced augmentation** (no code changes)
2. **Compare results** - expect 3-5% improvement in validation accuracy
3. **If still overfitting**, try V4 (remove layer4) configuration
4. **Fine-tune hyperparameters** (LR=1e-3, warmup=10) if needed

**Priority Order:** Data Augmentation > Architecture Tuning > Hyperparameter Adjustment

---

## Files Generated

- `training_history-baseline.csv` - 109 epochs
- `training_history-v1.csv` - 126 epochs
- `training_history-v2.csv` - 104 epochs
- `training_history-v3.csv` - 72 epochs

Each CSV contains: epoch, train_loss, train_acc, val_loss, val_acc, lr

---

## Summary

**Winner: V3 (ResNet50 without layer3)**
- ✅ Best validation accuracy: 95.24%
- ✅ Fastest convergence: 72 epochs
- ✅ Best generalization: smallest train-val gap
- ✅ Most efficient: 36% fewer parameters

**Main Challenge: Overfitting**
- All models memorize training data (100% train acc)
- Validation performance plateaus around 90-95%
- Solution: Data augmentation + architectural simplification

**Key Lesson:** For small datasets, simpler models with proper regularization often outperform complex architectures.
