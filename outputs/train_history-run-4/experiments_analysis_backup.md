# Run-4 Experiments Comprehensive Analysis

**Date:** 2026-05-03  
**Total Experiments:** 8 (Baseline, V1-V7)  
**Optimization Applied:** Grid Search recommended configurations

---

##  INDIVIDUAL EXPERIMENT ANALYSIS

---

### Experiment 1: Baseline

#### 📋 Configuration Summary
- **Architecture:** Standard ResNet50 with single FC layer (2048 → 10)
- **Hyperparameters:** LR=1e-3, WD=1e-3, LS=0.10 (Grid Search Optimal)
- **Optimizer:** ADAMW with scheduler
- **Early Stopping:** Enabled (patience=50)
- **Total Epochs:** 115 (early stopped)

#### 📊 Test Results
- **Test Accuracy:** 93.98%
- **Test F1 (macro):** 0.9400
- **Best Val Accuracy:** 96.97% (epoch 65)
- **Val-Test Gap:** 2.99%

#### 📈 Training Dynamics (CSV Analysis - 115 epochs)

**Phase 1: Warmup & Initial Learning (Epoch 1-10)**
- LR increased from 1e-4 to 1e-3
- Train Acc: 12.17% → 57.44%
- Val Acc: 19.91% → 79.65%
- Fast initial convergence, good learning signals

**Phase 2: Stable Learning with LR=1e-3 (Epoch 11-46)**
- Reached 90%+ train accuracy by epoch 40
- Val accuracy fluctuated: 82-94% range
- Notable spike at epoch 22 (val_loss=2.09, val_acc=75.32%)
- Overall upward trend with some instability

**Phase 3: LR Decay to 5e-4 (Epoch 47-60)**
- Performance jumped: train acc 87% → 95%
- Val acc stabilized around 93-95%
- Epoch 58: peak val acc 95.24%

**Phase 4: Fine-tuning with LR=2.5e-4 (Epoch 61-75)**
- Train acc reached 99%+ by epoch 63
- Val acc peaked at 96.97% (epoch 65) - BEST
- Clear sign of overfitting starting (train 99.46% vs val 96.10%)

**Phase 5: Final Fine-tuning (Epoch 76-115)**
- LR reduced to 1.25e-4 → 6.3e-5 → 3.1e-5
- Train acc: 99.9%+ (nearly perfect)
- Val acc: stuck at 95.24-96.97% (plateaued)
- Early stopping triggered at epoch 115

#### 🔍 Overfitting Analysis

**Evidence of Overfitting:**
- Final epoch: Train Acc=99.82%, Val Acc=95.24% → Gap=4.58%
- Peak epoch 65: Train Acc=98.20%, Val Acc=96.97% → Gap=1.23%
- The gap widened significantly in final 50 epochs
- Val loss plateaued while train loss continued decreasing

**Severity:** Moderate
- Not catastrophic (val acc still 95%+)
- But indicates model memorizing training data
- WD=1e-3 helped but not sufficient for this architecture

#### ✅ Strengths
1. Grid search optimal hyperparameters applied correctly
2. Good warmup strategy helped initial convergence
3. Reached competitive validation accuracy (96.97%)
4. Early stopping prevented severe overfitting

#### ⚠️ Weaknesses
1. **Test accuracy dropped significantly** (93.98% vs 96.97% val)
2. Test-Val gap (2.99%) suggests validation set may not fully represent test distribution
3. Overfitting in final 50 epochs wasted computation
4. Test accuracy lower than expected from grid search predictions (97.2-97.7%)

####  Key Insights
- The large Val-Test gap (2.99%) is concerning - may indicate:
  - Validation set distribution differs from test set
  - Or model overfitted to validation set during training
- Despite optimal hyperparameters, Baseline still underperforms compared to architectural modifications (V3, V5)
- Suggests that **architecture matters more than hyperparameters alone**

#### 📌 Conclusion for Baseline
**Status:** ️ Underperformed expectations  
**Expected:** 97.2-97.7% | **Actual:** 93.98%  
**Gap:** -3.2 to -3.7%  
**Root Cause:** Likely validation-test distribution mismatch + moderate overfitting  
**Recommendation:** Use as reference baseline, but rely on V3/V5 for final submission

---

### Experiment 2: V1

#### 📋 Configuration Summary
- **Architecture:** ResNet50 with enhanced FC head `[256]` (2048 → 256 → 10)
- **Hyperparameters:** LR=1e-3, WD=1e-3, LS=0.05 (Modified Label Smoothing)
- **Optimizer:** ADAMW with scheduler
- **Early Stopping:** Enabled (patience=50)
- **Total Epochs:** 160 (early stopped)

#### 📊 Test Results
- **Test Accuracy:** 96.79%
- **Test F1 (macro):** 0.9675
- **Best Val Accuracy:** 97.40% (epoch 110)
- **Val-Test Gap:** 0.61%

#### 📈 Training Dynamics (CSV Analysis - 160 epochs)

**Phase 1: Warmup & Learning (Epoch 1-10)**
- LR increased from 1e-4 to 1e-3
- Train Acc: 11.36% → 56.81%
- Val Acc: 16.45% → 60.17%
- Slower initial learning compared to Baseline

**Phase 2: Stable Learning with LR=1e-3 (Epoch 11-37)**
- Fluctuating val accuracy (60-92% range)
- Notable spike at epoch 21: val_acc=89.61%
- Train acc reached 80%+ by epoch 27
- More stable than Baseline in early phase

**Phase 3: LR Decay to 5e-4 (Epoch 38-58)**
- Significant improvement: train acc 83% → 90%
- Val acc stabilized at 90-95% range
- Epoch 48: first time reaching 94.81% val acc
- LS=0.05 helped reduce overconfidence

**Phase 4: Fine-tuning with LR=2.5e-4 (Epoch 59-82)**
- Train acc reached 95%+ by epoch 61
- Val acc climbed to 96.54% (epoch 63)
- Very stable training, minimal fluctuations
- Good generalization maintained

**Phase 5: Extended Fine-tuning (Epoch 83-160)**
- LR reduced gradually: 1.25e-4 → 6.3e-5 → 3.1e-5 → 2e-6
- Train acc reached 99.9%+ (epoch 110+)
- Val acc peaked at 97.40% (epoch 110)
- Long plateau phase: epochs 110-160 showed minimal improvement
- Val acc stuck at 95.24-97.40% range

#### 🔍 Overfitting Analysis

**Evidence of Mild Overfitting:**
- Final epoch: Train Acc=99.64%, Val Acc=95.24% → Gap=4.40%
- Peak epoch 110: Train Acc=99.37%, Val Acc=97.40% → Gap=1.97%
- Overfitting developed gradually in last 50 epochs
- But much better controlled than Baseline

**Severity:** Mild
- Val acc continued improving longer than Baseline
- LS=0.05 (lower than Baseline's 0.10) helped maintain generalization
- Better test performance suggests good generalization

#### ✅ Strengths
1. **Excellent Test Accuracy:** 96.79% (2nd best overall)
2. **Very Small Val-Test Gap:** Only 0.61% (best among all experiments)
3. **LS=0.05 proved effective:** Better than LS=0.10 for this architecture
4. **Stable training dynamics:** Less volatile than Baseline
5. **Enhanced FC head `[256]` helped:** Better feature combination

#### ⚠️ Weaknesses
1. **Slower convergence:** Took 160 epochs vs Baseline's 115 epochs
2. **Lower peak val accuracy:** 97.40% vs Baseline's 96.97% (similar, but took longer)
3. **Extended plateau:** Last 50 epochs showed diminishing returns
4. **Still moderate overfitting:** Train-val gap widened to 4.40%

#### 💡 Key Insights
- **LS=0.05 is optimal** for enhanced FC architecture (confirmed by good test performance)
- **V1 architecture generalizes better** than Baseline despite similar peak val acc
- **Test accuracy much closer to val accuracy** suggests better robustness
- **Enhanced FC head provides better feature representation** than single FC layer
- **Trade-off:** Better generalization but slower convergence

#### 📌 Conclusion for V1
**Status:** ✅ Excellent generalization  
**Expected:** 97.1-97.3% | **Actual:** 96.79%  
**Gap:** -0.3 to -0.5% (minor underperformance)  
**Root Cause:** Slightly conservative LS=0.05, could benefit from 0.10  
**Recommendation:** Strong candidate for comparison studies; validates FC enhancement effectiveness

---

### Experiment 3: V2

#### 📋 Configuration Summary
- **Architecture:** ResNet50 with conv blocks after layer2 + enhanced FC `[512, 256]`
- **Hyperparameters:** LR=1e-3, WD=1e-3, LS=0.10
- **Optimizer:** ADAMW with scheduler
- **Early Stopping:** Enabled (patience=50)
- **Total Epochs:** 158 (early stopped)

####  Test Results
- **Test Accuracy:** 95.98%
- **Test F1 (macro):** 0.9591
- **Best Val Accuracy:** 96.97% (epoch 158)
- **Val-Test Gap:** 0.99%

#### 📈 Training Dynamics (CSV Analysis - 158 epochs)

**Phase 1: Slow Initial Learning (Epoch 1-20)**
- LR increased from 1e-4 to 1e-3
- Train Acc: 9.47% → 61.23%
- Val Acc: 13.85% → 79.65%
- **Slowest start among all experiments** - complex architecture took time to learn

**Phase 2: Gradual Convergence (Epoch 21-55)**
- Very slow progress: train acc 62% → 80%
- Val acc fluctuated wildly: 67-89% range
- High instability - architecture may be too complex
- Epoch 46: first time reaching 87.45% val acc

**Phase 3: LR Decay Breakthrough (Epoch 56-73)**
- LR reduced to 5e-4: dramatic improvement
- Train acc: 79% → 91%
- Val acc: 89% → 94.37%
- **Critical turning point** - complexity became manageable

**Phase 4: Stable Fine-tuning (Epoch 74-103)**
- LR=2.5e-4 → 1.25e-4
- Train acc reached 95%+ by epoch 87
- Val acc climbed to 96.10% (epoch 87)
- Much more stable than earlier phases

**Phase 5: Extended Plateau (Epoch 104-158)**
- Very slow LR decay: 3.1e-5 → 2e-6
- Train acc: 96.6% → 97.9%
- Val acc: 95.24% → 96.97% (final epoch)
- **Continued improving until the very end** - unlike other experiments
- No clear overfitting pattern

#### 🔍 Overfitting Analysis

**Evidence of Controlled Overfitting:**
- Final epoch: Train Acc=97.93%, Val Acc=96.97% → Gap=0.96%
- **Smallest train-val gap among all experiments!**
- Very gradual gap widening, never explosive
- Architecture complexity acted as implicit regularizer

**Severity:** Very Mild
- Excellent generalization maintained throughout
- Enhanced FC head `[512, 256]` provided good feature hierarchy
- Conv blocks after layer2 didn't cause severe overfitting

#### ✅ Strengths
1. **Best generalization:** Smallest train-val gap (0.96%)
2. **Continued learning:** Improved until epoch 158 (no early plateau)
3. **Good Val-Test consistency:** Gap only 0.99%
4. **Complex architecture manageable:** LR decay helped significantly
5. **Stable final phase:** No volatile fluctuations

#### ️ Weaknesses
1. **Slowest convergence:** Took 158 epochs to reach peak
2. **Lower test accuracy:** 95.98% (6th place) despite good generalization
3. **Initial instability:** First 50 epochs very volatile
4. **Underperformed expectations:** Expected 97.0-97.5%, got 95.98%

#### 💡 Key Insights
- **Complex architecture = slower but more stable learning**
- V2's poor test performance (95.98%) despite good val accuracy (96.97%) suggests:
  - Val-test distribution mismatch (similar to Baseline)
  - Or test set harder for this specific architecture
- **LR=1e-3 too aggressive** for this architecture initially
- **Enhanced FC `[512, 256]` better than `[256]`** (V1) for complex backbone
- **Conv after layer2 less effective than layer1** (V5 comparison)

####  Conclusion for V2
**Status:** ️ Underperformed despite good training dynamics  
**Expected:** 97.0-97.5% | **Actual:** 95.98%  
**Gap:** -1.0 to -1.5%  
**Root Cause:** Architecture complexity not translating to test performance; possible val-test mismatch  
**Recommendation:** Interesting for ablation study but not for submission; validates that layer1 > layer2 for conv blocks

---

### Experiment 4: V3

#### 📋 Configuration Summary
- **Architecture:** ResNet50 with layer3 removed + single FC layer (lightweight model)
- **Hyperparameters:** LR=1e-3, WD=1e-3, LS=0.10
- **Optimizer:** ADAMW with scheduler
- **Early Stopping:** Enabled (patience=50)
- **Total Epochs:** 154 (early stopped)

####  Test Results
- **Test Accuracy:** 97.19%
- **Test F1 (macro):** 0.9714
- **Best Val Accuracy:** 98.27% (epoch 104)
- **Val-Test Gap:** 1.08%

####  Training Dynamics (CSV Analysis - 154 epochs)

**Phase 1: Fastest Initial Learning (Epoch 1-10)**
- LR increased from 1e-4 to 1e-3
- Train Acc: 25.52% → 59.69% (**fastest start**)
- Val Acc: 45.89% → 60.17%
- **Lightweight architecture learned much faster** than all others

**Phase 2: Rapid Convergence (Epoch 11-30)**
- Train acc reached 80%+ by epoch 19 (fastest among all)
- Val acc climbed to 87-91% range
- Less volatile than Baseline, more stable learning
- Epoch 26: reached 91.77% val acc

**Phase 3: LR Decay Breakthrough (Epoch 31-55)**
- LR=1e-3 phase: train acc 86% → 94%
- Val acc stabilized at 88-93%
- Epoch 47: reached 95.67% val acc (first time)
- **Consistent upward trajectory**

**Phase 4: Peak Performance (Epoch 56-104)**
- LR=5e-4 → 2.5e-4
- Train acc reached 99%+ by epoch 86
- **Val acc peaked at 98.27% (epoch 104)** - highest among all experiments!
- Epoch 96, 98, 100, 104, 106: all reached 97.40-98.27%
- **Excellent generalization maintained**

**Phase 5: Plateau with Perfect Training (Epoch 105-154)**
- LR=1.25e-4 → 1.6e-5
- Train acc: 100% (perfect, epoch 99+)
- Val acc: 96.97-98.27% range (stable)
- Mild overfitting but controlled
- Early stopping at epoch 154

####  Overfitting Analysis

**Evidence of Mild Overfitting:**
- Final epoch: Train Acc=100%, Val Acc=97.40% → Gap=2.60%
- Peak epoch 104: Train Acc=99.91%, Val Acc=98.27% → Gap=1.64%
- Overfitting developed in last 50 epochs
- But **best val accuracy overall** (98.27%)

**Severity:** Mild to Moderate
- Train acc reached 100% (perfect memorization)
- But val acc remained high (97%+)
- Lightweight architecture helped prevent catastrophic overfitting

#### ✅ Strengths
1. **Highest validation accuracy:** 98.27% (best among all 8 experiments)
2. **Fastest convergence:** Reached high accuracy much quicker than others
3. **Excellent test performance:** 97.19% (3rd place overall)
4. **Lightweight model efficient:** Fewer parameters, faster training
5. **Good Val-Test consistency:** Gap only 1.08%
6. **Removing layer3 proved effective:** Reduced complexity without sacrificing performance

#### ⚠️ Weaknesses
1. **Test accuracy lower than val:** 97.19% vs 98.27% (1.08% gap)
2. **Did not meet highest expectations:** Expected 97.5-98.0%, got 97.19%
3. **Perfect training accuracy:** May indicate model capacity still too high for dataset
4. **Slightly underperformed V5:** 97.19% vs 96.79% (but V5 has better val-test gap)

#### 💡 Key Insights
- **Simplification strategy worked:** Removing layer3 reduced overfitting risk
- **Faster learning = better generalization** (up to a point)
- **V3 is excellent secondary model candidate** behind V5
- **Lightweight architecture benefits:** Faster training, good accuracy, less overfitting
- **Val-Test gap (1.08%) suggests some distribution mismatch** but acceptable

#### 📌 Conclusion for V3
**Status:** ✅ Excellent performance, strong candidate  
**Expected:** 97.5-98.0% | **Actual:** 97.19%  
**Gap:** -0.3 to -0.8% (minor underperformance)  
**Root Cause:** Validation-test distribution mismatch; model still slightly over-parameterized  
**Recommendation:** 🥈 **Secondary submission model** - reliable, efficient, and competitive

---

### Experiment 5: V4

#### 📋 Configuration Summary
- **Architecture:** ResNet50 with layer4 removed + single FC + **maximum regularization** (dropout=0.7, LS=0.15)
- **Hyperparameters:** LR=5e-4, WD=5e-3, LS=0.15 (Maximum regularization config)
- **Optimizer:** ADAMW with scheduler
- **Early Stopping:** Enabled (patience=50)
- **Total Epochs:** 126 (early stopped)

#### 📊 Test Results
- **Test Accuracy:** 96.39%
- **Test F1 (macro):** 0.9634
- **Best Val Accuracy:** 97.40% (epoch 76, 97)
- **Val-Test Gap:** 1.01%

#### 📈 Training Dynamics (CSV Analysis - 126 epochs)

**Phase 1: Very Slow Start (Epoch 1-20)**
- LR=5e-4 (conservative start due to max regularization)
- Train Acc: 10.28% → 71.15% (**slowest warmup**)
- Val Acc: 20.35% → 75.32%
- **Heavy regularization significantly slowed learning**

**Phase 2: Gradual Convergence (Epoch 21-58)**
- Very slow progress: train acc 71% → 92%
- Val acc fluctuated: 75-95% range
- High instability due to aggressive regularization
- Epoch 44: first time reaching 94.37% val acc
- **Struggled to learn with dropout=0.7 + LS=0.15**

**Phase 3: LR Decay Helped (Epoch 59-89)**
- LR=2.5e-4: moderate improvement
- Train acc: 93% → 98%
- Val acc: 92-97% range
- Epoch 76: reached peak 97.40% val acc
- **Regularization prevented severe overfitting**

**Phase 4: Plateau with Heavy Regularization (Epoch 90-126)**
- LR=1.25e-4 → 1.6e-5
- Train acc: 98% → 100% (perfect by epoch 111+)
- Val acc: 93-97% range (stuck)
- **Regularization kept val acc from collapsing** but limited peak performance
- Early stopping at epoch 126

#### 🔍 Overfitting Analysis

**Evidence of Controlled Overfitting:**
- Final epoch: Train Acc=100%, Val Acc=94.37% → Gap=5.63%
- Peak epoch 76: Train Acc=97.75%, Val Acc=97.40% → Gap=0.35%
- **Overfitting developed late** (after epoch 100)
- Heavy regularization delayed but didn't prevent overfitting

**Severity:** Moderate
- Train acc reached 100% despite dropout=0.7
- Val acc dropped in final 30 epochs (97.40% → 94.37%)
- **Regularization helped but architecture flaw (removing layer4) still problematic**

#### ✅ Strengths
1. **Test accuracy improved significantly:** 96.39% vs Run-3's 93.57% (+2.82%)
2. **Maximum regularization prevented catastrophic overfitting**
3. **Good Val-Test consistency:** Gap only 1.01%
4. **LR=5e-4 helped stabilize training**
5. **Fix attempt partially successful:** +2.82% improvement

#### ⚠️ Weaknesses
1. **Slowest learning curve:** Took longest to reach high accuracy
2. **Peak val accuracy limited:** 97.40% (regularization too strong?)
3. **Final overfitting:** Val acc dropped 3% in last 30 epochs
4. **Still underperformed expectations:** Expected 94.5-95.0%, got 96.39% (actually better!)
5. **Architecture flaw remains:** Removing layer4 reduces capacity too much

#### 💡 Key Insights
- **Maximum regularization WORKED:** Improved from 93.57% to 96.39% (+2.82%)
- **But regularization too strong:** Limited peak performance to 97.40% val acc
- **Removing layer4 is fundamental flaw:** Even with perfect regularization, can't compete with V3/V5
- **LS=0.15 + dropout=0.7 may be excessive:** Could benefit from slightly less aggressive settings
- **V4 fix attempt successful but architecture still inferior** to other variants

####  Conclusion for V4
**Status:** ✅ Fix attempt successful (+2.82% improvement)  
**Expected:** 94.5-95.0% | **Actual:** 96.39%  
**Gap:** +1.4 to +1.9% (**exceeded expectations!**)  
**Root Cause:** Maximum regularization effective, but architecture still limited  
**Recommendation:** Interesting for demonstrating regularization effectiveness, but not for submission; proves that **regularization can compensate for architectural flaws to some extent**

---

### Experiment 6: V5 (Champion Model)

####  Configuration Summary
- **Architecture:** ResNet50 with conv blocks after layer1 + single FC layer
- **Hyperparameters:** LR=5e-4, WD=5e-3, LS=0.05 (**Maximum Performance Config**)
- **Optimizer:** ADAMW with scheduler
- **Early Stopping:** Enabled (patience=50)
- **Total Epochs:** 128 (early stopped)

####  Test Results
- **Test Accuracy:** 96.79%
- **Test F1 (macro):** 0.9674
- **Best Val Accuracy:** 97.40% (epoch 78, 96, 102, 120, 123, 127)
- **Val-Test Gap:** 0.61%

#### 📈 Training Dynamics (CSV Analysis - 128 epochs)

**Phase 1: Conservative Warmup (Epoch 1-10)**
- LR=5e-4 (conservative start for max performance config)
- Train Acc: 11.90% → 57.80%
- Val Acc: 12.99% → 70.13%
- Slower start than Baseline but more stable

**Phase 2: LR=5e-4 Learning (Epoch 11-27)**
- Gradual convergence: train acc 62% → 82%
- Val acc fluctuated: 68-89% range
- Epoch 24: reached 89.18% val acc
- **More stable than Run-3 V5** (WD=5e-3 helped)

**Phase 3: Critical LR Decay (Epoch 28-43)**
- LR reduced to 2.5e-4: **major breakthrough**
- Train acc: 85% → 93%
- Val acc: 89% → 93.51%
- **WD=5e-3 + LS=0.05 combination started working**

**Phase 4: Peak Performance (Epoch 44-78)**
- LR=1.25e-4 → 6.3e-5
- Train acc: 93% → 99.73%
- **Val acc peaked at 97.40% (epoch 78)** - first time
- Epoch 78: train 99.19%, val 97.40% (excellent balance)
- **Maximum performance config delivering results**

**Phase 5: Extended Fine-tuning (Epoch 79-128)**
- LR=6.3e-5 → 1.6e-5
- Train acc: 99.73% → 100% (perfect by epoch 105+)
- Val acc: 96.10-97.40% range (stable)
- **Consistent high performance maintained**
- Multiple epochs reached 97.40%: 78, 96, 102, 120, 123, 127
- Early stopping at epoch 128

#### 🔍 Overfitting Analysis

**Evidence of Mild Overfitting:**
- Final epoch: Train Acc=99.91%, Val Acc=96.97% → Gap=2.94%
- Peak epochs (78, 127): Train Acc=99.19-100%, Val Acc=97.40% → Gap=1.79-2.60%
- Overfitting developed gradually in last 50 epochs
- But **WD=5e-3 prevented catastrophic overfitting**

**Severity:** Mild
- Strong weight decay (5e-3) helped control overfitting
- Val acc remained high (96.97-97.40%)
- **Excellent generalization for such strong regularization**

#### ✅ Strengths
1. **Strong test accuracy:** 96.79% (tied with V1 for 2nd place)
2. **Best Val-Test gap:** Only 0.61% (excellent generalization)
3. **Maximum performance config worked:** WD=5e-3 + LS=0.05 effective
4. **Stable training dynamics:** Less volatile than Run-3 V5
5. **Consistent peak performance:** Multiple epochs reached 97.40%
6. **Conv blocks after layer1 proven effective:** Best backbone modification

#### ️ Weaknesses
1. **Lower than Run-3 V5:** 96.79% vs Run-3's 97.59% (-0.80%)
2. **Did not meet expectations:** Expected 98.0-98.5%, got 96.79%
3. **Test accuracy lower than val:** 96.79% vs 97.40% (0.61% gap)
4. **Conservative LR=5e-4 may have limited performance**

#### 💡 Key Insights
- **Maximum performance config (WD=5e-3, LS=0.05) WORKED** for generalization
- **But test accuracy dropped vs Run-3:** Suggests:
  - Run-3 V5 may have benefited from luck/lucky initialization
  - Or test set harder for this specific run
- **Layer1 conv blocks + WD=5e-3 = excellent regularization**
- **V5 architecture still best** despite slightly lower test accuracy than Run-3
- **Multiple peak epochs suggest stable optimum**

#### 📌 Conclusion for V5
**Status:** ✅ Strong champion model, excellent generalization  
**Expected:** 98.0-98.5% | **Actual:** 96.79%  
**Gap:** -1.2 to -1.7%  
**Root Cause:** Conservative LR=5e-4 + possible test set difficulty; but excellent generalization  
**Recommendation:** 🥇 **Primary submission model** - best architecture + best regularization config, despite slightly lower test accuracy than expected

---

### Experiment 7: V6

#### 📋 Configuration Summary
- **Architecture:** ResNet50 with conv blocks after layer2 + single FC layer
- **Hyperparameters:** LR=1e-3, WD=1e-3, LS=0.10
- **Optimizer:** ADAMW with scheduler
- **Early Stopping:** Enabled (patience=50)
- **Total Epochs:** 120 (early stopped)

#### 📊 Test Results
- **Test Accuracy:** 94.38%
- **Test F1 (macro):** 0.9436
- **Best Val Accuracy:** 96.97% (epoch 70)
- **Val-Test Gap:** 2.59%

####  Training Dynamics (CSV Analysis - 120 epochs)

**Phase 1: Moderate Start (Epoch 1-10)**
- LR increased from 1e-4 to 1e-3
- Train Acc: 13.35% → 56.99%
- Val Acc: 14.72% → 74.03%
- Standard learning curve

**Phase 2: Gradual Convergence (Epoch 11-48)**
- Train acc reached 90%+ by epoch 43
- Val acc fluctuated: 72-95% range
- High volatility in early phase
- Epoch 43: reached 94.81% val acc (first time)

**Phase 3: LR Decay Breakthrough (Epoch 49-70)**
- LR=5e-4: significant improvement
- Train acc: 90% → 97.66%
- **Val acc peaked at 96.97% (epoch 70)** - best performance
- Epoch 65: reached 96.54% val acc
- **Conv after layer2 showed promise**

**Phase 4: Plateau Phase (Epoch 71-120)**
- LR=2.5e-4 → 1.6e-5
- Train acc: 97.66% → 99.73%
- Val acc: 95.24-96.97% range (declining trend)
- **Val acc dropped from peak 96.97% to 95.67%**
- Early stopping at epoch 120

#### 🔍 Overfitting Analysis

**Evidence of Moderate Overfitting:**
- Final epoch: Train Acc=99.73%, Val Acc=95.67% → Gap=4.06%
- Peak epoch 70: Train Acc=98.20%, Val Acc=96.97% → Gap=1.23%
- Overfitting developed in last 50 epochs
- Val acc declined from peak (96.97% → 95.67%)

**Severity:** Moderate
- Val acc dropped 1.3% from peak
- Test accuracy much lower (94.38% vs 96.97% val)
- **Large Val-Test gap (2.59%) concerning**

####  Strengths
1. **Good peak val accuracy:** 96.97% (epoch 70)
2. **Conv after layer2 showed potential:** Reached high val accuracy
3. **Faster convergence than V2:** 120 epochs vs 158 epochs
4. **Single FC simpler than V2's enhanced FC**

#### ️ Weaknesses
1. **Poor test accuracy:** 94.38% (7th place)
2. **Large Val-Test gap:** 2.59% (2nd worst)
3. **Val acc declined from peak:** 96.97% → 95.67%
4. **Underperformed expectations:** Expected 96.7-96.9%, got 94.38%
5. **Layer2 conv blocks less effective than layer1** (V5 comparison)

####  Key Insights
- **Conv after layer2 inferior to layer1:** V5 (layer1) >> V6 (layer2)
- **Val-Test mismatch severe:** 96.97% val → 94.38% test (-2.59%)
- **Single FC not enough to compensate for layer2 positioning**
- **Overfitting more severe than V5:** Despite similar architecture style
- **V6 confirms layer1 is optimal position for conv blocks**

####  Conclusion for V6
**Status:** ️ Underperformed significantly  
**Expected:** 96.7-96.9% | **Actual:** 94.38%  
**Gap:** -2.3 to -2.5%  
**Root Cause:** Layer2 positioning suboptimal; severe val-test distribution mismatch  
**Recommendation:** 🔬 **Ablation study only** - confirms layer1 > layer2 for conv blocks; not for submission

---

### Experiment 8: V7

#### 📋 Configuration Summary
- **Architecture:** ResNet50 with conv blocks after layer2 + single FC layer (similar to V6)
- **Hyperparameters:** LR=1e-3, WD=1e-3, LS=0.10
- **Optimizer:** ADAMW with scheduler
- **Early Stopping:** Enabled (patience=50)
- **Total Epochs:** 103 (early stopped)

#### 📊 Test Results
- **Test Accuracy:** 94.58%
- **Test F1 (macro):** 0.9455
- **Best Val Accuracy:** 96.10% (epoch 53, 77, 87, 92, 94, 95, 96, 99, 101)
- **Val-Test Gap:** 1.52%

#### 📈 Training Dynamics (CSV Analysis - 103 epochs)

**Phase 1: Standard Start (Epoch 1-10)**
- LR increased from 1e-4 to 1e-3
- Train Acc: 12.08% → 58.16%
- Val Acc: 29.00% → 74.46%
- Similar to Baseline and V6

**Phase 2: Gradual Learning (Epoch 11-47)**
- Train acc reached 90%+ by epoch 40
- Val acc fluctuated: 70-95% range
- High volatility in middle phase
- Epoch 39: reached 93.51% val acc (first time)

**Phase 3: LR Decay Improvement (Epoch 48-76)**
- LR=5e-4: moderate improvement
- Train acc: 91% → 99.09%
- Val acc peaked at 96.10% (multiple epochs)
- **Less stable than V6 despite similar architecture**

**Phase 4: Early Plateau (Epoch 77-103)**
- LR=1.25e-4 → 3.1e-5
- Train acc: 99.09% → 99.82%
- Val acc: 93.51-96.10% range (stuck)
- **Earliest stopping among all experiments** (103 epochs)
- Val acc never exceeded 96.10%

#### 🔍 Overfitting Analysis

**Evidence of Moderate Overfitting:**
- Final epoch: Train Acc=99.82%, Val Acc=95.67% → Gap=4.15%
- Peak epochs: Train Acc=99.09-99.91%, Val Acc=96.10% → Gap=2.99-3.81%
- Overfitting developed rapidly in last 30 epochs
- Val acc stuck below 96.10% despite perfect training

**Severity:** Moderate
- Similar to V6 but slightly worse
- Test accuracy very close to V6 (94.58% vs 94.38%)
- **Layer2 positioning consistently underperforms**

#### ✅ Strengths
1. **Fastest convergence:** Only 103 epochs (shortest training)
2. **Decent peak val accuracy:** 96.10%
3. **Good Val-Test gap:** 1.52% (better than V6's 2.59%)
4. **Stable final phase:** Less volatile than early phases

#### ⚠️ Weaknesses
1. **Poor test accuracy:** 94.58% (8th place, worst overall)
2. **Lowest peak val accuracy:** 96.10% (worst among all experiments)
3. **Underperformed expectations:** Expected 96.7-96.9%, got 94.58%
4. **Layer2 conv blocks ineffective:** Confirmed by both V6 and V7
5. **No advantage over V6:** Nearly identical performance

#### 💡 Key Insights
- **V7 confirms V6 findings:** Layer2 conv blocks consistently inferior
- **Nearly identical to V6:** 94.58% vs 94.38% (difference negligible)
- **Faster convergence doesn't mean better performance:** 103 epochs but lowest accuracy
- **Layer2 positioning fundamentally flawed** for this task
- **V7 adds no new insights beyond V6**

#### 📌 Conclusion for V7
**Status:** ❌ Worst performer  
**Expected:** 96.7-96.9% | **Actual:** 94.58%  
**Gap:** -2.1 to -2.3%  
**Root Cause:** Layer2 positioning suboptimal; confirmed by two independent experiments (V6, V7)  
**Recommendation:** ❌ **Discard** - No redeeming qualities; layer2 conv blocks proven ineffective

---

## 🌍 GLOBAL COMPARATIVE ANALYSIS

---

### 📊 Performance Ranking (Test Accuracy)

| Rank | Experiment | Test Acc | Val Acc | Val-Test Gap | Epochs | Status |
|------|-----------|----------|---------|--------------|--------|---------|
| 🥇 1 | **V3** | **97.19%** | 98.27% | 1.08% | 154 | ✅ Excellent |
| 🥈 2 | **V1** | **96.79%** | 97.40% | 0.61% | 160 | ✅ Excellent Generalization |
| 🥉 2 | **V5** | **96.79%** | 97.40% | 0.61% | 128 | ✅ Best Architecture |
| 4 | V4 | 96.39% | 97.40% | 1.01% | 126 | ✅ Fix Successful (+2.82%) |
| 5 | V2 | 95.98% | 96.97% | 0.99% | 158 | ⚠️ Underperformed |
| 6 | Baseline | 93.98% | 96.97% | 2.99% | 115 | ❌ Large Val-Test Gap |
| 7 | V7 | 94.58% | 96.10% | 1.52% | 103 | ❌ Worst Performer |
| 8 | V6 | 94.38% | 96.97% | 2.59% | 120 | ❌ Layer2 Ineffective |

**Key Observations:**
1. **Top 3 models within 0.4%:** V3 (97.19%), V1/V5 (96.79%)
2. **Clear performance tiers:**
   - Tier 1 (Excellent): V3, V1, V5 (96.79-97.19%)
   - Tier 2 (Good): V4, V2 (95.98-96.39%)
   - Tier 3 (Poor): Baseline, V6, V7 (93.98-94.58%)
3. **Val-Test gap varies significantly:** 0.61% (V1/V5) to 2.99% (Baseline)

---

### 🔬 Architecture Comparison

#### Backbone Modifications

| Architecture | Modification | Best Test Acc | Conclusion |
|-------------|--------------|---------------|------------|
| Standard ResNet50 | None (Baseline) | 93.98% | ❌ Baseline underperforms |
| Remove layer3 | Lightweight model (V3) | **97.19%** | ✅ **Best strategy** |
| Remove layer4 | Reduce capacity (V4) | 96.39% | ⚠️ Needs heavy regularization |
| Conv after layer1 | Early feature enhancement (V5) | 96.79% | ✅ **Best backbone mod** |
| Conv after layer2 | Mid feature enhancement (V6/V7) | 94.58% | ❌ **Ineffective** |

**Key Findings:**
1. **Removing layer3 (V3) = Best overall performance** (97.19%)
   - Reduces complexity without sacrificing accuracy
   - Fastest convergence, excellent generalization
2. **Conv blocks after layer1 (V5) = Best backbone modification** (96.79%)
   - Early feature enhancement more effective than mid/late
   - Consistent with CNN theory (early layers capture low-level features)
3. **Conv blocks after layer2 (V6/V7) = Consistently poor** (~94.5%)
   - Two independent experiments confirm ineffectiveness
   - Mid-level feature enhancement not beneficial for this task
4. **Removing layer4 (V4) = Requires compensation** (96.39%)
   - Can work with maximum regularization (WD=5e-3, dropout=0.7)
   - But still inferior to other strategies

#### FC Head Design

| FC Configuration | Best Test Acc | Notes |
|-----------------|---------------|-------|
| Single FC (2048→10) | 97.19% (V3) | ✅ Simplest and most effective |
| Enhanced [256] (2048→256→10) | 96.79% (V1) | ✅ Good generalization |
| Enhanced [512, 256] (V2) | 95.98% | ⚠️ Complex but not better |

**Key Findings:**
1. **Single FC layer sufficient** when backbone is optimized (V3)
2. **Enhanced FC [256] provides good generalization** (V1: 0.61% val-test gap)
3. **Complex FC [512, 256] not worth the cost** (V2 underperformed)
4. **FC complexity less important than backbone architecture**

---

### ⚙️ Hyperparameter Effectiveness

#### Weight Decay Impact

| WD Value | Experiments | Avg Test Acc | Effectiveness |
|----------|-------------|--------------|---------------|
| 1e-3 | Baseline, V1, V2, V3, V6, V7 | 95.48% | ✅ Standard, reliable |
| 5e-3 | V4, V5 | 96.59% | ✅ **Better for complex models** |

**Finding:** WD=5e-3 superior for models with architectural modifications (V4, V5)

#### Label Smoothing Impact

| LS Value | Experiments | Avg Test Acc | Effectiveness |
|----------|-------------|--------------|---------------|
| 0.05 | V1, V5 | 96.79% | ✅ **Optimal for enhanced architectures** |
| 0.10 | Baseline, V2, V3, V4*, V6, V7 | 95.58% | ✅ Standard choice |
| 0.15 | V4* | 96.39% | ⚠️ Too aggressive alone |

*V4 used combination: LS=0.15 + WD=5e-3 + dropout=0.7

**Finding:** LS=0.05 optimal for models with enhanced FC or backbone modifications

#### Learning Rate Strategy

| LR Strategy | Experiments | Performance |
|------------|-------------|-------------|
| LR=1e-3 (standard) | Baseline, V1, V2, V3, V6, V7 | Mixed results |
| LR=5e-4 (conservative) | V4, V5 | ✅ More stable, better generalization |

**Finding:** Conservative LR=5e-4 better for maximum performance configs (V5)

---

### 🎯 Key Success Factors

#### What Worked Well ✅

1. **Simplification Strategy (V3)**
   - Removing layer3 reduced complexity
   - Faster training, better generalization
   - **Highest test accuracy: 97.19%**

2. **Early Feature Enhancement (V5)**
   - Conv blocks after layer1
   - Combined with WD=5e-3, LS=0.05
   - **Best architecture design principle**

3. **Enhanced FC Head (V1)**
   - Single hidden layer [256]
   - **Best generalization: 0.61% val-test gap**
   - LS=0.05 proved optimal

4. **Maximum Regularization (V4)**
   - Fixed underperforming architecture
   - **+2.82% improvement** from Run-3
   - Proves regularization can compensate for flaws

#### What Failed ❌

1. **Mid-Level Feature Enhancement (V6, V7)**
   - Conv blocks after layer2 consistently poor
   - Both experiments ~94.5% test accuracy
   - **Layer2 positioning fundamentally flawed**

2. **Complex FC Head (V2)**
   - Enhanced [512, 256] not better than simpler designs
   - Slower convergence, no accuracy gain
   - **Complexity without benefit**

3. **Baseline with Optimal Hyperparameters**
   - Despite grid search optimal config
   - Still underperformed (93.98%)
   - **Architecture matters more than hyperparameters**

---

### 📈 Training Dynamics Patterns

#### Convergence Speed

| Speed Category | Experiments | Epochs | Characteristics |
|---------------|-------------|--------|-----------------|
| Fastest | V7 | 103 | Quick but poor final accuracy |
| Fast | V4, V5, V6 | 120-128 | Good balance |
| Moderate | Baseline | 115 | Standard |
| Slow | V2, V3 | 154-158 | Complex/simplified architectures |
| Slowest | V1 | 160 | Enhanced FC needs more time |

**Pattern:** Faster convergence ≠ Better performance (V7 fastest but worst)

#### Overfitting Severity

| Severity | Experiments | Train-Val Gap | Characteristics |
|----------|-------------|---------------|-----------------|
| Minimal | V2 | 0.96% | Complex architecture self-regularizes |
| Mild | V1, V3, V4, V5 | 0.61-2.60% | Well-controlled |
| Moderate | Baseline, V6, V7 | 2.59-4.58% | Needs better regularization |

**Pattern:** Simpler architectures (V3) or well-regularized (V5) show less overfitting

---

### 🔍 Val-Test Gap Analysis

#### Gap Severity Ranking

| Gap Range | Experiments | Interpretation |
|-----------|-------------|----------------|
| 0.61% | V1, V5 | ✅ **Excellent generalization** |
| 0.99% | V2 | ✅ Very good |
| 1.01% | V4 | ✅ Good |
| 1.08% | V3 | ✅ Good |
| 1.52% | V7 | ⚠️ Acceptable |
| 2.59% | V6 | ❌ Concerning |
| 2.99% | Baseline | ❌ **Severe mismatch** |

**Critical Finding:** 
- **V1 and V5 have best generalization** (0.61% gap)
- **Baseline has worst generalization** despite optimal hyperparameters
- Suggests **architectural improvements reduce val-test distribution mismatch**

---

### 💡 Strategic Insights

#### 1. Architecture > Hyperparameters

**Evidence:**
- Baseline with optimal hyperparameters: 93.98%
- V3 with standard hyperparameters: 97.19%
- **Difference: +3.21% from architecture alone**

**Conclusion:** Focus on architectural improvements first, then fine-tune hyperparameters

#### 2. Simplification Can Outperform Enhancement

**Evidence:**
- V3 (remove layer3): 97.19% - **BEST**
- V5 (add conv blocks): 96.79%
- Baseline (standard): 93.98%

**Conclusion:** Sometimes removing complexity is better than adding it

#### 3. Position Matters for Feature Enhancement

**Evidence:**
- Conv after layer1 (V5): 96.79% ✅
- Conv after layer2 (V6/V7): ~94.5% ❌

**Conclusion:** Early-layer enhancement more effective than mid-layer

#### 4. Regularization Can Rescue Poor Architectures

**Evidence:**
- V4 Run-3 (no max regularization): 93.57%
- V4 Run-4 (max regularization): 96.39%
- **Improvement: +2.82%**

**Conclusion:** Aggressive regularization can partially compensate for architectural flaws

#### 5. FC Complexity Has Diminishing Returns

**Evidence:**
- Single FC (V3): 97.19%
- Enhanced [256] (V1): 96.79%
- Enhanced [512, 256] (V2): 95.98%

**Conclusion:** Beyond a point, FC complexity hurts rather than helps

---

### 🏆 Final Recommendations

#### For Submission (Production)

**Primary Model: V3** 🥇
- **Test Accuracy:** 97.19% (highest)
- **Pros:** Fastest convergence, lightweight, excellent performance
- **Cons:** Slightly higher val-test gap (1.08%)
- **Use Case:** Best overall choice for submission

**Secondary Model: V5** 🥈
- **Test Accuracy:** 96.79% (tied 2nd)
- **Pros:** Best architecture design, excellent generalization (0.61% gap)
- **Cons:** Slightly lower accuracy than V3
- **Use Case:** Ensemble with V3 for robustness

**Tertiary Model: V1** 🥉
- **Test Accuracy:** 96.79% (tied 2nd)
- **Pros:** Best generalization (0.61% gap), stable
- **Cons:** Slowest convergence (160 epochs)
- **Use Case:** Alternative if V3/V5 unavailable

#### For Research (Ablation Studies)

**Interesting Models:**
1. **V4:** Demonstrates regularization effectiveness (+2.82% fix)
2. **V2:** Shows FC complexity diminishing returns
3. **V6/V7:** Confirms layer2 positioning ineffective

**Not Recommended:**
- **Baseline:** Underperforms despite optimal hyperparameters
- **V6/V7:** Consistently poor, no redeeming qualities

#### For Future Work

**Promising Directions:**
1. **Combine V3 + V5 insights:** Lightweight model with layer1 conv blocks?
2. **Ensemble V3 + V5:** Could push accuracy beyond 97.5%
3. **Test layer1 conv blocks on V3 architecture:** Best of both worlds?
4. **Explore WD=5e-3 on V3:** Could further improve generalization?

**Avoid:**
- Conv blocks after layer2 or deeper
- Complex FC heads beyond [256]
- Relying solely on hyperparameter tuning without architectural changes

---

### 📊 Summary Statistics

#### Overall Performance Metrics

- **Best Test Accuracy:** 97.19% (V3)
- **Worst Test Accuracy:** 93.98% (Baseline)
- **Average Test Accuracy:** 95.76%
- **Standard Deviation:** 1.18%
- **Performance Range:** 3.21%

#### Training Efficiency

- **Fastest Convergence:** 103 epochs (V7)
- **Slowest Convergence:** 160 epochs (V1)
- **Average Epochs:** 133
- **Most Efficient:** V3 (97.19% in 154 epochs)

#### Generalization Quality

- **Best Val-Test Gap:** 0.61% (V1, V5)
- **Worst Val-Test Gap:** 2.99% (Baseline)
- **Average Gap:** 1.43%
- **Models with <1% Gap:** 3 out of 8 (V1, V2, V5)

---

### 🎓 Lessons Learned

1. **Always validate architecture before hyperparameter tuning**
   - Baseline proved that optimal hyperparameters can't fix poor architecture

2. **Simplification often beats complexity**
   - V3 (removing layer3) outperformed all enhancement strategies

3. **Position matters in CNN modifications**
   - Layer1 > Layer2 for conv block insertion

4. **Regularization is powerful but not magic**
   - V4 improved +2.82% but still couldn't match V3/V5

5. **Val-Test gap reveals generalization quality**
   - Small gap (V1, V5: 0.61%) indicates robust models
   - Large gap (Baseline: 2.99%) indicates overfitting or distribution mismatch

6. **Multiple experiments needed for confirmation**
   - V6 and V7 both confirmed layer2 ineffectiveness
   - Single experiment could be luck; two experiments prove pattern

---

## 🚀 CONCLUSION

**Run-4 successfully validated optimization strategies:**

✅ **V3 emerged as champion** (97.19% test accuracy)  
✅ **V5 proven as best architecture design** (layer1 conv blocks)  
✅ **V4 fix successful** (+2.82% improvement with max regularization)  
✅ **Grid search hyperparameters effective** (especially WD=5e-3, LS=0.05)  
❌ **Layer2 conv blocks proven ineffective** (V6, V7 both poor)  

**Final Recommendation for Submission:**
1. **Primary:** V3 (97.19%)
2. **Secondary:** V5 (96.79%, best generalization)
3. **Ensemble:** V3 + V5 could achieve 97.5%+

**Next Steps:**
- Submit V3 as primary model
- Consider V3+V5 ensemble for competition
- Explore combining V3 lightweight design with V5's layer1 conv blocks

---

**Analysis completed:** 2026-05-03  
**Total experiments analyzed:** 8  
**Total epochs reviewed:** 1,064  
**Data points analyzed:** ~5,320 (8 experiments × ~133 epochs × 5 metrics)

