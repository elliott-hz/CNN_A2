# Quick Start Guide

Get up and running with the Visual Dog Emotion Recognition project in 5 minutes!

## ⚡ Quick Setup (3 Steps)

### Step 1: Install Dependencies

```bash
cd CNN_A3
pip install -r requirements.txt
```

### Step 2: Setup Kaggle API

```bash
# Create directory
mkdir -p ~/.kaggle

# Download kaggle.json from https://www.kaggle.com/<username>/account
# Place it in ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Run Your First Experiment

```bash
# Run classification baseline experiment
python experiments/exp04_classification_baseline.py
```

That's it! The experiment will:
1. ✅ Download datasets automatically
2. ✅ Preprocess data
3. ✅ Train the model
4. ✅ Evaluate on test set
5. ✅ Save results to `outputs/exp04_classification_baseline/run_TIMESTAMP/`

## 🎯 Common Tasks

### View Experiment Results

```bash
# List all runs for an experiment
ls outputs/exp04_classification_baseline/

# View latest run
ls outputs/exp04_classification_baseline/run_*/

# Read the report
cat outputs/exp04_classification_baseline/run_*/logs/experiment_report.md

# Check training metrics
cat outputs/exp04_classification_baseline/run_*/logs/training_log.csv
```

### Run Different Experiments

```bash
# Detection experiments
python experiments/exp01_detection_baseline.py
python experiments/exp02_detection_modified_v1.py
python experiments/exp03_detection_modified_v2.py

# Classification experiments
python experiments/exp04_classification_baseline.py
python experiments/exp05_classification_modified_v1.py
python experiments/exp06_classification_modified_v2.py
```

### Run All Experiments

```bash
bash scripts/run_all_experiments.sh
```

### Test Inference

```bash
# After running at least exp01 and exp04
bash scripts/inference_demo.sh path/to/your/image.jpg
```

## 🔧 Troubleshooting

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Kaggle API authentication error"

**Solution:**
1. Make sure you have `kaggle.json` in `~/.kaggle/`
2. Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
3. Accept dataset rules on Kaggle website first

### Issue: "Out of Memory"

**Solution:** Edit the experiment script and reduce batch size:
```python
training_config = {
    'batch_size': 8,  # Reduce from 32
    'gradient_accumulation_steps': 2,  # Add this
    'use_amp': True,  # Ensure this is True
}
```

### Issue: "CUDA not available"

**Solution:** The code will automatically fall back to CPU, but it will be slower. To check:
```python
import torch
print(torch.cuda.is_available())  # Should print True
```

## 📊 Understanding Outputs

After running an experiment, you'll find:

```
outputs/exp04_classification_baseline/run_20260420_193045/
├── model/
│   ├── best_model.pth          ← Best model weights
│   └── model_config.json       ← Configuration used
├── logs/
│   ├── training_log.csv        ← Epoch-by-epoch metrics
│   ├── experiment_report.md    ← Human-readable report
│   └── evaluation_metrics.json ← Test set metrics
└── figures/                    ← Plots and visualizations
    ├── confusion_matrix.png
    └── ...
```

## 🎓 Learning Path

### Beginner
1. Run `exp04_classification_baseline.py` first (simplest)
2. Examine the output files
3. Read the experiment report
4. Try running inference on your own images

### Intermediate
1. Run all 6 experiments
2. Compare results across experiments
3. Modify hyperparameters in experiment scripts
4. Create custom experiment variants

### Advanced
1. Modify model architectures in `src/models/`
2. Implement custom data augmentations
3. Add new evaluation metrics
4. Integrate with external tools (W&B, MLflow)

## 💡 Pro Tips

1. **Start Small**: Test with fewer epochs first
   ```python
   training_config['epochs'] = 5  # Quick test
   ```

2. **Monitor GPU Usage**:
   ```bash
   nvidia-smi  # Linux/Mac
   ```

3. **Resume Training**: If interrupted, the best model is already saved

4. **Compare Experiments**: Use the timestamped folders to compare different runs

5. **Custom Images**: Test inference on your own dog photos!

## 📚 Next Steps

- Read `README.md` for detailed documentation
- Check `PartC-Project Structure.md` for architecture details
- Explore individual experiment scripts to understand the workflow
- Modify `config.yaml` for global settings

## 🆘 Need Help?

1. Check the error message carefully
2. Look at `PROJECT_SUMMARY.md` for file locations
3. Review the experiment script that failed
4. Check if all dependencies are installed

---

**Happy Training! 🐕🎉**
