"""
Quick Logic Validation Script
Tests the core components without full training
Run this on CPU to verify everything works before GPU training
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test all module imports"""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)
    
    try:
        from src.models.detection_model import YOLOv8Detector
        print("✅ YOLOv8Detector imported successfully")
        
        from src.models.classification_model import ResNet50Classifier
        print("✅ ResNet50Classifier imported successfully")
        
        from src.training.classification_trainer import ClassificationTrainer
        print("✅ ClassificationTrainer imported successfully")
        
        from src.evaluation.classification_evaluator import ClassificationEvaluator
        print("✅ ClassificationEvaluator imported successfully")
        
        from src.utils.logger import setup_logger
        print("✅ Logger utilities imported successfully")
        
        from src.utils.file_utils import create_experiment_dir
        print("✅ File utilities imported successfully")
        
        print("\n✅ TEST 1 PASSED: All imports successful\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}\n")
        return False


def test_model_creation():
    """Test model instantiation"""
    print("=" * 60)
    print("TEST 2: Model Creation")
    print("=" * 60)
    
    try:
        # Test classification model creation
        from src.models.classification_model import ResNet50Classifier
        
        config = {
            'num_classes': 4,
            'pretrained': False,  # Use random weights for quick test
            'dropout_rate': 0.5,
            'freeze_backbone': True
        }
        
        model = ResNet50Classifier(config)
        print(f"✅ ResNet50Classifier created successfully")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        print(f"✅ Forward pass successful: input shape {dummy_input.shape} -> output shape {output.shape}")
        
        print("\n✅ TEST 2 PASSED: Model creation and forward pass work\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_data_utils():
    """Test data loading utilities"""
    print("=" * 60)
    print("TEST 3: Data Utilities")
    print("=" * 60)
    
    try:
        from src.data_processing.dataset_utils import load_numpy_data
        
        # Create dummy data files for testing
        test_dir = Path('data/processed/test_dummy')
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dummy arrays
        X_dummy = np.random.rand(10, 224, 224, 3).astype(np.float32)
        y_dummy = np.random.randint(0, 4, size=(10,))
        
        np.save(test_dir / 'X_train.npy', X_dummy)
        np.save(test_dir / 'y_train.npy', y_dummy)
        
        # Try loading
        X_loaded, y_loaded = load_numpy_data(str(test_dir), 'train')
        print(f"✅ Data loading successful: X shape {X_loaded.shape}, y shape {y_loaded.shape}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        print("✅ Cleanup successful")
        
        print("\n✅ TEST 3 PASSED: Data utilities work\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_output_structure():
    """Test output directory creation"""
    print("=" * 60)
    print("TEST 4: Output Structure")
    print("=" * 60)
    
    try:
        from src.utils.file_utils import create_experiment_dir
        
        # Test directory creation
        exp_name = "test_exp_validation"
        exp_dir = create_experiment_dir(exp_name)
        print(f"✅ Experiment directory created: {exp_dir}")
        
        # Check subdirectories exist
        assert (exp_dir / 'model').exists(), "model/ dir missing"
        assert (exp_dir / 'logs').exists(), "logs/ dir missing"
        assert (exp_dir / 'figures').exists(), "figures/ dir missing"
        print("✅ All subdirectories created correctly")
        
        # Cleanup
        import shutil
        shutil.rmtree(exp_dir.parent)
        print("✅ Cleanup successful")
        
        print("\n✅ TEST 4 PASSED: Output structure works\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n" + "=" * 60)
    print("🧪 VISUAL DOG EMOTION RECOGNITION - LOGIC VALIDATION")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"NumPy Version: {np.__version__}")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Data Utilities", test_data_utils()))
    results.append(("Output Structure", test_output_structure()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ready for training.")
        print("\nNext steps:")
        print("1. Setup Kaggle API credentials")
        print("2. Run an experiment: python experiments/exp04_classification_baseline.py")
        print("3. Or run on GPU (AWS SageMaker) for full training\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check errors above.\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
