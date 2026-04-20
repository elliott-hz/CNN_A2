"""
CPU Test Configuration
Use this for quick logic validation on CPU before full GPU training
"""

# Quick test configuration for CPU validation
CPU_TEST_CONFIG = {
    # Use smaller dataset subset
    'use_subset': True,
    'subset_size': 100,  # Only use 100 samples per split
    
    # Minimal training
    'epochs': 3,         # Just 3 epochs to verify flow
    'batch_size': 4,     # Small batch for CPU
    
    # Simplified model
    'use_pretrained': False,  # Random init for faster testing
    
    # Disable optimizations not needed on CPU
    'use_amp': False,
    'num_workers': 0,    # No multiprocessing on CPU
    
    # Quick evaluation
    'early_stopping_patience': 2,
}

# How to use in experiment scripts:
"""
from config_cpu_test import CPU_TEST_CONFIG

# Override default config with test config
if args.test_mode:
    training_config.update(CPU_TEST_CONFIG)
"""
