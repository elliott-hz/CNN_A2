"""
File Utilities
Helper functions for file and directory operations
"""

from pathlib import Path
from datetime import datetime
import json
import yaml


def create_experiment_dir(experiment_name: str, base_output_dir: str = "outputs") -> Path:
    """
    Create timestamped experiment output directory.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'exp01_detection_baseline')
        base_output_dir: Base output directory
        
    Returns:
        Path to created experiment run directory
    """
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory structure
    exp_dir = Path(base_output_dir) / experiment_name / f"run_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "model").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    
    return exp_dir


def save_config(config: dict, output_path: str):
    """
    Save configuration to JSON or YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix in ['.yaml', '.yml']:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {output_path}")
