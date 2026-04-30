"""
Evaluation Package

Provides evaluation tools for detection and classification models.
"""

# Classification evaluator
from .classification_evaluator import ClassificationEvaluator

# Detection evaluator
from .detection_evaluator import DetectionEvaluator

__all__ = [
    'ClassificationEvaluator',
    'DetectionEvaluator'
]
