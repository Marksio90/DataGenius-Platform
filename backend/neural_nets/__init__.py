"""
Neural Networks module - PyTorch i TensorFlow integration.
"""

__version__ = "3.0.0"

from .pytorch_trainer import PyTorchTrainer
from .tensorflow_trainer import TensorFlowTrainer
from .automl_neural import AutoMLNeural

__all__ = [
    'PyTorchTrainer',
    'TensorFlowTrainer',
    'AutoMLNeural',
]