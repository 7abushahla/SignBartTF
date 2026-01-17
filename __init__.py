"""
SignBART TensorFlow/Keras Implementation
==========================================

A complete TensorFlow conversion of SignBART for sign language recognition.

Key Features:
- Full transformer encoder-decoder architecture
- Multiple attention mechanisms (self, cross, causal)
- Clean TFLite conversion with quantization
- Efficient tf.data pipeline
- Production-ready deployment

Modules:
--------
- model: Main SignBart model
- encoder: Transformer encoder
- decoder: Transformer decoder  
- attention: Attention mechanisms
- layers: Core layers (positional, projection, etc.)
- dataset: Data loading and preprocessing
- utils: Utility functions and metrics
- train: Training script
- convert_to_tflite: TFLite conversion with quantization

Quick Start:
------------
>>> from model import SignBart
>>> import tensorflow as tf
>>> 
>>> # Load config
>>> config = {
...     'd_model': 256,
...     'encoder_layers': 6,
...     'decoder_layers': 6,
...     'num_labels': 64,
...     'joint_idx': list(range(100)),
...     # ... other config params
... }
>>> 
>>> # Create model
>>> model = SignBart(config)
>>> 
>>> # Forward pass
>>> keypoints = tf.random.normal((2, 10, 100, 2))
>>> mask = tf.ones((2, 10))
>>> logits = model(keypoints, mask, training=False)

For more information, see README.md
"""

__version__ = "1.0.0"
__author__ = "SignBART Team"
__description__ = "TensorFlow implementation of SignBART for sign language recognition"

# Import main components for easier access
try:
    from .model import SignBart
    from .encoder import Encoder, EncoderLayer
    from .decoder import Decoder, DecoderLayer
    from .attention import SelfAttention, CrossAttention, CausalSelfAttention
    from .layers import PositionalEmbedding, Projection, ClassificationHead
    from .dataset import SignDataset, create_data_loaders
    from .utils import (
        accuracy, top_k_accuracy, 
        get_keypoint_config, count_model_parameters
    )
    
    __all__ = [
        'SignBart',
        'Encoder', 'EncoderLayer',
        'Decoder', 'DecoderLayer',
        'SelfAttention', 'CrossAttention', 'CausalSelfAttention',
        'PositionalEmbedding', 'Projection', 'ClassificationHead',
        'SignDataset', 'create_data_loaders',
        'accuracy', 'top_k_accuracy',
        'get_keypoint_config', 'count_model_parameters',
    ]
except ImportError:
    # TensorFlow not installed yet
    pass

