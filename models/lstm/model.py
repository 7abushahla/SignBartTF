"""
e-LSTM Model for Sign Language Recognition
Adapted from the original paper for 2D keypoints (90 × 2 = 180 features)

Original Architecture:
- 75 keypoints × 3 (x,y,z) = 225 features
- LSTM(1024) → LSTM(512) → Attention(64) → Dense(64) → Softmax

Adapted for our dataset:
- 90 keypoints × 2 (x,y) = 180 features
- Same architecture, adjusted input dimension
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(layers.Layer):
    """
    Additive (Bahdanau-style) attention layer with 64 hidden units
    
    This attention mechanism helps the model focus on particular time steps
    that contain the most discriminative information for gesture recognition.
    """
    
    def __init__(self, units=64, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        
        # Attention weights (initialized in build method)
        self.W1 = None
        self.W2 = None
        self.V = None
    
    def build(self, input_shape):
        """Build layer weights"""
        # W1 projects values: (batch, time, lstm_units) -> (batch, time, units)
        # W2 projects query: (batch, lstm_units) -> (batch, units)
        # V projects combined: (batch, time, units) -> (batch, time, 1)
        
        self.W1 = layers.Dense(self.units, use_bias=False, name='attention_W1')
        self.W2 = layers.Dense(self.units, use_bias=False, name='attention_W2')
        self.V = layers.Dense(1, use_bias=False, name='attention_V')
        
        super(BahdanauAttention, self).build(input_shape)
    
    def call(self, query, values):
        """
        Args:
            query: Last hidden state from LSTM, shape (batch, lstm_units)
            values: All hidden states from LSTM, shape (batch, time_steps, lstm_units)
        
        Returns:
            context_vector: Weighted sum of values, shape (batch, lstm_units)
            attention_weights: Attention weights, shape (batch, time_steps, 1)
        """
        # Expand query to match time dimension: (batch, 1, lstm_units)
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # Compute attention scores
        # score shape: (batch, time_steps, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(query_with_time_axis)
        ))
        
        # Attention weights (softmax over time dimension)
        # attention_weights shape: (batch, time_steps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector: weighted sum of values
        # context_vector shape: (batch, lstm_units)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)
        
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_elstm_model(
    num_classes,
    seq_len,
    num_keypoints=90,
    coords_per_keypoint=2,
    lstm1_units=1024,
    lstm2_units=512,
    attention_units=64,
    fc_units=64,
    dropout_rate=0.3
):
    """
    Build e-LSTM model for sign language recognition
    
    Args:
        num_classes: Number of sign language classes
        seq_len: Fixed sequence length (number of frames)
        num_keypoints: Number of keypoints per frame (default: 90)
        coords_per_keypoint: Coordinates per keypoint (default: 2 for x,y)
        lstm1_units: Units in first LSTM layer (default: 1024)
        lstm2_units: Units in second LSTM layer (default: 512)
        attention_units: Hidden units in attention layer (default: 64)
        fc_units: Units in fully connected layer (default: 64)
        dropout_rate: Dropout rate (default: 0.3)
    
    Returns:
        model: Compiled Keras model
    """
    # Input shape: (batch, seq_len, num_keypoints, coords)
    inputs = keras.Input(shape=(seq_len, num_keypoints, coords_per_keypoint), name='keypoint_input')
    
    # Flatten keypoints: (batch, seq_len, 180)
    # 90 keypoints × 2 coords = 180 features per frame
    x = layers.Reshape((seq_len, num_keypoints * coords_per_keypoint))(inputs)
    
    # LSTM Layer 1: 1024 units, return sequences
    lstm1 = layers.LSTM(
        lstm1_units,
        return_sequences=True,
        return_state=True,
        dropout=dropout_rate * 0.5,  # Recurrent dropout
        name='lstm1'
    )(x)
    lstm1_output = lstm1[0]  # (batch, seq_len, 1024)
    lstm1_output = layers.BatchNormalization(name='bn1')(lstm1_output)  # Batch norm for stability
    
    # LSTM Layer 2: 512 units, return sequences and final state
    lstm2 = layers.LSTM(
        lstm2_units,
        return_sequences=True,
        return_state=True,
        dropout=dropout_rate * 0.5,  # Recurrent dropout
        name='lstm2'
    )(lstm1_output)
    lstm2_output = lstm2[0]  # (batch, seq_len, 512)
    lstm2_output = layers.BatchNormalization(name='bn2')(lstm2_output)  # Batch norm for stability
    lstm2_final_state = lstm2[1]  # (batch, 512) - final hidden state
    
    # Additive Attention (Bahdanau-style) with 64 hidden units
    attention_layer = BahdanauAttention(units=attention_units, name='attention')
    context_vector, attention_weights = attention_layer(lstm2_final_state, lstm2_output)
    
    # Dropout for regularization
    x = layers.Dropout(dropout_rate)(context_vector)
    
    # Fully Connected Layer: Dense(64, ReLU) for dimensionality reduction
    x = layers.Dense(fc_units, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='bn3')(x)  # Batch norm after FC
    x = layers.Dropout(dropout_rate)(x)
    
    # Additional FC layer for more expressiveness
    x = layers.Dense(fc_units // 2, activation='relu', name='fc2')(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)  # Lighter dropout before output
    
    # Output Layer: Softmax for classification
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Build model
    model = keras.Model(inputs=inputs, outputs=outputs, name='e-LSTM')
    
    return model


def compile_model(model, learning_rate=0.001, label_smoothing=0.1, clipnorm=1.0):
    """
    Compile model with Adam optimizer and categorical cross-entropy loss
    
    Args:
        model: Keras model
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        label_smoothing: Label smoothing factor (default: 0.1) - helps with overfitting
        clipnorm: Gradient clipping norm (default: 1.0) - prevents exploding gradients
    
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')]
    )
    return model


if __name__ == "__main__":
    # Example usage
    NUM_CLASSES = 10  # Adjust based on your dataset
    SEQ_LEN = 64  # Fixed sequence length
    
    # Build model with custom Bahdanau attention (faithful to paper)
    model = build_elstm_model(
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        num_keypoints=90,
        coords_per_keypoint=2,
        lstm1_units=1024,
        lstm2_units=512,
        attention_units=64,
        fc_units=64,
        dropout_rate=0.3
    )
    
    # Compile
    model = compile_model(model, learning_rate=0.001)
    
    # Print summary
    print("\n" + "="*80)
    print("e-LSTM Model Architecture (Adapted for 2D Keypoints)")
    print("="*80)
    model.summary()
    
    print("\n" + "="*80)
    print("Model Configuration:")
    print("="*80)
    print(f"Input: 90 keypoints × 2 coords = 180 features per frame")
    print(f"Sequence Length: {SEQ_LEN} frames")
    print(f"LSTM 1: {1024} units (return sequences)")
    print(f"LSTM 2: {512} units (return sequences)")
    print(f"Attention: Bahdanau-style with 64 hidden units")
    print(f"FC Layer: 64 units (ReLU)")
    print(f"Output: {NUM_CLASSES} classes (Softmax)")
    print(f"Optimizer: Adam (LR=0.001)")
    print(f"Loss: Categorical Cross-Entropy")
    print("="*80 + "\n")

