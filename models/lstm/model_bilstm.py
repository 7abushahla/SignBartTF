"""
Improved e-LSTM model with Bidirectional LSTMs and Temporal Attention

Key improvements over original:
- Bidirectional LSTMs (process sequences forward and backward)
- Frame embedding layer with LayerNorm
- Temporal attention with proper masking
- Fewer parameters (~1-2M instead of ~10M)
- TFLite-friendly operations only
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable()
class TemporalAttention(layers.Layer):
    """
    Simple temporal attention over LSTM outputs.
    
    Computes attention weights over time steps and returns weighted context vector.
    Supports masking for padded frames.
    
    Args:
        units: Hidden units for attention scoring (default: 128)
    """
    
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense_tanh = layers.Dense(units, activation="tanh", name="attn_tanh")
        self.dense_score = layers.Dense(1, name="attn_score")
    
    def call(self, inputs, mask=None):
        """
        Args:
            inputs: (batch, time, features) - LSTM outputs
            mask: (batch, time) - True for valid frames, False for padded
        
        Returns:
            context: (batch, features) - weighted sum over time
        """
        # inputs: (B, T, F)
        # score:  (B, T, 1)
        score = self.dense_score(self.dense_tanh(inputs))
        
        # Apply mask if provided (from Masking layer)
        if mask is not None:
            # mask: (B, T) -> (B, T, 1)
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
            # Large negative value to "hide" padded positions
            score += (1.0 - mask) * -1e9
        
        # Attention weights over time
        weights = tf.nn.softmax(score, axis=1)  # (B, T, 1)
        
        # Context vector: weighted sum over time
        context = tf.reduce_sum(weights * inputs, axis=1)  # (B, F)
        return context
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_elstm_model(
    num_classes: int,
    seq_len: int,
    num_keypoints: int,
    coords_per_keypoint: int,
    lstm1_units: int = 256,
    lstm2_units: int = 128,
    attention_units: int = 128,
    fc_units: int = 128,
    dropout_rate: float = 0.5,
):
    """
    Improved e-LSTM model with Bidirectional LSTMs for sign language recognition.
    
    Architecture:
    1. Input: (seq_len, num_keypoints, coords_per_keypoint)
    2. Flatten + Masking: Handle variable-length sequences
    3. Frame Embedding: LayerNorm + Dense + Dropout
    4. Bidirectional LSTM Stack: 2 layers with dropout
    5. Temporal Attention: Weighted pooling over time
    6. Classification Head: Dense + Dropout + Softmax
    
    Args:
        num_classes: Number of sign language classes
        seq_len: Fixed sequence length (number of frames)
        num_keypoints: Number of keypoints per frame (default: 90)
        coords_per_keypoint: Coordinates per keypoint (default: 2 for x,y)
        lstm1_units: Units in first LSTM layer (default: 256, bidirectional = 512)
        lstm2_units: Units in second LSTM layer (default: 128, bidirectional = 256)
        attention_units: Hidden units in attention layer (default: 128)
        fc_units: Units in fully connected layer (default: 128)
        dropout_rate: Dropout rate (default: 0.5)
    
    Returns:
        model: Keras model
    """
    # ------------------------------------------------------------------
    # 1) Input and reshape
    # ------------------------------------------------------------------
    inputs = keras.Input(
        shape=(seq_len, num_keypoints, coords_per_keypoint),
        name="keypoints",
    )  # (B, T, 90, 2)
    
    # Flatten keypoints per frame: (B, T, 90*2 = 180)
    x = layers.Reshape(
        (seq_len, num_keypoints * coords_per_keypoint),
        name="flatten_kpts"
    )(inputs)
    
    # Mask padded frames (all zeros after preprocessing)
    x = layers.Masking(mask_value=0.0, name="frame_masking")(x)
    
    # ------------------------------------------------------------------
    # 2) Frame embedding
    # ------------------------------------------------------------------
    # Small dense projection to learn better representation per frame
    emb_dim = 128
    x = layers.LayerNormalization(epsilon=1e-6, name="frame_ln")(x)
    x = layers.Dense(emb_dim, activation="relu", name="frame_embedding")(x)
    x = layers.Dropout(0.3, name="frame_dropout")(x)
    
    # ------------------------------------------------------------------
    # 3) Bidirectional LSTM stack
    # ------------------------------------------------------------------
    # BiLSTM 1: 2 * lstm1_units outputs (forward + backward)
    x = layers.Bidirectional(
        layers.LSTM(
            lstm1_units,
            return_sequences=True,
            name="lstm1",
        ),
        name="bilstm1",
    )(x)
    x = layers.Dropout(0.4, name="lstm1_dropout")(x)
    
    # BiLSTM 2: 2 * lstm2_units outputs (forward + backward)
    x = layers.Bidirectional(
        layers.LSTM(
            lstm2_units,
            return_sequences=True,
            name="lstm2",
        ),
        name="bilstm2",
    )(x)
    x = layers.Dropout(0.4, name="lstm2_dropout")(x)
    
    # Note: Keras automatically passes the mask through layers
    # TemporalAttention will receive the mask from the LSTM
    
    # ------------------------------------------------------------------
    # 4) Temporal attention
    # ------------------------------------------------------------------
    context = TemporalAttention(
        units=attention_units,
        name="temporal_attention"
    )(x)
    
    # ------------------------------------------------------------------
    # 5) Classification head
    # ------------------------------------------------------------------
    x = layers.Dense(fc_units, activation="relu", name="fc1")(context)
    x = layers.Dropout(dropout_rate, name="fc_dropout")(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="logits")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="e_lstm_bilstm")
    return model


def compile_model(model, learning_rate=0.0002, label_smoothing=0.1, clipnorm=1.0):
    """
    Compile model with Adam optimizer and categorical cross-entropy loss
    
    Args:
        model: Keras model
        learning_rate: Learning rate for Adam optimizer (default: 0.0002)
        label_smoothing: Label smoothing factor (default: 0.1)
        clipnorm: Gradient clipping norm (default: 1.0)
    
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=clipnorm
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')
        ]
    )
    return model


if __name__ == "__main__":
    # Example usage
    NUM_CLASSES = 10
    SEQ_LEN = 64
    NUM_KEYPOINTS = 90
    COORDS = 2
    
    print("Building BiLSTM e-LSTM model...")
    model = build_elstm_model(
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        num_keypoints=NUM_KEYPOINTS,
        coords_per_keypoint=COORDS,
        lstm1_units=256,
        lstm2_units=128,
        attention_units=128,
        fc_units=128,
        dropout_rate=0.5
    )
    
    model = compile_model(model, learning_rate=0.0002)
    model.summary()
    
    print(f"\n✅ Model built successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Expected: ~1-2M parameters (much smaller than original 10M)")

