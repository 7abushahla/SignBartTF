"""
TFLite-friendly Functional API implementation of SignBART.
This version uses TFLite-compatible operations (no gather_nd).
Mathematically equivalent to the original but with TFLite-friendly ops.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from layers import PositionalEmbedding, ClassificationHead, Projection
from encoder import Encoder
from decoder import Decoder


class ExtractLastValidTokenTFLite(layers.Layer):
    """
    TFLite-friendly version that uses boolean masking instead of gather_nd.
    Mathematically equivalent to the original but TFLite-compatible.
    """
    
    def call(self, inputs):
        decoder_outputs, attention_mask = inputs
        
        # Find last valid position for each sequence
        seq_lengths = tf.cast(tf.reduce_sum(attention_mask, axis=1) - 1, tf.int32)
        seq_lengths = tf.maximum(seq_lengths, 0)  # Ensure non-negative
        
        # Get sequence length dimension
        seq_len = tf.shape(decoder_outputs)[1]
        
        # Create one-hot mask for last valid position (TFLite friendly)
        # This replaces gather_nd with a weighted sum
        position_mask = tf.one_hot(seq_lengths, depth=seq_len)  # (batch, seq_len)
        position_mask = tf.expand_dims(position_mask, -1)  # (batch, seq_len, 1)
        
        # Weighted sum effectively selects the last valid token
        # This is mathematically equivalent to gather_nd but TFLite-compatible
        last_output = tf.reduce_sum(
            decoder_outputs * position_mask, 
            axis=1
        )  # (batch, d_model)
        
        return last_output
    
    def get_config(self):
        return super().get_config()


def build_signbart_functional_tflite(config):
    """
    Build TFLite-friendly SignBART model using Functional API.
    Uses boolean masking instead of gather_nd for TFLite compatibility.
    
    Args:
        config: dict with model configuration
        
    Returns:
        keras.Model: TFLite-compatible Functional API model
    """
    # Define inputs
    keypoints_input = keras.Input(
        shape=(None, len(config['joint_idx']), 2),
        dtype=tf.float32,
        name='keypoints'
    )
    attention_mask_input = keras.Input(
        shape=(None,),
        dtype=tf.float32,
        name='attention_mask'
    )
    
    # Projection
    projection = Projection(config, name='projection')
    x_embed, y_embed = projection(keypoints_input)
    
    # Encoder
    encoder = Encoder(config, name='encoder')
    encoder_outputs = encoder(
        x_embed=x_embed,
        attention_mask=attention_mask_input,
        training=None  # Will be set by model.fit()
    )
    
    # Decoder
    decoder = Decoder(config, name='decoder')
    decoder_outputs = decoder(
        encoder_hidden_states=encoder_outputs,
        encoder_attention_mask=attention_mask_input,
        y_embed=y_embed,
        attention_mask=attention_mask_input,
        training=None
    )
    
    # Extract last valid token (TFLite-friendly version)
    extract_last_token = ExtractLastValidTokenTFLite(name='extract_last_token')
    last_output = extract_last_token([decoder_outputs, attention_mask_input])
    
    # Classification head
    classification_head = ClassificationHead(
        config['d_model'],
        config['num_labels'],
        config['classifier_dropout'],
        name='classification_head'
    )
    logits = classification_head(last_output)
    
    # Create the Functional model
    model = keras.Model(
        inputs={'keypoints': keypoints_input, 'attention_mask': attention_mask_input},
        outputs=logits,
        name='signbart_functional_tflite'
    )
    
    return model


if __name__ == "__main__":
    print("Testing TFLite-friendly SignBART model...")
    
    # Example config
    config = {
        'joint_idx': list(range(100)),
        'd_model': 144,
        'num_labels': 10,
        'encoder_layers': 2,
        'decoder_layers': 2,
        'encoder_attention_heads': 8,
        'decoder_attention_heads': 8,
        'encoder_ffn_dim': 144,
        'decoder_ffn_dim': 144,
        'dropout': 0.3,
        'activation_dropout': 0.3,
        'attention_dropout': 0.0,
        'classifier_dropout': 0.7,
        'encoder_layerdrop': 0.4,
        'decoder_layerdrop': 0.4,
        'max_position_embeddings': 256,
        'pe': 'learn'
    }
    
    print("\n1. Building TFLite-friendly model...")
    model = build_signbart_functional_tflite(config)
    print("✓ Model built")
    
    print("\n2. Testing inference...")
    test_data = {
        'keypoints': tf.random.normal((2, 10, 100, 2)),
        'attention_mask': tf.ones((2, 10))
    }
    output = model(test_data, training=False)
    print(f"✓ Output shape: {output.shape}")
    
    print("\n3. Model summary:")
    model.summary(line_length=100)
    
    print("\n✓ TFLite-friendly model is working!")

