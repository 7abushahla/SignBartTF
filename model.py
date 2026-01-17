"""
TensorFlow/Keras implementation of SignBART model.
Converted from PyTorch implementation.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from encoder import Encoder
from decoder import Decoder
from layers import Projection, ClassificationHead


class SignBart(keras.Model):
    """
    SignBART model for sign language recognition.
    Encoder-decoder transformer architecture with keypoint projection.
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config_dict = config
        self.joint_idx = config['joint_idx']
        
        # Main components
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        self.classification_head = ClassificationHead(
            config['d_model'], 
            config['num_labels'], 
            config['classifier_dropout']
        )
        self.projection = Projection(config)
        
        # Loss function
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    def call(self, inputs, training=None):
        """
        Forward pass of SignBART.
        
        Args:
            inputs: Can be either:
                - dict with keys: 'keypoints', 'attention_mask', 'labels' (optional)
                - tuple/list: (keypoints, attention_mask, labels) where labels is optional
            training: bool - whether in training mode
        
        Returns:
            if labels is None:
                logits: (batch_size, num_classes)
            else:
                (loss, logits): tuple of loss and logits
        """
        # Unpack inputs - support both dict and tuple/list formats
        if isinstance(inputs, dict):
            keypoints = inputs['keypoints']
            attention_mask = inputs['attention_mask']
            labels = inputs.get('labels', None)
        else:
            # Assume tuple/list format
            if len(inputs) == 2:
                keypoints, attention_mask = inputs
                labels = None
            else:
                keypoints, attention_mask, labels = inputs
        
        batch_size = tf.shape(keypoints)[0]
        
        # Project keypoints to embedding space
        x_embed, y_embed = self.projection(keypoints, training=training)
        
        # Encode
        encoder_outputs = self.encoder(
            x_embed=x_embed,
            attention_mask=attention_mask,
            training=training
        )
        
        # Decode (using same mask for decoder)
        decoder_attention_mask = attention_mask
        
        decoder_outputs = self.decoder(
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            y_embed=y_embed,
            attention_mask=decoder_attention_mask,
            training=training
        )
        
        # Get last valid token for each sequence
        # Find the last position where mask == 1
        last_indices = tf.cast(
            tf.argmax(
                tf.cumsum(tf.cast(decoder_attention_mask, tf.float32), axis=1),
                axis=1
            ),
            tf.int32
        )
        
        # Gather the last decoder output for each sequence
        batch_indices = tf.range(batch_size)
        gather_indices = tf.stack([batch_indices, last_indices], axis=1)
        last_decoder_outputs = tf.gather_nd(decoder_outputs, gather_indices)
        
        # Classification
        logits = self.classification_head(last_decoder_outputs, training=training)
        
        # Compute loss if labels provided
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            return loss, logits
        else:
            return logits
    
    def compute_loss(self, logits, labels):
        """
        Compute cross-entropy loss.
        
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
        
        Returns:
            loss: scalar tensor
        """
        return self.loss_fn(labels, logits)
    
    def train_step(self, data):
        """
        Custom training step for Keras model.fit().
        
        Args:
            data: dict with 'keypoints', 'attention_mask', 'labels'
        
        Returns:
            dict of metrics
        """
        labels = data['labels']
        
        with tf.GradientTape() as tape:
            # Forward pass - pass entire data dict
            loss, logits = self(data, training=True)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update loss tracker (built-in Keras metric)
        self.compiled_loss(labels, logits)
        
        # Update metrics
        self.compiled_metrics.update_state(labels, logits)
        
        # Return all metrics including loss
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss  # Add loss explicitly
        return metrics
    
    def test_step(self, data):
        """
        Custom test step for Keras model.fit() validation.
        
        Args:
            data: dict with 'keypoints', 'attention_mask', 'labels'
        
        Returns:
            dict of metrics
        """
        labels = data['labels']
        
        # Forward pass - pass entire data dict
        loss, logits = self(data, training=False)
        
        # Update loss tracker
        self.compiled_loss(labels, logits)
        
        # Update metrics
        self.compiled_metrics.update_state(labels, logits)
        
        # Return all metrics including loss
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss  # Add loss explicitly
        return metrics
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'config_dict': self.config_dict,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from config dict (for deserialization)."""
        # Extract the actual model config from the saved config
        config_dict = config.get('config_dict', config)
        return cls(config_dict)


def load_pretrained_weights(model, pretrained_path):
    """
    Load pretrained weights from a checkpoint file.
    
    Args:
        model: SignBart model
        pretrained_path: path to checkpoint file (.h5 or SavedModel)
    
    Returns:
        model: model with loaded weights
    """
    try:
        model.load_weights(pretrained_path)
        print(f"Loaded checkpoint from: {pretrained_path}")
    except Exception as e:
        print(f"Could not load weights: {e}")
        print("Model will use random initialization.")
    
    return model


if __name__ == "__main__":
    import yaml
    import numpy as np
    
    print("Testing TensorFlow SignBART Model...")
    
    # Load config (you'll need to adjust path)
    # For testing, we'll use a dummy config
    config = {
        'd_model': 256,
        'encoder_attention_heads': 8,
        'encoder_ffn_dim': 1024,
        'encoder_layers': 3,
        'decoder_attention_heads': 8,
        'decoder_ffn_dim': 1024,
        'decoder_layers': 3,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.1,
        'encoder_layerdrop': 0.0,
        'decoder_layerdrop': 0.0,
        'max_position_embeddings': 512,
        'num_labels': 64,
        'classifier_dropout': 0.1,
        'joint_idx': list(range(100)),  # 100 keypoints
    }
    
    # Create model
    print("\n1. Creating SignBart model:")
    model = SignBart(config)
    
    # Test forward pass without labels (inference)
    print("\n2. Testing inference (without labels):")
    batch_size = 2
    seq_len = 10
    num_joints = 100
    
    test_keypoints = tf.random.normal((batch_size, seq_len, num_joints, 2))
    test_mask = tf.ones((batch_size, seq_len))
    
    logits = model(test_keypoints, test_mask, training=False)
    print(f"   Keypoints shape: {test_keypoints.shape}")
    print(f"   Mask shape: {test_mask.shape}")
    print(f"   Logits shape: {logits.shape}")
    
    # Test forward pass with labels (training)
    print("\n3. Testing training (with labels):")
    test_labels = tf.constant([5, 10], dtype=tf.int32)
    loss, logits = model(test_keypoints, test_mask, labels=test_labels, training=True)
    print(f"   Labels shape: {test_labels.shape}")
    print(f"   Loss: {loss.numpy():.4f}")
    print(f"   Logits shape: {logits.shape}")
    
    # Test with padding
    print("\n4. Testing with padding:")
    test_mask = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=tf.float32)
    loss, logits = model(test_keypoints, test_mask, labels=test_labels, training=True)
    print(f"   Mask with padding: {test_mask.shape}")
    print(f"   Loss: {loss.numpy():.4f}")
    print(f"   Logits shape: {logits.shape}")
    
    # Count parameters
    print("\n5. Model summary:")
    total_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   Total trainable parameters: {total_params:,}")
    
    print("\n✓ All model tests passed!")
    
    # Test model compilation
    print("\n6. Testing model compilation:")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    print("   ✓ Model compiled successfully!")

