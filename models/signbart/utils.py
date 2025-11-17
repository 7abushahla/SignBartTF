import os
import logging
import cv2
import numpy as np
import torch
from tqdm import tqdm

# MediaPipe Holistic keypoint structure
total_body_idx = 33  # Pose landmarks: 0-32
total_hand = 21      # Hand landmarks per hand: 21 each
total_face = 25      # Selected face landmarks: 75-99 (NEW for 100 keypoints)

# ============================================================================
# Keypoint Index Definitions (for reference and fallback)
# ============================================================================

# POSE KEYPOINTS (0-32) - Full body pose from MediaPipe
pose_idx = list(range(0, 33))

# Upper body pose only (shoulders, elbows, wrists) - 6 keypoints
upper_body_idx = list(range(11, 17))

# HAND KEYPOINTS
# Left hand: 33-53 (21 keypoints)
lefthand_idx = [x + total_body_idx for x in range(0, 21)]
# Right hand: 54-74 (21 keypoints)  
righthand_idx = [x + 21 for x in lefthand_idx]

# FACE KEYPOINTS (75-99) - 25 selected face landmarks (NEW)
# These correspond to specific MediaPipe Face Mesh indices
face_idx = list(range(75, 100))

# ============================================================================
# Common Keypoint Configurations
# ============================================================================

# Hands only (42 keypoints) - Original approach
hands_only_idx = lefthand_idx + righthand_idx

# Upper body + hands (48 keypoints) - Body context with hands
pose_hands_idx = upper_body_idx + lefthand_idx + righthand_idx

# Full body 75 keypoints - All MediaPipe Holistic (pose + hands)
full_75_idx = pose_idx + lefthand_idx + righthand_idx

# Full body 100 keypoints - MediaPipe Holistic + selected face landmarks (NEW)
full_100_idx = pose_idx + lefthand_idx + righthand_idx + face_idx

# Grouped versions (for normalization) - each sublist is normalized independently
hands_only_groups = [lefthand_idx, righthand_idx]
pose_hands_groups = [upper_body_idx, lefthand_idx, righthand_idx]
full_75_groups = [pose_idx, lefthand_idx, righthand_idx]
full_100_groups = [pose_idx, lefthand_idx, righthand_idx, face_idx]  # NEW

# Legacy aliases for backward compatibility
body_idx = upper_body_idx
total_idx = pose_hands_idx
all_keypoints_idx = full_75_idx  # Legacy: referred to 75 keypoints
all_keypoints_groups = full_75_groups  # Legacy: 3 groups

# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

def top_k_accuracy(logits, labels, k=5):
    top_k_preds = torch.topk(logits, k, dim=1).indices
    correct = (top_k_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
    total = labels.size(0)
    return correct / total

def save_checkpoints(model, optimizer, path_dir, epoch, name=None):
    if not os.path.exists(path_dir):
        print(f"Making directory {path_dir}")
        os.makedirs(path_dir)
    if name is None:
        filename = f'{path_dir}/checkpoints_{epoch}.pth'
    else:
        filename = f'{path_dir}/checkpoints_{epoch}_{name}.pth'
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, filename)

def load_checkpoints(model, optimizer, path, resume=True):
    if not os.path.exists(path):
        raise FileNotFoundError
    if os.path.isdir(path):
        # Get all checkpoint files
        checkpoint_files = [f for f in os.listdir(path) if f.endswith('.pth')]
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {path}")
        
        # Try to find best_val checkpoint first, then best_train, then latest
        best_val = [f for f in checkpoint_files if 'best_val' in f]
        best_train = [f for f in checkpoint_files if 'best_train' in f]
        
        if best_val:
            filename = f'{path}/{best_val[0]}'
            print(f'Loaded best validation checkpoint: {best_val[0]}')
        elif best_train:
            filename = f'{path}/{best_train[0]}'
            print(f'Loaded best training checkpoint: {best_train[0]}')
        else:
            # Extract epoch numbers from filenames
            epochs = []
            for f in checkpoint_files:
                try:
                    epoch_str = f.replace('checkpoints_', '').replace('.pth', '')
                    epoch_num = int(epoch_str.split('_')[0])
                    epochs.append((epoch_num, f))
                except (ValueError, IndexError):
                    continue
            
            if not epochs:
                raise FileNotFoundError(f"No valid checkpoint files found in {path}")
            
            latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
            filename = f'{path}/{latest_file}'
            print(f'Loaded latest checkpoint: epoch {latest_epoch}')

        checkpoints = torch.load(filename)
    else:
        print(f"Load checkpoint from file : {path}")
        checkpoints = torch.load(path)

    model.load_state_dict(checkpoints['model'])
    optimizer.load_state_dict(checkpoints['optimizer'])
    if resume:
        return checkpoints['epoch'] + 1
    else:
        return 1

def train_epoch(model, dataloader, optimizer, scheduler=None, epoch=0, epochs=0):
    device = next(model.parameters()).device
    all_loss, all_acc, all_top_5_acc = 0.0, 0.0, 0.0
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True, desc=f"Training epoch {epoch + 1}/{epochs}: ")
    for i, data in loop:
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        labels = data["labels"]
        optimizer.zero_grad()
        loss, logits = model(**data)
        loss.backward()
        optimizer.step()
        all_loss += loss.item()

        acc = accuracy(logits, labels)
        top_5_acc = top_k_accuracy(logits, labels, k=5)

        all_acc += acc
        all_top_5_acc += top_5_acc

        loop.set_postfix_str(f"Loss: {loss.item():.3f}, Acc: {acc:.3f}, Top 5 Acc: {top_5_acc:.3f}")

    if scheduler:
        scheduler.step(loss)

    all_loss /= len(dataloader)
    all_acc /= len(dataloader)
    all_top_5_acc /= len(dataloader)

    return all_loss, all_acc, all_top_5_acc

def evaluate(model, dataloader, epoch=0, epochs=0, log_per_class=True):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        epoch: Current epoch number
        epochs: Total number of epochs
        log_per_class: Whether to log per-class accuracies
    
    Returns:
        tuple: (loss, accuracy, top5_accuracy)
    """
    device = next(model.parameters()).device
    all_loss, all_acc, all_top_5_acc = 0.0, 0.0, 0.0
    
    # For per-class accuracy
    all_preds = []
    all_labels = []
    
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True,
                desc=f"Evaluation epoch {epoch + 1}/{epochs}: ")

    for i, data in loop:
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        labels = data["labels"]
        loss, logits = model(**data)
    
        all_loss += loss.item()
        acc = accuracy(logits, labels)
        top_5_acc = top_k_accuracy(logits, labels, k=5)

        all_acc += acc
        all_top_5_acc += top_5_acc
        
        # Collect predictions and labels for per-class metrics
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loop.set_postfix_str(f"Loss: {loss.item():.3f}, Acc: {acc:.3f}, Top 5 Acc: {top_5_acc:.3f}")

    all_loss /= len(dataloader)
    all_acc /= len(dataloader)
    all_top_5_acc /= len(dataloader)
    
    # Calculate and log per-class accuracy
    if log_per_class and len(all_preds) > 0:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Get unique classes present in the data
        unique_classes = np.unique(all_labels)
        
        logging.info("Per-class accuracy:")
        print("\nPer-class accuracy:")
        
        for class_idx in sorted(unique_classes):
            class_mask = all_labels == class_idx
            class_correct = (all_preds[class_mask] == all_labels[class_mask]).sum()
            class_total = class_mask.sum()
            class_acc = class_correct / class_total if class_total > 0 else 0
            
            # Map class index to gesture name (G01, G02, etc.)
            class_name = f"G{class_idx+1:02d}"
            
            log_msg = f"  Class {class_name}: Acc = {class_acc:.4f} ({class_correct}/{class_total})"
            logging.info(log_msg)
            print(log_msg)

    return all_loss, all_acc, all_top_5_acc

def create_attention_mask(mask, dtype, tgt_len = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def create_causal_attention_mask(attention_mask, input_shape, inputs_embeds):
    batch_size, query_length = input_shape[0], input_shape[1]

    expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, query_length, query_length).to(
        dtype=inputs_embeds.dtype
    )
    inverted_mask = 1.0 - expanded_mask
    expanded_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(inputs_embeds.dtype).min)
    
    causal_mask = torch.tril(torch.ones((query_length, query_length), device=inputs_embeds.device, dtype=inputs_embeds.dtype))
    expanded_mask += causal_mask[None, None, :, :]

    return expanded_mask

def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total_params, "trainable": trainable_params}

# ============================================================================
# Utility function to get keypoint configuration by name
# ============================================================================

def get_keypoint_config(config_name):
    """
    Get keypoint indices and groups by configuration name.
    
    Args:
        config_name: One of 'hands_only', 'pose_hands', 'full_75', 'full_100', 
                     'all_keypoints' (legacy, refers to 75)
    
    Returns:
        tuple: (flat_indices, grouped_indices) where:
            - flat_indices: list of all keypoint indices
            - grouped_indices: list of lists for normalization groups
    
    Examples:
        >>> flat, groups = get_keypoint_config('full_100')
        >>> print(len(flat))  # 100
        >>> print(len(groups))  # 4 (pose, left hand, right hand, face)
        
        >>> flat, groups = get_keypoint_config('full_75')
        >>> print(len(flat))  # 75
        >>> print(len(groups))  # 3 (pose, left hand, right hand)
    """
    configs = {
        'hands_only': (hands_only_idx, hands_only_groups),
        'pose_hands': (pose_hands_idx, pose_hands_groups),
        'full_75': (full_75_idx, full_75_groups),
        'full_100': (full_100_idx, full_100_groups),
        'all_keypoints': (full_75_idx, full_75_groups),  # Legacy alias (75 keypoints)
        'full': (full_75_idx, full_75_groups),  # Legacy alias (75 keypoints)
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Valid options: {list(configs.keys())}")
    
    return configs[config_name]

def print_keypoint_summary():
    """Print a summary of available keypoint configurations."""
    print("=" * 80)
    print("Keypoint Configuration Summary")
    print("=" * 80)
    print(f"MediaPipe Holistic Structure:")
    print(f"  Pose landmarks:       {total_body_idx} keypoints (indices 0-32)")
    print(f"  Left hand landmarks:  {total_hand} keypoints (indices 33-53)")
    print(f"  Right hand landmarks: {total_hand} keypoints (indices 54-74)")
    print(f"  Face landmarks:       {total_face} keypoints (indices 75-99)")
    print()
    print(f"Available Configurations:")
    print(f"  'hands_only':   {len(hands_only_idx):3d} keypoints in {len(hands_only_groups)} groups")
    print(f"  'pose_hands':   {len(pose_hands_idx):3d} keypoints in {len(pose_hands_groups)} groups")
    print(f"  'full_75':      {len(full_75_idx):3d} keypoints in {len(full_75_groups)} groups")
    print(f"  'full_100':     {len(full_100_idx):3d} keypoints in {len(full_100_groups)} groups (NEW)")
    print("=" * 80)

if __name__ == "__main__":
    # Print keypoint configuration summary
    print_keypoint_summary()
    
    # Test keypoint configurations
    print("\nTesting get_keypoint_config():")
    for config_name in ['hands_only', 'pose_hands', 'full_75', 'full_100']:
        flat, groups = get_keypoint_config(config_name)
        print(f"  {config_name:15s}: {len(flat):3d} keypoints, {len(groups)} groups")
    
    # Test attention mask creation
    print("\nTesting attention mask creation:")
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    input_embeds = torch.randn(2, 5, 768)
    expand_mask = create_causal_attention_mask(mask, (2, 5), input_embeds)
    print(f"  Attention mask shape: {expand_mask.shape}")
    print(f"  Attention mask dtype: {expand_mask.dtype}")