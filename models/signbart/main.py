import os
import argparse
import random
import logging
import torch
import numpy as np
import yaml
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

from torch.utils.data import DataLoader
from pathlib import Path

from dataset import Datasets
from model import SignBart
from utils import train_epoch, evaluate
from utils import body_idx, lefthand_idx, righthand_idx, save_checkpoints, load_checkpoints

def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--config_path", type=str, default="",
                        help="Path to the config model to be used")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to the config model to be used")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")
    parser.add_argument("--task", type=str, default=False, choices=["train", "eval"],
                        help="Whether to train or evaluate the model")

    parser.add_argument("--data_path", type=str, default="",
                        help="Path to the training dataset")

    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the model training")

    parser.add_argument("--resume_checkpoints", type=str, default="",
                        help="Path to the checkpoints to be used for resuming training")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=float, default=0.1,
                        help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")
    
    # Validation control
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable validation during training (training only)")
    
    # Checkpoint control
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--save_all_checkpoints", action="store_true",
                        help="Save checkpoint at every epoch (disk intensive)")

    return parser

def setup_logging(experiment_name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(experiment_name + ".log")
        ]
    )

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def load_model(config_path, pretrained_path, device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Verify config is properly loaded
    print(f"Loaded config from {config_path}")
    print(f"  Config keys: {list(config.keys())}")
    
    model = SignBart(config)
    model.train(True)
    model.to(device)
    
    if pretrained_path:
        print(f"Load checkpoint from file : {pretrained_path}")
        state_dict = torch.load(pretrained_path)
        ret = model.load_state_dict(state_dict, strict=False)
        
        print("Missing keys: ", ret.missing_keys)
        print("Unexpected keys: ", ret.unexpected_keys)
        
    return model, config

def prepare_data_loaders(data_path, joint_idx, generator, no_validation=False):
    train_datasets = Datasets(data_path, "train", shuffle=True, joint_idxs=joint_idx, augment=True)
    train_loader = DataLoader(train_datasets, shuffle=True, generator=generator,
                              batch_size=1, collate_fn=train_datasets.data_collator)
    
    if no_validation:
        return train_loader, None
    
    val_datasets = Datasets(data_path, "test", shuffle=True, joint_idxs=joint_idx)
    val_loader = DataLoader(val_datasets, shuffle=True, generator=generator,
                            batch_size=1, collate_fn=val_datasets.data_collator)
    return train_loader, val_loader

def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def determine_keypoint_groups(config_joint_idx):
    """
    Determine how to group keypoints for normalization.
    Returns a list of lists, where each inner list is a group to normalize together.
    
    Logic:
    - Body pose keypoints (0-32): normalize together
    - Left hand keypoints (33-53): normalize together
    - Right hand keypoints (54-74): normalize together
    - Face keypoints (75-99): normalize together
    """
    groups = []
    
    # Separate keypoints by type
    body_kpts = []
    left_hand_kpts = []
    right_hand_kpts = []
    face_kpts = []
    
    for idx in config_joint_idx:
        if idx < 33:
            body_kpts.append(idx)
        elif idx < 54:
            left_hand_kpts.append(idx)
        elif idx < 75:
            right_hand_kpts.append(idx)
        else:  # idx >= 75
            face_kpts.append(idx)
    
    # Add non-empty groups
    if body_kpts:
        groups.append(body_kpts)
    if left_hand_kpts:
        groups.append(left_hand_kpts)
    if right_hand_kpts:
        groups.append(right_hand_kpts)
    if face_kpts:
        groups.append(face_kpts)
    
    return groups

def main(args):
    g = set_random_seed(args.seed)
    setup_logging(args.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")
    if args.no_validation:
        print(f"Validation: DISABLED (training only)")
    print(f"{'='*80}\n")
    
    # Load config first to get joint indices
    model, config = load_model(args.config_path, args.pretrained_path, device)
    
    # Use joint indices from config file if available, otherwise fall back to hardcoded
    if 'joint_idx' in config and config['joint_idx']:
        config_joint_idx = config['joint_idx']
        print(f"Using joint indices from config: {len(config_joint_idx)} keypoints")
        
        # Determine keypoint groups for normalization
        joint_idx = determine_keypoint_groups(config_joint_idx)
        
        # Report what we found
        group_names = []
        for i, group in enumerate(joint_idx):
            if group[0] < 33:
                name = f"body pose (indices {group[0]}-{group[-1]}, {len(group)} keypoints)"
                group_names.append(name)
            elif group[0] < 54:
                name = f"left hand (indices {group[0]}-{group[-1]}, {len(group)} keypoints)"
                group_names.append(name)
            elif group[0] < 75:
                name = f"right hand (indices {group[0]}-{group[-1]}, {len(group)} keypoints)"
                group_names.append(name)
            else:  # group[0] >= 75
                name = f"face (indices {group[0]}-{group[-1]}, {len(group)} keypoints)"
                group_names.append(name)
        
        print(f"Keypoint groups for normalization ({len(joint_idx)} groups):")
        for i, name in enumerate(group_names):
            print(f"  Group {i+1}: {name}")
    else:
        # Fallback to hardcoded indices (hands only)
        print("Using hardcoded joint indices (left + right hand only)")
        joint_idx = [lefthand_idx, righthand_idx]
    
    checkpoint_dir = "checkpoints_" + args.experiment_name

    train_loader, val_loader = prepare_data_loaders(args.data_path, joint_idx, g, args.no_validation)
    
    # Count and report parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: ~{total_params * 4 / (1024**2):.2f} MB (float32)\n")
    
    logging.info(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Report configuration
    print(f"Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: 1")
    print(f"  Optimizer: AdamW")
    print(f"  Scheduler: ReduceLROnPlateau (factor={args.scheduler_factor}, patience={args.scheduler_patience})")
    print(f"  Seed: {args.seed}")
    print(f"  Validation: {'DISABLED' if args.no_validation else 'ENABLED'}")
    print(f"  Checkpoint strategy: Save every {args.save_every} epochs + best + latest + final")
    if args.save_all_checkpoints:
        print(f"  WARNING: Saving ALL epoch checkpoints (disk intensive!)")
    
    print(f"\nModel Architecture (from {args.config_path}):")
    print(f"  d_model: {config.get('d_model', 'N/A')}")
    print(f"  Encoder layers: {config.get('encoder_layers', 'N/A')}")
    print(f"  Decoder layers: {config.get('decoder_layers', 'N/A')}")
    print(f"  Attention heads (encoder): {config.get('encoder_attention_heads', 'N/A')}")
    print(f"  Attention heads (decoder): {config.get('decoder_attention_heads', 'N/A')}")
    print(f"  FFN dim (encoder): {config.get('encoder_ffn_dim', 'N/A')}")
    print(f"  FFN dim (decoder): {config.get('decoder_ffn_dim', 'N/A')}")
    print(f"  Dropout: {config.get('dropout', 'N/A')}")
    print(f"  Activation dropout: {config.get('activation_dropout', 'N/A')}")
    print(f"  Attention dropout: {config.get('attention_dropout', 'N/A')}")
    print(f"  Classifier dropout: {config.get('classifier_dropout', 'N/A')}")
    print(f"  Encoder layerdrop: {config.get('encoder_layerdrop', 'N/A')}")
    print(f"  Decoder layerdrop: {config.get('decoder_layerdrop', 'N/A')}")
    print(f"  Number of classes: {config.get('num_labels', 'N/A')}")
    print(f"  Max position embeddings: {config.get('max_position_embeddings', 'N/A')}")
    print(f"  Positional encoding: {config.get('pe', 'N/A')}")
    
    num_joints = len(config.get('joint_idx', []))
    if num_joints > 0:
        # Determine what keypoints are being used
        has_pose = any(idx < 33 for idx in config.get('joint_idx', []))
        has_left_hand = any(33 <= idx < 54 for idx in config.get('joint_idx', []))
        has_right_hand = any(54 <= idx < 75 for idx in config.get('joint_idx', []))
        has_face = any(idx >= 75 for idx in config.get('joint_idx', []))
        
        keypoint_types = []
        if has_pose:
            keypoint_types.append("body pose")
        if has_left_hand:
            keypoint_types.append("left hand")
        if has_right_hand:
            keypoint_types.append("right hand")
        if has_face:
            keypoint_types.append("face")
        
        desc = " + ".join(keypoint_types)
        print(f"  Joint indices: {num_joints} keypoints ({desc})")
    print()
    
    logging.info(f"Configuration: {config}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.scheduler_factor,
                                                     patience=args.scheduler_patience)

    list_train_loss, list_train_acc, list_val_loss, list_val_acc = [], [], [], []
    top_train_acc, top_val_acc = 0, 0
    lr_progress = []
    epochs = args.epochs

    if args.resume_checkpoints:
        print(f"Resume training from file : {args.resume_checkpoints}")
        resume_epoch = load_checkpoints(model, optimizer, args.resume_checkpoints, resume=True)
    else:
        resume_epoch = 0

    if args.task == "eval":
        if val_loader is None:
            print("ERROR: Cannot evaluate without validation data. Remove --no_validation flag.")
            return
        
        print("Evaluate model..!")
        model.train(False)
        start_time = time.time()
        val_loss, val_acc, val_top5_acc = evaluate(model, val_loader, epoch=0, epochs=0)
        inference_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"Evaluation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Top-1 Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Top-5 Accuracy: {val_top5_acc:.4f} ({val_top5_acc*100:.2f}%)")
        print(f"  Inference time: {inference_time:.2f} seconds")
        print(f"  Time per sample: {inference_time/len(val_loader):.4f} seconds")
        print(f"{'='*80}\n")
        
        logging.info(f"Evaluation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Top-5: {val_top5_acc:.4f}, Time: {inference_time:.2f}s")
        return

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path("out-imgs/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output images directory: out-imgs/{args.experiment_name}/\n")
    
    # Training
    total_train_time = 0
    
    for epoch in range(resume_epoch, epochs):
        epoch_start_time = time.time()
        
        train_loss, train_acc, train_top5_acc = train_epoch(model, train_loader, optimizer, scheduler, epoch=epoch, epochs=epochs)
        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)

        # Validation (if enabled)
        if val_loader is not None:
            model.train(False)
            val_loss, val_acc, val_top5_acc = evaluate(model, val_loader, epoch=epoch, epochs=epochs)
            model.train(True)

            list_val_loss.append(val_loss)
            list_val_acc.append(val_acc)
        else:
            val_loss, val_acc, val_top5_acc = 0.0, 0.0, 0.0
        
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time

        # Save best training checkpoint
        if train_acc > top_train_acc:
            top_train_acc = train_acc
            save_checkpoints(model, optimizer, checkpoint_dir, epoch, name="best_train")
            print(f"  → Saved best training checkpoint (acc: {train_acc:.4f})")
            logging.info(f"Saved best training checkpoint at epoch {epoch+1} (acc: {train_acc:.4f})")

        # Save best validation checkpoint
        if val_loader is not None and val_acc > top_val_acc:
            top_val_acc = val_acc
            save_checkpoints(model, optimizer, checkpoint_dir, epoch, name="best_val")
            print(f"  → Saved best validation checkpoint (acc: {val_acc:.4f})")
            logging.info(f"Saved best validation checkpoint at epoch {epoch+1} (acc: {val_acc:.4f})")

        # Save periodic checkpoint (every N epochs)
        if args.save_all_checkpoints or (args.save_every > 0 and (epoch + 1) % args.save_every == 0):
            save_checkpoints(model, optimizer, checkpoint_dir, epoch, name=f"epoch_{epoch+1}")
            print(f"  → Saved periodic checkpoint (epoch {epoch+1})")
            logging.info(f"Saved periodic checkpoint at epoch {epoch+1}")
        
        # Always save latest checkpoint (overwrites previous)
        save_checkpoints(model, optimizer, checkpoint_dir, epoch, name="latest")

        print(f"[{epoch + 1}/{epochs}] TRAIN  loss: {train_loss:.4f} acc: {train_acc:.4f} top5: {train_top5_acc:.4f} | time: {epoch_time:.1f}s")
        logging.info(f"[{epoch + 1}/{epochs}] TRAIN  loss: {train_loss:.4f} acc: {train_acc:.4f} top5: {train_top5_acc:.4f}")
        
        if val_loader is not None:
            print(f"[{epoch + 1}/{epochs}] VAL    loss: {val_loss:.4f} acc: {val_acc:.4f} top5: {val_top5_acc:.4f}")
            logging.info(f"[{epoch + 1}/{epochs}] VAL    loss: {val_loss:.4f} acc: {val_acc:.4f} top5: {val_top5_acc:.4f}")
        
        print("")
        logging.info("")

        lr_progress.append(optimizer.param_groups[0]["lr"])
    
    # Save final checkpoint after training completes
    print("Saving final checkpoint...")
    save_checkpoints(model, optimizer, checkpoint_dir, epochs-1, name="final")
    logging.info("Final checkpoint saved")
    
    # Final summary
    avg_epoch_time = total_train_time / (epochs - resume_epoch)
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Total training time: {total_train_time/3600:.2f} hours ({total_train_time:.1f} seconds)")
    print(f"Average time per epoch: {avg_epoch_time:.1f} seconds")
    print(f"Best train accuracy: {top_train_acc:.4f} ({top_train_acc*100:.2f}%)")
    if val_loader is not None:
        print(f"Best validation accuracy: {top_val_acc:.4f} ({top_val_acc*100:.2f}%)")
    
    # List saved checkpoints
    print(f"\nSaved checkpoints in {checkpoint_dir}:")
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        for ckpt in checkpoints:
            size_mb = os.path.getsize(os.path.join(checkpoint_dir, ckpt)) / (1024**2)
            print(f"  - {ckpt} ({size_mb:.2f} MB)")
    
    print(f"{'='*80}\n")
    
    logging.info(f"Training complete - Total time: {total_train_time:.1f}s, Best train acc: {top_train_acc:.4f}")
    if val_loader is not None:
        logging.info(f"Best val acc: {top_val_acc:.4f}")

    # Plot training curves
    fig, ax = plt.subplots()
    ax.plot(range(1, len(list_train_loss) + 1), list_train_loss, c="#D64436", label="Training loss")
    ax.plot(range(1, len(list_train_acc) + 1), list_train_acc, c="#00B09B", label="Training accuracy")
    if val_loader and list_val_acc:
        ax.plot(range(1, len(list_val_acc) + 1), list_val_acc, c="#E0A938", label="Validation accuracy")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
    ax.grid()
    fig.savefig("out-imgs/" + args.experiment_name + "_loss.png")
    plt.close()

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
    ax1.set(xlabel="Epoch", ylabel="LR", title="")
    ax1.grid()
    fig1.savefig("out-imgs/" + args.experiment_name + "_lr.png")
    plt.close()

    print("\nPlots saved to out-imgs/")
    logging.info("Experiment finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.experiment_name:
        print("ERROR: --experiment_name is required")
        exit(1)
    if not args.config_path:
        print("ERROR: --config_path is required")
        exit(1)
    if not args.data_path:
        print("ERROR: --data_path is required")
        exit(1)
    if not args.task:
        print("ERROR: --task is required (train or eval)")
        exit(1)
    
    main(args)