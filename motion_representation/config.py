"""
Configuration file for motion representation learning.
"""

import os
import argparse
from pathlib import Path


def get_config():
    """Get configuration arguments."""
    parser = argparse.ArgumentParser(description='Train Motion VAE with GRU')
    
    # Data paths
    parser.add_argument('--lmdb_dir', type=str,
                        default='dataset_processed_New/lmdb_Salsa_pair/lmdb_train',
                        help='Path to LMDB database directory')
    parser.add_argument('--is_MDM', action='store_true', default=True,
                        help='Use MDM data format (includes HML3D vectors)')
    parser.add_argument('--parent_dir', type=str, default='.',
                        help='Parent directory for relative paths')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=263,
                        help='Input feature dimension (HumanML3D)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension (GRU hidden_dim or Transformer d_model)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers (GRU layers or Transformer layers)')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Dimension of latent representation')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='Sequence length (window size)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Architecture selection
    parser.add_argument('--encoder_type', type=str, default='gru',
                        choices=['gru', 'transformer'],
                        help='Encoder architecture: gru or transformer')
    parser.add_argument('--decoder_type', type=str, default='gru',
                        choices=['gru', 'transformer'],
                        help='Decoder architecture: gru or transformer')
    parser.add_argument('--use_vae', action='store_true', default=False,
                        help='Use VAE with reparameterization. If not set, uses vanilla autoencoder (no KL loss)')
    
    # Transformer-specific parameters
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads (for transformer)')
    parser.add_argument('--ff_size', type=int, default=2048,
                        help='Feed-forward network size (for transformer)')
    parser.add_argument('--activation', type=str, default='gelu',
                        choices=['gelu', 'relu'],
                        help='Activation function for transformer')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision (AMP) for training to reduce memory usage')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=[None, 'cosine', 'step'],
                        help='Learning rate scheduler (None, cosine, or step)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--kl_weight', type=float, default=0.0001,
                        help='Weight for KL divergence loss')
    
    # Data parameters
    parser.add_argument('--window_size', type=int, default=20,
                        help='Size of motion window')
    parser.add_argument('--stride', type=int, default=10,
                        help='Stride for window sampling')
    parser.add_argument('--use_both_roles', action='store_true', default=True,
                        help='Use both leader and follower motions (joint training)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str,
                        default='motion_representation/checkpoints',
                        help='Base directory to save checkpoints')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name suffix (e.g., "Vanilla_GRU"). Will be appended to checkpoint_dir')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to specific checkpoint to resume from. If not specified, automatically finds latest checkpoint in checkpoint_dir')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=100,
                        help='Log training progress every N iterations')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cpu, etc.)')
    
    args = parser.parse_args()
    
    # Ensure device is set as attribute (for compatibility with existing code)
    if not hasattr(args, 'device'):
        args.device = args.device
    
    # Append model name to checkpoint directory if provided
    if args.model_name:
        args.checkpoint_dir = f"{args.checkpoint_dir}_{args.model_name}"
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    return args

