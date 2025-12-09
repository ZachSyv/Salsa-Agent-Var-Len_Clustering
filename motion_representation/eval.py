"""
Evaluation script for Motion VAE.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import sys
from pathlib import Path

# Add motion_representation to path
motion_rep_path = Path(__file__).parent
sys.path.insert(0, str(motion_rep_path.parent))

from motion_representation.models import MotionModel
# Backward compatibility
MotionVAE = MotionModel
from motion_representation.data.motion_dataset import create_dataloader
from motion_representation.utils.eval import (
    evaluate_model,
    visualize_reconstruction,
    visualize_latent_space,
    compute_reconstruction_metrics,
    sample_and_visualize
)
from motion_representation.config import get_config


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Motion VAE')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--lmdb_dir', type=str,
                        default='dataset_processed_New/lmdb_Salsa_pair/lmdb_train',
                        help='Path to LMDB database')
    parser.add_argument('--is_MDM', action='store_true', default=True,
                        help='Use MDM data format')
    parser.add_argument('--output_dir', type=str, default='motion_representation/eval_outputs',
                        help='Directory to save evaluation outputs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    
    eval_args = parser.parse_args()
    
    # Get base config (but don't parse command line args again)
    # Create a minimal config object
    class EvalConfig:
        def __init__(self):
            self.lmdb_dir = eval_args.lmdb_dir
            self.is_MDM = eval_args.is_MDM
            self.batch_size = eval_args.batch_size
            self.window_size = 20
            self.stride = 10
            self.use_both_roles = True
            self.recon_weight = 1.0
            self.kl_weight = 0.0001
            self.input_dim = 263
            self.hidden_dim = 512
            self.num_layers = 2
            self.latent_dim = 2048
            self.seq_len = 20
            self.dropout = 0.1
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    config = EvalConfig()
    args = eval_args
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint['config']
    
    model = MotionModel(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        latent_dim=model_config['latent_dim'],
        seq_len=model_config['seq_len'],
        dropout=model_config['dropout'],
        encoder_type=model_config.get('encoder_type', 'gru'),
        decoder_type=model_config.get('decoder_type', 'gru'),
        num_heads=model_config.get('num_heads', 8),
        ff_size=model_config.get('ff_size', 2048),
        activation=model_config.get('activation', 'gelu'),
        use_vae=model_config.get('use_vae', False),  # Default to False (vanilla AE) if not in checkpoint
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Create data loader
    print("Loading dataset...")
    # Create args object for data loader
    class DataLoaderArgs:
        def __init__(self):
            self.is_MDM = config.is_MDM
            self.device = config.device
    
    data_args = DataLoaderArgs()
    eval_loader = create_dataloader(
        args=data_args,
        lmdb_dir=config.lmdb_dir,
        window_size=config.window_size,
        stride=config.stride,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        use_both_roles=config.use_both_roles,
        normalize=True,  # Use normalized data (default, matches training)
    )
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, eval_loader, device, config)
    
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Reconstruction Loss: {metrics['recon_loss']:.4f}")
    print(f"KL Loss: {metrics['kl_loss']:.6f}")
    
    # Detailed reconstruction metrics
    if metrics['originals'] is not None and metrics['reconstructions'] is not None:
        recon_metrics = compute_reconstruction_metrics(
            metrics['originals'],
            metrics['reconstructions']
        )
        print(f"\nDetailed Metrics:")
        print(f"MSE: {recon_metrics['mse']:.6f}")
        print(f"MAE: {recon_metrics['mae']:.6f}")
        print(f"MSE per sample (mean): {recon_metrics['mse_per_sample'].mean():.6f}")
        print(f"MSE per sample (std): {recon_metrics['mse_per_sample'].std():.6f}")
    
    # Visualizations
    if metrics['originals'] is not None and metrics['reconstructions'] is not None:
        print("\nGenerating visualizations...")
        
        # Reconstruction visualization
        visualize_reconstruction(
            metrics['originals'][:args.num_samples],
            metrics['reconstructions'][:args.num_samples],
            save_path=os.path.join(args.output_dir, 'reconstructions.png'),
            num_samples=args.num_samples
        )
        
        # Latent space visualization
        if metrics['latents'] is not None:
            visualize_latent_space(
                metrics['latents'],
                save_path=os.path.join(args.output_dir, 'latent_space.png')
            )
    
    # Sample from model
    print("\nSampling from model...")
    generated = sample_and_visualize(
        model, device, config,
        num_samples=args.num_samples,
        save_dir=args.output_dir
    )
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")
    
    # Update args reference
    args = eval_args


if __name__ == '__main__':
    main()

