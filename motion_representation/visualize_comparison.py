"""
Visualize original vs reconstructed motions side by side.
Creates comparison videos showing original and reconstructed motions.
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from motion_representation.models import MotionModel
# Backward compatibility
MotionVAE = MotionModel
from motion_representation.data.motion_dataset import create_dataloader
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain


def create_side_by_side_comparison(
    original_motion,
    reconstructed_motion,
    output_path,
    title_original="Original",
    title_reconstructed="Reconstructed",
    fps=20
):
    """
    Create a side-by-side comparison video of original and reconstructed motions.
    
    Args:
        original_motion: Original motion in HumanML3D format (seq_len, 263) or 3D keypoints (seq_len, 22, 3)
        reconstructed_motion: Reconstructed motion in same format
        output_path: Path to save the comparison video
        title_original: Title for original motion
        title_reconstructed: Title for reconstructed motion
        fps: Frames per second for the video
    """
    # Convert to 3D keypoints if needed
    if original_motion.shape[-1] == 263:
        # HumanML3D format - convert to 3D keypoints
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        original_keypoints = recover_from_ric(
            torch.from_numpy(original_motion).float().to(device),
            22
        ).cpu().numpy()
    else:
        original_keypoints = original_motion
    
    if reconstructed_motion.shape[-1] == 263:
        # HumanML3D format - convert to 3D keypoints
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        reconstructed_keypoints = recover_from_ric(
            torch.from_numpy(reconstructed_motion).float().to(device),
            22
        ).cpu().numpy()
    else:
        reconstructed_keypoints = reconstructed_motion
    
    # Create separate videos for original and reconstructed
    original_path = output_path.replace('.mp4', '_original.mp4')
    reconstructed_path = output_path.replace('.mp4', '_reconstructed.mp4')
    
    print(f"Creating original motion video: {original_path}")
    plot_3d_motion(
        original_path,
        t2m_kinematic_chain,
        original_keypoints,
        title=title_original,
        fps=fps,
        radius=4
    )
    
    print(f"Creating reconstructed motion video: {reconstructed_path}")
    plot_3d_motion(
        reconstructed_path,
        t2m_kinematic_chain,
        reconstructed_keypoints,
        title=title_reconstructed,
        fps=fps,
        radius=4
    )
    
    print(f"\nVideos saved:")
    print(f"  Original: {original_path}")
    print(f"  Reconstructed: {reconstructed_path}")
    print(f"\nYou can view them side by side or use a video editor to create a split-screen comparison.")


def visualize_samples(
    model,
    dataloader,
    device,
    num_samples=5,
    output_dir="motion_representation/visualizations",
    fps=20
):
    """
    Visualize original vs reconstructed motions for multiple samples.
    
    Args:
        model: Trained VAE model
        dataloader: DataLoader for motion windows
        device: Device to run on
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
        fps: Frames per second for videos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, motion_batch in enumerate(tqdm(dataloader, desc="Visualizing")):
            if sample_count >= num_samples:
                break
            
            motion_batch = motion_batch.to(device)
            batch_size = motion_batch.size(0)
            
            # Reconstruct
            # Note: model.forward() automatically extracts first frame and uses it for decoder
            recon_motion, mean, logvar, z = model(motion_batch)
            
            # Process each sample in the batch
            for i in range(min(batch_size, num_samples - sample_count)):
                original = motion_batch[i].cpu().numpy()  # (20, 263)
                reconstructed = recon_motion[i].cpu().numpy()  # (20, 263)
                
                # Compute reconstruction error
                mse = np.mean((original - reconstructed) ** 2)
                
                output_path = os.path.join(
                    output_dir,
                    f"comparison_sample_{sample_count:03d}_mse_{mse:.4f}.mp4"
                )
                
                print(f"\nSample {sample_count + 1}/{num_samples}")
                print(f"MSE: {mse:.6f}")
                
                create_side_by_side_comparison(
                    original,
                    reconstructed,
                    output_path,
                    title_original=f"Original (Sample {sample_count + 1})",
                    title_reconstructed=f"Reconstructed (MSE: {mse:.4f})",
                    fps=fps
                )
                
                sample_count += 1
    
    print(f"\n✓ Visualization complete! {sample_count} samples saved to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize original vs reconstructed motions')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--lmdb_dir', type=str,
                        default='dataset_processed_New/lmdb_Salsa_pair/lmdb_train',
                        help='Path to LMDB database')
    parser.add_argument('--is_MDM', action='store_true', default=True,
                        help='Use MDM data format')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str,
                        default='motion_representation/visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for loading data')
    parser.add_argument('--fps', type=int, default=20,
                        help='Frames per second for videos')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Create data loader
    print("Loading dataset...")
    class DataLoaderArgs:
        def __init__(self):
            self.is_MDM = args.is_MDM
            self.device = device
    
    data_args = DataLoaderArgs()
    dataloader = create_dataloader(
        args=data_args,
        lmdb_dir=args.lmdb_dir,
        window_size=20,
        stride=10,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        use_both_roles=True,
    )
    
    # Visualize
    print(f"\nVisualizing {args.num_samples} samples...")
    visualize_samples(
        model,
        dataloader,
        device,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        fps=args.fps
    )


if __name__ == '__main__':
    main()




'''
python -m motion_representation.visualize_comparison \
    --checkpoint motion_representation/checkpoints/best_checkpoint.pth \
    --lmdb_dir dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
    --is_MDM \
    --num_samples 5 \
    --output_dir motion_representation/checkpoints/visualizations \
    --fps 20
'''