"""
Utility script to check for corrupted data samples in the dataset.
This can help identify if data corruption is causing NaN gradients.
"""
import torch
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_representation.data.motion_dataset import create_dataloader
from motion_representation.config import get_config


def check_data_corruption(lmdb_dir, batch_size=32, num_batches=None, window_size=20, stride=10):
    """
    Check for corrupted samples in the dataset.
    
    Args:
        lmdb_dir: Path to the LMDB dataset directory
        batch_size: Batch size for loading
        num_batches: Number of batches to check (None = all)
        window_size: Motion window size (default: 20)
        stride: Stride for window sampling (default: 10)
    """
    print("="*80)
    print("DATA CORRUPTION CHECK")
    print("="*80)
    
    # Create a minimal args object for the dataset
    class Args:
        pass
    args = Args()
    
    # Load dataset using the same function as training
    print(f"\nLoading dataset from: {lmdb_dir}")
    dataloader = create_dataloader(
        args=args,
        lmdb_dir=lmdb_dir,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for debugging to avoid multiprocessing issues
        use_both_roles=True,
    )
    
    dataset = dataloader.dataset
    
    corrupted_samples = []
    total_samples = 0
    total_batches = 0
    
    print(f"\nChecking {len(dataset)} samples in batches of {batch_size}...")
    
    for batch_idx, motion in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break
            
        total_batches += 1
        batch_size_actual = motion.shape[0]
        total_samples += batch_size_actual
        
        # Check each sample in the batch
        for sample_idx in range(batch_size_actual):
            sample = motion[sample_idx]  # (seq_len, dim)
            
            # Check for NaN
            has_nan = torch.isnan(sample).any().item()
            nan_count = torch.isnan(sample).sum().item() if has_nan else 0
            
            # Check for Inf
            has_inf = torch.isinf(sample).any().item()
            inf_count = torch.isinf(sample).sum().item() if has_inf else 0
            
            # Check for extreme values
            abs_max = sample.abs().max().item()
            has_extreme = abs_max > 1e6
            
            # Check for all zeros or constant values
            is_all_zero = (sample == 0).all().item()
            std_val = sample.std().item()
            is_constant = std_val < 1e-8
            
            if has_nan or has_inf or has_extreme or is_all_zero or is_constant:
                global_idx = batch_idx * batch_size + sample_idx
                corrupted_samples.append({
                    'global_idx': global_idx,
                    'batch_idx': batch_idx,
                    'sample_idx': sample_idx,
                    'has_nan': has_nan,
                    'nan_count': nan_count,
                    'has_inf': has_inf,
                    'inf_count': inf_count,
                    'has_extreme': has_extreme,
                    'abs_max': abs_max,
                    'is_all_zero': is_all_zero,
                    'is_constant': is_constant,
                    'std': std_val,
                    'mean': sample.mean().item(),
                    'min': sample.min().item(),
                    'max': sample.max().item()
                })
        
        # Progress update
        if (batch_idx + 1) % 100 == 0:
            print(f"  Checked {batch_idx + 1} batches ({total_samples} samples), found {len(corrupted_samples)} corrupted samples...")
    
    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total batches checked: {total_batches}")
    print(f"Total samples checked: {total_samples}")
    print(f"Corrupted samples found: {len(corrupted_samples)}")
    print(f"Corruption rate: {len(corrupted_samples)/total_samples*100:.2f}%")
    
    if corrupted_samples:
        print(f"\n" + "="*80)
        print("CORRUPTED SAMPLES DETAILS")
        print("="*80)
        
        # Group by corruption type
        nan_samples = [s for s in corrupted_samples if s['has_nan']]
        inf_samples = [s for s in corrupted_samples if s['has_inf']]
        extreme_samples = [s for s in corrupted_samples if s['has_extreme']]
        zero_samples = [s for s in corrupted_samples if s['is_all_zero']]
        constant_samples = [s for s in corrupted_samples if s['is_constant']]
        
        print(f"\nNaN samples: {len(nan_samples)}")
        print(f"Inf samples: {len(inf_samples)}")
        print(f"Extreme value samples (>1e6): {len(extreme_samples)}")
        print(f"All-zero samples: {len(zero_samples)}")
        print(f"Constant samples (std < 1e-8): {len(constant_samples)}")
        
        # Show first 10 corrupted samples
        print(f"\nFirst 10 corrupted samples:")
        for i, sample_info in enumerate(corrupted_samples[:10]):
            print(f"\n  Sample {i+1} (global_idx={sample_info['global_idx']}, batch={sample_info['batch_idx']}, sample_in_batch={sample_info['sample_idx']}):")
            print(f"    NaN: {sample_info['has_nan']} (count: {sample_info['nan_count']})")
            print(f"    Inf: {sample_info['has_inf']} (count: {sample_info['inf_count']})")
            print(f"    Extreme: {sample_info['has_extreme']} (abs_max: {sample_info['abs_max']:.2e})")
            print(f"    All-zero: {sample_info['is_all_zero']}")
            print(f"    Constant: {sample_info['is_constant']} (std: {sample_info['std']:.2e})")
            print(f"    Stats: mean={sample_info['mean']:.4f}, min={sample_info['min']:.4f}, max={sample_info['max']:.4f}")
        
        # Save corrupted sample indices
        try:
            import os
            output_file = 'corrupted_samples.txt'
            with open(output_file, 'w') as f:
                f.write("Corrupted Sample Indices\n")
                f.write("="*80 + "\n")
                for sample_info in corrupted_samples:
                    f.write(f"Global idx: {sample_info['global_idx']}, Batch: {sample_info['batch_idx']}, Sample in batch: {sample_info['sample_idx']}\n")
                    f.write(f"  NaN: {sample_info['has_nan']}, Inf: {sample_info['has_inf']}, Extreme: {sample_info['has_extreme']}\n")
                    f.write(f"  All-zero: {sample_info['is_all_zero']}, Constant: {sample_info['is_constant']}\n")
                    f.write(f"  Stats: mean={sample_info['mean']:.4f}, std={sample_info['std']:.4f}, min={sample_info['min']:.4f}, max={sample_info['max']:.4f}\n")
                    f.write("-"*80 + "\n")
            print(f"\nSaved corrupted sample indices to: {output_file}")
        except Exception as e:
            print(f"\nCould not save corrupted samples list: {e}")
    else:
        print("\n✓ No corrupted samples found!")
    
    return corrupted_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check dataset for corrupted samples')
    parser.add_argument('--lmdb_dir', type=str, required=True, help='Path to LMDB dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for checking')
    parser.add_argument('--num_batches', type=int, default=None, help='Number of batches to check (None = all)')
    parser.add_argument('--window_size', type=int, default=20, help='Motion window size')
    parser.add_argument('--stride', type=int, default=10, help='Stride for window sampling')
    
    args = parser.parse_args()
    
    check_data_corruption(
        args.lmdb_dir, 
        args.batch_size, 
        args.num_batches,
        args.window_size,
        args.stride
    )

