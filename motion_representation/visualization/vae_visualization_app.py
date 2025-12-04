"""
Web application for visualizing VAE model reconstructions and latent space.
"""

import os
import sys
import gradio as gr
import numpy as np
import torch
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from motion_representation.models import MotionModel
# Backward compatibility
MotionVAE = MotionModel
from motion_representation.data.motion_dataset import create_dataloader
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain


class VAEVisualizationApp:
    """Main VAE visualization application."""
    
    def __init__(self):
        self.model: Optional[MotionVAE] = None
        self.dataloader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp_dir = tempfile.mkdtemp()
        self.current_idx = 0
        self.config = None
        
        # Default paths
        self.default_lmdb_dir = "dataset_processed_New/lmdb_Salsa_pair/lmdb_train"
        self.default_checkpoint_dir = "motion_representation/checkpoints"
    
    def load_model(self, checkpoint_path: str) -> Tuple[str, bool]:
        """
        Load VAE model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Status message and success flag
        """
        try:
            if not os.path.exists(checkpoint_path):
                return f"Error: Checkpoint not found: {checkpoint_path}", False
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.config = checkpoint['config']
            
            self.model = MotionModel(
                input_dim=self.config['input_dim'],
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                latent_dim=self.config['latent_dim'],
                seq_len=self.config['seq_len'],
                dropout=self.config['dropout'],
                encoder_type=self.config.get('encoder_type', 'gru'),
                decoder_type=self.config.get('decoder_type', 'gru'),
                num_heads=self.config.get('num_heads', 8),
                ff_size=self.config.get('ff_size', 2048),
                activation=self.config.get('activation', 'gelu'),
                use_vae=self.config.get('use_vae', False),
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            
            # Build detailed config string
            config_str = "Model Configuration:\n"
            config_str += f"  Encoder: {self.config.get('encoder_type', 'gru')}\n"
            config_str += f"  Decoder: {self.config.get('decoder_type', 'gru')}\n"
            config_str += f"  Use VAE: {self.config.get('use_vae', False)}\n"
            config_str += f"  Input Dim: {self.config.get('input_dim', 263)}\n"
            config_str += f"  Hidden Dim: {self.config.get('hidden_dim', 512)}\n"
            config_str += f"  Num Layers: {self.config.get('num_layers', 2)}\n"
            config_str += f"  Latent Dim: {self.config.get('latent_dim', 512)}\n"
            config_str += f"  Seq Len: {self.config.get('seq_len', 20)}\n"
            config_str += f"  Dropout: {self.config.get('dropout', 0.1)}\n"
            if self.config.get('encoder_type') == 'transformer' or self.config.get('decoder_type') == 'transformer':
                config_str += f"  Num Heads: {self.config.get('num_heads', 8)}\n"
                config_str += f"  FF Size: {self.config.get('ff_size', 2048)}\n"
                config_str += f"  Activation: {self.config.get('activation', 'gelu')}\n"
            
            status = f"Model loaded successfully!\n\n"
            status += config_str + "\n"
            status += f"Training Info:\n"
            status += f"  Epoch: {epoch}\n"
            status += f"  Loss: {loss:.4f}\n"
            status += f"  Device: {self.device}\n"
            status += f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            
            return status, True
        except Exception as e:
            import traceback
            error_msg = f"Error loading model: {str(e)}\n{traceback.format_exc()}"
            return error_msg, False
    
    def load_dataset(self, lmdb_dir: str, is_MDM: bool = True) -> Tuple[str, int]:
        """
        Load dataset.
        
        Args:
            lmdb_dir: Path to LMDB directory
            is_MDM: Whether data is in MDM format
            
        Returns:
            Status message and total samples
        """
        try:
            if not os.path.exists(lmdb_dir):
                return f"Error: LMDB path does not exist: {lmdb_dir}", 0
            
            # Create args object
            class DataLoaderArgs:
                def __init__(self, is_MDM_val, device_val):
                    self.is_MDM = is_MDM_val
                    self.device = device_val
            
            data_args = DataLoaderArgs(is_MDM, self.device)
            
            self.dataloader = create_dataloader(
                args=data_args,
                lmdb_dir=lmdb_dir,
                window_size=20,
                stride=10,
                batch_size=32,
                shuffle=False,
                num_workers=2,
                use_both_roles=True,
            )
            
            total_samples = len(self.dataloader.dataset)
            status = f"Dataset loaded successfully!\n"
            status += f"Total samples: {total_samples}\n"
            status += f"LMDB: {lmdb_dir}"
            
            return status, total_samples
        except Exception as e:
            import traceback
            error_msg = f"Error loading dataset: {str(e)}\n{traceback.format_exc()}"
            return error_msg, 0
    
    def visualize_reconstruction(self, idx: int) -> Tuple[Optional[str], Optional[str], str]:
        """
        Visualize original vs reconstructed motion.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (original_video_path, reconstructed_video_path, info_string)
        """
        if self.model is None:
            return None, None, "Error: Model not loaded. Please load a model first."
        
        if self.dataloader is None:
            return None, None, "Error: Dataset not loaded. Please load dataset first."
        
        try:
            if idx < 0 or idx >= len(self.dataloader.dataset):
                return None, None, f"Error: Index {idx} out of range (0-{len(self.dataloader.dataset)-1})"
            
            # Get sample
            motion = self.dataloader.dataset[idx]  # (20, 263)
            motion = motion.unsqueeze(0).to(self.device)  # (1, 20, 263)
            
            # Reconstruct
            with torch.no_grad():
                recon_motion, mean, logvar, z = self.model(motion)
            
            # Convert to numpy
            original = motion[0].cpu().numpy()  # (20, 263)
            reconstructed = recon_motion[0].cpu().numpy()  # (20, 263)
            
            # Compute metrics
            mse = np.mean((original - reconstructed) ** 2)
            mae = np.mean(np.abs(original - reconstructed))
            
            # Convert to 3D keypoints
            original_keypoints = recover_from_ric(
                torch.from_numpy(original).float().to(self.device),
                22
            ).cpu().numpy()
            
            reconstructed_keypoints = recover_from_ric(
                torch.from_numpy(reconstructed).float().to(self.device),
                22
            ).cpu().numpy()
            
            # Create videos
            original_path = os.path.join(self.temp_dir, f"original_vae_{idx}.mp4")
            reconstructed_path = os.path.join(self.temp_dir, f"reconstructed_vae_{idx}.mp4")
            
            plot_3d_motion(
                original_path,
                t2m_kinematic_chain,
                original_keypoints,
                title=f"Original (Sample {idx})",
                fps=20,
                radius=4
            )
            
            plot_3d_motion(
                reconstructed_path,
                t2m_kinematic_chain,
                reconstructed_keypoints,
                title=f"Reconstructed (MSE: {mse:.4f})",
                fps=20,
                radius=4
            )
            
            info = f"Sample Index: {idx}\n"
            info += f"MSE: {mse:.6f}\n"
            info += f"MAE: {mae:.6f}\n"
            info += f"Latent z shape: {z.shape}\n"
            info += f"Latent mean norm: {torch.norm(mean).item():.4f}\n"
            info += f"Latent std: {torch.exp(0.5 * logvar).mean().item():.4f}"
            
            return original_path, reconstructed_path, info
            
        except Exception as e:
            import traceback
            error_msg = f"Error visualizing reconstruction: {str(e)}\n{traceback.format_exc()}"
            return None, None, error_msg
    
    def visualize_latent_space(self, num_samples: int) -> Tuple[Optional[str], str]:
        """
        Visualize latent space using t-SNE.
        
        Args:
            num_samples: Number of samples to encode and visualize
            
        Returns:
            Tuple of (plot_path, info_string)
        """
        if self.model is None:
            return None, "Error: Model not loaded. Please load a model first."
        
        if self.dataloader is None:
            return None, "Error: Dataset not loaded. Please load dataset first."
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            
            total_samples = len(self.dataloader.dataset)
            num_samples = min(num_samples, total_samples)
            
            # Randomly sample indices
            indices = np.random.choice(total_samples, num_samples, replace=False)
            
            info = f"Encoding {num_samples} samples...\n"
            
            # Encode samples
            latents = []
            self.model.eval()
            
            with torch.no_grad():
                for idx in tqdm(indices, desc="Encoding samples"):
                    motion = self.dataloader.dataset[idx]  # (20, 263)
                    motion = motion.unsqueeze(0).to(self.device)  # (1, 20, 263)
                    
                    # Encode to latent
                    mean, logvar = self.model.encoder(motion)
                    z = self.model.reparameterize(mean, logvar)
                    
                    latents.append(z[0].cpu().numpy())  # (latent_dim,)
            
            latents = np.array(latents)  # (num_samples, latent_dim)
            
            info += f"Latent shape: {latents.shape}\n"
            info += f"Computing t-SNE...\n"
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_samples - 1))
            latents_2d = tsne.fit_transform(latents)
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.6, s=50)
            plt.xlabel('t-SNE Component 1', fontsize=12)
            plt.ylabel('t-SNE Component 2', fontsize=12)
            plt.title(f'VAE Latent Space Visualization (t-SNE)\n{num_samples} samples', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            info += f"t-SNE complete!\n"
            info += f"Latent dimension: {latents.shape[1]}\n"
            info += f"Latent mean: {latents.mean(axis=0).mean():.4f}\n"
            info += f"Latent std: {latents.std(axis=0).mean():.4f}\n"
            info += f"Latent range: [{latents.min():.4f}, {latents.max():.4f}]"
            
            # Save plot
            plot_path = os.path.join(self.temp_dir, f"latent_space_tsne_{num_samples}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path, info
            
        except ImportError:
            return None, "Error: sklearn not available. Please install: pip install scikit-learn"
        except Exception as e:
            import traceback
            error_msg = f"Error visualizing latent space: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg
    
    def navigate_sample(self, direction: str, current_idx: int, total_samples: int) -> int:
        """
        Navigate to next/previous sample.
        
        Args:
            direction: 'next' or 'prev'
            current_idx: Current sample index
            total_samples: Total number of samples
            
        Returns:
            New sample index
        """
        if direction == 'next':
            new_idx = min(current_idx + 1, total_samples - 1)
        elif direction == 'prev':
            new_idx = max(current_idx - 1, 0)
        else:
            new_idx = current_idx
        
        self.current_idx = new_idx
        return new_idx


def create_interface():
    """Create the Gradio interface."""
    app = VAEVisualizationApp()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1400px;
    }
    h1 {
        font-size: 28px !important;
        font-weight: 600 !important;
        margin-bottom: 10px !important;
    }
    h3 {
        font-size: 20px !important;
        font-weight: 500 !important;
        margin-top: 15px !important;
        margin-bottom: 10px !important;
    }
    """
    
    with gr.Blocks(title="VAE Motion Representation Visualization") as demo:
        gr.HTML(f"<style>{custom_css}</style>", visible=False)
        gr.Markdown("# 🎭 VAE Motion Representation Visualization")
        gr.Markdown("Visualize VAE reconstructions and explore the latent space of motion representations.")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Model & Dataset Loading")
                checkpoint_path = gr.Textbox(
                    label="Checkpoint Path",
                    value="motion_representation/checkpoints/best_checkpoint.pth",
                    placeholder="Path to VAE checkpoint"
                )
                load_model_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False, lines=3)
                
                lmdb_dir = gr.Textbox(
                    label="LMDB Directory",
                    value="dataset_processed_New/lmdb_Salsa_pair/lmdb_train",
                    placeholder="Path to LMDB directory"
                )
                is_MDM = gr.Checkbox(label="Is MDM Format", value=True)
                load_dataset_btn = gr.Button("Load Dataset", variant="primary")
                dataset_status = gr.Textbox(label="Dataset Status", interactive=False, lines=2)
            
            with gr.Column(scale=1):
                total_samples = gr.Number(label="Total Samples", value=0, interactive=False)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Reconstruction Visualization")
                with gr.Row():
                    prev_btn = gr.Button("◀ Previous", size="sm")
                    sample_idx = gr.Number(
                        label="Sample Index",
                        value=0,
                        minimum=0,
                        maximum=0,
                        step=1,
                        precision=0
                    )
                    next_btn = gr.Button("Next ▶", size="sm")
                    go_btn = gr.Button("Go", size="sm")
                
                visualize_recon_btn = gr.Button("Visualize Reconstruction", variant="primary")
                
                with gr.Row():
                    original_video = gr.Video(label="Original Motion", scale=1)
                    reconstructed_video = gr.Video(label="Reconstructed Motion", scale=1)
                
                recon_info = gr.Textbox(label="Reconstruction Info", interactive=False, lines=6)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Latent Space Visualization (t-SNE)")
                num_samples_tsne = gr.Number(
                    label="Number of Samples",
                    value=100,
                    minimum=10,
                    maximum=10000,
                    step=10,
                    precision=0
                )
                visualize_latent_btn = gr.Button("Generate Latent Space Visualization", variant="primary")
                
                latent_plot = gr.Image(label="Latent Space t-SNE Visualization")
                latent_info = gr.Textbox(label="Latent Space Info", interactive=False, lines=8)
        
        # Event handlers
        def on_load_model(checkpoint_path_val):
            status, success = app.load_model(checkpoint_path_val)
            return status
        
        def on_load_dataset(lmdb_dir_val, is_MDM_val):
            status, total = app.load_dataset(lmdb_dir_val, is_MDM_val)
            max_val = max(0, int(total) - 1) if total > 0 else 0
            return status, total, gr.update(maximum=max_val, value=0)
        
        def on_navigate(direction, current_idx_val, total_samples_val):
            try:
                current = int(current_idx_val) if current_idx_val is not None else 0
                total = int(total_samples_val) if total_samples_val is not None else 0
                new_idx = app.navigate_sample(direction, current, total)
                return new_idx
            except Exception as e:
                return 0
        
        def on_visualize_reconstruction(idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                original, reconstructed, info = app.visualize_reconstruction(idx)
                return original, reconstructed, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return None, None, error_msg
        
        def on_visualize_latent(num_samples_val):
            try:
                num_samples = int(num_samples_val) if num_samples_val is not None else 100
                plot_path, info = app.visualize_latent_space(num_samples)
                return plot_path, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return None, error_msg
        
        def on_go(idx_val):
            return on_visualize_reconstruction(idx_val)
        
        # Bind events
        load_model_btn.click(
            fn=on_load_model,
            inputs=[checkpoint_path],
            outputs=[model_status]
        )
        
        load_dataset_btn.click(
            fn=on_load_dataset,
            inputs=[lmdb_dir, is_MDM],
            outputs=[dataset_status, total_samples, sample_idx]
        )
        
        prev_btn.click(
            fn=lambda idx, total: on_navigate('prev', idx, total),
            inputs=[sample_idx, total_samples],
            outputs=[sample_idx]
        )
        
        next_btn.click(
            fn=lambda idx, total: on_navigate('next', idx, total),
            inputs=[sample_idx, total_samples],
            outputs=[sample_idx]
        )
        
        go_btn.click(
            fn=on_go,
            inputs=[sample_idx],
            outputs=[original_video, reconstructed_video, recon_info]
        )
        
        visualize_recon_btn.click(
            fn=on_visualize_reconstruction,
            inputs=[sample_idx],
            outputs=[original_video, reconstructed_video, recon_info]
        )
        
        # Auto-visualize when index changes
        sample_idx.change(
            fn=on_visualize_reconstruction,
            inputs=[sample_idx],
            outputs=[original_video, reconstructed_video, recon_info]
        )
        
        visualize_latent_btn.click(
            fn=on_visualize_latent,
            inputs=[num_samples_tsne],
            outputs=[latent_plot, latent_info]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7862)

