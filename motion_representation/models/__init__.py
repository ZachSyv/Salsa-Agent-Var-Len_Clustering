"""Motion representation models."""

from .motion_model import MotionModel, vae_loss
from .encdec_gru import GRUEncoder, GRUDecoder
from .encdec_transformer import TransformerEncoder, TransformerDecoder
from .positional_encoding import PositionalEncoding

# Backward compatibility alias
MotionVAE = MotionModel

__all__ = [
    'MotionModel',
    'MotionVAE',  # Backward compatibility
    'vae_loss',
    'GRUEncoder', 
    'GRUDecoder',
    'TransformerEncoder',
    'TransformerDecoder',
    'PositionalEncoding'
]

