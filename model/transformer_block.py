import torch, torch.nn as nn, torch.nn.functional as F

from .mha import MultiheadAttention
from .channel_layer_norm import ChannelLayerNorm

class TransformerBlock(nn.Module):
    in_channels: int
    hidden_channels: int
    kernel_size: int
    num_heads: int
    p_dropout: float

    attn: MultiheadAttention
    norm1: ChannelLayerNorm
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    norm2: ChannelLayerNorm

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_heads: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.p_dropout = p_dropout

        assert kernel_size % 2 != 0

        self.attn = MultiheadAttention(
            in_channels,
            in_channels,
            num_heads,
            p_dropout,
        )
        self.norm1 = ChannelLayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(hidden_channels, in_channels, kernel_size, padding="same")
        self.norm2 = ChannelLayerNorm(in_channels)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Input shapes:
            - seq: `(batch, in_channels, seq_len)`
            - mask: `(batch, 1, seq_len)`
        
        Returned shapes:
            - output: `(batch, in_channels, seq_len)`
        """

        # Apply self-attention.
        # residual: (batch, in_channels, seq_len)
        # seq: (batch, in_channels, seq_len)
        residual = seq
        seq = self.attn(seq, seq, seq)

        # Apply dropout and layer norm.
        # seq: (batch, in_channels, seq_len)
        seq = F.dropout(seq, self.p_dropout, self.training)
        seq = self.norm1(seq + residual)

        # Process attention context.
        # residual: (batch, in_channels, seq_len)
        # seq: (batch, in_channels, seq_len)
        residual = seq
        seq = self.conv1(seq)
        seq = torch.relu(seq)
        seq = F.dropout(seq, self.p_dropout, self.training)
        seq = self.conv2(seq)

        # Apply dropout and layer norm.
        # seq: (batch, seq_len, embed_dim)
        seq = F.dropout(seq, self.p_dropout, self.training)
        seq = self.norm2(seq + residual)

        return seq
