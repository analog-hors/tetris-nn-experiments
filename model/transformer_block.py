import torch, torch.nn as nn, torch.nn.functional as F

class TransformerBlock(nn.Module):
    embed_dim: int
    hiddem_dim: int
    kernel_size: int
    num_heads: int
    p_dropout: float

    attn: nn.MultiheadAttention
    norm1: nn.LayerNorm
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    norm2: nn.LayerNorm

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        kernel_size: int,
        num_heads: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hiddem_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.p_dropout = p_dropout

        assert kernel_size % 2 != 0

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, p_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(hidden_dim, embed_dim, kernel_size, padding="same")
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Input shapes:
            - seq: `(batch, seq_len, embed_dim)`
        
        Returned shapes:
            - output: `(batch, seq_len, embed_dim)`
        """

        # Apply self-attention.
        # residual: (batch, in_channels, seq_len)
        # seq: (batch, in_channels, seq_len)
        residual = seq
        seq = self.attn(seq, seq, seq, need_weights=False)[0]

        # Apply dropout and layer norm.
        # seq: (batch, in_channels, seq_len)
        seq = F.dropout(seq, self.p_dropout, self.training)
        seq = self.norm1(seq + residual)

        # Process attention context.
        # residual: (batch, in_channels, seq_len)
        # seq: (batch, in_channels, seq_len)
        residual = seq
        seq = seq.transpose(-1, -2)
        seq = self.conv1(seq)
        seq = torch.relu(seq)
        seq = F.dropout(seq, self.p_dropout, self.training)
        seq = self.conv2(seq)
        seq = seq.transpose(-1, -2)

        # Apply dropout and layer norm.
        # seq: (batch, seq_len, embed_dim)
        seq = F.dropout(seq, self.p_dropout, self.training)
        seq = self.norm2(seq + residual)

        return seq
