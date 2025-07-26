import torch, torch.nn as nn, torch.nn.functional as F

class RotaryMultiheadAttention(nn.Module):
    in_channels: int
    out_channels: int
    num_heads: int
    p_dropout: float

    conv_q: nn.Conv1d
    conv_k: nn.Conv1d
    conv_v: nn.Conv1d
    conv_o: nn.Conv1d

    _cached_sin_cos: torch.Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.p_dropout = p_dropout

        assert in_channels % self.num_heads == 0
        assert in_channels // self.num_heads % 2 == 0

        self.conv_q = nn.Conv1d(in_channels, in_channels, 1)
        self.conv_k = nn.Conv1d(in_channels, in_channels, 1)
        self.conv_v = nn.Conv1d(in_channels, in_channels, 1)
        self.conv_o = nn.Conv1d(in_channels, out_channels, 1)

        self._cached_sin_cos = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input shapes:
            - query: `(batch, in_channels, seq_len)`
            - key: `(batch, in_channels, seq_len)`
            - value: `(batch, in_channels, seq_len)`

        Returned shapes:
            - output: `(batch, in_channels, seq_len)`
        """

        batch, in_channels, seq_len = query.shape
        head_channels = in_channels // self.num_heads

        # Project query, key, and value tensors.
        # q: (batch, in_channels, seq_len)
        # k: (batch, in_channels, seq_len)
        # v: (batch, in_channels, seq_len)
        q = self.conv_q(query)
        k = self.conv_k(key)
        v = self.conv_v(value)

        # Split channels across heads and move channel dimension to last.
        # q: (batch, num_heads, seq_len, head_channels)
        # k: (batch, num_heads, seq_len, head_channels)
        # v: (batch, num_heads, seq_len, head_channels)
        q = q.reshape((batch, self.num_heads, head_channels, seq_len)).transpose(2, 3)
        k = k.reshape((batch, self.num_heads, head_channels, seq_len)).transpose(2, 3)
        v = v.reshape((batch, self.num_heads, head_channels, seq_len)).transpose(2, 3)

        # Rotate query and key tensors to encode relative position.
        # q: (batch, num_heads, seq_len, head_channels)
        # k: (batch, num_heads, seq_len, head_channels)
        q = self._rotate(q)
        k = self._rotate(k)

        # Compute context matrix.
        # context: (batch, num_heads, seq_len, head_channels)
        attn_dropout = self.p_dropout if self.training else 0.0
        context = F.scaled_dot_product_attention(q, k, v, None, attn_dropout)
        
        # Rejoin heads and project output.
        # context: (batch, out_channels, seq_len)
        context = context.transpose(2, 3).reshape((batch, in_channels, seq_len))
        context = self.conv_o(context)

        return context

    def _rotate(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Input shapes:
            - seq: `(batch, num_heads, seq_len, head_channels)`

        Returned shapes:
            - output: `(batch, num_heads, seq_len, head_channels)`
        """

        _batch, _num_heads, seq_len, head_channels = seq.shape
        pairs = head_channels // 2

        if self._cached_sin_cos is None or self._cached_sin_cos.shape[1] < seq_len:
            # Compute freqs and indices.
            # freqs: (pairs)
            # indices: (seq_len)
            freqs = 10_000 ** -(torch.arange(pairs, device=seq.device).float() / pairs)
            indices = torch.arange(seq_len, device=seq.device)
            
            # Compute angles and sin-cos tensor.
            # angles: (seq_len, pairs)
            # _cached_sin_cos: (2, seq_len, pairs)
            angles = indices.unsqueeze(1) * freqs
            self._cached_sin_cos = torch.stack((
                torch.sin(angles),
                torch.cos(angles),
            ))

        # Unpack x and y.
        # x: (batch, num_heads, seq_len, pairs)
        # y: (batch, num_heads, seq_len, pairs)
        x = seq[:, :, :, :pairs]
        y = seq[:, :, :, pairs:]

        # Unpack sin and cos.
        # sin: (seq_len, pairs)
        # cos: (seq_len, pairs)
        sin = self._cached_sin_cos[0, :seq_len]
        cos = self._cached_sin_cos[1, :seq_len]

        # Rotate pairs and recombine.
        # output: (batch, num_heads, seq_len, head_channels)
        output = torch.cat((
            x * cos - y * sin,
            x * sin + y * cos,
        ), -1)

        return output
