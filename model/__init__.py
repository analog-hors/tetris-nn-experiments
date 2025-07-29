import torch, torch.nn as nn, torch.nn.functional as F

from .transformer_block import TransformerBlock

class Model(nn.Module):
    piece_embed_dim: int

    piece_embed: nn.Embedding
    y_embed: nn.Parameter
    proj_in: nn.Linear
    proj_out: nn.Linear
    blocks: nn.ModuleList

    def __init__(self, piece_embed_dim: int):
        super().__init__()
        self.piece_embed_dim = piece_embed_dim

        self.piece_embed = nn.Embedding(7, piece_embed_dim)
        self.y_embed = nn.Parameter(torch.randn((20, 128)) * 0.02)

        self.proj_in = nn.Linear(10 + 5 * piece_embed_dim, 128)
        self.proj_out = nn.Linear(128, 7 * 4 * 10)

        self.blocks = nn.ModuleList(
            TransformerBlock(128, 128, 3, 2, 0.05)
            for _ in range(6)
        )

    def forward(self, field: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        """
        Input shapes:
            - field: `(batch, 20, 10)`
            - queue: `(batch, 5)`

        Returned shapes:
            - output: `(batch, 7, 4, 20, 10)`
        """

        # Embed each piece and flatten.
        # queue: (batch, 5 * piece_embed_dim)
        queue = self.piece_embed(queue)
        queue = queue.flatten(1)

        # Combine field with queue expanded per-row.
        # combined: (batch, 20, 10 + 5 * piece_embed_dim)
        combined = queue.unsqueeze(-2).expand((-1, 20, -1))
        combined = torch.cat((field, combined), -1)

        # Apply blocks.
        # pred: (batch, 20, 10 * 7 * 4)
        pred = self.proj_in(combined)
        for block in self.blocks:
            pred = block(pred + self.y_embed)
        pred = self.proj_out(pred)

        # Flatten and apply log softmax.
        pred = pred.flatten(1)
        pred = F.log_softmax(pred, 1)

        # Reshape and permute to output shape.
        # pred: (batch, 7, 4, 20, 10)
        pred = pred.reshape((-1, 20, 10, 7, 4))
        pred = pred.permute((0, 3, 4, 1, 2))

        return pred
