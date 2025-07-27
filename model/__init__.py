import torch, torch.nn as nn, torch.nn.functional as F

from .transformer_block import TransformerBlock

class Model(nn.Module):
    piece_embed_dim: int

    piece_embed: nn.Embedding
    y_embed: nn.Parameter
    proj_in: nn.Conv1d
    proj_out: nn.Conv1d
    blocks: nn.ModuleList

    def __init__(self, piece_embed_dim: int):
        super().__init__()
        self.piece_embed_dim = piece_embed_dim

        self.piece_embed = nn.Embedding(7, piece_embed_dim)
        self.y_embed = nn.Parameter(torch.randn((128, 20)) * 0.02)

        self.proj_in = nn.Conv1d(10 + 5 * piece_embed_dim, 128, 1)
        self.proj_out = nn.Conv1d(128, 7 * 4 * 10, 1)

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

        # Transpose field to interpret columns as channels.
        # field: (batch, 10, 20)
        field = field.transpose(1, 2)

        # Combine field with queue expanded per-row.
        # combined: (batch, 10 + 5 * piece_embed_dim, 20)
        combined = queue.unsqueeze(2).expand((-1, -1, 20))
        combined = torch.cat((field, combined), 1)

        # Apply blocks.
        # pred: (batch, 7 * 4 * 10, 20)
        pred = self.proj_in(combined)
        for block in self.blocks:
            pred = block(pred + self.y_embed)
        pred = self.proj_out(pred)

        # Flatten and apply log softmax.
        pred = pred.flatten(1)
        pred = F.log_softmax(pred, 1)

        # Reshape and transpose to output shape.
        # pred: (batch, 7, 4, 20, 10)
        pred = pred.reshape((-1, 7, 4, 10, 20))
        pred = pred.transpose(-1, -2)

        return pred
