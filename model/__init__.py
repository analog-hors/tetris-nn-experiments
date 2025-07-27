import torch, torch.nn as nn, torch.nn.functional as F

from .rotary_transformer_block import RotaryTransformerBlock

class Model(nn.Module):
    piece_embed_dim: int

    piece_embed: nn.Embedding
    proj_in: nn.Conv1d
    proj_out: nn.Conv1d
    block1: RotaryTransformerBlock
    block2: RotaryTransformerBlock
    block3: RotaryTransformerBlock
    block4: RotaryTransformerBlock
    block5: RotaryTransformerBlock
    block6: RotaryTransformerBlock

    def __init__(self, piece_embed_dim: int):
        super().__init__()
        self.piece_embed_dim = piece_embed_dim

        self.piece_embed = nn.Embedding(7, piece_embed_dim)

        self.proj_in = nn.Conv1d(10 + 5 * piece_embed_dim, 128, 1)
        self.proj_out = nn.Conv1d(128, 7 * 4 * 10, 1)

        self.block1 = RotaryTransformerBlock(128, 128, 3, 2, 0.05)
        self.block2 = RotaryTransformerBlock(128, 128, 3, 2, 0.05)
        self.block3 = RotaryTransformerBlock(128, 128, 3, 2, 0.05)
        self.block4 = RotaryTransformerBlock(128, 128, 3, 2, 0.05)
        self.block5 = RotaryTransformerBlock(128, 128, 3, 2, 0.05)
        self.block6 = RotaryTransformerBlock(128, 128, 3, 2, 0.05)

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
        x = self.proj_in(combined)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        pred = self.proj_out(x)
        
        # Flatten and apply log softmax.
        pred = pred.flatten(1)
        pred = F.log_softmax(pred, 1)

        # Reshape and transpose to output shape.
        # pred: (batch, 7, 4, 20, 10)
        pred = pred.reshape((-1, 7, 4, 10, 20))
        pred = pred.transpose(-1, -2)

        return pred
