import torch, torch.nn as nn, torch.nn.functional as F

class Model(nn.Module):
    piece_embed_dim: int

    piece_embed: nn.Embedding
    down: nn.Conv1d
    down2: nn.Conv1d
    mid: nn.Conv1d
    up1: nn.ConvTranspose1d
    up2: nn.Conv1d

    def __init__(self, piece_embed_dim: int):
        super().__init__()
        self.piece_embed_dim = piece_embed_dim

        self.piece_embed = nn.Embedding(7, piece_embed_dim)

        self.down1 = nn.Conv1d(10 + 5 * piece_embed_dim, 32, kernel_size=3, padding=1)
        self.down2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        self.mid = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        
        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.up2 = nn.Conv1d(64, 10, kernel_size=3, padding=1)

    def forward(self, field: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        """
        Input shapes:
            - field: `(batch, 200)`
            - queue: `(batch, 5)`

        Returned shapes:
            - output: `(batch, in_channels, seq_len)`
        """

        # Embed each piece and flatten.
        # queue: (batch, 5 * piece_embed_dim)
        queue = self.piece_embed(queue)
        queue = queue.flatten(1)

        # Reshape and transpose field to interpret columns as channels.
        # field: (batch, 10, 20)
        field = field.reshape((-1, 20, 10))
        field = field.transpose(1, 2)

        # Combine field with queue expanded per-row.
        # combined: (batch, 10 + 5 * piece_embed_dim, 20)
        combined = queue.unsqueeze(2).expand((-1, -1, 20))
        combined = torch.cat((field, combined), 1)

        # Apply down1.
        # down1: (batch, 32, 20)
        down1 = self.down1(combined)
        down1 = F.relu(down1)

        # Apply down2.
        # down2: (batch, 64, 10)
        down2 = F.max_pool1d(down1, 2)
        down2 = self.down2(down2)
        down2 = F.relu(down2)

        # Apply mid.
        # mid: (batch, 64, 10)
        mid = self.mid(down2)

        # Apply up1.
        # up1: (batch, 32, 20)
        up1 = self.up1(mid)

        # Apply up2.
        # up2: (batch, 10, 20)
        up2 = torch.cat((up1, down1), 1)
        up2 = self.up2(up2)
        
        # Transpose and flatten back to input shape.
        # pred: (batch, 200)
        pred = up2.transpose(1, 2)
        pred = pred.flatten(1)

        return pred
