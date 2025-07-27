import torch
from torch.utils.data import Dataset
from typing import BinaryIO

class TetrisDataset(Dataset):
    def __init__(self, io: BinaryIO):
        self.io = io

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.io.seek(index * 209)
        buffer = self.io.read(209)

        field_bytes = buffer[:200]
        queue_bytes = buffer[200:205] 
        move_bytes = buffer[205:]

        field = torch.frombuffer(bytearray(field_bytes), dtype=torch.uint8)
        field = field.reshape((20, 10)).float()

        queue = torch.frombuffer(bytearray(queue_bytes), dtype=torch.uint8)
        queue = queue.long()

        target = torch.zeros((7, 4, 20, 10))
        target[*move_bytes] = 1.0

        return field, queue, target

    def __len__(self) -> int:
        self.io.seek(0, 2)
        return self.io.tell() // 209
