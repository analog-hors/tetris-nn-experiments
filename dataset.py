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
        self.io.seek(index * 205)
        buffer = self.io.read(205)

        field_bytes = torch.frombuffer(bytearray(buffer[:200]), dtype=torch.uint8)
        queue_bytes = torch.frombuffer(bytearray(buffer[200:]), dtype=torch.uint8)

        field = (field_bytes == 1).float()
        queue = queue_bytes.long()
        target = (field_bytes == 2).float()

        return field, queue, target

    def __len__(self) -> int:
        self.io.seek(0, 2)
        return self.io.tell() // 205
