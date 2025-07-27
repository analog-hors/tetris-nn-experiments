import torch, torch.nn.functional as F, sys, time
from torch.utils.data import DataLoader, RandomSampler

from model import Model
from dataset import TetrisDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 10
CHECKPOINT_INTERVAL = 1000

if __name__ == "__main__":
    dataset = TetrisDataset(open("data1.bin", "rb"))
    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        sampler=RandomSampler(dataset, replacement=True, num_samples=1_000_000_000),
        pin_memory=True,
        num_workers=1,
    )

    model = Model(8).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters())
    batches = 0

    if len(sys.argv) == 2:
        checkpoint = torch.load(sys.argv[1])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        batches = checkpoint["batches"]

    model.train()
    running_start = time.time()
    running_loss = 0.0
    for field, queue, target in dataloader:
        field = field.to(DEVICE, non_blocking=True)
        queue = queue.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        pred = model(field, queue)
        loss = F.binary_cross_entropy_with_logits(pred, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches += 1

        if batches % LOG_INTERVAL == 0:
            avg_running_loss = running_loss / LOG_INTERVAL
            batches_per_sec = LOG_INTERVAL / (time.time() - running_start)
            print(f"[{batches}] loss: {avg_running_loss}, {batches_per_sec:.2f} batches/sec", flush=True)

            running_start = time.time()
            running_loss = 0.0

        if batches % CHECKPOINT_INTERVAL == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "batches": batches,
            }, f"checkpoints/{batches:06}-model.pth")
