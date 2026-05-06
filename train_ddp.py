import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    # torchrun sets these environment variables automatically
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    n_images = int(1e4)
    images = np.random.random((n_images, 64, 64)).astype(np.float32)
    dataset = ImageDataset(images)

    sampler = DistributedSampler(dataset, num_replicas=world_size, 
                                 rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=128, sampler=sampler)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 64, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    ddp_model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    optimizer = optim.Adam(ddp_model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    n_epochs = 3
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        total_loss = 0.0

        for batch in loader:
            x = batch.to(device)
            y = torch.zeros(x.size(0), dtype=torch.long, device=device)

            logits = ddp_model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {total_loss / len(loader):.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()