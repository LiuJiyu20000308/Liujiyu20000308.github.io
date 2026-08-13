"""可使用 torchrun 启动的最小单机多卡 DDP 示例。"""

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class ToyDataset(Dataset):
    """按样本编号生成确定性数据，避免依赖 notebook 隐藏状态。"""

    def __init__(self, size=4096, features=32):
        self.size = size
        self.features = features

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        x = torch.randn(self.features, generator=generator)
        y = (x.sum() > 0).long()
        return x, y


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("该示例需要支持 CUDA 的 PyTorch")

    # torchrun 为每个进程提供 LOCAL_RANK、RANK 和 WORLD_SIZE。
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # env:// 从 torchrun 设置的环境变量中读取进程组连接信息。
    dist.init_process_group("nccl", init_method="env://", device_id=device)
    rank = dist.get_rank()

    try:
        dataset = ToyDataset()
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_size=128,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
        )

        model = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2)
        ).to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(5):
            # 所有 rank 使用同一 epoch 种子，但仍取得互不重叠的数据分片。
            sampler.set_epoch(epoch)
            model.train()

            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                # DDP 在反向传播期间自动对已就绪的梯度桶执行 All-Reduce。
                loss.backward()
                optimizer.step()

            # 日志和 checkpoint 只由 rank 0 负责，避免重复写入。
            if rank == 0:
                print(f"epoch={epoch + 1}, last_loss={loss.item():.4f}")

        if rank == 0:
            torch.save(model.module.state_dict(), "toy_ddp.pt")
    finally:
        # 即使训练发生异常，也释放 NCCL 进程组资源。
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
