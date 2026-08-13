---
layout: post
title: "D2L 计算性能学习总结：从硬件、异步执行到多机 DDP"
date: 2026-08-13 10:00 +0800
tags: [D2L, PyTorch, 深度学习, GPU, DDP, 分布式训练]
toc: true
math: true
permalink: /deep-learning/computational-performance/
---

## 本篇逻辑主线

D2L“计算性能”一章把模型之外的系统问题串成了一条完整链路：硬件决定计算与搬运的上限，PyTorch 后端把张量算子交给 CPU 线程池或 CUDA 内核，异步提交让 Python 不必逐项等待；单卡不够时，数据并行把 batch 分给多张 GPU，而梯度同步又把问题从“算得快”推进到“传得快”。`DataParallel`（DP）用单进程简化多卡调用，`DistributedDataParallel`（DDP）则以一进程一 GPU 和 All-Reduce 获得更好的扩展性；跨机器后基本算法不变，但网络成为更显著的约束。

这篇总结沿着“硬件基础 → PyTorch 并行与异步计算 → 单机多卡 → DP 与 DDP → 多机训练”复盘知识框架，不复述 Notebook 的逐段实现，也不引用其中依赖旧硬件的性能数字。

## 1. 计算硬件：算力不是唯一指标

CPU追求低延迟和通用控制能力：核心较少但复杂，有分支预测、层级缓存和SIMD向量单元，适合操作系统、数据预处理和分支较多的任务。GPU用大量相对简单的执行单元换取吞吐量，尤其适合把同一种运算应用到大批数据。ASIC则针对固定计算模式设计电路，以通用性换取更高能效，例如面向张量运算的专用加速器。

| 设备 | 核心目标 | 并行与通用性 | 能效 | 典型任务 |
|---|---|---|---|---|
| CPU | 低延迟、复杂控制 | 少量强核心，通用性最高 | 中等 | 数据准备、控制逻辑、小规模计算 |
| GPU | 高吞吐数据并行 | 大量轻量执行单元，适合规则计算 | 矩阵任务上较高 | GEMM、卷积、训练与批量推理 |
| ASIC | 固定任务极致优化 | 并行结构专用，通用性最低 | 特定任务上最高 | 张量运算、低延迟或低功耗推理 |

深度学习的主干是矩阵乘法、卷积和逐元素运算：数据结构规则、相同指令可作用于大量元素，恰好能填满GPU的大量执行单元；Tensor Core等专用单元又进一步加速低精度矩阵乘加。但峰值FLOPS不等于真实速度。一轮训练还要从存储读取样本、经CPU内存和PCIe搬到显存，再读写参数与激活；多卡还要经过PCIe、NVLink或网络同步梯度。任何一环的带宽、延迟或容量不足，计算单元都会等待。

因此性能分析要同时问三件事：**算子算得多快、数据喂得上吗、设备之间传得动吗**。批量连续访问通常比大量小而随机的访问高效；足够大的batch也更容易摊薄内核启动和传输的固定成本。

## 2. PyTorch的自动并行：Python只描述工作

用户写的是Python，真正的张量计算主要落在PyTorch的C++后端及其调用的底层库中：CPU算子可利用线程池、向量指令和BLAS，CUDA张量则调用CUDA、cuBLAS、cuDNN等实现。一次大型矩阵乘法会在GPU内部启动大量CUDA线程，通常不需要用户手写线程调度。

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(2048, 2048, device=device)
y = x @ x  # CPU上可使用多线程；CUDA上由GPU内核并行执行
print(y.shape, y.device)
```

这不意味着“放到GPU一定更快”。小算子计算时间可能小于CUDA内核启动和CPU/GPU传输开销；频繁执行`.cpu()`、`.item()`或许多细碎算子，也会破坏流水。应让数据尽量长期驻留在计算设备上，并用真实输入做端到端基准。

一个普通Python进程也能向两张GPU分别提交独立任务：先提交`cuda:0`运算，再提交`cuda:1`运算，由于CUDA调用通常异步，两张卡可以同时工作，不需要DP或DDP。DP/DDP解决的是模型复制、数据划分和梯度同步，不是多设备并行存在的前提。

### 编译不是默认的“免费加速”

本章旧版Notebook用`torch.jit.script(net)`说明命令式与符号式编程的折中，但其中实测并没有显示稳定加速。编译有冷启动成本，动态控制流、输入shape变化和不支持的操作还可能导致失败、图断裂或重新编译。当前PyTorch已将TorchScript标记为弃用：运行时优化通常考虑[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)，跨环境导出则考虑[`torch.export`](https://docs.pytorch.org/docs/stable/export.html)。二者仍应由用户按场景主动启用并实测，Eager模式继续保留最好的Python兼容性和调试体验。

## 3. 异步计算：提交完成不等于计算完成

CUDA调用通常是异步的。CPU执行`y = x @ x`时，往往只是把任务加入当前设备的CUDA stream，随后即可继续提交工作；GPU在后台真正执行。只有出现依赖或显式同步时，CPU才等待GPU，例如读取`.item()`、把结果复制到CPU，或调用`torch.cuda.synchronize()`。

这会让普通墙钟计时低估耗时，因为测到的可能只是“提交任务”的时间。CUDA Event在设备时间线上记录事件，适合测量GPU工作；若用`time.perf_counter()`测端到端墙钟，则必须在开始前和结束后同步。

```python
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(2048, 2048, device=device)

if device.type == "cuda":
    _ = x @ x                         # 预热
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    y = x @ x
    end.record()
    torch.cuda.synchronize()
    print(f"GPU时间: {start.elapsed_time(end):.3f} ms")
else:
    start = time.perf_counter()
    y = x @ x
    print(f"CPU时间: {(time.perf_counter() - start) * 1000:.3f} ms")
```

异步与并行不是同义词：异步描述“调用方是否等待结果”，并行描述“任务是否在同一时刻执行”。异步提交为重叠创造条件，但有数据依赖的算子仍必须按序；不同GPU、不同stream或计算与拷贝使用不同资源时，才可能真正重叠。过度排队还会让中间结果堆积并增加显存压力。

## 4. 单机多卡：先拆数据，再同步状态

数据并行的一次迭代可以压缩成五步：

1. 在每张GPU放置一份相同模型；
2. 将全局batch切成多个数据分片；
3. 各GPU并行执行前向和反向传播，得到局部梯度；
4. 聚合局部梯度，使各模型副本得到一致梯度；
5. 各副本更新参数，并进入下一次迭代。

若有$k$个进程、每个进程本地batch为$b$，一次同步更新看到的全局batch通常是$k b$。GPU增加后若同时放大全局batch，优化行为也会变化，学习率、warmup和BatchNorm策略不能只按“卡数翻倍”机械处理。

数据并行复制完整模型，适合模型能放进单卡、但希望提高样本吞吐的场景。模型并行则把层、张量或流水阶段拆到不同设备，主要解决单卡放不下模型的问题，但会引入激活传输、切分策略和更多同步点。

多卡不会保证线性加速。实际收益取决于模型计算量是否足以覆盖梯度通信，数据加载能否跟上，GPU是否同构、分片是否均衡，以及batch是否大到能有效利用每张卡。小模型或小batch常常通信占比过高。

## 5. DataParallel：简单但中心化

`nn.DataParallel`使用一个Python进程和多个执行线程。输入先到主GPU，再被scatter到其他GPU；每次前向时复制模型，各卡计算后把输出gather回主GPU，反向梯度也归约到主模型，由主进程更新参数。

```python
import torch
from torch import nn

if torch.cuda.device_count() >= 2:
    model = nn.DataParallel(nn.Linear(32, 2), device_ids=[0, 1]).cuda(0)
    x = torch.randn(256, 32, device="cuda:0")
    logits = model(x)
else:
    print("DataParallel示例需要至少两张可见GPU")
```

DP的优点是改动少，适合快速验证。主要代价是GPU 0额外承担输入切分、输出与梯度聚合，并保存主模型，负载和显存不均衡；单进程内的Python调度也受到GIL影响。模型复制和中心化通信使它难以随GPU数量扩展，而且不能自然扩展到多机。

## 6. DistributedDataParallel：一进程一GPU

DDP本身不创建进程。命令中的`--nproc-per-node=2`让`torchrun`在本机启动两个Python进程，并分别设置`LOCAL_RANK=0/1`；脚本据此绑定`cuda:0/1`。`init_process_group()`也不扫描并接管所有GPU，它只是读取`RANK`、`WORLD_SIZE`、主节点地址等信息，让已经启动的进程建立通信组。

```bash
torchrun --standalone --nproc-per-node=2 assets/code/single-node-ddp.py
```

下面是完整源码，不依赖Notebook中的变量；为了让重点落在DDP流程上，数据集用样本编号生成确定性数据：

```python
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class ToyDataset(Dataset):
    def __init__(self, size=4096, features=32):
        self.size, self.features = size, features

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        x = torch.randn(self.features, generator=generator)
        return x, (x.sum() > 0).long()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("该示例需要支持CUDA的PyTorch")

    # torchrun为每个进程设置LOCAL_RANK、RANK和WORLD_SIZE。
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", init_method="env://", device_id=device)
    rank = dist.get_rank()

    try:
        dataset = ToyDataset()
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(
            dataset, batch_size=128, sampler=sampler,
            num_workers=2, pin_memory=True)

        model = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2)
        ).to(device)
        model = DDP(model, device_ids=[local_rank])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(5):
            sampler.set_epoch(epoch)
            model.train()
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()          # DDP在这里自动同步梯度
                optimizer.step()

            if rank == 0:
                print(f"epoch={epoch + 1}, last_loss={loss.item():.4f}")

        if rank == 0:
            torch.save(model.module.state_dict(), "toy_ddp.pt")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

这份代码也可以从[源码文件]({{ '/assets/code/single-node-ddp.py' | relative_url }})下载。`DistributedSampler`保证各rank读取不同数据；`set_epoch()`让所有rank在新一轮使用协调后的新洗牌，否则每轮顺序会重复。DDP构造时会将rank 0的模型状态同步到其他rank；此后每个进程保存完整模型，并用相同的聚合梯度独立更新参数。

### Ring All-Reduce到底做了什么

设有4个rank，每个rank根据自己的数据算出局部梯度$g_0,g_1,g_2,g_3$。目标是让每个rank最终都拥有

$$
g=\frac{g_0+g_1+g_2+g_3}{4}.
$$

把每个局部梯度按相同位置切成4块，并把rank连成逻辑环。Ring All-Reduce通常包含两个阶段：

1. **Reduce-Scatter**：共走3轮。每轮每个rank向下一个邻居发送一块，同时接收上一邻居的一块并累加；结束后，rank 0到3各自只持有一块已经汇总了全部局部梯度的结果。
2. **All-Gather**：再走3轮。各rank沿环传递自己负责的最终块；结束后，每个rank都收齐4个块，重新得到完整的全局梯度。

```text
局部梯度：
rank 0: [a0, b0, c0, d0]     rank 1: [a1, b1, c1, d1]
rank 2: [a2, b2, c2, d2]     rank 3: [a3, b3, c3, d3]

Reduce-Scatter结束：
rank 0持有 Σa    rank 1持有 Σb    rank 2持有 Σc    rank 3持有 Σd

All-Gather结束：
所有rank都持有 [Σa, Σb, Σc, Σd]
```

若梯度共$S$字节、rank数为$n$，每个rank在两个阶段传输的数据量约为

$$
2\frac{n-1}{n}S,
$$

而不是把自己的完整梯度分别广播给其余$n-1$个rank。所有设备共同传输和归约，没有专门的“梯度主GPU”。PyTorch DDP还会把参数梯度组织成多个bucket：反向传播从网络后部向前计算，当某个bucket就绪时，DDP钩子通过`ProcessGroupNCCL`立即发起All-Reduce，同时GPU继续计算其他bucket，从而让通信与反向计算重叠。

Ring是All-Reduce的一种算法，不是DDP协议本身。NCCL会结合消息规模和PCIe、NVLink、网络拓扑选择Ring、Tree或其他策略；rank 0主要用于一次性日志、指标展示和checkpoint保存，并不集中处理训练梯度。

## 7. 为什么正式训练通常选择DDP

| 对比项 | DP | DDP |
|---|---|---|
| 执行模型 | 单进程、多线程 | 多进程，通常一进程一GPU |
| Python GIL | 多线程调度受影响 | 进程各有解释器，影响较小 |
| 主卡瓶颈 | GPU 0负责scatter/gather与主模型更新 | 梯度集合通信去中心化，rank职责基本对称 |
| 梯度同步 | 归约到主GPU | NCCL All-Reduce后各rank都有一致梯度 |
| 通信计算重叠 | 能力有限 | 梯度桶可与反向计算重叠 |
| 负载与显存 | 主GPU更重 | 通常更均衡，每卡一份模型与本地数据 |
| 多机扩展 | 不支持 | 支持 |
| 适用场景 | 快速原型、简单实验 | 正式单机多卡和多机训练 |

DDP的代价是工程边界更明确：必须用`torchrun`等启动器创建进程，数据要分片，指标要跨rank归约，日志和保存要避免重复，异常时还要正确清理通信组。它也不保证任何情况下更快；当模型很小、单卡本地batch太小、输入流水慢或网络通信占比过高时，多卡收益可能有限甚至为负。正确结论不是“DDP永远快”，而是“在可扩展的同步数据并行中，DDP通常比DP拥有更合理的执行与通信结构”。

## 8. 多机训练：多机DDP与参数服务器

### 8.1 多机DDP

几个名称先统一：**node**是一台机器；**local rank**是进程在本机的编号，通常对应本机GPU；**rank**是进程在整个作业中的唯一编号；**world size**是全部节点的进程总数。例如两台机器、每台4个进程时，world size为8，每台机器的local rank都是0到3，但全局rank是0到7。

多机DDP仍使用上一节的完整脚本，数据分片、参数与梯度交互分别由下面三处定义，并不是只配置IP就会自动发生：

```python
# 1. 所有进程根据torchrun环境变量建立通信组。
dist.init_process_group("nccl", init_method="env://", device_id=device)

# 2. 各rank读取同一个逻辑数据集中的不同分片。
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=128, sampler=sampler)

# 3. 同步初始模型；backward时自动All-Reduce梯度。
model = DDP(model.to(device), device_ids=[local_rank])
loss.backward()
```

真实项目中，每台机器必须有相同版本的代码，并能从本地副本、共享文件系统或对象存储访问同一个逻辑数据集；DDP不会在节点间传输原始样本。若还要计算全局验证指标，则要显式汇总统计量：

```python
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    stats = torch.zeros(2, dtype=torch.float64, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        stats[0] += (model(x).argmax(1) == y).sum()  # 本rank正确数
        stats[1] += y.numel()                        # 本rank样本数
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1]).item()
```

两台机器分别启动同一份脚本，并指定相同的主节点地址和端口：

```bash
# 节点0；请替换GPU数量、IP和端口占位符
torchrun --nnodes=2 --node_rank=0 \
  --nproc-per-node=<GPUS_PER_NODE> \
  --master_addr=<NODE0_IP> --master_port=<PORT> train.py

# 节点1
torchrun --nnodes=2 --node_rank=1 \
  --nproc-per-node=<GPUS_PER_NODE> \
  --master_addr=<NODE0_IP> --master_port=<PORT> train.py
```

`--nnodes`是机器数，`--node_rank`标识当前机器，`--nproc-per-node`是每台机器启动的进程数。所有进程通过同一rendezvous信息组成进程组，仍然各自读取数据分片、计算局部梯度并执行All-Reduce；区别只是部分通信跨越了机器边界，带宽通常更低、延迟更高，因此梯度大小、网络拓扑和计算通信重叠更加重要。

多机故障常落在几个地方：节点间`master_port`不通；CUDA、PyTorch、NCCL或代码版本不一致；rank/world size配置错误导致集合通信永久等待；未正确使用`DistributedSampler`造成重复数据；所有rank同时打印或覆盖同一个checkpoint。遇到“卡住”时应先核对进程是否全部启动、网络端口是否互通，再检查每个rank是否以相同顺序进入集合通信。

### 8.2 参数服务器

D2L `parameterserver.ipynb`的“多机训练”主要描述另一种架构。Worker计算局部梯度后执行`push`，参数服务器聚合梯度、执行更新，再由Worker通过`pull`取得新参数：

```text
机器0 Worker ──push梯度──┐
机器1 Worker ──push梯度──┼→ 参数服务器：聚合并更新参数
机器2 Worker ──push梯度──┘              │
       ↑                                │
       └────────pull新参数───────────────┘
```

单个参数服务器容易形成网络与计算瓶颈，因此可以把参数按key切片，让多台服务器各自负责一部分。参数服务器还可以允许Worker异步push/pull，减少慢节点等待，但Worker可能使用不同版本的参数，梯度陈旧会改变优化行为。

它与多机DDP共享“不同Worker处理不同数据并聚合梯度”的数学目标，但系统实现不同：DDP没有中央更新者，各rank通过All-Reduce得到相同梯度并独立执行相同的`optimizer.step()`；参数服务器集中或分片保存状态，以Push/Pull完成交互。今天密集神经网络的同步训练通常优先DDP、FSDP等集合通信方案，而参数服务器仍适合超大稀疏Embedding、推荐系统或需要异步更新的场景。

## 9. 知识地图

可以把本章压缩成五句话：

1. 深度学习适合GPU，是因为主要算子具有规则、密集、可批量的数据并行结构；真实性能还受内存和互连限制。
2. 自动并行让底层库替我们使用CPU线程或GPU线程；异步执行让CPU无需等待每个CUDA任务结束，从而有机会重叠提交、计算与通信。
3. 单机数据并行让各GPU处理不同样本、计算局部梯度，再同步成一致梯度；通信和负载决定扩展效率。
4. DP胜在简单，DDP胜在去中心化、多进程、通信重叠和多机扩展；正式多GPU训练通常优先DDP，但结论必须由基准验证。
5. 单机DDP扩展到多机时训练逻辑不变，只是rank分布到不同node，网络从实现细节升级为主要性能约束。

## 10. 参数服务器章节练习思考

### 10.1 怎样进一步提高环同步性能？

最直接的方法是使用**双向环**：把梯度再切成两组，一组顺时针传输，另一组逆时针传输，同时利用两个方向的链路带宽。进一步还可以根据NVLink、PCIe和跨机网络的真实拓扑构造多个环；先在机器内归约、再跨机器归约、最后在机器内广播，形成分层All-Reduce。工程上还应调整bucket大小，把足够小的梯度合并成大消息以摊薄启动延迟，并尽早同步已就绪的bucket，让通信与反向计算重叠。优化目标不是固定使用某一种环，而是让可用链路尽量同时繁忙，并减少跨越低带宽边界的数据量。

### 10.2 能否在计算仍进行时异步通信？有什么影响？

可以。DDP的梯度bucket就是典型例子：后面层的梯度先算好后，立即用非阻塞All-Reduce同步，前面层继续反向计算；只需在优化器读取该bucket前等待通信完成。理想情况下，被计算覆盖的通信不再增加关键路径时间。但bucket过小会产生大量通信启动开销，过大又会推迟第一次同步；不同rank的计算速度不一致仍会让快rank等待慢rank。

还要区分“异步通信”和“异步参数更新”。前者只是重叠计算与传输，更新前仍使用本轮完整梯度，通常不改变同步SGD语义；参数服务器若允许Worker不等待其他Worker就更新，则会出现陈旧梯度，虽然吞吐可能提高，但收敛行为和可复现性都会改变。

### 10.3 长时间训练丢失一台服务器时怎样避免从头重启？

同步DDP的集合通信要求全部rank参与，一个rank消失后不能让其余rank无条件继续；需要停止当前进程组，通过弹性rendezvous按新成员关系重建进程组，再从最近checkpoint恢复。checkpoint应定期写入可靠的共享或对象存储，至少包含模型、优化器、学习率调度器、混合精度scaler、epoch/step、随机数状态和数据采样进度。日志与checkpoint写入还要具有原子性或版本号，避免故障留下半个文件。

参数服务器架构还可以采用主从复制或分片副本、心跳与故障检测、写前日志和幂等更新；某个参数分片失效后由副本接管。无论采用哪种架构，容错都不是简单“忽略丢失节点”，而是保存足够状态、检测故障、重新确定成员关系，并保证恢复后不会重复或漏掉关键更新。

## 对应资料

- D2L PyTorch版：`chapter_computational-performance`中的硬件、自动并行、异步计算、多GPU训练、简洁多GPU实现和参数服务器。
- [PyTorch DistributedDataParallel文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch分布式通信文档](https://docs.pytorch.org/docs/stable/distributed.html)
- [torchrun文档](https://docs.pytorch.org/docs/stable/elastic/run.html)
- [torch.compile文档](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
