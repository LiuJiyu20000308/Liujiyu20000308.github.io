---
layout: post
title: "D2L 微调学习总结：从迁移学习到热狗识别"
date: 2026-08-13 16:00 +0800
tags: [D2L, PyTorch, 深度学习, 迁移学习, 微调, 计算机视觉]
toc: true
math: true
permalink: /deep-learning/fine-tuning/
---

## 本篇主线

训练图像模型时，经常遇到一个不对称：手里的目标数据只有几千张，想使用的网络却有上千万个参数。从随机参数开始训练，模型既要重新学习边缘、纹理和形状，又要学习目标类别，很容易在小数据上过拟合。**微调（fine-tuning）**的核心思想，是把大型源数据集上已经学到的视觉知识作为起点，只让模型针对目标任务做必要调整。

本文以D2L的热狗二分类为主线，回答四个问题：预训练模型究竟迁移了什么；PyTorch中怎样完成标准微调；怎样复用ImageNet中已有的“hot dog”分类权重；什么时候应该冻结部分网络。最后结合受控实验回答Notebook练习1～3。

## 1. 微调到底迁移了什么

设预训练网络可以拆成

$$
z=f_\theta(x),\qquad y=Wz+b.
$$

$f_\theta$是特征提取器，输出层$W,b$把特征映射成源数据集的类别。卷积网络靠近输入的层通常学习边缘、颜色和纹理，中间层逐渐组合局部形状，靠近输出的层更依赖具体类别。ImageNet训练不仅给出了一个1000分类器，也给模型提供了一套有用的视觉表示。

当目标任务变成“热狗/非热狗”时，ResNet-18的卷积结构和大部分参数仍然有价值，但原来的1000维输出不再匹配目标标签。因此标准微调分四步：

```text
ImageNet预训练模型
        ↓ 保留网络结构和特征参数
替换1000分类输出层为2分类输出层
        ↓
用较大学习率训练新输出层
        +
用较小学习率调整预训练特征
```

这里不是简单“接着训练”。真正需要做的决策包括：源任务和目标任务是否相似、哪些层应该更新、不同参数组使用多大学习率，以及新输出层怎样初始化。

## 2. 热狗数据集怎样进入模型

D2L热狗数据集包含`hotdog`和`not-hotdog`两个目录，`ImageFolder`根据目录名建立标签。训练图像的尺寸和长宽比不同，所以训练阶段使用随机裁剪和水平翻转；验证阶段使用确定性的缩放与中心裁剪。预训练ResNet要求输入采用与ImageNet一致的通道归一化。

```python
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

train_set = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, "train"), transform=train_transform)
test_set = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, "test"), transform=test_transform)

print(train_set.class_to_idx)  # 不要凭印象硬编码标签顺序
```

随机增广只用于训练集。若验证集也随机裁剪，同一模型每次验证会看到不同输入，指标难以比较。归一化参数也不是随意的常数，而是与预训练权重的输入分布配套；使用错误预处理会削弱迁移效果。

## 3. PyTorch中的标准微调

现代`torchvision`通过权重枚举加载预训练模型。ResNet-18的`fc`原本接收512维特征并输出1000个ImageNet类别；热狗任务只需把它替换成2分类层。

```python
from torchvision.models import ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
net = torchvision.models.resnet18(weights=weights)
net.fc = nn.Linear(net.fc.in_features, 2)
nn.init.xavier_uniform_(net.fc.weight)
```

新分类层从头学习，预训练骨干只需小幅调整，所以常用**差分学习率**：骨干使用基础学习率，输出层使用其10倍。倍数不是定律，只是清楚表达“新层多走一点、旧层少走一点”的起点。

```python
base_lr = 5e-4

backbone_params = [
    parameter
    for name, parameter in net.named_parameters()
    if not name.startswith("fc.")
]

optimizer = torch.optim.SGD(
    [
        {"params": backbone_params, "lr": base_lr},
        {"params": net.fc.parameters(), "lr": base_lr * 10},
    ],
    weight_decay=1e-3,
)
criterion = nn.CrossEntropyLoss()
```

训练循环本身与普通监督学习相同：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

for images, labels in train_loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(net(images), labels)
    loss.backward()
    optimizer.step()
```

迁移学习的收益不是“冻结了一份答案”，而是优化起点更接近有用解：小数据不必再次发明通用视觉特征，梯度主要用于适配目标域和重新划分类别边界。

## 4. Trick一：复用ImageNet的热狗输出权重

标准做法会丢弃整个1000分类输出层。不过ImageNet本身含有`hot dog`类，在当前ResNet-18类别顺序中索引为934。源分类层第934行

$$
z_{hotdog}=w_{934}^{\mathsf T}h+b_{934}
$$

已经学会在512维特征中寻找“像热狗”的方向，因此可以把它作为新二分类层热狗输出的初值，同时复制偏置。

二分类Softmax比较的是两个logit，而不是只看热狗logit的绝对值。只复制热狗行、让非热狗行保持任意随机尺度并不完整。一个实用近似是：热狗行复制第934类，其余999类权重和偏置的均值作为“通用非热狗”初值。

```python
weights = torchvision.models.ResNet18_Weights.DEFAULT
net = torchvision.models.resnet18(weights=weights)

# 替换输出层前先保留ImageNet分类层。
source_fc = net.fc
net.fc = nn.Linear(source_fc.in_features, 2)

# ImageFolder按目录建立标签，读取映射比硬编码0和1更安全。
hotdog_label = train_set.class_to_idx["hotdog"]
not_hotdog_label = train_set.class_to_idx["not-hotdog"]

# 选择除ImageNet hot dog外的其他类别。
imagenet_hotdog_index = 934
other_classes = torch.ones(source_fc.out_features, dtype=torch.bool)
other_classes[imagenet_hotdog_index] = False

# 参数初始化不应进入自动微分计算图。
with torch.no_grad():
    net.fc.weight[hotdog_label].copy_(
        source_fc.weight[imagenet_hotdog_index])
    net.fc.bias[hotdog_label].copy_(
        source_fc.bias[imagenet_hotdog_index])

    net.fc.weight[not_hotdog_label].copy_(
        source_fc.weight[other_classes].mean(dim=0))
    net.fc.bias[not_hotdog_label].copy_(
        source_fc.bias[other_classes].mean())
```

这个变换不是数学上的精确合并。原1000分类器中“不是热狗”的联合分数对应其余999个logit的`logsumexp`，它不是一个线性神经元；取均值只是为继续微调提供一个有语义的背景起点。

这项技巧适用于源标签和目标标签有明确对应关系时。如果目标任务是“食物/非食物”，单独复制热狗行就不再对应完整正类；如果类别名称相同但拍摄域差异很大，也不应假设一定提升。本次单次实验中，权重复用与普通Xavier初始化的最终准确率接近，说明它是**更有信息的初始化选择，而不是保证增益**。

## 5. Trick二：冻结部分网络

目标数据很少或算力有限时，可以把预训练骨干当作固定特征提取器，只训练新分类层：

```python
# 先冻结全部参数。
for parameter in net.parameters():
    parameter.requires_grad = False

# 再明确解冻新输出层。
for parameter in net.fc.parameters():
    parameter.requires_grad = True

# 优化器只接收真正需要更新的参数。
optimizer = torch.optim.SGD(
    net.fc.parameters(), lr=5e-3, weight_decay=1e-3)
```

也可以只冻结较通用的前半部分，让靠近输出的层适配新任务。例如保留`layer4`和`fc`可训练：

```python
for name, parameter in net.named_parameters():
    parameter.requires_grad = name.startswith(("layer4.", "fc."))

optimizer = torch.optim.SGD(
    (p for p in net.parameters() if p.requires_grad),
    lr=1e-3,
    weight_decay=1e-3,
)
```

冻结的优点是减少反向计算和梯度显存，降低小数据破坏通用特征的风险；代价是模型适配目标域的能力下降。当目标数据与ImageNet相似且非常少时，固定特征提取器往往是可靠起点；数据增多或目标域差异变大时，应逐步解冻靠后的层，再考虑完整微调。

还有一个容易漏掉的细节：`requires_grad=False`只停止参数梯度，不会自动阻止BatchNorm在训练模式下更新`running_mean`和`running_var`。若希望骨干状态完全固定，需要让冻结部分保持`eval()`；而每次调用整个模型的`train()`后，又要重新把冻结模块切回`eval()`。冻结“参数”和冻结“模块状态”不是同一件事。

## 6. 练习1～3：实验结果与结论

为了回答练习而不依赖Notebook中残留的历史输出，我在当前环境做了一次受控复现：RTX 5070 Ti、PyTorch 2.7.1、当前`ResNet18_Weights.DEFAULT`、随机种子42、batch size 128、训练5个epoch。表中是最后一个epoch的测试准确率；每个设置只运行一次，因此数字用于解释趋势，不是性能基准。

| 设置 | 骨干学习率 | 分类头学习率 | 测试准确率 |
|---|---:|---:|---:|
| 完整微调 | $5\times10^{-5}$ | $5\times10^{-4}$ | 0.7450 |
| 完整微调 | $5\times10^{-4}$ | $5\times10^{-3}$ | 0.8600 |
| 完整微调 | $5\times10^{-3}$ | $5\times10^{-2}$ | 0.9187 |
| 完整微调，学习率过大 | $5\times10^{-2}$ | $5\times10^{-1}$ | 0.5000 |
| 从零训练 | $5\times10^{-4}$ | 同左 | 0.7037 |
| 从零训练，调高学习率 | $5\times10^{-3}$ | 同左 | 0.8488 |
| 冻结骨干，只训练分类头 | 不更新 | $5\times10^{-3}$ | 0.8387 |

### 练习1：继续提高微调学习率会怎样

结论不是单调上升或单调下降，而是存在合适区间。学习率太小时，5个epoch内模型尚未充分适配；适当提高后收敛更快；继续增大则会跨过较好解，甚至迅速破坏预训练表示。本次实验中$5\times10^{-3}$优于更小设置，但$5\times10^{-2}$直接退化到二分类随机水平。不能据此把$5\times10^{-3}$当作所有任务的最佳值，正确做法是在验证集上按数量级搜索，并分别关注骨干和分类头的学习率。

### 练习2：分别调整微调和从零训练的超参数后，差距还存在吗

差距会缩小，但小数据下通常仍存在。将从零训练学习率从$5\times10^{-4}$调到$5\times10^{-3}$后，准确率从0.7037提高到0.8488，说明“不公平的默认超参数”会夸大迁移学习优势；在同一组实验里，完整微调仍达到0.9187，说明预训练表示带来的数据效率没有被超参数调整完全替代。

这不是“微调永远胜出”的定理。若目标数据足够大、源域与目标域差异很大，或从零模型获得更长训练、更合适的调度器和正则化，差距可能消失甚至反转。公平比较至少要控制数据划分、随机种子、增广、训练预算和模型选择规则，并为两种方法分别调参。

### 练习3：冻结输出层之前的参数会怎样

只训练分类头仍能得到有竞争力的结果，本次为0.8387，但低于完整微调的0.8600和更充分调参后的0.9187。原因是ImageNet特征已经能分辨大量视觉模式，线性分类器足以完成大部分工作；但热狗数据的拍摄风格和“非热狗”类别边界并不完全等于ImageNet，固定骨干限制了进一步适配。

Notebook提示代码若把`finetune_net.parameters()`全部设为`False`，也会连`fc`一起冻结，随后反向传播将没有可训练参数。正确做法是冻结骨干后重新解冻`fc`，并只把可训练参数交给优化器。冻结策略还应使用比完整微调更积极的分类头学习率；本次将头部学习率保持在$5\times10^{-4}$时只有0.7200，提高到$5\times10^{-3}$后才达到0.8387。

## 7. 怎样选择微调策略

| 条件 | 推荐起点 |
|---|---|
| 数据很少，目标域接近ImageNet | 冻结骨干，只训练分类头 |
| 数据中等，目标域较接近 | 冻结前部，训练后部和分类头 |
| 数据较多或域差异明显 | 小学习率完整微调 |
| 源类别与目标类别明确重合 | 尝试复用对应输出权重，再用验证集确认 |
| 训练不稳定 | 降低骨干学习率，使用warmup、调度器或逐步解冻 |

实践中可以从最保守的线性探测开始，再逐步开放模型容量：先只训头部，接着解冻最后一个stage，最后才完整微调。每一步都在相同验证协议下比较，才能判断增加的适配能力是否值得额外计算和过拟合风险。

## 8. 最后总结

微调的知识链可以压缩成：

```text
大规模源数据学习通用表示
        ↓
保留预训练骨干，替换任务输出层
        ↓
分类头大学习率 + 骨干小学习率
        ↓
按数据量和域差异决定冻结、部分解冻或完整微调
        ↓
若源目标类别重合，可进一步复用源输出权重
```

热狗例子最重要的结论不是某个固定准确率，而是三种先验可以逐层利用：预训练骨干提供通用视觉特征；冻结策略控制保留多少旧知识；ImageNet热狗输出权重还能提供类别级先验。先验越具体，越需要检查源任务与目标任务的语义是否真的一致，最终仍应让独立验证集决定是否采用。

## 对应资料

- [D2L：微调](https://zh.d2l.ai/chapter_computer-vision/fine-tuning.html)
- [PyTorch：Transfer Learning for Computer Vision Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Torchvision：ResNet-18与预训练权重](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [PyTorch：`nn.Module`训练与评估模式](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
