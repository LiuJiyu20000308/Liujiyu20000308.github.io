---
layout: post
title: "CNN 从卷积到 DenseNet：经典架构为何这样演进"
date: 2026-08-11 09:40 +0800
tags: [D2L, PyTorch, 深度学习, 神经网络, CNN, 计算机视觉]
toc: true
math: true
permalink: /deep-learning/cnn-evolution/
---

## 本篇逻辑主线

图像包含局部邻域和空间位置，直接展平后接全连接层既丢掉结构先验，又让参数量随像素数爆炸。卷积把局部连接与参数共享写进模型，通道用不同特征图描述同一空间，padding、stride 和 pooling 再控制分辨率与感受野。LeNet 首次把这些组件组合成可训练的识别系统；AlexNet 借助数据、GPU、ReLU、Dropout 和增强把 CNN 推到大规模视觉；VGG 用重复的 $3\times3$ 块把架构模块化；NiN 用 $1\times1$ 卷积和全局平均池化减少巨大全连接层；GoogLeNet/Inception 用并行分支同时处理多尺度，并用瓶颈控制成本；BatchNorm 让更深网络更易优化；ResNet 用恒等捷径解决深度退化与梯度通路；DenseNet 再把相加改成通道拼接，让每层直接复用之前的全部特征。整条演进不是“模型越来越复杂”，而是在表达能力、优化难度、参数量、计算量与显存之间不断重新分配预算。

### 怎样阅读下面的图和代码

每个经典架构都先用图回答“张量经过哪些 stage、在哪里改变高宽或通道、分支如何汇合”，再给出保留核心拓扑的 PyTorch 实现。代码统一输出 **logits**，不在模型末尾添加 softmax，因为训练时通常直接交给 `nn.CrossEntropyLoss`。除 LeNet 外，示例默认彩色输入 `[N,3,H,W]`；分类数通过 `num_classes` 调整。它们适合学习与 shape 验证，不等同于逐行复刻论文的训练细节或预训练权重配方。

后文代码共用以下导入：

```python
import torch
from torch import nn
from torch.nn import functional as F
```

## 1. 为什么图像不适合直接交给全连接层

若一张 RGB 图像为 $1000\times1000$，展平后有 300 万个输入。仅连接到 1000 个隐藏单元就需要约 30 亿个权重。更重要的是，全连接层把左上角像素与右下角像素当成和相邻像素同等普通的特征关系，没有利用图像的两个强先验：

1. **局部性（locality）**：边缘、纹理等低层模式主要由相邻像素决定。
2. **平移结构**：同一个检测器应能在不同位置发现同一种模式。

卷积用局部窗口解决第一点，用同一组核权重扫描所有位置解决第二点。严格说，深度学习框架里的“卷积”通常计算互相关，但核是从数据学习的，名称差异不影响模型表达。

对单通道输入，一个局部窗口可写成

$$
Y_{i,j}=b+\sum_{a=-\Delta_h}^{\Delta_h}\sum_{c=-\Delta_w}^{\Delta_w}
K_{a,c}X_{i+a,j+c}.
$$

把核索引改写为从 0 到 $K_h-1$、$K_w-1$ 只是坐标平移，不会漏掉窗口前半部分；$\Delta=1$ 对应覆盖偏移 $-1,0,1$，也就是大小 3。以水平边缘核 `[1, -1]` 为例，窗口落在两个相同像素上时输出为 0，跨过亮暗边缘才得到正或负响应；图中的“移一格后为 0”通常是局部差分抵消，不是平移操作凭空补了 0。

### 平移等变性不等于平移不变性

若输入平移，卷积特征图也相应平移，这叫平移等变（equivariance），不是输出完全不变。池化、全局汇聚、数据增强以及最终分类头会逐步降低对精确位置的敏感性，才得到近似的平移不变性。

## 2. 卷积核、神经元与输出通道

二维卷积层的输入、权重与输出 shape 是：

```text
input : [N, C_in,  H_in,  W_in]
weight: [C_out, C_in/groups, K_h, K_w]
bias  : [C_out]
output: [N, C_out, H_out, W_out]
```

在普通 `groups=1` 时，一个输出通道拥有一组形状 `[C_in, K_h, K_w]` 的核。它在每个位置同时观察全部输入通道，各通道互相关结果相加，再加一个偏置，得到该输出通道的一个像素。不同空间位置的“神经元”共享这同一组权重；所以卷积核不是一个神经元，而是一组被大量空间神经元复用的连接参数。

一个核组产生一个输出通道；想得到 $C_{out}$ 种特征响应，就学习 $C_{out}$ 个核组：

```python
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    padding=1,
)
X = torch.randn(32, 3, 224, 224)
Y = conv(X)  # [32, 64, 224, 224]
```

`nn.Conv2d` 的核心构造参数是 `in_channels`、`out_channels`、`kernel_size`、`stride`、`padding`、`dilation`、`groups`、`bias` 和 `padding_mode`。`groups` 改变输入输出通道的连接分组；`dilation` 拉开核元素的采样间距；二者都会影响权重或有效感受野，使用时应重新核对 shape。

每个输出通道通常不能简单命名为一个独立“边缘检测器”：通道是联合训练的分布式表示，但这个直觉有助于理解为什么深层网络随深度增加通道数。

### `Conv3d` 与 `Conv2d`

`Conv2d` 在两个空间轴滑动，常处理图像，输入 `[N,C,H,W]`。`Conv3d` 在深度、高度、宽度三个轴滑动，输入 `[N,C,D,H,W]`，权重 `[C_out,C_in,K_d,K_h,K_w]`，常用于视频片段或体数据。把 RGB 的 3 个颜色通道误当作 Conv3d 的“深度轴”会改变语义；普通彩色图像仍使用 `Conv2d(in_channels=3, ...)`。

## 3. Padding、stride 与输出尺寸

设输入高为 $H$、总 padding 为每侧 $P$、卷积核大小 $K$、stride 为 $S$、dilation 为 $D$，则

$$
H_{out}=\left\lfloor
\frac{H+2P-D(K-1)-1}{S}+1
\right\rfloor,
$$

宽度同理。D2L 常在 dilation=1 时写成等价的简化式。

- **padding** 在边界补值，常用零。奇数核 $K$、stride=1 时取 $P=(K-1)/2$，可保持高宽。
- **stride** 控制窗口移动间隔；stride=2 常把空间尺寸约减半，同时增大相邻输出中心在原图上的间距。

对一维音频，stride=2 表示卷积窗口每次跨过两个采样点，只保留一半左右的输出位置；它降低计算和时间分辨率，同时必须警惕混叠。二维图像上的收益相同：输出高宽下降后，后续层计算与激活显存都显著减少。

边缘像素在无 padding 时参与卷积的次数少，堆叠多层后还会快速丢失。padding 既保留尺寸，也让边缘信息获得更多计算机会；但补零本身是一种边界假设，不是凭空创造信息。

## 4. Pooling、全局平均池化与感受野

池化在每个通道独立滑动，不混合通道，也没有可学习卷积核：

- 最大池化保留窗口中最强响应，早期 CNN 常用来容忍小位移。
- 平均池化保留局部平均，更像低通和聚合。
- 全局平均池化（global average pooling, GAP）把每个通道的整个 $H\times W$ 平均为一个值：`[N,C,H,W] -> [N,C,1,1]`。

```python
gap = nn.AdaptiveAvgPool2d((1, 1))
Y = gap(X).flatten(1)  # [N, C]
```

现代网络也常直接使用 stride 卷积下采样，因此“CNN 必须有 pooling”是错误的。

某个单元的**感受野**是所有可能影响它的输入位置。连续堆叠小卷积会逐层扩大感受野：两个 $3\times3$ 卷积在 stride=1 时拥有约 $5\times5$ 的理论感受野，同时插入一次额外非线性，参数也比一个同通道的 $5\times5$ 卷积少。

### $1\times1$ 卷积为什么有用

$1\times1$ 卷积不看空间邻居，而是在每个像素位置独立进行通道线性组合：

$$
\mathbb R^{C_{in}}\rightarrow\mathbb R^{C_{out}}.
$$

跨位置共享同一权重，所以空间结构保留。它可以改变通道数、增加逐像素非线性、构造瓶颈并控制昂贵卷积的输入宽度，是 NiN、Inception、ResNet 和 DenseNet 的共同语言。

## 5. LeNet：把卷积组件组成完整识别器

### 前一阶段的问题

softmax 回归和 MLP 处理 Fashion-MNIST 时先把 $28\times28$ 图像展平，空间邻域被抹掉。LeNet 的突破是保留二维结构，并用卷积逐步提取局部模式。

### 核心结构与实现

D2L 的 LeNet 由两个“卷积 + sigmoid + 平均池化”阶段和三个全连接层构成：

<figure class="network-figure network-figure-wide">
  <img src="{{ '/assets/deep-learning/cnn/lenet.svg' | relative_url }}" alt="LeNet 从输入图像经过两组卷积与池化，再进入三个全连接层的完整数据流">
  <figcaption>LeNet 数据流。卷积部分逐步降低空间分辨率并增加通道，全连接部分把最终特征映射为 10 类输出。图源：D2L《卷积神经网络》Notebook。</figcaption>
</figure>

```python
lenet = nn.Sequential(
    nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(6, 16, 5), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10),
)

X = torch.randn(4, 1, 28, 28)
logits = lenet(X)  # [4, 10]
```

shape 链为 `[N,1,28,28] -> [N,6,28,28] -> [N,6,14,14] -> [N,16,10,10] -> [N,16,5,5] -> [N,10]`。

### 收益、局限与下一步

LeNet 证明局部连接、共享参数和下采样能形成端到端图像识别器，参数远少于对原图直接全连接。但它面向小型灰度图、层数浅，使用易饱和 sigmoid 和平均池化，无法直接承担 ImageNet 规模的多类别自然图像。更大数据、GPU 和新的训练技巧到位后，AlexNet 才能把同一基本范式放大。

LeNet 训练时 batch size 过大还可能表现为“按 epoch 收敛变慢”：数据量为 $N$ 时每轮只有约 $N/B$ 次参数更新，增大 $B$ 却保持 epoch 和学习率不变，会显著减少更新次数。大 batch 梯度更平稳、GPU 吞吐可能更高，但常需配合学习率缩放与 warmup；应区分按 epoch、按更新步数和按墙钟时间三种快慢。

## 6. AlexNet：让深度 CNN 在大规模视觉上成立

### 为什么出现

LeNet 之后很长时间，手工特征加传统分类器仍占主流。缺的并不只是“多几层”：还缺大规模标注数据、GPU 并行卷积、可训练的激活与正则化策略。

### 核心设计

AlexNet 有 5 个卷积层和 3 个全连接层，论文计为 8 个有参数层；前面使用较大的 $11\times11$、$5\times5$ 核，之后使用连续 $3\times3$ 卷积，通道远多于 LeNet。它用 ReLU 替代 sigmoid，用最大池化、Dropout 和强数据增强，并在 GPU 上训练。

<figure class="network-figure">
  <img src="{{ '/assets/deep-learning/cnn/alexnet.svg' | relative_url }}" alt="LeNet 与 AlexNet 的架构对比，AlexNet 更深、更宽并使用 ReLU、最大池化和 Dropout">
  <figcaption>从 LeNet 到 AlexNet：基本的“卷积提特征—池化降采样—全连接分类”没有改变，但深度、通道数、激活函数和训练规模都被显著放大。图源：D2L《现代卷积神经网络》Notebook。</figcaption>
</figure>

```python
def alexnet(num_classes=10, in_channels=3):
    return nn.Sequential(
        # 输入 [N,3,224,224]
        nn.Conv2d(in_channels, 96, 11, stride=4, padding=2), nn.ReLU(),
        nn.MaxPool2d(3, stride=2),                         # [N,96,27,27]
        nn.Conv2d(96, 256, 5, padding=2), nn.ReLU(),
        nn.MaxPool2d(3, stride=2),                         # [N,256,13,13]
        nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(3, stride=2),                         # [N,256,6,6]
        nn.Flatten(),
        nn.Linear(256 * 6 * 6, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, num_classes),
    )

net = alexnet()
logits = net(torch.randn(2, 3, 224, 224))  # [2, 10]
```

这段代码采用单 GPU、单分支的现代 PyTorch 写法，省略原论文因当时双 GPU 显存限制而使用的分组细节；它保留 5 个卷积层、3 个全连接层与 Dropout 的核心拓扑。

### 参数主要在哪里

AlexNet 的卷积计算量很大，但参数主要集中在最后的全连接层。D2L 简化版展平后为 6400 维，仅 `6400 -> 4096` 就约 2621 万个权重，再加 `4096 -> 4096` 约 1678 万。全连接头因此同时带来存储、过拟合和输入尺寸固定问题。

### 收益、代价与下一步

AlexNet 证明学习到的分层特征能显著超越手工视觉流水线，ReLU、Dropout、增强和 GPU 成为新范式。代价是结构中核大小、通道数和层安排较经验化，巨大全连接层参数昂贵。VGG 接下来要回答：能否用一个重复、可扩展的模块取代这种“每层都单独设计”的方式？

## 7. VGG：用重复的小卷积块建立设计模板

### 从 AlexNet 到 VGG 块

VGG 把“若干个保持分辨率的 $3\times3$ 卷积 + ReLU，再接 $2\times2$ 最大池化”定义成可重复块：

<figure class="network-figure">
  <img src="{{ '/assets/deep-learning/cnn/vgg.svg' | relative_url }}" alt="AlexNet 与 VGG 的块结构对比，VGG 使用重复的三乘三卷积和二乘二最大池化">
  <figcaption>VGG 把 AlexNet 中不规则的核和层安排改造成可重复的块：同一 stage 保持高宽，池化时高宽减半、通道增加。图源：D2L《现代卷积神经网络》Notebook。</figcaption>
</figure>

```python
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers += [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU()]
        in_channels = out_channels
    layers += [nn.MaxPool2d(2, stride=2)]
    return nn.Sequential(*layers)

def vgg16(num_classes=10, in_channels=3):
    # 2+2+3+3+3 = 13 个卷积层
    architecture = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
    features = []
    cin = in_channels
    for num_convs, cout in architecture:
        features.append(vgg_block(num_convs, cin, cout))
        cin = cout

    return nn.Sequential(
        *features,                                      # [N,512,7,7]
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, num_classes),
    )

net = vgg16()
logits = net(torch.randn(2, 3, 224, 224))  # [2, 10]
```

连续小核在相近感受野下加入更多非线性，结构规则、容易加深。空间尺寸每个块减半，通道通常逐块增加。

### VGG-16 的“16”是什么

VGG-16 指 13 个卷积层加 3 个全连接层，共 16 个有可训练参数的层；ReLU 和池化不计入这个数字。它不是“16 个卷积层”。

### 收益、代价与下一步

VGG 奠定了“用块设计网络”的工程范式，结构清楚，特征也长期被用于迁移学习。但它仍保留昂贵全连接头，计算与显存消耗大。更关键的是，一旦过早展平或使用普通全连接层，二维邻域关系就不再显式存在，后续无法继续用卷积共享空间结构。NiN 因而尝试把“全连接的表达力”留在每个像素位置，并取消巨大全连接分类头。

## 8. NiN：在每个像素上做小型网络

### 核心思想

Network in Network（NiN）用普通卷积提取邻域信息，再接多个 $1\times1$ 卷积，在每个空间位置上完成通道 MLP：

<figure class="network-figure">
  <img src="{{ '/assets/deep-learning/cnn/nin.svg' | relative_url }}" alt="VGG、AlexNet 与 NiN 的架构对比，NiN 使用一乘一卷积组成局部通道网络并以全局平均池化分类">
  <figcaption>NiN 与 AlexNet/VGG 的关键差异：中间不急于展平，使用 $1\times1$ 卷积在每个空间位置混合通道；末端用全局平均池化替代巨大全连接头。图源：D2L《现代卷积神经网络》Notebook。</figcaption>
</figure>

```python
def nin_block(cin, cout, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, padding), nn.ReLU(),
        nn.Conv2d(cout, cout, 1), nn.ReLU(),
        nn.Conv2d(cout, cout, 1), nn.ReLU(),
    )

def nin(num_classes=10, in_channels=3):
    return nn.Sequential(
        nin_block(in_channels, 96, 11, 4, 0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, 5, 1, 2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, 3, 1, 1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        # 最后一个块直接把通道变成类别数
        nin_block(384, num_classes, 3, 1, 1),
        nn.AdaptiveAvgPool2d((1, 1)),  # [N,K,H,W] -> [N,K,1,1]
        nn.Flatten(),                  # [N,K]
    )

net = nin()
logits = net(torch.randn(2, 3, 224, 224))  # [2, 10]
```

最后一个块把通道数变成类别数，GAP 把每个类别特征图汇聚成对应 logit。这样既保留空间组织到网络末端，又显著减少全连接参数。

### 收益、代价与下一步

NiN 增加逐像素非线性，减少参数和全连接层过拟合，并推广了 $1\times1$ 卷积与 GAP。局限是它仍串行选择单一尺度的核，某层到底该看 $1\times1$、$3\times3$ 还是 $5\times5$ 仍要人工决定；参数少也不必然训练更快。GoogLeNet 将不同尺度同时放入并行分支。

## 9. GoogLeNet / Inception：让网络自己组合尺度

### 为什么并行分支

小核擅长局部细节，大核拥有更大感受野，池化提供不同的汇聚归纳偏置。Inception V1 不强迫整层选择唯一尺度，而是并行计算：

1. $1\times1$ 卷积分支；
2. $1\times1 \rightarrow 3\times3$；
3. $1\times1 \rightarrow 5\times5$；
4. $3\times3$ 最大池化 $\rightarrow 1\times1$。

各分支保持相同高宽，最后沿通道维拼接：

<figure class="network-figure">
  <img src="{{ '/assets/deep-learning/cnn/inception-v1-block.svg' | relative_url }}" alt="Inception V1 模块的四条并行路径，包括一乘一卷积、三乘三卷积、五乘五卷积和池化路径">
  <figcaption>Inception V1 基本块。四条路径必须产生相同的 $H\times W$，才能在通道维 `dim=1` 拼接；$1\times1$ 瓶颈负责控制后续大卷积的输入通道。图源：D2L《现代卷积神经网络》Notebook。</figcaption>
</figure>

大卷积前的 $1\times1$ 是瓶颈：先压缩通道，再做昂贵的空间卷积，显著减少参数和计算。多尺度的意义不是简单“核越多越好”，而是让同一层同时获得不同感受野的证据，再由后续层联合使用。

GoogLeNet V1 堆叠 9 个 Inception 块，中间用最大池化下采样，末端使用 GAP。原论文还设置辅助分类器帮助深层训练；D2L 为突出主干省略了它们。

<figure class="network-figure network-figure-tall">
  <img src="{{ '/assets/deep-learning/cnn/googlenet-v1.svg' | relative_url }}" alt="GoogLeNet V1 从卷积 stem、九个 Inception 模块到全局平均池化的完整架构">
  <figcaption>GoogLeNet V1 主干：Inception 模块按 2、5、2 分成三个 stage，stage 间池化下采样，末端使用全局平均池化。图源：D2L《现代卷积神经网络》Notebook。</figcaption>
</figure>

下面的实现保留 D2L 主干的通道配置，省略两个辅助分类器：

```python
class InceptionV1Block(nn.Module):
    def __init__(self, cin, c1, c2, c3, c4):
        super().__init__()
        self.p1 = nn.Conv2d(cin, c1, 1)
        self.p2 = nn.Sequential(
            nn.Conv2d(cin, c2[0], 1), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], 3, padding=1),
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(cin, c3[0], 1), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], 5, padding=2),
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(cin, c4, 1),
        )

    def forward(self, X):
        paths = [self.p1(X), self.p2(X), self.p3(X), self.p4(X)]
        return F.relu(torch.cat(paths, dim=1))

def googlenet_v1(num_classes=10, in_channels=3):
    stem1 = nn.Sequential(
        nn.Conv2d(in_channels, 64, 7, stride=2, padding=3), nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1),
    )
    stem2 = nn.Sequential(
        nn.Conv2d(64, 64, 1), nn.ReLU(),
        nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1),
    )
    stage3 = nn.Sequential(
        InceptionV1Block(192, 64, (96, 128), (16, 32), 32),   # -> 256
        InceptionV1Block(256, 128, (128, 192), (32, 96), 64), # -> 480
        nn.MaxPool2d(3, stride=2, padding=1),
    )
    stage4 = nn.Sequential(
        InceptionV1Block(480, 192, (96, 208), (16, 48), 64),
        InceptionV1Block(512, 160, (112, 224), (24, 64), 64),
        InceptionV1Block(512, 128, (128, 256), (24, 64), 64),
        InceptionV1Block(512, 112, (144, 288), (32, 64), 64),
        InceptionV1Block(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(3, stride=2, padding=1),
    )
    stage5 = nn.Sequential(
        InceptionV1Block(832, 256, (160, 320), (32, 128), 128),
        InceptionV1Block(832, 384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
    )
    return nn.Sequential(stem1, stem2, stage3, stage4, stage5,
                         nn.Linear(1024, num_classes))

net = googlenet_v1()
logits = net(torch.randn(2, 3, 224, 224))  # [2, 10]
```

### Inception V1 到 V3

V3 延续多分支思想，但更系统地提高计算效率和训练质量：用两个 $3\times3$ 代替昂贵 $5\times5$，把 $n\times n$ 分解成 $1\times n$ 与 $n\times1$，设计更高效的网格下采样，并把 **Conv → BatchNorm → ReLU** 作为基础卷积单元；训练还可使用辅助分类器和标签平滑。核心变化不是“分支更多”，而是把大卷积分解并谨慎安排降采样，使计算预算用在更丰富的表示上。

<figure class="network-figure">
  <img src="{{ '/assets/deep-learning/cnn/inception-v3.svg' | relative_url }}" alt="Inception V3 从二百九十九像素输入、Stem、Inception A B C 和 Reduction 模块到分类头的主干架构">
  <figcaption>Inception V3 主干。TorchVision 的命名对应 `Mixed_5b~5d`（A×3）、`Mixed_6a`（Reduction-A）、`Mixed_6b~6e`（B×4）、`Mixed_7a`（Reduction-B）和 `Mixed_7b~7c`（C×2）。</figcaption>
</figure>

<figure class="network-figure">
  <img src="{{ '/assets/deep-learning/cnn/inception-v3-factorization.svg' | relative_url }}" alt="Inception V3 把五乘五拆为两个三乘三，并把 n 乘 n 拆为一乘 n 和 n 乘一的卷积分解示意">
  <figcaption>V3 的卷积分解：减少参数和乘加次数，同时插入更多非线性。后期分支再拆成 $1\times3$ 与 $3\times1$，让横向和纵向证据并行保留。</figcaption>
</figure>

一个 V3 基础卷积和 Inception-A 块可以写成：

```python
class BasicConv2d(nn.Sequential):
    def __init__(self, cin, cout, **conv_kwargs):
        super().__init__(
            nn.Conv2d(cin, cout, bias=False, **conv_kwargs),
            nn.BatchNorm2d(cout, eps=0.001),
            nn.ReLU(inplace=True),
        )

class InceptionA(nn.Module):
    def __init__(self, cin, pool_features):
        super().__init__()
        self.b1 = BasicConv2d(cin, 64, kernel_size=1)
        self.b2 = nn.Sequential(
            BasicConv2d(cin, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5, padding=2),
        )
        self.b3 = nn.Sequential(
            BasicConv2d(cin, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1),
        )
        self.pool_proj = BasicConv2d(cin, pool_features, kernel_size=1)

    def forward(self, X):
        p1 = self.b1(X)
        p2 = self.b2(X)
        p3 = self.b3(X)
        p4 = self.pool_proj(F.avg_pool2d(X, 3, stride=1, padding=1))
        return torch.cat([p1, p2, p3, p4], dim=1)
```

完整 V3 包含多种 A/B/C 与 Reduction 块，工程中应直接使用 TorchVision 经过测试的实现，避免手抄数百行通道配置：

```python
from torchvision.models import inception_v3, Inception_V3_Weights

# 从头训练；V3 的标准输入为 [N,3,299,299]
net = inception_v3(weights=None, num_classes=10, aux_logits=False,
                   init_weights=False)
logits = net(torch.randn(2, 3, 299, 299))  # [2, 10]

# 迁移学习时让权重对象提供匹配的预处理规则
weights = Inception_V3_Weights.DEFAULT
pretrained = inception_v3(weights=weights)
preprocess = weights.transforms()
```

这里的 Inception V3 仍属于 Inception 家族；它使用 BatchNorm，但**不包含 ResNet 的残差相加**。ResNet 是后面独立演进出的结构路线；不要把它和后来出现的 Inception-ResNet 混为一谈。

### 收益、代价与下一步

Inception 以较低参数量获得强多尺度能力，但分支和通道配置复杂，修改成本高，而且网络更深后仍面临激活尺度和优化困难。BatchNorm 先解决训练条件问题；随后 ResNet 用更简单的残差块建立直接跨层通路。

## 10. BatchNorm：把中间激活放回可训练尺度

<figure class="network-figure network-figure-wide">
  <img src="{{ '/assets/deep-learning/cnn/batchnorm-flow.svg' | relative_url }}" alt="卷积输出经过 BatchNorm 标准化、可学习缩放平移和 ReLU 的数据流，并区分训练与推理统计量">
  <figcaption>BatchNorm 数据流。训练与推理的差异发生在统计量来源；$gamma$、$eta$ 在两种模式下都是同一组可学习参数。</figcaption>
</figure>

对卷积输出的每个通道，训练时在 batch 与空间位置 $(N,H,W)$ 上计算均值和方差：

$$
\mu_\mathcal{B}=\frac1m\sum_i x_i,
\qquad
\sigma_\mathcal{B}^2=\frac1m\sum_i(x_i-\mu_\mathcal{B})^2,
$$

$$
y=\gamma\frac{x-\mu_\mathcal{B}}
{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}+\beta.
$$

$\gamma$ 和 $\beta$ 是每通道可学习缩放与平移，允许网络恢复合适的分布，而不是永久强制零均值单位方差。

```python
class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=1):
        super().__init__(
            # BN 已有可学习 beta，卷积 bias 通常可以省略
            nn.Conv2d(cin, cout, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

block = ConvBNReLU(64, 128)
X = torch.randn(8, 64, 28, 28)

block.train()
Y_train = block(X)  # 使用当前 batch 统计并更新 running stats

block.eval()
with torch.no_grad():
    Y_eval = block(X[:1])  # 使用 running_mean / running_var
```

训练时使用当前 batch 统计并更新 running mean/variance；推理时使用运行统计，保证单样本输出确定。因此 `train()` 与 `eval()` 的区别会直接改变结果，小 batch 也可能让估计不稳定。BatchNorm 常允许更大学习率、改善优化条件，并带来一定正则化噪声，但它不是 Dropout 的等价替代，也不应把“内部协变量偏移”当作已经完全证实的唯一解释。

BatchNorm 让深网更好训，却没有改变“新增层能否至少表示旧模型”的结构问题。ResNet 从函数类和梯度通路直接处理这个问题。

## 11. ResNet：学习残差，而不是重学恒等映射

### 深度退化与残差思想

更深模型理论容量更大，但普通堆叠网络可能连训练误差都变差，这不是典型过拟合，而是优化退化。若新块至少能实现恒等映射，扩大后的函数类才容易包含旧模型。

残差块输出：

$$
\mathbf y=F(\mathbf x)+\mathbf x.
$$

当理想变换接近恒等时，只需把 $F$ 学到接近 0。反向传播也得到一条不必穿过所有权重层的恒等通路。

<figure class="network-figure">
  <img src="{{ '/assets/deep-learning/cnn/residual-block.svg' | relative_url }}" alt="普通卷积块与残差块对比，残差块增加从输入直接到加法节点的捷径连接">
  <figcaption>普通块与残差块。右侧 shortcut 绕过权重层，在加法节点与 $F(X)$ 融合；若 shape 改变，shortcut 需要 $1\times1$ 投影。图源：D2L《现代卷积神经网络》Notebook。</figcaption>
</figure>

```python
class Residual(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.skip = (
            nn.Identity() if cin == cout and stride == 1
            else nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )
        )

    def forward(self, X):
        return torch.relu(self.main(X) + self.skip(X))

def resnet_stage(cin, cout, blocks, first_stage=False):
    layers = []
    for i in range(blocks):
        stride = 1 if first_stage or i > 0 else 2
        layers.append(Residual(cin if i == 0 else cout, cout, stride))
    return nn.Sequential(*layers)

def resnet18(num_classes=10, in_channels=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1),
        resnet_stage(64, 64, 2, first_stage=True),
        resnet_stage(64, 128, 2),
        resnet_stage(128, 256, 2),
        resnet_stage(256, 512, 2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(), nn.Linear(512, num_classes),
    )

net = resnet18()
logits = net(torch.randn(2, 3, 224, 224))  # [2, 10]
```

<figure class="network-figure network-figure-tall">
  <img src="{{ '/assets/deep-learning/cnn/resnet18.svg' | relative_url }}" alt="ResNet 十八从输入卷积、四个残差 stage、全局平均池化到全连接输出的架构">
  <figcaption>ResNet-18：四个 stage 各有 2 个残差块；除第一个 stage 外，每个 stage 的首块用 stride=2 下采样并增加通道。图源：D2L《现代卷积神经网络》Notebook。</figcaption>
</figure>

### ResNet 是否需要 pooling，怎样下采样

经典 ResNet 开头有 stride=2 的 $7\times7$ 卷积和 stride=2 的最大池化；进入后续 stage 时，首个残差块的主支和 $1\times1$ shortcut 都用 stride=2，使高宽减半、通道增加；末端用 GAP。也就是说，它有 pooling，但不靠每个残差块后的 pooling 下采样。较小图像版本还常去掉开头最大池化。

### 残差连接是否保证性能不变

结构上，残差参数化让恒等映射更容易包含在函数类中；这不等于实际 SGD 一定找到不差的解，也不保证有限验证集上的性能不下降。初始化、正则化、数据和优化超参数仍会影响结果。shortcut 帮助梯度传播也不是“梯度永远为 1”：总梯度还包含残差分支、后续层与激活的作用。

### 收益、代价与下一步

ResNet 用简单、可堆叠的块训练百层网络，结构比 Inception 更容易扩展，残差思想也进入几乎所有深层架构。相加要求两支 shape 一致，且把新旧特征融合在同一通道表示中。DenseNet 进一步问：如果不相加压缩，而是把所有旧特征原样保留给后续层，会怎样？

## 12. DenseNet：所有层直接复用之前的特征

### 稠密连接

DenseNet 第 $l$ 层接收所有先前特征图的通道拼接：

<figure class="network-figure">
  <img src="{{ '/assets/deep-learning/cnn/densenet-connections.svg' | relative_url }}" alt="ResNet 使用逐元素相加，而 DenseNet 使用通道拼接的跨层连接对比">
  <figcaption>ResNet 与 DenseNet 的连接差异：ResNet 用逐元素相加融合两条路径；DenseNet 沿通道维拼接，因此旧特征仍可被后续层单独访问。图源：D2L《现代卷积神经网络》Notebook。</figcaption>
</figure>

$$
\mathbf x_l=H_l([\mathbf x_0,\mathbf x_1,\ldots,\mathbf x_{l-1}]).
$$

与 ResNet 的逐元素相加不同，`torch.cat(..., dim=1)` 保留每一批特征的独立通道。若每层新增 $k$ 个通道，$k$ 称为增长率（growth rate）：

```python
class DenseBlock(nn.Module):
    def __init__(self, layers, cin, growth):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(cin + i * growth), nn.ReLU(),
                nn.Conv2d(cin + i * growth, growth, 3, padding=1),
            )
            for i in range(layers)
        ])

    def forward(self, X):
        for block in self.blocks:
            X = torch.cat([X, block(X)], dim=1)
        return X
```

<figure class="network-figure">
  <img src="{{ '/assets/deep-learning/cnn/densenet-block.svg' | relative_url }}" alt="DenseNet 稠密块中每一层都接收此前所有层的特征图">
  <figcaption>稠密块：第 $l$ 层的输入通道数会随此前层数线性增长，每层只新增 `growth_rate` 个通道。图源：D2L《现代卷积神经网络》Notebook。</figcaption>
</figure>

### 为什么参数可能更少

每层只需新增较窄的 `growth` 个特征，旧特征无需被后层反复重新学习；在合适增长率和 bottleneck 设计下，DenseNet 可以用较窄层实现强特征复用，因此参数可能少于同等精度的宽 ResNet。注意“可能”不是无条件：层数、增长率和压缩率仍决定总参数与计算。

### Transition layer 为什么用平均池化

稠密块后通道持续增加，transition layer 用 $1\times1$ 卷积压缩通道，再用 stride=2 的平均池化减半高宽：

```python
def transition(cin, cout):
    return nn.Sequential(
        nn.BatchNorm2d(cin), nn.ReLU(),
        nn.Conv2d(cin, cout, 1),
        nn.AvgPool2d(2, stride=2),
    )

def densenet_small(num_classes=10, in_channels=3,
                   growth=32, layers_per_block=(4, 4, 4, 4)):
    stem = nn.Sequential(
        nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1),
    )

    channels = 64
    body = []
    for i, num_layers in enumerate(layers_per_block):
        body.append(DenseBlock(num_layers, channels, growth))
        channels += num_layers * growth
        if i != len(layers_per_block) - 1:
            body.append(transition(channels, channels // 2))
            channels //= 2

    return nn.Sequential(
        stem, *body,
        nn.BatchNorm2d(channels), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(), nn.Linear(channels, num_classes),
    )

net = densenet_small()
logits = net(torch.randn(2, 3, 224, 224))  # [2, 10]
```

平均池化平滑汇总所有响应，与稠密复用“尽量保留信息”的思路一致；最大池化只保留局部最大值，更具选择性。它不是数学上唯一可行的选择，而是原架构的有效设计取舍。

### 为什么显存更大

参数较少不等于显存更少。训练反向传播必须保存很多中间激活；每层都读取越来越长的通道拼接，`cat` 还会产生大特征图和内存访问。朴素实现因此常比 ResNet 更耗显存，尤其在高分辨率阶段。checkpointing、内存高效实现和 bottleneck/压缩可以缓解，但通常以额外计算或实现复杂度为代价。

## 13. 模型演进对比

| 模型 | 要解决的问题 | 核心设计 | 主要收益 | 代价或局限 |
|---|---|---|---|---|
| LeNet | 展平图像丢空间结构 | 卷积、激活、池化、全连接组成端到端 CNN | 少参数地学习局部特征 | 浅、面向小图，sigmoid 易饱和 |
| AlexNet | CNN 尚未在大规模视觉成立 | 更深更宽、ReLU、GPU、Dropout、数据增强 | 证明分层学习特征胜过手工特征 | 全连接参数巨大，结构经验化 |
| VGG | 缺少可复用的深网模板 | 重复 $3\times3$ 卷积块 | 规则、易扩展，深小核有效 | 计算、显存和全连接参数昂贵 |
| NiN | 全连接头丢空间且参数多 | $1\times1$ 通道 MLP + GAP | 保留空间到末端，显著减参 | 串行单尺度，未必训练更快 |
| GoogLeNet / Inception V1 | 单一卷积尺度难选择 | 四路多尺度分支、$1\times1$ 瓶颈、GAP | 参数效率高，多尺度表示强 | 分支和通道配置复杂 |
| Inception V3 | 大核和粗糙下采样的计算成本高 | 卷积分解、A/B/C 模块、Reduction、BN | 更有效地使用计算预算，训练更稳定 | 结构更复杂；标准输入为 $299\times299$ |
| BatchNorm | 深层激活尺度和优化困难 | batch 标准化 + 可学习 $\gamma,\beta$ | 更稳定、更快的深层训练 | 训练/推理行为不同，依赖 batch 统计 |
| ResNet | 深度增加引起优化退化 | $F(x)+x$ 恒等 shortcut | 梯度通路直接，块简单可堆叠 | shape 对齐要求；不保证实际性能不降 |
| DenseNet | 相加会融合旧新特征 | 所有旧特征沿通道稠密拼接 | 特征复用强，参数可更少 | 激活与拼接导致显存、带宽成本高 |

## 本篇知识链总结

卷积以局部连接和参数共享利用图像结构；padding、stride、pooling 与通道共同控制 shape、感受野和计算预算。LeNet 建立基本流水线，AlexNet 证明规模化有效，VGG 把深网变成重复块，NiN 去掉巨大全连接头，Inception 并行组合尺度，BatchNorm 改善深层优化，ResNet 提供恒等捷径，DenseNet 用拼接实现极致特征复用。理解每一步解决的旧问题，比记住每个网络的层数更重要。

## 常见误区

- 把“卷积平移等变”直接说成输出天然平移不变。
- 认为一个卷积核只连接一个输入通道，或把 RGB 当作 Conv3d 的深度轴。
- 忽略 PyTorch `Conv2d` 权重 shape 是 `[C_out,C_in,K_h,K_w]`。
- 认为 padding 会恢复已经不存在的边缘信息。
- 认为 pooling 会混合通道，或现代 CNN 每个 stage 都必须池化。
- 把 VGG-16 理解成 16 个卷积层。
- 认为 $1\times1$ 卷积没有用，因为它不看邻居。
- 认为 BatchNorm 推理时仍使用当前样本均值方差。
- 认为残差连接从理论上保证任何训练结果都不差。
- 因 DenseNet 参数较少，就推断其训练显存一定较少。

## 系列导航

- 上一篇：[PyTorch 深度学习计算与工程实践]({{ '/deep-learning/pytorch-engineering/' | relative_url }})
- [系列总览]({{ '/deep-learning/' | relative_url }})
- 下一篇：[D2L 计算性能：从硬件、异步执行到多机 DDP]({{ '/deep-learning/computational-performance/' | relative_url }})

## 对应章节与参考资料

- D2L：[卷积神经网络](https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html)、[现代卷积神经网络](https://zh.d2l.ai/chapter_convolutional-modern/index.html)。文中的 LeNet、AlexNet、VGG、NiN、Inception V1、ResNet、DenseNet 图来自对应 D2L Notebook；BatchNorm 与 Inception V3 总图为本文按相同知识链重绘。
- PyTorch：[`Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)、[`Conv3d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)、[`AdaptiveAvgPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html)、[`torchvision.models.inception_v3`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.inception_v3.html)
- 原始论文：[AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)、[VGG](https://arxiv.org/abs/1409.1556)、[Network in Network](https://arxiv.org/abs/1312.4400)、[GoogLeNet / Inception V1](https://arxiv.org/abs/1409.4842)、[Inception V3](https://arxiv.org/abs/1512.00567)、[Batch Normalization](https://arxiv.org/abs/1502.03167)、[ResNet](https://arxiv.org/abs/1512.03385)、[DenseNet](https://arxiv.org/abs/1608.06993)
