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

```python
# 简化开头
nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
nn.ReLU()
nn.MaxPool2d(3, stride=2)
```

### 参数主要在哪里

AlexNet 的卷积计算量很大，但参数主要集中在最后的全连接层。D2L 简化版展平后为 6400 维，仅 `6400 -> 4096` 就约 2621 万个权重，再加 `4096 -> 4096` 约 1678 万。全连接头因此同时带来存储、过拟合和输入尺寸固定问题。

### 收益、代价与下一步

AlexNet 证明学习到的分层特征能显著超越手工视觉流水线，ReLU、Dropout、增强和 GPU 成为新范式。代价是结构中核大小、通道数和层安排较经验化，巨大全连接层参数昂贵。VGG 接下来要回答：能否用一个重复、可扩展的模块取代这种“每层都单独设计”的方式？

## 7. VGG：用重复的小卷积块建立设计模板

### 从 AlexNet 到 VGG 块

VGG 把“若干个保持分辨率的 $3\times3$ 卷积 + ReLU，再接 $2\times2$ 最大池化”定义成可重复块：

```python
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers += [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU()]
        in_channels = out_channels
    layers += [nn.MaxPool2d(2, stride=2)]
    return nn.Sequential(*layers)
```

连续小核在相近感受野下加入更多非线性，结构规则、容易加深。空间尺寸每个块减半，通道通常逐块增加。

### VGG-16 的“16”是什么

VGG-16 指 13 个卷积层加 3 个全连接层，共 16 个有可训练参数的层；ReLU 和池化不计入这个数字。它不是“16 个卷积层”。

### 收益、代价与下一步

VGG 奠定了“用块设计网络”的工程范式，结构清楚，特征也长期被用于迁移学习。但它仍保留昂贵全连接头，计算与显存消耗大。更关键的是，一旦过早展平或使用普通全连接层，二维邻域关系就不再显式存在，后续无法继续用卷积共享空间结构。NiN 因而尝试把“全连接的表达力”留在每个像素位置，并取消巨大全连接分类头。

## 8. NiN：在每个像素上做小型网络

### 核心思想

Network in Network（NiN）用普通卷积提取邻域信息，再接多个 $1\times1$ 卷积，在每个空间位置上完成通道 MLP：

```python
def nin_block(cin, cout, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, padding), nn.ReLU(),
        nn.Conv2d(cout, cout, 1), nn.ReLU(),
        nn.Conv2d(cout, cout, 1), nn.ReLU(),
    )
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

```python
return torch.cat([p1, p2, p3, p4], dim=1)
```

大卷积前的 $1\times1$ 是瓶颈：先压缩通道，再做昂贵的空间卷积，显著减少参数和计算。多尺度的意义不是简单“核越多越好”，而是让同一层同时获得不同感受野的证据，再由后续层联合使用。

GoogLeNet V1 堆叠 9 个 Inception 块，中间用最大池化下采样，末端使用 GAP。原论文还设置辅助分类器帮助深层训练；D2L 为突出主干省略了它们。

### Inception V1 到 V3

V3 延续多分支思想，但更系统地提高计算效率和训练质量：用两个 $3\times3$ 代替昂贵 $5\times5$，把 $n\times n$ 分解成 $1\times n$ 与 $n\times1$，设计更高效的网格下采样，并配合 BatchNorm、辅助分类器和标签平滑等训练策略。核心变化不是“分支更多”，而是把大卷积分解并谨慎安排降采样，使计算预算用在更丰富的表示上。

### 收益、代价与下一步

Inception 以较低参数量获得强多尺度能力，但分支和通道配置复杂，修改成本高，而且网络更深后仍面临激活尺度和优化困难。BatchNorm 先解决训练条件问题；随后 ResNet 用更简单的残差块建立直接跨层通路。

## 10. BatchNorm：把中间激活放回可训练尺度

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
block = nn.Sequential(
    nn.Conv2d(cin, cout, 3, padding=1, bias=False),
    nn.BatchNorm2d(cout),
    nn.ReLU(),
)
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
            else nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
        )

    def forward(self, X):
        return torch.relu(self.main(X) + self.skip(X))
```

### ResNet 是否需要 pooling，怎样下采样

经典 ResNet 开头有 stride=2 的 $7\times7$ 卷积和 stride=2 的最大池化；进入后续 stage 时，首个残差块的主支和 $1\times1$ shortcut 都用 stride=2，使高宽减半、通道增加；末端用 GAP。也就是说，它有 pooling，但不靠每个残差块后的 pooling 下采样。较小图像版本还常去掉开头最大池化。

### 残差连接是否保证性能不变

结构上，残差参数化让恒等映射更容易包含在函数类中；这不等于实际 SGD 一定找到不差的解，也不保证有限验证集上的性能不下降。初始化、正则化、数据和优化超参数仍会影响结果。shortcut 帮助梯度传播也不是“梯度永远为 1”：总梯度还包含残差分支、后续层与激活的作用。

### 收益、代价与下一步

ResNet 用简单、可堆叠的块训练百层网络，结构比 Inception 更容易扩展，残差思想也进入几乎所有深层架构。相加要求两支 shape 一致，且把新旧特征融合在同一通道表示中。DenseNet 进一步问：如果不相加压缩，而是把所有旧特征原样保留给后续层，会怎样？

## 12. DenseNet：所有层直接复用之前的特征

### 稠密连接

DenseNet 第 $l$ 层接收所有先前特征图的通道拼接：

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

### 为什么参数可能更少

每层只需新增较窄的 `growth` 个特征，旧特征无需被后层反复重新学习；在合适增长率和 bottleneck 设计下，DenseNet 可以用较窄层实现强特征复用，因此参数可能少于同等精度的宽 ResNet。注意“可能”不是无条件：层数、增长率和压缩率仍决定总参数与计算。

### Transition layer 为什么用平均池化

稠密块后通道持续增加，transition layer 用 $1\times1$ 卷积压缩通道，再用 stride=2 的平均池化减半高宽：

```python
nn.Sequential(
    nn.BatchNorm2d(cin), nn.ReLU(),
    nn.Conv2d(cin, cout, 1),
    nn.AvgPool2d(2, stride=2),
)
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
| GoogLeNet / Inception | 单一卷积尺度难选择 | 多尺度并行分支、$1\times1$ 瓶颈、GAP | 参数效率高，多尺度表示强 | 分支和通道配置复杂 |
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

- 上一篇：[PyTorch 工程实践与完整建模流程]({{ '/deep-learning/pytorch-engineering/' | relative_url }})
- [系列总览]({{ '/deep-learning/' | relative_url }})
- 下一篇：无（本篇是系列终点）

## 对应章节与参考资料

- D2L：[卷积神经网络](https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html)、[现代卷积神经网络](https://zh.d2l.ai/chapter_convolutional-modern/index.html)
- PyTorch：[`Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)、[`Conv3d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)、[`AdaptiveAvgPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html)
- 原始论文：[AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)、[VGG](https://arxiv.org/abs/1409.1556)、[Network in Network](https://arxiv.org/abs/1312.4400)、[GoogLeNet](https://arxiv.org/abs/1409.4842)、[Batch Normalization](https://arxiv.org/abs/1502.03167)、[ResNet](https://arxiv.org/abs/1512.03385)、[DenseNet](https://arxiv.org/abs/1608.06993)
