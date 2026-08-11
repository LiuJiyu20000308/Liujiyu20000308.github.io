---
layout: post
title: "PyTorch 与神经网络训练基础：从 Tensor Shape 到完整训练循环"
date: 2026-08-11 09:00 +0800
tags: [D2L, PyTorch, 深度学习, 神经网络]
toc: true
math: true
permalink: /deep-learning/foundations/
---

## 本篇逻辑主线

神经网络训练并不是一串需要背诵的 API。数据先被表示为带 shape 的张量，广播和矩阵乘法规定了它们怎样组合；层把张量变成预测，损失把预测压缩成一个可优化的标量；自动微分沿前向计算动态建立的计算图反向传播梯度；优化器保存参数引用和更新规则，于是 `zero_grad()`、`backward()`、`step()` 形成一次训练迭代。最后，`Dataset` 与 `DataLoader` 提供批数据，`train()`、`eval()` 和 `no_grad()` 区分训练与验证，我们才得到一条完整且不泄漏状态的训练流水线。

## 1. Tensor：数值之外首先看 shape

张量（tensor）是带有 dtype、device 和 shape 的多维数组。对神经网络代码，先写 shape 往往比先算数值更有效：

```python
X = torch.randn(32, 784)      # [batch, features]
W = torch.randn(784, 10)      # [in_features, out_features]
b = torch.zeros(10)           # [out_features]
logits = X @ W + b             # [32, 10]
```

`X @ W` 的内维必须相等：`[32, 784] @ [784, 10] -> [32, 10]`。`b` 没有 batch 维，却能加到每一行，是广播在工作。

### 广播不是随意复制

从末尾维度开始比较，两维相等、其中一维为 1，或某一方缺少该维时才可广播。例如：

```python
X = torch.randn(32, 10)
b = torch.randn(10)          # 视作 [1, 10]
Y = X + b                    # [32, 10]

scale = torch.randn(32, 1)
Z = X * scale                # 每个样本使用一个缩放值
```

广播通常只创建逻辑上的扩展视图，并不真的复制整块数据。但它也容易制造“代码能跑、语义错了”的 bug：`[32, 1]` 与 `[32]` 相减会广播成 `[32, 32]`，而不是期望的逐样本误差。比较张量前先对齐 shape。

### `reshape`、`view`、`unsqueeze`

- `view` 只在内存布局兼容时返回视图，转置后的非连续张量常需先 `contiguous()`。
- `reshape` 优先返回视图，必要时会复制，更适合作为通用接口；不要假设它一定与原张量共享存储。
- `unsqueeze(dim)` 增加长度为 1 的维度，`squeeze` 删除长度为 1 的维度；它们常用于显式准备广播。

```python
images = torch.randn(32, 1, 28, 28)  # [N, C, H, W]
flat = images.reshape(32, -1)         # [32, 784]
```

`-1` 表示让 PyTorch 根据元素总数推断这一维。reshape 前后元素总数必须一致。

### `sum`、`mean` 与 `keepdim`

```python
X = torch.randn(32, 10)
X.mean(dim=0).shape                 # [10]，对 batch 求均值
X.mean(dim=0, keepdim=True).shape   # [1, 10]，便于广播回 X
```

`dim=0` 消掉第 0 维，不是“保留第 0 维”。`keepdim=True` 保留一个长度为 1 的占位维。这个细节在标准化、BatchNorm 和注意力掩码中都会反复出现。

### 几个常见生成与截断运算

`torch.normal(mean, std, size=...)` 从正态分布采样，`torch.randn` 是均值 0、标准差 1 的快捷形式。`torch.clamp(X, min, max)` 把有限值截到区间内，例如 `clamp(pred, min=1)` 可避免随后 `log(pred)` 遇到非正数；但 `clamp` 不会自动把 `NaN` 修好，正无穷在设定有限 `max` 时才会被截断。数值异常应定位产生位置，而不是只在输出端掩盖。

## 2. 微积分与概率：训练公式背后的两套语言

导数描述标量输入的局部变化率，梯度把多变量偏导数组成向量。若 $f(\mathbf b)=\lVert\mathbf b\rVert_2^2=\mathbf b^\top\mathbf b$，则对列向量采用“梯度仍为列向量”的约定，有

$$
\nabla_{\mathbf b}f=2\mathbf b.
$$

有些资料使用分子布局把导数写成 $2\mathbf b^\top$；差异来自矩阵微积分的行/列约定，而不是数值结论矛盾。链式法则负责把局部导数组合成整体梯度。

自动微分可正向累积或反向累积。若函数有很多输入、一个标量损失，反向模式一次从输出回传就能得到所有参数梯度，正是神经网络训练的场景；代价是前向时通常要保存反向所需的中间值。它保存的是运算关系、输入和必要中间量，不必预先把每个导数数值都算好，反向时再调用对应运算的梯度规则。正向模式的成本更接近“输入方向数”，适合输入很少、输出很多的情形。

概率则描述不确定性：随机变量的期望是长期平均，方差描述围绕均值的波动，概率分布把可能结果与概率对应起来。最大似然、交叉熵、Dropout 的期望和 BatchNorm 的样本统计都建立在这套语言上。经验均值是对真实期望的有限样本估计，不能把二者混为一谈。

## 3. 层、模型、损失和优化器各自负责什么

`nn.Linear(in_features, out_features)` 实现

$$
Y = XW^\top + b,
$$

因此权重的 PyTorch shape 是 `[out_features, in_features]`：

```python
layer = nn.Linear(784, 10)
layer.weight.shape  # [10, 784]
layer.bias.shape    # [10]
```

`nn.Sequential` 适合纯串行结构：前一层输出自动成为后一层输入。

```python
net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
```

模型输出预测，损失函数回答“预测离目标多远”，优化器回答“根据梯度怎样改参数”。三者不要混为一谈。优化器在创建时接收了 `net.parameters()` 中的 `Parameter` 对象引用：

```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
```

所以 `optimizer.step()` 不需要再次传入模型。它已经持有同一批参数对象，并从每个参数的 `.grad` 读取梯度。

## 4. Dataset、DataLoader、iterator 和 generator

`Dataset` 定义“一个样本是什么”和“共有多少样本”；`DataLoader` 定义如何分批、打乱、并行加载和拼接样本。

```python
dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=True
)

for X_batch, y_batch in loader:
    # X_batch: [B, ...]，最后一个批次 B 可能小于 64
    ...
```

`iter(loader)` 创建迭代器，`next(iterator)` 取下一个 batch。`DataLoader` 本身是可迭代对象，并不等于某个永久迭代器；每轮 `for` 循环会创建新迭代器。Python generator 是一种惰性迭代器：执行到 `yield` 返回一个值，再次 `next()` 时从暂停处继续。理解这一点，就不会把“数据集已经全部载入内存”和“每次只迭代一个 batch”混为一谈。

训练集通常 `shuffle=True`；验证集不必打乱。`drop_last=True` 会丢弃不足一个 batch 的最后一批，只在算法确实要求固定 batch size 时使用。

## 5. 计算图与自动微分

当参与运算的张量需要梯度时，PyTorch 在前向执行过程中动态记录运算关系。若标量损失为 $L$，`L.backward()` 用链式法则计算每个叶子参数的

$$
\frac{\partial L}{\partial \theta}.
$$

前向传播不是只“算预测”：它还保存反向传播所需的中间信息。反向传播也不会神秘地“改变权重”，它只把梯度写入参数的 `.grad`；真正更新发生在 `optimizer.step()`。

### 叶子张量与原地修改

用户直接创建、且 `requires_grad=True` 的张量通常是叶子张量；`nn.Parameter` 也是叶子张量。对需要梯度的叶子张量做原地修改会破坏 autograd 对版本和历史的追踪，因此通常报错。

参数更新应交给优化器，或在明确的无梯度环境中完成：

```python
with torch.no_grad():
    w -= lr * w.grad
```

不要用 `.data` 绕开检查；它可能静默破坏计算图。现代 PyTorch 代码优先使用 `no_grad()`。

### `detach()` 与 `no_grad()`

`x.detach()` 返回与 `x` 共享底层存储、但与当前计算图断开的张量；它适合把某个中间结果当常量使用。由于共享存储，对 detached 张量的原地修改仍可能影响原值。

`torch.no_grad()` 是一个上下文，表示其中的新运算不构建梯度图，适合验证、推理和手动参数更新。两者作用范围不同：一个切断特定张量的历史，一个关闭一段代码的梯度记录。

## 6. 为什么梯度会累积

PyTorch 默认把新梯度加到已有 `.grad`，而不是覆盖。这使多个小 batch 的梯度累积成为可能，但普通训练若忘记清零，更新方向就会混入之前的 batch。

一次标准迭代的顺序是：

```python
optimizer.zero_grad()  # 1. 清除旧梯度
logits = net(X)        # 2. 前向传播
loss = criterion(logits, y)
loss.backward()        # 3. 写入/累加梯度
optimizer.step()       # 4. 更新参数
```

把 `zero_grad()` 放在 `step()` 后也能工作，只要每次 `backward()` 前已清零；放在迭代开头更容易审计。

### `sum`、`mean`、batch size 与梯度尺度

若单样本损失为 $\ell_i$，则

$$
L_{\text{sum}}=\sum_{i=1}^{B}\ell_i,
\qquad
L_{\text{mean}}=\frac{1}{B}\sum_{i=1}^{B}\ell_i.
$$

两者梯度相差约 $B$ 倍。`reduction='mean'` 让梯度尺度较少依赖 batch size，是多数 PyTorch 损失的默认值；`sum` 不是错误，但学习率必须与其尺度配套。做梯度累积时，若希望模拟一个大 batch，通常把每个 micro-batch 的 mean loss 再除以累积步数。

## 7. `train()`、`eval()` 和验证模式

`model.train()` 与 `model.eval()` 不控制是否求梯度，而是切换模块行为：Dropout 在训练时随机丢弃，评估时关闭；BatchNorm 在训练时使用 batch 统计量并更新运行统计，评估时使用已保存的运行统计。

因此验证通常同时需要：

```python
model.eval()
with torch.no_grad():
    ...
```

只用 `eval()` 仍会构图；只用 `no_grad()` 则可能让 Dropout 和 BatchNorm 保持训练行为。进入下一轮训练前再调用 `model.train()`。

## 8. 一条完整、可复用的训练与验证循环

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Flatten(),                  # [B, 1, 28, 28] -> [B, 784]
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 10),            # logits: [B, 10]
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_count = 0.0, 0, 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)   # y: [B], dtype long
        optimizer.zero_grad()
        logits = model(X)                   # [B, 10]
        loss = criterion(logits, y)         # scalar
        loss.backward()
        optimizer.step()

        batch_size = X.shape[0]
        train_loss += loss.item() * batch_size
        train_correct += (logits.argmax(dim=1) == y).sum().item()
        train_count += batch_size

    model.eval()
    val_loss, val_correct, val_count = 0.0, 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            val_loss += loss.item() * X.shape[0]
            val_correct += (logits.argmax(dim=1) == y).sum().item()
            val_count += X.shape[0]

    print(
        epoch,
        train_loss / train_count,
        train_correct / train_count,
        val_loss / val_count,
        val_correct / val_count,
    )
```

这里先用 `loss.item() * batch_size` 恢复该批损失总和，最后再除以真实样本数，避免最后一个小 batch 让“各批 mean 的简单平均”产生偏差。

## 本篇知识链总结

`shape` 决定张量是否合法组合；层把输入映射为预测；损失把预测变成标量目标；计算图保存链式法则所需关系；`backward()` 计算并累积梯度；优化器持有参数引用并执行更新；`DataLoader` 持续提供 batch；训练模式和验证模式隔离随机层、运行统计与梯度记录。把这条链说清楚，绝大多数“训练循环为什么这样写”的问题就不再需要死记。

## 常见误区

- 认为 `backward()` 会更新权重；它只计算梯度。
- 忘记梯度默认累积，或在 `backward()` 后、`step()` 前清零。
- 只看元素数量，不核对广播后的 shape。
- 认为 `eval()` 等于 `no_grad()`。
- 用 `.data` 规避 autograd 的原地修改保护。
- 把 batch loss 的 mean 再等权平均，忽略最后一批大小不同。
- 认为 `reshape` 一定共享存储，或认为 `view` 可处理任意非连续张量。

## 系列导航

- 上一篇：无（本篇是系列起点）
- [系列总览]({{ '/deep-learning/' | relative_url }})
- 下一篇：[从线性回归到 softmax 分类]({{ '/deep-learning/linear-models/' | relative_url }})

## 对应章节与参考资料

- D2L：[预备知识](https://zh.d2l.ai/chapter_preliminaries/index.html)、[线性神经网络](https://zh.d2l.ai/chapter_linear-networks/index.html)
- PyTorch：[Autograd](https://docs.pytorch.org/docs/stable/autograd.html)、[`DataLoader`](https://docs.pytorch.org/docs/stable/data.html)、[`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
