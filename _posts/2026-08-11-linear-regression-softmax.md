---
layout: post
title: "从线性回归到 Softmax 分类：损失、概率与决策边界"
date: 2026-08-11 09:10 +0800
tags: [D2L, PyTorch, 深度学习, 神经网络]
toc: true
math: true
permalink: /deep-learning/linear-models/
---

## 本篇逻辑主线

线性回归从一个最简单的问题出发：用特征的加权和预测连续值。平方损失把预测误差变成可微目标，高斯噪声又给它一个最大似然解释，随机梯度下降据此学习参数。分类任务的目标不再是任意实数，于是线性层先输出 logits，softmax 再把互斥类别的相对证据变成概率，交叉熵用正确类别的负对数概率训练模型。沿着这条线还可以看清 sigmoid、BCE、多标签分类的适用边界，以及感知机、SVM 等线性分类器为什么只是换了损失或间隔思想，并没有突破线性决策边界。

## 1. 线性回归：用线性关系预测连续值

给定 $d$ 个特征 $\mathbf{x}\in\mathbb{R}^d$，线性回归写作

$$
\hat y = \mathbf{w}^\top\mathbf{x}+b.
$$

一个 batch 的实现是：

```python
X = torch.randn(64, d)       # [B, d]
net = nn.Linear(d, 1)
y_hat = net(X)               # [B, 1]
```

这里的“线性”通常包含偏置，因此数学上是仿射映射。模型不要求原始世界完全线性，只要求在所选特征空间里，用这个形式作为可接受的近似。

### 平方损失、平均与 $1/2$

单样本平方损失常写成

$$
\ell(y,\hat y)=\frac{1}{2}(y-\hat y)^2.
$$

$1/2$ 只为求导后抵消平方带来的 2，并不改变最优点。对 batch 求平均：

$$
L=\frac{1}{B}\sum_{i=1}^{B}\ell_i,
$$

是为了让梯度尺度较少随 batch size 改变；求和也可以，但学习率需要相应调整。PyTorch 的 `nn.MSELoss()` 默认直接计算均方误差，不额外乘 $1/2$。

```python
loss_fn = nn.MSELoss(reduction="mean")
loss = loss_fn(y_hat, y.reshape(-1, 1))
```

shape 必须真正匹配。若 `y_hat` 是 `[B, 1]`、`y` 是 `[B]`，广播可能生成 `[B, B]` 并只给出警告，这不是正确的逐样本误差。

### 为什么平方损失对应最大似然

假设观测满足

$$
y=\mathbf{w}^\top\mathbf{x}+b+\epsilon,
\qquad \epsilon\sim\mathcal N(0,\sigma^2),
$$

则最大化所有样本的高斯似然，等价于最小化负对数似然；去掉与参数无关的常数后，正好得到误差平方和。于是 MSE 不只是“看起来平滑”，它隐含了独立同方差高斯噪声假设。

当离群点很多时，平方会让大误差支配梯度。Huber loss 在小误差区使用平方、在大误差区转为线性：

$$
L_\delta(r)=
\begin{cases}
\frac12r^2,& |r|\le\delta,\\
\delta(|r|-\frac12\delta),& |r|>\delta,
\end{cases}
$$

它在平滑优化和异常值鲁棒性之间折中，对应 `nn.HuberLoss`。

## 2. SGD：不必每一步都看完所有数据

批量随机梯度下降用一个随机小批量估计全数据梯度：

$$
\mathbf{w}\leftarrow\mathbf{w}-\eta\nabla_{\mathbf{w}}L_\mathcal{B}.
$$

batch 越大，梯度估计通常越稳定但单步计算和显存越多；batch 越小，噪声更大却能更频繁更新。学习率 $\eta$ 与损失 reduction、batch size 和优化器共同决定实际步长，不能孤立理解。

线性回归从零实现与简洁实现本质一致：前者手动维护 `w.grad` 并更新，后者把参数注册为 `Parameter`，交给 `torch.optim`。

## 3. 从回归到分类：先保留 logits

分类不能把类别编号当连续量做回归。例如猫=1、狗=2、鸟=3 并不意味着鸟比猫“大两倍”。对 $K$ 个互斥类别，线性层输出 $K$ 个实数：

$$
\mathbf{o}=W\mathbf{x}+\mathbf{b},
\qquad \mathbf{o}\in\mathbb{R}^K.
$$

这些未归一化分数称为 logits。它们没有必须落在 $[0,1]$ 的限制，且整体加上同一常数不会改变 softmax 概率。

```python
classifier = nn.Linear(d, K)
logits = classifier(X)       # [B, K]
```

### Softmax：比较互斥类别的相对证据

$$
p_k=\frac{\exp(o_k)}{\sum_{j=1}^{K}\exp(o_j)},
\qquad \sum_k p_k=1.
$$

softmax 把 logits 转成概率分布，但指数可能溢出。数值稳定的做法是先减去每个样本的最大 logit：

$$
p_k=\frac{\exp(o_k-m)}{\sum_j\exp(o_j-m)},
\qquad m=\max_j o_j.
$$

更进一步，交叉熵实现通常直接使用 `logsumexp`，避免先计算接近 0 的概率再取对数。

## 4. 交叉熵：训练正确类别的对数概率

若真实标签用 one-hot 向量 $\mathbf y$ 表示，交叉熵是

$$
H(\mathbf y,\mathbf p)=-\sum_{k=1}^{K}y_k\log p_k
=-\log p_{y}.
$$

将 softmax 代入可得单样本稳定形式：

$$
\ell(\mathbf o,y)=-o_y+\log\sum_j\exp(o_j).
$$

这也是多项类别分布的负对数似然。它既惩罚正确类概率低，也通过归一化让各类别竞争。

### `CrossEntropyLoss` 为什么接收 logits

`nn.CrossEntropyLoss` 在一个数值稳定的运算中完成 `log_softmax + NLLLoss`：

```python
logits = model(X)                  # [B, K], 任意实数
y = torch.tensor([...]).long()     # [B], 值域 0..K-1
loss = nn.CrossEntropyLoss()(logits, y)
```

不要在前面再手动 `softmax`。传概率不仅重复计算，还会损失数值稳定性，并改变该函数所期待的输入语义。推理或展示置信度时才使用：

```python
probs = logits.softmax(dim=1)       # [B, K]
pred = logits.argmax(dim=1)         # 与 probs.argmax(dim=1) 相同
```

`argmax` 只适用于“恰选一个最大类”的决策。它不保证最大概率足够高；若业务允许拒绝预测，需要另设置信度阈值。

在医疗等代价不对称场景，“概率最大的类别”也不一定是最优行动。应根据误诊/漏诊代价最小化期望风险，并在独立验证集上确定阈值；分类概率、最终决策和业务动作是三个层次。

### 信息量、熵与软标签

事件概率越小，观察到它的信息量 $I(x)=-\log p(x)$ 越大。熵

$$
H(P)=\mathbb E_{x\sim P}[-\log P(x)]
$$

正是信息量在真实分布下的期望；交叉熵 $H(P,Q)$ 则用模型分布 $Q$ 给来自 $P$ 的事件编码。one-hot 标签只是特殊的目标分布。软标签允许多个类别拥有非零目标概率，可表达类别相似性或 teacher 的不确定性；label smoothing 把少量质量从正确类分给其余类别，抑制过度自信，但平滑强度仍需验证。

当类别数极大（如下一词预测）时，输出权重、完整 softmax 的计算和显存都随词表规模增长，长尾类别数据又很稀疏。工程上会使用子词词表、采样近似、分层 softmax 或更高效的输出结构，而不是仅盲目增大词表。

## 5. Sigmoid、Softmax 与 BCE 的任务边界

最常混淆的不是公式，而是“标签是否互斥”。

| 任务 | 输出 shape | 概率变换 | 典型损失 | 决策 |
|---|---:|---|---|---|
| 二分类（一个 logit） | `[B]` 或 `[B,1]` | 每样本 sigmoid | `BCEWithLogitsLoss` | sigmoid 概率与阈值比较 |
| 二分类（两个互斥类） | `[B,2]` | 类别间 softmax | `CrossEntropyLoss` | `argmax(dim=1)` |
| $K$ 类单标签 | `[B,K]` | 类别间 softmax | `CrossEntropyLoss` | `argmax(dim=1)` |
| $K$ 标签可并存 | `[B,K]` | 每个标签独立 sigmoid | `BCEWithLogitsLoss` | 每个标签独立阈值 |

sigmoid 为单个 logit 给出

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

二分类用一个 sigmoid logit 与用两个 softmax logits 在表达上可以互相转换，但参数化和接口不同。多标签任务中“猫”和“室内”可以同时为真，因此不能用 softmax 强迫它们概率和为 1。

`nn.BCEWithLogitsLoss` 把 sigmoid 与二元交叉熵融合，也应直接接收 logits：

```python
logits = model(X)                     # [B, K]
targets = targets.float()             # [B, K]，元素为 0/1
loss = nn.BCEWithLogitsLoss()(logits, targets)
pred = logits.sigmoid() >= threshold
```

阈值 0.5 等价于 logit 阈值 0，但类别不平衡、误报与漏报代价不同时，应在验证集上选择阈值，而不是默认 `argmax`。

## 6. 线性分类器能做什么，不能做什么

无论是 softmax 回归、逻辑回归还是感知机，只要输入只经过一次线性变换，类别之间的决策边界都是超平面。换损失会改变训练准则，却不会自动获得弯曲边界。

### 感知机

感知机对被误分类样本更新权重，适用于线性可分数据；若数据不可分，经典算法可能不收敛。它强调“分对”，但不提供校准概率。

### SVM 与最大间隔

支持向量机不仅要求分对，还最大化离决策边界最近样本的间隔。软间隔 SVM 用 hinge loss 与正则项容忍部分错误；核方法把输入隐式映射到更高维，从而在原空间得到非线性边界。它与神经网络的共同点是都在学习决策函数，区别在于特征映射是预先选定的核，还是由多层网络端到端学习。

### VC dimension 的位置

VC 维描述一个函数类打散样本的能力，是理论容量指标，不是模型参数个数的简单同义词。它提醒我们：表达能力越大，拟合训练集越容易，但泛化还取决于数据量、归纳偏置、优化和正则化。现代深度网络的实践不能仅靠一个 VC 维数字解释。

线性边界的根本限制，正是下一篇引入隐藏层和非线性激活的原因。

## 本篇知识链总结

线性回归用仿射映射预测连续值，MSE 对应高斯噪声下的最大似然，SGD 从小批量估计梯度；分类先输出 logits，softmax 表达互斥类别的相对概率，交叉熵等价于正确类别的负对数似然；sigmoid/BCE 适合一个二元事件或多个可并存标签。不同线性分类损失提供了不同训练偏好，但只有引入非线性特征变换，才能突破超平面决策边界。

## 常见误区

- 把类别编号当连续值做 MSE 回归。
- 在 `CrossEntropyLoss` 前手动 softmax。
- 把多标签任务当成多分类，用 softmax 强迫标签竞争。
- 只因 `argmax` 返回一个类就认为模型足够自信。
- 忽略 `[B]` 与 `[B,1]` 广播成错误损失矩阵的风险。
- 认为 MSE 中 $1/2$ 或 reduction 的选择会改变最优解；它们主要改变梯度尺度。
- 认为换成 SVM 损失就自动获得非线性边界。

## 系列导航

- 上一篇：[PyTorch 与神经网络训练基础]({{ '/deep-learning/foundations/' | relative_url }})
- [系列总览]({{ '/deep-learning/' | relative_url }})
- 下一篇：[多层感知机、泛化与稳定训练]({{ '/deep-learning/mlp-generalization/' | relative_url }})

## 对应章节与参考资料

- D2L：[线性回归](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html)、[Softmax 回归](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html)
- PyTorch：[`CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)、[`BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)、[`HuberLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)
