---
layout: post
title: "D2L 循环神经网络：从序列数据与 BPTT 到 GRU、LSTM、深层和双向建模"
date: 2026-08-20 09:00 +0800
tags: [D2L, PyTorch, 深度学习, RNN, GRU, LSTM, 序列模型]
toc: true
math: true
permalink: /deep-learning/recurrent-neural-networks/
---

## 本篇逻辑主线

图像模型通常把一个样本一次性送入网络；序列模型却必须回答两个额外问题：当前输入之前发生了什么，以及这段历史应该保留多久。语言模型把联合概率按时间拆成一连串“预测下一个词元”的条件概率，RNN 再用隐状态压缩此前信息。这样做带来了贯穿本篇的三组矛盾：序列很长但计算图不能无限增长，参数需要跨时间共享但数据规律未必完全平稳，历史既要长期保留又要在边界处及时覆盖。

本文沿着下面的因果链整理 D2L 的循环神经网络基础和现代循环网络：

```text
序列预测与因果方向
    ↓
文本 → 词元 → 词表 → 语料索引
    ↓
语言模型、n 元语法、稀疏计数与平滑
    ↓
随机采样 / 顺序分区，X 与 Y 错开一个时间步
    ↓
RNN 隐状态、时间参数共享、字符级语言模型
    ↓
BPTT、detach、梯度裁剪、困惑度
    ↓
GRU：选择性读取历史 + 选择性更新状态
    ↓
LSTM：长期记忆的保留、准备、写入与读取
    ↓
深层 RNN 增加表示深度，双向 RNN 引入未来上下文
```

真正需要贯穿始终检查的是：**当前 tensor 的每一维是什么，当前输出允许看见哪些时间位置，状态值与状态的梯度历史是否都要跨 batch 传递。**

### 对应 Notebook

| 主题 | Notebook | 本文位置 |
|---|---|---|
| 序列模型与多步预测 | `sequence.ipynb` | 第 1 节 |
| 文本读取、词元化与词表 | `text-preprocessing.ipynb` | 第 2 节 |
| 语言模型、n 元语法与序列采样 | `language-models-and-dataset.ipynb` | 第 2～3 节 |
| RNN 原理 | `rnn.ipynb` | 第 4 节 |
| 从零实现与训练 | `rnn-scratch.ipynb` | 第 4～6 节 |
| PyTorch 简洁实现 | `rnn-concise.ipynb` | 第 6 节 |
| 通过时间反向传播 | `bptt.ipynb` | 第 6 节 |
| GRU | `gru.ipynb` | 第 7 节 |
| LSTM | `lstm.ipynb` | 第 8 节 |
| 深层循环网络 | `deep-rnn.ipynb` | 第 10 节 |
| 双向循环网络 | `bi-rnn.ipynb` | 第 11 节 |

## 1. 序列建模：一步预测容易，多步外推困难

给定序列 $x_1,\ldots,x_T$，联合概率总能按时间方向分解：

$$
P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_1,\ldots,x_{t-1}).
$$

自回归模型直接用过去的观测预测未来。若只保留最近 $\tau$ 个值，可以写成

$$
\hat x_t=f(x_{t-\tau},\ldots,x_{t-1}).
$$

训练时的单步预测总是使用真实历史，误差不会进入下一次输入；多步预测则必须把自己的输出重新作为输入：

$$
\hat x_{t+2}=f(x_{t-\tau+2},\ldots,x_t,\hat x_{t+1}).
$$

因此预测步数越远，分布偏移和误差累积越严重。训练误差低只说明模型会在“真实历史附近”做局部预测，并不保证它在自己的预测轨迹上仍然稳定。时间序列划分也必须尊重时间方向：未来数据不能进入训练当前预测的特征、归一化统计或超参数选择。

固定窗口把历史长度写死在输入 shape 中。隐状态模型则递归更新

$$
h_t=f(x_t,h_{t-1}),
$$

用固定长度的 $h_t$ 概括可变长度历史。这是 RNN 的出发点，也是它的限制：历史不会消失在依赖关系上，却可能在有限容量和反复非线性变换中逐渐丢失。

## 2. 从文本到训练语料：词元、词表和 n 元语法

### 2.1 词元化与词频

文本首先被拆成字符、单词或子词。`collections.Counter(tokens)` 统计一维词元列表中每个词元的出现次数：

```python
import collections

tokens = ["the", "time", "the", "machine"]
counter = collections.Counter(tokens)
print(counter)                # Counter({'the': 2, 'time': 1, 'machine': 1})
print(counter.most_common())  # 按频率从高到低排列
```

如果输入是“句子列表的列表”，需要先展平；内部列表不可哈希，不能直接作为 `Counter` 的键：

```python
lines = [["the", "time"], ["the", "machine"]]
tokens = [token for line in lines for token in line]
counter = collections.Counter(tokens)
```

词表维护两个方向的映射：

```text
token_to_idx：词元 → 整数编号
idx_to_token：整数编号 → 词元
```

常见的 `__getitem__` 写法同时支持单个词元和词元列表：

```python
def __getitem__(self, tokens):
    if not isinstance(tokens, (list, tuple)):
        return self.token_to_idx.get(tokens, self.unk)
    return [self[token] for token in tokens]
```

于是 `vocab['time']` 返回一个编号，`vocab[['time', 'machine']]` 则递归返回编号列表。方括号访问会自动调用 `__getitem__`，未知词通过字典 `get` 的默认值映射到 `<unk>`。

### 2.2 n 元语法只是相邻切片的配对

二元词组可以由两个错位切片组成：

```python
corpus = ["i", "love", "deep", "learning"]
bigram_tokens = list(zip(corpus[:-1], corpus[1:]))
# [('i', 'love'), ('love', 'deep'), ('deep', 'learning')]
```

三元词组同理：

```python
trigram_tokens = list(zip(
    corpus[:-2], corpus[1:-1], corpus[2:]
))
```

n 元语言模型用最近 $n-1$ 个词元近似完整历史，例如

$$
P(x_t\mid x_1,\ldots,x_{t-1})
\approx P(x_t\mid x_{t-n+1},\ldots,x_{t-1}).
$$

窗口变长能表达更多局部结构，但可能组合数按 $|V|^n$ 增长，而真实语料中的绝大多数组合很少出现。齐普夫定律又使词频呈明显长尾：少数词极常见，大量词和词组只出现一两次。

### 2.3 拉普拉斯平滑解决的是零频，不是语义

若直接按计数估计

$$
\hat P(w\mid h)=\frac{n(h,w)}{n(h)},
$$

训练集中未出现的合理词组会得到零概率，整条测试序列的乘积也随之变成零。加一平滑改为

$$
\hat P(w\mid h)=\frac{n(h,w)+1}{n(h)+|V|},
$$

从高频事件拿出少量概率质量，分给低频和未观察事件。“处理结构丰富而频率不足的低频词组”指的是缓解这种零频和数据稀疏，而不是理解了词义。可能的 n 元组极多，统一加常数会给大量不合理组合分配过多概率，所以真实语言模型更常使用回退、插值、Kneser--Ney 或神经表示。

## 3. 长序列怎样组成 batch

### 3.1 为什么标签 `Y` 也是一个序列

语言模型在每个时间步都预测下一个词元：

```text
X = [1, 2, 3, 4, 5]
Y = [2, 3, 4, 5, 6]
```

这不是只做五次互不相关的“单值到单值”预测。RNN 的隐藏状态逐步累积历史，所以五个监督信号分别表示：

```text
[1]             → 2
[1, 2]          → 3
[1, 2, 3]       → 4
[1, 2, 3, 4]    → 5
[1, 2, 3, 4, 5] → 6
```

一次前向传播因此能利用所有位置，而不是读完整段后只得到一个训练目标。若下一段从 `[6,7,8,...]` 开始，顺序采样会把上一段末尾的隐藏状态继续传入；此时用 `6` 预测 `7` 仍可间接利用更早的 `5`。随机采样的相邻 batch 没有原序列连续关系，所以必须重置状态。

### 3.2 顺序分区并不会枚举所有滑动窗口

`load_data_time_machine(batch_size, num_steps)` 默认使用顺序分区：选择一个随机偏移，把保留下来的语料分成 `batch_size` 条长行，再沿列方向每次截取 `num_steps`。一个 epoch 会处理所保留的词元，却不会枚举所有可能的重叠窗口；开头、结尾和不足完整 batch 的部分还会被丢弃。

下面给出核心实现，返回的 `X,Y` shape 都是 `[B,T]`：

```python
import random
import torch

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)

    num_batches = Xs.shape[1] // num_steps
    for start in range(0, num_batches * num_steps, num_steps):
        yield (Xs[:, start:start + num_steps],
               Ys[:, start:start + num_steps])
```

随机偏移改变固定切分边界，提高跨 epoch 的覆盖率，但并不产生完美均匀的所有窗口分布。若确实需要所有合法窗口，可以使用步幅为 1 的重叠滑动窗口：

```python
def all_subsequences(corpus, num_steps):
    for start in range(len(corpus) - num_steps):
        X = corpus[start:start + num_steps]
        Y = corpus[start + 1:start + num_steps + 1]
        yield X, Y
```

步幅小于 `num_steps` 时，相邻窗口会重复计算相同词元。它增加的是窗口起点和上下文组合的覆盖，不是单个样本的上下文长度。若步幅为 `stride>1`，可以让偏移量在 `0,...,stride-1` 间轮换；步幅为 1 时所有起点已被覆盖，无需额外随机偏移。

## 4. 普通 RNN：同一状态转移在时间上重复使用

设当前输入、上一隐状态分别为 $X_t\in\mathbb R^{B\times d}$、$H_{t-1}\in\mathbb R^{B\times h}$，普通 RNN 为

$$
H_t=\tanh(X_tW_{xh}+H_{t-1}W_{hh}+b_h),
$$

$$
O_t=H_tW_{hq}+b_q.
$$

每个时间步都使用同一组 $W_{xh},W_{hh},W_{hq}$。沿时间展开看起来像很多层，但这些层共享参数，因此参数量不随时间步数 $T$ 增加；计算量和训练时保存的激活却会随 $T$ 增加。

参数共享表达了“不同时间位置大体遵循同一种转移规律”的归纳偏置。它不意味着每个位置输出相同，因为 $X_t$ 和 $H_{t-1}$ 不同。同一词元在不同上下文会产生不同状态。若市场阶段、季节或绝对位置真的改变了生成机制，共享参数可能在不同片段间折中；可以加入时间或阶段特征，使用更强门控结构、混合专家或分段模型。

### 4.1 独热编码与嵌入查询

嵌入表示用固定长度实数向量表示离散词元。若嵌入表 $E\in\mathbb R^{V\times d_e}$，词元 $i$ 的表示就是第 $i$ 行 $E[i,:]$。独热向量 $e_i$ 与矩阵相乘恰好选择对应行：

$$
e_i^TW_{xh}=W_{xh}[i,:].
$$

因此 `one_hot(token) @ W_xh` 在计算效果上就是嵌入查表；显式 `nn.Embedding` 避免构造稀疏的 $V$ 维张量，并允许嵌入维度与隐藏维度分离，但不会仅因换了 API 就自动提高准确率。

### 4.2 从零实现时每个 `X` 是一个时间步的整个 batch

原始 `X` 为 `[B,T]`，转置并独热编码后是 `[T,B,V]`。因此下面循环中的 `X_t` 不是整条序列，而是当前时间步上所有 $B$ 个样本：

```python
import torch.nn.functional as F

def rnn_forward(inputs, state, params):
    # inputs: [T, B, V]；H: [B, H]
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    for X_t in inputs:  # X_t: [B, V]
        H = torch.tanh(X_t @ W_xh + H @ W_hh + b_h)
        Y_t = H @ W_hq + b_q       # [B, V]
        outputs.append(Y_t)

    # 时间优先地拼成 [T*B, V]，只返回最后的H供下一段使用
    return torch.cat(outputs, dim=0), (H,)

inputs = F.one_hot(X.T, num_classes=vocab_size).float()
logits, state = rnn_forward(inputs, state, params)
```

`H` 的每一行是一个样本的状态，矩阵乘法只是并行处理 batch，不会混合不同行。每个时间步都要预测下一个词元，所以所有 `Y_t` 都放入 `outputs`；隐藏状态本身则不断覆盖，最终状态传给下一段。

## 5. 输出、标签、交叉熵与困惑度

设 `B=2,T=3`：

```text
X = [[a,b,c], [u,v,w]]
Y = [[b,c,d], [v,w,x]]
```

RNN 按时间步拼接输出，顺序是 `[b,v,c,w,d,x]` 对应的六行 logits，所以：

```python
y = Y.T.reshape(-1)  # [b,v,c,w,d,x]
```

标签必须先从 `[B,T]` 转为 `[T,B]` 再展平，才能与 `torch.cat(outputs, dim=0)` 的时间优先顺序一致。最终 shape 为：

```text
logits: [T*B, V]
targets: [T*B]
```

它们故意不完全相同。多分类交叉熵要求每个样本提供 $V$ 个类别分数，而标签只保存一个正确类别索引：

$$
\ell_i=-\log\frac{\exp(\text{logits}_{i,y_i})}
{\sum_{j=1}^{V}\exp(\text{logits}_{i,j})}.
$$

PyTorch 的 `CrossEntropyLoss` 内部组合了 `log_softmax` 和正确类别索引选择，不需要把标签显式转成 `[T*B,V]` 的独热矩阵。

平均交叉熵为

$$
L=-\frac1N\sum_{i=1}^{N}\log p_i,
$$

困惑度定义为

$$
\operatorname{PPL}=\exp(L)
=\left(\prod_{i=1}^{N}\frac1{p_i}\right)^{1/N}.
$$

指数来自对数概率的反变换，不是因为标签只有 0 和 1。若模型每次都给正确字符概率 $1/V$，困惑度就是 $V$；完美预测时困惑度为 1。训练困惑度接近 1 也可能只是记住了很小的语料，模型选择仍应查看按时间划分的验证集。

## 6. BPTT：状态值可以跨 batch，计算图不能无限跨 batch

### 6.1 为什么只需要 `detach(state)`

顺序分区中，上一批的最终状态是下一批唯一复用的中间结果：

```text
batch 1：X1 → H1 → loss1
                 ↓ 保留数值
batch 2：X2 + H1 → H2 → loss2
```

若不分离，`loss2.backward()` 会继续穿过 `H1` 回到 batch 1，计算图随 batch 不断增长；普通代码还会遇到“再次通过已释放计算图反向传播”或参数原地更新的版本错误。`detach` 保留状态值但切断此前的梯度历史，这正是截断的通过时间反向传播：

```python
def detach_state(state):
    if isinstance(state, tuple):       # LSTM: (H, C)
        return tuple(s.detach() for s in state)
    return state.detach()              # RNN / GRU: H
```

其他量不需要分离：`X,Y` 默认不求梯度；当前 `logits,loss` 不传给下一批；模型参数必须保留梯度，否则不能训练，只需要在每轮更新前清空 `.grad`。

从零实现为了统一接口，常把单个状态写成 `(H,)`，其外层是 tuple、内部才是二维 Tensor。PyTorch 内置单层单向 `nn.RNN/nn.GRU` 返回三维状态 `[1,B,H]`，`nn.LSTM` 返回两个三维张量 `(H,C)`。第一维统一表示 `num_layers * num_directions`；`detach` 对二维或三维都一样有效。

### 6.2 梯度为什么消失或爆炸

跨越多个时间步的梯度含有状态雅可比的连乘：

$$
\frac{\partial H_t}{\partial H_k}
=\prod_{j=k+1}^{t}\frac{\partial H_j}{\partial H_{j-1}}.
$$

若这些变换持续缩小向量，梯度指数趋近于 0，早期词元难以学习；若持续放大，梯度会突然爆炸。完整 BPTT 还需要保存整条序列的激活，时间和内存代价都很高。固定长度截断带来有偏但更稳定的梯度估计，实践中通常比无限反向传播更可控。

梯度裁剪限制整体范数：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

它能阻止一次巨大更新破坏参数，却不能恢复已经接近零的梯度。GRU、LSTM 的门控和近似恒等通路才是缓解长期梯度消失的结构性方法。

### 6.3 一个统一的训练 epoch

下面的训练函数同时适用于内置 RNN、GRU 和 LSTM 包装模型：

```python
import math
from torch import nn

def train_epoch(model, train_iter, optimizer, device,
                use_random_iter=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    state = None
    loss_sum = token_count = 0

    for X, Y in train_iter:                 # X,Y: [B,T]
        X, Y = X.to(device), Y.to(device)

        if state is None or use_random_iter:
            state = None                    # 内置循环层会创建零状态
        else:
            state = detach_state(state)     # 保留值，截断跨batch梯度

        targets = Y.T.reshape(-1)            # [T*B]
        logits, state = model(X, state)      # [T*B,V]
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n = targets.numel()
        loss_sum += loss.item() * n
        token_count += n

    return math.exp(loss_sum / token_count)
```

随机采样的相邻 batch 不连续，所以每批都把 `state=None`；顺序分区才保留状态值并 `detach`。

## 7. GRU：候选状态怎样读历史，最终状态怎样写更新

GRU 先计算重置门与更新门：

$$
R_t=\sigma(X_tW_{xr}+H_{t-1}W_{hr}+b_r),
$$

$$
Z_t=\sigma(X_tW_{xz}+H_{t-1}W_{hz}+b_z).
$$

候选状态和最终状态为

$$
\tilde H_t=\tanh\left(X_tW_{xh}+(R_t\odot H_{t-1})W_{hh}+b_h\right),
$$

$$
H_t=Z_t\odot H_{t-1}+(1-Z_t)\odot\tilde H_t.
$$

两个门都涉及过去，但作用位置不同：

| 门 | 控制的问题 | 接近 0 | 接近 1 |
|---|---|---|---|
| 重置门 $R_t$ | 构造候选内容时，读取多少旧记忆？ | 候选主要依赖当前输入 | 候选结合完整旧状态 |
| 更新门 $Z_t$ | 候选算好后，最终写入多少新内容？ | 使用候选状态 | 直接保留旧状态 |

重置门不是旧状态在最终结果中的混合比例。即使 $R_t=0$，只要 $Z_t=1$，仍有 $H_t=H_{t-1}$：候选内容被重建了，但根本没有写入。只有 $R_t=0,Z_t=0$ 时，模型才根据当前输入重建最终状态。

更新门还提供从 $H_{t-1}$ 到 $H_t$ 的加法直通路径。其导数包含

$$
\frac{\partial L}{\partial H_{t-1}}
=Z_t\odot\frac{\partial L}{\partial H_t}+\text{其他路径的梯度项}.
$$

当需要长期记忆的维度上 $Z_t\approx1$ 时，梯度近似原样传回，而不必在每步都乘容易收缩的 `tanh` 导数和循环矩阵。跨多步的直接路径系数约为 $\prod_j Z_j$。这只能缓解而不能保证消除梯度消失或爆炸。

PyTorch 简洁实现只需替换循环层：

```python
gru = nn.GRU(
    input_size=vocab_size,
    hidden_size=256,
    num_layers=1,
)
```

输入是 `[T,B,V]`，输出是每个时间步的 `[T,B,H]`，最终状态为 `[num_layers,B,H]`。GRU 比普通 RNN 多计算两个门，参数与主要矩阵乘法约为普通循环部分的三倍，但通常更容易学习长期依赖。

## 8. LSTM：长期记忆的保留、准备、写入和读取

LSTM 把内部长期记忆 $C_t$ 与对外可见状态 $H_t$ 分开。三个 sigmoid 门并行地由 $X_t,H_{t-1}$ 计算：

$$
\begin{aligned}
I_t&=\sigma(X_tW_{xi}+H_{t-1}W_{hi}+b_i),\\
F_t&=\sigma(X_tW_{xf}+H_{t-1}W_{hf}+b_f),\\
O_t&=\sigma(X_tW_{xo}+H_{t-1}W_{ho}+b_o).
\end{aligned}
$$

候选记忆也是并行分支：

$$
\tilde C_t=\tanh(X_tW_{xc}+H_{t-1}W_{hc}+b_c).
$$

输入门并不先参与候选记忆计算。“输入门”指它控制候选内容进入记忆元，而不是控制 $X_t$ 是否进入候选分支。二者计算完成后才组合：

$$
C_t=F_t\odot C_{t-1}+I_t\odot\tilde C_t,
$$

$$
H_t=O_t\odot\tanh(C_t).
$$

| 量 | 理论意义 |
|---|---|
| 遗忘门 $F_t$ | 旧记忆保留多少 |
| 候选记忆 $\tilde C_t$ | 准备写入什么具体内容；它不是比例门 |
| 输入门 $I_t$ | 候选内容实际写入多少 |
| 记忆元 $C_t$ | 更新后的内部长期存储 |
| 输出门 $O_t$ | 当前记忆对外读取多少 |
| 隐状态 $H_t$ | 当前对输出层和下一步门控可见的工作状态 |

典型组合很直观：$F=1,I=0$ 保持旧记忆；$F=0,I=1$ 用候选内容替换旧记忆；$O=0$ 表示“记住但暂不输出”，并不删除 $C_t$。

候选 $\tilde C_t$ 虽被限制在 $(-1,1)$，记忆元 $C_t$ 却是跨时间加法累积的结果，并不保证仍在该区间。输出前再次使用 `tanh(C_t)`，一方面把对外状态限制在稳定范围，另一方面保留 $C_t$ 内部较宽的累积动态；输出门再选择暴露哪些维度。

```python
lstm = nn.LSTM(
    input_size=vocab_size,
    hidden_size=256,
    num_layers=1,
)

outputs, state = lstm(inputs)
H, C = state
# outputs: [T,B,H]
# H, C:   [num_layers,B,H]
```

LSTM 有输入、遗忘、输出和候选四组变换，循环部分的计算和参数量约为普通 RNN 的四倍。它提供更明确的长期存储和读取控制，但不意味着在所有数据集上必然优于更简单的 GRU。

## 9. 一套统一的 PyTorch 语言模型

RNN、GRU 和 LSTM 可以共用同一包装层。输入仍使用字符独热编码，以便与 D2L 从零实现保持一致：

```python
import torch
from torch import nn
import torch.nn.functional as F

class RecurrentLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256,
                 cell="gru", num_layers=1):
        super().__init__()
        cells = {
            "rnn": nn.RNN,
            "gru": nn.GRU,
            "lstm": nn.LSTM,
        }
        if cell not in cells:
            raise ValueError(f"unknown cell: {cell}")

        self.vocab_size = vocab_size
        self.rnn = cells[cell](
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, X, state=None):
        # X: [B,T] → one_hot: [T,B,V]
        inputs = F.one_hot(
            X.T, num_classes=self.vocab_size
        ).float()
        outputs, state = self.rnn(inputs, state)  # [T,B,H]
        logits = self.proj(outputs.reshape(-1, outputs.shape[-1]))
        return logits, state                    # [T*B,V]
```

训练时用第 6 节的 `train_epoch`。生成时先用真实前缀预热状态，再把每次采样结果反馈给模型：

```python
@torch.no_grad()
def generate(prefix, num_preds, model, vocab, device, alpha=1.0):
    if not prefix or alpha <= 0:
        raise ValueError("prefix不能为空且alpha必须为正")

    model.eval()
    state = None
    outputs = [vocab[prefix[0]]]

    def last_input():
        return torch.tensor(
            [[outputs[-1]]], device=device
        )

    # 输入真实前缀，但不把模型预测当作前缀内容
    for token in prefix[1:]:
        _, state = model(last_input(), state)
        outputs.append(vocab[token])

    for _ in range(num_preds):
        logits, state = model(last_input(), state)
        # q_i ∝ p_i**alpha，等价于softmax(alpha * logits)
        probs = torch.softmax(alpha * logits[-1], dim=0)
        next_token = torch.multinomial(probs, 1).item()
        outputs.append(next_token)

    return "".join(vocab.idx_to_token[i] for i in outputs)
```

`alpha=1` 按模型原分布采样；`alpha>1` 使分布更尖锐、更保守；`0<alpha<1` 增加多样性和错误概率；`alpha→∞` 接近 `argmax`。训练时输入真实上一词元，生成时输入模型自己的结果，所以自回归生成仍会累积错误。

## 10. 深层循环网络：同时沿时间和层传播

多层 RNN 在每个时间步把第 $l-1$ 层的输出送入第 $l$ 层，同时每一层都把自己的状态送到下一时间步：

$$
H_t^{(l)}=\phi_l\left(
H_t^{(l-1)}W_{xh}^{(l)}
+H_{t-1}^{(l)}W_{hh}^{(l)}+b_h^{(l)}
\right),
$$

其中 $H_t^{(0)}=X_t$。依赖方向可以写成：

```text
时间方向：H[t-1,l] → H[t,l]
层间方向：H[t,l-1] → H[t,l]
```

增加层数提升的是每个时间步的表示深度，并不自动延长模型能够稳定记忆的时间距离。更深网络还增加参数、激活内存和梯度路径，通常需要 dropout、谨慎初始化和更多调参。

```python
deep_lstm = nn.LSTM(
    input_size=vocab_size,
    hidden_size=256,
    num_layers=2,
    dropout=0.1,  # 作用于层与层之间；只有num_layers>1时有效
)
```

两层单向 LSTM 的 `H,C` shape 都是 `[2,B,H]`。从零实现时应为每一层维护独立状态：先在时间步 $t$ 更新第一层，再把其输出送入第二层；不能让两层错误共享同一个 $W_{hh}$ 或状态张量。

## 11. 双向循环网络：未来可见时才成立

双向 RNN 在同一序列上运行两条递归：

$$
\overrightarrow H_t=f(X_t,\overrightarrow H_{t-1}),
\qquad
\overleftarrow H_t=g(X_t,\overleftarrow H_{t+1}).
$$

第 $t$ 个位置的表示拼接两个方向：

$$
H_t=[\overrightarrow H_t;\overleftarrow H_t]\in\mathbb R^{B\times2h}.
$$

```python
encoder = nn.GRU(
    input_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=2,
    bidirectional=True,
)

outputs, state = encoder(embedded_tokens)
# outputs: [T,B,2H]
# state:   [2 * num_layers,B,H]
```

它适合整段输入已经可用的编码任务，例如文本分类、命名实体识别、缺词填充或机器翻译编码器。它不适合自回归“预测下一个词元”：训练第 $t$ 个输出时，反向状态已经看见 $x_{t+1}$ 及更后面的真实词元，标签发生泄漏；实际生成时未来不存在，训练与推理条件不一致。合理的训练困惑度也不能证明这种错误用法有效。

双向与深层是两个独立轴：`num_layers=2,bidirectional=True` 表示每层都有前后两个方向，状态第一维为 $2\times2=4$。方向数翻倍会增加循环计算、状态和输出通道；下一层的 `input_size` 也必须接收上一层拼接后的 $2H$ 特征，内置 API 会自动处理。

## 12. 怎样选择 RNN、GRU、LSTM、深层和双向结构

| 需求 | 合适起点 | 主要理由 | 主要代价或风险 |
|---|---|---|---|
| 教学、小数据、短期依赖 | 单层 RNN | 结构最简单、计算最少 | 长期梯度与记忆较弱 |
| 需要门控但重视简洁和速度 | GRU | 更新门提供长期直通路径，状态只有 $H$ | 循环计算约为 RNN 三倍 |
| 需要显式区分内部记忆与对外状态 | LSTM | $C$ 与 $H$ 分离，读写控制最完整 | 四组变换，参数和计算最多 |
| 单层容量不足 | 深层 GRU/LSTM | 增加每个时间步的表示深度 | 更慢、更耗内存、更难调参 |
| 当前位置允许使用完整左右上下文 | 双向 GRU/LSTM | 同时编码过去和未来 | 不能做在线因果生成，输出维翻倍 |
| 在线预测或自回归生成 | 单向 RNN/GRU/LSTM | 不使用未来信息，训练推理条件一致 | 时间方向难以完全并行 |

模型复杂度应服从任务的信息边界。门越多、层越深并不自动更好；先确认目标是否需要长期依赖、未来上下文是否真实可用，再比较验证集困惑度、吞吐量、参数量和生成质量。

## 13. 常见误区与调试顺序

- 把 `Counter(tokens)` 当成词表映射；它只统计频数，编号仍由 `Vocab` 决定。
- 把 `zip(corpus[:-1], corpus[1:])` 看成神秘语法；它只是把相邻切片配成二元组。
- 认为 `[1,2,3,4,5]` 的标签只应是 `6`；语言模型在每个时间步都提供下一词元监督。
- 认为顺序数据迭代器会返回所有可能滑动窗口；它只返回某个随机偏移下的非重叠分区。
- 看到循环内 `X_t:[B,V]` 就误以为把整段序列一起更新状态；它只是当前时间步的整个 batch。
- 认为 `logits:[T*B,V]` 和 `targets:[T*B]` shape 不同就不能算 loss；交叉熵的后一维是类别分数，标签只保存类别索引。
- 把困惑度取指数归因于独热标签；指数是平均负对数似然的反变换。
- 认为共享参数会让所有时间步产生相同结果；参数相同，但输入与历史状态不同。
- 认为 `detach(state)` 会清空记忆；它只切断梯度，状态数值仍传到下一批。
- 认为三维内置状态与二维从零状态冲突；前者只是显式增加了层数/方向维。
- 认为梯度裁剪也能解决梯度消失；它只限制过大的梯度范数。
- 把 GRU 重置门和更新门都理解成“旧内容占比”；前者控制候选如何读取历史，后者控制最终是否写入候选。
- 认为 LSTM 输入门先计算候选记忆；两者由同一输入并行计算，输入门控制候选写入 $C_t$ 的比例。
- 把双向网络直接用于下一词生成；这会在训练时泄漏未来标签。

遇到 shape 问题时，按下面顺序打印通常最快：

```python
print("tokens", X.shape)          # [B,T]
print("time major", X.T.shape)   # [T,B]
print("outputs", outputs.shape)  # [T,B,D*H]
print("state", state[0].shape if isinstance(state, tuple)
      else state.shape)            # [L*D,B,H]
print("logits", logits.shape)    # [T*B,V]
print("targets", Y.T.reshape(-1).shape)  # [T*B]
```

## 本篇知识链总结

循环神经网络的核心不是“把前一个输出再输入一次”，而是建立一条受任务信息边界约束的状态通路：文本被编号并切成错位的特征与标签，RNN 用共享转移函数压缩历史，交叉熵在每个时间步提供监督；顺序分区让状态值跨 batch 延续，`detach` 则把梯度限制在可计算的时间范围内。

普通 RNN 的长梯度链容易消失或爆炸。GRU 将“构造候选时读多少历史”和“最终写多少候选”分开，更新门提供直接保留状态的路径；LSTM 再把内部记忆与对外状态分离，用遗忘、输入、输出三道门管理保留、写入和读取。深层结构沿层方向增加表示能力，双向结构沿时间反方向增加未来上下文，但只有在完整序列真实可用时才合法。

最终可以把整章压缩成三个问题：

```text
数据问题：X、Y、时间、batch 和词表维怎样对齐？
优化问题：状态值需要保留多久，梯度需要反传多久？
建模问题：过去、未来和内部记忆中的哪些信息在任务中真实可用？
```

## 系列导航

- 上一篇：[计算机视觉任务全景：从数据增强、目标检测到语义分割与风格迁移]({{ '/deep-learning/computer-vision-tasks/' | relative_url }})
- [系列总览]({{ '/deep-learning/' | relative_url }})
- 下一篇：[D2L Seq2Seq 机器翻译：从双语数据、编码器—解码器到掩码损失与自回归生成]({{ '/deep-learning/seq2seq-machine-translation/' | relative_url }})

## 对应章节与参考资料

- D2L：[循环神经网络](https://zh.d2l.ai/chapter_recurrent-neural-networks/index.html)
- D2L：[现代循环神经网络](https://zh.d2l.ai/chapter_recurrent-modern/index.html)
- PyTorch：[`nn.RNN`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)、[`nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)、[`nn.LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- 原始论文：[LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)、[Bidirectional RNN](https://ieeexplore.ieee.org/document/650093)
