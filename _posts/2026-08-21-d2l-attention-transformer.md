---
layout: post
title: "D2L 注意力机制：从 QKV、多头自注意力到 Transformer"
date: 2026-08-21 15:00 +0800
tags: [D2L, PyTorch, 深度学习, NLP, 注意力机制, Self-Attention, Transformer]
toc: true
math: true
permalink: /deep-learning/attention-transformer/
---

## 本篇逻辑主线

基础 Seq2Seq 把整条源序列压进一个固定长度的最终状态，解码每个目标词元时都读取同一份上下文。注意力机制改变了这个接口：解码器先提出当前的查询（query），再用查询与一组键（key）计算匹配分数，经 softmax 得到权重，最后按权重汇聚与键一一配对的值（value）。Bahdanau 注意力让 RNN 解码器在每一步动态读取编码器的全部时间步；自注意力进一步让同一序列同时提供 Q、K、V；多头注意力让模型在多个可学习子空间中并行建立关系；Transformer 则完全用这些模块替代循环计算。

本文沿着下面的知识链整理 D2L `chapter_attention-mechanisms`：

```text
query提出“当前要找什么”
    ↓
score(query, key)衡量匹配程度
    ↓ masked softmax
attention weights：每个query对所有有效key的概率分布
    ↓ 与value批量矩阵乘
attention output：按需读取的信息
    ↓
Bahdanau：每个解码步动态读取源序列
    ↓
self-attention：Q、K、V来自同一序列
    ↓
multi-head：在多个投影子空间中并行建立关系
    ↓
positional encoding：补回自注意力缺少的顺序
    ↓
Transformer：编码器自注意力 + 解码器因果自注意力 + 交叉注意力
```

全文统一使用下面的 shape 记号：

| 符号 | 含义 |
|---|---|
| $B$ | batch size，独立样本数量 |
| $T_s$ | 源序列长度或 key-value 对数量 |
| $T_t$ | 目标序列长度或 query 数量 |
| $H$ | `num_hiddens`，Transformer 的总特征维度 |
| $h$ | `num_heads`，注意力头数 |
| $D_h=H/h$ | 每个注意力头的特征维度 |
| $V$ | 目标词表大小 |

贯穿全文最重要的检查方式不是记类名，而是始终回答四个问题：**Q、K、V 分别来自哪里；当前 softmax 沿哪个维度归一化；每个位置允许看见哪些 key；残差相加前两条支路的 shape 是否相同。**

### 对应 Notebook

| 主题 | Notebook | 本文位置 |
|---|---|---|
| QKV 框架与权重可视化 | `attention-cues.ipynb` | 第 1 节 |
| 核回归形式的注意力汇聚 | `nadaraya-waston.ipynb` | 第 2 节 |
| masked softmax、加性与点积评分 | `attention-scoring-functions.ipynb` | 第 3～4 节 |
| RNN Seq2Seq 的动态上下文 | `bahdanau-attention.ipynb` | 第 5 节 |
| 多头投影与 shape 变换 | `multihead-attention.ipynb` | 第 7 节 |
| 自注意力与位置编码 | `self-attention-and-positional-encoding.ipynb` | 第 6、8 节 |
| Transformer 编码器、解码器与训练 | `transformer.ipynb` | 第 9～13 节 |

## 1. Q、K、V：注意力到底在做什么

可以把注意力理解为“带着问题去查一组带标签的资料”：

- query 表示当前想找什么；
- key 是每份资料用于匹配查询的标签或地址；
- value 是匹配后真正要读取的内容。

给定第 $i$ 个 query 和第 $j$ 个 key，先计算标量分数 $s_{ij}$，再对同一个 query 的全部 key 做 softmax：

$$
\alpha_{ij}
=\frac{\exp(s_{ij})}{\sum_{k=1}^{T_s}\exp(s_{ik})}.
$$

最后对 value 加权求和：

$$
\mathbf{o}_i=\sum_{j=1}^{T_s}\alpha_{ij}\mathbf{v}_j.
$$

所以需要严格区分三个量：

```text
attention score   s_ij：softmax之前，可正可负，不要求和为1
attention weight  α_ij：softmax之后，非负且每个query对应的一行和为1
attention output  o_i：使用权重对value加权后的向量
```

若注意力权重矩阵为 `[B,T_t,T_s]`，元素 `weights[b,i,j]` 表示第 $b$ 个样本的第 $i$ 个 query 对第 $j$ 个 key 的参考程度。热力图的每一行对应一个 query，每一列对应一个 key；颜色越深，表示相应 value 对当前输出贡献通常越大。热力图展示的是“模型从哪里读信息”，不是预测值或误差。

## 2. Nadaraya-Watson：从距离分数到加权汇聚

Nadaraya-Watson 核回归给出了最容易观察的完整注意力。训练样本 $(x_i,y_i)$ 分别充当 key-value 对，新测试输入 $x$ 是 query：

$$
\hat y(x)=\sum_i\alpha(x,x_i)y_i.
$$

使用高斯核时，未归一化注意力分数是：

$$
s(x,x_i)=-\frac{1}{2}(x-x_i)^2.
$$

它确实是注意力分数，而不是最终权重。query 与 key 越近，分数越接近 0；距离越远，分数越负。对所有 $i$ 做 softmax 后才得到权重：

$$
\alpha(x,x_i)=\operatorname{softmax}_i
\left(-\frac{1}{2}(x-x_i)^2\right).
$$

D2L 的带参数版本进一步学习宽度参数 $w$：

$$
s(x,x_i)=-\frac{1}{2}\big((x-x_i)w\big)^2.
$$

$|w|$ 越大，距离差异在 softmax 前被放得越大，权重越集中；$|w|$ 越小，模型会平均参考更宽范围的样本。这个例子建立了之后所有注意力模块的共同模板：**评分函数决定怎样匹配，softmax 把分数变成概率分布，value 的维度决定汇聚输出的维度。**

## 3. masked softmax：哪些 key 根本不应参与竞争

文本 batch 为了对齐长度会补 `<pad>`，解码器训练还必须屏蔽未来词元。若无效位置直接参加 softmax，它们也会分走概率，因此要先把这些位置的分数替换成极大负数：

```python
def masked_softmax(X, valid_lens):
    # X: [B, num_queries, num_keys]
    if valid_lens is None:
        return torch.softmax(X, dim=-1)

    shape = X.shape
    if valid_lens.dim() == 1:
        # [B] -> [B * num_queries]
        # 同一样本的所有query共用一个有效key数量。
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        # [B, num_queries] -> [B * num_queries]
        # 每个query可以拥有不同的可见范围。
        valid_lens = valid_lens.reshape(-1)

    # 合并batch与query维，每一行对应一个query的全部key分数。
    X = X.reshape(-1, shape[-1])
    X = d2l.sequence_mask(X, valid_lens, value=-1e6)

    # exp(-1e6)约等于0，无效位置的注意力权重也约等于0。
    return torch.softmax(X.reshape(shape), dim=-1)
```

对二维矩阵，`softmax(dim=1)` 和 `sum(dim=1)` 都沿同一个轴工作：它们都在每一行的列元素之间计算。`sum(dim=0)` 才是从上到下把不同行相加，得到每一列的和。对注意力分数 `[B,Q,K]`，`dim=-1` 就是沿 key 维归一化，因此每个 query 在所有有效 key 上的权重之和为 1。

## 4. 加性注意力与缩放点积注意力

### 4.1 加性注意力：用一个小 MLP 比较 Q 和 K

当 query 和 key 的原始特征维度不相同时，可以分别投影到同一个隐藏空间：

$$
s(\mathbf q,\mathbf k)
=\mathbf w_v^\top\tanh(W_q\mathbf q+W_k\mathbf k).
$$

这里的 `num_hiddens` 是**注意力评分网络内部的隐藏维度**，不是 RNN 时间状态的同义词。假设：

```text
queries: [2,1,20]  两个样本，每个样本1个query，每个query为20维
keys:    [2,10,2]  两个样本，每个样本10个key，每个key为2维
values:  [2,10,4]  每个key对应一个4维value
```

若 `num_hiddens=8`，则：

```text
W_q(queries): [2,1,8]
W_k(keys):    [2,10,8]
广播组合：    [2,1,10,8]
w_v打分：     [2,1,10]
softmax：     [2,1,10]
weights @ V: [2,1,4]
```

`keys` 最后的 `2` 是每一个 key 的特征维度，不是 key 的数量；key 的数量是中间的 `10`。key 与 value 必须在数量维一一配对，但二者的特征维可以不同。最终输出最后一维由 value 的维度 `4` 决定。

### 4.2 缩放点积注意力：矩阵乘法友好的评分函数

若 Q 和 K 已具有相同特征维度 $d$，可用：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V.
$$

除以 $\sqrt d$ 是为了控制点积方差。维度增大时，未经缩放的点积绝对值容易变大，使 softmax 过早饱和到近似 one-hot，梯度也随之变弱。

二者的主要区别是匹配假设：Nadaraya-Watson 使用“距离是否接近”，缩放点积使用“投影后的方向和幅度是否匹配”。加性注意力允许 Q、K 原始维度不同且评分网络更灵活；缩放点积可以直接使用高效矩阵乘法，因此成为 Transformer 的标准选择。

## 5. Bahdanau 注意力：不再把整句压进同一个上下文

基础 RNN Seq2Seq 在每个解码步都使用同一个编码器最终状态。Bahdanau 注意力改为：

$$
\mathbf c_{t'}
=\sum_{t=1}^{T_s}
\alpha(\mathbf s_{t'-1},\mathbf h_t)\mathbf h_t.
$$

对应关系是：

```text
query：上一解码步顶层隐藏状态 s_(t'-1)  [B,1,H]
keys： 编码器所有源时间步输出          [B,Ts,H]
values：同一组编码器输出                [B,Ts,H]
context：当前解码步动态汇聚的上下文      [B,1,H]
```

`hidden_state` 的形状为 `[num_layers,B,H]`。`hidden_state[-1]` 选择最后一层并消去层维，得到 `[B,H]`；再 `unsqueeze(1)` 才得到一个 query 的 `[B,1,H]`。这个状态没有 `num_steps`，因为 PyTorch GRU 把两种结果分开返回：`output:[T,B,H]` 保存顶层的所有时间步，`hidden_state:[L,B,H]` 保存所有层的最后时间步。

当前目标词嵌入 `[B,1,E]` 与 context `[B,1,H]` 沿最后一维拼接，所以 GRU 每步输入是 `[1,B,E+H]`。这里的 `E+H` 属于 Bahdanau RNN 解码器；Transformer 不会把 embedding 与 hidden state 这样拼接。

## 6. 自注意力：同一序列既提问，也提供答案

自注意力令 Q、K、V 都来自同一个输入 $X$：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

“都来自 $X$”不等于三者数值相同。三套独立参数把同一词元投影成三种角色：Q 描述它要寻找什么，K 描述它用什么特征接受匹配，V 描述它真正提供什么信息。

对于 `X:[B,T,H]`，自注意力输出仍是 `[B,T,H]`。每个位置都可直接读取任意其他位置，因此最大信息路径长度是 $O(1)$，而 RNN 需要沿时间走 $O(T)$ 步；所有位置也能并行计算。但注意力矩阵为 `[T,T]`，标准自注意力的时间与主要注意力内存复杂度随序列长度二次增长，核心计算量为 $O(T^2H)$。

仅有自注意力时，若同时置换输入顺序和输出位置，计算关系也会跟着置换；模型本身不知道“第一个词”和“第三个词”的区别。因此 Transformer 还必须注入位置信息。

## 7. 多头注意力：重点不是复制，而是学习多个关系子空间

### 7.1 `num_hiddens` 为什么还要拆成多个头

在多头注意力中，`num_hiddens=H` 是 Q、K、V 投影后的**总特征宽度**，通常也等于 Transformer 的模型维度 $d_{model}$。标准实现先用大矩阵一次得到全部头的特征，再按头拆分：

```text
W_q(X): [B,T,H]
         ↓ H = h × D_h
        [B,T,h,D_h]
```

这不是随意切原始 embedding。`W_q/W_k/W_v` 在切分之前已经学习如何把适合不同头的特征组织到不同区间。每个头随后独立产生一套 `[T_t,T_s]` 注意力分布，可以分别学习局部搭配、长距离依赖、句法关系或指代关系。

标准设计令 $D_h=H/h$，所以 $h$ 个头拼回后仍为 $H$。若每个头都使用完整的 $H$ 维，计算量、参数量和拼接宽度都会随头数成倍增加。固定总宽度可以在近似相同预算下获得多个注意力视角，并要求 `H % h == 0`。

### 7.2 `transpose_qkv` 到底改变了什么

设 `B=2,T=4,H=8,h=2`，则每头维度 `D_h=4`：

```text
[2,4,8]
   reshape → [2,4,2,4]  # [B,T,h,D_h]
   permute → [2,2,4,4]  # [B,h,T,D_h]
   reshape → [4,4,4]    # [B*h,T,D_h]
```

对应实现是：

```python
def transpose_qkv(X, num_heads):
    # [B,T,H] -> [B,T,h,D_h]
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # [B,T,h,D_h] -> [B,h,T,D_h]
    X = X.permute(0, 2, 1, 3)

    # 把“样本编号+头编号”折叠成独立批量项。
    return X.reshape(-1, X.shape[2], X.shape[3])
```

合并 batch 和 head 不会混合不同样本。新的第一维只是：

```text
0 = 样本0的头0
1 = 样本0的头1
2 = 样本1的头0
3 = 样本1的头1
```

`torch.bmm` 对第一维逐项独立计算，只会执行 `Q[p] @ K[p].T`，不会让 `Q[0]` 与 `K[2]` 相乘。这样无需 Python 循环，就能把所有样本的所有头作为一个大 batch 并行处理。`valid_lens` 也按每个样本重复 $h$ 次，保持同样的对应顺序。

注意力完成后，`transpose_output` 执行逆变换：

```text
[B*h,T,D_h] -> [B,h,T,D_h] -> [B,T,h,D_h] -> [B,T,H]
```

### 7.3 为什么 `W_o` 输入输出都是 `num_hiddens`

拼接多个头只得到：

```text
[head_0 | head_1 | ... | head_(h-1)]
```

`W_o` 用一个可学习线性层融合不同头的信息：

$$
\operatorname{MHA}(Q,K,V)
=\operatorname{Concat}(head_1,\ldots,head_h)W_O.
$$

输入和输出都为 $H$ 不代表它是恒等映射；$H\times H$ 权重会重新组合所有头的特征。输出保持 $H$ 的主要原因是之后需要残差相加：

$$
Y=X+\operatorname{MHA}(X,X,X).
$$

技术上可以把 `W_o` 改成 `Linear(H,H_out)`，但若 $H_{out}\ne H$，残差分支也必须投影，后续 LayerNorm、FFN 和下一层输入维度都要同步修改。

## 8. 位置编码：相加发生在模块内部

D2L 使用固定正弦—余弦位置编码：

$$
p_{i,2j}=\sin\left(\frac{i}{10000^{2j/H}}\right),\qquad
p_{i,2j+1}=\cos\left(\frac{i}{10000^{2j/H}}\right).
$$

Transformer 中这一行：

```python
X = self.pos_encoding(
    self.embedding(X) * math.sqrt(self.num_hiddens)
)
```

展开后是：

```python
token_vectors = self.embedding(X)                # [B,T,H]
token_vectors *= math.sqrt(self.num_hiddens)
X = token_vectors + self.P[:, :T, :]             # 广播到[B,T,H]
X = self.dropout(X)
```

相加写在 `PositionalEncoding.forward()` 内部。乘 $\sqrt H$ 是为了让初始 embedding 的尺度与取值在 $[-1,1]$ 的固定位置编码更匹配，避免位置信号在相加时压过内容表示。

这个 `P` 不是 `nn.Parameter`，所以位置编码本身不会被梯度更新；模型学习的是怎样通过后续 Q、K、V 投影利用它。若改用 `nn.Embedding(max_len,H)` 存储位置向量，位置编码本身也可以学习。相加不改变 shape：

```text
token embedding [B,T,H]
+ position       [1,T,H]
= Transformer X [B,T,H]
```

正余弦方案同时提供绝对位置和可线性表达的相对偏移：固定偏移 $\delta$ 对每一对 sin/cos 分量都对应一个与位置 $i$ 无关的二维旋转矩阵。

## 9. Transformer 的总维度契约

Transformer 把模型宽度统一为 $H$：

```text
token ids              [B,T]
embedding + position   [B,T,H]
multi-head attention   [B,T,H]
AddNorm                [B,T,H]
position-wise FFN      [B,T,H_ff] -> [B,T,H]
AddNorm                [B,T,H]
```

这里没有 `embed_size + num_hiddens`。那是 Bahdanau RNN 解码器把词嵌入与 context 沿特征维拼接的结果。Transformer 的 embedding 直接使用 `num_hiddens` 维，位置编码是相加，注意力与 FFN 也都回到 $H$，以便每个子层都满足残差连接的 shape 要求。

`nn.Linear` 可以直接接收三维张量，因为它总是变换最后一维并保留前置维：

```text
Linear(H,H_ff): [B,T,H]    -> [B,T,H_ff]
Linear(H_ff,H): [B,T,H_ff] -> [B,T,H]
```

它等价于对每个 `[b,t,:]` 向量独立使用同一组参数，不会混合 batch 或词元位置。

## 10. 编码器：让每个源词元读取完整源序列

一个 D2L `EncoderBlock` 包含两个子层：

```python
def forward(self, X, valid_lens):
    # Q=K=V=X：编码器多头自注意力。
    attn = self.attention(X, X, X, valid_lens)  # [B,Ts,H]
    Y = self.addnorm1(X, attn)                  # [B,Ts,H]

    # 每个位置独立通过同一个两层MLP。
    ffn_out = self.ffn(Y)                       # [B,Ts,H]
    return self.addnorm2(Y, ffn_out)            # [B,Ts,H]
```

`addnorm1` 和 `addnorm2` 的结构一样，但不是同一个模块。二者的 LayerNorm 分别拥有独立的缩放参数 $\gamma$ 和平移参数 $\beta$：第一套适配注意力子层的分布，第二套适配 FFN 子层的分布。强行复用会让两个位置共享参数，形成标准 Transformer 没有的约束。

D2L 的 `AddNorm` 是原论文的 Post-LN 形式：

$$
\operatorname{LayerNorm}(X+\operatorname{Dropout}(\operatorname{Sublayer}(X))).
$$

训练配置中 `norm_shape=[H]`，所以对每个词元的 $H$ 个特征计算均值和方差，不跨 batch，也不跨时间。`normalized_shape` 不是输出 shape；它指定输入末尾哪些维度一起参与归一化。

完整编码器先执行 embedding、缩放与位置编码，再堆叠多个 EncoderBlock。最终只返回：

```text
enc_outputs: [B,Ts,H]
```

Transformer 没有 RNN 循环状态，因此不返回 `[L,B,H]` 的 `hidden_state`。整个源序列的最终表示就是提供给解码器的 memory。

## 11. 解码器：三个子层解决三个不同问题

每个 DecoderBlock 有三个子层：

```text
目标表示X
  ↓ 掩蔽多头自注意力：只能读取已生成的目标前缀
AddNorm1
  ↓ 编码器—解码器交叉注意力：读取完整源序列
AddNorm2
  ↓ 逐位置FFN：变换每个目标位置的通道特征
AddNorm3
```

### 11.1 `state` 从哪里来

外层 `EncoderDecoder` 先执行编码器，再调用解码器的 `init_state`：

```python
enc_outputs = encoder(enc_X, enc_valid_lens)  # [B,Ts,H]
state = decoder.init_state(enc_outputs, enc_valid_lens)
```

TransformerDecoder 将其组织为：

```python
state = [
    enc_outputs,                 # [B,Ts,H]
    enc_valid_lens,              # [B]
    [None] * num_decoder_layers  # 每层的目标侧历史缓存
]
```

`enc_valid_lens` 不是编码器额外返回的状态，而是数据 batch 原本提供的源序列有效长度；外层接口把它同时交给编码器掩码和解码器初始化。Encoder 返回的变量虽然名为 `X`，被调用方接住后就是 `enc_outputs`。

### 11.2 训练：全部目标位置并行，但必须使用因果掩码

训练时目标输入为 `[B,T_t,H]`，所有位置一次进入解码器。第 $i$ 个位置不能看到 $i$ 之后的真实目标词，否则会泄漏答案，因此构造：

```text
dec_valid_lens[b] = [1,2,...,T_t]
```

它让第一个 query 只看第一个 key，第二个看前两个，以此类推。此时：

```text
masked self-attention:
Q = X                         [B,Tt,H]
K = V = X                     [B,Tt,H]
weights                       [B,h,Tt,Tt]

cross-attention:
Q = decoder representation    [B,Tt,H]
K = V = enc_outputs           [B,Ts,H]
weights                       [B,h,Tt,Ts]
```

### 11.3 预测：query 只有当前词元，K/V 缓存完整前缀

自回归预测一次只输入一个新词元，当前 `X:[B,1,H]`。每个解码器块把它接到自己的历史缓存：

```text
第1步cache：[B,1,H]
第2步cache：[B,2,H]
第3步cache：[B,3,H]
```

于是当前 query 可以读取全部过去目标表示。未来词元根本还不存在，因此 D2L 在预测分支令 `dec_valid_lens=None`。每层必须维护独立 cache，因为不同层保存的是不同表示层次上的目标状态。

训练时 cache 不会无限增长：每个 batch 都重新执行 `init_state`，每个 DecoderBlock 只对完整目标序列调用一次，对应 cache 只是 `[B,T_t,H]`。下一个 batch 会重新得到 `[None]*num_layers`。逐步增长只发生在自回归预测中，它让当前 query 能访问完整的已生成前缀。D2L 的教学实现缓存的是每层历史表示；更优化的推理实现还会直接缓存投影后的 K/V，避免每一步重新投影全部历史。

## 12. TransformerDecoder.forward 的端到端数据流

可以把解码器压缩成下面的 shape 流程：

```text
目标词元编号              [B,Tt]
  ↓ Embedding × sqrt(H)
目标向量                  [B,Tt,H]
  ↓ + positional encoding
带位置的目标表示          [B,Tt,H]
  ↓ DecoderBlock × L
解码器最终表示            [B,Tt,H]
  ↓ Linear(H,V)
目标词表logits            [B,Tt,V]
```

编码器输出没有初始化某个循环隐藏状态，而是在每个解码器块的交叉注意力中反复作为 K 和 V：

```python
Y2 = attention2(
    queries=Y,
    keys=enc_outputs,
    values=enc_outputs,
    valid_lens=enc_valid_lens,
)
```

残差连接中的 `X` 是进入当前子层、尚未经过内部 WQKV 的原始表示。内部计算会创建 `Q=W_q(X)`、`K=W_k(X)`、`V=W_v(X)`，但不会覆盖旁路上的 `X`。标准 Transformer 让旁路和注意力输出都为 $H$ 维，所以可以直接相加。

## 13. 如何读 Transformer 的注意力权重

多头实现内部为批量矩阵乘而把 batch 与 head 合并，权重物理 shape 常为：

```text
[B*h, num_queries, num_keys]
```

恢复逻辑维度后是：

```text
[B, h, num_queries, num_keys]
```

不同模块的矩阵语义如下：

| 模块 | query 轴 | key 轴 | 典型 shape |
|---|---|---|---|
| 编码器自注意力 | 源词元位置 | 源词元位置 | `[B,h,Ts,Ts]` |
| 解码器掩蔽自注意力 | 目标词元位置 | 目标前缀位置 | `[B,h,Tt,Tt]` |
| 解码器交叉注意力 | 目标词元位置 | 源词元位置 | `[B,h,Tt,Ts]` |

编码器权重会屏蔽源序列 `<pad>`；解码器自注意力热力图的未来区域应为 0，形成下三角结构；交叉注意力则可以观察每个目标位置主要读取了哪些源位置。不同头的权重不同，正是多头设计希望学习到的多种关系。

注意力权重是理解模型数据流的工具，但不能自动等同于因果解释：后续还有 value 投影、`W_o`、残差、FFN 和多层组合，单张热力图不能完整证明某个词元对最终预测的因果贡献。

## 14. 自注意力、RNN 与卷积怎样选择

设序列长度为 $T$、表示维度为 $H$、卷积核宽度为 $k$：

| 架构 | 单层主要复杂度 | 顺序操作 | 最长信息路径 | 主要特点 |
|---|---:|---:|---:|---|
| RNN | $O(TH^2)$ | $O(T)$ | $O(T)$ | 因果状态自然，但时间并行困难 |
| 一维卷积 | $O(kTH^2)$ | $O(1)$ | 堆叠后约 $O(T/k)$ | 局部先验强，可并行 |
| 自注意力 | $O(T^2H)$ | $O(1)$ | $O(1)$ | 任意位置直接交互，但长序列二次增长 |

Transformer 的优势不是“所有情况下都更省计算”，而是消除了 RNN 的时间依赖链，让每层所有位置并行，并把远距离信息路径缩到一步。序列很长时，`T×T` 分数矩阵会成为时间和显存瓶颈；这也是稀疏、局部、线性注意力以及 KV cache 等后续研究的出发点。

## 15. 常见误区

- 把 attention score 和 attention weight 当成同一个量；前者还没 softmax，后者才是概率分布。
- 把 `softmax(dim=1)` 与 `sum(dim=1)`说成一个按行、一个按列；二者都沿第 1 维操作。
- 把 `[B,10,2]` 中最后的 `2` 当成 key 数量；它是每个 key 的特征维，key 数量是 `10`。
- 混淆 batch size 与 query 数量；batch 是独立样本数，query 数是每个样本内部要查询多少次，注意力不会跨 batch 匹配。
- 认为 `num_hiddens` 永远指 RNN 状态；在加性注意力中它是评分 MLP 宽度，在 Transformer 中通常是总模型宽度 $H$。
- 认为多头只是复制同一注意力；每个头拥有不同的 Q/K/V 投影子空间和独立注意力分布。
- 看到 `[B*h,T,D_h]` 就认为样本混在一起；批量矩阵乘不会跨第一维计算。
- 认为 `W_o:H→H` 没有作用；它会学习融合所有头，只是为残差连接保持 shape。
- 认为位置编码与 embedding 拼接；D2L 中二者相加，shape 始终为 `[B,T,H]`。
- 认为固定正余弦位置编码也被优化器学习；固定 `P` 不更新，模型学习的是如何使用它。
- 认为三维张量不能直接传给 `nn.Linear`；Linear 只变换最后一维，保留 batch 与时间维。
- 认为 Encoder 返回的 `X` 不是 `enc_outputs`；它只是被外层调用者换了变量名。
- 认为 Transformer 解码器需要 GRU 风格的 `hidden_state`；它通过编码器 memory、目标侧 self-attention 和每层 cache 传递信息。
- 认为训练时目标 cache 会跨 batch 无限增长；每次 `init_state` 都重新创建缓存。
- 认为两个 AddNorm 可以共享；它们分别拥有独立 LayerNorm 参数，适配不同子层分布。
- 把 `normalized_shape` 当成输出 shape；它只指定 LayerNorm 对输入最后哪些维度归一化。

## 本篇知识链总结

注意力机制把固定汇聚改成查询驱动的动态读取。评分函数先比较 Q 与 K，masked softmax 决定哪些 key 有资格参与并把分数转成概率，矩阵乘再用这些概率汇聚 V。Bahdanau 注意力借此让每个 RNN 解码步读取不同源位置；自注意力则让同一序列内部的每个词元都能查询其他词元。

多头注意力先通过可学习的 WQKV 把总宽度 $H$ 组织成 $h$ 个子空间，再用 `transpose_qkv` 把“样本—头”组合折叠进 batch 维并行计算，最后恢复、拼接并用 $W_O$ 融合。位置编码补回顺序，残差连接要求所有子层回到相同的 $H$ 维，LayerNorm 和逐位置 FFN 让这些模块可以稳定堆叠。

Transformer 编码器让每个源位置通过多头自注意力读取完整源序列；解码器先用因果自注意力读取目标前缀，再用交叉注意力把当前目标 query 对齐到编码器 memory。训练时用 `[1,2,...,T_t]` 掩码并行处理所有目标位置，预测时用每层 cache 保存已经生成的目标表示。最终 `[B,T_t,H]` 经线性层变成 `[B,T_t,V]` logits，仍然落回 Seq2Seq 的条件生成目标。

整章可以压缩成四个调试问题：

```text
来源：当前Q、K、V各自来自哪个序列或哪一层？
形状：batch、head、query、key、feature分别在哪个维度？
可见性：padding和未来位置是否在softmax前被正确掩蔽？
接口：注意力、FFN、残差与最终词表投影的最后一维是否对齐？
```

## 系列导航

- 上一篇：[计算机视觉任务全景：从数据增强、目标检测到语义分割与风格迁移]({{ '/deep-learning/computer-vision-tasks/' | relative_url }})
- [系列总览]({{ '/deep-learning/' | relative_url }})
- 下一篇：无（本篇是当前已发布系列终点）

## 对应章节与参考资料

- D2L：[注意力机制](https://zh.d2l.ai/chapter_attention-mechanisms/index.html)，覆盖注意力提示、Nadaraya-Watson 核回归、评分函数、Bahdanau 注意力、多头注意力、自注意力、位置编码与 Transformer。
- PyTorch：[`nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)、[`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)、[`nn.LayerNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)、[`nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- Bahdanau、Cho、Bengio：[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Vaswani 等：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
