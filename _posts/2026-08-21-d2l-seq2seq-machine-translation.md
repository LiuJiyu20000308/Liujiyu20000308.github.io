---
layout: post
title: "D2L Seq2Seq 机器翻译：从双语数据、编码器—解码器到掩码损失与自回归生成"
date: 2026-08-21 09:00 +0800
tags: [D2L, PyTorch, 深度学习, NLP, Seq2Seq, 机器翻译, 编码器-解码器]
toc: true
math: true
permalink: /deep-learning/seq2seq-machine-translation/
---

## 本篇逻辑主线

语言模型把一条序列向后错开一个词元，用同一种语言的历史预测下一个词；机器翻译却要把一条源语言序列映射成长度可能完全不同的目标语言序列。Seq2Seq 因此必须同时解决四件事：怎样把变长源句压缩成可传递的状态，解码器为什么除了编码结果还需要目标侧输入，填充位置为什么不能进入损失，以及训练时能看到真实目标词、预测时只能看到自身输出所造成的差异。

本文沿着下面的数据流整理 D2L 的 RNN Seq2Seq 实现：

```text
英法平行句对
    ↓ 词元化、双词表、<eos>、<pad>、valid_len
源词元 [B,Ts] ──Embedding──> 源向量 [Ts,B,E]
    ↓
GRU Encoder ──> 全部输出 [Ts,B,H] + 最终状态 [L,B,H]
                                              ↓ 初始化
目标前缀 [B,Tt] ──Embedding──> GRU Decoder ──> logits [B,Tt,Vt]
                                              ↓
真实目标 [B,Tt] ──valid_len 掩码──> 交叉熵
                                              ↓
训练：teacher forcing；预测：<bos> 后逐词自回归，遇 <eos> 停止
                                              ↓
BLEU：短句惩罚 + n 元语法精确率
```

贯穿全文需要分清三个边界：**源序列和目标序列是两条不同的时间轴；状态值与当前词元输入承担不同职责；训练的并行解码与预测的逐步解码使用不同信息。**

本文承接上一篇的 RNN、GRU、隐状态和交叉熵知识，集中解释 RNN Seq2Seq。注意力、束搜索和 Transformer 只在边界处说明，不在本文展开。

### 对应 Notebook

| 主题 | Notebook | 本文位置 |
|---|---|---|
| 双语数据、双词表、截断与填充 | `machine-translation-and-dataset.ipynb` | 第 1～2 节 |
| 编码器—解码器通用接口 | `encoder-decoder.ipynb` | 第 3 节 |
| RNN 编码器和解码器 | `seq2seq.ipynb` | 第 4～5 节 |
| 掩码损失与强制教学 | `seq2seq.ipynb` | 第 6～8 节 |
| 贪心预测与 BLEU | `seq2seq.ipynb` | 第 9～11 节 |
| 练习中的结构改进 | `seq2seq.ipynb` | 第 12 节 |

## 1. 机器翻译的数据契约：两条变长序列怎样组成一个 batch

设批量大小为 $B$，源序列最大长度为 $T_s$，目标序列最大长度为 $T_t$。数据迭代器返回四个张量：

```text
X:           [B,Ts]   源语言词元编号
X_valid_len: [B]      每条源句的有效长度
Y:           [B,Tt]   目标语言词元编号
Y_valid_len: [B]      每条目标句的有效长度
```

源语言和目标语言分别建立词表，因为同一整数在两个词表中没有共同含义：

```text
src_vocab[5] 可能是 "go"
tgt_vocab[5] 可能是 "je"
```

预处理先把每个句子的词元转换成编号，再添加结束标记：

```python
def build_array_nmt(lines, vocab, num_steps):
    # 每个词元列表转换成整数编号列表。
    lines = [vocab[line] for line in lines]

    # 新建列表，并在每条序列末尾添加句子结束标记。
    lines = [line + [vocab['<eos>']] for line in lines]

    # 截断或填充为统一长度，得到[B,T]。
    array = torch.tensor([
        truncate_pad(line, num_steps, vocab['<pad>'])
        for line in lines
    ])

    # <pad>以外的位置才算有效词元，得到[B]。
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(dim=1)
    return array, valid_len
```

这里两次赋值给 `lines` 不是重复操作：第一次完成“词元到编号”，第二次完成“追加结束标记”。旧的中间结果之后不再需要，所以复用变量名。

### 1.1 `num_steps` 直接截断并不总是合适

当前实现先添加 `<eos>`，再执行 `line[:num_steps]`。若序列过长，句尾语义和 `<eos>` 都可能被截掉：目标侧无法从该样本学习正确的停止位置，源侧和目标侧分别截断还可能破坏平行句对的语义对齐。

教学数据可以用固定截断简化 shape，但真实任务应先统计长度分布，再选择能覆盖大多数句子的上限。更稳妥的策略包括：

- 按长度分桶，每个 batch 只填充到本批最长序列；
- 过滤整对超长平行句，而不是只截一侧；
- 把长段落拆成语义完整的短句；
- 在显存允许时提高最大长度。

写成 `line[:num_steps - 1] + [eos]` 只能保证张量末尾有结束标记，无法恢复已删除的语义，甚至可能让模型学习“不完整翻译也应立即结束”。填充本身不会丢信息，只要后续用 `valid_len` 或布尔掩码排除 `<pad>`。

## 2. `vocab_size`、`embed_size` 和 `num_hiddens` 分别控制什么

三个常见维度不能混为一谈：

| 名称 | 含义 | 来源或选择方式 |
|---|---|---|
| `vocab_size`，记为 $V$ | 词表中有多少个词元 | 由语料、词元化和最低频率决定 |
| `embed_size`，记为 $E$ | 每个词元向量有多少维 | 模型超参数 |
| `num_hiddens`，记为 $H$ | GRU 隐状态有多少维 | 模型超参数 |

```python
embedding = nn.Embedding(vocab_size, embed_size)
```

内部维护一个可训练矩阵：

```text
embedding.weight: [V,E]
```

输入整数编号 `i` 就是查询第 `i` 行。对批量序列：

```text
输入编号：[B,T]
嵌入输出：[B,T,E]
```

这等价于用词元的独热向量选择权重矩阵的一行：

$$
e_i^T E_{	ext{table}}=E_{	ext{table}}[i,:].
$$

`nn.Embedding` 不记录词频；词频由 `Counter` 或词表对象保存。高频词只是在训练中被查询和更新得更频繁。创建嵌入层时 PyTorch 已自动初始化 `embedding.weight`，且它是 `requires_grad=True` 的模型参数。即使本节的自定义 Xavier 初始化函数只显式处理 `Linear` 和 `GRU`，嵌入矩阵仍保留自己的默认随机初始化，并通过

```python
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
```

与其他参数一起训练。

## 3. 为什么 `EncoderDecoder` 同时需要 `enc_X` 和 `dec_X`

通用接口只有几行：

```python
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        # enc_X：源语言序列，回答“要翻译什么”。
        enc_outputs = self.encoder(enc_X, *args)

        # 把编码器输出转换成解码器初始状态。
        dec_state = self.decoder.init_state(enc_outputs, *args)

        # dec_X：目标语言前缀，回答“目标句已经生成了什么”。
        return self.decoder(dec_X, dec_state)
```

只有 `enc_X` 不能完整决定每一步解码计算，因为目标序列按链式法则分解为

$$
P(y_1,ldots,y_{T_t}\mid x_1,ldots,x_{T_s})
=\prod_{t=1}^{T_t}P(y_t\mid y_1,ldots,y_{t-1},c),
$$

其中 $c$ 是源句的编码信息。解码器不仅需要知道源句说了什么，还需要知道目标句此前已经出现了哪些词元。

训练时，`dec_X` 是向右错开一位的真实目标序列；预测时，它先是 `<bos>`，之后是模型上一时刻的预测。源语言和目标语言的词序、长度都可能不同，因此不存在“编码器第 $t$ 步状态固定对应解码器第 $t$ 步”的规则。

## 4. 编码器：从词元编号到最终状态

RNN 编码器的任务是把变长源句压缩进状态。设层数为 $L$：

```python
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size, num_hiddens, num_layers,
            dropout=dropout
        )

    def forward(self, X):
        # [B,Ts] -> [B,Ts,E]：每个编号查询一个嵌入向量。
        X = self.embedding(X)

        # GRU默认采用时间优先布局：[B,Ts,E] -> [Ts,B,E]。
        X = X.permute(1, 0, 2)

        # 未显式传初始状态时，PyTorch创建全0状态。
        output, state = self.rnn(X)

        # output：[Ts,B,H]，保存最后一层每个时间步的输出。
        # state： [L,B,H]，保存每一层在最后时间步的状态。
        return output, state
```

在单向 GRU 中，`output[-1]` 等于最后一层的最终状态 `state[-1]`；但两者的整体含义不同：`output` 沿时间保留最后一层的轨迹，`state` 沿层保留每一层的最终状态。

本节的基础解码器只使用 `state`。这把整个源句压进固定大小 `[L,B,H]`，长句信息容易成为瓶颈；注意力机制的核心改进正是让解码器重新访问 `output` 中各个源时间步，而不是只依赖最终状态。

## 5. 解码器：初始状态来自编码器，后续状态属于解码器

基础解码器接收目标词元和状态：

```python
class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 每一步输入由目标词嵌入E和上下文H拼接而成。
        self.rnn = nn.GRU(
            embed_size + num_hiddens,
            num_hiddens,
            num_layers,
            dropout=dropout
        )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # [L,B,H]：编码器每一层的最终状态。
        return enc_outputs[1]

    def forward(self, X, state):
        # 目标编号[B,Tt] -> 嵌入[B,Tt,E] -> 时间优先[Tt,B,E]。
        X = self.embedding(X).permute(1, 0, 2)

        # state[-1]为顶层状态[B,H]；沿目标时间复制成[Tt,B,H]。
        context = state[-1].repeat(X.shape[0], 1, 1)

        # 每一步同时输入目标词嵌入和上下文：[Tt,B,E+H]。
        X_and_context = torch.cat((X, context), dim=2)

        # 传入的state只负责初始化；GRU内部沿目标时间轴持续更新状态。
        output, state = self.rnn(X_and_context, state)

        # [Tt,B,H] -> [Tt,B,Vt] -> [B,Tt,Vt]。
        logits = self.dense(output).permute(1, 0, 2)
        return logits, state
```

这里传入的 `state` 不是一组“与解码器各时间步对应的编码器状态”。第一次解码时有

```text
S_dec,0 = H_enc,final
```

之后是

```text
S_dec,1 = GRU(dec_X1, S_dec,0)
S_dec,2 = GRU(dec_X2, S_dec,1)
...
```

源时间 $T_s$ 和目标时间 $T_t$ 是两条独立轴。编码器最终状态只初始化解码器，后续状态由解码器自己更新。

### 5.1 D2L 这段上下文代码的调用边界

训练时一次把完整 `dec_X:[B,Tt]` 交给解码器，`context` 在进入 GRU 前一次性从编码器状态复制，所以整段目标序列使用固定的编码上下文。

预测时每次只输入一个词元，并把返回的新 `state` 再传入。按上面的原始写法，第二次调用起 `state[-1]` 已是解码器状态，因此重新构造的 `context` 不再是原始编码器最终状态。模型仍能依靠递归状态携带源信息，但这与“固定编码上下文在每一步拼接”的理论描述不完全相同。

若要严格固定上下文，可以把“会更新的解码器状态”和“不会更新的编码上下文”分开保存：

```python
def init_state(self, enc_outputs, *args):
    hidden = enc_outputs[1]       # [L,B,H]，作为GRU初始状态
    context = hidden[-1]          # [B,H]，始终保留编码器信息
    return hidden, context

def forward(self, X, state):
    hidden, fixed_context = state
    X = self.embedding(X).permute(1, 0, 2)       # [Tt,B,E]
    context = fixed_context.repeat(X.shape[0], 1, 1)
    output, hidden = self.rnn(
        torch.cat((X, context), dim=2), hidden
    )
    logits = self.dense(output).permute(1, 0, 2)
    return logits, (hidden, fixed_context)
```

这也解释了为什么现代注意力解码器的 `state` 往往不是单个张量，而是包含编码器输出、有效长度、缓存和解码器隐状态的结构。

## 6. 强制教学：`dec_input` 和标签 `Y` 怎样错开

假设目标句编号为：

```text
Y = [je, suis, ici, <eos>]
```

训练解码器的输入是：

```text
dec_input = [<bos>, je, suis, ici]
```

每个位置的监督关系为：

```text
编码信息 + <bos>          -> je
编码信息 + <bos>, je      -> suis
编码信息 + ..., suis      -> ici
编码信息 + ..., ici       -> <eos>
```

批量代码只需把 `<bos>` 拼到 `Y` 去掉最后一列后的结果前面：

```python
# Y：[B,Tt]，其中有效序列末尾已经含有<eos>。
bos = torch.full(
    (Y.shape[0], 1),
    tgt_vocab['<bos>'],
    dtype=Y.dtype,
    device=Y.device
)

# [B,1]和[B,Tt-1]拼接，结果仍为[B,Tt]。
dec_input = torch.cat((bos, Y[:, :-1]), dim=1)

# Y_hat：[B,Tt,Vt]；Y：[B,Tt]。
Y_hat, _ = net(X, dec_input, X_valid_len)
```

这叫**强制教学**（teacher forcing）：训练第 $t$ 步时输入真实的 $y_{t-1}$，而不是模型刚预测的 $\hat y_{t-1}$。这样可以一次并行准备所有目标输入，早期错误不会污染后续位置，优化更稳定。

`Y` 必须是完整序列而不是单个最终词元，因为解码器在每个目标时间步都执行一次词表分类。一个长度为 $T_t$ 的句对提供 $T_t$ 个监督信号。

## 7. 掩码交叉熵：三维 logits 为什么可以直接计算 loss

不同目标句经过填充后都有 `[B,Tt]`，但 `<pad>` 只是为了对齐张量，不是真实翻译目标。若把它计入损失，模型会因大量容易预测的 `<pad>` 获得奖励，并偏向过早输出填充词元。

先根据有效长度构造掩码：

```python
def sequence_mask(X, valid_len, value=0):
    """把每行有效长度之后的位置改成value。"""
    max_len = X.size(1)

    # positions：[1,T]；valid_len[:,None]：[B,1]。
    # 广播比较后mask为[B,T]。
    positions = torch.arange(max_len, device=X.device)[None, :]
    mask = positions < valid_len[:, None]

    # 注意：该实现原地修改X。
    X[~mask] = value
    return X
```

例如 `valid_len=[2,4]`、`T=4` 时：

```text
weights = [[1,1,0,0],
           [1,1,1,1]]
```

带掩码的交叉熵为：

```python
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """忽略目标序列中填充位置的交叉熵。"""
    def forward(self, pred, label, valid_len):
        # pred：[B,T,V]；label：[B,T]；valid_len：[B]。
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)

        # 保留每个(batch,time)位置的损失，而不是提前聚合。
        self.reduction = 'none'

        # CrossEntropyLoss要求类别维C位于第1维：
        # 输入[N,C,d1,...]，标签[N,d1,...]。
        # 所以[B,T,V] -> [B,V,T]，标签仍为[B,T]。
        unweighted_loss = super().forward(
            pred.permute(0, 2, 1),
            label
        )                                      # [B,T]

        # 有效位置乘1，<pad>位置乘0；每条序列返回一个数[B]。
        return (unweighted_loss * weights).mean(dim=1)
```

三维交叉熵本质上仍是 $B\times T_t$ 次独立的 $V_t$ 分类：

```text
logits：[B,Vt,Tt]  -> 每个(batch,time)位置有Vt个类别分数
label： [B,Tt]     -> 每个(batch,time)位置只有一个正确类别编号
loss：  [B,Tt]     -> reduction='none'时保留每个位置的损失
```

不需要把标签变成 `[B,Tt,Vt]` 的独热张量，也不要提前调用 softmax；`CrossEntropyLoss` 内部已经组合 `log_softmax` 和正确类别索引选择。

### 7.1 `.mean(dim=1)` 的归一化含义

D2L 的写法把掩码后的损失除以固定 $T_t$：

$$
L_b=\frac{1}{T_t}\sum_{t=1}^{T_t}m_{b,t}\ell_{b,t}.
$$

因此有效词元更多的句子通常贡献更大的样本损失，而不是每条句子各自对有效词元求平均。若想得到严格的“每个样本有效词元平均损失”，可以改成：

```python
token_loss = unweighted_loss * weights
loss_per_sample = token_loss.sum(dim=1) / valid_len.clamp_min(1)
```

若目标是整个 batch 的每词元平均值，更直接的聚合是：

```python
loss = token_loss.sum() / weights.sum().clamp_min(1)
```

三种方式的梯度尺度和样本加权不同，应让训练损失、日志统计和验证指标采用一致定义。

## 8. 一次训练更新中发生了什么

把数据、前向传播、损失和优化连起来：

```python
def train_step(net, batch, optimizer, tgt_vocab, device):
    net.train()
    optimizer.zero_grad()

    X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

    # teacher forcing输入：[<bos>, y1, ..., y(T-1)]。
    bos = torch.full(
        (Y.shape[0], 1), tgt_vocab['<bos>'],
        dtype=Y.dtype, device=device
    )
    dec_input = torch.cat((bos, Y[:, :-1]), dim=1)  # [B,Tt]

    # Encoder处理X；Decoder用编码状态和dec_input预测整个目标序列。
    logits, _ = net(X, dec_input, X_valid_len)      # [B,Tt,Vt]

    # 返回[B]后求和，得到backward要求的标量。
    loss_per_sample = MaskedSoftmaxCELoss()(
        logits, Y, Y_valid_len
    )
    loss = loss_per_sample.sum()
    loss.backward()

    # 循环网络仍可能梯度爆炸，因此更新前裁剪范数。
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.detach()
```

与语言模型顺序分区不同，每个机器翻译样本是一条独立句对。下一个 batch 不延续上一个 batch 的句子，因此不需要把解码器状态跨 batch 保存，也不需要在批次之间 `detach(state)`。

本节初始化函数对 `Linear` 和 `GRU` 权重使用 Xavier 初始化，对偏置和 `Embedding` 保持模块默认初始化。初始化只发生在训练开始前；若每个 epoch 都重新初始化，模型永远无法累积学习结果。

## 9. 预测：为什么 `dec_X` 可以覆盖，历史去了哪里

预测时没有真实 `Y` 可供强制教学。模型先编码源句，再从 `<bos>` 开始逐词生成：

```python
@torch.no_grad()
def greedy_decode(net, enc_X, enc_valid_len, tgt_vocab, num_steps):
    net.eval()

    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    # 第一次输入只有<bos>，shape为[1,1]。
    dec_X = torch.tensor(
        [[tgt_vocab['<bos>']]],
        dtype=torch.long,
        device=enc_X.device
    )

    output_ids = []
    for _ in range(num_steps):
        logits, dec_state = net.decoder(dec_X, dec_state)  # [1,1,Vt]

        # 沿词表维选择最高分编号：[1,1,Vt] -> [1,1]。
        next_token = logits.argmax(dim=2)

        # 单句、单步张量只有一个元素，取出Python整数。
        pred_id = next_token.item()
        if pred_id == tgt_vocab['<eos>']:
            break

        output_ids.append(pred_id)  # 保存完整输出序列。
        dec_X = next_token          # 只把最新词元送入下一步。

    return tgt_vocab.to_tokens(output_ids)
```

`dec_X = next_token` 确实覆盖了前一个输入，但不会清除历史：

```text
dec_X       保存下一时间步需要读取的最新词元
dec_state   保存源句信息和此前目标词元形成的循环状态
output_ids  保存已经生成的完整结果
```

RNN 解码器若同时保留整个目标前缀并继续使用已更新的状态，会重复处理旧词元。Transformer 没有同样形式的循环隐状态，通常传入完整前缀，或使用 KV cache 保存此前计算。

原 Notebook 写成：

```python
pred = dec_X.squeeze(dim=0).type(torch.int32).item()
```

对 `[1,1]` 张量而言，`squeeze(dim=0)` 得到 `[1]`，`type(torch.int32)` 改变张量类型，`item()` 再取得 Python 整数。由于张量只有一个元素，直接 `dec_X.item()` 即可；若批量大于 1，则不能调用 `item()`，应保留整批编号。

这里每步使用 `argmax`，所以是**贪心解码**。它只选择当前最优词，不保证整条序列概率最大；束搜索会同时保留若干候选前缀，是下一阶段的内容。

## 10. Teacher forcing 与自回归预测为什么存在落差

训练和推理的信息条件不同：

| 阶段 | 第 $t$ 步输入 | 优点 | 风险 |
|---|---|---|---|
| 强制教学训练 | 真实 $y_{t-1}$ | 稳定、可一次处理整段目标 | 从未练习如何从自身错误恢复 |
| 自回归预测 | 预测 $\hat y_{t-1}$ | 符合真实生成条件 | 错误会逐步累积 |
| 完全自回归训练 | 预测 $\hat y_{t-1}$ | 训练条件更接近推理 | 初期输入近似随机、训练慢且不稳定 |

这种差异常称为**暴露偏差**。若训练时完全用上一时刻预测，需要把整段解码改成显式时间循环：

```python
dec_X = bos                           # [B,1]
step_outputs = []

for _ in range(Y.shape[1]):
    step_logits, dec_state = net.decoder(dec_X, dec_state)
    step_outputs.append(step_logits)  # 每项[B,1,Vt]
    dec_X = step_logits.argmax(dim=2) # [B,1]

Y_hat = torch.cat(step_outputs, dim=1)  # [B,Tt,Vt]
```

`argmax` 是离散选择，后面时间步的损失不能穿过“选了哪个编号”这一步对先前 logits 求导；梯度只能经连续的解码器状态路径传播。训练早期错误还会污染整个后缀。

计划采样在真实词和预测词之间随机选择，并逐渐提高预测词比例，是一种直观折中，但会引入额外概率计划和随机性，不保证总能优于充分训练的强制教学。无论采用哪种策略，最终都应在完全自回归条件下验证。

## 11. BLEU 在测量什么

给定预测长度 $m$、参考长度 $r$，D2L 使用的 BLEU 形式为

$$
\operatorname{BLEU}
=\exp\left(\min\left(0,1-\frac{r}{m}\right)\right)
\prod_{n=1}^{k}p_n^{1/2^n},
$$

其中 $p_n$ 是预测序列中 $n$ 元语法的裁剪精确率。它包含两部分：

1. **短句惩罚**：若 $m<r$，即使预测的少数词都正确，也会因遗漏内容而降分。
2. **n 元语法匹配**：不仅检查单词是否出现，还检查更长局部片段是否与参考一致。

“裁剪”很重要。若参考句中某个词只出现一次，预测重复十次，最多只能匹配一次，不能用重复输出刷高精确率。

BLEU 为 1 表示在当前参考句和所选 $k$ 下完全匹配；BLEU 低不必然表示语义错误，因为正确翻译可能有多种措辞。它也不能单独诊断模型究竟错在词义、语序、遗漏还是停止位置。小规模示例可用它做一致比较，真实系统还需要多参考、人工评价或其他语义指标。

实现时还要防止预测为空或 `len_pred < n` 导致除零；教学函数默认生成结果足够长，并不等于生产实现可以忽略边界检查。

## 12. 从练习走向更通用的 Seq2Seq

### 12.1 编码器和解码器 shape 不一致时怎样传状态

直接 `return enc_outputs[1]` 要求两侧层数和隐藏维相同。若不一致，应增加**桥接层**，而不是强行 reshape：

```python
# 将编码器顶层状态[B,Henc]映射到解码器需要的[B,Hdec]。
top = enc_state[-1]
dec_top = torch.tanh(self.bridge(top))

# 若解码器有Ld层，可复制、分别投影，或为各层学习独立初始化。
dec_state = dec_top.unsqueeze(0).repeat(num_dec_layers, 1, 1)
```

双向编码器还要先融合正反向状态。LSTM 的状态是 `(H,C)`，隐状态和记忆元都必须初始化或映射。

### 12.2 输出层不只有一个 `Linear(H,V)`

基础输出层把每个隐状态直接投影为词表 logits。其他选择包括：

- **权重绑定**：用目标嵌入矩阵的转置作为输出权重，若 $H\ne E$ 则先投影到 $E$；
- **非线性输出头**：`Linear -> Tanh/GELU -> Linear`，提高表达能力；
- **融合上下文**：把解码器状态、注意力上下文和当前输入嵌入共同用于预测；
- **大词表近似**：adaptive、hierarchical 或 sampled softmax 降低完整归一化代价；
- **复制机制**：在固定词表生成与从源句复制之间学习混合概率。

权重绑定示意：

```python
# decoder_outputs：[B,T,E]；embedding.weight：[V,E]。
logits = decoder_outputs @ self.embedding.weight.T + self.output_bias
```

无论输出头如何设计，常规词表生成最终仍产生 `[B,T,V]` logits，标签仍是 `[B,T]` 整数编号。

### 12.3 GRU、LSTM、注意力各自改变什么

- 把 GRU 换成 LSTM 不改变 Seq2Seq 的数据契约，但状态从单个 `H` 变成 `(H,C)`。
- 增加层数提升每个时间步的表示深度，不消除固定上下文瓶颈。
- 注意力让每个解码步动态读取所有编码器输出，解决“整句只压进最后状态”的主要限制。
- 束搜索改变解码策略，不改变训练模型本身。
- Transformer 同样遵循编码器—解码器和自回归概率分解，但用注意力与缓存替代循环状态。

## 13. 常见误区与 shape 调试顺序

- 认为 `dec_X` 与 `enc_X` 重复；前者是目标侧历史，后者是源句。
- 认为解码器第 $t$ 步使用编码器第 $t$ 步状态；基础模型只用编码器最终状态初始化解码器。
- 认为 `nn.Embedding` 保存词频；它保存 `[V,E]` 可训练向量，词频由词表统计。
- 没看到手写初始化就以为嵌入没有权重；模块构造时已自动初始化。
- 把 `vocab_size` 当成 `embed_size`；一个是词元数量，一个是每个词元的表示维度。
- 把 `[B,T,V]` logits 与 `[B,T]` 标签 shape 不同视为错误；交叉熵要求每个位置有 $V$ 个分数、一个正确编号。
- 对 logits 先做 softmax 再送进 `CrossEntropyLoss`；这会重复归一化并削弱数值稳定性。
- 不屏蔽 `<pad>` 损失；模型会把填充模式当成真实学习目标。
- 认为覆盖 `dec_X` 会清空生成历史；循环历史在 `dec_state`，输出历史在 `output_seq`。
- 训练时用真实目标前缀，评估时却也喂真实目标；这不是自回归翻译，会高估性能。
- 认为贪心每步最优等于整句概率最优；局部选择无法回溯。
- 把 BLEU 当成语义正确性的充分条件；它主要比较参考译文的局部 n 元语法。

遇到问题时按数据流打印：

```python
print("enc tokens", X.shape)               # [B,Ts]
print("enc valid", X_valid_len.shape)      # [B]
print("target", Y.shape)                   # [B,Tt]
print("dec input", dec_input.shape)        # [B,Tt]
print("enc output", enc_output.shape)      # [Ts,B,H]
print("enc state", enc_state.shape)        # [L,B,H]
print("logits", logits.shape)              # [B,Tt,Vt]
print("labels", Y.shape)                    # [B,Tt]
print("token loss", token_loss.shape)      # [B,Tt]
```

若元素数相同但 loss 异常，再核对时间顺序、词表是否用错、`<bos>/<eos>` 是否错位、`valid_len` 是否包含 `<eos>`，而不是继续盲目 reshape。

## 本篇知识链总结

Seq2Seq 把“一个序列预测另一个序列”拆成两条相连但不同的递归。源词元先由源嵌入层转换成向量，编码器沿源时间轴更新状态；最终编码状态初始化解码器。目标嵌入层再把 `<bos>` 或已有目标词元变成输入，解码器沿目标时间轴生成 `[B,T_t,V_t]` logits。

训练时，真实目标序列右移一位形成 `dec_input`，这让所有位置都获得稳定监督；`valid_len` 构造掩码，使三维交叉熵只保留真实词元。预测时没有真实目标，模型从 `<bos>` 开始，把每一步 `argmax` 得到的编号反馈给下一步；最新输入可以覆盖，因为循环状态保存历史，生成列表保存结果。这个训练—推理信息差异解释了暴露偏差，也解释了为什么 BLEU 必须在真正的自回归输出上计算。

整篇可以压缩成四个检查问题：

```text
数据：源词表、目标词表、<eos>、<pad>和valid_len是否一致？
状态：编码器最终状态怎样初始化并持续影响解码器？
监督：dec_input与Y是否恰好错开一个词元，<pad>是否被屏蔽？
推理：模型是否只使用自身已经生成的信息，何时遇到<eos>停止？
```

## 系列导航

- 上一篇：[D2L 循环神经网络：从序列数据与 BPTT 到 GRU、LSTM、深层和双向建模]({{ '/deep-learning/recurrent-neural-networks/' | relative_url }})
- [系列总览]({{ '/deep-learning/' | relative_url }})
- 下一篇：[D2L 注意力机制：从 QKV、多头自注意力到 Transformer]({{ '/deep-learning/attention-transformer/' | relative_url }})

## 对应章节与参考资料

- D2L：[机器翻译与数据集](https://zh.d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html)、[编码器—解码器架构](https://zh.d2l.ai/chapter_recurrent-modern/encoder-decoder.html)、[序列到序列学习](https://zh.d2l.ai/chapter_recurrent-modern/seq2seq.html)
- PyTorch：[`nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)、[`nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)、[`CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- Sutskever、Vinyals、Le：[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- Cho 等：[Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- Papineni 等：[BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
