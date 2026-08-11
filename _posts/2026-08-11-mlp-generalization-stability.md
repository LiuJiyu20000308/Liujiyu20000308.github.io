---
layout: post
title: "多层感知机、泛化与稳定训练：从非线性到残差思想"
date: 2026-08-11 09:20 +0800
tags: [D2L, PyTorch, 深度学习, 神经网络]
toc: true
math: true
permalink: /deep-learning/mlp-generalization/
---

## 本篇逻辑主线

线性模型的决策边界受限，于是我们叠加隐藏层；但若层与层之间没有非线性，多层线性变换仍可合并成一层，深度并没有增加表达能力。激活函数让 MLP 能拟合复杂关系，也带来模型容量、过拟合与梯度传播问题。可靠训练因此需要两条并行链：一条用训练/验证/测试划分、K 折和防泄漏流程估计泛化；另一条用 L1/L2、权重衰减、Dropout、合适初始化、BatchNorm、梯度裁剪和残差连接控制容量并稳定优化。最终目标不是让训练误差最低，而是用没有被调参过程污染的数据，验证模型能否在未知样本上稳定工作。

## 1. 为什么多层线性层仍是一层

假设两层之间没有激活函数：

$$
\mathbf h=W_1\mathbf x+\mathbf b_1,
\qquad
\mathbf y=W_2\mathbf h+\mathbf b_2.
$$

代入后：

$$
\mathbf y=(W_2W_1)\mathbf x+(W_2\mathbf b_1+\mathbf b_2),
$$

仍是一个仿射变换。堆再多层也只会改变线性映射的参数化，而不会产生弯曲决策边界。MLP 的关键不是“多几次矩阵乘法”，而是在线性层之间插入非线性激活：

```python
net = nn.Sequential(
    nn.Linear(d_in, 256),
    nn.ReLU(),
    nn.Linear(256, d_out),
)
```

隐藏单元数、层数、连接方式和输入特征共同决定模型容量。

宽度没有通用的“必须逐层压缩”公式。第一层先扩展到较宽表示、之后缓慢压缩是一种常见启发式：它给模型空间组合输入特征，再逐步形成任务表示；但窄瓶颈也可能过早丢信息，过宽又增加计算与过拟合。候选深度和宽度应作为超参数，用可靠验证协议比较，而不是只凭网络示意图。

## 2. 激活函数：让网络不再可合并

### ReLU、sigmoid、tanh

$$
\operatorname{ReLU}(x)=\max(0,x),
$$

$$
\sigma(x)=\frac{1}{1+e^{-x}},
\qquad
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}.
$$

| 激活 | 输出范围 | 优点 | 主要问题 |
|---|---|---|---|
| ReLU | $[0,\infty)$ | 正半轴梯度为 1、计算简单、稀疏激活 | 负半轴梯度为 0，可能出现“死亡 ReLU” |
| sigmoid | $(0,1)$ | 适合作为二元概率输出 | 大正/负输入进入饱和区，梯度接近 0；输出非零中心 |
| tanh | $(-1,1)$ | 零中心 | 两端仍饱和，深层网络易梯度消失 |

ReLU 更常见不是因为它“处处可导”，而是因为其正区间不饱和、计算便宜，配合合理初始化更容易训练深层网络。输出层是否使用激活由任务和损失决定：`CrossEntropyLoss` 前保留 logits；二元概率展示时才 sigmoid。

### 不可导点怎么办

ReLU 在 $x=0$ 的数学导数不存在，但优化只需要框架选取一个次梯度约定。PyTorch 在该点返回 0。精确落在不可导点通常不是训练的主要障碍；更重要的是负区间长期为 0 导致单元收不到梯度。Leaky ReLU、PReLU 等变体给负区间一个小斜率。

自定义分段激活应使用张量运算，如 `torch.where` 或 `clamp`，不要把张量转成 Python 标量再分支，否则会断开向量化和计算图。

## 3. 容量、偏差、方差与不可约噪声

预测误差可以用一条有用但简化的分解来理解：

$$
\mathbb E[(y-\hat f(x))^2]
=\operatorname{Bias}^2+\operatorname{Variance}+\sigma^2.
$$

- 偏差（bias）高：模型或特征过于简单，训练集也拟合不好，表现为欠拟合。
- 方差（variance）高：模型对训练样本波动太敏感，训练误差低而验证误差高，表现为过拟合。
- 不可约噪声 $\sigma^2$：标签噪声或未观测因素导致，增加模型复杂度也无法消除。

增加层数、隐藏单元或训练时长常会降低偏差，却可能增大方差；更多高质量数据、合适正则化和更符合任务的归纳偏置通常能降低方差。不要把“训练更久”当成总会改善泛化的按钮。

多项式回归很好地展示容量变化：把标量 $x$ 映射为 $[1,x,x^2,\ldots,x^d]$，再做线性回归，就能拟合 $d$ 次多项式。模型对这些“新特征”仍是线性的，但对原始 $x$ 已是非线性；次数太低易欠拟合，次数很高而数据很少则易过拟合。代码里的 `train_features.shape[-1]` 就是在读取最后一维的特征数，供 `nn.Linear(in_features, 1)` 使用。

## 4. 训练集、验证集和测试集：三者职责不同

- 训练集：计算梯度，拟合参数。
- 验证集：选择架构、正则强度、学习率、轮数和阈值。
- 测试集：在所有方案冻结后，只做最终一次无偏评估。

反复查看测试结果再改模型，测试集就事实上变成了验证集。最终数字会对这个测试集过拟合。

### 数据泄漏往往发生在预处理

标准化的均值/方差、缺失值填充值、类别词表、特征选择、降维、数据增强参数和阈值都必须只用训练部分拟合，再应用到验证/测试部分。先在全数据上标准化再划分，即使没有显式使用标签，也向训练过程泄露了验证分布信息。

时间序列、同一受试者的多条记录、同一原图的增强版本还需要按时间或组划分，不能随机把高度相关样本分到两侧。

### K 折交叉验证有什么用

数据较少时，将开发数据分成 $K$ 份，每次用一份验证、其余训练，得到 $K$ 个验证分数：

$$
\bar s=\frac1K\sum_{k=1}^{K}s_k.
$$

它降低“恰好分到一个容易/困难验证集”的偶然性，用于比较超参数或模型方案。K 折不是用来让同一个测试集反复参与选择。

选定方案后，应使用全部开发数据（原训练集与验证集）从头重训最终模型，因为验证样本的使命已经完成，把它们纳入训练能利用更多信息。测试集仍保持封存。若 K 折要做集成，也可以保留各折模型，但这是另一种明确的最终方案。

超参数训练不是对超参数求普通反向梯度，而是在外层比较不同配置。网格/随机搜索直接试候选；贝叶斯优化用已观察到的“配置—验证分数”拟合代理模型，再在探索未知区域与利用当前好区域之间选择下一次昂贵实验。无论搜索多智能，最终依据仍是未泄漏的验证分数，泛化误差则是模型在真实未知分布上的期望误差，只能由有限验证/测试样本近似估计。

## 5. L1、L2、权重衰减与约束视角

### L2 / Ridge

$$
L_{\text{total}}=L_{\text{data}}+\frac{\lambda}{2}\lVert\mathbf w\rVert_2^2.
$$

其梯度多出 $\lambda\mathbf w$，使参数持续向 0 收缩，通常得到许多小而非精确为 0 的权重。在线性回归中称 Ridge；在神经网络中常称 L2 正则或权重衰减。

经典 SGD 下：

$$
\mathbf w\leftarrow(1-\eta\lambda)\mathbf w-\eta\nabla L_{\text{data}},
$$

所以表现为每步先按比例衰减。对 Adam 等自适应优化器，直接把 L2 项混入梯度与解耦权重衰减并不完全等价；PyTorch 中需要明确选择优化器及其语义（如 `AdamW`）。

### L1 / Lasso

$$
L_{\text{total}}=L_{\text{data}}+\lambda\lVert\mathbf w\rVert_1.
$$

L1 在 0 附近有尖点，更容易把部分系数推到精确 0，因此具有稀疏选择效果。在线性模型中称 Lasso。它并不保证在高度相关特征中稳定选出“真正那一个”。

### 拉格朗日与 KKT 的必要直觉

“限制参数范数不超过某值”与“在损失中惩罚参数范数”是约束形式和惩罚形式的两种视角：

$$
\min_{\mathbf w}L(\mathbf w)
\quad\text{s.t.}\quad R(\mathbf w)\le c
$$

可通过拉格朗日函数写成

$$
\mathcal L(\mathbf w,\lambda)=L(\mathbf w)+\lambda(R(\mathbf w)-c),
\qquad \lambda\ge0.
$$

KKT 条件的直觉是：若约束没有卡住，乘子可为 0；若最优点正落在边界，惩罚梯度与数据损失梯度达到平衡。实际训练不需要手推每个 KKT 条件，但这个视角解释了 $\lambda$ 为什么代表“拟合与复杂度之间的价格”。

通常不衰减偏置和 BatchNorm 的缩放/平移参数，因为它们规模小、角色不同；应通过优化器参数组显式控制。

## 6. Dropout：随机子网络与 inverted scaling

训练时，inverted dropout 对激活 $h$ 使用

$$
h'=\frac{m}{1-p}h,
\qquad m\sim\operatorname{Bernoulli}(1-p).
$$

因此

$$
\mathbb E[h']=h.
$$

训练时已除以 $1-p$，评估时直接关闭 Dropout，不再缩放。这就是 `nn.Dropout(p)` 的行为。

保持期望不等于保持方差。对固定 $h$：

$$
\operatorname{Var}(h')=\frac{p}{1-p}h^2,
$$

随机掩码故意注入噪声，阻止单元形成脆弱的共适应关系。它是一种训练期正则化，不是模型压缩；推理时网络结构并没有因此变小。

```python
net = nn.Sequential(
    nn.Linear(d, 256), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, out_dim),
)
```

必须用 `train()` / `eval()` 切换行为。

## 7. 梯度消失、梯度爆炸与数值异常

深层复合函数的梯度包含大量雅可比矩阵乘积。若这些乘积的典型尺度小于 1，梯度随深度衰减；大于 1，则可能指数放大。

稳定性分析重点看激活 $h_l$ 和损失对激活的梯度 $\partial L/\partial h_l$，因为它们是穿过每一层不断传播的“信号”；单层权重梯度通常由该层输入激活与输出梯度共同构成。若信号早已消失或爆炸，即使优化器最终更新的是 $W,b$，也拿不到合适的参数梯度。

- sigmoid/tanh 饱和区导数接近 0，容易梯度消失。
- 权重尺度过大或重复乘法可能导致梯度爆炸。
- 过大激活、除以接近 0、`log(0)`、指数溢出会产生 `Inf`；`Inf-Inf`、`0/0` 等进一步产生 `NaN`。

调试时先定位第一次非有限值：

```python
assert torch.isfinite(loss)
for name, p in model.named_parameters():
    if p.grad is not None:
        assert torch.isfinite(p.grad).all(), name
```

降低学习率、使用稳定的融合损失（如 `CrossEntropyLoss`）、加入 `eps`、检查输入尺度和初始化，通常比在最后盲目替换 NaN 更有效。

### Xavier 与 Kaiming/He 初始化

全零初始化隐藏层会产生对称性：同层神经元收到相同梯度，始终学成相同特征。需要随机初始化打破对称。

Xavier 试图同时保持前向激活和反向梯度的方差，适合近似线性或 tanh：

$$
\operatorname{Var}(W_{ij})\approx\frac{2}{n_{\text{in}}+n_{\text{out}}}.
$$

Kaiming/He 针对 ReLU 约有一半激活被截断的情况：

$$
\operatorname{Var}(W_{ij})\approx\frac{2}{n_{\text{in}}}.
$$

```python
nn.init.xavier_uniform_(linear.weight)
nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
```

初始化不是为了保证永不爆炸，而是让训练起点落在可传播的尺度范围。

### 梯度裁剪

全局范数裁剪把过大的梯度缩放到阈值：

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

它是防止偶发爆炸的保险丝，不会修复错误数据、错误损失或长期不合适的学习率。

## 8. BatchNorm 与残差连接为何有助于深层训练

Batch Normalization 对一个通道/特征在训练 batch 上标准化，再学习缩放和平移：

$$
\operatorname{BN}(x)=\gamma\frac{x-\mu_\mathcal{B}}
{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}+\beta.
$$

它改善激活尺度和优化条件，并引入 batch 统计噪声；“减少内部协变量偏移”是早期直觉，并非充分的因果解释。训练时使用当前 batch 统计并更新 running mean/variance，推理时使用运行统计，所以 batch 太小会不稳定，`train()` / `eval()` 也不可省略。

running mean/variance 用 momentum 控制新 batch 统计写入移动平均的比例，`eps` 防止除以零；它们是 buffer，不通过梯度学习。$\gamma,\beta$ 与前层参数会通过标准化运算共同获得梯度，而且同一 batch 样本因均值/方差计算而相互耦合。batch size=1 时，全连接特征的方差会退化，BatchNorm 无法提供有效标准化；LayerNorm 改为在单个样本的特征维上统计，因而不依赖 batch size。

Xavier/Kaiming 只设置训练起点的权重尺度，BatchNorm 则在整个训练过程中动态重整中间激活并学习缩放；两者都关心信号尺度，但作用时机和机制不同。BatchNorm 常让损失曲面条件更友好、对尺度更不敏感，因此能够尝试相对更大的学习率，但不意味着学习率可以无限增大。

残差块学习

$$
\mathbf y=F(\mathbf x)+\mathbf x.
$$

恒等路径让信号和梯度有一条更直接的通路，也让新块在 $F\approx0$ 时接近恒等映射。它让深层模型更容易优化，但不保证“加层后测试性能绝不会变差”：有限数据、优化失败和过拟合仍然存在。CNN 中的具体下采样与 shape 对齐将在[第五篇]({{ '/deep-learning/cnn-evolution/' | relative_url }})展开。

## 9. 剪枝与知识蒸馏不是 Dropout

- 剪枝（pruning）：从已经训练的模型中删除权重、通道或结构，目标是减少计算/存储；常需微调恢复精度。
- 知识蒸馏（knowledge distillation）：训练较小的 student 模型去匹配 teacher 的软输出或中间表示，知识被迁移到另一套参数中。
- Dropout：训练时随机屏蔽激活以正则化，推理时仍使用完整模型。

三者都可能与“减少依赖、提高效率”相关，但机制和产物完全不同。

## 本篇知识链总结

非线性激活让多层网络真正扩大函数类；容量扩大后，必须用独立验证流程判断泛化，用 L1/L2 和 Dropout 等方法约束模型；深度增加后，又要用合理初始化、稳定损失、BatchNorm、梯度裁剪和残差路径维护信号与梯度。模型选择与数值稳定不是训练结束后的补丁，而是同一建模流程的两面。

## 常见误区

- 认为多叠几层 `Linear` 即使没有激活也会更强。
- 把验证集或测试集统计量用于标准化、填补和特征选择。
- 选择超参数后不重训，白白丢掉验证数据；或反过来把测试集也纳入重训。
- 认为权重衰减对所有优化器都与 L2 惩罚完全等价。
- 认为 inverted dropout 同时保持均值和方差。
- 用全零权重初始化隐藏层，导致神经元无法打破对称。
- 把梯度裁剪当作掩盖 NaN 根因的万能修复。
- 认为残差连接从数学上保证任何数据集上的最终性能不下降。

## 系列导航

- 上一篇：[从线性回归到 softmax 分类]({{ '/deep-learning/linear-models/' | relative_url }})
- [系列总览]({{ '/deep-learning/' | relative_url }})
- 下一篇：[PyTorch 工程实践与完整建模流程]({{ '/deep-learning/pytorch-engineering/' | relative_url }})

## 对应章节与参考资料

- D2L：[多层感知机](https://zh.d2l.ai/chapter_multilayer-perceptrons/index.html)、[数值稳定性和模型初始化](https://zh.d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html)、[批量规范化](https://zh.d2l.ai/chapter_convolutional-modern/batch-norm.html)
- PyTorch：[`Dropout`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)、[`BatchNorm2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)、[初始化函数](https://docs.pytorch.org/docs/stable/nn.init.html)
