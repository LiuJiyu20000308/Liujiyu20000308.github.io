---
layout: post
title: "PyTorch 工程实践：从 nn.Module 到可复现建模流程"
date: 2026-08-11 09:30 +0800
tags: [D2L, PyTorch, 深度学习, 神经网络]
toc: true
math: true
permalink: /deep-learning/pytorch-engineering/
---

## 本篇逻辑主线

会写一个 `Linear` 层并不等于拥有可维护的模型。PyTorch 用 `nn.Module` 把层、参数和子块组织成树，`Sequential` 负责简单串联，自定义 `forward` 负责分支、复用和控制流；注册后的参数才能被初始化、迁移到 GPU、保存并交给优化器。模型外部还需要同样严格的数据流程：预处理规则只能在训练集上拟合，验证集负责选择方案，环境与随机性必须记录，checkpoint 要能恢复。房价预测提供了一条完整范例——从 pandas 表格、缺失值、标准化和 one-hot，走到 K 折、重训、保存和部署——也揭示了版本、dtype、高基数类别与分布偏移这些“代码之外”的工程约束。

## 1. `nn.Module`：层、块和模型是同一种抽象

在 PyTorch 中，单层、残差块和完整网络都可以是 `nn.Module`。一个模块至少负责：

- 保存子模块和需要学习的 `Parameter`；
- 定义从输入到输出的 `forward`；
- 递归暴露参数、状态和设备迁移能力；
- 配合 autograd 自动完成反向传播。

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.hidden = nn.Linear(in_features, hidden)
        self.output = nn.Linear(hidden, out_features)

    def forward(self, X):
        X = torch.relu(self.hidden(X))
        return self.output(X)
```

通过 `self.hidden = ...` 赋值后，子层会被注册。若把层放在普通 Python list 里，PyTorch 不一定能发现它们；动态层列表应使用 `nn.ModuleList`，键值结构使用 `nn.ModuleDict`。

### `Sequential` 与自定义 `forward`

纯串行结构适合：

```python
net = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 10),
)
```

若有多分支、跳跃连接、同一层复用或输入相关控制流，应写自定义模块。调用 `net(X)` 而不是直接 `net.forward(X)`，因为 `Module.__call__` 还负责 hooks、autocast 等框架逻辑。

`torch.nn.functional.relu(X)`（常写作 `F.relu`）是无状态函数，适合在自定义 `forward` 中临时调用；`nn.ReLU()` 是可注册进 `Sequential` 的模块。对 ReLU 这种无参数运算二者数学上相同，但模块形式更便于打印结构、挂 hook 和统一配置；带可学习参数或持久状态的层应使用模块并在 `__init__` 中注册。

## 2. 参数访问、注册和共享

```python
for name, param in net.named_parameters():
    print(name, param.shape, param.requires_grad)

state = net.state_dict()
```

`named_parameters()` 返回参与训练的参数；`state_dict()` 还包含 BatchNorm 的 running mean/variance 等持久 buffer。普通 tensor 属性不会自动成为参数：

```python
self.weight = nn.Parameter(torch.randn(in_dim, out_dim))
self.register_buffer("running_scale", torch.ones(out_dim))
```

共享参数必须复用同一个层对象，而不是复制相同数值：

```python
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8), nn.ReLU(),
    shared, nn.ReLU(),
    shared, nn.ReLU(),
)
```

两个位置引用同一个参数，因此反向传播得到的梯度贡献会累加到同一 `.grad`。

## 3. 初始化：尺度与对称性都要处理

PyTorch 层有合理默认初始化，但深层网络常需根据激活函数显式选择 Xavier 或 Kaiming：

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

net.apply(init_weights)
```

不能把同一层所有神经元的权重初始化为相同值。即使它不是 0，对称神经元也会收到相同梯度，始终学到相同特征。偏置初始化为 0 通常没有这个问题，因为随机权重已经打破对称。

### 延后初始化

有时构建模型时还不知道输入维度。`nn.LazyLinear(out_features)` 和 `nn.LazyConv2d` 会在第一次看到输入时推断缺失 shape：

```python
net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(), nn.Linear(64, 10))
net(torch.randn(2, 20))  # 首次前向后，第一层权重确定为 [64, 20]
```

第一次前向前参数处于未初始化状态，依赖参数 shape 的初始化、优化器或保存逻辑要放在 materialize 之后，或严格按 Lazy module 文档处理。延后初始化减少硬编码，却不能替代 shape 检查。

## 4. 自定义层：让参数留在计算图里

不带参数的中心化层：

```python
class CenteredLayer(nn.Module):
    def forward(self, X):
        return X - X.mean()
```

带参数的自定义线性层：

```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, X):
        return torch.relu(X @ self.weight.T + self.bias)
```

前向传播中不要使用 `self.weight.data`，否则运算可能绕开 autograd。只有明确不参与梯度的状态才注册为 buffer 或普通常量。

## 5. 保存与加载：优先保存 `state_dict`

推荐保存模型与优化器状态，而不是依赖任意 Python 对象的整体序列化：

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "config": config,
}, "checkpoint.pt")
```

恢复时先用代码重建相同架构，再加载：

```python
checkpoint = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_epoch = checkpoint["epoch"] + 1
model.eval()
```

只做推理时可以只保存模型 `state_dict`。要无缝续训，还应保存 scheduler、混合精度 scaler、随机状态和数据划分标识。加载外部文件前确认来源可信，并以当前 PyTorch 文档为准处理 `torch.load` 的安全选项。

## 6. Device 与 GPU：模型和数据必须在同一设备

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for X, y in loader:
    X = X.to(device)
    y = y.to(device)
    logits = model(X)
```

`.to(device)` 返回位于目标设备的张量；对 tensor 写 `X.to(device)` 却不接收返回值，`X` 仍在原设备。`.cuda()` 是 CUDA 专用快捷方式，`.to(device)` 更便于 CPU/GPU 统一和多卡编号。

只有同一运算的张量都在同一 device 上才能计算。频繁把小张量 `.cpu().numpy()` 或 `.item()` 会触发设备同步，成为性能瓶颈；日志应适度聚合后再传回 CPU。

### 行列数事先不确定时怎样拼接

不要在循环里反复 `torch.cat`/`np.concatenate`：每次都要重新分配并复制之前的全部结果，累计成本可能接近平方级。若最终 shape 可计算，先 `empty` 预分配再按切片写入；若确实未知，先把每块放入 Python list，循环结束后只 `torch.cat(chunks, dim=...)` 一次。数据量超出内存时应流式写入分块文件或增量存储，而不是构造一个无限增长的大矩阵。拼接前除目标维以外的 shape 必须一致。

## 7. 房价预测：表格数据的完整流程

房价预测把工程中的关键边界放在同一任务里。假设训练表有标签 `SalePrice`，测试表没有：

```python
train_features = train_df.drop(columns=["SalePrice"])
test_features = test_df.copy()
```

### 数值特征：只用训练集拟合规则

```python
num_cols = train_features.select_dtypes(include="number").columns
means = train_features[num_cols].mean()
stds = train_features[num_cols].std().replace(0, 1)

train_features[num_cols] = (train_features[num_cols] - means) / stds
test_features[num_cols] = (test_features[num_cols] - means) / stds

fill_values = train_features[num_cols].mean()
train_features[num_cols] = train_features[num_cols].fillna(fill_values)
test_features[num_cols] = test_features[num_cols].fillna(fill_values)
```

实践中应先从原始训练部分计算填充值和缩放量，再固定应用。不能各自在训练集和测试集上标准化，否则同一数值在两边代表不同坐标。

### 类别特征与 one-hot

`pd.get_dummies` 会受 pandas 版本、类别 dtype、缺失值选项和数据中实际类别集合影响。最关键的是让训练与测试列严格对齐：

```python
all_raw = pd.concat([train_features, test_features], axis=0)
all_encoded = pd.get_dummies(all_raw, dummy_na=True, dtype="float32")

n_train = len(train_features)
X_train_df = all_encoded.iloc[:n_train]
X_test_df = all_encoded.iloc[n_train:]
```

竞赛中把无标签测试特征一起用于确定类别列通常不涉及标签泄漏，但严格生产流程更稳妥的做法是：只在训练集拟合类别词表，为未知类别保留专门编码，再 `reindex` 测试列。无论采用哪种协议，都要在实验记录中说明。

旧教程里固定出现“331 列”不代表任何环境都必须相同：pandas 版本、源数据版本、`dummy_na`、数值/类别 dtype 判断都可能改变列数。应该验证列名、dtype 和对齐关系，而不是追逐一个硬编码数字。

转换 tensor 前保证没有 `object` dtype：

```python
X_train = torch.tensor(X_train_df.to_numpy(dtype="float32"))
X_test = torch.tensor(X_test_df.to_numpy(dtype="float32"))
y_train = torch.tensor(train_df["SalePrice"].to_numpy(dtype="float32"))
```

`numpy.object_` 转换错误通常意味着混入字符串、布尔扩展类型或未编码类别。先检查 `df.dtypes` 和非数值列，不能靠强制 `torch.tensor(..., dtype=...)` 猜测修复。

### One-hot 的内存边界

低基数类别适合 one-hot；高基数列会产生巨大稀疏矩阵。可选方案包括：

- 删除无意义的 ID；
- 合并罕见类别并设置 unknown；
- 使用稀疏表示；
- 对类别索引使用 embedding；
- 使用适合类别特征的树模型或专门编码器。

合并大表再 one-hot 还会产生中间副本，应估算 `rows × columns × bytes_per_value`，尽早转成 `float32`。

### K 折、选型与最终重训

在每一折内部重新拟合预处理器和模型；用验证 log-RMSE 比较学习率、权重衰减、epoch 等方案。选定配置后，在全部有标签训练数据上从头重训，再预测无标签测试集。提交文件只是推理产物，不是新的训练标签。

完整训练循环的模式见[第一篇]({{ '/deep-learning/foundations/' | relative_url }})，泛化和 K 折的边界见[第三篇]({{ '/deep-learning/mlp-generalization/' | relative_url }})。

## 8. 环境隔离、版本与 Jupyter kernel

“终端里安装成功，Notebook 仍然 import 失败”通常因为安装命令和 kernel 使用的不是同一个解释器。先在 Notebook 检查：

```python
import sys, torch
print(sys.executable)
print(torch.__version__)
```

一个可复现项目至少记录：

- Python、PyTorch、pandas、D2L 的版本；
- CPU/CUDA 平台信息；
- 依赖锁文件或环境文件；
- Notebook 选择的 kernel；
- 随机种子、数据版本、划分方法和关键配置。

D2L 的书稿、`d2l` 包与 PyTorch API 都会演化。旧 Notebook 的输出列数、默认 dtype、函数签名或 `.data` 写法不能被当成当前行为的唯一标准；先区分“概念是否改变”和“接口是否改变”。

## 9. 分布偏移与 AutoML 的正确位置

训练与部署分布不一致时，模型即使在验证集上很好也可能失败：

- 协变量偏移：$P(X)$ 变了，但 $P(Y\mid X)$ 近似不变；
- 标签偏移：$P(Y)$ 变了，但 $P(X\mid Y)$ 近似不变；
- 概念偏移：$P(Y\mid X)$ 本身改变。

工程上需要监控输入范围、类别未知率、预测分布和延迟标签表现，并让验证划分尽量模拟部署条件。

AutoML 可以自动搜索预处理、模型和超参数，适合作为强基线与搜索工具；它不会自动修复数据泄漏、错误指标、分布偏移或不合理任务定义。搜索空间和验证协议若错，自动化只会更快地选出错误答案。

## 10. 让代码和知识长期可用

建议把一次实验拆成四类可追踪对象：

1. 源代码：模型、数据管道、训练入口；
2. 配置：超参数、路径的相对约定、随机种子；
3. 环境：依赖锁定与硬件信息；
4. 产物：checkpoint、指标、图表和简洁结论。

Notebook 适合探索和解释，正式训练逻辑应逐步抽到可测试的模块中。知识记录不要只保存“最后能跑的代码”，还要写明 shape、假设、验证协议和曾经踩过的坑；这正是本系列把对话与 Notebook 重组为文章的目的。

## 本篇知识链总结

`nn.Module` 让参数和子块成为可递归管理的树；注册决定参数能否被优化、保存和迁移；初始化与延后初始化解决模型建立时的尺度和 shape；`state_dict` 与 device 管理让训练可恢复、可部署。模型之外，预处理器、数据划分、环境版本和监控同样是系统状态。只有它们共同被记录和验证，一段 Notebook 才真正成为可复现建模流程。

## 常见误区

- 把子层放进普通 list，导致参数未注册。
- 直接调用 `forward`，绕过 `Module.__call__` 的框架机制。
- 在自定义层中使用 `.data` 参与前向计算。
- `X.to(device)` 后不接收返回值。
- 只保存模型参数，却期望无缝恢复优化器动量和训练轮数。
- 在测试集上分别拟合标准化或类别编码规则。
- 看到 one-hot 列数与教程不同就认为 PyTorch 出错，而不检查 pandas、数据和 dtype。
- 把 AutoML 当成数据质量和验证设计的替代品。

## 系列导航

- 上一篇：[多层感知机、泛化与稳定训练]({{ '/deep-learning/mlp-generalization/' | relative_url }})
- [系列总览]({{ '/deep-learning/' | relative_url }})
- 下一篇：[CNN：从卷积到 DenseNet]({{ '/deep-learning/cnn-evolution/' | relative_url }})

## 对应章节与参考资料

- D2L：[深度学习计算](https://zh.d2l.ai/chapter_deep-learning-computation/index.html)、[Kaggle 房价预测](https://zh.d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html)、[环境和分布偏移](https://zh.d2l.ai/chapter_multilayer-perceptrons/environment.html)
- PyTorch：[`state_dict` 保存与加载](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、[CUDA 语义](https://docs.pytorch.org/docs/stable/notes/cuda.html)、[Lazy modules](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html)
