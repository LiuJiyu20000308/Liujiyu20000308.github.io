---
layout: post
title: "vLLM 量化与推测解码：怎样减少容量、带宽与串行轮次"
date: 2026-09-02 09:50 +0800
tags: [vLLM, LLM 推理, 量化, FP8, AWQ, GPTQ, Speculative Decoding]
toc: true
math: true
permalink: /vllm/quantization-speculative-decoding/
---

## 本篇要回答什么

模型名写着 INT4，整台服务的显存能否直接除以四？FP8 与 INT8 同为 1 byte，为什么误差和 kernel 完全不同？推测解码先由小模型“猜”多个 tokens，最终输出为什么仍能服从目标模型分布？

量化减少某类数据的表示宽度；推测解码尝试减少目标模型串行 decode 的轮数。二者改变的是不同成本项，都需要把容量、正确性、kernel 支持和端到端性能分开验证。

## 1. 从实数到有限编码

仿射量化可写为：

$$
q=\operatorname{clamp}
\left(\operatorname{round}(x/s)+z,q_{min},q_{max}\right),
\qquad
\hat x=s(q-z).
$$

$s$ 是 scale，表示整数刻度的一格对应多大实数范围；$z$ 是 zero point；round 和 clamp 说明信息通常不可逆。

若把校准区间 $[x_{min},x_{max}]$ 铺到整数范围，一种常见选择是

$$
s=\frac{x_{max}-x_{min}}{q_{max}-q_{min}},
\qquad
z=\operatorname{round}(q_{min}-x_{min}/s).
$$

这只是建立概念的 min-max 方案，不等于 AWQ、GPTQ 或所有生产 kernel 的完整算法。scale 可以按 tensor、channel、group 或 block 计算；粒度越细通常越能适应局部动态范围，但 metadata 和 kernel 寻址也更复杂。

### bit、byte 与编码数量

$b$ bits 最多表示 $2^b$ 个不同编码；8 bits=1 byte，所以 INT8 有 256 个 bit patterns，INT4 有 16 个，两个 INT4 values 理论上可打包进 1 byte。编码数量不是可表示实数的“精度保证”：scale、zero point、动态范围、舍入、饱和和数据分布共同决定误差。

以 unsigned $b$-bit 仿射量化为例，$q_{min}=0,q_{max}=2^b-1$。要求 $x_{min}\mapsto q_{min}$、$x_{max}\mapsto q_{max}$：

$$q_{min}\approx x_{min}/s+z,\qquad q_{max}\approx x_{max}/s+z.$$

两式相减得到

$$s=\frac{x_{max}-x_{min}}{q_{max}-q_{min}},$$

再代回得到 $z\approx q_{min}-x_{min}/s$，随后 round 并 clamp 到合法整数范围。对称量化常固定 $z=0$，用最大绝对值确定 $s$；非对称量化可更充分覆盖偏置区间，但 zero-point 运算/metadata 可能更复杂。

### 可执行的 per-tensor 与 per-group 量化

```python
def affine_quantize(x, bits=8, group_size=None):
    flat = [float(v) for v in x]
    if group_size is None:
        group_size = len(flat)
    qmin, qmax = 0, 2**bits - 1
    q, xhat, meta = [], [], []
    for start in range(0, len(flat), group_size):
        g = flat[start:min(start + group_size, len(flat))]
        lo, hi = min(min(g), 0.0), max(max(g), 0.0)  # 保证实数 0 可表示
        scale = (hi - lo) / (qmax - qmin) if hi != lo else 1.0
        zero = max(qmin, min(qmax, round(qmin - lo / scale)))
        qg = [max(qmin, min(qmax, round(v / scale) + zero)) for v in g]
        q.extend(qg)
        xhat.extend(scale * (code - zero) for code in qg)
        meta.append((scale, zero))
    return q, xhat, meta

x = [-8.0, -0.2, 0.1, 0.3, 7.0, 7.1, 7.2, 7.3]
_, tensor_hat, tensor_meta = affine_quantize(x, bits=4)
_, group_hat, group_meta = affine_quantize(x, bits=4, group_size=4)
mae = lambda a, b: sum(abs(u-v) for u, v in zip(a, b)) / len(a)
print("per-tensor MAE", mae(x, tensor_hat), tensor_meta)
print("per-group  MAE", mae(x, group_hat), group_meta)
assert len(tensor_meta) == 1 and len(group_meta) == 2
```

输入是 float tensor、bits 和 group size；输出是量化 codes、反量化近似与每组 `(scale,zero)`。shape 不变，但最后一维被逻辑分组。状态无变化。它实现 min-max affine 关系，不能模拟 INT4 packed storage、outlier 策略、AWQ/GPTQ 校准或真实 GPU kernel；per-group 也不保证对每组数据误差一定更小，只提供更细局部范围。

## 2. 权重、激活、累加器和 KV 必须分开

线性层

$$
Y=XW^T
$$

至少涉及：常驻权重 $W$、随本轮 tokens 变化的激活 $X/Y$、点积累加器，以及服务生命周期中增长的 KV Cache。一个 INT4 checkpoint 通常只说明权重表示。

| 对象 | 主要随什么增长 | 量化后的直接影响 |
|---|---|---|
| 权重 | 参数量 | 常驻容量和权重读取带宽 |
| 激活 | 本轮 token/batch shape | 中间容量、通信和算子输入 |
| accumulator | 点积实现 | 数值范围和精度 |
| KV Cache | 活跃请求的历史 tokens | 并发容量与 attention 读取带宽 |

`W4A16` 通常表示 4-bit 权重、16-bit 激活；仍未说明 scale、zero point、打包布局、accumulator 和 KV dtype。因此不能从“权重四倍压缩”推出总显存四倍下降。

权重 payload 可粗写为

$$
B_{weight}=N_{param}b_w/8+B_{meta},
$$

而普通 KV payload 为

$$
B_{KV}=2LTN_{kv}De+B_{scale/alignment}.
$$

两者增长规律不同。

activation（激活）泛指模型 forward 中由当前输入产生的中间 tensor，例如 embedding、hidden states、Q/K/V、MLP 中间值和 layer output。激活函数（如 SiLU/GELU）是生成其中一部分激活的算子；二者有关但不是同义词。权重随模型保存，激活随请求和本轮 shape 产生。

显存（HBM）是 GPU 芯片外接/封装内的高带宽存储，不是执行乘加的 Tensor Core/CUDA core。简化路径是：HBM 中的权重/激活 → cache/shared memory/register → 计算单元 → 结果写回。decode 每次只处理少量 tokens，却要逐层读取大量权重，因此搬运常很重要。

W4A16 不要求先把整份 INT4 权重永久展开成 BF16 副本。专用 kernel 通常读取 packed INT4 与 group scales，在寄存器/片上 tile 内解包和缩放，与 BF16/FP16 activation 相乘，并用更高精度累加；展开值用完即丢弃。若 backend 不支持而真的预先全量反量化，容量优势会大幅丢失，这正是必须核对实际 kernel/fallback 的原因。

1024 个 INT4 参数的纯 payload 是 $1024\times4/8=512$ bytes。`group_size=128` 时有 8 groups；若每组保存 2-byte scale + 2-byte zero，则 metadata 32 bytes，总计 544 bytes。4096 个 INT4、group size 64 则 payload 2048 bytes、64 组 metadata 256 bytes、合计 2304 bytes。真实 checkpoint 还含 tensor headers、对齐和未量化参数。

## 3. FP8、INT8 与 INT4

FP8 和 INT8 都占 1 byte，只说明存储宽度相同：

- INT8 把 bits 解释为整数编码，常配合显式 scale/zero point；
- FP8 按符号、指数、尾数解释，E4M3 倾向更多有效数字，E5M2 倾向更大动态范围；
- INT4 只有 16 个编码，打包、group size 和反量化路径更关键。

同样减少一半 byte，延迟也不会自动减半。若某算子 80% 时间用于读权重、20% 是其他成本，即使权重读取理想减半，总时间也只是

$$
0.8/2+0.2=0.6,
$$

即约 1.67 倍；scale、解包、反量化和对齐还会改变结果。prefill 偏计算受限、decode 偏带宽受限时，同一量化方案的收益也可能不同。

| 格式 | 典型编码 | 动态范围/精度倾向 | 常见额外信息 |
|---|---|---|---|
| INT4 | 16 个整数编码 | 极窄，强依赖 group scale | packed layout、scale、可选 zero |
| INT8 | 256 个整数编码 | 均匀格点 | scale、可选 zero |
| FP8 E4M3 | 1 sign/4 exponent/3 mantissa | 较多尾数，范围较小 | tensor/block scale、饱和策略 |
| FP8 E5M2 | 1/5/2 | 范围更大，尾数更少 | 同上 |
| BF16 | 1/8/7 | 范围接近 FP32 | 通常无需显式量化 scale |

格式名仍不足以定义数值行为：是否保留 Inf/NaN、subnormal、有限值变体、舍入模式与 scaling recipe 都要看硬件和 kernel 契约。

## 4. AWQ、GPTQ 与 Marlin 是两层问题

AWQ 与 GPTQ 主要回答“怎样从浮点权重得到误差较小的低比特权重”：

- AWQ 使用校准激活统计识别更敏感的权重通道，并通过缩放等方法保护它们；`activation-aware` 不表示最终一定把激活存成 INT4；
- GPTQ 是训练后量化方法，利用二阶信息近似并在逐层处理时补偿已引入误差，不是简单逐元素四舍五入。

Marlin 更接近执行层：怎样让打包权重、scale/zero point 与半精度激活在 GPU 上高效完成量化矩阵乘。

```text
校准数据 → AWQ / GPTQ → 量化权重、scale、zero、布局
                                  │
输入激活 ─────────────────────────┼→ Marlin 等 kernel → 输出激活
```

因此 `GPTQ-Marlin` 并不矛盾：前者描述权重如何产生，后者描述兼容时怎样执行。两个 4-bit checkpoints 也未必可互换，元数据协议和打包布局可能不同。

vLLM v0.20.0 可从 `model_executor/layers/quantization/__init__.py` 的方法注册开始，继续检查对应 `QuantizationConfig` 的 activation dtype、最低计算能力、bits/group size 和 `get_quant_method`。注册存在只证明框架认识该方法；启动日志、实际层选择和 profiler 才能证明本次动态路径。

更细的责任图是：

```text
离线量化/模型制作者
  calibration samples
       ├─ AWQ：用激活统计寻找敏感通道，选择保护/缩放方案
       └─ GPTQ：逐层近似最小化输出误差，用 Hessian/二阶近似补偿
             ↓
  packed qweight + scales + zeros + group_size + metadata
             ↓ checkpoint load
vLLM QuantizationConfig
  验证格式/平台/activation dtype，给具体 layer 选择 quant method
             ↓
Marlin 或其他 execution kernel
  半精度 activation + packed low-bit weight
  → 解包/缩放/矩阵乘/高精度累加 → output activation
```

GPTQ 的直觉不是“更聪明的 round”一句话，而是某列权重量化产生误差后，根据近似 Hessian $H\approx X^TX$ 评估它对层输出的影响，并调整尚未量化的权重来补偿。AWQ 则利用激活观测到“少数权重通道对输出尤其重要”，通过等价缩放等方式降低这些通道量化误差。二者都不能仅靠名字推断 group size、zero point 或最终 kernel。

## 5. KV Cache 量化的容量账

以每 token 112 KiB 的 BF16 KV 为例，1-byte 格式的理论 payload 为 56 KiB/token。4096 tokens 从 448 MiB 降到 224 MiB，但真实运行时至少还有：

$$
B_{runtime}=B_{KV\ payload}+B_{scales}+B_{block\ alignment}+B_{allocator/reserve}.
$$

更多可用 byte 能否变成更多 blocks，还取决于启动 profiling 后剩余显存、block size、backend layout、graph/workspace 和模型常驻对象。

K 的量化误差会改变 attention logits，V 的误差会改变加权结果，并经过后续投影和层继续传播。因此验证至少分成：

1. 质量：固定 tokenized prompts 和采样，比较 logits、输出或任务指标；
2. 容量：记录分配 blocks、最大 token 容量和 OOM 边界；
3. 性能：固定到达过程和输入输出长度，比较 TTFT、ITL、吞吐。

CLI 能解析 `--kv-cache-dtype` 不等于具体 backend 能消费该格式。应沿 `CacheConfig.cache_dtype`、KV quant mode、backend capability 和实际 forward/log 逐层确认。

容量核算器：

```python
from math import ceil

def kv_capacity(pool_bytes, *, layers, kv_heads, head_dim,
                bits, block_size, scale_bytes_per_block=0):
    payload_per_token = 2 * layers * kv_heads * head_dim * bits / 8
    payload_per_block = payload_per_token * block_size
    block_bytes = payload_per_block + scale_bytes_per_block
    blocks = int(pool_bytes // block_bytes)
    return {"bytes_per_token_payload": payload_per_token,
            "blocks": blocks,
            "token_slots": blocks * block_size,
            "unused_pool_bytes": pool_bytes - blocks * block_bytes}

bf16 = kv_capacity(1 << 30, layers=28, kv_heads=8, head_dim=128,
                   bits=16, block_size=16)
fp8 = kv_capacity(1 << 30, layers=28, kv_heads=8, head_dim=128,
                  bits=8, block_size=16)
assert bf16["bytes_per_token_payload"] == 112 * 1024
assert fp8["bytes_per_token_payload"] == 56 * 1024
assert fp8["token_slots"] == 2 * bf16["token_slots"]
```

在假设 scale overhead 为 0 且 pool 固定 1 GiB 时，slot 数理想翻倍。真实 pool 预算、scale 粒度、对齐、backend 和 allocator 取整会使结果偏离；更多 slots 也只有在 workload 受 KV 容量限制时才可能提高并发。

### v0.20.0 中 KV dtype、scale 与真实分配在哪里

按固定版本从配置到执行核对：

```text
CLI / EngineArgs: --kv-cache-dtype
  → CacheConfig.cache_dtype                 # 用户意图/规范化配置
  → v1/kv_cache_interface.py quant mode     # 按 dtype 分类 cache spec
  → AttentionBackend.supports_kv_cache_dtype
  → 具体 backend.supported_kv_cache_dtypes  # 静态能力与 layout 约束
  → GPUModelRunner / worker 初始化 cache tensors、scale 路径
  → 启动日志 + forward trace                # 动态选择证据
```

量化方式、scale 来源/布局不能只从一个 flag 推出：可能来自 checkpoint tensor/metadata、cache config、backend 定义和 runner 初始化。`calculate_kv_scales` 等旧配置在固定版本若已有弃用提示，就应继续查 backend/checkpoint，而不是沿用旧版经验。

“初始化时已经分好 blocks”与“请求时还要分配”指两个层次：engine 启动时预分配整个 GPU KV pool 和固定 physical block slots；请求到来时，Scheduler 只是在这批既有 slots 中给请求分配 logical ownership/block ids。后者通常不新建 CUDA 大 tensor，但会修改 free queue、ref count 与 block table。

## 6. 推测解码减少目标模型串行轮次

普通 decode 必须先得到 $x_1$ 才能构造 $x_2$ 的上下文。推测解码让较便宜的 proposer 先给出 $k$ 个候选，再让 target model 一次并行验证多个位置：

```text
proposer:     x1? → x2? → x3?
target:       [x1?, x2?, x3?] 一次前向获得各位置 logits
sampler:      从左到右接受；首次拒绝后，后续候选全部作废
```

后一个候选依赖前一个候选作为上下文，所以不能跳过前面的拒绝继续接受后面的 token。

大小模型的概率不是“提前全知道”。proposer 每生成一个 draft token 时保留该位置的 $q(\cdot\mid context)$ 或至少候选概率；拿到整段 draft 后，target 用一次 batched forward 对这些候选上下文并行产生各位置的 $p(\cdot\mid context)$。然后 sampler 才同时拥有对应位置的 p/q，逐位置做接受检验。

```text
小模型串行提出 d1,d2,d3，并保存 q1(d1),q2(d2),q3(d3)
              ↓
大模型对 [context+d1+d2+d3] 一次前向
得到 p1(.),p2(.),p3(.) 以及下一位置 p4(.)
              ↓
从 d1 开始接受/拒绝
```

target “一次并行验证”仍计算每个候选位置的 logits，并非不用大模型。

## 7. 拒绝采样为什么保持目标分布

设 proposer 分布为 $q$，target 分布为 $p$。先从 $q$ 抽到候选 $x$，以

$$
a(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
$$

接受。候选直接成为输出 $x$ 的概率质量为

$$
q(x)a(x)=\min(q(x),p(x)).
$$

若拒绝，从正残差分布采样：

$$
r(x)=
\frac{\max(p(x)-q(x),0)}
{\sum_y\max(p(y)-q(y),0)}.
$$

取

$$
p=[0.6,0.3,0.1],\qquad q=[0.3,0.4,0.3].
$$

直接接受质量是 `[0.3,0.3,0.1]`，总计 0.7；拒绝质量为 0.3。正残差归一化为 `[1,0,0]`，最终：

$$
[0.3,0.3,0.1]+0.3[1,0,0]
=[0.6,0.3,0.1]=p.
$$

因此“先猜再验”在使用正确接受/修正规则时不会把最终随机采样分布替换成 $q$。

源码中的三类结果要分开：accepted 是从左到右连续通过的 draft tokens；recovered 是首次拒绝处从残差分布得到的 token；bonus 只可能在所有候选接受后附加。

证明中的关键恒等式是

$$
\sum_x\min(p_x,q_x)=1-\sum_x(p_x-q_x)_+
=1-\sum_x(q_x-p_x)_+.
$$

令拒绝总概率 $Z=\sum_x(p_x-q_x)_+$。最终输出 $x$ 的质量为

$$
\underbrace{\min(p_x,q_x)}_{\text{直接接受}}
+\underbrace{Z\frac{(p_x-q_x)_+}{Z}}_{\text{拒绝后恢复}}
=\min(p_x,q_x)+(p_x-q_x)_+=p_x.
$$

### 精确枚举与 Monte Carlo

```python
import random

p = [0.6, 0.3, 0.1]
q = [0.3, 0.4, 0.3]
accepted_mass = [min(px, qx) for px, qx in zip(p, q)]
reject_prob = 1.0 - sum(accepted_mass)
positive = [max(px-qx, 0.0) for px, qx in zip(p, q)]
residual = [x/sum(positive) for x in positive]
exact = [a + reject_prob*r for a, r in zip(accepted_mass, residual)]
assert all(abs(a-b) < 1e-12 for a, b in zip(exact, p))

rng = random.Random(7)
counts = [0, 0, 0]
for _ in range(300_000):
    x = rng.choices(range(3), weights=q)[0]
    if rng.random() < min(1.0, p[x]/q[x]):
        out = x
    else:
        out = rng.choices(range(3), weights=residual)[0]
    counts[out] += 1
empirical = [n/sum(counts) for n in counts]
print(exact, empirical)
assert max(abs(a-b) for a, b in zip(empirical, p)) < 0.005
```

输入是同一上下文下 target 分布 $p$ 与 proposer 分布 $q$；输出是一个最终 token。循环状态只有计数器。枚举证明单 token 修正分布，Monte Carlo 检查实现。真实推测解码对一串候选从左到右重复条件化，并处理数值稳定、batch、top-k/p、grammar、bonus 和 KV commit；这段代码不证明完整 serving path。

完整候选时间线：

```text
draft: d1 d2 d3 d4
verify d1: accept
verify d2: accept
verify d3: reject → sample recovered r3
d4: 依赖被拒的 d3，直接作废
本轮提交: [d1,d2,r3]

若 d1..d4 全接受：可再从 target 的下一个位置取 bonus token
```

`accepted=2` 并不表示目标模型“只算了 2 个位置”；target 可能已并行给出候选位置 logits，但控制状态/KV 只能提交保持正确上下文的连续接受前缀和恢复 token。

第三个候选被拒绝时，并不缺“第三个位置其他 token 的概率”：target forward 已产生完整 $p_3(\cdot)$，proposer 也提供 $q_3(\cdot)$。sampler 用两者构造正残差 $[p_3-q_3]_+$，从中抽 recovered token。第四个候选依赖被拒绝的第三个上下文，所以作废；下一轮以 recovered token 为真实历史继续。

## 8. 接受率高仍不保证更快

粗略成本为

$$
\text{cost per final token}
\approx
\frac{C_{draft}+C_{verify}+C_{sampling}}
{E[N_{output\ per\ step}]}.
$$

只有每轮有效输出数的增长快于 proposer、target verification、logits、KV 和 padding 成本，端到端才加速。大并发下，target decode 本来就可能填满 GPU，多验证候选还会抢占其他请求的 token budget。

v0.20.0 可从 `config/speculative.py::SpeculativeConfig` 看候选数、method 和 draft 配置，从 `v1/spec_decode` 看 proposer，从 `v1/sample/rejection_sampler.py` 看验证与拒绝采样。

## 9. 可归因的优化实验

不要同时打开五个开关再把全部收益归给其中一个。先固定模型、commit、GPU、tokenized workload、到达率、采样、warm-up 和测量窗口，然后每次改变一个主变量：

| 实验 | 权重 | KV | 推测解码 | 问题 |
|---|---|---|---|---|
| A | BF16 | BF16 | 关闭 | 基线 |
| B | INT4 | BF16 | 关闭 | 权重量化净影响 |
| C | BF16 | FP8 | 关闭 | KV 量化净影响 |
| D | INT4 | FP8 | 关闭 | 两种量化交互 |
| E | INT4 | FP8 | 开启 | 推测解码增量 |

同时记录正确性、峰值显存、TTFT/ITL/E2E 分位数、request/token throughput、preemption、cache hit、实际 backend/kernel 和 speculative acceptance。吞吐提高而 P99 TTFT 变差、blocks 增加但质量下降、接受率高却不加速，都是解释机制的重要反指标。

实验还需设置门槛：先确认同一 tokenized workload、相同 sampling/seed（或统计等价）、无错误/OOM，再比较性能。量化实验要报告模型质量或 logits 差异；推测解码要报告 target-call 次数、draft/accepted/recovered/bonus tokens、每轮有效输出和 verification batch shape。否则“acceptance=90%”无法说明 proposer 占了多少时间，也无法解释为何没有加速。

## 常见误区

- “INT4 模型的所有显存都变为四分之一”：激活、KV、未量化层和 workspace 不会同比变化。
- “FP8 就是占 1 byte 的 INT8”：编码规则与动态范围不同。
- “checkpoint 更小必然更快”：执行 kernel、shape 和瓶颈决定速度。
- “speculative acceptance 就是 APC hit”：一个是候选验证，一个是前缀 KV 复用。

## 本篇总结

量化应先回答“哪个对象、什么格式、何种 scale、由哪个 kernel 消费”；推测解码应先回答“proposer 成本、target 验证和每轮有效输出如何平衡”。任何开关都只能在固定 workload 和正确性门槛下，通过容量、延迟与吞吐的对照实验建立收益。

---

[上一篇：Scheduler]({{ '/vllm/scheduler/' | relative_url }}) · [系列首页]({{ '/vllm/' | relative_url }}) · [下一篇：FlashAttention]({{ '/vllm/flashattention/' | relative_url }})

## 资料

- [AWQ](https://arxiv.org/abs/2306.00978) 与 [GPTQ](https://arxiv.org/abs/2210.17323)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [vLLM v0.20.0 源码](https://github.com/vllm-project/vllm/tree/v0.20.0)
