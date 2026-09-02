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

## 6. 推测解码减少目标模型串行轮次

普通 decode 必须先得到 $x_1$ 才能构造 $x_2$ 的上下文。推测解码让较便宜的 proposer 先给出 $k$ 个候选，再让 target model 一次并行验证多个位置：

```text
proposer:     x1? → x2? → x3?
target:       [x1?, x2?, x3?] 一次前向获得各位置 logits
sampler:      从左到右接受；首次拒绝后，后续候选全部作废
```

后一个候选依赖前一个候选作为上下文，所以不能跳过前面的拒绝继续接受后面的 token。

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
