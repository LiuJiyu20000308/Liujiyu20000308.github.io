---
layout: page
title: 从神经网络基础到视觉与 Transformer：D2L 与 PyTorch 学习总结
permalink: /deep-learning/
lang: zh-CN
math: true
---

这组文章面向已经粗略学过一遍深度学习、但知识仍散落在公式、Notebook 和 API 之间的读者。它不按函数名罗列知识点，而是沿着一条问题驱动的路线重新组织：张量如何承载数据，梯度如何穿过计算图，线性模型为何能回归却难以表达复杂边界，非线性网络如何获得容量并保持泛化，工程代码怎样成为可复现的训练系统，为什么从 LeNet 一路演化出 ResNet 和 DenseNet，这些视觉表示如何进一步支持目标检测、语义分割和图像生成，以及注意力机制怎样从 QKV 汇聚发展为多头自注意力与 Transformer。

学习边界包括 D2L 的预备知识、线性神经网络、多层感知机、深度学习计算、卷积基础、现代 CNN、计算性能、视觉微调、目标检测、语义/实例分割和神经风格迁移，以及注意力汇聚、Bahdanau 注意力、自注意力、多头注意力、位置编码与 Transformer。

## 知识路线

整条路线可以压缩成下面的依赖链：

$$
\text{张量与自动微分}
\rightarrow \text{线性回归与分类}
\rightarrow \text{MLP 与泛化}
\rightarrow \text{可复现工程流程}
\rightarrow \text{卷积与现代 CNN}
\rightarrow \text{并行与分布式训练}
\rightarrow \text{迁移学习与微调}
\rightarrow \text{检测、分割与生成}.
$$

注意力机制形成另一条从“动态读取”走向并行序列建模的路线：

$$
\text{QKV 与注意力汇聚}
\rightarrow \text{masked softmax 与评分函数}
\rightarrow \text{Bahdanau 动态上下文}
\rightarrow \text{多头自注意力与位置编码}
\rightarrow \text{Transformer 编码器—解码器}.
$$

其中，“训练基础”给出所有模型共享的执行机制；“线性模型”建立损失与概率解释；“MLP”回答表达能力、泛化和稳定性；“深度学习计算与工程实践”完整对应 D2L 的 `chapter_deep-learning-computation`，把层与块、参数、延后初始化、自定义层、读写文件和 GPU 落到可保存、可迁移、可复现的代码；“CNN 演进”观察分类骨干如何在真实约束下逐步演化，视觉任务文章再把共享特征扩展为锚框、RoI 和逐像素预测；注意力文章则从评分、掩码和 tensor shape 出发，解释自注意力、多头并行与 Transformer 编解码器如何组成完整序列模型。

## 推荐阅读顺序

1. [PyTorch 与神经网络训练基础]({{ '/deep-learning/foundations/' | relative_url }})
   从 tensor shape、广播与矩阵乘法出发，贯通 `Dataset`、`DataLoader`、自动微分、梯度累积以及完整训练/验证循环。

2. [从线性回归到 softmax 分类]({{ '/deep-learning/linear-models/' | relative_url }})
   从连续值预测走到 logits、softmax、交叉熵，并分清二分类、多分类、多标签和不同损失函数的边界。

3. [多层感知机、泛化与稳定训练]({{ '/deep-learning/mlp-generalization/' | relative_url }})
   解释非线性为何必要，怎样识别欠拟合与过拟合，以及正则化、初始化、BatchNorm 和残差连接如何让深层网络可训练。

4. [PyTorch 深度学习计算与工程实践]({{ '/deep-learning/pytorch-engineering/' | relative_url }})
   完整覆盖 D2L“深度学习计算”：层与块、参数管理、延后初始化、自定义层、读写文件和 GPU，并将它们与数据预处理、验证和环境隔离组合成可复现建模流程。

5. [CNN：从卷积到 DenseNet]({{ '/deep-learning/cnn-evolution/' | relative_url }})
   以“上一代遇到什么问题”为线索，串起卷积基础、LeNet、AlexNet、VGG、NiN、GoogLeNet、BatchNorm、ResNet 和 DenseNet。

6. [D2L 计算性能：从硬件、异步执行到多机 DDP]({{ '/deep-learning/computational-performance/' | relative_url }})
   从 CPU、GPU 和数据搬运出发，梳理 PyTorch 自动并行与异步执行、单机多卡、DP/DDP 选择、NCCL All-Reduce 以及多机训练。

7. [D2L 微调：从迁移学习到热狗识别]({{ '/deep-learning/fine-tuning/' | relative_url }})
   解释预训练表示如何迁移到小数据任务，以热狗二分类串起差分学习率、复用ImageNet类别权重、冻结部分网络及微调实验结论。

8. [计算机视觉任务全景：从数据增强、目标检测到语义分割与风格迁移]({{ '/deep-learning/computer-vision-tasks/' | relative_url }})
   从 shape、坐标和监督对齐出发，贯通边界框、锚框、SSD、R-CNN 系列、RoI Align、转置卷积、FCN、VOC 数据与神经风格迁移，并收束到 Kaggle 数据工程流程。

9. [D2L 注意力机制：从 QKV、多头自注意力到 Transformer]({{ '/deep-learning/attention-transformer/' | relative_url }})
   从查询、键、值和 masked softmax 出发，解释加性与缩放点积评分、Bahdanau 动态上下文、自注意力、多头拆分、位置编码，以及 Transformer 编码器、因果解码器和预测缓存的完整 shape 流程。

## 如何使用这组文章

- 如果代码能运行但总分不清 shape，先读第一篇，再用第二篇核对损失输入输出。
- 如果训练误差下降、验证误差却恶化，重点读第三篇的模型选择、数据泄漏和正则化。
- 如果不清楚 `nn.Module` 怎样管理参数，Notebook 换环境就报错，或不知道该保存什么，直接查第四篇。
- 如果只记得经典网络名字，却说不清为什么会有下一代架构，从第五篇的因果链开始。
- 如果目标数据不多，不确定应该从零训练、冻结骨干还是完整微调，阅读第七篇。
- 如果分不清锚框与特征图、RoI Pooling 与 RoI Align、转置卷积与上采样，或不知道检测和分割 loss 的 shape，阅读第八篇。
- 如果分不清 score、weight 和 output，QKV 的来源，多头为何拆维，`transpose_qkv` 是否混合 batch，或解码器怎样屏蔽未来并读取编码器输出，阅读第九篇。

每篇末尾都有知识链回顾、常见误区、前后篇导航和 D2L 对应章节。系列中的公式服务于理解和实现，不追求教材式的完整证明；代码则尽量短，并明确关键 tensor shape。

## 主要资料

- [《动手学深度学习》中文版](https://zh.d2l.ai/)
- [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/)
- [系列第一篇：PyTorch 与神经网络训练基础]({{ '/deep-learning/foundations/' | relative_url }})
- [计算机视觉任务全景]({{ '/deep-learning/computer-vision-tasks/' | relative_url }})
- [注意力机制：从 QKV 到 Transformer]({{ '/deep-learning/attention-transformer/' | relative_url }})
