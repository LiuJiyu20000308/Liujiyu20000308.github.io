---
layout: page
title: 从线性模型到现代卷积神经网络：D2L 与 PyTorch 学习总结
permalink: /deep-learning/
lang: zh-CN
math: true
---

这组文章面向已经粗略学过一遍深度学习、但知识仍散落在公式、Notebook 和 API 之间的读者。它不按函数名罗列知识点，而是沿着一条问题驱动的路线重新组织：张量如何承载数据，梯度如何穿过计算图，线性模型为何能回归却难以表达复杂边界，非线性网络如何获得容量并保持泛化，工程代码怎样成为可复现的训练系统，最后又为什么从 LeNet 一路演化出 AlexNet、VGG、NiN、GoogLeNet、ResNet 和 DenseNet。

学习边界到现代卷积神经网络结束，包括 D2L 的预备知识、线性神经网络、多层感知机、深度学习计算、卷积基础和现代 CNN；不进入 RNN、GRU、LSTM 等循环神经网络。

## 知识路线

整条路线可以压缩成下面的依赖链：

$$
\text{张量与自动微分}
\rightarrow \text{线性回归与分类}
\rightarrow \text{MLP 与泛化}
\rightarrow \text{可复现工程流程}
\rightarrow \text{卷积与现代 CNN}.
$$

其中，“训练基础”给出所有模型共享的执行机制；“线性模型”建立损失与概率解释；“MLP”回答表达能力、泛化和稳定性；“工程实践”把概念落到可保存、可迁移、可复现的代码；“CNN 演进”则把这些组件放进图像任务，观察架构如何在真实约束下逐步演化。

## 推荐阅读顺序

1. [PyTorch 与神经网络训练基础]({{ '/deep-learning/foundations/' | relative_url }})
   从 tensor shape、广播与矩阵乘法出发，贯通 `Dataset`、`DataLoader`、自动微分、梯度累积以及完整训练/验证循环。

2. [从线性回归到 softmax 分类]({{ '/deep-learning/linear-models/' | relative_url }})
   从连续值预测走到 logits、softmax、交叉熵，并分清二分类、多分类、多标签和不同损失函数的边界。

3. [多层感知机、泛化与稳定训练]({{ '/deep-learning/mlp-generalization/' | relative_url }})
   解释非线性为何必要，怎样识别欠拟合与过拟合，以及正则化、初始化、BatchNorm 和残差连接如何让深层网络可训练。

4. [PyTorch 工程实践与完整建模流程]({{ '/deep-learning/pytorch-engineering/' | relative_url }})
   将层、块、参数、保存、GPU、数据预处理、验证和环境隔离组合成一个可复现的建模流程。

5. [CNN：从卷积到 DenseNet]({{ '/deep-learning/cnn-evolution/' | relative_url }})
   以“上一代遇到什么问题”为线索，串起卷积基础、LeNet、AlexNet、VGG、NiN、GoogLeNet、BatchNorm、ResNet 和 DenseNet。

## 如何使用这组文章

- 如果代码能运行但总分不清 shape，先读第一篇，再用第二篇核对损失输入输出。
- 如果训练误差下降、验证误差却恶化，重点读第三篇的模型选择、数据泄漏和正则化。
- 如果 Notebook 换环境就报错，或不知道该保存什么，直接查第四篇。
- 如果只记得经典网络名字，却说不清为什么会有下一代架构，从第五篇的因果链开始。

每篇末尾都有知识链回顾、常见误区、前后篇导航和 D2L 对应章节。系列中的公式服务于理解和实现，不追求教材式的完整证明；代码则尽量短，并明确关键 tensor shape。

## 主要资料

- [《动手学深度学习》中文版](https://zh.d2l.ai/)
- [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/)
- [系列第一篇：PyTorch 与神经网络训练基础]({{ '/deep-learning/foundations/' | relative_url }})
