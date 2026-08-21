---
layout: post
title: "计算机视觉任务全景：从数据增强、目标检测到语义分割与风格迁移"
date: 2026-08-19 09:00 +0800
tags: [D2L, PyTorch, 深度学习, CNN, 计算机视觉, 目标检测, 语义分割]
toc: true
math: true
permalink: /deep-learning/computer-vision-tasks/
---

## 本篇逻辑主线

图像分类只需要回答“整张图是什么”，而真实视觉任务还要回答“目标在哪里”“每个像素是什么”“同类目标是不是同一个实例”，甚至“怎样生成一张同时保留内容和风格的新图”。任务一旦从整图标签走向空间预测，CNN 输出就不能过早压缩成一个向量：目标检测要在多个尺度上为大量候选框预测类别和偏移，语义分割要把低分辨率语义特征恢复到逐像素输出，实例分割还要保持 RoI 与原图的精确对齐。

本文把 D2L `chapter_computer-vision` 中除已有独立文章的微调之外的内容，整理成一条统一知识链：

```text
数据与增广
    ↓
边界框、IoU、锚框与偏移量
    ↓
多尺度特征图与 SSD
    ↓
R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN
    ↓
语义分割数据 → 转置卷积 → FCN
    ↓
固定预训练网络、直接优化像素 → 神经风格迁移
```

看似不同的模型都在处理同一组约束：**shape 如何对应任务语义，空间坐标怎样在不同分辨率之间映射，训练标签如何与预测槽位对齐，哪些计算能够共享，损失应当作用于哪些有效样本。**

### 对应 Notebook

| 主题 | Notebook | 本文位置 |
|---|---|---|
| 数据增广 | `image-augmentation.ipynb` | 第 2 节 |
| 边界框与检测数据 | `bounding-box.ipynb`、`object-detection-dataset.ipynb` | 第 3 节 |
| 锚框、IoU、NMS | `anchor.ipynb` | 第 4～5 节 |
| 多尺度检测与 SSD | `multiscale-object-detection.ipynb`、`ssd.ipynb` | 第 6～7 节 |
| R-CNN 系列 | `rcnn.ipynb` | 第 8 节 |
| 语义分割数据 | `semantic-segmentation-and-dataset.ipynb` | 第 9 节 |
| 转置卷积与 FCN | `transposed-conv.ipynb`、`fcn.ipynb` | 第 10～11 节 |
| 神经风格迁移 | `neural-style.ipynb` | 第 12 节 |
| 比赛工程流程 | `kaggle-cifar10.ipynb`、`kaggle-dog.ipynb` | 第 13 节 |

## 1. 先用输出 shape 区分视觉任务

设输入图像批量为

```text
X: [N, 3, H, W]
```

不同任务对输出的要求完全不同：

| 任务 | 典型输出 | 一个输出单元表示什么 |
|---|---|---|
| 图像分类 | `[N,K]` | 一张图属于每个类别的 logits |
| 目标检测 | 每张图若干 `[class,score,x1,y1,x2,y2]` | 一个目标实例的类别与框 |
| 语义分割 | `[N,K,H,W]` | 每个像素属于每个类别的 logits |
| 实例分割 | 检测结果 + 每个实例的 mask | 同类目标也分别拥有独立掩码 |
| 风格迁移 | `[N,3,H,W]` | 直接生成的新图像 |

分类模型最终可以通过全局平均池化把空间维压成 `1×1`。检测和分割不能这样做：它们必须保留或重建空间位置。由此产生后续三条核心路径：

1. **检测**：把特征图的每个空间位置解释为一组候选框中心。
2. **分割**：把每个空间位置解释为一个像素类别预测，并上采样回原图。
3. **实例任务**：在共享特征图上裁取每个 RoI，同时保持坐标对齐。

## 2. 数据与增广：空间任务必须同步变换标签

### 2.1 输入图片为什么先 `/255` 再 Normalize

`torchvision.io.read_image` 返回 `uint8` 张量，shape 通常为 `[3,H,W]`，像素值在 0～255。预训练 CNN 常使用基于 0～1 像素统计得到的均值和标准差，因此预处理顺序是：

```python
image = image.float() / 255
image = normalize(image)
```

按通道写作

$$
x'_{c,h,w}=\frac{x_{c,h,w}/255-\mu_c}{\sigma_c}.
$$

训练和预测必须使用同一套预处理。这里的 `/255` 归一化的是颜色强度；检测数据中 `/256` 则可能是在 256 像素宽高的图片上归一化边界框坐标，二者不能混用。

### 2.2 分类、检测和分割的数据增广不同

随机翻转、裁剪、颜色抖动等增广的作用不是简单“制造更多图片”，而是把合理的不变性写进训练分布：

- 分类中，水平翻转后类别通常不变，颜色变化也常不改变物体类别。
- 检测中，几何变换必须同步更新边界框；若裁剪只剩目标极小部分，还要决定删除、截断还是保留该框。
- 分割中，图片与标签 mask 必须使用完全相同的随机裁剪和翻转；颜色抖动只作用于输入照片，不能改变类别 mask 的颜色编码。

同步随机裁剪的核心接口类似：

```python
image, mask = random_crop(image, mask, height, width)
```

而不是分别调用两次随机裁剪，否则图片位置和像素标签会错位。

### 2.3 训练集、验证集和测试集不能混用

Kaggle 两个实战 Notebook 反复强调同一条工程边界：训练集用于更新参数，验证集用于选择模型与超参数，测试集只用于最终预测。重组文件夹、复制图片只是为了适配 `ImageFolder` 的目录接口；真正重要的是划分协议不泄漏。

数据增强通常只用于训练集。验证与测试应使用确定性的 resize、center crop 和 normalization，否则同一模型每次评估的输入分布都在随机变化。

## 3. 边界框与检测数据：先统一坐标语义

### 3.1 两种常见边界框格式

角点格式：

```text
[x_min, y_min, x_max, y_max]
```

中心格式：

```text
[center_x, center_y, width, height]
```

二者转换为

$$
c_x=\frac{x_{min}+x_{max}}2,\qquad
c_y=\frac{y_{min}+y_{max}}2,
$$

$$
w=x_{max}-x_{min},\qquad
h=y_{max}-y_{min}.
$$

反向转换：

$$
x_{min}=c_x-\frac w2,\quad x_{max}=c_x+\frac w2,
$$

$$
y_{min}=c_y-\frac h2,\quad y_{max}=c_y+\frac h2.
$$

代码中必须确认函数期望哪一种格式。IoU、绘图和 NMS 常用角点格式，边界框偏移编码常先转成中心格式。

### 3.2 归一化坐标与像素坐标

对于宽为 $W$、高为 $H$ 的图片，角点坐标可以归一化为：

```python
scale = torch.tensor([W, H, W, H])
normalized_box = pixel_box / scale
pixel_box = normalized_box * scale
```

归一化后，同一组坐标可以独立于具体图片大小表示相对位置。完整图片的连续边界可写为 `[0,0,W,H]`，因此空间坐标除以图片尺寸；这与像素颜色最大值 255 没有关系。

### 3.3 COCO 为什么用一个 JSON 管理标注

COCO 检测标注主要由三张逻辑表组成：

```text
annotations.image_id    → images.id
annotations.category_id → categories.id
```

其中 `annotations` 的 `bbox` 使用

```text
[x_min, y_min, width, height]
```

而不是 `xyxy`。转换时要显式计算：

```python
x, y, w, h = annotation['bbox']
box_xyxy = [x, y, x + w, y + h]
```

原始 COCO 的类别编号也不是连续的 0～79，训练时通常建立从原始 `category_id` 到连续类别下标的映射。图片编号、标注编号和类别编号是三种不同 ID，不能通过文件名或数组位置猜测它们的关系。

## 4. 锚框：把连续检测问题变成大量固定预测槽位

目标数量不固定，但卷积网络需要固定 shape 的输出。锚框提供了一组预先定义的参考框：网络不从零生成框，而是对每个参考框预测“是什么”和“应当怎样调整”。

### 4.1 一个特征位置为什么生成多个框

同一中心可能出现小物体、大物体、横向物体或纵向物体，因此每个位置生成多个尺度 $s$ 和宽高比 $r$ 的锚框。若以图像高度 $H$ 为基准，像素宽高可定义为

$$
w_a=Hs\sqrt r,\qquad h_a=\frac{Hs}{\sqrt r}.
$$

于是

$$
\frac{w_a}{h_a}=r,\qquad w_ah_a=(Hs)^2.
$$

在非正方形图片中，横纵坐标分别按 $W,H$ 归一化，所以归一化宽度还要乘 $H/W$，才能保持像素意义上的宽高比。

### 4.2 `multibox_prior` 的 shape

若特征图空间大小为 $H_f\times W_f$，每个位置生成 $A$ 个锚框，则输出为

```text
anchors: [1, H_f × W_f × A, 4]
```

最后一维是归一化的 `[xmin,ymin,xmax,ymax]`。最前面的 1 是共享锚框模板的批量维；`anchors[0]` 取出第一组中的全部锚框，而 `anchors[0,0]` 才是第一个锚框。

靠近图片边缘的大锚框可能出现小于 0 或大于 1 的坐标，这是正常现象。边界处的真实目标也可能被图片截断；训练匹配时通常保留这些锚框，最终显示或输出时再裁到图片范围。

## 5. 从锚框到训练标签：IoU、偏移量和 NMS

### 5.1 IoU 衡量两个框的重叠

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

对两组框 `boxes1:[N,4]`、`boxes2:[M,4]`，两两 IoU 的输出是 `[N,M]`。代码中的

```python
boxes1[:, None, :2]
```

把 `[N,2]` 变成 `[N,1,2]`，与 `[M,2]` 广播后得到 `[N,M,2]`，从而一次计算所有框对的交集左上角。`None` 等价于 `unsqueeze(1)`。

### 5.2 锚框分配需要阈值匹配和强制匹配

典型规则包括：

1. 对每个锚框，若它与某个真实框的最大 IoU 高于阈值，就分配给该真实框。
2. 对每个真实框，额外强制分配当前 IoU 最大的尚可用锚框，保证小目标或形状特殊的目标不至于完全没有正样本。
3. 未匹配锚框作为背景；某些实现还会把阈值中间区域标为 ignore。

类别标签通常把 0 留给背景，所以真实目标类别会整体加 1。

### 5.3 为什么偏移量这样编码

设锚框中心与宽高为 $(x_a,y_a,w_a,h_a)$，匹配真实框为 $(x_g,y_g,w_g,h_g)$。D2L 示例使用

$$
t_x=10\frac{x_g-x_a}{w_a},\qquad
t_y=10\frac{y_g-y_a}{h_a},
$$

$$
t_w=5\log\frac{w_g}{w_a},\qquad
t_h=5\log\frac{h_g}{h_a}.
$$

中心差除以锚框宽高，使平移成为与绝对尺寸无关的相对位移；宽高使用比值再取对数，使放大与缩小成为近似对称的加性变量；10 和 5 是调整目标尺度的经验因子，解码时必须对应除回去：

$$
\hat x=x_a+\frac{\hat t_x}{10}w_a,\qquad
\hat y=y_a+\frac{\hat t_y}{10}h_a,
$$

$$
\hat w=w_a\exp(\hat t_w/5),\qquad
\hat h=h_a\exp(\hat t_h/5).
$$

每个锚框有 4 个回归分量，所以 5 个锚框的 `bbox_mask` 有 $5\times4=20$ 个元素。背景锚框对应的四项全为 0，它们不参与边界框回归损失。

### 5.4 NMS 删除同一目标的重复预测

非极大值抑制（NMS）按置信度从高到低处理预测框：保留最高分框，删除与它 IoU 超过阈值的同类框，再处理剩余框。它解决的是“多个锚框同时命中一个目标”，而不是提高分类概率。

```text
分类阈值：过滤不可信预测
NMS 阈值：过滤与高分框过度重叠的重复预测
```

两个阈值作用不同。NMS 一般按类别分别执行，否则一只猫和一只高度重叠的狗可能被错误地互相抑制。

## 6. 多尺度特征图：小目标密集预测，大目标稀疏预测

特征图的空间位置决定锚框中心，特征值则用于预测这些锚框的类别和偏移量。若一个特征层为 `[N,C,H_f,W_f]`，每个位置有 $A$ 个锚框，就会生成 $H_fW_fA$ 个预测槽位。

浅层或高分辨率特征图通常搭配小锚框：中心密集，能覆盖小目标可能出现的更多位置。深层或低分辨率特征图通常搭配大锚框：中心较少，但单元感受野更大，适合判断大目标。

| 特征图 | 每位置锚框数 | 总锚框数 | 常见用途 |
|---|---:|---:|---|
| `4×4` | 3 | 48 | 较小、较密集的框 |
| `2×2` | 3 | 12 | 中等框 |
| `1×1` | 3 | 3 | 覆盖大范围的框 |

这种配对不是 `multibox_prior` 自动推断的：模型设计者为每个特征层指定 `sizes`，网络训练后才学会利用该层特征完成相应尺度的预测。

## 7. SSD：一次前向传播完成多尺度密集检测

### 7.1 分类与回归预测头的通道数

设每个位置有 $A$ 个锚框、目标类别数为 $K$。使用 softmax 分类时，每个锚框需要 $K+1$ 个分数，其中额外一类是背景，因此分类卷积输出通道为

$$
A(K+1).
$$

```python
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(
        num_inputs,
        num_anchors * (num_classes + 1),
        kernel_size=3,
        padding=1,
    )
```

边界框预测每个锚框需要 4 个数，所以输出通道为 $4A$。

卷积输出暂时把“锚框”和“类别”折叠进通道维：

```text
[N, A(K+1), H, W]
```

后续必须先变成 NHWC 顺序再展开：

```python
pred = pred.permute(0, 2, 3, 1)
pred = torch.flatten(pred, start_dim=1)
```

这样线性序列按“位置 → 锚框 → 类别”排列，最后才能安全地恢复为：

```python
cls_preds = cls_preds.reshape(N, -1, K + 1)
# [N, 所有尺度的锚框总数, K+1]
```

若不先 `permute`，NCHW 直接展开会先放完某个通道的所有空间位置，每连续 $K+1$ 个数就不再属于同一个锚框。

### 7.2 TinySSD 的完整 shape

D2L 的 TinySSD 对 `256×256` 输入使用五个尺度，空间大小为：

```text
32×32, 16×16, 8×8, 4×4, 1×1
```

每个位置生成 4 个锚框，总数为

$$
(32^2+16^2+8^2+4^2+1^2)\times4=5444.
$$

香蕉数据集只有一个目标类别，因此输出为：

```text
anchors   : [1, 5444, 4]
cls_preds : [N, 5444, 2]       # 背景、香蕉
bbox_preds: [N, 5444 × 4]
```

### 7.3 SSD 的训练损失

类别损失对锚框做交叉熵，边界框损失只在正锚框掩码处计算：

$$
L=L_{cls}+\lambda L_{bbox}.
$$

简化实现为：

```python
cls = cross_entropy(cls_preds, cls_labels)
bbox = l1_loss(bbox_preds * bbox_masks,
               bbox_labels * bbox_masks)
loss = cls + bbox
```

真实 SSD 还需要面对严重的正负样本不平衡，常使用难负样本挖掘或 focal loss。检测精度也不能只看锚框分类准确率：背景数量巨大时，即使模型几乎都预测背景，准确率仍可能很高。实际评估更关注按类别的 precision-recall 与 mAP。

## 8. R-CNN 系列：不断提高共享计算与端到端程度

### 8.1 四代模型解决了什么问题

| 模型 | 提议区域来源 | 特征提取 | 最终预测 | 主要瓶颈 |
|---|---|---|---|---|
| R-CNN | 选择性搜索 | 每个 RoI 分别经过同一个 CNN | 每类 SVM + 线性框回归 | 上千次重复卷积、分阶段训练 |
| Fast R-CNN | 选择性搜索 | 整图卷积一次，RoI Pooling | softmax + 框回归联合训练 | 选择性搜索仍慢 |
| Faster R-CNN | RPN | 与 RPN 共享特征图 | RoI Head 分类与回归 | 两阶段结构仍较复杂 |
| Mask R-CNN | RPN | 共享特征图 + RoI Align | 再增加实例 mask 分支 | 像素标注和计算成本更高 |

原始 R-CNN 微调 CNN 时的 softmax 头与后续 SVM 看似重复，角色却不同：临时 softmax 通过反向传播让 CNN 特征适应检测区域；CNN 固定后，每类 SVM 才是最终分类器。Fast R-CNN 删除了这种历史性的重复流程。

### 8.2 RoI Pooling 为什么能接全连接层

不同提议区域映射到共享特征图后可能分别是 `[C,8,12]`、`[C,15,20]`。全连接层要求固定输入长度，RoI Pooling 因此把每个区域划为固定网格，例如 `7×7`，每格做最大汇聚：

```text
任意大小 RoI 特征 → [C,7,7]
N 个 RoI         → [N,C,7,7]
```

相同的是 shape，不是特征数值。大 RoI 的一个格子汇总更大范围，小 RoI 的一个格子汇总更小范围。

`spatial_scale` 把原图坐标映射到特征图。例如主干总步幅为 16 时，原图坐标 160 对应特征图坐标 10。

### 8.3 RoI Align 为什么对实例分割重要

若原图坐标 100 在 stride=16 的特征图上对应 6.25，RoI Pooling 会对 RoI 边界和网格边界取整，可能将它变成 6，即映射回原图的 96。分类对几像素误差可能不敏感，像素级 mask 却会明显错位。

RoI Align 保留 6.25 这样的浮点坐标，在每个网格内选采样点，并用周围四个特征值做双线性插值。它避免由连续取整造成的空间偏移；“对齐”不表示完全无信息损失，而是输出网格与原图 RoI 保持更准确的位置对应。

### 8.4 Faster R-CNN 为什么有四项损失

RPN 只负责“哪里可能有目标”，RoI Head 才判断具体类别并精修框，因此有两次分类和两次回归：

$$
L=L_{rpn\_obj}+\lambda_1L_{rpn\_box}
 +L_{roi\_cls}+\lambda_2L_{roi\_box}.
$$

| 损失 | 样本 | 学习目标 |
|---|---|---|
| `loss_objectness` | 正、负锚框 | RPN 判断前景或背景 |
| `loss_rpn_box_reg` | 仅正锚框 | 锚框调整成提议框 |
| `loss_classifier` | 正、负 RoI | 背景或具体类别 |
| `loss_box_reg` | 仅正 RoI | 提议框调整成最终框 |

RoI Head 若使用类别相关回归，会输出 `K×4` 个偏移量。它们通常都会是非零实数；训练时只取真实类别对应的一组，推理时按候选类别取相应一组。其他类别的输出不必强制为零。

现代实现把四项损失相加后一次反向传播。RPN 与 RoI Head 都更新共享 CNN；NMS 和提议框索引选择通常不可微，RPN 主要依靠自己的两项损失学习生成候选区域。

## 9. 语义分割数据：RGB 标签图不是普通照片

### 9.1 “图像分割”在教材中的狭义用法

广义上，image segmentation 是语义、实例和全景分割的总称。D2L 本节把它狭义地用于“根据颜色、纹理等低层相似性把图片分区，但不保证区域具有类别语义”的传统分割：

| 任务 | 每个像素回答什么 | 是否区分同类实例 |
|---|---|---|
| 非语义区域分割 | 属于哪个视觉相似区域 | 不涉及 |
| 语义分割 | 属于哪个预定义类别 | 否 |
| 实例分割 | 属于哪个类别的哪个实例 | 是 |
| 全景分割 | stuff 语义 + things 实例 | 对可数目标区分 |

### 9.2 VOC 标签的读取与转换

VOC 输入照片来自 `JPEGImages`，标签来自 `SegmentationClass`。二者读取后都可能是 `[3,H,W]`，但含义不同：

```text
feature RGB：真实颜色，作为模型输入
label RGB  ：规定颜色，每种颜色编码一个类别
```

标签 PNG 强制使用 `ImageReadMode.RGB`，是为了得到明确的三个颜色通道。随后把每个 RGB 三元组编码成整数索引：

```python
idx = (R * 256 + G) * 256 + B
class_mask = colormap2label[idx]
```

shape 由 `[3,H,W]` 变成 `[H,W]`，dtype 为整数类别。这个转换用于训练或验证真实标签，不处理模型预测；预测 logits 直接沿类别维 `argmax` 得到类别编号。

图片需要 `/255` 和 ImageNet normalization，类别 mask 绝不能做这些连续数值归一化。训练时二者采用相同几何裁剪，但只对图片做颜色变化和标准化。

## 10. 转置卷积：卷积稀疏矩阵的转置

忽略偏置，把普通卷积输入、输出按行展开为向量，可以写成

$$
y=Wx,
$$

其中 $W$ 是由卷积核在不同位置展开而成的巨大稀疏矩阵。普通卷积从输入窗口收集加权和；转置矩阵 $W^T$ 则把每个输入值按同样的核系数散布回更大的输出空间，重叠位置相加：

$$
z=W^Ty.
$$

这也是其名称来源。设损失为 $L$、上游梯度为 $g_y$，则

$$
dL=g_y^Tdy=g_y^TWdx=(W^Tg_y)^Tdx,
$$

所以

$$
\frac{\partial L}{\partial x}=W^T\frac{\partial L}{\partial y}.
$$

转置卷积的前向计算正是普通卷积对输入反向传播时使用的线性运算。它不是逆卷积：一般有 $W^TW\ne I$，因此 `tconv(conv(X))` 可以恢复 shape，却不能自动恢复被下采样丢失的数值。

单个空间维的转置卷积输出为

$$
H_{out}=(H_{in}-1)s-2p+d(k-1)+output\_padding+1.
$$

`kernel_size` 不需要小于输入高宽，因为转置卷积不是在小输入中截取一个大窗口，而是让每个输入值在输出画布上散布一个核形状的贡献。

## 11. FCN：把分类骨干改造成逐像素分类器

D2L 的 FCN 使用去掉全局池化和全连接层的 ResNet-18。对 `[N,3,320,480]` 输入，主干下采样 32 倍：

```text
[N,3,320,480]
      ↓ ResNet 主干
[N,512,10,15]
      ↓ 1×1 卷积
[N,21,10,15]
      ↓ 转置卷积
[N,21,320,480]
```

$1\times1$ 卷积在每个低分辨率位置把 512 维特征映射为 21 个类别 logits，不改变空间大小。转置卷积使用

```python
nn.ConvTranspose2d(
    21, 21,
    kernel_size=64,
    padding=16,
    stride=32,
)
```

虽然输入只有 `10×15`，`64×64` 核仍然合法：每个低分辨率元素向输出散布一个 `64×64` 权重图，相邻中心间隔 32，贡献重叠后相加。输出尺寸为

$$
(10-1)\times32-2\times16+64=320,
$$

$$
(15-1)\times32-2\times16+64=480.
$$

卷积核先初始化为二维双线性插值“帐篷”权重，再参与训练。该简单模型属于 FCN-32s：它恢复了输出分辨率，却不能凭空找回主干下采样丢失的边缘细节。更精细的分割网络会融合浅层高分辨率特征。

分割交叉熵把每个像素当作一个分类样本：

```text
logits: [N,K,H,W]
labels: [N,H,W]
```

损失沿类别维计算，再对有效像素平均。预测时 `logits.argmax(dim=1)` 得到 `[N,H,W]` 类别图。

## 12. 神经风格迁移：模型参数不动，优化输入图像

风格迁移与前面的监督学习方向相反：预训练 CNN 作为固定的特征度量，不更新网络参数；真正被优化的是一张合成图像。

```python
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
```

`nn.Parameter` 让像素张量出现在 `model.parameters()` 中，优化器因而可以直接更新图像。`super().__init__()` 必须先初始化 `nn.Module` 的参数注册结构。

损失由三部分组成：

1. **内容损失**：合成图在较深层的特征接近内容图。
2. **风格损失**：合成图与风格图在多个层的 Gram 矩阵接近。
3. **全变分损失**：相邻像素不要剧烈跳变，抑制噪点。

若特征为 $F\in\mathbb{R}^{C\times HW}$，Gram 矩阵为

$$
G=FF^T,
$$

它描述通道之间的整体相关性，弱化了精确空间位置，因此适合表示纹理与风格。总损失为

$$
L=\alpha L_{content}+\beta L_{style}+\gamma L_{tv}.
$$

权重决定生成图更忠于内容、风格还是平滑性。训练循环每步先从合成图提取特征，再对合成图像素反向传播；CNN 参数应冻结，否则“尺子”和“被测对象”会同时变化。

## 13. Kaggle 实战提供的工程闭环

CIFAR-10 和犬种识别 notebook 的价值不只在最终分数，而在把模型放进一条可提交、可复现的数据流水线：

```text
读取 CSV 标签
  → 划分训练/验证
  → 按目录或自定义 Dataset 组织图片
  → 训练集随机增广，验证集确定性预处理
  → 保存验证最优配置
  → 使用全部训练数据重训
  → 按测试文件顺序生成 submission.csv
```

常见错误包括：

- 重组目录时把同一原图同时放入训练和验证。
- 测试预测顺序与提交模板 ID 顺序不一致。
- 在验证集继续使用随机裁剪，导致指标波动。
- 类别名到整数下标的映射在训练与提交阶段不一致。
- 只保存模型权重，却没有保存 normalization、类别映射和输入尺寸。

犬种识别类别多、单类样本少，更依赖预训练模型和稳定的数据增广；CIFAR-10 图片小，从头训练可行，但仍需要清晰的验证协议。二者共同说明：模型只是实验的一部分，数据索引和预处理契约同样属于模型。

## 14. 怎样选择检测或分割路线

| 需求 | 合适起点 | 原因 |
|---|---|---|
| 只判断整张图类别 | 分类 CNN | 输出最简单，标注成本低 |
| 需要所有目标框且重视速度 | SSD 一类单阶段检测器 | 一次密集预测，无独立 RoI 阶段 |
| 需要更灵活的两阶段精修 | Faster R-CNN | RPN 提议 + RoI Head 分类回归 |
| 每个像素只要语义类别 | FCN 或现代语义分割网络 | 输出 `[N,K,H,W]` |
| 同类目标也要分别分割 | Mask R-CNN | 检测实例与 mask 分支结合 |
| 需要合成内容与纹理 | 神经风格迁移 | 固定表示网络，直接优化像素 |

任务定义应先于模型选择。若标注只有图片类别，就不能直接监督像素 mask；若只需要道路整体区域，实例分割的额外复杂度未必有价值；若部署延迟严格，两阶段检测器的精度收益也要与计算预算比较。

## 本篇知识链总结

空间视觉任务的主线可以压缩为：

```text
图片与标签先建立严格的 shape、坐标和预处理契约
        ↓
锚框把不定数量目标变成固定数量预测槽位
        ↓
IoU 分配训练责任，偏移量学习相对几何，NMS 删除重复结果
        ↓
多尺度特征图让小目标密集预测、大目标稀疏预测
        ↓
SSD 在所有尺度一次输出；R-CNN 系列逐步提高共享计算和端到端程度
        ↓
分割把输出从“每个框”推进到“每个像素”
        ↓
转置卷积恢复空间尺寸，RoI Align保持坐标对齐
        ↓
风格迁移进一步展示：预训练 CNN 也可以作为固定损失网络来优化输入
```

真正贯穿整章的不是某个模型名，而是四个检查问题：当前张量的每一维表示什么；坐标属于原图还是特征图；哪些预测槽位拥有监督；损失的梯度最终更新哪个模块。

## 常见误区

- 把 `/255` 的颜色归一化与“边界框坐标除以图片尺寸”当成同一件事。
- 认为 `anchors[0]` 是第一个锚框，而不是删除最前面的批量维。
- 认为超出图片边界的锚框一定非法，训练前必须全部裁剪。
- 忘记 SSD 分类通道中每个锚框都有 `K+1` 个类别槽位。
- 在 NCHW 上直接 flatten，再错误地把不同空间位置组合成同一锚框类别。
- 认为 R-CNN 为每个提议区域创建一个不同 CNN；实际是同一个 CNN 被重复调用。
- 认为 RoI Pooling 后不同区域的特征相同；相同的只有 shape。
- 认为 `K×4` 回归输出中未选类别的偏移量必须为 0。
- 把转置卷积称为卷积的逆运算，或认为卷积核必须小于输入特征图。
- 对语义分割标签执行图片 normalization，破坏整数类别编号。
- 认为上采样到原图大小就自动恢复了所有边缘细节。
- 风格迁移时同时更新预训练 CNN 和合成图片，使特征度量本身漂移。

## 系列导航

- 上一篇：[D2L 微调：从迁移学习到热狗识别]({{ '/deep-learning/fine-tuning/' | relative_url }})
- [系列总览]({{ '/deep-learning/' | relative_url }})
- 下一篇：[D2L 注意力机制：从 QKV、多头自注意力到 Transformer]({{ '/deep-learning/attention-transformer/' | relative_url }})

## 对应章节与参考资料

- D2L：[计算机视觉](https://zh.d2l.ai/chapter_computer-vision/index.html)
- D2L：[图像增广](https://zh.d2l.ai/chapter_computer-vision/image-augmentation.html)、[边界框](https://zh.d2l.ai/chapter_computer-vision/bounding-box.html)、[锚框](https://zh.d2l.ai/chapter_computer-vision/anchor.html)
- D2L：[多尺度目标检测](https://zh.d2l.ai/chapter_computer-vision/multiscale-object-detection.html)、[SSD](https://zh.d2l.ai/chapter_computer-vision/ssd.html)、[R-CNN 系列](https://zh.d2l.ai/chapter_computer-vision/rcnn.html)
- D2L：[语义分割数据集](https://zh.d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html)、[转置卷积](https://zh.d2l.ai/chapter_computer-vision/transposed-conv.html)、[FCN](https://zh.d2l.ai/chapter_computer-vision/fcn.html)
- D2L：[神经风格迁移](https://zh.d2l.ai/chapter_computer-vision/neural-style.html)
- 原始论文：[R-CNN](https://arxiv.org/abs/1311.2524)、[Fast R-CNN](https://arxiv.org/abs/1504.08083)、[Faster R-CNN](https://arxiv.org/abs/1506.01497)、[Mask R-CNN](https://arxiv.org/abs/1703.06870)、[FCN](https://arxiv.org/abs/1411.4038)
