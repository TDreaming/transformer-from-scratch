# 从零实现 Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songyaolun/transformer-from-scratch/blob/main/transformer_from_scratch.ipynb)

本系列教程将带你从零开始，用 PyTorch 手写一个完整的 Transformer 模型，完成一个中英翻译任务。

## 写在前面

写这系列文章的初衷是夏洛铁烦恼和《万物发明指南》带来的灵感。不管你穿越回过去的哪个时间点，你总要有点真本事。当前 AI 巨变的基座就是 Transformer，你提前把论文发出来，就能让 Transformer 的作者感觉活在你的阴影之下。（哈哈哈，烂梗）

## 系列文章

| 序号 | 文章 | 内容 |
|------|------|------|
| 1 | [PyTorch 基础与神经网络模块](01-pytorch-basics.md) | 张量操作、nn.Module、常用层、训练流程 |
| 2 | [数据处理与 Transformer 输入层](02-data-and-input-layer.md) | 词表构建、Padding、Embedding、位置编码 |
| 3 | [多头注意力机制与核心组件](03-multi-head-attention.md) | 注意力公式、FFN、残差连接、Encoder/Decoder Layer |
| 4 | [Transformer 模型组装](04-transformer-assembly.md) | Mask、Encoder、Decoder、完整 Transformer |
| 5 | [训练、推理与可视化](05-training-and-inference.md) | 损失函数、训练循环、Greedy Decode、注意力可视化 |

## 一键运行

点击下方徽章，在 Google Colab 中打开包含全部代码的 Notebook：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songyaolun/transformer-from-scratch/blob/main/transformer_from_scratch.ipynb)

## 环境要求

- Python 3.9+
- PyTorch 2.0+
- matplotlib, seaborn（可视化部分）
