# 从零实现 Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songyaolun/transformer-from-scratch/blob/main/transformer_from_scratch.ipynb)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://songyaolun.github.io/transformer-from-scratch/)

从零开始用 PyTorch 手写一个完整的 Transformer 模型，完成中英翻译任务。

## 在线阅读

**[https://songyaolun.github.io/transformer-from-scratch/](https://songyaolun.github.io/transformer-from-scratch/)**

## 系列文章

1. **PyTorch 基础与神经网络模块** — 张量操作、nn.Module、常用层、训练流程
2. **数据处理与 Transformer 输入层** — 词表构建、Padding、Embedding、位置编码
3. **多头注意力机制与核心组件** — 注意力公式、FFN、残差连接、Encoder/Decoder Layer
4. **Transformer 模型组装** — Mask、Encoder、Decoder、完整 Transformer
5. **训练、推理与可视化** — 损失函数、训练循环、Greedy Decode、注意力可视化

## 一键运行

点击上方 **Open in Colab** 徽章，在 Google Colab 中打开包含全部代码的 Notebook，无需本地环境配置。

## 本地运行

```bash
# 克隆仓库
git clone https://github.com/songyaolun/transformer-from-scratch.git
cd transformer-from-scratch

# 安装依赖
pip install torch numpy matplotlib seaborn

# 打开 Notebook
jupyter notebook transformer_from_scratch.ipynb
```

## 环境要求

- Python 3.9+
- PyTorch 2.0+
- matplotlib, seaborn（可视化部分）
