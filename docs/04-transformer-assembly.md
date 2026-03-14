# 从零实现 Transformer（四）：模型组装

> 我们已经造好了零件，准备好了原材料，准备合体！（我来组成头部，我来组成。。。烂梗王正是在下）

## 系列目录

1. [PyTorch 基础与神经网络模块](01-pytorch-basics.md)
2. [数据处理与 Transformer 输入层](02-data-and-input-layer.md)
3. [多头注意力机制与核心组件](03-multi-head-attention.md)
4. **Transformer 模型组装**（本篇）
5. [训练、推理与可视化](05-training-and-inference.md)

---

## 1. Mask 构造函数

在组装模型之前，我们先实现两个关键的 mask 函数。

### 1.1 Padding Mask

屏蔽 `<pad>` 位置，让模型不关注无意义的填充：

```python
import torch
import torch.nn as nn
import numpy as np


def make_pad_mask(q, k, pad_idx=0):
    """
    构造 Padding Mask
    k.ne(pad_idx): 不等于 pad 的位置为 True(1)，是 pad 为 False(0)
    输出 shape: (Batch, 1, 1, K_Len)，会在 head 和 Q_len 维度广播
    """
    mask = k.ne(pad_idx).unsqueeze(1).unsqueeze(2)
    return mask
```

### 1.2 Subsequent Mask（因果掩码）

下三角矩阵，防止 Decoder 看到未来信息：

```python
def make_subsequent_mask(seq_len):
    """
    构造因果掩码（下三角矩阵）
    输出 shape: (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8))
    return mask.unsqueeze(0).unsqueeze(0)
```

### 测试 Mask

```python
q_sim = torch.tensor([[1, 2, 3]])
k_sim = torch.tensor([[1, 2, 0, 0]])  # 0 是 pad

pad_mask = make_pad_mask(q_sim, k_sim, pad_idx=0)
print("Pad Mask:", pad_mask.squeeze())
# tensor([True, True, False, False])

subsequent_mask = make_subsequent_mask(5)
print("Subsequent Mask:\n", subsequent_mask.squeeze())
# tensor([[1, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0],
#         [1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1]])
```

---

## 2. 堆叠 Encoder

Encoder 就是堆叠多个 EncoderLayer：

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)
```

---

## 3. 堆叠 Decoder

Decoder 类似，但多了一个**投影层**——将 `d_model` 映射到 `vocab_size`，这样才能找到对应的词表索引：

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            tgt, attn_map = layer(tgt, enc_output, src_mask, tgt_mask)
        tgt = self.norm(tgt)
        return self.projection(tgt), attn_map
```

---

## 4. 究极合体：完整 Transformer

### 4.1 Combined Mask

Decoder 需要同时满足两种 mask：padding 不应该看到，也不能看到未来的 token。所以要重叠两个矩阵，都是 True 的位置才有效。

```
tgt_mask (padding):          subsequent_mask (causal):
[[T, T, F, F]]               [[T, F, F, F],
                               [T, T, F, F],
 ↓ 广播到 (4,4)               [T, T, T, F],
[[T, T, F, F],                 [T, T, T, T]]
 [T, T, F, F],
 [T, T, F, F],
 [T, T, F, F]]

逐元素 & 操作 →
combined_mask:
[[T, F, F, F],
 [T, T, F, F],
 [T, T, F, F],    ← 位置2 的第3个token 本是 T&T=T，但 padding mask 说 F
 [T, T, F, F]]
```

### 4.2 为什么拆分 encode 和 decode 方法？

**训练阶段**：一把梭，Encoder 和 Decoder 一起跑：
```
src → Encoder → enc_output
tgt（完整） → Decoder → 输出
```

**推理阶段**：逐词生成
```
第1步：输入 <BOS>           → 生成 "I"
第2步：输入 <BOS> I         → 生成 "love"
第3步：输入 <BOS> I love    → 生成 "you"
```

每一步都要重新跑 Decoder，但 **Encoder 的输出始终不变**（源句子没变）。拆开后：
- Encoder 跑 **1 次**
- Decoder 跑 **N 次**，每次复用 enc_output

### 4.3 完整代码

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()

        self.src_embedding = TransformerInputLayer(src_vocab_size, d_model, dropout=dropout)
        self.tgt_embedding = TransformerInputLayer(tgt_vocab_size, d_model, dropout=dropout)

        self._encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, n_layers, dropout)
        self._decoder = Decoder(tgt_vocab_size, d_model, n_heads, d_ff, n_layers, dropout)

        self.src_pad_idx = 0
        self.tgt_pad_idx = 0

    def make_masks(self, src, tgt):
        src_mask = make_pad_mask(src, src, self.src_pad_idx)
        tgt_mask = make_pad_mask(tgt, tgt, self.tgt_pad_idx)
        subsequent_mask = make_subsequent_mask(tgt.size(1)).to(src.device)
        combined_mask = tgt_mask & subsequent_mask
        return src_mask, combined_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.make_masks(src, tgt)
        enc_output = self._encoder(self.src_embedding(src), src_mask)
        dec_output, _ = self._decoder(self.tgt_embedding(tgt), enc_output, src_mask, tgt_mask)
        return dec_output

    # 专门暴露给推理阶段用的方法
    def encode(self, src, src_mask):
        return self._encoder(self.src_embedding(src), src_mask)

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        dec_output, _ = self._decoder(self.tgt_embedding(tgt), enc_output, src_mask, tgt_mask)
        return dec_output
```

---

## 5. 全流程测试

用随机数据跑一遍整个模型：

```python
src_vocab_size = 100
tgt_vocab_size = 100
d_model = 512
n_layers = 2
n_head = 8

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff=2048, n_layers=n_layers)
model.to(device)

# 伪造数据：Batch=2, Src_Len=5, Tgt_Len=6
# 0 是 pad，1 是 sos，2 是 eos
src = torch.tensor([[1, 5, 6, 2, 0], [1, 9, 2, 0, 0]]).to(device)
tgt = torch.tensor([[1, 7, 3, 4, 8, 2], [1, 6, 8, 2, 0, 0]]).to(device)

print("Source Shape:", src.shape)  # (2, 5)
print("Target Shape:", tgt.shape)  # (2, 6)

output = model(src, tgt)
print("Output Shape:", output.shape)  # (2, 6, 100) → (batch, tgt_len, vocab_size)
```

**输出解读**：`(2, 6, 100)` 表示 2 个样本，每个样本 6 个位置，每个位置输出对 100 个词的概率分布。

---

## 小结

至此，你手中已经有了一个**完整的 Transformer 模型**！

- **输入**：源语言索引序列 + 目标语言索引序列
- **输出**：每个位置下一个词的概率分布

我们实现了：
- **Mask 函数**：Padding Mask + Subsequent Mask + Combined Mask
- **Encoder**：堆叠 N 个 EncoderLayer
- **Decoder**：堆叠 N 个 DecoderLayer + 投影层
- **Transformer 主类**：整合 Embedding、Encoder、Decoder，支持训练和推理

下一篇将赋予模型智慧——用数据训练它，并实现推理和注意力可视化。

---

上一篇：[<< 多头注意力机制与核心组件](03-multi-head-attention.md)

下一篇：[训练、推理与可视化 >>](05-training-and-inference.md)
