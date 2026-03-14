# 从零实现 Transformer（三）：多头注意力机制与核心组件

> 上一篇我们把人类的句子变成了 `(Batch, Seq_len, Dim)` 的张量，现在我们要创建 Transformer 的大脑——多头注意力机制，并搭建 Encoder 和 Decoder 的核心组件。

## 系列目录

1. [PyTorch 基础与神经网络模块](01-pytorch-basics.md)
2. [数据处理与 Transformer 输入层](02-data-and-input-layer.md)
3. **多头注意力机制与核心组件**（本篇）
4. [Transformer 模型组装](04-transformer-assembly.md)
5. [训练、推理与可视化](05-training-and-inference.md)

---

## 1. 什么是注意力机制？

### 1.1 核心公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 1.2 QKV 直觉理解

用查百科大全来类比：

| 角色 | 类比 | 含义 |
|------|------|------|
| **Q** (Query) | 你心中的问题——"让我查查香蕉是什么？" | "我要找什么" |
| **K** (Key) | 百科大全中词条的关键字——"香蕉" | "这个元素怎么表述" |
| **V** (Value) | 词条下的具体内容——黄色、热带水果、弯的 | "这个元素的实际信息" |

### 1.3 计算过程

1. **Q @ K^T**：用"问题"和所有"词条"做点积，算出相关性分数
2. **/ √d_k**：缩放。维度 d_k 越大，点积绝对值越大，会导致 softmax 接近 one-hot（梯度消失）

| 输入 | softmax 输出 | 特征 |
|------|-------------|------|
| `[1, 2, 3]` | `[0.09, 0.24, 0.67]` | 分布均匀，梯度正常 |
| `[10, 20, 30]` | `[0.00, 0.00, 1.00]` | 接近 one-hot，梯度几乎为 0 |

3. **softmax**：把分数归一化为概率分布（加起来等于 1）
4. **@ V**：用概率对所有"内容"做加权求和

### 1.4 多头的直觉

把 QKV 拆分成多个"头"（比如 8 个），相当于让模型有 8 个不同的大脑：
- 有的学习语法
- 有的学习语义
- 有的学习指代关系

最后把所有头拼接到一起，综合全部专家建议。

---

## 2. 多头注意力机制实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_k = d_model // num_heads  # 每个头的维度，如 512 / 8 = 64
        self.n_heads = num_heads
        self.d_model = d_model

        # 定义 QKV 的权重矩阵
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 线性变换 + 拆分头
        # (Batch, Len, D_model) → (Batch, Len, n_head, d_k) → (Batch, n_head, Len, d_k)
        Q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # 计算点积注意力
        # Q: (Batch, n_head, Len, d_k), K^T: (Batch, n_head, d_k, Len)
        # scores: (Batch, n_head, Len, Len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用 mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        # 加权求和
        context = torch.matmul(attn, V)

        # 拼接多个头
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(context)
        return output, attn
```

### 测试

```python
d_model = 512
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)

x = torch.randn(2, 10, 512)  # (batch, seq_len, dim)
out, attn_map = mha(x, x, x, mask=None)

print(f"Input:  {x.shape}")           # (2, 10, 512)
print(f"Output: {out.shape}")          # (2, 10, 512)
print(f"Attention Map: {attn_map.shape}")  # (2, 8, 10, 10)
```

**维度解读**：
- `(2, 8, 10, 10)` — 第 2 个样本、第 8 个头中，Q 的第 i 个位置对 K 的第 j 个位置的注意力分数

---

## 3. 前馈神经网络（FFN）

每个 Encoder/Decoder 层中，除了 Attention，还有一个前馈网络：

```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

FFN 的作用是增加模型的非线性能力，`d_ff` 通常是 `d_model` 的 4 倍（如 512 → 2048）。

---

## 4. 残差连接与层归一化

在 Transformer 中，每个子层的输出都要经过 **Add & Norm** 操作。

### 4.1 残差连接

名字很高大上，实现很简单：**把输入和子层输出直接相加**。

```python
residual = x + sublayer_output
```

为什么需要？随着网络层数加深，梯度可能消失或爆炸。残差提供了一条"安全隧道"，让梯度可以直接回流到浅层。

### 4.2 层归一化（LayerNorm）

本质就是：`输出 = (x - 均值) / 标准差`，把数据拉到均值 0、方差 1 的范围。

**LayerNorm vs BatchNorm 的区别在于：沿哪个方向算均值和标准差。**

```
数据矩阵 (batch=3, features=4):

         特征1  特征2  特征3  特征4
句子1  [  -    -    -    -  ]  ← LayerNorm: 沿这个方向算
句子2  [  -    -    -    -  ]  ← LayerNorm: 沿这个方向算
句子3  [  -    -    -    -  ]  ← LayerNorm: 沿这个方向算
         |      |      |     |
         BatchNorm: 沿这个方向算
```

```python
x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10., 11., 12.]],
                   [[0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2]]])

layer_norm = nn.LayerNorm(4)
output = layer_norm(x)

print(f"归一化前均值：{x[0,0,:].mean():.4f}")      # 2.5000
print(f"归一化后均值：{output[0,0,:].mean():.4f}")   # 0.0000
```

### 4.3 Norm 的位置

| 方案 | 公式 | 说明 |
|------|------|------|
| Post-Norm（原始论文） | `x = Norm(x + SubLayer(x))` | 先计算，再归一化 |
| Pre-Norm（后来主流） | `x = x + SubLayer(Norm(x))` | 先归一化，再计算 |

Pre-Norm 训练更稳定，是现代 Transformer 的主流选择。本教程先用 Post-Norm 方便对照论文。

---

## 5. 组装 Encoder Layer

有了 Attention、FFN、LayerNorm、残差连接，我们就能搭建 Encoder 的一层积木了：

**Encoder = Self-Attention + Add & Norm + FFN + Add & Norm**

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x
```

---

## 6. 组装 Decoder Layer

Decoder 稍微复杂一点，有**三个子层**：

1. **Masked Self-Attention**：掩码防止看到未来信息
2. **Cross-Attention**：Q 来自 Decoder，K 和 V 来自 Encoder
3. **FFN**：和之前一样

### 6.1 两种 Mask 的作用

**自注意力用 tgt_mask**（下三角矩阵，防止看到未来）：

```
Q 看 K →   <sos>   I    eat   apple
  <sos>  [  1      0     0      0  ]   ← 只能看自己
  I      [  1      1     0      0  ]   ← 看不到 eat 和 apple
  eat    [  1      1     1      0  ]   ← 看不到 apple
  apple  [  1      1     1      1  ]   ← 全部能看
```

应用 mask 后，被遮蔽的位置填 `-1e9`，softmax 后变为 0。

**交叉注意力用 src_mask**（只遮蔽 padding）：

```
Q 看 K →     我     吃    苹果   <pad>
  <sos>  [   1      1      1      0   ]
  I      [   1      1      1      0   ]
  eat    [   1      1      1      0   ]
  apple  [   1      1      1      0   ]
```

### 6.2 Decoder Layer 代码

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 自注意力
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 交叉注意力
        attn_output, attn_map = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x, attn_map
```

---

## 7. 测试核心组件

```python
d_model = 512
num_heads = 8
d_ff = 2048
batch_size = 2
src_len = 10
tgt_len = 5

# 测试 Encoder Layer
enc_layer = EncoderLayer(d_model, num_heads, d_ff)
src_input = torch.randn(batch_size, src_len, d_model)
enc_output = enc_layer(src_input, mask=None)
print(f"Encoder Input:  {src_input.shape}")   # (2, 10, 512)
print(f"Encoder Output: {enc_output.shape}")   # (2, 10, 512)

# 测试 Decoder Layer
dec_layer = DecoderLayer(d_model, num_heads, d_ff)
tgt_input = torch.randn(batch_size, tgt_len, d_model)
dec_output, cross_attn = dec_layer(tgt_input, enc_output, src_mask=None, tgt_mask=None)
print(f"Decoder Input:  {tgt_input.shape}")    # (2, 5, 512)
print(f"Decoder Output: {dec_output.shape}")    # (2, 5, 512)
print(f"Cross Attention: {cross_attn.shape}")   # (2, 8, 5, 10)
```

---

## 小结

恭喜你完成了 Transformer 中最难的数学部分！我们现在有了：
- **多头注意力机制**：Transformer 的核心计算单元
- **前馈网络 FFN**：增加非线性能力
- **残差连接 + LayerNorm**：稳定训练
- **Encoder Layer**：Self-Attention + FFN
- **Decoder Layer**：Masked Self-Attention + Cross-Attention + FFN

有了计算单元，只要像叠积木一样把多个 Encoder 和 Decoder 串联起来，就能得到完整的"变形金刚"了。

---

上一篇：[<< 数据处理与 Transformer 输入层](02-data-and-input-layer.md)

下一篇：[Transformer 模型组装 >>](04-transformer-assembly.md)
