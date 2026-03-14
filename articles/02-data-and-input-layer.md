# 从零实现 Transformer（二）：数据处理与 Transformer 输入层

> 上一篇我们掌握了 PyTorch 的基础操作，本篇将构建一个迷你翻译数据集，并实现 Transformer 的输入层——包括词嵌入（Token Embedding）和位置编码（Positional Encoding）。

## 系列目录

1. [PyTorch 基础与神经网络模块](01-pytorch-basics.md)
2. **数据处理与 Transformer 输入层**（本篇）
3. [多头注意力机制与核心组件](03-multi-head-attention.md)
4. [Transformer 模型组装](04-transformer-assembly.md)
5. [训练、推理与可视化](05-training-and-inference.md)

---

## 1. 构建迷你翻译数据集和词表

为了演示就先不用完整的数据集了，我们创建一个微型的"Demo 数据集"，看清 token 是如何被处理的。你就知道为什么大模型很多都是用 Token 这种奇奇怪怪的单位来计费了，因为 **Token 才是大模型的计算原子**。

```python
import torch
import torch.nn as nn
import torch.utils.data as data
import math
import numpy as np

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"设备: {device}")
```

### 1.1 原始数据与词表构建

```python
# 构建原始数据集，source 是英文，target 是中文
raw_data = [
    ("I love deep learning", "我爱深度学习"),
    ("Transformer is all you need", "你需要的就是变形金刚"),
    ("Attention mechanism is great", "注意力机制很棒"),
    ("hello world", "你好世界"),
    ("PyTorch is easy to learn", "PyTorch很容易学")
]

# 构建词表
# pad: 填充, sos: 开始, eos: 结束
src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
tar_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}

# 扫描数据集，构建词表
for src, tar in raw_data:
    for word in src.split():      # 英文用空格分词
        if word not in src_vocab:
            src_vocab[word] = len(src_vocab)
    for word in tar:              # 中文按单个字符遍历
        if word not in tar_vocab:
            tar_vocab[word] = len(tar_vocab)

# 创建反向词表（索引 → 单词）
idx2src = {v: k for k, v in src_vocab.items()}
idx2tar = {v: k for k, v in tar_vocab.items()}

print(f"源语言词表大小: {len(src_vocab)}")
print(f"目标语言词表大小: {len(tar_vocab)}")
```

> **关于词表大小**：英文日常交流使用 3000-4000 个单词就可以覆盖 90% 的口语场景。GPT-3 使用约 50K 的词表（主要英文），Qwen3 已经扩展到 150K（中英文），GPT-4o 更是 200K。词表越大，训练数据量也越大。

---

## 2. Dataset 与 DataLoader（Padding）

很多初学者都有疑问：怎么把长短不一的句子塞到同一个 Batch 中呢？

答案很朴素：**反正只是为了长度一样，随便塞点无意义的 0 进去就好了。**

### 2.1 自定义 Dataset

```python
class ToyTranslationDataset(data.Dataset):
    def __init__(self, data, src_vocab, tar_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tar_vocab = tar_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tar_text = self.data[idx]
        # 查找每个 token 在词表中的索引
        src_indices = [self.src_vocab[word] for word in src_text.split()]
        tar_indices = [self.tar_vocab[word] for word in tar_text]
        return torch.tensor(src_indices), torch.tensor(tar_indices)
```

### 2.2 Collate 函数与 Padding

神经网络要求输入是规整的矩阵，不能是长度参差不齐的列表。

```python
def collate_fn(batch):
    """自定义 batch 处理：补前后标识 + 补 0"""
    src_batch, tar_batch = [], []

    for src_sample, tar_sample in batch:
        src_batch.append(src_sample)
        # 目标序列拼上 <sos> 和 <eos>
        tar_processed = torch.cat([
            torch.tensor([tar_vocab['<sos>']]),
            tar_sample,
            torch.tensor([tar_vocab['<eos>']])
        ])
        tar_batch.append(tar_processed)

    # padding_value=0 就是词表中的 <pad>
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    tar_batch = nn.utils.rnn.pad_sequence(tar_batch, padding_value=0, batch_first=True)
    return src_batch, tar_batch


dataset = ToyTranslationDataset(raw_data, src_vocab, tar_vocab)
loader = data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

src_batch, tar_batch = next(iter(loader))

print("处理后 Batch 数据的形状")
print(f"Source Batch Shape: {src_batch.shape}  (Batch, Src_len)")
print(f"Target Batch Shape: {tar_batch.shape}  (Batch, Tar_len)")
print("\nSource:\n", src_batch)
print("Target:\n", tar_batch)
```

**解读**：batch=2，源数据（英文）一个长度为 4，一个长度为 5，短的句子最后补 0。目标数据加了 `<sos>` 和 `<eos>`，最长的句子长度比原来多了 2，其他句子也补 0 到同一长度。

---

## 3. 词嵌入（Token Embedding）

经过 DataLoader，我们已经把人类的语言初步转换成了数字（每个 Token 对应一个词表 Index）。但我们能直接用这些 Index 去训练吗？

**不能。** 类比一下：老师给你布置了一本寒假作业，只给了你目录，让你把整本作业连题目带答案都写出来——信息太少了。

我们需要**把词扩展到更高维度**，让这个词在现实生活中的众多含义都能被数字表示。比如"香蕉"的含义有水果、黄色、热带、弯的等等，用一个数字编号很难表征这些含义。

| 整数索引 | Embedding 向量 |
|---|---|
| 1 维，只有"编号"信息 | 768 维，每个维度都可以编码不同的语义特征 |
| 无法表达词与词之间的相似度 | 向量距离天然反映语义距离（如 `king - man + woman ≈ queen`） |

### 3.1 Embedding 的本质就是查表

```python
# 词表大小 10，向量维度 4
emb_layer = nn.Embedding(num_embeddings=10, embedding_dim=4)

print("Embedding 权重矩阵形状：", emb_layer.weight.shape)  # (10, 4)

input_ids = torch.tensor([3, 0, 7])
outputs = emb_layer(input_ids)

print(f"输出形状: {outputs.shape}")  # (3, 4)

# 验证：Embedding 本质就是查表
print(torch.equal(emb_layer.weight.data[3], outputs.data[0]))  # True
```

### 3.2 缩放因子

论文中有一个 `Embedding * sqrt(d_model)` 的操作。为什么？为了让语义信息给予更大的权重，避免被位置信息"功高盖主"。

打个比方：你喝一杯咖啡，加了一杯糖浆，你还能喝出啥？你要是加一大杯咖啡（放大语义），一点糖（位置编码），味道就正好了。

```python
d_model = 512
emb_large = nn.Embedding(100, d_model)
sample = emb_large(torch.tensor([1]))

print(f"未缩放的标准差：{sample.std().item():.4f}")          # ≈ 1.0
sample_scaled = sample * math.sqrt(d_model)
print(f"缩放后的标准差：{sample_scaled.std().item():.4f}")    # ≈ 22.6
print(f"缩放因子 sqrt({d_model}) = {math.sqrt(d_model):.2f}")
```

---

## 4. 位置编码（Positional Encoding）

你已经把人类语言转化成了电脑能看懂的向量，但还没告诉电脑这些数字的**顺序**。

"但我提供的数据不就是按顺序排列的吗？" 好问题！如果是串行读（像 RNN、LSTM），模型确实可以根据读入顺序推断，但那样**太慢了**。Transformer 是**一次性并行处理所有 token** 的，所以没有位置编码的话，`I love you` 和 `Love I you` 对它来说一模一样。

### 4.1 位置编码公式

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

> 推荐 B 站视频讲解（只有 10 分钟）：https://b23.tv/qpalhST
>
> 一句话理解：基于 sin 和 cos 的周期性质，通过叠加不同尺度的正弦余弦来对位置编码，值范围控制在固定范围方便归一化，且任意 pos+K 的编码都能通过 pos 的编码线性推导获得。

### 4.2 手动推导（简化版）

以 `max_len=4, d_model=8` 为例：

```
position × div_term:
pos=0: [0.0,  0.0,  0.0,   0.0  ]
pos=1: [1.0,  0.1,  0.01,  0.001]
pos=2: [2.0,  0.2,  0.02,  0.002]
pos=3: [3.0,  0.3,  0.03,  0.003]
```

偶数位用 sin，奇数位用 cos，交错拼接后：

```
列:     sin₀   cos₀   sin₁   cos₁   sin₂   cos₂   sin₃   cos₃
pos=0: [ 0.0,   1.0,   0.0,   1.0,   0.0,   1.0,   0.0,   1.0 ]
pos=1: [ 0.84,  0.54,  0.10,  0.99,  0.01,  1.0,   0.001, 1.0 ]
pos=2: [ 0.91, -0.42,  0.20,  0.98,  0.02,  1.0,   0.002, 1.0 ]
pos=3: [ 0.14, -0.99,  0.30,  0.96,  0.03,  1.0,   0.003, 1.0 ]
```

左边的列频率高，不同位置差异明显（秒针）；右边的列频率低，差异较小（时针）。通过不同频率的组合，可以为每个位置生成唯一的"位置指纹"。

**会不会有重复？** 单个维度是周期性的，但所有不同频率组合的最小公倍数趋于无穷大（周期是无理数），所以对于我们的小模型完全够用。新的大模型有更先进的位置编码方案（如 RoPE）。

### 4.3 代码实现

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数下标用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数下标用 cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)  # 不需要学习的参数

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
```

### 4.4 Forward 推导

假设 `x` 的 shape 是 `(32, 50, 512)` (batch=32, seq_len=50, d_model=512)：

```
self.pe 的 shape: (1, 5000, 512)
截取: self.pe[:, :50, :] → (1, 50, 512)
相加: x + self.pe[:, :50, :] → (32, 50, 512)  # batch 维度广播
```

位置编码和句子的具体内容无关——"你爱我"、"我爱你"、"蜜雪冰城甜蜜蜜"，句首的字都用 index=0 的位置编码。

---

## 5. 组合 Embedding + Positional Encoding

将 Embedding 层和位置编码层封装为一个完整的输入层：

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class TransformerInputLayer(nn.Module):
    def __init__(self, vocab_size, emb_size, max_len=5000, dropout=0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, emb_size)
        self.pos_emb = PositionalEncoding(emb_size, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.token_emb(x)
        out = self.pos_emb(out)
        return self.dropout(out)
```

---

## 小结

本篇我们完成了 Transformer 的"原材料"准备：
- **词表构建**：将文本转换为数字索引
- **Dataset & DataLoader**：处理变长序列的 Padding 策略
- **Token Embedding**：将一维索引映射到高维向量空间
- **Positional Encoding**：用正弦余弦函数为每个位置生成唯一编码
- **输入层组合**：Embedding + 位置编码 + Dropout

只要逐行写完了上面的代码，运行没有报错，我们就可以进入第三篇——手写多头注意力机制。

---

上一篇：[<< PyTorch 基础与神经网络模块](01-pytorch-basics.md)

下一篇：[多头注意力机制与核心组件 >>](03-multi-head-attention.md)
