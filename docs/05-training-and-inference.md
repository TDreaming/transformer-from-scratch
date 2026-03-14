# 从零实现 Transformer（五）：训练、推理与可视化

> 来到最终章啦，我们怎么把模型赋予智慧呢？现在我们有了一个种子，需要用人类的知识将它浇灌为参天大树。本篇会介绍训练、推理和注意力可视化的完整流程。

## 系列目录

1. [PyTorch 基础与神经网络模块](01-pytorch-basics.md)
2. [数据处理与 Transformer 输入层](02-data-and-input-layer.md)
3. [多头注意力机制与核心组件](03-multi-head-attention.md)
4. [Transformer 模型组装](04-transformer-assembly.md)
5. **训练、推理与可视化**（本篇）

---

## 1. 损失函数与优化器

想象你在一个漆黑的山谷里，目标是走到最低点（下山）：
- **损失函数**：脚下的"地形图"，告诉你离谷底还有多远
- **优化器**：你的"下山策略"，决定每步往哪走、步子迈多大

### 1.1 不同模型的偏好

| 环节 | CNN 场景 | Transformer 场景 |
|------|---------|-----------------|
| 损失函数 | Cross-Entropy / Smooth L1 | Cross-Entropy（预测下一个 token） |
| 优化器 | SGD + Momentum / Adam | 几乎统一用 AdamW |
| 学习率 | StepLR / CosineAnnealing | Warmup + 衰减（先升后降） |

### 1.2 训练流程总览

```
输入数据 → 模型 → 预测结果
                       ↓
                  损失函数 ← 真实标签
                       ↓
                   损失值(一个数字)
                       ↓
                  反向传播(算梯度)
                       ↓
                  优化器(更新参数)
                       ↓
                  回到第一步，循环
```

### 1.3 代码

```python
import torch
from torch import nn
import torch.optim as optim

# 损失函数：ignore_index=0 忽略 padding 的 loss
criterion = nn.CrossEntropyLoss(ignore_index=0)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

model.train()
```

---

## 2. Teacher Forcing 训练

训练时，Decoder 的输入不是模型自己的预测，而是**真实的目标序列**（像老师手把手教）。

### 为什么不让模型用自己的预测？

```
不用 Teacher Forcing：
  t=1: 输入 <start>    → 预测 "我" ✅
  t=2: 输入 "我"        → 预测 "恨" ❌ ← 错了！
  t=3: 输入 "恨"        → 预测 "狗" ❌ ← 错上加错！

用 Teacher Forcing：
  t=1: 输入 <start>    → 预测 "我" ✅
  t=2: 输入 "我"(正确)  → 预测 "恨" ❌（但没关系！）
  t=3: 输入 "爱"(正确)  → 预测 "猫" ✅（不受上一步影响）
```

即使某步预测错了，下一步输入仍是正确答案，不会被带偏。

### 训练循环

```python
epochs = 100

for epoch in range(epochs):
    epoch_loss = 0.0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        # 核心：构造 Decoder 的输入和标签
        tgt_input = tgt[:, :-1]   # 去掉最后一个元素
        tgt_label = tgt[:, 1:]    # 去掉第一个元素

        # 1. 清空梯度
        optimizer.zero_grad()

        # 2. 前向传播
        output = model(src, tgt_input)

        # 3. 计算损失（展平为 2D）
        loss = criterion(
            output.contiguous().view(-1, len(tar_vocab)),
            tgt_label.contiguous().view(-1)
        )

        # 4. 反向传播
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(loader):.4f}')

print("训练完成！")
```

### 为什么 epochs 要大于 1？

类比学习和备考：不会只看一遍课本就上考场。

- **第一遍**：粗略理解概念（loss 很高）
- **反复刷题**：大量练习，纠错（多轮 epoch）
- **模拟考试**：用没见过的题检验（验证集/测试集）

不同 epoch 学到的东西不一样：
- 早期：最显著的粗粒度模式
- 中期：更细微的特征和边界情况
- 后期：精调决策边界

但也不能看太多遍——**过拟合风险**！需要验证集监控、Early Stopping、正则化等手段。

---

## 3. CrossEntropyLoss 展平计算详解

### 为什么要展平？

`CrossEntropyLoss` 要求输入是 `(N, C)` 和 `(N,)`，但模型输出是 3D 的 `(Batch, Seq_Len, Vocab_Size)`。

假设 Batch=2, Seq_Len=3, Vocab_Size=5：

```python
# 模型输出 (2, 3, 5) → 展平为 (6, 5)
output.view(-1, 5)

# 标签 (2, 3) → 展平为 (6,)
tgt_label.view(-1)
```

本质：把「2 个句子 × 3 个位置」拉成「6 个独立的分类问题」，每个问题从 5 个词里选 1 个正确的。

### 为什么翻译是分类任务？

在 Decoder 的每个时间步，任务是：从整个词汇表（V 个候选词）中选出一个概率最高的词。这个"从 V 个候选中选一个"的操作，本质就是一个 **V 分类问题**。整个翻译过程 = 连续执行 T 次分类。

---

## 4. 训练循环每一步的详解

| 步骤 | 权重 W | 梯度 W.grad | Loss | 说明 |
|------|--------|-------------|------|------|
| `zero_grad()` | 不变 | 归零 | 无 | 防止梯度累加 |
| `forward` | 不变 | 0 | 无 | 用当前权重做矩阵运算 |
| `criterion(...)` | 不变 | 0 | 计算得出 | 衡量预测与真实的差距 |
| `backward()` | **不变** | 0 → 具体值 | 不变 | 链式法则计算梯度 |
| `step()` | **更新！** | 不变 | 过时的旧值 | 唯一修改权重的步骤 |

**核心认知**：前向传播、Loss 计算、反向传播都不修改权重，只有 `optimizer.step()` 才真正更新参数。

### 跨 Epoch 的 Loss 变化趋势

```
Epoch  1 | Loss: 8.9234    ← 接近 ln(vocab_size)，在"瞎猜"
Epoch 10 | Loss: 4.2156    ← 快速下降
Epoch 20 | Loss: 1.8923    ← 学到了主要模式
Epoch 30 | Loss: 0.6741    ← 精细调整
Epoch 50 | Loss: 0.0847    ← 在训练集上拟合良好
```

---

## 5. 推理（Greedy Decode）

训练好的模型有答案，但推理时没有标准答案。流程是：

1. 先把源句子编码成 Memory
2. 给 Decoder 输入 `<sos>`
3. 模型预测第一个词
4. 把预测的词追加到输入，预测下一个词
5. 直到生成 `<eos>` 或达到最大长度

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    # Encoder 只计算一次
    memory = model.encode(src, src_mask)

    # 初始化 Decoder 输入：只有 <sos>
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for i in range(max_len - 1):
        tgt_mask = make_subsequent_mask(ys.size(1)).to(device)
        out = model.decode(ys, memory, src_mask, tgt_mask)

        # 只关心最后一个时间步的输出
        prob = out[:, -1, :]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

        if next_word == tar_vocab['<eos>']:
            break

    return ys
```

### 封装翻译函数

```python
def translate(sentence):
    model.eval()

    src_indices = [src_vocab[w] for w in sentence.split()]
    src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)
    src_mask = make_pad_mask(src_tensor, src_tensor, 0)

    out_tensor = greedy_decode(model, src_tensor, src_mask, max_len=100, start_symbol=tar_vocab['<sos>'])

    out_indices = out_tensor.squeeze().tolist()
    translation = []
    for idx in out_indices:
        if idx == tar_vocab['<sos>']: continue
        if idx == tar_vocab['<eos>']: break
        translation.append(idx2tar[idx])

    return "".join(translation)


# 见证奇迹的时刻
print("\n=== 测试模型翻译能力 ===")

test_sentences = [
    "I love deep learning",
    "Transformer is all you need",
    "hello world"
]

for s in test_sentences:
    print(f"原文: {s}  -->  翻译: {translate(s)}")
```

---

## 6. 可视化注意力

还记得之前 `forward` 返回的 `attn_map` 吗？现在能展示模型在翻译某个中文词时，"盯着"哪个英文词看。

```python
import matplotlib.pyplot as plt
import seaborn as sns


def plot_attention(sentence, translation, attention_weights):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attention_weights, cmap='viridis',
                xticklabels=sentence.split(),
                yticklabels=list(translation))
    plt.xlabel('Source Words')
    plt.ylabel('Target Words')
    plt.title('Attention Weights')
    plt.show()


# 获取 Attention Map
model.eval()
src_text = "I love deep learning"
src_indices = torch.tensor([src_vocab[w] for w in src_text.split()]).unsqueeze(0).to(device)
src_mask = make_pad_mask(src_indices, src_indices, 0)

enc_out = model.encode(src_indices, src_mask)

tgt_indices = torch.tensor([[
    tar_vocab['<sos>'],
    tar_vocab['我'], tar_vocab['爱'],
    tar_vocab['深'], tar_vocab['度'],
    tar_vocab['学'], tar_vocab['习']
]]).to(device)
tgt_mask = make_subsequent_mask(tgt_indices.size(1)).to(device)

output, attn_weights = model._decoder(model.tgt_embedding(tgt_indices), enc_out, src_mask, tgt_mask)

# 取第一个样本，平均所有头的注意力
attn_avg = attn_weights[0].mean(dim=0).detach().cpu().numpy()

plot_attention(src_text, "<sos>我爱深度学习", attn_avg[1:])
```

---

## 7. 结语

至此，你已经从零开始构建并训练了一个 Transformer 模型，并完成了一次翻译任务。虽然这个模型很小，数据集也很小，但**原理和 ChatGPT、Claude 模型一样**。你把模型放大，数据放大，训练时间变长，就会得到一个非常强大的模型。

用一首现代诗结尾：

> Embedding 把世界万物编织成向量，
> 让语言有了形状，让含义有了方向。
>
> Attention 在向量的海洋中捕捞关联，
> 让每个词回望全文，找到自己的锚点。
>
> Training 让模型一次次预测下一个 Token，
> 在错误与修正之间，逼近语言的本能。
>
> Loss 是黑暗中唯一的灯塔，
> 梯度沿着它的方向，一步步下山。
>
> Scaling 是一个朴素的信仰——
> 参数再多一些，数据再大一些，智能就涌现了。
>
> Inference 是训练之后的独白，
> 模型终于开口，一个 Token 接一个 Token，
> 像孩子学会造句，
> 像河流学会入海。
>
> 我们用矩阵乘法搭建了一座巴别塔，
> 它不通天，
> 但它开始理解。

### 课后作业

去 Kaggle 或开源数据集下载一些中英、中日或你喜欢的数据集，把模型扩大，看下能不能训练一个自己的翻译模型。

---

上一篇：[<< Transformer 模型组装](04-transformer-assembly.md)

---

## 全系列回顾

1. [PyTorch 基础与神经网络模块](01-pytorch-basics.md) — 张量操作、nn.Module、常用层
2. [数据处理与 Transformer 输入层](02-data-and-input-layer.md) — 词表、Padding、Embedding、位置编码
3. [多头注意力机制与核心组件](03-multi-head-attention.md) — 注意力、FFN、残差连接、Encoder/Decoder Layer
4. [Transformer 模型组装](04-transformer-assembly.md) — Mask、Encoder、Decoder、完整 Transformer
5. [训练、推理与可视化](05-training-and-inference.md) — 损失函数、训练循环、推理、注意力可视化
