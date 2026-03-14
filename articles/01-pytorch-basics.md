# 从零实现 Transformer（一）：PyTorch 基础与神经网络模块

> 本系列文章将带你从零开始，用 PyTorch 手写一个完整的 Transformer 模型。本篇是第一篇，覆盖 PyTorch 张量操作、`nn.Module` 机制、常用层、训练流程等基础知识。如果你已经熟悉 PyTorch，可以直接跳到第二篇。

## 系列目录

1. **PyTorch 基础与神经网络模块**（本篇）
2. [数据处理与 Transformer 输入层](02-data-and-input-layer.md)
3. [多头注意力机制与核心组件](03-multi-head-attention.md)
4. [Transformer 模型组装](04-transformer-assembly.md)
5. [训练、推理与可视化](05-training-and-inference.md)

---

## 写在前面

写这系列文章的初衷是夏洛铁烦恼和《万物发明指南》带来的灵感，希望能帮助到大家。不管你穿越回过去的哪个时间点，你总要有点真本事，才能在这个快节奏的世界里保持领先。我在想如果我穿回过去，我只知道 AI 会来，但是不知道 AI 是怎么来的，我想这就是我写这系列文章的初衷。当前 AI 巨变的基座就是 Transformer，你提前把论文发出来，就能让 Transformer 的作者感觉活在你的阴影之下。（哈哈哈，烂梗）

## 1. 环境准备

```python
import torch
import torch.nn as nn
import math

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"设备: {device}")
```

我的个人电脑是 M1 系列的 Mac 电脑，上面的执行结果是：`设备: mps`。如果你用有 NVIDIA GPU 的电脑，结果应该是 `设备: cuda`。

## 2. PyTorch 张量操作

### 2.1 创建张量

```python
import torch

# 从列表创建
x = torch.tensor([1, 2, 3])
print(x)         # tensor([1, 2, 3])
print(x.shape)   # torch.Size([3])

# 从二维列表创建
matrix = torch.tensor([[1, 2], [3, 4]])
print(matrix)        # tensor([[1, 2], [3, 4]])
print(matrix.shape)  # torch.Size([2, 2])
```

常用的创建方法：

```python
x = torch.zeros(2, 3)    # 全 0 张量
x = torch.ones(3, 2)     # 全 1 张量
x = torch.randn(3, 4)    # 标准正态分布张量
```

### 2.2 张量属性

```python
x = torch.randn(2, 4, 5)
print(x.shape)    # torch.Size([2, 4, 5])
print(x.dim())    # 3 (维度)
print(x.numel())  # 40 (元素总数)
print(x.dtype)    # torch.float32 (数据类型)
```

### 2.3 张量索引

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x[0, 1])     # 2 (访问单个元素)
print(x[0])         # tensor([1, 2, 3]) (第一行)
print(x[:, 1:])     # tensor([[2, 3], [5, 6]]) (切片)
```

### 2.4 形状操作

#### view() — 变换形状

```python
x = torch.randn(2, 3, 4)
x_flat = x.view(2, -1)         # (2, 12)，-1 自动推断
x_high = x_flat.view(2, 2, -1, 2)  # (2, 2, 3, 2)
```

**什么时候用？** 拆分多头注意力的时候：

```python
B, T, C = 2, 3, 8
x = torch.randn(B, T, C)
n_heads = 2
head_dim = C // n_heads
x_heads = x.view(B, T, n_heads, head_dim)  # (2, 3, 2, 4)
```

#### transpose() — 交换两个维度

```python
x = torch.randn(2, 3, 4)
x_t = x.transpose(1, 2)  # (2, 4, 3)
```

#### permute() — 重排所有维度

```python
x = torch.randn(2, 3, 4, 5)
x_p = x.permute(0, 3, 1, 2)  # (2, 5, 3, 4)
```

**transpose vs permute：** `transpose` 只交换两个维度，写法简洁；`permute` 可以重排所有维度。真实 Transformer 中的用法：

```python
B, T, C = 2, 3, 512
n_head = 8
head_dim = C // n_head
x = torch.randn(B, T, C)
x = x.view(B, T, n_head, head_dim)
x = x.permute(0, 2, 1, 3)  # (2, 8, 3, 64)
```

#### unsqueeze() / squeeze() — 增/减维度

```python
x = torch.tensor([1, 2, 3])   # shape: (3,)
x1 = x.unsqueeze(0)           # shape: (1, 3)
x2 = x.unsqueeze(1)           # shape: (3, 1)

x = torch.randn(1, 3, 1, 4)  # shape: (1, 3, 1, 4)
x = x.squeeze()                # shape: (3, 4)，删除所有大小为 1 的维度
```

为什么有了 `view` 还要 `squeeze/unsqueeze`？因为 `unsqueeze(1)` 不需要关心其他维度的 size，而 `view` 必须声明所有维度的值。

#### torch.cat() — 拼接张量

```python
x1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
x2 = torch.tensor([[7, 8, 9], [10, 11, 12]])

torch.cat([x1, x2], dim=0)  # (4, 3)
torch.cat([x2, x1], dim=1)  # (2, 6)
```

#### contiguous() — 确保内存连续

```python
x = torch.randn(2, 3, 4)
x = x.transpose(0, 1)  # 转置后内存不连续
# x.view(2, 4, 3)  # 报错！
x = x.contiguous()
x = x.view(2, 4, 3)    # 正常
```

`transpose`、`permute` 等操作改变的是内存布局，而 `view` 要求内存连续，所以需要 `contiguous()`。

---

## 3. 神经网络模块 (nn.Module)

`nn.Module` 是所有神经网络的基类，有点类似于 Java 的 Object 类。

### 3.1 基本结构

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()           # 必须调用！
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = MyModel()
```

`super().__init__()` 是**必须调用**的，它初始化了参数注册、buffer 管理等功能。如果不调用，会直接报错：

```python
class BrokenModel(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(10, 5)  # AttributeError!
```

### 3.2 为什么用 model(x) 而不是 model.forward(x)？

当你运行 `model(x)` 时，Python 实际调用的是 `model.__call__(x)`。在 `nn.Module` 的 `__call__` 方法中，会先运行各种 Hook，最后才调用你写的 `forward(x)`。

```python
# nn.Module 的简化实现（仅供理解）
class Module:
    def __call__(self, *args, **kwargs):
        # 1. 执行前置操作（各种 Hook）
        result = self.forward(*args, **kwargs)
        # 2. 执行后置操作
        return result
```

### 3.3 完整的调用链演示

```python
class MyLayer(nn.Module):
    def __init__(self, in_features, out_features, name="MyLayer"):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.name = name
        print(f"MyLayer 初始化完成，weight shape: {self.weight.shape}")

    def forward(self, x):
        print(f"MyLayer {self.name} 的forward被调用，input shape: {x.shape}")
        return x @ self.weight.t()


class MyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = MyLayer(in_features, 40, "layer1")
        self.layer2 = MyLayer(40, 5, "layer2")

    def forward(self, x):
        print(f"MyModel 的forward被调用，input shape: {x.shape}")
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)


model = MyModel(10, 20)
x = torch.randn(2, 10)
output = model(x)
```

调用链路：

```
model(x) → model.__call__(x) → model.forward(x) → self.layer1(x)
  → layer1.__call__(x) → layer1.forward(x) → relu(x) → self.layer2(x)
  → layer2.__call__(x) → layer2.forward(x)
```

### 3.4 关于 `__setattr__` 的细节

在 `nn.Module` 的 `__init__` 中，你会看到这样的代码：

```python
super().__setattr__("training", True)
super().__setattr__("_parameters", {})
super().__setattr__("_buffers", {})
```

为什么不直接用 `self.training = True`？因为 PyTorch 重写了 `__setattr__` 方法，添加了对 parameters、submodules、buffers 的特殊处理。`__init__` 中只是想单纯赋值，所以调用 `object` 的 `__setattr__` 来避免无效开销。

---

## 4. 常用层

### 4.1 nn.Linear — 全连接层

```python
# y = wx + b
linear = nn.Linear(10, 5)
x = torch.randn(2, 10)
output = linear(x)  # shape: (2, 5)
```

### 4.2 nn.Embedding — 词嵌入层

将离散的词索引映射到连续的向量空间，本质就是**查表操作**：

```python
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)

indices = torch.tensor([1, 5, 3, 10])
output = embedding(indices)  # shape: (4, 64)

batch_indices = torch.tensor([[1, 4, 6], [10, 5, 19]])
output = embedding(batch_indices)  # shape: (2, 3, 64)
```

### 4.3 nn.LayerNorm — 层归一化

对每个样本的特征进行归一化（均值为 0，方差为 1）：

```python
dim = 5
layer_norm = nn.LayerNorm(dim)
x = torch.randn(2, 10, dim)
output = layer_norm(x)
```

### 4.4 nn.Dropout — 随机丢弃

训练时随机将部分参数设为 0，避免过拟合：

```python
drop_out = nn.Dropout(0.5)  # 50% 概率置零
output = drop_out(x)
```

注意：非零值会被放大 `1/(1-p)` 倍，保证特征期望不变。

### 4.5 激活函数

```python
import torch.nn.functional as F

# ReLU：负数置零，正数不变
x = torch.tensor([-1.0, -0.5, 1.0, 2.0])
F.relu(x)  # tensor([0., 0., 1., 2.])

# Softmax：转为概率分布（和为1）
x = torch.tensor([1.0, 2.0, 3.0])
F.softmax(x, dim=-1)  # tensor([0.0900, 0.2447, 0.6652])
```

---

## 5. 训练相关

### 5.1 设备管理

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

x = x.to(device)    # 张量需要重新赋值
model.to(device)     # 模型是原地操作
```

为什么张量需要 `x = x.to(device)` 而模型只需要 `model.to(device)`？

- **tensor** 从内存转移到显存，地址不一样了，不是 inplace 操作
- **model** 内部的参数 tensor 换了，但 model 这个"容器"还是同一个对象引用

### 5.2 损失函数

```python
# 分类任务用 CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
output = torch.tensor([[1.5, 0.2, -0.5], [0.1, 2.0, 0.3]])
target = torch.tensor([0, 2])
loss = criterion(output, target)

# 回归任务用 MSELoss
criterion = nn.MSELoss()
output = torch.tensor([[200.0], [400.0]])
target = torch.tensor([[220.0], [380.0]])
loss = criterion(output, target)  # 400.0
```

### 5.3 优化器与训练循环

```python
import torch.optim as optim

model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()       # 1. 清空梯度
    outputs = model(inputs)     # 2. 前向传播
    loss = criterion(outputs, targets)  # 3. 计算损失
    loss.backward()             # 4. 反向传播
    optimizer.step()            # 5. 更新参数
```

### 5.4 训练与评估模式

```python
model.train()  # 启用 Dropout, BatchNorm 等
model.eval()   # 禁用 Dropout, BatchNorm 等
```

| 特性 | model.eval() | torch.no_grad() |
|------|-------------|----------------|
| 作用对象 | 层的功能行为 | 梯度计算引擎 |
| 具体影响 | Dropout 不丢弃；BatchNorm 用统计均值 | 禁止构建计算图 |
| 是否省显存 | 否 | 是 |

两者通常成对出现。新版 PyTorch 中可用 `torch.inference_mode()` 替代 `torch.no_grad()`，性能更好。

### 5.5 梯度相关操作

```python
# 反向传播计算梯度
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor([4.])，对 y=x² 求导，x=2 时梯度为 4

# detach 分离梯度
z = y.detach()  # 不参与梯度计算

# 冻结预训练模型参数（迁移学习）
for param in resnet.parameters():
    param.requires_grad_(False)
```

---

## 6. 矩阵运算

### 6.1 矩阵乘法

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[7, 8], [9, 10], [11, 12]])

a @ b              # 等价于 torch.matmul(a, b)
# tensor([[ 58,  64],
#         [139, 154]])

# 批量矩阵乘法
A = torch.randn(10, 2, 3)
B = torch.randn(10, 3, 4)
C = torch.matmul(A, B)  # (10, 2, 4)
```

### 6.2 掩码操作

```python
x = torch.randn(2, 3)
mask = torch.tensor([[True, False, True], [False, True, False]])
result = x.masked_fill(mask, -1e9)  # mask 为 True 的位置填 -1e9
```

三角掩码（Transformer 中用于防止看到未来信息）：

```python
x = torch.ones(4, 4)
torch.triu(x)  # 上三角
torch.tril(x)  # 下三角
```

---

## 7. 一个完整的神经网络

把上面的知识串起来：

```python
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(32, 10)
y = torch.randint(0, 2, (32,))

model.train()
optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()

print(f"损失值: {loss.item()}")
```

---

## 8. torch.compile 简介

PyTorch 2.0 引入了 `torch.compile`，可以将模型编译为更高效的代码：

```python
@torch.compile
def foo(x):
    return torch.sin(x) + 1

foo(torch.randn(10).cpu())
```

在 CPU 上会生成优化的 C++ 代码，在 GPU 上会生成 Triton 代码。

---

## 小结

本篇介绍了 PyTorch 的核心基础：
- **张量**：创建、索引、形状变换、拼接
- **nn.Module**：继承、forward、调用链
- **常用层**：Linear、Embedding、LayerNorm、Dropout
- **训练流程**：损失函数、优化器、梯度、设备管理

掌握这些，你就有了手写 Transformer 的全部"原材料"。下一篇我们将构建翻译数据集和 Transformer 的输入层。

---

下一篇：[数据处理与 Transformer 输入层 >>](02-data-and-input-layer.md)
