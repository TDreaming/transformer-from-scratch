# NumOCR

一个用 PyTorch 实现的简易数字识别示例，包含：

- 合成数字数据集生成
- CNN 模型定义
- 模型训练
- 模型预测

## 建议先读

- `README.md`: 快速命令入口
- `GUIDE.md`: 设计思路、实现考虑和阅读指南

## 文件说明

- `data.py`: 生成 28x28 的七段数码管风格数字图像
- `model.py`: 简单卷积神经网络
- `train.py`: 训练并保存模型
- `predict.py`: 加载模型并执行预测
- `tests/test_numocr.py`: 单元测试
- `GUIDE.md`: 完整设计与上手文档

## 训练

在项目根目录执行：

```bash
python3 -m mylearn.numocr.train
```

你也可以缩短训练时间，例如：

```bash
python3 -m mylearn.numocr.train --epochs 4 --train-size 6000 --val-size 1000
```

## 预测

用程序先生成一个测试数字再预测：

```bash
python3 -m mylearn.numocr.predict --digit 8
```

如果你有本地图片，也可以直接预测：

```bash
python3 -m mylearn.numocr.predict --image path/to/digit.png
```

说明：

- 训练数据是代码动态生成的，不依赖网络下载
- 模型权重默认保存在 `artifacts/numocr_model.pt`
- 图片预测建议使用黑底白字；如果是白底黑字，脚本会自动尝试反相

## 单元测试

执行：

```bash
python3 -m unittest discover -s mylearn/numocr/tests -v
```
