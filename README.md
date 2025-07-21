# QLoRA微调Llama2-7B框架

这是一个使用QLoRA（Quantized Low-Rank Adaptation）方法微调Llama2-7B大模型的完整代码框架。

## 特性

- 🚀 **QLoRA微调**: 使用4-bit量化和LoRA适配器，大幅减少显存需求
- 📊 **自动数据处理**: 支持多种开源数据集格式
- 🔧 **灵活配置**: 易于调整的超参数和训练设置
- 📈 **训练监控**: 集成WandB进行训练过程监控
- 💾 **模型保存**: 自动保存最佳模型和LoRA权重
- 🤖 **推理测试**: 提供交互式聊天和批量测试功能

## 环境要求

- Python 3.8+
- CUDA 11.8+ (推荐)
- 至少16GB显存 (使用QLoRA后)
- 至少32GB系统内存

## 安装

1. 克隆项目：
```bash
git clone <repository-url>
cd FineTuning
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 设置Hugging Face访问权限（需要访问Llama2模型）：
```bash
huggingface-cli login
```

## 数据集选择

本框架默认使用 `timdettmers/openassistant-guanaco` 数据集，这是一个高质量的指令微调数据集。您也可以在 `config.py` 中修改为其他数据集：

### 推荐数据集：
- `timdettmers/openassistant-guanaco` - OpenAssistant对话数据
- `microsoft/DialoGPT-medium` - 对话生成数据
- `databricks/databricks-dolly-15k` - Dolly指令数据
- `tatsu-lab/alpaca` - Alpaca指令数据

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练过程将：
- 自动下载Llama2-7B模型
- 加载并预处理数据集
- 应用QLoRA配置
- 开始微调训练
- 保存最佳模型到 `./llama2-7b-qlora-finetuned/`

### 2. 测试模型

```bash
python inference.py
```

提供两种测试模式：
- **交互式聊天**: 与模型进行实时对话
- **批量测试**: 运行预设的测试用例

## 配置说明

### 主要配置参数 (`config.py`)

#### 模型配置
- `model_name`: 基础模型名称
- `output_dir`: 模型保存路径

#### 训练配置
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `per_device_train_batch_size`: 每设备批次大小
- `gradient_accumulation_steps`: 梯度累积步数

#### QLoRA配置
- `lora_r`: LoRA秩 (rank)
- `lora_alpha`: LoRA缩放参数
- `lora_dropout`: LoRA dropout率
- `target_modules`: 应用LoRA的目标模块

#### 量化配置
- `bf16`: 使用bfloat16精度
- `fp16`: 使用fp16精度

## 文件结构

```
FineTuning/
├── config.py              # 配置文件
├── data_processor.py      # 数据处理模块
├── model_setup.py         # 模型设置模块
├── train.py              # 主训练脚本
├── inference.py          # 推理测试脚本
├── requirements.txt      # 依赖包列表
└── README.md            # 说明文档
```

## 训练监控

训练过程会自动记录到WandB，包括：
- 训练损失曲线
- 学习率变化
- 梯度范数
- 模型参数统计

## 性能优化建议

### 显存优化
- 降低 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用更小的 `lora_r` 值

### 训练效果优化
- 调整学习率和训练轮数
- 尝试不同的数据集
- 调整LoRA参数

## 常见问题

### Q: 显存不足怎么办？
A: 降低批次大小或增加梯度累积步数，也可以减小LoRA的rank值。

### Q: 如何更换数据集？
A: 在 `config.py` 中修改 `dataset_name` 参数。

### Q: 训练中断后如何继续？
A: 训练器会自动从检查点恢复，只需重新运行 `python train.py`。

### Q: 如何调整模型输出质量？
A: 在 `inference.py` 中调整生成参数如 `temperature`、`top_p` 等。

## 许可证

本项目遵循MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个框架！

## 致谢

- [QLoRA论文](https://arxiv.org/abs/2305.14314)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT库](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) 