#!/bin/bash

# QLoRA微调训练启动脚本

echo "=== QLoRA微调训练启动脚本 ==="
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到NVIDIA GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠ 警告: 未检测到NVIDIA GPU，训练可能很慢"
fi

echo ""

# 检查依赖
echo "检查依赖包..."
python -c "import torch, transformers, peft, bitsandbytes" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖包..."
    pip install -r requirements.txt
fi

echo ""

# 检查Hugging Face登录
echo "检查Hugging Face访问权限..."
python -c "from huggingface_hub import HfApi; api = HfApi(); api.whoami()" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "请先登录Hugging Face:"
    echo "huggingface-cli login"
    echo ""
    read -p "按回车键继续..."
fi

echo ""

# 开始训练
echo "开始QLoRA微调训练..."
echo "训练日志将显示在下方，按Ctrl+C可中断训练"
echo ""

python train.py

echo ""
echo "训练完成！"
echo "模型已保存到: ./llama2-7b-qlora-finetuned/"
echo ""
echo "运行以下命令测试模型:"
echo "python inference.py" 