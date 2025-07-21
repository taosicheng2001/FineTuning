"""
推理脚本 - 测试微调后的模型
"""
import torch
from transformers import AutoTokenizer
from model_setup import load_finetuned_model
from config import TrainingConfig


def generate_response(model, tokenizer, prompt: str, max_length: int = 512) -> str:
    """生成回复"""
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    
    # 移动到GPU（如果可用）
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 移除输入部分，只返回生成的回复
    response = response[len(prompt):].strip()
    
    return response


def interactive_chat(model, tokenizer):
    """交互式聊天"""
    print("=== 微调模型聊天界面 ===")
    print("输入 'quit' 退出聊天")
    print("-" * 50)
    
    while True:
        user_input = input("\n用户: ")
        
        if user_input.lower() == 'quit':
            print("再见！")
            break
        
        if user_input.strip():
            # 构建提示
            prompt = f"### Human: {user_input}\n\n### Assistant:"
            
            print("助手: ", end="", flush=True)
            
            # 生成回复
            response = generate_response(model, tokenizer, prompt)
            print(response)


def test_examples(model, tokenizer):
    """测试示例"""
    test_prompts = [
        "请解释什么是机器学习？",
        "如何制作一杯咖啡？",
        "写一首关于春天的诗",
        "解释量子计算的基本原理",
        "如何学习一门新的编程语言？"
    ]
    
    print("=== 测试示例 ===")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n测试 {i}: {prompt}")
        print("-" * 40)
        
        full_prompt = f"### Human: {prompt}\n\n### Assistant:"
        response = generate_response(model, tokenizer, full_prompt)
        print(f"回复: {response}")
        print("-" * 40)


def main():
    """主函数"""
    config = TrainingConfig()
    
    # 检查模型路径
    model_path = config.output_dir
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        print("请先运行训练脚本: python train.py")
        return
    
    print("正在加载微调后的模型...")
    
    try:
        # 加载模型
        model, tokenizer = load_finetuned_model(model_path, config.model_name)
        print("模型加载成功！")
        
        # 选择模式
        print("\n请选择模式:")
        print("1. 交互式聊天")
        print("2. 测试示例")
        
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            interactive_chat(model, tokenizer)
        elif choice == "2":
            test_examples(model, tokenizer)
        else:
            print("无效选择，退出程序")
            
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("请确保模型已正确训练并保存")


if __name__ == "__main__":
    import os
    main() 