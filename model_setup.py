"""
模型设置模块 - QLoRA配置
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from config import TrainingConfig


def setup_tokenizer(model_name: str) -> AutoTokenizer:
    """设置分词器"""
    print(f"正在加载分词器: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def setup_model(config: TrainingConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """设置模型和分词器"""
    # 设置量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载模型
    print(f"正在加载模型: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 加载分词器
    tokenizer = setup_tokenizer(config.model_name)
    
    return model, tokenizer


def setup_lora_config(config: TrainingConfig) -> LoraConfig:
    """设置LoRA配置"""
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    return lora_config


def apply_lora(model: AutoModelForCausalLM, lora_config: LoraConfig) -> AutoModelForCausalLM:
    """应用LoRA到模型"""
    print("正在应用LoRA配置...")
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model


def setup_training_model(config: TrainingConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """完整设置训练模型"""
    # 设置模型和分词器
    model, tokenizer = setup_model(config)
    
    # 设置LoRA配置
    lora_config = setup_lora_config(config)
    
    # 应用LoRA
    model = apply_lora(model, lora_config)
    
    return model, tokenizer


def save_model(model, tokenizer, output_dir: str):
    """保存模型"""
    print(f"正在保存模型到: {output_dir}")
    
    # 保存LoRA权重
    model.save_pretrained(output_dir)
    
    # 保存分词器
    tokenizer.save_pretrained(output_dir)
    
    print("模型保存完成！")


def load_finetuned_model(model_path: str, base_model_name: str = "meta-llama/Llama-2-7b-hf"):
    """加载微调后的模型"""
    from peft import PeftModel
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer 