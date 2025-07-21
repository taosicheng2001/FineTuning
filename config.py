"""
QLoRA微调配置文件
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # 模型配置
    model_name: str = "meta-llama/Llama-2-7b-hf"
    output_dir: str = "./llama2-7b-qlora-finetuned"
    
    # 数据集配置
    dataset_name: str = "timdettmers/openassistant-guanaco"
    dataset_split: str = "train"
    max_length: int = 2048
    
    # 训练配置
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # QLoRA配置
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list = None
    
    # 量化配置
    bf16: bool = True
    fp16: bool = False
    
    # 其他配置
    seed: int = 42
    max_grad_norm: float = 0.3
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

@dataclass
class DataConfig:
    # 数据处理配置
    max_length: int = 2048
    truncation: bool = True
    padding: str = "max_length"
    return_tensors: str = "pt"
    
    # 模板配置
    template: str = "### Human: {instruction}\n\n### Assistant: {response}"
    
    # 过滤配置
    min_length: int = 100
    max_length_filter: int = 2048 