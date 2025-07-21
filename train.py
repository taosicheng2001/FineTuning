"""
QLoRA微调主训练脚本
"""
import os
import torch
import wandb
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from accelerate import Accelerator
from config import TrainingConfig, DataConfig
from data_processor import DataProcessor, create_data_collator
from model_setup import setup_training_model, save_model


def setup_wandb(config: TrainingConfig):
    """设置WandB"""
    wandb.init(
        project="llama2-qlora-finetuning",
        config={
            "model_name": config.model_name,
            "dataset_name": config.dataset_name,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "batch_size": config.per_device_train_batch_size,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
        }
    )


def main():
    """主训练函数"""
    # 设置配置
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    # 设置随机种子
    torch.manual_seed(training_config.seed)
    
    # 设置WandB
    setup_wandb(training_config)
    
    print("=== QLoRA微调开始 ===")
    print(f"模型: {training_config.model_name}")
    print(f"数据集: {training_config.dataset_name}")
    print(f"输出目录: {training_config.output_dir}")
    
    # 设置模型和分词器
    print("\n1. 设置模型和分词器...")
    model, tokenizer = setup_training_model(training_config)
    
    # 设置数据处理器
    print("\n2. 设置数据处理器...")
    data_processor = DataProcessor(tokenizer, data_config)
    
    # 准备训练数据
    print("\n3. 准备训练数据...")
    train_dataset = data_processor.prepare_dataset(
        training_config.dataset_name,
        training_config.dataset_split
    )
    
    # 创建数据整理器
    data_collator = create_data_collator(tokenizer)
    
    # 设置训练参数
    print("\n4. 设置训练参数...")
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_steps=training_config.warmup_steps,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=training_config.bf16,
        fp16=training_config.fp16,
        dataloader_pin_memory=training_config.dataloader_pin_memory,
        remove_unused_columns=training_config.remove_unused_columns,
        report_to="wandb",
        run_name=f"llama2-qlora-{training_config.dataset_name.split('/')[-1]}",
    )
    
    # 创建训练器
    print("\n5. 创建训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("\n6. 开始训练...")
    trainer.train()
    
    # 保存模型
    print("\n7. 保存模型...")
    save_model(model, tokenizer, training_config.output_dir)
    
    # 关闭WandB
    wandb.finish()
    
    print("\n=== 训练完成！ ===")
    print(f"模型已保存到: {training_config.output_dir}")


if __name__ == "__main__":
    main() 