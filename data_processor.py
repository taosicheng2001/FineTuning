"""
数据处理模块
"""
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional
from config import DataConfig


class DataProcessor:
    def __init__(self, tokenizer: AutoTokenizer, config: DataConfig):
        self.tokenizer = tokenizer
        self.config = config
        
    def load_dataset(self, dataset_name: str, split: str = "train") -> Dataset:
        """加载数据集"""
        print(f"正在加载数据集: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        return dataset
    
    def format_prompt(self, example: Dict) -> str:
        """格式化提示"""
        if "instruction" in example and "response" in example:
            return self.config.template.format(
                instruction=example["instruction"],
                response=example["response"]
            )
        elif "text" in example:
            return example["text"]
        else:
            # 尝试其他常见字段
            if "prompt" in example and "completion" in example:
                return f"### Human: {example['prompt']}\n\n### Assistant: {example['completion']}"
            elif "question" in example and "answer" in example:
                return f"### Human: {example['question']}\n\n### Assistant: {example['answer']}"
            else:
                # 如果都不匹配，返回原始文本
                return str(example)
    
    def filter_dataset(self, dataset: Dataset) -> Dataset:
        """过滤数据集"""
        def filter_function(example):
            text = self.format_prompt(example)
            length = len(text.split())
            return self.config.min_length <= length <= self.config.max_length_filter
        
        filtered_dataset = dataset.filter(filter_function)
        print(f"过滤后数据集大小: {len(filtered_dataset)}")
        return filtered_dataset
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """分词函数"""
        texts = [self.format_prompt(example) for example in examples]
        
        tokenized = self.tokenizer(
            texts,
            truncation=self.config.truncation,
            padding=self.config.padding,
            max_length=self.config.max_length,
            return_tensors=self.config.return_tensors
        )
        
        # 设置labels为input_ids的副本
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def prepare_dataset(self, dataset_name: str, split: str = "train") -> Dataset:
        """准备训练数据集"""
        # 加载数据集
        dataset = self.load_dataset(dataset_name, split)
        
        # 过滤数据集
        dataset = self.filter_dataset(dataset)
        
        # 分词处理
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset


def create_data_collator(tokenizer):
    """创建数据整理器"""
    def data_collator(features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["labels"] = torch.stack([f["labels"] for f in features])
        return batch
    
    return data_collator 