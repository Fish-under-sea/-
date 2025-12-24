#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电影评论情感分析 - BERT模型训练脚本
修复版本 - 适配最新transformers库
已添加详细的训练进度显示
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
import json
import os
import time
from tqdm import tqdm
from datetime import datetime
import sys

print("=" * 70)
print("电影评论情感分析 - BERT模型训练")
print("=" * 70)

# 强制使用CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = torch.device('cpu')
print(f"[INFO] 使用设备: {device}")

# 检查transformers版本，选择正确的AdamW导入方式
try:
    from transformers import AdamW
    print("[INFO] 使用transformers.AdamW")
except ImportError:
    from torch.optim import AdamW
    print("[INFO] 使用torch.optim.AdamW")

# 配置
CONFIG = {
    "model_name": "bert-base-chinese",
    "model_path": "./models/bert-base-chinese",  # 本地模型路径
    "finetuned_path": "./models/finetuned",
    "data_path": "./datasets",
    "num_labels": 8,
    "max_length": 128,
    "batch_size": 4,  # 减小批大小，避免内存问题
    "learning_rate": 2e-5,
    "num_epochs": 8,
    "warmup_steps": 100
}

# 数据集类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """加载数据"""
    print("[INFO] 加载数据集...")
    
    try:
        # 加载训练集
        train_file = f"{CONFIG['data_path']}/processed/train.csv"
        if not os.path.exists(train_file):
            print(f"[ERROR] 找不到训练文件: {train_file}")
            return None, None, None
        
        train_df = pd.read_csv(train_file)
        print(f"[OK] 训练集: {len(train_df)} 条")
        
        # 加载验证集
        val_file = f"{CONFIG['data_path']}/processed/val.csv"
        if os.path.exists(val_file):
            val_df = pd.read_csv(val_file)
            print(f"[OK] 验证集: {len(val_df)} 条")
        else:
            print(f"[WARNING] 找不到验证文件，使用训练集的一部分作为验证")
            # 从训练集分一部分作为验证
            split_idx = int(len(train_df) * 0.8)
            val_df = train_df[split_idx:]
            train_df = train_df[:split_idx]
            print(f"[INFO] 重新划分: 训练集 {len(train_df)} 条，验证集 {len(val_df)} 条")
        
        # 加载测试集
        test_file = f"{CONFIG['data_path']}/processed/test.csv"
        if os.path.exists(test_file):
            test_df = pd.read_csv(test_file)
            print(f"[OK] 测试集: {len(test_df)} 条")
        else:
            print(f"[WARNING] 找不到测试文件")
            test_df = pd.DataFrame()
        
        return train_df, val_df, test_df
        
    except Exception as e:
        print(f"[ERROR] 加载数据失败: {e}")
        return None, None, None

def create_simple_dataset():
    """创建简单数据集（备选方案）"""
    print("[INFO] 创建简单示例数据集...")
    
    simple_data = [
        {"text": "打戏很真，看的很过瘾", "label_id": 0},
        {"text": "电影票价过高", "label_id": 1},
        {"text": "是个喜剧电影", "label_id": 2},
    ]
    
    df = pd.DataFrame(simple_data)
    split_idx = int(len(df) * 0.7)
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    return train_df, val_df, pd.DataFrame()

def display_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    """显示进度条"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '░' * (length - filled_length)
    print(f'\r{prefix} [{bar}] {percent}% {suffix}', end='')
    if iteration == total:
        print()

class TrainingProgress:
    """训练进度管理器"""
    def __init__(self, total_epochs, total_batches):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.start_time = time.time()
        self.epoch_start_time = None
        
    def start_epoch(self, epoch):
        """开始新epoch"""
        self.epoch_start_time = time.time()
        self.current_epoch = epoch + 1
        print(f"\n[EPOCH] Epoch {self.current_epoch}/{self.total_epochs}")
        print("-" * 50)
        
    def update_batch(self, batch_idx, loss, accuracy):
        """更新批次进度"""
        elapsed = time.time() - self.epoch_start_time
        eta = (elapsed / (batch_idx + 1)) * (self.total_batches - batch_idx - 1)
        
        progress = (batch_idx + 1) / self.total_batches * 100
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        
        sys.stdout.write(f"\r[PROGRESS] 批次 {batch_idx+1}/{self.total_batches} "
                        f"| 损失: {loss:.4f} | 准确率: {accuracy:.2%} | "
                        f"进度: {progress:.1f}% | ETA: {eta_str}")
        sys.stdout.flush()
    
    def end_epoch(self, epoch_loss, epoch_accuracy):
        """结束epoch"""
        epoch_time = time.time() - self.epoch_start_time
        epoch_time_str = time.strftime("%H:%M:%S", time.gmtime(epoch_time))
        
        print(f"\n[RESULT] Epoch完成 | 时间: {epoch_time_str} | "
              f"平均损失: {epoch_loss:.4f} | 准确率: {epoch_accuracy:.2%}")
    
    def summary(self):
        """训练总结"""
        total_time = time.time() - self.start_time
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
        
        print(f"\n[SUMMARY] 训练总时间: {total_time_str}")
        print(f"[SUMMARY] 平均每epoch时间: {total_time/self.total_epochs:.1f}秒")

def main():
    """主函数"""
    
    print(f"[INFO] PyTorch版本: {torch.__version__}")
    
    # 1. 检查模型文件
    print(f"[INFO] 检查模型文件: {CONFIG['model_path']}")
    if not os.path.exists(CONFIG["model_path"]):
        print(f"[ERROR] 模型路径不存在: {CONFIG['model_path']}")
        print("[INFO] 请先运行: python setup_bert_model.py")
        return
    
    # 2. 加载tokenizer
    print("\n[INFO] 加载BERT tokenizer...")
    try:
        # 首先尝试从本地加载
        tokenizer_path = CONFIG["model_path"]
        if os.path.exists(os.path.join(tokenizer_path, "vocab.txt")):
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            print(f"[OK] 从本地加载tokenizer，词汇表大小: {tokenizer.vocab_size}")
        else:
            # 如果本地没有，从网络加载
            tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
            print(f"[OK] 从网络加载tokenizer，词汇表大小: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"[ERROR] 加载tokenizer失败: {e}")
        return
    
    # 3. 加载数据
    train_df, val_df, test_df = load_data()
    if train_df is None or len(train_df) < 3:
        print("[WARNING] 数据不足或加载失败，使用简单数据集...")
        train_df, val_df, test_df = create_simple_dataset()
    
    print(f"[OK] 训练数据: {len(train_df)} 条，验证数据: {len(val_df)} 条")
    
    # 4. 创建数据集
    print("\n[INFO] 创建数据集...")
    
    # 确保label_id是整数类型
    train_df['label_id'] = train_df['label_id'].astype(int)
    if len(val_df) > 0:
        val_df['label_id'] = val_df['label_id'].astype(int)
    
    train_dataset = SentimentDataset(
        train_df['text'].values,
        train_df['label_id'].values,
        tokenizer,
        max_length=CONFIG["max_length"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )
    
    if len(val_df) > 0:
        val_dataset = SentimentDataset(
            val_df['text'].values,
            val_df['label_id'].values,
            tokenizer,
            max_length=CONFIG["max_length"]
        )
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    else:
        val_loader = None
        print("[WARNING] 没有验证数据")
    
    # 5. 加载BERT模型
    print("\n[INFO] 加载BERT模型...")
    try:
        # 尝试从本地加载
        model = BertForSequenceClassification.from_pretrained(
            CONFIG["model_path"],
            num_labels=CONFIG["num_labels"]
        )
        print("[OK] 从本地加载BERT模型")
    except Exception as e:
        print(f"[WARNING] 本地加载失败: {e}")
        print("[INFO] 尝试从网络加载...")
        try:
            model = BertForSequenceClassification.from_pretrained(
                CONFIG["model_name"],
                num_labels=CONFIG["num_labels"]
            )
            print("[OK] 从网络加载BERT模型")
        except Exception as e2:
            print(f"[ERROR] 加载模型失败: {e2}")
            return
    
    model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 总参数量: {total_params:,}")
    print(f"[INFO] 可训练参数: {trainable_params:,}")
    print(f"[INFO] 模型大小: {total_params * 4 / (1024*1024):.2f} MB")
    
    # 6. 设置优化器
    print("\n[INFO] 设置优化器...")
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=0.01
    )
    
    # 7. 训练循环
    print(f"\n[INFO] 开始训练，共 {CONFIG['num_epochs']} 个epoch")
    print("=" * 70)
    
    best_val_accuracy = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # 初始化进度管理器
    progress = TrainingProgress(CONFIG["num_epochs"], len(train_loader))
    
    for epoch in range(CONFIG["num_epochs"]):
        progress.start_epoch(epoch)
        
        # 训练模式
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 获取数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            _, predictions = torch.max(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # 更新进度
            current_acc = correct / total if total > 0 else 0
            current_loss = epoch_loss / (batch_idx + 1)
            progress.update_batch(batch_idx, current_loss, current_acc)
        
        # 计算训练准确率
        train_accuracy = correct / total if total > 0 else 0
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_accuracy)
        
        progress.end_epoch(avg_loss, train_accuracy)
        
        # 验证
        if val_loader is not None and len(val_loader) > 0:
            print("\n[INFO] 验证模型...")
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    _, predictions = torch.max(outputs.logits, dim=1)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
                    
                    # 显示验证进度
                    progress = (batch_idx + 1) / len(val_loader) * 100
                    print(f"\r[VALIDATION] 进度: {progress:.1f}%", end='')
            
            if val_total > 0:
                val_accuracy = val_correct / val_total
                history['val_acc'].append(val_accuracy)
                print(f"\n[RESULT] 验证准确率: {val_accuracy:.2%}")
                
                # 保存最佳模型
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    print(f"[SAVE] 保存最佳模型 (准确率: {val_accuracy:.2%})")
                    
                    os.makedirs(CONFIG["finetuned_path"], exist_ok=True)
                    
                    # 保存模型
                    model.save_pretrained(CONFIG["finetuned_path"])
                    tokenizer.save_pretrained(CONFIG["finetuned_path"])
                    
                    # 保存配置
                    config_file = os.path.join(CONFIG["finetuned_path"], "training_config.json")
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(CONFIG, f, ensure_ascii=False, indent=2)
                    
                    print(f"[OK] 模型保存到: {CONFIG['finetuned_path']}")
        else:
            print("[INFO] 跳过验证步骤")
    
    # 显示训练总结
    progress.summary()
    
    # 8. 如果之前没有保存模型，现在保存
    if not os.path.exists(CONFIG["finetuned_path"]) or len(os.listdir(CONFIG["finetuned_path"])) == 0:
        print("\n[INFO] 保存最终模型...")
        os.makedirs(CONFIG["finetuned_path"], exist_ok=True)
        model.save_pretrained(CONFIG["finetuned_path"])
        tokenizer.save_pretrained(CONFIG["finetuned_path"])
        print(f"[OK] 最终模型保存到: {CONFIG['finetuned_path']}")
    
    # 9. 测试（如果有测试数据）
    if test_df is not None and len(test_df) > 0:
        print("\n" + "=" * 70)
        print("[TEST] 测试模型")
        print("=" * 70)
        
        test_df['label_id'] = test_df['label_id'].astype(int)
        test_dataset = SentimentDataset(
            test_df['text'].values,
            test_df['label_id'].values,
            tokenizer,
            max_length=CONFIG["max_length"]
        )
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])
        
        model.eval()
        test_correct = 0
        test_total = 0
        
        print("[INFO] 测试模型...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, predictions = torch.max(outputs.logits, dim=1)
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)
                
                progress = (batch_idx + 1) / len(test_loader) * 100
                print(f"\r[TESTING] 进度: {progress:.1f}%", end='')
        
        if test_total > 0:
            test_accuracy = test_correct / test_total
            print(f"\n[RESULT] 测试准确率: {test_accuracy:.2%}")
        else:
            test_accuracy = 0
            print("[WARNING] 没有测试数据")
    else:
        test_accuracy = 0
        print("[WARNING] 跳过测试步骤")
    
    # 10. 保存训练结果
    results = {
        "best_val_accuracy": float(best_val_accuracy),
        "test_accuracy": float(test_accuracy),
        "num_epochs": CONFIG["num_epochs"],
        "batch_size": CONFIG["batch_size"],
        "learning_rate": CONFIG["learning_rate"],
        "train_samples": len(train_df),
        "val_samples": len(val_df) if val_df is not None else 0,
        "test_samples": len(test_df) if test_df is not None else 0,
        "training_history": history
    }
    
    os.makedirs("results", exist_ok=True)
    results_path = 'results/training_results.json'
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("[DONE] 训练完成!")
    print("=" * 70)
    print(f"最佳验证准确率: {best_val_accuracy:.2%}")
    print(f"测试准确率: {test_accuracy:.2%}")
    print(f"模型保存到: {CONFIG['finetuned_path']}")
    print(f"训练结果: {results_path}")
    
    print("\n[INFO] 下一步:")
    print("1. 评估模型: python scripts/evaluate_cpu.py --mode eval")
    print("2. 交互演示: python scripts/evaluate_cpu.py --mode demo")
    print("3. 启动Web应用: python app.py")

if __name__ == "__main__":
    main()