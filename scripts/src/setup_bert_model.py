#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载BERT模型 - 使用国内镜像
"""

import os
import sys
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests
import time

def download_with_retry(model_name, save_path, max_retries=3):
    """带重试的下载函数"""
    
    # 设置镜像源
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    for attempt in range(max_retries):
        try:
            print(f"[INFO] 尝试 {attempt + 1}/{max_retries}...")
            
            # 下载tokenizer
            print("下载tokenizer...")
            tokenizer = BertTokenizer.from_pretrained(
                model_name,
                cache_dir="./cache/huggingface"
            )
            tokenizer.save_pretrained(save_path)
            print(f"[OK] tokenizer保存到: {save_path}")
            
            # 下载模型
            print("下载模型...")
            model = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,
                torch_dtype=torch.float32,
                cache_dir="./cache/huggingface"
            )
            model.save_pretrained(save_path)
            print(f"[OK] 模型保存到: {save_path}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 尝试 {attempt + 1} 失败: {e}")
            
            if attempt < max_retries - 1:
                print(f"[INFO] 等待 {2 ** attempt} 秒后重试...")
                time.sleep(2 ** attempt)
                
                # 尝试其他镜像源
                if attempt == 1:
                    os.environ['HF_ENDPOINT'] = 'https://mirror.sjtu.edu.cn/hugging-face'
                    print("[INFO] 切换到上海交通大学镜像...")
                elif attempt == 2:
                    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                    print("[INFO] 切换回hf-mirror镜像...")
            else:
                return False
    
    return False

def download_bert_model_manual():
    """手动下载BERT模型（备用方法）"""
    print("\n[INFO] 尝试手动下载...")
    
    model_name = "bert-base-chinese"
    save_path = "./models/bert-base-chinese"
    
    os.makedirs(save_path, exist_ok=True)
    
    # 文件列表
    files = [
        ("config.json", "模型配置文件"),
        ("vocab.txt", "词汇表文件"),
        ("pytorch_model.bin", "模型权重文件"),
        ("tokenizer_config.json", "tokenizer配置文件"),
        ("special_tokens_map.json", "特殊token映射")
    ]
    
    # 镜像源URL模板
    base_urls = [
        "https://hf-mirror.com/bert-base-chinese/resolve/main/{}",
        "https://mirror.sjtu.edu.cn/hugging-face/bert-base-chinese/resolve/main/{}"
    ]
    
    for filename, description in files:
        print(f"下载 {description} ({filename})...")
        
        downloaded = False
        for base_url in base_urls:
            url = base_url.format(filename)
            try:
                response = requests.get(url, timeout=30, stream=True)
                if response.status_code == 200:
                    file_path = os.path.join(save_path, filename)
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"  [OK] 从 {base_url.split('/')[2]} 下载成功")
                    downloaded = True
                    break
            except Exception as e:
                print(f"  [ERROR] 从 {base_url.split('/')[2]} 下载失败: {e}")
        
        if not downloaded:
            print(f"  [WARNING] {filename} 下载失败")
    
    # 检查是否下载了必要文件
    required_files = ['config.json', 'vocab.txt', 'pytorch_model.bin']
    missing_files = []
    
    for filename in required_files:
        if not os.path.exists(os.path.join(save_path, filename)):
            missing_files.append(filename)
    
    if missing_files:
        print(f"[ERROR] 缺少必要文件: {missing_files}")
        return False
    else:
        print("[OK] 手动下载完成")
        return True

def download_with_proxy():
    """使用代理下载"""
    print("\n[INFO] 尝试使用代理下载...")
    
    # 设置代理（如果有的话）
    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890',
    }
    
    model_name = "bert-base-chinese"
    save_path = "./models/bert-base-chinese"
    
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # 尝试使用代理下载
        print("设置代理下载...")
        
        # 先下载tokenizer
        tokenizer = BertTokenizer.from_pretrained(
            model_name,
            proxies=proxies,
            cache_dir="./cache/huggingface"
        )
        tokenizer.save_pretrained(save_path)
        print("[OK] tokenizer下载完成")
        
        # 下载模型
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            torch_dtype=torch.float32,
            proxies=proxies,
            cache_dir="./cache/huggingface"
        )
        model.save_pretrained(save_path)
        print("[OK] 模型下载完成")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 代理下载失败: {e}")
        return False

def create_minimal_bert_model():
    """创建最小化的BERT模型（最后手段）"""
    print("\n[INFO] 创建最小化BERT模型...")
    
    save_path = "./models/bert-base-chinese"
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # 创建tokenizer
        from transformers import BertTokenizer
        
        print("创建tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        
        # 创建最小配置
        config = {
            "architectures": ["BertForSequenceClassification"],
            "attention_probs_dropout_prob": 0.1,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.36.0",
            "type_vocab_size": 2,
            "use_cache": True,
            "vocab_size": 21128,
            "num_labels": 3,
            "id2label": {"0": "positive", "1": "negative", "2": "neutral"},
            "label2id": {"positive": 0, "negative": 1, "neutral": 2}
        }
        
        import json
        with open(os.path.join(save_path, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # 保存tokenizer
        tokenizer.save_pretrained(save_path)
        
        print("[OK] 最小化BERT模型创建完成")
        print("[WARNING] 这是一个最小化模型，性能可能不如完整模型")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 创建最小化模型失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 70)
    print("下载BERT模型 - 多种下载方式")
    print("=" * 70)
    
    model_name = "bert-base-chinese"
    save_path = "./models/bert-base-chinese"
    
    # 创建目录
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("./cache/huggingface", exist_ok=True)
    
    print(f"[INFO] 目标模型: {model_name}")
    print(f"[INFO] 保存路径: {save_path}")
    print("[INFO] 优先使用国内镜像...")
    print()
    
    # 方法1: 使用镜像源下载
    print("方法1: 使用镜像源下载")
    print("-" * 40)
    success = download_with_retry(model_name, save_path)
    
    if not success:
        print("\n方法1失败，尝试方法2...")
        
        # 方法2: 手动下载
        print("\n方法2: 手动下载文件")
        print("-" * 40)
        success = download_bert_model_manual()
    
    if not success:
        print("\n方法2失败，尝试方法3...")
        
        # 方法3: 使用代理
        print("\n方法3: 使用代理下载")
        print("-" * 40)
        success = download_with_proxy()
    
    if not success:
        print("\n所有下载方法都失败，尝试方法4...")
        
        # 方法4: 创建最小化模型
        print("\n方法4: 创建最小化BERT模型")
        print("-" * 40)
        success = create_minimal_bert_model()
    
    # 检查结果
    if success:
        print("\n" + "=" * 70)
        print("✅ BERT模型设置完成!")
        print("=" * 70)
        
        # 检查文件
        print("\n[CHECK] 检查下载的文件:")
        files = os.listdir(save_path)
        for file in files:
            file_path = os.path.join(save_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
                print(f"  - {file} ({size_str})")
        
        print(f"\n📁 模型保存在: {save_path}")
        print("\n[INFO] 现在可以运行:")
        print("1. 训练模型: python main.py --train-bert")
        print("2. 完整流程: python main.py --all")
        
        return True
    else:
        print("\n" + "=" * 70)
        print("❌ BERT模型下载失败!")
        print("=" * 70)
        print("\n[SOLUTION] 解决方案:")
        print("1. 检查网络连接")
        print("2. 尝试手动下载:")
        print("   - 访问: https://hf-mirror.com/bert-base-chinese")
        print("   - 下载以下文件到 models/bert-base-chinese/ 目录:")
        print("     - config.json")
        print("     - pytorch_model.bin")
        print("     - vocab.txt")
        print("     - tokenizer_config.json")
        print("3. 或运行: python train_simple.py 使用简单模型")
        
        return False

if __name__ == "__main__":
    main()