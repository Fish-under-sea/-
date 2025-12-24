#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电影评论情感分析 - 数据预处理脚本
优化版本，支持更好的数据分割和验证
修复数据预处理 - 最终版本
"""

import pandas as pd
import json
import re
import os
import random
import numpy as np

print("=" * 60)
print("修复数据预处理 - 最终版本")
print("=" * 60)

# 配置
CONFIG = {
    "data_path": "./datasets",
    "label_map": {
        "positive": 0,
        "negative": 1,
        "neutral": 2
    },
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "seed": 42
}

def clean_text(text):
    """清洗文本"""
    if not isinstance(text, str):
        return ""
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 保留中文、英文、数字和基本标点
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；："\'、（）《》【】]', '', text)
    return text.strip()

def get_label_counts(df):
    """获取标签计数（转换为Python原生类型）"""
    counts = {}
    for label in ['positive', 'negative', 'neutral']:
        count = len(df[df['label'] == label])
        counts[label] = int(count)  # 转换为int
    return counts

def main():
    """主函数"""
    
    print("[INFO] 开始数据预处理...")
    
    # 1. 创建目录
    raw_dir = f"{CONFIG['data_path']}/raw"
    processed_dir = f"{CONFIG['data_path']}/processed"
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"[OK] 创建目录: {raw_dir}")
    print(f"[OK] 创建目录: {processed_dir}")
    
    # 2. 保存标签文件
    labels_file = f"{CONFIG['data_path']}/labels.json"
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(CONFIG["label_map"], f, ensure_ascii=False, indent=2)
    print(f"[OK] 标签文件已保存: {labels_file}")
    
    # 3. 检查原始语料
    raw_file = f"{raw_dir}/campus_corpus.txt"
    
    if not os.path.exists(raw_file):
        print(f"[ERROR] 找不到语料文件: {raw_file}")
        return
    
    print(f"[INFO] 读取语料文件: {raw_file}")
    
    # 4. 读取和解析数据
    data = []
    current_label = None
    
    try:
        with open(raw_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"[INFO] 读取了 {len(lines)} 行")
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 检测标签行
                if line.startswith('# 正面评价'):
                    current_label = 'positive'
                elif line.startswith('# 负面评价'):
                    current_label = 'negative'
                elif line.startswith('# 中性评价'):
                    current_label = 'neutral'
                elif line.startswith('#'):
                    continue
                elif current_label is not None:
                    cleaned = clean_text(line)
                    if cleaned:
                        data.append({
                            'text': cleaned,
                            'label': current_label,
                            'label_id': CONFIG["label_map"][current_label]
                        })
        
        print(f"[INFO] 解析出 {len(data)} 条有效数据")
        
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {str(e)}")
        return
    
    if not data:
        print("[ERROR] 没有找到有效数据")
        return
    
    # 5. 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 6. 检查数据分布
    print("\n[INFO] 数据分布统计:")
    print("-" * 40)
    
    for label in ['positive', 'negative', 'neutral']:
        count = len(df[df['label'] == label])
        print(f"  {label}: {count} 条")
    
    print(f"  总计: {len(df)} 条")
    
    # 7. 随机分割数据
    print(f"\n[INFO] 分割数据集 ({CONFIG['train_ratio']:.0%}/{CONFIG['val_ratio']:.0%}/{CONFIG['test_ratio']:.0%})...")
    
    random.seed(CONFIG["seed"])
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    train_end = int(len(df) * CONFIG["train_ratio"])
    val_end = train_end + int(len(df) * CONFIG["val_ratio"])
    
    train_df = df.iloc[indices[:train_end]]
    val_df = df.iloc[indices[train_end:val_end]]
    test_df = df.iloc[indices[val_end:]]
    
    # 8. 保存数据
    train_file = f"{processed_dir}/train.csv"
    val_file = f"{processed_dir}/val.csv"
    test_file = f"{processed_dir}/test.csv"
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    # 9. 打印统计
    print("\n[STATS] 详细统计:")
    print("=" * 60)
    
    train_counts = get_label_counts(train_df)
    val_counts = get_label_counts(val_df)
    test_counts = get_label_counts(test_df)
    
    print(f"总样本数: {len(df)}")
    print(f"  正面评价: {len(df[df['label']=='positive'])} 条")
    print(f"  负面评价: {len(df[df['label']=='negative'])} 条")
    print(f"  中性评价: {len(df[df['label']=='neutral'])} 条")
    
    print(f"\n训练集: {len(train_df)} 条")
    for label, count in train_counts.items():
        proportion = count / len(train_df) if len(train_df) > 0 else 0
        print(f"  {label}: {count} 条 ({proportion:.1%})")
    
    print(f"\n验证集: {len(val_df)} 条")
    for label, count in val_counts.items():
        proportion = count / len(val_df) if len(val_df) > 0 else 0
        print(f"  {label}: {count} 条 ({proportion:.1%})")
    
    print(f"\n测试集: {len(test_df)} 条")
    for label, count in test_counts.items():
        proportion = count / len(test_df) if len(test_df) > 0 else 0
        print(f"  {label}: {count} 条 ({proportion:.1%})")
    
    print(f"\n文件路径:")
    print(f"  训练集: {train_file}")
    print(f"  验证集: {val_file}")
    print(f"  测试集: {test_file}")
    print("=" * 60)
    
    # 10. 保存统计信息（修复JSON序列化问题）
    stats = {
        "total_samples": int(len(df)),
        "positive_samples": int(len(df[df['label']=='positive'])),
        "negative_samples": int(len(df[df['label']=='negative'])),
        "neutral_samples": int(len(df[df['label']=='neutral'])),
        "train_samples": int(len(train_df)),
        "train_label_distribution": train_counts,
        "val_samples": int(len(val_df)),
        "val_label_distribution": val_counts,
        "test_samples": int(len(test_df)),
        "test_label_distribution": test_counts,
        "split_method": "random",
        "split_ratios": {
            "train": float(CONFIG["train_ratio"]),
            "val": float(CONFIG["val_ratio"]),
            "test": float(CONFIG["test_ratio"])
        }
    }
    
    stats_file = f"{processed_dir}/dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] 统计信息已保存: {stats_file}")
    
    # 11. 数据质量检查
    print("\n[QUALITY] 数据质量检查:")
    print("-" * 60)
    
    # 检查文本长度
    df['text_length'] = df['text'].str.len()
    avg_length = float(df['text_length'].mean())
    min_length = int(df['text_length'].min())
    max_length = int(df['text_length'].max())
    
    print(f"文本长度统计:")
    print(f"  平均长度: {avg_length:.1f} 字符")
    print(f"  最短文本: {min_length} 字符")
    print(f"  最长文本: {max_length} 字符")
    
    # 检查类别平衡
    class_counts = df['label'].value_counts()
    print(f"\n类别分布:")
    for label in ['positive', 'negative', 'neutral']:
        count = class_counts.get(label, 0)
        proportion = count / len(df) if len(df) > 0 else 0
        print(f"  {label}: {count} 条 ({proportion:.1%})")
    
    print("\n[DONE] 数据预处理完成!")
    print("\n[INFO] 下一步:")
    print("1. 训练模型: python main.py --train-bert")
    print("2. 完整流程: python main.py --all")

if __name__ == "__main__":
    main()