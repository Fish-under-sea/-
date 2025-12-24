#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电影评论情感分析 - CPU专用评估脚本
支持BERT模型
"""

import torch
import pandas as pd
import numpy as np
import json
import os
import argparse
from pathlib import Path

print("=" * 70)
print("电影评论情感分析 - CPU评估")
print("=" * 70)

# ==================== 强制使用CPU ====================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = torch.device('cpu')
print("[INFO] 使用CPU进行评估")

# ==================== 配置 ====================
CONFIG = {
    "model_path": "./models/finetuned",
    "data_path": "./datasets",
    "max_length": 128,
    "batch_size": 8
}

# ==================== 数据集类 ====================
class SentimentDataset(torch.utils.data.Dataset):
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
        
        # 使用BERT tokenizer的encode_plus方法
        encoding = self.tokenizer.encode_plus(
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

# ==================== 模型加载辅助函数 ====================
def load_model_and_tokenizer(model_path):
    """加载模型和tokenizer，支持BERT和Qwen"""
    print("[INFO] 加载模型和tokenizer...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # 检查模型配置文件
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            print(f"[ERROR] 找不到模型配置文件: {config_path}")
            return None, None
        
        # 读取配置判断模型类型
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        model_type = config_data.get('model_type', '').lower()
        arch = config_data.get('architectures', [''])[0].lower()
        
        is_qwen = 'qwen' in model_type or 'qwen' in arch
        is_bert = 'bert' in model_type or 'bert' in arch or 'Bert' in arch
        
        print(f"[INFO] 检测到模型类型: {model_type}")
        
        # 根据模型类型加载
        if is_qwen:
            print("[INFO] 使用Qwen模型配置")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        elif is_bert:
            print("[INFO] 使用BERT模型配置")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            print("[INFO] 使用默认配置")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        model.to(device)
        model.eval()
        print("[OK] 模型和tokenizer加载成功")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"[ERROR] 加载模型失败: {e}")
        
        # 尝试简化加载
        try:
            print("[INFO] 尝试简化加载...")
            from transformers import BertTokenizer, BertForSequenceClassification
            
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            model = BertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3
            )
            model.to(device)
            model.eval()
            print("[OK] 使用BERT简化加载成功")
            return model, tokenizer
        except Exception as e2:
            print(f"[ERROR] 简化加载也失败: {e2}")
            return None, None

# ==================== 评估函数 ====================
def evaluate_model(model_path=None, device_type='cpu', trust_remote_code=False):
    """评估模型性能"""
    
    # 如果传入了model_path，更新CONFIG
    if model_path:
        CONFIG["model_path"] = model_path
    
    # 检查模型
    if not os.path.exists(CONFIG["model_path"]):
        print(f"[ERROR] 找不到微调模型: {CONFIG['model_path']}")
        print("[INFO] 请先运行训练脚本")
        return None
    
    print(f"[INFO] 模型路径: {CONFIG['model_path']}")
    print(f"[INFO] 使用设备: {device_type}")
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(CONFIG["model_path"])
    if model is None or tokenizer is None:
        print("[ERROR] 无法加载模型，评估终止")
        return None
    
    # 加载测试数据
    print("[INFO] 加载测试数据...")
    try:
        test_path = f"{CONFIG['data_path']}/processed/test.csv"
        if not os.path.exists(test_path):
            print(f"[ERROR] 找不到测试数据: {test_path}")
            print("[INFO] 请先运行数据预处理")
            return None
            
        test_df = pd.read_csv(test_path)
        
        # 加载标签映射
        label_path = f"{CONFIG['data_path']}/labels.json"
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            id_to_label = {v: k for k, v in label_map.items()}
        else:
            print("[WARNING] 找不到标签文件，使用默认标签映射")
            id_to_label = {0: 'positive', 1: 'negative', 2: 'neutral'}
            
        print(f"[OK] 加载 {len(test_df)} 条测试数据")
    except Exception as e:
        print(f"[ERROR] 加载测试数据失败: {e}")
        return None
    
    # 创建测试集
    test_dataset = SentimentDataset(
        test_df['text'].values,
        test_df['label_id'].values,
        tokenizer,
        max_length=CONFIG["max_length"]
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )
    
    # 评估
    print("\n[INFO] 评估模型...")
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 获取预测结果
            probabilities = torch.softmax(outputs.logits, dim=1)
            _, predictions = torch.max(outputs.logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # 显示进度
            progress = (batch_idx + 1) / len(test_loader) * 100
            print(f"[PROGRESS] {progress:5.1f}%", end='\r')
    
    print("\n" + "=" * 70)
    print("[RESULTS] 评估结果")
    print("=" * 70)
    
    # 计算指标
    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"测试准确率: {accuracy:.2%}")
        print(f"测试样本数: {len(test_df)}")
        
        # 分类报告
        print("\n[REPORT] 分类报告:")
        print("-" * 50)
        
        test_labels_named = [id_to_label.get(label, f'unknown_{label}') for label in all_labels]
        test_preds_named = [id_to_label.get(pred, f'unknown_{pred}') for pred in all_predictions]
        
        target_names = ['positive', 'negative', 'neutral']
        
        report = classification_report(
            test_labels_named,
            test_preds_named,
            target_names=target_names,
            digits=4
        )
        
        print(report)
        
        # 混淆矩阵
        print("[MATRIX] 混淆矩阵:")
        print("-" * 50)
        
        cm = confusion_matrix(test_labels_named, test_preds_named, labels=target_names)
        
        # 打印矩阵
        print("       预测: positive  negative  neutral")
        for i, row in enumerate(cm):
            print(f"真实: {target_names[i]:8} {row[0]:9} {row[1]:9} {row[2]:7}")
        
        # 错误分析
        print("\n[ANALYSIS] 错误分析:")
        print("-" * 50)
        
        errors = []
        for i, (true, pred, text, probs) in enumerate(zip(all_labels, all_predictions, test_df['text'].values, all_probabilities)):
            if true != pred:
                errors.append({
                    'index': i,
                    'text': text,
                    'true_label': id_to_label.get(true, f'unknown_{true}'),
                    'pred_label': id_to_label.get(pred, f'unknown_{pred}'),
                    'confidence': float(probs[pred]),
                    'probabilities': {
                        'positive': float(probs[0]),
                        'negative': float(probs[1]),
                        'neutral': float(probs[2])
                    }
                })
        
        print(f"总错误数: {len(errors)}/{len(test_df)}")
        print(f"错误率: {len(errors)/len(test_df):.2%}")
        
        if errors:
            print("\n错误示例:")
            for i, error in enumerate(errors[:3]):  # 显示前3个错误
                print(f"{i+1}. 文本: {error['text']}")
                print(f"   真实: {error['true_label']}")
                print(f"   预测: {error['pred_label']} (置信度: {error['confidence']:.2%})")
                print(f"   概率分布: 正面{error['probabilities']['positive']:.2%}, "
                      f"负面{error['probabilities']['negative']:.2%}, "
                      f"中性{error['probabilities']['neutral']:.2%}")
        
        # 保存结果
        results = {
            'accuracy': float(accuracy),
            'total_samples': len(test_df),
            'error_count': len(errors),
            'error_rate': len(errors)/len(test_df),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'error_examples': errors[:5]  # 保存前5个错误
        }
        
    except Exception as e:
        print(f"[WARNING] 指标计算失败: {e}")
        print(f"[INFO] 基本准确率: {sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels):.2%}")
        results = {
            'accuracy': float(sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)),
            'total_samples': len(test_df),
            'error_count': sum(np.array(all_predictions) != np.array(all_labels)),
            'error_rate': sum(np.array(all_predictions) != np.array(all_labels)) / len(test_df)
        }
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    results_path = 'results/evaluation_results.json'
    
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 评估完成! 结果保存到: {results_path}")
    except Exception as e:
        print(f"[ERROR] 保存结果失败: {e}")
    
    return model, tokenizer, results

# ==================== 交互式演示 ====================
def interactive_demo(model_path=None):
    """交互式情感分析演示"""
    
    print("\n" + "=" * 70)
    print("[DEMO] 交互式情感分析演示")
    print("=" * 70)
    
    # 如果传入了model_path，更新CONFIG
    if model_path:
        CONFIG["model_path"] = model_path
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(CONFIG["model_path"])
    if model is None or tokenizer is None:
        print("[ERROR] 无法加载模型，演示终止")
        return
    
    # 加载标签
    try:
        label_path = f"{CONFIG['data_path']}/labels.json"
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            id_to_label = {v: k for k, v in label_map.items()}
        else:
            id_to_label = {0: 'positive', 1: 'negative', 2: 'neutral'}
    except:
        id_to_label = {0: 'positive', 1: 'negative', 2: 'neutral'}
    
    # 情感中文映射
    sentiment_cn = {
        'positive': '正面',
        'negative': '负面',
        'neutral': '中性'
    }
    
    print("\n请输入电影相关评论进行分析")
    print("输入 '退出' 或 'quit' 结束")
    print("-" * 50)
    
    while True:
        user_input = input("\n请输入文本: ").strip()
        
        if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
            print("\n感谢使用，再见！")
            break
        
        if not user_input:
            print("请输入有效的文本")
            continue
        
        try:
            # 处理输入文本
            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CONFIG["max_length"]
            ).to(device)
            
            # 预测
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(outputs.logits, dim=1)
            
            sentiment_id = prediction.item()
            sentiment = id_to_label.get(sentiment_id, f'unknown_{sentiment_id}')
            confidence = probabilities[0, sentiment_id].item()
            
            # 显示结果
            print("\n[RESULT] 分析结果:")
            print(f"  评论: {user_input}")
            print(f"  情感: {sentiment_cn.get(sentiment, '未知')}")
            print(f"  置信度: {confidence:.2%}")
            
            # 显示概率分布
            print("\n  概率分布:")
            probs = probabilities[0].cpu().numpy()
            labels = ['正面', '负面', '中性']
            for i, label in enumerate(labels):
                prob = probs[i] if i < len(probs) else 0
                print(f"    {label}: {prob:.2%}")
            
            # 显示情感判断
            print("\n  判断依据:")
            if sentiment == 'positive':
                print("    ✓ 表达了积极情绪或满意")
            elif sentiment == 'negative':
                print("    ✗ 表达了消极情绪或不满意")
            else:
                print("    ○ 表达中性信息或事实描述")
                
        except Exception as e:
            print(f"[ERROR] 预测出错: {e}")
            print("[INFO] 请尝试输入其他评论")

# ==================== 主函数 ====================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CPU模型评估脚本')
    parser.add_argument('--model-path', type=str, default="./models/finetuned",
                       help='模型路径')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备类型')
    parser.add_argument('--mode', type=str, default='eval',
                       choices=['eval', 'demo'],
                       help='评估模式: eval(全面评估) 或 demo(交互式演示)')
    parser.add_argument('--trust-remote-code', action='store_true',
                       help='信任远程代码 (用于Qwen等模型)')
    parser.add_argument('--max-length', type=int, default=128,
                       help='最大文本长度')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批大小')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.max_length:
        CONFIG["max_length"] = args.max_length
    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
    
    if args.mode == 'eval':
        print(f"[INFO] 开始评估模型...")
        print(f"[INFO] 模型路径: {args.model_path}")
        print(f"[INFO] 设备: {args.device}")
        print(f"[INFO] 最大长度: {CONFIG['max_length']}")
        print(f"[INFO] 批大小: {CONFIG['batch_size']}")
        
        results = evaluate_model(args.model_path, args.device, args.trust_remote_code)
        if results:
            print("\n[INFO] 评估完成，现在可以运行交互式演示")
            print(f"命令: python scripts/evaluate_cpu.py --mode demo --model-path {args.model_path}")
    elif args.mode == 'demo':
        print(f"[INFO] 开始交互式演示...")
        print(f"[INFO] 模型路径: {args.model_path}")
        interactive_demo(args.model_path)

if __name__ == "__main__":
    main()