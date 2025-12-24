from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os
import sys
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

app = Flask(__name__, 
           template_folder=os.path.join(current_dir, 'templates'),
           static_folder=os.path.join(current_dir, 'static'))

# 配置
CONFIG = {
    "model_path": "./models/finetuned",
    "tokenizer_path": "./models/finetuned",
    "data_path": "./datasets",
    "max_length": 128
}

# 初始化模型和tokenizer
model = None
tokenizer = None
id_to_label = None

def load_model():
    """加载模型和tokenizer"""
    global model, tokenizer, id_to_label
    
    print("[INFO] 正在加载模型...")
    
    try:
        # 检查模型路径
        if not os.path.exists(CONFIG["model_path"]):
            print(f"[ERROR] 模型路径不存在: {CONFIG['model_path']}")
            print("[INFO] 请先运行训练脚本: python main.py --train-bert")
            return False
        
        # 加载tokenizer
        tokenizer = BertTokenizer.from_pretrained(CONFIG["tokenizer_path"])
        print("[OK] Tokenizer加载成功")
        
        # 加载模型
        model = BertForSequenceClassification.from_pretrained(CONFIG["model_path"])
        model.eval()  # 设置为评估模式
        print("[OK] 模型加载成功")
        
        # 加载标签映射
        labels_path = os.path.join(CONFIG["data_path"], "labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            id_to_label = {v: k for k, v in label_map.items()}
            print("[OK] 标签映射加载成功")
        else:
            # 使用默认标签映射
            id_to_label = {0: 'positive', 1: 'negative', 2: 'neutral'}
            print("[WARNING] 使用默认标签映射")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 加载模型失败: {e}")
        return False

def predict_sentiment(text):
    """预测评论情感"""
    if model is None or tokenizer is None:
        return {"error": "模型未加载", "status": "error"}
    
    try:
        # 预处理文本
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONFIG["max_length"]
        )
        
        # 预测
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(outputs.logits, dim=1)
        
        sentiment_id = prediction.item()
        sentiment = id_to_label.get(sentiment_id, f'unknown_{sentiment_id}')
        confidence = probabilities[0, sentiment_id].item()
        
        # 获取所有类别的概率
        probs = probabilities[0].cpu().numpy()
        
        # 情感中文映射
        sentiment_cn = {
            'positive': '正面',
            'negative': '负面',
            'neutral': '中性',
            'unknown_0': '正面',
            'unknown_1': '负面',
            'unknown_2': '中性'
        }
        
        # 判断依据
        reasoning = ""
        if sentiment == 'positive':
            reasoning = "表达了积极情绪或满意"
        elif sentiment == 'negative':
            reasoning = "表达了消极情绪或不满意"
        else:
            reasoning = "表达中性信息或事实描述"
        
        return {
            "text": text,
            "sentiment": sentiment,
            "sentiment_chinese": sentiment_cn.get(sentiment, "未知"),
            "confidence": confidence,
            "probabilities": {
                "正面": float(probs[0]) if len(probs) > 0 else 0,
                "负面": float(probs[1]) if len(probs) > 1 else 0,
                "中性": float(probs[2]) if len(probs) > 2 else 0
            },
            "reasoning": reasoning,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return {"error": f"预测失败: {str(e)}", "status": "error"}

@app.route('/')
def home():
    """主页"""
    return render_template('index.html', 
                         title="电影评论情感分析系统",
                         subtitle="分析电影评论的情感倾向")

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "请输入文本", "status": "error"})
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "文本不能为空", "status": "error"})
        
        # 预测情感
        result = predict_sentiment(text)
        
        if "error" in result:
            return jsonify(result)
        
        return jsonify({
            "status": "success",
            "data": result
        })
        
    except Exception as e:
        return jsonify({
            "error": f"服务器错误: {str(e)}",
            "status": "error"
        })

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测"""
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({"error": "请输入文本列表", "status": "error"})
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({"error": "texts必须是一个列表", "status": "error"})
        
        if len(texts) > 100:  # 限制批量大小
            return jsonify({"error": "一次最多处理100条文本", "status": "error"})
        
        results = []
        for text in texts:
            result = predict_sentiment(text.strip())
            results.append(result)
        
        return jsonify({
            "status": "success",
            "count": len(results),
            "results": results
        })
        
    except Exception as e:
        return jsonify({
            "error": f"批量预测失败: {str(e)}",
            "status": "error"
        })

@app.route('/stats')
def stats():
    """系统统计"""
    stats_data = {
        "model_loaded": model is not None,
        "model_path": CONFIG["model_path"],
        "tokenizer_path": CONFIG["tokenizer_path"],
        "max_length": CONFIG["max_length"],
        "labels": id_to_label,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 尝试加载训练结果
    try:
        results_path = os.path.join(parent_dir, "results", "training_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                training_results = json.load(f)
                stats_data["training_results"] = training_results
    except:
        pass
    
    return jsonify({
        "status": "success",
        "stats": stats_data
    })

# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "资源不存在", "status": "error"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "服务器内部错误", "status": "error"}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("校园情感分析系统 - Web应用")
    print("=" * 60)
    
    # 加载模型
    if load_model():
        print("[INFO] 模型加载成功，启动Web应用...")
        print(f"[INFO] 访问地址: http://localhost:8000")
        print(f"[INFO] 健康检查: http://localhost:8000/health")
        print("[INFO] 按 Ctrl+C 停止应用")
        print("-" * 60)
        
        app.run(debug=True, host='0.0.0.0', port=8000)
    else:
        print("[ERROR] 模型加载失败，请检查配置")
        sys.exit(1)