"""模型配置文件 - 使用轻量级BERT模型"""

MODEL_CONFIG = {
    # 模型信息 - 使用BERT
    "model_name": "bert-base-chinese",
    "model_type": "bert",
    
    # 模型路径
    "pretrained_model_path": "./models/bert-base-chinese",
    "finetuned_model_path": "./models/finetuned",
    "tokenizer_path": "./models/bert-base-chinese",
    
    # BERT配置
    "trust_remote_code": False,  # BERT不需要这个
    
    # 训练配置
    "num_labels": 3,
    "max_length": 128,  # BERT支持较短长度
    "batch_size": 8,    # BERT可以更大的batch
    "learning_rate": 2e-5,  # BERT适合这个学习率
    "num_epochs": 3,
    "gradient_accumulation_steps": 1,
    
    # 数据配置
    "data_path": "./datasets",
    "train_file": "train.csv",
    "val_file": "val.csv",
    "test_file": "test.csv",
    
    # 标签映射
    "label_map": {
        "positive": 0,
        "negative": 1,
        "neutral": 2
    },
    
    # 设备配置
    "device": "cpu",
    
    # 简化配置
    "use_local_qwen_tokenizer": False,
    "auto_load_tokenizer": True
}