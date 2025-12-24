#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复环境问题
"""

import os
import sys
import subprocess

def fix_pytorch_tensorflow_conflict():
    """修复PyTorch和TensorFlow冲突"""
    print("=" * 60)
    print("修复PyTorch和TensorFlow冲突")
    print("=" * 60)
    
    # 卸载TensorFlow（如果有的话）
    print("\n[1/4] 检查TensorFlow...")
    try:
        import tensorflow as tf
        print(f"✅ 找到TensorFlow版本: {tf.__version__}")
        print("[INFO] TensorFlow可能会干扰PyTorch，建议卸载或使用虚拟环境")
        
        choice = input("是否卸载TensorFlow？ (y/n): ").strip().lower()
        if choice == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "tensorflow", "-y"])
            print("✅ TensorFlow已卸载")
        else:
            print("⚠️  保留TensorFlow，但可能会影响PyTorch")
    except ImportError:
        print("✅ 未安装TensorFlow")
    
    # 重新安装PyTorch
    print("\n[2/4] 重新安装PyTorch...")
    
    torch_cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch==2.0.1+cpu",
        "torchvision==0.15.2+cpu",
        "torchaudio==2.0.2+cpu",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ]
    
    try:
        subprocess.check_call(torch_cmd)
        print("✅ PyTorch重新安装成功")
    except Exception as e:
        print(f"❌ PyTorch安装失败: {e}")
        print("[INFO] 尝试安装其他版本...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==1.13.1+cpu", "torchvision==0.14.1+cpu", "torchaudio==0.13.1+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"])
    
    # 安装transformers的特定版本
    print("\n[3/4] 安装transformers...")
    
    # 先卸载现有的
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "transformers", "-y"])
    
    # 安装支持Qwen2的版本
    transformers_cmd = [
        sys.executable, "-m", "pip", "install",
        "transformers==4.36.0",  # 支持Qwen2的版本
        "accelerate>=0.24.0",
        "safetensors>=0.4.0",
        "tokenizers>=0.15.0"
    ]
    
    try:
        subprocess.check_call(transformers_cmd)
        print("✅ transformers安装成功")
    except Exception as e:
        print(f"❌ transformers安装失败: {e}")
    
    # 验证安装
    print("\n[4/4] 验证安装...")
    try:
        import torch
        import transformers
        
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ Transformers版本: {transformers.__version__}")
        
        # 测试简单的张量运算
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        z = x + y
        print(f"✅ 张量运算测试通过: {z.tolist()}")
        
        print("\n✅ 环境修复成功!")
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False
    
    return True

def set_offline_mode():
    """设置离线模式，避免网络连接问题"""
    print("\n" + "=" * 60)
    print("设置离线模式")
    print("=" * 60)
    
    # 设置环境变量
    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_CACHE": "./cache/transformers",
        "HF_HOME": "./cache/huggingface"
    }
    
    print("设置以下环境变量:")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    # 创建缓存目录
    os.makedirs("./cache/transformers", exist_ok=True)
    os.makedirs("./cache/huggingface", exist_ok=True)
    
    print("\n✅ 离线模式已设置")
    print("⚠️  注意：这将禁用所有HuggingFace网络连接")

if __name__ == "__main__":
    print("正在修复环境问题...")
    
    # 设置离线模式
    set_offline_mode()
    
    # 修复PyTorch冲突
    if fix_pytorch_tensorflow_conflict():
        print("\n" + "=" * 60)
        print("修复完成!")
        print("=" * 60)
        print("\n现在可以尝试运行:")
        print("1. 测试环境: python -c \"import torch; print('PyTorch:', torch.__version__)\"")
        print("2. 运行训练: python scripts/train_qwen_local.py")
    else:
        print("\n❌ 修复失败，请检查错误信息")