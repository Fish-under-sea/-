#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复PyTorch DLL加载问题
"""

import os
import sys
import subprocess
import platform

def fix_pytorch_issue():
    """修复PyTorch DLL问题"""
    print("=" * 70)
    print("修复PyTorch DLL加载问题")
    print("=" * 70)
    
    # 检查系统环境
    print(f"操作系统: {platform.system()} {platform.version()}")
    print(f"Python版本: {sys.version}")
    
    # 卸载有问题的PyTorch
    print("\n[1/4] 卸载有问题的PyTorch...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
        print("✅ PyTorch卸载成功")
    except:
        print("⚠️  PyTorch卸载失败，继续...")
    
    # 清理缓存
    print("\n[2/4] 清理pip缓存...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "cache", "purge"])
        print("✅ 缓存清理成功")
    except:
        print("⚠️  缓存清理失败，继续...")
    
    # 安装正确的PyTorch版本
    print("\n[3/4] 安装正确的PyTorch版本...")
    
    # 检查Python版本
    python_version = sys.version_info
    
    if python_version >= (3, 10):
        print("[INFO] Python 3.10+ 检测到")
        
        # 对于Windows，安装CPU-only版本
        torch_cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch==2.0.1", 
            "torchvision==0.15.2", 
            "torchaudio==2.0.2",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
    else:
        print("[INFO] Python 3.9或更早版本")
        torch_cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch==1.13.1+cpu", 
            "torchvision==0.14.1+cpu", 
            "torchaudio==0.13.1+cpu",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
    
    try:
        subprocess.check_call(torch_cmd)
        print("✅ PyTorch安装成功")
    except Exception as e:
        print(f"❌ PyTorch安装失败: {e}")
        print("[INFO] 尝试使用conda安装...")
        
        # 如果pip安装失败，尝试conda
        try:
            subprocess.check_call(["conda", "install", "pytorch", "torchvision", "torchaudio", "cpuonly", "-c", "pytorch", "-y"])
            print("✅ PyTorch (conda) 安装成功")
        except:
            print("❌ Conda安装也失败")
            print("[INFO] 尝试安装更简单的版本...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==1.13.1"])
    
    # 验证安装
    print("\n[4/4] 验证PyTorch安装...")
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        print(f"✅ 设备数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        
        # 测试简单操作
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        z = x + y
        print(f"✅ 张量运算测试: {z.tolist()}")
        
        print("\n✅ PyTorch修复成功!")
        
    except Exception as e:
        print(f"❌ PyTorch验证失败: {e}")
        return False
    
    return True

def install_minimal_requirements():
    """安装最小化依赖"""
    print("\n" + "=" * 70)
    print("安装最小化依赖")
    print("=" * 70)
    
    # 基础依赖（不需要torch的）
    base_packages = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
    ]
    
    for package in base_packages:
        print(f"安装 {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  ✅ {package} 安装成功")
        except Exception as e:
            print(f"  ❌ {package} 安装失败: {e}")
    
    print("\n✅ 最小化依赖安装完成")

if __name__ == "__main__":
    print("正在修复PyTorch DLL问题...")
    
    # 先安装基础包
    install_minimal_requirements()
    
    # 修复PyTorch
    if fix_pytorch_issue():
        print("\n" + "=" * 70)
        print("修复完成!")
        print("=" * 70)
        print("\n现在可以尝试运行:")
        print("1. 测试PyTorch: python -c \"import torch; print('PyTorch版本:', torch.__version__)\"")
        print("2. 运行数据预处理: python scripts/data_preprocess.py")
        print("3. 运行完整流程: python main.py --all")
    else:
        print("\n❌ 修复失败，请尝试手动安装PyTorch")