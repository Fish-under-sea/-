#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校园情感分析系统 - 主程序
支持BERT/轻量级模型训练
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
import json
import time

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def load_model_config():
    """加载模型配置"""
    config_paths = [
        os.path.join(current_dir, "app/configs/model_config.py"),
        os.path.join(current_dir, "app/configs/model_config.yaml"),
        os.path.join(current_dir, "configs/model_config.py"),
        os.path.join(current_dir, "configs/model_config.yaml"),
    ]
    
    config = None
    
    # 尝试加载Python配置文件
    for config_path in config_paths:
        if os.path.exists(config_path):
            if config_path.endswith('.py'):
                try:
                    # 动态导入Python配置文件
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("model_config", config_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    config = module.MODEL_CONFIG
                    print(f"📋 加载配置文件: {config_path}")
                    break
                except Exception as e:
                    print(f"⚠️  无法加载Python配置文件 {config_path}: {e}")
            elif config_path.endswith('.yaml'):
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    print(f"📋 加载YAML配置文件: {config_path}")
                    break
                except Exception as e:
                    print(f"⚠️  无法加载YAML配置文件 {config_path}: {e}")
    
    # 如果没有找到配置文件，使用默认配置
    if config is None:
        print("⚠️  未找到配置文件，使用默认配置")
        config = {
            "model_name": "bert-base-chinese",
            "model_type": "bert",
            "pretrained_model_path": "./models/bert-base-chinese",
            "finetuned_model_path": "./models/finetuned",
            "tokenizer_path": "./models/bert-base-chinese",
            "device": "cpu",
            "num_labels": 3,
            "max_length": 128,
            "batch_size": 8,
            "learning_rate": 2e-5,
            "num_epochs": 3,
            "trust_remote_code": False,
        }
    
    return config

def check_model_files(model_path):
    """检查模型文件是否存在"""
    print(f"[INFO] 检查模型文件: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    # 检查必要文件
    required_files = ['config.json', 'vocab.txt', 'pytorch_model.bin']
    optional_files = ['tokenizer_config.json', 'special_tokens_map.json']
    
    found_files = []
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            found_files.append(file)
        else:
            missing_files.append(file)
    
    # 列出所有文件
    all_files = os.listdir(model_path)
    print(f"[INFO] 找到 {len(all_files)} 个文件:")
    for file in all_files[:10]:  # 只显示前10个文件
        file_path = os.path.join(model_path, file)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            size_str = f"({file_size/1024:.1f} KB)" if file_size < 1024*1024 else f"({file_size/(1024*1024):.1f} MB)"
            print(f"  - {file} {size_str}")
    
    if len(all_files) > 10:
        print(f"  ... 还有 {len(all_files) - 10} 个文件")
    
    if missing_files:
        print(f"⚠️  缺少必要文件: {', '.join(missing_files)}")
        
        # 对于BERT模型，可以自动下载
        if "bert" in model_path.lower():
            print("[INFO] BERT模型文件不完整，可能需要下载")
            print("[INFO] 运行: python setup_bert_model.py")
        
        return False
    
    print("✅ 模型文件检查通过")
    return True

def display_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    """显示进度条"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '░' * (length - filled_length)
    print(f'\r{prefix} [{bar}] {percent}% {suffix}', end='')
    if iteration == total:
        print()

def run_subprocess_with_progress(cmd, process_name="进程"):
    """运行子进程并显示进度"""
    import subprocess
    
    print(f"🚀 启动{process_name}...")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时显示输出
        last_line = ""
        for line in process.stdout:
            line = line.strip()
            if line:
                # 保存最后一行用于进度显示
                last_line = line
                
                # 显示重要信息
                if any(keyword in line for keyword in ["[INFO]", "[OK]", "[ERROR]", "[RESULT]", "[SAVE]", "准确率:", "测试样本数:", "损失:"]):
                    print(f"  {line}")
                elif "[EPOCH]" in line or "Epoch" in line:
                    print(f"\n  {line}")
                elif "[PROGRESS]" in line or "批次" in line:
                    # 进度信息，更新同一行
                    print(f"\r  {line}", end='')
                    sys.stdout.flush()
        
        # 等待进程结束
        process.wait()
        
        elapsed_time = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        
        if process.returncode == 0:
            print(f"\n✅ {process_name}完成 (耗时: {elapsed_str})")
            return True
        else:
            print(f"\n❌ {process_name}失败，返回码: {process.returncode} (耗时: {elapsed_str})")
            return False
            
    except Exception as e:
        print(f"\n❌ 运行{process_name}出错: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='校园情感分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --prepare-data       # 准备数据
  %(prog)s --train              # 训练模型
  %(prog)s --evaluate           # 评估模型
  %(prog)s --demo               # 交互式演示
  %(prog)s --all                # 运行完整流程
  %(prog)s --list-config        # 显示当前配置
  %(prog)s --fix-env            # 修复环境问题
  %(prog)s --setup-model        # 设置本地模型

训练选项:
  --train-bert                  # 使用BERT模型训练
  --train-simple               # 使用简单模型训练
        """
    )
    
    # 主功能参数
    parser.add_argument('--prepare-data', action='store_true',
                       help='准备数据集')
    parser.add_argument('--train', action='store_true',
                       help='训练模型（自动选择最佳方式）')
    parser.add_argument('--train-bert', action='store_true',
                       help='使用BERT模型训练')
    parser.add_argument('--train-simple', action='store_true',
                       help='使用简单模型训练')
    parser.add_argument('--evaluate', action='store_true',
                       help='评估模型')
    parser.add_argument('--demo', action='store_true',
                       help='交互式演示')
    parser.add_argument('--all', action='store_true',
                       help='运行完整流程（准备数据 -> 训练 -> 评估）')
    
    # 工具参数
    parser.add_argument('--list-config', action='store_true',
                       help='显示当前配置并退出')
    parser.add_argument('--fix-env', action='store_true',
                       help='修复环境问题')
    parser.add_argument('--setup-model', action='store_true',
                       help='设置本地模型')
    
    # 可选覆盖配置的参数
    parser.add_argument('--pretrained-model', type=str,
                       help='覆盖配置中的预训练模型路径')
    parser.add_argument('--finetuned-model', type=str,
                       help='覆盖配置中的微调模型路径')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       help='覆盖设备配置 (cpu 或 cuda)')
    parser.add_argument('--batch-size', type=int,
                       help='覆盖批大小配置')
    parser.add_argument('--num-epochs', type=int,
                       help='覆盖训练轮数')
    parser.add_argument('--max-length', type=int,
                       help='覆盖最大文本长度')
    parser.add_argument('--model-type', type=str, choices=['bert', 'qwen', 'simple'],
                       help='选择模型类型')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_model_config()
    
    # 如果有命令行参数，覆盖配置
    if args.pretrained_model:
        config['pretrained_model_path'] = args.pretrained_model
    if args.finetuned_model:
        config['finetuned_model_path'] = args.finetuned_model
    if args.device:
        config['device'] = args.device
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.max_length:
        config['max_length'] = args.max_length
    if args.model_type:
        config['model_type'] = args.model_type
    
    # 显示配置
    if args.list_config:
        print("=" * 70)
        print("📋 当前配置:")
        print("=" * 70)
        for key, value in config.items():
            print(f"  {key}: {value}")
        return 0
    
    # 修复环境
    if args.fix_env:
        print("=" * 70)
        print("🔧 修复环境问题")
        print("=" * 70)
        try:
            import subprocess
            result = subprocess.run([sys.executable, "fix_environment.py"], 
                                  capture_output=False, text=True)
            return 0
        except:
            print("❌ 修复环境失败，请手动运行: python fix_environment.py")
            return 1
    
    # 设置本地模型
    if args.setup_model:
        print("=" * 70)
        print("🔧 设置本地模型")
        print("=" * 70)
        try:
            import subprocess
            result = subprocess.run([sys.executable, "setup_local_model.py"], 
                                  capture_output=False, text=True)
            return 0
        except:
            print("❌ 设置本地模型失败，请手动运行: python setup_local_model.py")
            return 1
    
    # 如果没有指定任何操作参数，显示帮助
    if not any([args.prepare_data, args.train, args.train_bert, args.train_simple, 
                args.evaluate, args.demo, args.all]):
        parser.print_help()
        return
    
    print("=" * 70)
    print("🏫 电影评论情感分析系统")
    print("=" * 70)
    print(f"🤖 模型类型: {config.get('model_type', 'bert')}")
    print(f"📁 预训练模型: {config['pretrained_model_path']}")
    print(f"📁 微调模型: {config['finetuned_model_path']}")
    print(f"⚙️  设备: {config['device']}")
    print(f"📊 批大小: {config.get('batch_size', 8)}")
    print(f"📏 文本长度: {config.get('max_length', 128)}")
    print(f"🔄 训练轮数: {config.get('num_epochs', 3)}")
    print("=" * 70)
    
    # 将配置保存为环境变量，供子进程使用
    os.environ['MODEL_CONFIG'] = json.dumps(config)
    
    try:
        # 准备数据
        if args.prepare_data or args.all:
            print("\n📊 准备数据集...")
            print("=" * 50)
            
            success = run_subprocess_with_progress(
                [sys.executable, "scripts/data_preprocess.py"],
                "数据预处理"
            )
            
            if success:
                print("✅ 数据准备完成")
            else:
                print("❌ 数据准备失败")
                if not args.all:
                    return 1
        
        # 确定训练脚本
        train_script = None
        if args.train_bert:
            train_script = "scripts/train_bert.py"
            print("\n🤖 训练模型 (BERT模式)...")
        elif args.train_simple:
            train_script = "scripts/train_simple.py"
            print("\n🤖 训练模型 (简单模式)...")
        elif args.train or args.all:
            # 自动选择最佳训练脚本
            if os.path.exists("scripts/train_bert.py"):
                train_script = "scripts/train_bert.py"
                print("\n🤖 训练模型 (自动选择: BERT模式)...")
            elif os.path.exists("scripts/train_simple.py"):
                train_script = "scripts/train_simple.py"
                print("\n🤖 训练模型 (自动选择: 简单模式)...")
            else:
                train_script = "scripts/train_cpu.py"
                print("\n🤖 训练模型 (自动选择: 标准模式)...")
        
        # 执行训练
        if train_script:
            print("=" * 50)
            print(f"📁 预训练模型路径: {config['pretrained_model_path']}")
            print(f"📁 微调模型保存路径: {config['finetuned_model_path']}")
            print(f"⚙️  使用设备: {config['device']}")
            print(f"📊 批大小: {config.get('batch_size', 8)}")
            print(f"📏 文本长度: {config.get('max_length', 128)}")
            print(f"🔄 训练轮数: {config.get('num_epochs', 3)}")
            print("-" * 50)
            
            # 检查模型文件
            if "bert" in train_script.lower():
                if not check_model_files(config['pretrained_model_path']):
                    print("❌ 模型文件检查失败")
                    print("⚠️  请先下载BERT模型:")
                    print("   1. 运行: python setup_bert_model.py")
                    print("   2. 或手动下载: bert-base-chinese")
                    
                    if not args.all:
                        return 1
                    print("⚠️  尝试继续训练...")
            
            # 检查训练脚本是否存在
            if not os.path.exists(train_script):
                print(f"❌ 训练脚本不存在: {train_script}")
                if not args.all:
                    return 1
                print("⚠️  跳过训练步骤...")
            else:
                success = run_subprocess_with_progress(
                    [sys.executable, train_script],
                    "模型训练"
                )
                
                if success:
                    print("✅ 模型训练完成")
                    
                    # 检查模型是否保存成功
                    if os.path.exists(config['finetuned_model_path']):
                        saved_files = os.listdir(config['finetuned_model_path'])
                        print(f"📁 模型已保存到: {config['finetuned_model_path']}")
                        print(f"📋 保存了 {len(saved_files)} 个文件")
                        if saved_files:
                            print("  主要文件:")
                            for file in saved_files[:5]:
                                if file.endswith(('.json', '.bin', '.txt', '.model')):
                                    print(f"    - {file}")
                else:
                    print("❌ 模型训练失败")
                    if not args.all:
                        return 1
        
        # 评估模型
        if args.evaluate or args.all:
            print("\n🧪 评估模型...")
            print("=" * 50)
            print(f"📁 微调模型路径: {config['finetuned_model_path']}")
            
            # 检查微调模型是否存在
            if not os.path.exists(config['finetuned_model_path']):
                print(f"⚠️  微调模型路径不存在: {config['finetuned_model_path']}")
                print("请先运行训练步骤")
                if not args.all:
                    return 1
                print("⚠️  跳过评估步骤...")
            else:
                # 选择合适的评估脚本
                eval_scripts = [
                    "scripts/evaluate_cpu.py",
                    "scripts/evaluate_simple.py"
                ]
                
                eval_script = None
                for script in eval_scripts:
                    if os.path.exists(script):
                        eval_script = script
                        break
                
                if eval_script is None:
                    print("❌ 找不到可用的评估脚本")
                    if not args.all:
                        return 1
                else:
                    print(f"📝 使用评估脚本: {eval_script}")
                    
                    # 检测模型类型，传递相应参数
                    config_path = os.path.join(config['finetuned_model_path'], "config.json")
                    trust_remote_code = False
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                model_config = json.load(f)
                                if "qwen" in model_config.get("model_type", "").lower():
                                    trust_remote_code = True
                        except:
                            pass
                    
                    cmd = [
                        sys.executable,
                        eval_script,
                        "--model-path", config['finetuned_model_path'],
                        "--device", config['device'],
                        "--mode", "eval"
                    ]
                    
                    if trust_remote_code:
                        cmd.extend(["--trust-remote-code", "true"])
                    
                    success = run_subprocess_with_progress(cmd, "模型评估")
                    
                    if success:
                        print("✅ 模型评估完成")
                        if os.path.exists("results/evaluation_results.json"):
                            print("📊 评估结果保存在: results/evaluation_results.json")
                        elif os.path.exists("results/simple_evaluation_results.json"):
                            print("📊 评估结果保存在: results/simple_evaluation_results.json")
                    else:
                        print("❌ 模型评估失败")
                        if not args.all:
                            return 1
        
        # 交互式演示
        if args.demo:
            print("\n🎮 交互式演示...")
            print("=" * 50)
            print(f"📁 使用微调模型: {config['finetuned_model_path']}")
            
            if not os.path.exists(config['finetuned_model_path']):
                print(f"⚠️  微调模型路径不存在: {config['finetuned_model_path']}")
                print("请先运行训练步骤")
                return 1
            
            # 选择合适的评估脚本
            eval_scripts = [
                "scripts/evaluate_cpu.py",
                "scripts/evaluate_simple.py"
            ]
            
            eval_script = None
            for script in eval_scripts:
                if os.path.exists(script):
                    eval_script = script
                    break
            
            if eval_script is None:
                print("❌ 找不到可用的评估脚本")
                return 1
            
            print(f"📝 使用评估脚本: {eval_script}")
            print("-" * 50)
            print("输入电影相关评论进行分析")
            print("输入 '退出' 或 'quit' 结束")
            print("=" * 50)
            
            # 检测模型类型，传递相应参数
            config_path = os.path.join(config['finetuned_model_path'], "config.json")
            trust_remote_code = False
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        model_config = json.load(f)
                        if "qwen" in model_config.get("model_type", "").lower():
                            trust_remote_code = True
                except:
                    pass
            
            cmd = [
                sys.executable,
                eval_script,
                "--model-path", config['finetuned_model_path'],
                "--device", config['device'],
                "--mode", "demo"
            ]
            
            if trust_remote_code:
                cmd.extend(["--trust-remote-code", "true"])
            
            try:
                import subprocess
                process = subprocess.run(cmd, capture_output=False, text=True)
                
                if process.returncode != 0:
                    print(f"\n❌ 演示失败，返回码: {process.returncode}")
            except Exception as e:
                print(f"❌ 演示过程出错: {e}")
        
        # 显示完成信息
        if args.all:
            print("\n" + "=" * 70)
            print("🎉 完整流程执行完毕!")
            print("=" * 70)
            
            # 检查各个步骤的结果
            steps = []
            
            if os.path.exists(f"{config['data_path']}/processed/train.csv"):
                try:
                    train_df = pd.read_csv(f"{config['data_path']}/processed/train.csv")
                    steps.append(f"📊 数据准备: 完成 ({len(train_df)} 条训练数据)")
                except:
                    steps.append("📊 数据准备: 完成")
            else:
                steps.append("📊 数据准备: 未完成")
            
            if os.path.exists(config['finetuned_model_path']):
                saved_files = os.listdir(config['finetuned_model_path'])
                steps.append(f"🤖 模型训练: 完成 ({len(saved_files)} 个文件)")
            else:
                steps.append("🤖 模型训练: 未完成")
            
            if os.path.exists("results/evaluation_results.json"):
                steps.append("🧪 模型评估: 完成")
            elif os.path.exists("results/simple_evaluation_results.json"):
                steps.append("🧪 模型评估: 完成 (简单模式)")
            else:
                steps.append("🧪 模型评估: 未完成")
            
            for step in steps:
                print(step)
            
            if os.path.exists(config['finetuned_model_path']):
                print(f"📁 微调模型: {config['finetuned_model_path']}")
            
            if os.path.exists("results/training_results.json"):
                print(f"📁 训练结果: results/training_results.json")
            
            if os.path.exists("results/evaluation_results.json"):
                print(f"📁 评估结果: results/evaluation_results.json")
            elif os.path.exists("results/simple_evaluation_results.json"):
                print(f"📁 评估结果: results/simple_evaluation_results.json")
            
            print("=" * 70)
            print("\n🎯 下一步:")
            if os.path.exists(config['finetuned_model_path']):
                print(f"1. 交互式演示: python main.py --demo")
            else:
                print(f"1. 训练模型: python main.py --train-bert")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        return 1
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())