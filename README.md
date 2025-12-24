# 🏫 电影评论情感分析系统 - 基于BERT模型
## 📚 项目简介

基于预训练语言模型的校电影评论情感分类系统，实现对电影相关评论的三分类情感分析（正面/负面/中性）。项目包含完整的机器学习流程：数据处理 → 模型微调 → 评估 → 部署。

本系统采用 **BERT模型**

## ✨ 主要特性

- **完整的工作流**: 数据准备 → 模型训练 → 评估 → 交互演示
- **优化数据处理**: 智能数据分割、数据质量检查
- **详细评估报告**: 准确率、分类报告、混淆矩阵、错误分析
- **交互式演示**: 实时文本情感分析
- **配置灵活**: 支持命令行参数和配置文件

## 📄 运行说明

### 环境配置

创建 Python 环境
~~~
conda create -n movies
~~~

激活 Python 环境
~~~
conda activate movies
~~~

安装依赖库
~~~
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r req.txt
~~~

### 代码运行

进入项目文件夹
~~~
cd 'D:\你的文件目录\电影评论情感分析{期末大作业}\'
~~~

运行完整训练流程
~~~
python .\main.py --all
~~~

交互式演示
~~~
python .\main.py --demo
~~~

交互式演示（网页版）
~~~
python .\app\app.py
~~~

## 📁 项目结构

~~~
电影评论情感分析系统/
│
├── 📂 app/                         # Web应用（可选）
│   ├── 📂 static/                 # 静态资源（CSS, JS, 图片等）
│   ├── 📂 templates/              # HTML模板
│   │   └── 📄 index.html             # 主页面
│   ├── app.py                     # Flask应用主文件
│   └── 📂 configs/                # 应用配置
│       ├── __pycache__/
│       ├── 📄 model_config.cpython-310.pyc
│       └── 📄 model_config.py        # 模型配置文件
│
├── 📂 cache/                      # 缓存目录
│   ├── 📄 huggingface/               # HuggingFace缓存
│   ├── 📄 transformers/              # Transformers缓存
│   └── 📄 version.txt               # 版本信息
│
├── 📂 datasets/                   # 数据集目录
│   ├── 📂 processed/             # 处理后的数据
│   │   ├── 📄 train.csv             # 训练集
│   │   ├── 📄 val.csv               # 验证集
│   │   ├── 📄 test.csv              # 测试集
│   │   └── 📄 dataset_stats.json    # 数据统计信息
│   ├── 📂 raw/                   # 原始数据
│   │   └── 📄 campus_corpus.txt     # 校园语料库原始文件
│   └── 📄 labels.json               # 标签定义文件
│
├── 📂 models/                     # 模型目录
│   ├── 📂 bert-base-chinese/     # BERT预训练模型（下载的）
│   ├── 📂 finetuned/             # 微调后的模型
│   └── 📂 tokenizer/             # Tokenizer文件
│
├── 📂 notebooks/                  # Jupyter笔记本
│   └── 📄 download.ipynb            # 数据下载和处理笔记本
│
├── 📂 results/                    # 训练结果
│   ├── 📄 training_results.json     # 训练过程结果
│   └── 📄 evaluation_results.json   # 模型评估结果
│
├── 📂 scripts/                    # 核心脚本
│   ├── 📂 src/                   # 源码目录（调优和工具脚本）
│   │   ├── 📄 setup_bert_model.py   # BERT模型下载脚本
│   │   ├── 📄 fix_environment.py    # 环境修复脚本
│   │   ├── 📄 fix_issues.py         # 问题修复脚本
│   │   └── 📄 fix_pytorch.py        # PyTorch相关修复
│   ├── 📄 data_preprocess.py        # 数据预处理脚本
│   ├── 📄 train_bert.py             # BERT模型训练脚本
│   └── 📄 evaluate_cpu.py           # CPU环境下的模型评估脚本
│
├── 📂 tests/                      # 测试目录
│
├── 📄 .gitignore                  # Git忽略文件
├── 📄 main.py                     # 主程序入口
├── 📄 req.txt                     # 依赖包列表
└── 📄 README.md                   # 说明文档
~~~

## 📦 文件详细说明
核心文件
main.py - 主程序入口，支持命令行参数运行完整流程

req.txt - Python依赖包列表

.gitignore - Git版本控制忽略文件配置