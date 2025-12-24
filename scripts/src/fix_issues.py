# fix_issues.py
import os
import shutil

def check_bert_model():
    """检查BERT模型完整性"""
    pretrained_path = "./models/bert-base-chinese"
    
    print("检查BERT模型文件:")
    if os.path.exists(pretrained_path):
        files = os.listdir(pretrained_path)
        print(f"找到 {len(files)} 个文件:")
        for file in files[:10]:
            print(f"  - {file}")
    else:
        print("  ⚠️  BERT模型目录不存在")
        print("  请下载BERT模型: bert-base-chinese")
        print("  或运行: python setup_bert_model.py")

if __name__ == "__main__":
    print("检查模型文件...")
    check_bert_model()