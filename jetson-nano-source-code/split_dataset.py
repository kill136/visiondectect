import os
import shutil
import random
from pathlib import Path

def split_dataset(src_root, dst_root, train_ratio=0.8):
    """Split dataset into train and val sets according to YOLO classification format."""
    # 确保目标目录存在
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    
    # 创建train和val目录
    train_dir = dst_root / 'train'
    val_dir = dst_root / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # 类别映射
    class_mapping = {
        'ink_bleeding': 'ink_accumulation',  # 将ink_bleeding映射到ink_accumulation
        'ink_break': 'ink_break',
        'normal': 'normal'
    }
    
    # 处理每个类别
    src_root = Path(src_root)
    for src_cls, dst_cls in class_mapping.items():
        src_path = src_root / 'train' / src_cls
        if not src_path.exists():
            print(f"Warning: Source path {src_path} does not exist")
            continue
            
        # 创建目标类别目录
        train_cls_dir = train_dir / dst_cls
        val_cls_dir = val_dir / dst_cls
        train_cls_dir.mkdir(exist_ok=True)
        val_cls_dir.mkdir(exist_ok=True)
        
        # 获取所有图片
        images = list(src_path.glob('*.png'))
        random.shuffle(images)
        
        # 计算分割点
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # 复制训练集图片
        for img in train_images:
            dst_path = train_cls_dir / img.name
            shutil.copy2(img, dst_path)
            print(f"Copied {img.name} to train/{dst_cls}")
            
        # 复制验证集图片
        for img in val_images:
            dst_path = val_cls_dir / img.name
            shutil.copy2(img, dst_path)
            print(f"Copied {img.name} to val/{dst_cls}")

if __name__ == "__main__":
    src_root = "balanced_dataset"
    dst_root = "datasets/ink_classification"
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 清空目标目录
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    
    # 分割数据集
    split_dataset(src_root, dst_root)
    
    # 创建数据集配置文件
    yaml_content = '''path: /home/wbj/Documents/AI/Smart-Ai-Pothole-Detecto/jetson-nano-source-code/datasets/ink_classification  # 数据集根目录
train: train  # 训练集目录
val: val    # 验证集目录

# 类别名称
names:
  0: ink_break
  1: ink_accumulation
  2: normal
'''
    
    with open(os.path.join('datasets', 'ink_classification.yaml'), 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print("\nDataset split completed!")
    print("You can now start training with the new dataset.")
