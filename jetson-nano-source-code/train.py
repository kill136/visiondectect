from ultralytics import YOLO
import torch
import os
import logging
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_model_files():
    """清理可能损坏的模型文件"""
    files_to_remove = ['yolov8n-cls.pt', 'yolov8n.pt']
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                logger.info(f"Removed existing model file: {file}")
            except Exception as e:
                logger.warning(f"Failed to remove {file}: {e}")

def main():
    # 清理可能损坏的模型文件
    clean_model_files()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确保数据集目录存在
    dataset_dir = os.path.join(current_dir, 'balanced_dataset')
    
    if not os.path.exists(dataset_dir):
        raise RuntimeError(f"Dataset directory not found: {dataset_dir}")
    
    logger.info("Loading YOLOv8 model...")
    try:
        # 使用更大的模型以提高性能
        model = YOLO('yolov8x-cls.pt')  # 使用最大的YOLOv8x分类模型
        training_args = {
            'data': dataset_dir,  # 使用目录路径而不是YAML文件
            'val': True,  # 仅启用验证模式
            'epochs': 100,
            'imgsz': 256,              # 使用实际数据集的尺寸
            'batch': 16,               # 因为模型变大，适当减小batch size
            'device': device,
            'project': 'runs/train',
            'name': 'ink_detection2',
            'save': True,
            'save_period': 10,
            'patience': 20,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.0005,            # 降低学习率以适应更大的模型
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,        # 增加预热轮数
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'augment': True,
            'dropout': 0.2,            # 增加dropout以防止过拟合
            'seed': 42,
            'workers': 4,
            'cos_lr': True,
            'close_mosaic': 10,
            'exist_ok': True,
            'translate': 0.1,
            'scale': 0.2,
            'mosaic': 0.3,
            'mixup': 0.1,
            'amp': True               # 使用自动混合精度训练
        }
        
        logger.info("Starting training with YOLOv8x model...")
        results = model.train(**training_args)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
