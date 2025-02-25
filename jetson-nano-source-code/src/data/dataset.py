import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import random

class InkDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', patch_size=256):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.patch_size = patch_size
        
        # 获取所有大图路径
        self.image_paths = []
        self.labels = []
        
        for class_name in ['normal', 'ink_bleeding', 'ink_break']:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            class_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.image_paths.extend(class_images)
            self.labels.extend([class_name] * len(class_images))
            
        self.class_to_idx = {'normal': 0, 'ink_bleeding': 1, 'ink_break': 2}
        
    def __len__(self):
        return len(self.image_paths)
        
    def random_crop(self, image):
        """从大图中随机裁剪一个patch_size大小的区域"""
        if image.size[0] < self.patch_size or image.size[1] < self.patch_size:
            # 如果图像小于目标大小，进行填充
            new_size = (max(image.size[0], self.patch_size), 
                       max(image.size[1], self.patch_size))
            new_image = Image.new('RGB', new_size)
            new_image.paste(image, ((new_size[0]-image.size[0])//2,
                                  (new_size[1]-image.size[1])//2))
            image = new_image
            
        # 随机裁剪
        left = random.randint(0, image.size[0] - self.patch_size)
        top = random.randint(0, image.size[1] - self.patch_size)
        return image.crop((left, top, left + self.patch_size, top + self.patch_size))
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.mode == 'train':
            # 训练模式：随机裁剪 + 数据增强
            image = self.random_crop(image)
            if self.transform:
                image = self.transform(image)
        else:
            # 验证模式：使用滑动窗口进行预测
            if self.transform:
                image = self.transform(image)
                
        return image, self.class_to_idx[label]

def create_data_loaders(data_dir, batch_size=32, num_workers=4):
    """创建训练和验证数据加载器"""
    train_transform = transforms.Compose([
        # 随机水平和垂直翻转
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # 随机旋转
        transforms.RandomRotation(10),
        # 随机调整亮度、对比度和饱和度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # 随机调整图像大小
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = InkDefectDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = InkDefectDataset(
        os.path.join(data_dir, 'val'),
        transform=val_transform,
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
