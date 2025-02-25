import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image

class InkDefectPredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.classes = ['normal', 'ink_bleeding', 'ink_break']
        self.model = self.load_model(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path):
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, len(self.classes))
        )
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            # 如果是完整的检查点文件
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果只有模型状态
            model.load_state_dict(checkpoint)
            
        model = model.to(self.device)
        return model
        
    def predict_patch(self, image_tensor):
        """预测单个图像块"""
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0).to(self.device))
            probabilities = torch.softmax(output, dim=1)
            return probabilities.cpu().numpy()[0]
            
    def sliding_window(self, image, window_size=256, stride=128):
        """使用滑动窗口生成图像块"""
        width, height = image.size
        patches = []
        positions = []
        
        # 如果图像小于窗口大小，直接调整图像大小
        if width < window_size or height < window_size:
            image = image.resize((max(width, window_size), max(height, window_size)), Image.BILINEAR)
            width, height = image.size
        
        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                # 确保至少有一个patch
                if x + window_size > width:
                    x = width - window_size
                if y + window_size > height:
                    y = height - window_size
                    
                patch = image.crop((x, y, x + window_size, y + window_size))
                patch_tensor = self.transform(patch)
                patches.append(patch_tensor)
                positions.append((x, y))
                
        # 如果没有生成任何patch，至少生成一个
        if not patches:
            patch = image.resize((window_size, window_size), Image.BILINEAR)
            patch_tensor = self.transform(patch)
            patches.append(patch_tensor)
            positions.append((0, 0))
            
        return torch.stack(patches), positions
        
    def aggregate_predictions(self, patch_predictions, positions, image_size, window_size=256):
        """聚合所有patch的预测结果"""
        width, height = image_size
        # 创建热力图
        heatmaps = np.zeros((len(self.classes), height, width))
        counts = np.zeros((height, width))
        
        for pred, (x, y) in zip(patch_predictions, positions):
            for c in range(len(self.classes)):
                heatmaps[c, y:y+window_size, x:x+window_size] += pred[c]
            counts[y:y+window_size, x:x+window_size] += 1
            
        # 避免除零
        counts = np.maximum(counts, 1)
        
        # 计算平均值
        for c in range(len(self.classes)):
            heatmaps[c] /= counts
            
        return heatmaps
        
    def predict(self, image_path, threshold=0.5):
        """预测大图中的缺陷"""
        image = Image.open(image_path).convert('RGB')
        patches, positions = self.sliding_window(image)
        
        # 批量预测所有patch
        patch_predictions = []
        for i in range(0, len(patches), 32):  # 批量大小为32
            batch = patches[i:i+32].to(self.device)
            with torch.no_grad():
                output = self.model(batch)
                probs = torch.softmax(output, dim=1)
                patch_predictions.extend(probs.cpu().numpy())
                
        # 聚合预测结果
        heatmaps = self.aggregate_predictions(patch_predictions, positions, image.size)
        
        # 生成结果
        results = {
            'class_predictions': {},
            'heatmaps': heatmaps,
            'detection_counts': {},
            'average_confidences': {}
        }
        
        # 对每个类别计算统计信息
        for i, class_name in enumerate(self.classes):
            # 计算高于阈值的区域
            detected_regions = (heatmaps[i] > threshold).sum()
            avg_confidence = float(heatmaps[i][heatmaps[i] > threshold].mean()) if detected_regions > 0 else 0.0
            
            results['class_predictions'][class_name] = {
                'confidence': float(np.mean(heatmaps[i])),
                'detected': detected_regions > 0
            }
            results['detection_counts'][class_name] = int(detected_regions)
            results['average_confidences'][class_name] = float(avg_confidence)
            
        return results

def create_model(num_classes=3, pretrained=True):
    """
    创建一个基于ResNet18的模型
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    Returns:
        model: 创建的模型
    """
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    return model
