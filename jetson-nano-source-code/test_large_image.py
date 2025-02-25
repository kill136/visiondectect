from ultralytics import YOLO
import os
import cv2
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sliding_window(image, window_size, stride):
    """使用滑动窗口遍历图像"""
    for y in range(0, image.shape[0] - window_size[1] + 1, stride[1]):
        for x in range(0, image.shape[1] - window_size[0] + 1, stride[0]):
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]

def process_large_image(model, image_path, window_size=(256, 256), stride=(128, 128), 
                       conf_threshold=0.6):
    """处理大图像并标记检测结果"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # 创建用于显示结果的图像副本
    result_image = image.copy()
    
    # 创建热力图
    heatmap = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    
    # 记录检测结果
    detections = []
    
    # 使用滑动窗口处理图像
    for x, y, window in sliding_window(image, window_size, stride):
        # 预处理窗口
        results = model.predict(window)
        
        for result in results:
            if result.probs is None:
                continue
            
            class_id = result.probs.top1
            confidence = result.probs.top1conf.item()
            class_name = result.names[class_id]
            
            if confidence >= conf_threshold:
                if class_name in ['ink_break', 'ink_accumulation']:
                    # 记录检测结果
                    detections.append({
                        'x': x,
                        'y': y,
                        'class': class_name,
                        'conf': confidence
                    })
                    
                    # 在热力图上标记
                    color = (0, 0, 1) if class_name == 'ink_break' else (1, 0, 0)
                    cv2.rectangle(heatmap, (x, y), (x + window_size[0], y + window_size[1]), 
                                color, -1)
    
    # 将热力图叠加到原图上
    alpha = 0.3
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
    result_image = cv2.addWeighted(result_image, 1, (heatmap_normalized * 255).astype(np.uint8), 
                                  alpha, 0)
    
    # 在检测位置添加标记和文本
    for det in detections:
        color = (0, 0, 255) if det['class'] == 'ink_break' else (255, 0, 0)
        cv2.rectangle(result_image, (det['x'], det['y']), 
                     (det['x'] + window_size[0], det['y'] + window_size[1]), 
                     color, 2)
        label = f"{det['class']}: {det['conf']:.2f}"
        cv2.putText(result_image, label, (det['x'], det['y'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_image, detections

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'runs/train/ink_detection2/weights/epoch50.pt')
    test_samples_dir = os.path.join(current_dir, 'test_samples')
    output_dir = os.path.join(current_dir, 'test_results')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = YOLO(model_path)
    logging.info("Model loaded successfully")
    
    # 处理测试样本目录中的所有图片
    for filename in os.listdir(test_samples_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(test_samples_dir, filename)
        logging.info(f"Processing image: {filename}")
        
        try:
            # 处理图像
            result_image, detections = process_large_image(model, image_path)
            
            # 保存结果
            output_path = os.path.join(output_dir, f"result_{filename}")
            cv2.imwrite(output_path, result_image)
            
            # 输出检测结果统计
            ink_breaks = sum(1 for d in detections if d['class'] == 'ink_break')
            ink_accums = sum(1 for d in detections if d['class'] == 'ink_accumulation')
            logging.info(f"Found {ink_breaks} ink breaks and {ink_accums} ink accumulations")
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            continue
    
    logging.info("Processing completed. Results saved in test_results directory")

if __name__ == "__main__":
    main()
