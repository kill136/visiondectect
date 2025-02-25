from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO
import time
import logging
from threading import Thread, Lock
from queue import Queue
import torch

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class VideoCamera:
    def __init__(self, model_path):
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.model = YOLO(model_path)
        self.lock = Lock()
        
    def __del__(self):
        self.camera.release()

    def process_frame(self, frame, window_size=(192, 192), stride=(128, 128)):
        height, width = frame.shape[:2]
        result_image = frame.copy()
        windows = []
        positions = []
        
        for y in range(0, height - window_size[1] + 1, stride[1]):
            for x in range(0, width - window_size[0] + 1, stride[0]):
                window = frame[y:y + window_size[1], x:x + window_size[0]]
                windows.append(window)
                positions.append((x, y))
        
        return windows, positions, result_image

    def draw_results(self, frame, results, positions, window_size=(192, 192)):
        overlay = frame.copy()
        alpha = 0.3
        
        for result, (x, y) in zip(results, positions):
            if result.probs is None:
                continue
                
            class_id = result.probs.top1
            confidence = result.probs.top1conf.item()
            class_name = result.names[class_id]
            
            if confidence >= 0.6 and class_name in ['ink_break', 'ink_accumulation']:
                color = (0, 0, 255) if class_name == 'ink_break' else (255, 0, 0)
                cv2.rectangle(overlay, (x, y), 
                             (x + window_size[0], y + window_size[1]), 
                             color, -1)
                
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def get_frame(self):
        with self.lock:
            success, frame = self.camera.read()
            if not success:
                return None

            start_time = time.time()
            
            # 处理帧
            windows, positions, result_frame = self.process_frame(frame)
            
            # 批量预测
            results = self.model.predict(windows, verbose=False)
            
            # 绘制结果
            frame = self.draw_results(result_frame, results, positions)
            
            # 计算和显示FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 编码图像
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.01)  # 小延迟以减少CPU使用

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera('runs/train/ink_detection2/weights/epoch50.pt')),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
