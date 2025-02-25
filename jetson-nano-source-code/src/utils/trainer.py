import torch
from pathlib import Path
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # 设置输出目录
        self.output_dir = Path(config['output']['model_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置TensorBoard
        if config['output']['tensorboard']:
            self.writer = SummaryWriter(
                log_dir=str(self.output_dir / 'tensorboard' / time.strftime('%Y%m%d-%H%M%S'))
            )
        
        self.best_val_acc = 0
        self.patience_counter = 0
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
        
        return total_loss/len(self.train_loader), 100.*correct/total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss/len(self.val_loader), 100.*correct/total
    
    def train(self):
        print(f"开始训练，使用设备: {self.device}")
        
        for epoch in range(self.config['training']['epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录到TensorBoard
            if self.config['output']['tensorboard']:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], epoch)
            
            print(f'\nEpoch {epoch+1}/{self.config["training"]["epochs"]}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, str(self.output_dir / 'best_model.pth'))
                print(f'保存最佳模型，验证准确率: {val_acc:.2f}%')
            else:
                self.patience_counter += 1
            
            # 保存检查点
            if (epoch + 1) % self.config['output']['save_interval'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, str(self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'))
            
            # 提前停止
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f'\n验证准确率在{self.config["training"]["early_stopping_patience"]}个epoch内没有提升，停止训练')
                break
        
        if self.config['output']['tensorboard']:
            self.writer.close()
        
        print(f'\n训练完成！最佳验证准确率: {self.best_val_acc:.2f}%')
        print(f'模型和日志保存在: {self.output_dir}')
