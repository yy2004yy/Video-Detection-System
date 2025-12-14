"""
训练脚本 - 一键启动训练
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from data import VideoDataset
from models import DeepFakeDetector
from utils import setup_logger, set_seed, save_checkpoint, load_checkpoint, load_config, setup_device, Accuracy, F1Score, AUC


def train_epoch(model, dataloader, criterion, optimizer, device, logger):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    accuracy = Accuracy()
    f1_score = F1Score()
    
    for batch_idx, batch in enumerate(dataloader):
        frames = batch['frames'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        accuracy.update(preds, labels)
        f1_score.update(preds, labels)
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy.compute()
    epoch_f1 = f1_score.compute()
    
    return epoch_loss, epoch_acc, epoch_f1


def validate(model, dataloader, criterion, device, logger):
    """验证"""
    model.eval()
    running_loss = 0.0
    accuracy = Accuracy()
    f1_score = F1Score()
    auc = AUC()
    
    with torch.no_grad():
        for batch in dataloader:
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(frames)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)
            
            accuracy.update(preds, labels)
            f1_score.update(preds, labels)
            auc.update(probs, labels)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy.compute()
    epoch_f1 = f1_score.compute()
    epoch_auc = auc.compute()
    
    return epoch_loss, epoch_acc, epoch_f1, epoch_auc


def main():
    parser = argparse.ArgumentParser(description='训练DeepFake检测模型')
    parser.add_argument('--config', type=str, default='configs/clip_baseline.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    logger = setup_logger(log_dir=config['training']['log_dir'])
    logger.info(f"使用配置文件: {args.config}")
    
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = setup_device(config['device']['gpu_id'])
    
    # 设置HuggingFace镜像（如果需要）
    if config.get('hf_mirror', {}).get('enabled', False):
        os.environ["HF_ENDPOINT"] = config['hf_mirror']['endpoint']
        logger.info(f"使用HuggingFace镜像: {config['hf_mirror']['endpoint']}")
    
    # 创建模型
    model = DeepFakeDetector(
        backbone=config['model']['backbone'],
        backbone_model_name=config['model'].get('clip_model_name') or config['model'].get('imagebind_model_name'),
        num_classes=2,
        device=device
    )
    logger.info(f"模型创建完成: {config['model']['name']}")
    
    # 创建数据加载器（这里需要根据实际数据路径调整）
    # TODO: 根据实际数据路径创建数据集
    logger.warning("⚠️  请根据实际数据路径修改数据集创建代码")
    
    # 示例：假设有视频路径和标签列表
    # video_paths = [...]  # 视频路径列表
    # labels = [...]  # 标签列表 (0=真实, 1=虚假)
    # 
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # 
    # train_dataset = VideoDataset(video_paths, labels, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # 为了演示，这里创建一个虚拟的数据加载器
    # 实际使用时请替换为真实数据
    logger.info("创建虚拟数据加载器（请替换为真实数据）")
    train_loader = None  # 需要实际数据
    val_loader = None    # 需要实际数据
    
    if train_loader is None:
        logger.error("❌ 请先配置数据集！")
        return
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 恢复训练（如果指定）
    start_epoch = 0
    best_auc = 0.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_auc = checkpoint.get('best_auc', 0.0)
        logger.info(f"从epoch {start_epoch}恢复训练")
    
    # 训练循环
    num_epochs = config['training']['num_epochs']
    save_dir = config['training']['save_dir']
    
    logger.info("=" * 50)
    logger.info("开始训练")
    logger.info("=" * 50)
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # 训练
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        
        # 验证
        if val_loader:
            val_loss, val_acc, val_f1, val_auc = validate(
                model, val_loader, criterion, device, logger
            )
            
            logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
            
            # 保存检查点
            is_best = val_auc > best_auc
            if is_best:
                best_auc = val_auc
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_auc': val_auc,
                'config': config
            }
            
            save_checkpoint(
                checkpoint,
                save_dir,
                filename=f"checkpoint_epoch_{epoch+1}.pth",
                is_best=is_best
            )
        else:
            logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    
    logger.info("=" * 50)
    logger.info("训练完成！")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

