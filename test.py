"""
测试/评估脚本 - 生成论文指标
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import yaml

from data import VideoDataset
from models import DeepFakeDetector
from utils import setup_logger, set_seed, load_checkpoint, load_config, setup_device, calculate_metrics


def test(model, dataloader, device, logger):
    """测试模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(frames)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"处理进度: [{batch_idx+1}/{len(dataloader)}]")
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='测试DeepFake检测模型')
    parser.add_argument('--config', type=str, default='configs/clip_baseline.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--output', type=str, default='test_results.txt',
                       help='结果输出文件')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    logger = setup_logger(log_dir=config['training']['log_dir'])
    logger.info(f"使用配置文件: {args.config}")
    logger.info(f"加载检查点: {args.checkpoint}")
    
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
    
    # 加载权重
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    logger.info("✅ 模型权重加载完成")
    
    # 创建测试数据加载器
    # TODO: 根据实际数据路径创建测试数据集
    logger.warning("⚠️  请根据实际数据路径修改测试数据集创建代码")
    
    # 示例：
    # test_video_paths = [...]  # 测试视频路径列表
    # test_labels = [...]  # 测试标签列表
    # 
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # 
    # test_dataset = VideoDataset(test_video_paths, test_labels, transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    test_loader = None  # 需要实际数据
    
    if test_loader is None:
        logger.error("❌ 请先配置测试数据集！")
        return
    
    # 测试
    logger.info("=" * 50)
    logger.info("开始测试")
    logger.info("=" * 50)
    
    metrics = test(model, test_loader, device, logger)
    
    # 打印结果
    logger.info("\n" + "=" * 50)
    logger.info("测试结果")
    logger.info("=" * 50)
    logger.info(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    logger.info(f"精确率 (Precision): {metrics['precision']:.4f}")
    logger.info(f"召回率 (Recall): {metrics['recall']:.4f}")
    logger.info(f"F1-score: {metrics['f1']:.4f}")
    logger.info(f"F1-score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    
    if 'confusion_matrix' in metrics:
        logger.info(f"\n混淆矩阵:")
        logger.info(f"{metrics['confusion_matrix']}")
    
    # 保存结果到文件
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("DeepFake检测模型测试结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"配置文件: {args.config}\n")
        f.write(f"检查点: {args.checkpoint}\n\n")
        f.write("指标:\n")
        f.write(f"  准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
        f.write(f"  精确率 (Precision): {metrics['precision']:.4f}\n")
        f.write(f"  召回率 (Recall): {metrics['recall']:.4f}\n")
        f.write(f"  F1-score: {metrics['f1']:.4f}\n")
        f.write(f"  F1-score (Macro): {metrics['f1_macro']:.4f}\n")
        f.write(f"  AUC: {metrics['auc']:.4f}\n")
        
        if 'confusion_matrix' in metrics:
            f.write(f"\n混淆矩阵:\n")
            for row in metrics['confusion_matrix']:
                f.write(f"  {row}\n")
    
    logger.info(f"\n✅ 结果已保存到: {args.output}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

