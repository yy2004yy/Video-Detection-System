"""
批量特征提取脚本 - 场景一：离线批量处理
用于从大量视频中提取CLIP特征并保存为.npy或.pkl文件

使用方法:
    python extract_features.py --video_dir ./data/videos --output_dir ./features --batch_size 64
"""
import os
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from data import VideoInferenceDataset
from models.backbones.clip_encoder import CLIPEncoder
from utils import setup_logger, setup_device, load_config


def extract_features_batch(model, dataloader, device, logger, save_format='npy'):
    """
    批量提取特征
    
    Args:
        model: CLIP编码器模型
        dataloader: 数据加载器
        device: 计算设备
        logger: 日志记录器
        save_format: 保存格式 ('npy' 或 'pkl')
    
    Returns:
        dict: 包含特征和路径的字典
    """
    model.eval()
    all_features = []
    all_paths = []
    
    logger.info(f"开始批量提取特征，批次大小: {dataloader.batch_size}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="提取特征")):
            frames = batch['frames'].to(device)  # [batch_size, num_frames, C, H, W]
            video_paths = batch['video_path']
            
            # 提取特征
            features = model(frames)  # [batch_size, feature_dim]
            
            # 转换为numpy并保存
            features_np = features.cpu().numpy()
            
            all_features.append(features_np)
            all_paths.extend(video_paths)
            
            # 每100个batch打印一次进度
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"已处理 {batch_idx + 1} 个批次，共 {len(all_paths)} 个视频")
    
    # 合并所有特征
    all_features = np.vstack(all_features)  # [total_videos, feature_dim]
    
    logger.info(f"✅ 特征提取完成！共提取 {len(all_paths)} 个视频的特征")
    logger.info(f"特征形状: {all_features.shape}")
    
    return {
        'features': all_features,
        'video_paths': all_paths
    }


def save_features(features_dict, output_dir, save_format='npy', logger=None):
    """
    保存特征到文件
    
    Args:
        features_dict: 包含特征和路径的字典
        output_dir: 输出目录
        save_format: 保存格式 ('npy' 或 'pkl')
        logger: 日志记录器
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if save_format == 'npy':
        # 保存为.npy格式
        features_path = os.path.join(output_dir, 'features.npy')
        paths_path = os.path.join(output_dir, 'video_paths.txt')
        
        np.save(features_path, features_dict['features'])
        with open(paths_path, 'w', encoding='utf-8') as f:
            for path in features_dict['video_paths']:
                f.write(f"{path}\n")
        
        if logger:
            logger.info(f"✅ 特征已保存到: {features_path}")
            logger.info(f"✅ 路径列表已保存到: {paths_path}")
    
    elif save_format == 'pkl':
        # 保存为.pkl格式
        output_path = os.path.join(output_dir, 'features.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(features_dict, f)
        
        if logger:
            logger.info(f"✅ 特征已保存到: {output_path}")
    
    else:
        raise ValueError(f"不支持的保存格式: {save_format}")


def collect_video_paths(video_dir, extensions=None):
    """
    收集视频文件路径
    
    Args:
        video_dir: 视频目录
        extensions: 视频文件扩展名列表
    
    Returns:
        list: 视频文件路径列表
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
    video_paths = []
    video_dir = Path(video_dir)
    
    for ext in extensions:
        video_paths.extend(video_dir.rglob(f'*{ext}'))
        video_paths.extend(video_dir.rglob(f'*{ext.upper()}'))
    
    video_paths = [str(p) for p in video_paths]
    video_paths.sort()
    
    return video_paths


def main():
    parser = argparse.ArgumentParser(description='批量提取CLIP特征')
    parser.add_argument('--config', type=str, default='configs/clip_baseline.yaml',
                       help='配置文件路径')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='视频目录路径')
    parser.add_argument('--output_dir', type=str, default='./features',
                       help='特征输出目录')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小（根据显存调整，A6000可以设置64-128）')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='每个视频提取的帧数')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载进程数')
    parser.add_argument('--save_format', type=str, default='npy', choices=['npy', 'pkl'],
                       help='保存格式: npy 或 pkl')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='GPU ID（如果为None则使用配置文件中的设置）')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    logger = setup_logger(log_dir=config['training']['log_dir'])
    logger.info("=" * 60)
    logger.info("批量特征提取 - 场景一：离线批量处理")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"视频目录: {args.video_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"批次大小: {args.batch_size}")
    
    # 设置设备
    gpu_id = args.gpu_id if args.gpu_id is not None else config['device']['gpu_id']
    device = setup_device(gpu_id)
    logger.info(f"使用设备: {device}")
    
    # 设置HuggingFace镜像（如果需要）
    if config.get('hf_mirror', {}).get('enabled', False):
        os.environ["HF_ENDPOINT"] = config['hf_mirror']['endpoint']
        logger.info(f"使用HuggingFace镜像: {config['hf_mirror']['endpoint']}")
    
    # 收集视频文件
    logger.info("正在收集视频文件...")
    video_paths = collect_video_paths(args.video_dir)
    logger.info(f"找到 {len(video_paths)} 个视频文件")
    
    if len(video_paths) == 0:
        logger.error("❌ 未找到任何视频文件！请检查视频目录路径。")
        return
    
    # 创建CLIP编码器
    logger.info("正在加载CLIP模型...")
    clip_model_name = config['model'].get('clip_model_name', 'openai/clip-vit-base-patch32')
    encoder = CLIPEncoder(
        model_name=clip_model_name,
        device=device
    )
    logger.info(f"✅ CLIP模型加载完成: {clip_model_name}")
    logger.info(f"特征维度: {encoder.get_feature_dim()}")
    
    # 创建数据加载器
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = VideoInferenceDataset(
        video_paths=video_paths,
        num_frames=args.num_frames,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.startswith('cuda') else False
    )
    
    # 批量提取特征
    logger.info("=" * 60)
    logger.info("开始批量提取特征...")
    logger.info("=" * 60)
    
    features_dict = extract_features_batch(
        model=encoder,
        dataloader=dataloader,
        device=device,
        logger=logger,
        save_format=args.save_format
    )
    
    # 保存特征
    logger.info("=" * 60)
    logger.info("正在保存特征...")
    logger.info("=" * 60)
    
    save_features(
        features_dict=features_dict,
        output_dir=args.output_dir,
        save_format=args.save_format,
        logger=logger
    )
    
    logger.info("=" * 60)
    logger.info("✅ 批量特征提取完成！")
    logger.info("=" * 60)
    logger.info(f"特征文件位置: {args.output_dir}")
    logger.info(f"特征形状: {features_dict['features'].shape}")
    logger.info(f"视频数量: {len(features_dict['video_paths'])}")


if __name__ == "__main__":
    main()

