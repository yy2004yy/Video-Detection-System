"""
单个视频检测脚本 - 给导师演示用
"""
import os
import argparse
import torch
from pathlib import Path
from PIL import Image
import yaml

from data.preprocess import extract_frames_from_video
from models import DeepFakeDetector
from utils import setup_logger, load_config, setup_device, load_checkpoint
from torchvision import transforms


def inference_single_video(model, video_path, device, num_frames=8, logger=None):
    """
    对单个视频进行推理
    
    Args:
        model: 训练好的模型
        video_path: 视频文件路径
        device: 计算设备
        num_frames: 提取的帧数
        logger: 日志记录器
    
    Returns:
        dict: 包含预测结果和置信度的字典
    """
    if logger:
        logger.info(f"📹 正在处理视频: {video_path}")
    
    # 提取视频帧
    frames = extract_frames_from_video(video_path, num_frames=num_frames)
    
    if logger:
        logger.info(f"✅ 提取了 {len(frames)} 帧")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 预处理所有帧
    processed_frames = [transform(frame) for frame in frames]
    
    # 堆叠为tensor: [num_frames, C, H, W]
    frames_tensor = torch.stack(processed_frames)
    
    # 添加batch维度: [1, num_frames, C, H, W]
    frames_tensor = frames_tensor.unsqueeze(0).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        logits = model(frames_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()
    
    # 结果
    class_names = ["真实", "虚假"]
    result = {
        'video_path': video_path,
        'prediction': class_names[pred_class],
        'pred_class': pred_class,
        'confidence': confidence,
        'probabilities': {
            '真实': probs[0][0].item(),
            '虚假': probs[0][1].item()
        }
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='单个视频检测推理')
    parser.add_argument('--config', type=str, default='configs/clip_baseline.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--video', type=str, required=True,
                       help='要检测的视频路径')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='提取的帧数')
    args = parser.parse_args()
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video):
        print(f"❌ 错误: 视频文件不存在: {args.video}")
        return
    
    # 加载配置
    config = load_config(args.config)
    logger = setup_logger(log_dir=config['training']['log_dir'])
    logger.info(f"使用配置文件: {args.config}")
    logger.info(f"加载检查点: {args.checkpoint}")
    logger.info(f"检测视频: {args.video}")
    
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
    
    # 推理
    logger.info("=" * 50)
    logger.info("开始推理")
    logger.info("=" * 50)
    
    result = inference_single_video(
        model, args.video, device, 
        num_frames=args.num_frames, 
        logger=logger
    )
    
    # 打印结果
    print("\n" + "=" * 50)
    print("检测结果")
    print("=" * 50)
    print(f"视频路径: {result['video_path']}")
    print(f"预测结果: {result['prediction']}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"\n概率分布:")
    print(f"  真实: {result['probabilities']['真实']:.2%}")
    print(f"  虚假: {result['probabilities']['虚假']:.2%}")
    print("=" * 50)
    
    logger.info("推理完成")


if __name__ == "__main__":
    main()

