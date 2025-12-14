"""
设置随机种子、保存模型等工具函数
"""
import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


def set_seed(seed: int = 42):
    """
    设置随机种子，确保实验可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(state: Dict[str, Any],
                   checkpoint_dir: str,
                   filename: str = "checkpoint.pth",
                   is_best: bool = False):
    """
    保存模型检查点
    
    Args:
        state: 要保存的状态字典（包含model_state_dict, optimizer_state_dict等）
        checkpoint_dir: 检查点保存目录
        filename: 文件名
        is_best: 是否为最佳模型
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存常规检查点
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    # 如果是最佳模型，额外保存
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_filepath)
        print(f"✅ 保存最佳模型到: {best_filepath}")


def load_checkpoint(checkpoint_path: str,
                   model: Optional[torch.nn.Module] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[str] = None) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 要加载权重的模型（可选）
        optimizer: 要加载状态的优化器（可选）
        device: 设备（可选）
    
    Returns:
        Dict[str, Any]: 加载的状态字典
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 模型权重已加载")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ 优化器状态已加载")
    
    return checkpoint


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Dict[str, Any]: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(gpu_id: int = 0) -> str:
    """
    设置计算设备
    
    Args:
        gpu_id: GPU ID
    
    Returns:
        str: 设备字符串（如 "cuda:0" 或 "cpu"）
    """
    print("-" * 50)
    if torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
            gpu_name = torch.cuda.get_device_name(gpu_id)
            total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            print(f"✅ 成功调用显卡 {gpu_id}: {gpu_name}")
            print(f"🚀 显存总量: {total_mem:.2f} GB")
        else:
            device = "cuda:0"
            print(f"⚠️  指定的GPU {gpu_id}不存在，使用GPU 0")
    else:
        device = "cpu"
        print("❌ 警告：未检测到显卡，正在使用 CPU (速度会很慢)")
    print("-" * 50)
    return device

