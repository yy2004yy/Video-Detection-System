"""
定义 PyTorch Dataset 类 (读取视频/音频)
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Tuple
from pathlib import Path
from .preprocess import extract_frames_from_video, extract_audio_from_video


class VideoDataset(Dataset):
    """
    视频数据集类，用于加载视频帧和标签
    """
    def __init__(self, 
                 video_paths: List[str],
                 labels: List[int],
                 num_frames: int = 8,
                 transform=None,
                 load_audio: bool = False,
                 audio_sample_rate: int = 16000):
        """
        Args:
            video_paths: 视频文件路径列表
            labels: 标签列表 (0=真实, 1=虚假)
            num_frames: 每个视频提取的帧数
            transform: 图像变换（如resize, normalize等）
            load_audio: 是否加载音频
            audio_sample_rate: 音频采样率
        """
        assert len(video_paths) == len(labels), "视频路径和标签数量必须一致"
        
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
        self.load_audio = load_audio
        self.audio_sample_rate = audio_sample_rate
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # 提取视频帧
        frames = extract_frames_from_video(video_path, num_frames=self.num_frames)
        
        # 应用变换
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # 转换为tensor
        if isinstance(frames[0], Image.Image):
            # 如果transform没有转换为tensor，手动转换
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            frames = [to_tensor(frame) for frame in frames]
        
        # 堆叠帧: [num_frames, C, H, W]
        frames_tensor = torch.stack(frames)
        
        result = {
            'frames': frames_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'video_path': video_path
        }
        
        # 如果需要音频
        if self.load_audio:
            try:
                audio = extract_audio_from_video(video_path, sample_rate=self.audio_sample_rate)
                result['audio'] = torch.tensor(audio, dtype=torch.float32)
            except Exception as e:
                # 如果音频提取失败，使用零向量
                result['audio'] = torch.zeros(self.audio_sample_rate * 3)  # 默认3秒
        
        return result


class VideoInferenceDataset(Dataset):
    """
    用于推理的数据集类（无标签）
    """
    def __init__(self,
                 video_paths: List[str],
                 num_frames: int = 8,
                 transform=None,
                 load_audio: bool = False,
                 audio_sample_rate: int = 16000):
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.transform = transform
        self.load_audio = load_audio
        self.audio_sample_rate = audio_sample_rate
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        # 提取视频帧
        frames = extract_frames_from_video(video_path, num_frames=self.num_frames)
        
        # 应用变换
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # 转换为tensor
        if isinstance(frames[0], Image.Image):
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            frames = [to_tensor(frame) for frame in frames]
        
        frames_tensor = torch.stack(frames)
        
        result = {
            'frames': frames_tensor,
            'video_path': video_path
        }
        
        if self.load_audio:
            try:
                audio = extract_audio_from_video(video_path, sample_rate=self.audio_sample_rate)
                result['audio'] = torch.tensor(audio, dtype=torch.float32)
            except Exception as e:
                result['audio'] = torch.zeros(self.audio_sample_rate * 3)
        
        return result

