"""
视频切帧、音频提取的预处理脚本
"""
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional
import librosa


def extract_frames_from_video(video_path: str, num_frames: int = 8, 
                              output_dir: Optional[str] = None) -> List[Image.Image]:
    """
    从视频中提取关键帧
    
    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数
        output_dir: 可选的输出目录，如果提供则保存帧到磁盘
    
    Returns:
        List[PIL.Image]: 提取的帧图像列表
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 计算采样间隔
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames // num_frames
        frame_indices = [i * step for i in range(num_frames)]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            
            # 如果指定了输出目录，保存帧
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                frame_path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
                pil_image.save(frame_path)
    
    cap.release()
    return frames


def extract_audio_from_video(video_path: str, output_path: Optional[str] = None,
                             sample_rate: int = 16000) -> np.ndarray:
    """
    从视频中提取音频
    
    Args:
        video_path: 视频文件路径
        output_path: 可选的音频输出路径
        sample_rate: 音频采样率
    
    Returns:
        np.ndarray: 音频波形数据
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 使用librosa提取音频
    audio, sr = librosa.load(video_path, sr=sample_rate)
    
    # 如果指定了输出路径，保存音频
    if output_path:
        import soundfile as sf
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, sample_rate)
    
    return audio


# 为了兼容性，保留旧函数名
extract_frames = extract_frames_from_video
extract_audio = extract_audio_from_video

