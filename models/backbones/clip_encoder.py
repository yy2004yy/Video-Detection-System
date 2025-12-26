"""
封装 CLIP 模型作为特征提取器
"""
import os
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
from typing import Optional, Tuple


class CLIPEncoder(nn.Module):
    """
    CLIP编码器，用于提取视频帧的特征
    """
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Args:
            model_name: CLIP模型名称
            cache_dir: 模型缓存目录
            device: 计算设备
        """
        super().__init__()
        
        # 设置模型缓存目录
        if cache_dir is None:
            PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
            cache_dir = str(PROJECT_ROOT / "models" / "clip")
        
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir
        
        # 加载CLIP模型
        self.model = CLIPModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_safetensors=True
        )
        
        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # 获取特征维度
        self.feature_dim = self.model.config.projection_dim  # 通常是512
        
        # 移动到指定设备
        if device:
            self.to(device)
        
        # 设置为评估模式
        self.eval()
    
    def forward(self, images):
        """
        提取图像特征
        
        Args:
            images: 输入图像，可以是PIL Image列表或tensor
        
        Returns:
            torch.Tensor: 图像特征 [batch_size, feature_dim]
        """
        # 如果输入是PIL Image列表，需要预处理
        if isinstance(images, list) and isinstance(images[0], type(images[0])):
            # 检查是否是PIL Image
            try:
                from PIL import Image
                if isinstance(images[0], Image.Image):
                    # 使用processor预处理
                    inputs = self.processor(images=images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model.get_image_features(**inputs)
                    return outputs
            except:
                pass
        
        # 如果输入已经是tensor，直接使用
        if isinstance(images, torch.Tensor):
            device = next(self.parameters()).device
            
            # 处理不同的tensor形状
            if images.dim() == 4:  # [batch_size, C, H, W]
                batch_size = images.shape[0]
                num_frames = 1
                images = images.unsqueeze(1)  # [batch_size, 1, C, H, W]
            elif images.dim() == 5:  # [batch_size, num_frames, C, H, W]
                batch_size, num_frames = images.shape[:2]
            else:
                raise ValueError(f"不支持的tensor维度: {images.dim()}, 期望4或5维")
            
            # 展平为 [batch_size * num_frames, C, H, W]
            images_flat = images.view(batch_size * num_frames, *images.shape[2:])
            
            # 确保输入在正确的设备上
            if images_flat.device != device:
                images_flat = images_flat.to(device)
            
            # CLIP的vision_model期望normalized的pixel values
            # 如果输入是[0,1]范围的tensor，需要转换为CLIP期望的格式
            # CLIP内部使用processor时会做normalization，但直接调用vision_model时
            # 需要确保输入格式正确（通常是[0,1]范围，然后内部会normalize）
            # 这里假设输入已经是[0,1]范围的tensor（经过ToTensor）
            
            # 使用vision encoder提取特征
            with torch.no_grad():
                vision_outputs = self.model.vision_model(pixel_values=images_flat)
                # 获取pooled特征
                image_features = vision_outputs.pooler_output  # [batch_size * num_frames, hidden_size]
                # 通过visual projection
                image_features = self.model.visual_projection(image_features)  # [batch_size * num_frames, feature_dim]
            
            # 重新reshape: [batch_size, num_frames, feature_dim]
            image_features = image_features.view(batch_size, num_frames, self.feature_dim)
            
            # 对帧特征进行平均池化
            image_features = image_features.mean(dim=1)  # [batch_size, feature_dim]
            
            return image_features
        
        raise ValueError(f"不支持的输入类型: {type(images)}")
    
    def encode_images(self, images):
        """
        编码图像（别名方法，与forward相同）
        """
        return self.forward(images)
    
    def get_feature_dim(self):
        """
        获取特征维度
        """
        return self.feature_dim

