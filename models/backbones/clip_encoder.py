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
            # 假设输入是 [batch_size, num_frames, C, H, W]
            batch_size, num_frames = images.shape[:2]
            images = images.view(batch_size * num_frames, *images.shape[2:])
            
            # 预处理图像（需要转换为PIL格式或使用vision transformer）
            # 这里简化处理，假设已经预处理过
            # 实际使用时可能需要更复杂的处理
            
            # 使用vision encoder
            with torch.no_grad():
                pixel_values = images
                if pixel_values.device != next(self.parameters()).device:
                    pixel_values = pixel_values.to(next(self.parameters()).device)
                
                outputs = self.model.vision_model(pixel_values=pixel_values)
                image_features = outputs.pooler_output
                image_features = self.model.visual_projection(image_features)
            
            # 重新reshape: [batch_size, num_frames, feature_dim]
            image_features = image_features.view(batch_size, num_frames, -1)
            
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

