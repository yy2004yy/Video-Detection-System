"""
定义最终的检测模型（把骨干网和分类头拼在一起）
"""
import torch
import torch.nn as nn
from typing import Optional
from .backbones.clip_encoder import CLIPEncoder
from .backbones.imagebind_encoder import ImageBindEncoder


class DeepFakeDetector(nn.Module):
    """
    DeepFake检测器，结合特征提取器和分类头
    """
    def __init__(self,
                 backbone: str = "clip",
                 backbone_model_name: Optional[str] = None,
                 num_classes: int = 2,
                 dropout: float = 0.5,
                 device: Optional[str] = None):
        """
        Args:
            backbone: 骨干网络类型 ("clip" 或 "imagebind")
            backbone_model_name: 骨干网络模型名称
            num_classes: 分类数量（通常是2：真实/虚假）
            dropout: Dropout比率
            device: 计算设备
        """
        super().__init__()
        
        self.backbone_type = backbone
        
        # 初始化骨干网络
        if backbone == "clip":
            model_name = backbone_model_name or "openai/clip-vit-base-patch32"
            self.backbone = CLIPEncoder(model_name=model_name, device=device)
            feature_dim = self.backbone.get_feature_dim()
        elif backbone == "imagebind":
            model_name = backbone_model_name or "facebook/imagebind-base"
            self.backbone = ImageBindEncoder(model_name=model_name, device=device)
            feature_dim = self.backbone.get_feature_dim()
        else:
            raise ValueError(f"不支持的骨干网络类型: {backbone}")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 4, num_classes)
        )
        
        # 移动到设备
        if device:
            self.to(device)
    
    def forward(self, frames, audio=None):
        """
        前向传播
        
        Args:
            frames: 视频帧 [batch_size, num_frames, C, H, W]
            audio: 音频（可选，仅用于imagebind）
        
        Returns:
            torch.Tensor: 分类logits [batch_size, num_classes]
        """
        # 提取特征
        if self.backbone_type == "clip":
            features = self.backbone(frames)
        elif self.backbone_type == "imagebind":
            features = self.backbone(images=frames, audio=audio)
        else:
            raise ValueError(f"不支持的骨干网络类型: {self.backbone_type}")
        
        # 分类
        logits = self.classifier(features)
        
        return logits
    
    def predict(self, frames, audio=None):
        """
        预测（返回概率）
        
        Args:
            frames: 视频帧
            audio: 音频（可选）
        
        Returns:
            torch.Tensor: 预测概率 [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(frames, audio)
            probs = torch.softmax(logits, dim=1)
        return probs
    
    def predict_class(self, frames, audio=None):
        """
        预测类别
        
        Args:
            frames: 视频帧
            audio: 音频（可选）
        
        Returns:
            torch.Tensor: 预测类别 [batch_size]
        """
        probs = self.predict(frames, audio)
        return probs.argmax(dim=1)

