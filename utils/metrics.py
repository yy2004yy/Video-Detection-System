"""
计算准确率、AUC、F1-score等指标
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from typing import Dict, List, Optional, Tuple


def calculate_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    计算各种评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 预测概率（用于计算AUC）
    
    Returns:
        Dict[str, float]: 包含各种指标的字典
    """
    metrics = {}
    
    # 准确率
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # F1-score
    metrics['f1'] = f1_score(y_true, y_pred, average='binary')
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    
    # 精确率和召回率
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    
    # AUC
    if y_proba is not None:
        try:
            # 如果是多分类，需要处理
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                # 使用正类的概率
                y_proba_binary = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba[:, -1]
            else:
                y_proba_binary = y_proba.flatten()
            
            metrics['auc'] = roc_auc_score(y_true, y_proba_binary)
        except Exception as e:
            metrics['auc'] = 0.0
            print(f"⚠️  AUC计算失败: {e}")
    else:
        metrics['auc'] = 0.0
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # 从混淆矩阵计算其他指标
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positive'] = int(tp)
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
    
    return metrics


class Accuracy:
    """准确率计算类"""
    def __init__(self):
        self.correct = 0
        self.total = 0
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        更新统计
        
        Args:
            preds: 预测标签 [batch_size]
            targets: 真实标签 [batch_size]
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        self.correct += np.sum(preds == targets)
        self.total += len(targets)
    
    def compute(self) -> float:
        """计算准确率"""
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def reset(self):
        """重置统计"""
        self.correct = 0
        self.total = 0


class F1Score:
    """F1-score计算类"""
    def __init__(self, average: str = 'binary'):
        self.average = average
        self.all_preds = []
        self.all_targets = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        更新统计
        
        Args:
            preds: 预测标签 [batch_size]
            targets: 真实标签 [batch_size]
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        self.all_preds.extend(preds)
        self.all_targets.extend(targets)
    
    def compute(self) -> float:
        """计算F1-score"""
        if len(self.all_preds) == 0:
            return 0.0
        return f1_score(self.all_targets, self.all_preds, average=self.average, zero_division=0)
    
    def reset(self):
        """重置统计"""
        self.all_preds = []
        self.all_targets = []


class AUC:
    """AUC计算类"""
    def __init__(self):
        self.all_probs = []
        self.all_targets = []
    
    def update(self, probs: torch.Tensor, targets: torch.Tensor):
        """
        更新统计
        
        Args:
            probs: 预测概率 [batch_size, num_classes] 或 [batch_size]
            targets: 真实标签 [batch_size]
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # 如果是多类概率，取正类概率
        if probs.ndim > 1 and probs.shape[1] > 1:
            probs = probs[:, 1] if probs.shape[1] == 2 else probs[:, -1]
        
        self.all_probs.extend(probs)
        self.all_targets.extend(targets)
    
    def compute(self) -> float:
        """计算AUC"""
        if len(self.all_probs) == 0:
            return 0.0
        try:
            return roc_auc_score(self.all_targets, self.all_probs)
        except Exception as e:
            print(f"⚠️  AUC计算失败: {e}")
            return 0.0
    
    def reset(self):
        """重置统计"""
        self.all_probs = []
        self.all_targets = []

