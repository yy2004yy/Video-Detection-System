"""
日志记录工具
"""
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "DeepFake_Project",
                 log_dir: Optional[str] = None,
                 level: int = logging.INFO,
                 log_to_file: bool = True) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件保存目录
        level: 日志级别
        log_to_file: 是否保存到文件
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志文件名包含时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件保存到: {log_file}")
    
    return logger


def get_logger(name: str = "DeepFake_Project") -> logging.Logger:
    """
    获取已存在的日志记录器，如果不存在则创建
    
    Args:
        name: 日志记录器名称
    
    Returns:
        logging.Logger: 日志记录器
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name)
    return logger

