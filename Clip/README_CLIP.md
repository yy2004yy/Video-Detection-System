# CLIP 模型部署说明

## 📋 概述

本项目实现了基于 CLIP (Contrastive Language-Image Pre-training) 模型的视频问答系统，支持输入视频和文字问题，输出文字回答。

## 🚀 快速开始

### 1. 创建 Conda 环境

使用提供的脚本自动部署：

```bash
cd /data1/yangye/DeepFake_Project/Clip
chmod +x deploy.sh
./deploy.sh
```

或者手动创建：

```bash
# 创建环境
conda env create -f environment_clip.yml

# 激活环境
conda activate clip_env

# 安装依赖（如果需要）
pip install -r requirements_clip.txt
```

### 2. 运行程序

```bash
# 确保已激活环境
conda activate clip_env

# 运行程序
cd /data1/yangye/DeepFake_Project/Clip
python deploy_clip.py
```

### 3. 使用交互式问答

程序启动后，你可以：

1. **首次输入**：提供视频路径和问题
   ```
   /path/to/video.mp4|这个视频是真实的还是伪造的?
   ```

2. **后续输入**：直接输入问题（会使用上次的视频）
   ```
   这个人的表情是什么?
   ```

3. **退出**：输入 `quit` 或 `exit`

## 📁 目录结构

```
Clip/
├── deploy_clip.py          # 主程序文件
├── environment_clip.yml     # Conda环境配置
├── requirements_clip.txt    # Python依赖列表
├── deploy.sh               # 自动部署脚本
└── README_CLIP.md          # 本说明文档
```

## 🔧 配置说明

### GPU 设置

默认使用 GPU 3 号卡。如需修改，编辑 `deploy_clip.py` 中的：

```python
device = setup_device(gpu_id=3)  # 修改这里的数字
```

### 模型缓存

所有模型权重会自动下载到：
```
DeepFake_Project/models/clip/
```

### 视频处理

- 默认从视频中提取 8 帧进行分析
- 支持常见视频格式（mp4, avi, mov 等）
- 自动处理视频帧采样

## 💡 功能特性

1. **视频输入支持**：自动从视频中提取关键帧
2. **多语言问答**：支持中英文问题
3. **智能答案匹配**：基于 CLIP 的视觉-语言理解能力
4. **交互式对话**：支持连续问答，无需重复加载视频

## 📊 支持的问答类型

- **真实性检测**：判断视频是否为 DeepFake
- **表情识别**：识别视频中人物的情绪
- **说话检测**：判断是否在说话，口型是否同步
- **通用问答**：基于视频内容的任意问题

## ⚠️ 注意事项

1. **首次运行**：需要从 HuggingFace 下载模型（约 500MB），请确保网络连接正常
2. **显存要求**：建议至少 4GB 显存
3. **视频格式**：支持 OpenCV 能读取的所有视频格式
4. **CLIP 限制**：CLIP 不是生成式模型，答案基于预定义的候选答案进行匹配

## 🔍 故障排除

### 问题：模型下载失败

**解决方案**：
- 检查网络连接
- 程序已配置国内镜像源（hf-mirror.com）
- 如果仍有问题，可以手动下载模型到 `models/clip/` 目录

### 问题：CUDA 错误

**解决方案**：
- 检查 GPU 驱动和 CUDA 版本
- 确认指定的 GPU 编号存在
- 尝试使用 CPU 模式（修改代码中的 device）

### 问题：视频无法读取

**解决方案**：
- 检查视频文件路径是否正确
- 确认视频文件格式是否支持
- 检查文件权限

## 📝 开发说明

### 自定义候选答案

在 `answer_question_with_clip` 函数中，可以自定义 `candidate_answers` 参数来提供更精确的答案选项。

### 扩展功能

- 可以修改 `extract_frames_from_video` 函数来改变帧提取策略
- 可以修改 `answer_question_with_clip` 函数来支持多帧融合分析

## 📞 联系支持

如有问题，请检查：
1. Conda 环境是否正确激活
2. 所有依赖是否安装完整
3. GPU 是否可用
4. 视频文件是否有效

