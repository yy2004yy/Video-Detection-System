#!/bin/bash

# CLIP模型部署脚本
# 用于创建conda环境并安装依赖

set -e

echo "=========================================="
echo "🚀 开始部署 CLIP 模型环境"
echo "=========================================="

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 项目根目录: $PROJECT_ROOT"

# 1. 创建conda环境
echo ""
echo "📦 步骤 1: 创建 conda 环境 'clip_env'..."
if conda env list | grep -q "^clip_env "; then
    echo "⚠️  环境 'clip_env' 已存在，是否删除并重新创建? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🗑️  删除现有环境..."
        conda env remove -n clip_env -y
    else
        echo "✅ 使用现有环境"
        exit 0
    fi
fi

echo "🔨 正在创建环境..."
conda env create -f "$SCRIPT_DIR/environment_clip.yml"

# 2. 激活环境并安装额外依赖
echo ""
echo "📦 步骤 2: 安装额外依赖..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clip_env

# 安装requirements中的包（如果conda环境没有完全安装）
pip install -r "$SCRIPT_DIR/requirements_clip.txt" --quiet

# 3. 创建模型缓存目录
echo ""
echo "📁 步骤 3: 创建模型缓存目录..."
mkdir -p "$PROJECT_ROOT/models/clip"
echo "✅ 模型缓存目录: $PROJECT_ROOT/models/clip"

# 4. 验证安装
echo ""
echo "🔍 步骤 4: 验证安装..."
python -c "import torch; import transformers; import cv2; print('✅ 所有依赖安装成功')" || {
    echo "❌ 依赖验证失败"
    exit 1
}

echo ""
echo "=========================================="
echo "🎉 CLIP 环境部署完成！"
echo "=========================================="
echo ""
echo "📝 使用说明:"
echo "  1. 激活环境: conda activate clip_env"
echo "  2. 运行程序: cd $SCRIPT_DIR && python deploy_clip.py"
echo "  3. 在交互界面中输入视频路径和问题"
echo ""
echo "💡 示例输入:"
echo "  /path/to/video.mp4|这个视频是真实的还是伪造的?"
echo ""

