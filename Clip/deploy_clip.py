import os

# 设置 Hugging Face 国内镜像源 (这行代码必须放在 import transformers 之前)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import sys
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path



# 设置模型缓存目录到项目文件夹
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "clip"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)

# ============================
# 1. 硬件检测与配置
# ============================
def setup_device(gpu_id=3):
    """设置计算设备"""
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

# ============================
# 2. 视频处理函数
# ============================
def extract_frames_from_video(video_path, num_frames=8):
    """
    从视频中提取关键帧
    
    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数
    
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
    
    cap.release()
    print(f"📹 从视频中提取了 {len(frames)} 帧 (总帧数: {total_frames}, FPS: {fps:.2f})")
    return frames

# ============================
# 3. CLIP模型加载
# ============================
def load_clip_model(device, model_name="openai/clip-vit-base-patch32"):
    """
    加载CLIP模型
    
    Args:
        device: 计算设备
        model_name: 模型名称
    
    Returns:
        model, processor: CLIP模型和处理器
    """
    print(f"⏳  正在从 HuggingFace 下载并加载 CLIP 模型: {model_name}")
    print(f"📁  模型缓存目录: {MODEL_CACHE_DIR}")
    
    try:
        # 加载模型到指定设备
        model = CLIPModel.from_pretrained(
            model_name,
            cache_dir=str(MODEL_CACHE_DIR),
            use_safetensors=True
        ).to(device)
        model.eval()
        
        # 加载预处理器
        processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=str(MODEL_CACHE_DIR)
        )
        print("🎉 模型加载成功！部署完成。")
        return model, processor
    except Exception as e:
        print(f"💥 模型加载失败，请检查网络: {e}")
        raise

# ============================
# 4. 问答功能
# ============================
def answer_question_with_clip(model, processor, video_frames, question, device, candidate_answers=None):
    """
    使用CLIP回答关于视频的问题
    
    Args:
        model: CLIP模型
        processor: CLIP处理器
        video_frames: 视频帧列表
        question: 用户问题
        device: 计算设备
        candidate_answers: 候选答案列表（如果为None，则使用默认答案）
    
    Returns:
        str: 回答文本
    """
    # 如果没有提供候选答案，使用默认的问答模板
    if candidate_answers is None:
        # 根据问题类型生成候选答案
        question_lower = question.lower()
        if "real" in question_lower or "fake" in question_lower or "deepfake" in question_lower or "真实" in question or "虚假" in question or "伪造" in question:
            candidate_answers = [
                "This is a real person speaking in the video.",
                "This is a fake or deepfake video with artificial manipulation.",
                "The video shows authentic human speech and facial movements.",
                "The video contains synthetic or AI-generated content."
            ]
        elif "emotion" in question_lower or "表情" in question or "情绪" in question:
            candidate_answers = [
                "The person appears happy and joyful.",
                "The person appears sad or melancholic.",
                "The person appears angry or frustrated.",
                "The person appears neutral or calm."
            ]
        elif "speaking" in question_lower or "说话" in question or "讲话" in question:
            candidate_answers = [
                "The person is speaking clearly and naturally.",
                "The person's mouth movements match the audio.",
                "The person's lip-sync appears synchronized.",
                "There is a mismatch between audio and video."
            ]
        else:
            # 通用答案
            candidate_answers = [
                "Yes, this is evident in the video.",
                "No, this is not evident in the video.",
                "The video shows this characteristic clearly.",
                "The video does not show this characteristic."
            ]
    
    # 将问题与候选答案组合
    text_inputs = [f"{question} {answer}" for answer in candidate_answers]
    
    # 处理视频帧（取第一帧或平均多帧）
    if len(video_frames) == 1:
        image = video_frames[0]
    else:
        # 如果有多帧，使用第一帧作为代表（也可以平均多帧）
        image = video_frames[0]
    
    # 预处理
    inputs = processor(
        text=text_inputs,
        images=image,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    # 找到最匹配的答案
    best_idx = probs.argmax(dim=1).item()
    confidence = probs[0][best_idx].item()
    best_answer = candidate_answers[best_idx]
    
    return best_answer, confidence, probs.cpu().numpy()[0]

# ============================
# 5. 交互式对话主函数
# ============================
def interactive_chat(model, processor, device):
    """
    交互式对话循环
    """
    print("\n" + "=" * 50)
    print("🤖 CLIP 视频问答系统已启动")
    print("=" * 50)
    print("使用说明:")
    print("  - 输入视频路径和问题，格式: 视频路径|问题")
    print("  - 例如: /path/to/video.mp4|这个视频是真实的还是伪造的?")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("=" * 50 + "\n")
    
    current_video_path = None
    current_frames = None
    
    while True:
        try:
            user_input = input("请输入 (视频路径|问题) 或直接输入问题 (使用上次视频): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not user_input:
                continue
            
            # 解析输入
            if '|' in user_input:
                parts = user_input.split('|', 1)
                video_path = parts[0].strip()
                question = parts[1].strip()
                
                # 加载新视频
                print(f"\n📹 正在加载视频: {video_path}")
                current_frames = extract_frames_from_video(video_path, num_frames=8)
                current_video_path = video_path
            else:
                # 使用上次的视频
                question = user_input
                if current_frames is None:
                    print("❌ 错误: 请先提供视频路径")
                    continue
            
            if not question:
                print("❌ 错误: 问题不能为空")
                continue
            
            # 回答问题
            print(f"\n❓ 问题: {question}")
            print("🤔 正在分析视频...")
            
            answer, confidence, probs = answer_question_with_clip(
                model, processor, current_frames, question, device
            )
            
            print(f"\n💬 CLIP回答: {answer}")
            print(f"📊 置信度: {confidence:.2%}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()

# ============================
# 主程序入口
# ============================
def main():
    # 设置GPU
    device = setup_device(gpu_id=3)
    
    # 加载模型
    model, processor = load_clip_model(device)
    
    # 启动交互式对话
    interactive_chat(model, processor, device)

if __name__ == "__main__":
    main()
