# Video Detection System (虚假视频检测系统)

> 🎓 毕业设计项目：基于多模态大模型的虚假说话人视频检测

## 📖 项目简介
本项目旨在利用深度学习技术（主要基于 CLIP 等多模态大模型），对视频内容进行分析，检测其中的深度伪造（DeepFake）成分，特别是针对虚假说话人的视频进行识别。

## 🛠️ 环境依赖
* **语言**: Python 3.x
* **框架**: PyTorch
* **关键库**: Transformers, OpenCV, NumPy

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone [https://github.com/yy2004yy/Video-Detection-System.git](https://github.com/yy2004yy/Video-Detection-System.git)
cd Video-Detection-System
```
### 2. 安装依赖
```bash
pip install -r requirements.txt
```
### 3. 运行代码
示例运行指令
``` bash
python main.py
```

## ⚠️ 关于模型权重 (Important)
由于 GitHub 文件大小限制，大模型权重文件（如 CLIP 预训练模型）未包含在此仓库中。 请在运行前确保：

手动下载所需的模型权重（.pth 或 .bin 文件）。

将权重文件放置在 models/ 目录下（参考代码中的路径配置）。

📝 待办事项 (To-Do)
[x] 项目初始化 & Git 仓库搭建

[ ] 完善数据预处理模块

[ ] 跑通基准模型 (Baseline)

[ ] 优化检测准确率


---

### 如何把它传上去？

既然你现在本地和服务器都已经同步好了，我建议你**在本地电脑**操作，体验一下“本地修改 -> 推送云端”的流程：

1.  **在本地文件夹里**新建一个文本文件，命名为 `README.md`（注意后缀是 `.md`，不是 `.txt`）。
2.  把上面的内容粘贴进去，保存。
3.  在文件夹里右键打开 Git Bash，执行“三部曲”：

```bash
git add README.md
git commit -m "添加项目说明文档"
git push
```
