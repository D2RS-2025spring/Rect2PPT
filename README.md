---

# Rect2PPT

Rect2PPT 是一个用于自动识别并裁剪会议或报告中拍摄的 PPT 照片的项目。该项目集成了先进的 [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) 模型，实现了从 PPT 照片中自动定位并提取 PPT 区域，并利用裁剪后的图片生成全新的 PPT 文件，从而大大简化了后期整理工作。

---

## 项目成员
黄俊豪（组长，学号：2024303110080），杨晓森（学号： 2024303110095），沈兵飞（学号：2024303110067），卢孟超（学号：2024303110085），郭旭龙 （学号：2024303110077）

---

## 特性

- **自动检测 PPT 区域**  
  利用 Grounded SAM 模型自动识别会议或报告中拍摄的 PPT 区域，省去人工标注的繁琐过程。

- **智能裁剪与处理**  
  对识别出的 PPT 区域进行透视变换与裁剪，保证生成的图片便于 PPT 的生成。

- **自动生成 PPT**  
  将处理后的图片整合生成 PPT 文件，让展示与汇报更为整洁专业。

- **自动化任务管理**  
  使用 [Invoke](http://www.pyinvoke.org/) 工具统一管理环境配置、图像处理与 PPT 生成任务，实现一站式自动化流程。

---

## 目录结构

```
Rect2PPT/
│
├── environment.yaml         # Conda 环境配置文件（记录所有依赖）
├── tasks.py                 # Invoke 任务定义文件，包含 setup-env、run-detect、run-create-ppt 和 all 任务
├── detect_rect.py           # 图片识别与 PPT 区域裁剪模块（集成 Grounded SAM 模型）
├── create_ppt.py            # 根据裁剪后的 PPT 区域图片生成 PPT 文件
├── setup_env.bat            # Windows 平台下的环境配置批处理脚本
├── web_app.py               # 基于 Gradio 的网页端入口，同时支持上传、处理与生成 PPT 下载
└── README.md                # 项目介绍文件
```

此外，项目运行时会自动创建如下用户目录结构（示例）：

```
BASE_DIR/
  upload_history/
    1/
      uploads/             # 上传的 PPT 原图
      mask_output/         # 自动生成的 mask 图片
      results/             # 裁剪处理后的 PPT 图片
      generated_ppt/       # 生成的 PPT 文件
  compressed_images/        # 临时存储 PPT 图片压缩结果（位于用户目录同级）
```

---

## 快速开始

### 1. 环境准备

- **安装 Conda**  
  请确保系统中已安装 [Conda](https://docs.conda.io)。

- **安装 Invoke**  
  使用 pip 安装 Invoke：
  ```bash
  pip install invoke
  ```
  
- **下载模型权重**  
  项目依赖两个关键的模型权重文件，请按照下面的命令下载：
  ```bash
  # 进入项目目录（如果该目录存在）
  cd Rect2PPT

  # 下载 SAM 模型的权重
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

  # 下载 GroundingDINO 模型的权重
  wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  ```
  
- **创建或激活虚拟环境**  
  根据 `environment.yaml` 文件创建一个 Conda 环境：
  ```bash
  invoke setup-env
  ```
  该任务会自动检测或创建目标环境，并提示如何激活。

### 2. 图片处理与 PPT 生成

项目提供两种调用方式：  
- **命令行方式（使用 Invoke）**  
- **网页端方式（基于 Gradio）**

#### 命令行方式

- **执行 PPT 区域识别与裁剪**  
  运行以下命令启动图像处理任务：
  ```bash
  invoke run-detect
  ```
  如果不传参数，则默认使用 `config.yaml` 中的路径设置。

- **生成 PPT 文件**  
  完成图像处理后，使用下面的命令生成 PPT 文件：
  ```bash
  invoke run-create-ppt
  ```
  同样，可使用 `config.yaml` 中的默认配置。

- **一键式执行所有任务**  
  若希望同时完成环境搭建、图像处理与 PPT 生成，可运行：
  ```bash
  invoke all
  ```

#### 网页端方式

运行以下命令启动 Gradio 网页服务：
```bash
python web_app.py
```
默认网址：[http://127.0.0.1:7860/](http://127.0.0.1:7860/)

在网页上，你可以：
- 上传图片（系统会自动创建用户专属目录，存放在 `upload_history` 下）。
- 点击“开始处理图片”启动图像处理任务。
- 点击“生成 PPT”生成 PPT 文件，并在网页上提供下载链接。

---

## 鸣谢

感谢 [Gradio](https://gradio.app/)、[python-pptx](https://python-pptx.readthedocs.io/)、[Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) 及其他开源项目对本项目的支持。

---

这份 README.md 文件概述了 Rect2PPT 项目的功能、目录结构、环境准备、使用说明与部署建议，帮助你快速上手并部署该项目。
## 适用场景

在现实中，会议或报告中的 PPT 照片往往由于拍摄角度和光照等因素难以直接用于展示或归档。Rect2PPT 通过自动检测和裁剪出 PPT 区域，再自动生成 PPT 文件，能够极大地简化整理和归档工作，提高工作效率。

