# Rect2PPT

Rect2PPT 是一个用于自动识别并裁剪会议或报告中拍摄的 PPT 照片的项目。该项目集成了先进的 [Grounded SAM](https://github.com/IDEA-Research/Grounded-SAM) 模型，实现了从 PPT 照片中自动定位并提取 PPT 区域，并利用裁剪后的图片生成全新的 PPT 文件，从而大大简化了后期整理工作。

## 特性

- **自动检测 PPT 区域**  
  使用 Grounded SAM 模型自动定位会议或报告中拍摄的 PPT 区域。

- **智能裁剪与处理**  
  对识别出的 PPT 区域进行裁剪，为后续 PPT 文件生成做好准备。

- **自动生成 PPT**  
  从裁剪后的图片自动生成 PPT 文件，让汇报内容更为整洁专业。

- **自动化任务管理**  
  结合 [Invoke](http://www.pyinvoke.org/) 工具管理各项任务，实现环境配置、图像处理与 PPT 生成的一站式自动化执行。

## 目录结构

```
Rect2PPT/
│
├── environment.yaml         # Conda 环境配置文件（记录所有依赖）
├── config.yaml              # 项目配置文件（路径设置：原始图片、裁剪后的图片、PPT 输出目录等）
├── tasks.py                 # Invoke 任务定义文件，包含 setup-env、run-detect、run-create-ppt 和 all 任务
├── detect_rect.py           # 图片识别与 PPT 区域裁剪模块（集成 Grounded SAM 模型）
├── create_ppt.py            # 根据裁剪后的 PPT 区域图片生成 PPT 文件
├── setup_env.bat            # Windows 平台下的环境配置批处理脚本
└── README.md                # 项目介绍文件
```

## 快速开始

### 1. 环境准备

- **安装 Conda**  
  请确保系统中已安装 [Conda](https://docs.conda.io)。

- **生成 environment.yaml**  
  在项目根目录下，通过如下命令生成环境配置文件（注意过滤掉带有本地路径的 `prefix` 字段）：
  ```bash
  conda env export --no-builds | grep -v "prefix:" > environment.yaml
  ```

- **安装 Invoke**  
  安装 Invoke 来管理各项任务：
  ```bash
  pip install invoke
  ```

### 2. 使用 Invoke 创建环境

在项目根目录下运行如下命令：
```bash
invoke setup-env
```

该命令将会：
- 读取 `environment.yaml` 中的目标环境名称；
- 检查该 Conda 环境是否已存在；
- 如果不存在，则自动创建该环境并提示激活命令。

### 3. 图片处理与 PPT 生成

#### 识别与裁剪操作

使用下面的命令调用 PPT 区域识别与裁剪模块：
```bash
invoke run-detect --input "your_input_folder" --tmp_dir "your_tmp_folder"
```
- 如果不传入参数，将默认使用 `config.yaml` 中 `Paths` 分区配置的路径设置。

#### PPT 自动生成

裁剪操作完成后，运行以下命令生成 PPT 文件：
```bash
invoke run-create-ppt --image_dir "your_image_dir" --output "your_output_folder"
```
- 同样，如果不传入参数，会自动使用 `config.yaml` 提供的默认设置。

#### 一键式执行所有任务

你也可以使用下面的命令一键执行环境配置、图像处理和 PPT 生成全部流程：
```bash
invoke all
```

## 配置文件说明

项目中使用 `config.yaml` 管理路径及其他配置，其主要结构如下：
```yaml
Paths:
  orig_folder: "path/to/original_images"   # 原始 PPT 照片存放目录
  mask_folder: "path/to/cropped_images"      # 裁剪后图片存放目录
  output_folder: "path/to/output_ppt"        # 生成 PPT 的输出目录
```

请确保根据实际情况调整配置文件中的路径。

## 适用场景

在现实中，会议或报告中的 PPT 照片往往由于拍摄角度和光照等因素难以直接用于展示或归档。Rect2PPT 通过自动检测和裁剪出 PPT 区域，再自动生成 PPT 文件，能够极大地简化整理和归档工作，提高工作效率。

