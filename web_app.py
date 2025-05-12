import gradio as gr
import os
import glob
import time
import shutil

# 获取当前文件所在的目录作为项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_new_user_directory():
    """
    在 BASE_DIR 下创建 upload_history 文件夹，
    并在其中创建最新的子文件夹，名称为 1, 2, 3, …（基于已有数字子文件夹自动递增）。
    在新的子文件夹内同时创建以下目录：
      - uploads：上传图片保存目录
      - mask_output：mask 保存目录
      - results：裁剪输出目录
      - generated_ppt：PPT 存储目录
    返回这四个目录的路径元组：(uploads_dir, mask_dir, output_dir, ppt_dir)
    """
    upload_history_root = os.path.join(BASE_DIR, "upload_history")
    if not os.path.exists(upload_history_root):
        os.makedirs(upload_history_root, exist_ok=True)

    # 查找已有数字子目录，选择下一个数字作为新目录名称
    existing_numbers = []
    for name in os.listdir(upload_history_root):
        path = os.path.join(upload_history_root, name)
        if os.path.isdir(path) and name.isdigit():
            existing_numbers.append(int(name))
    new_number = max(existing_numbers) + 1 if existing_numbers else 1
    new_folder_name = str(new_number)
    base_user_dir = os.path.join(upload_history_root, new_folder_name)

    # 创建指定的子目录
    uploads_dir = os.path.join(base_user_dir, "uploads")
    mask_dir = os.path.join(base_user_dir, "mask_output")
    output_dir = os.path.join(base_user_dir, "results")
    ppt_dir = os.path.join(base_user_dir, "generated_ppt")
    for directory in [uploads_dir, mask_dir, output_dir, ppt_dir]:
        os.makedirs(directory, exist_ok=True)

    return uploads_dir, mask_dir, output_dir, ppt_dir


# 使用 get_new_user_directory() 来生成用户专属目录
uploads_dir, mask_dir, output_dir, ppt_dir = get_new_user_directory()

# PPT 文件输出路径在用户目录下生成
PPT_OUTPUT_PATH = os.path.join(ppt_dir, "output.pptx")

# 创建用于 PPT 图片压缩的目录，位于用户目录（与 uploads 同级）
COMPRESSED_FOLDER = os.path.join(os.path.dirname(uploads_dir), "compressed_images")
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

# 相对路径设置：配置文件与模型权重（确保项目目录结构不变）
CONFIG_PATH = os.path.join(BASE_DIR, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
GROUNDED_CHECKPOINT = os.path.join(BASE_DIR, "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT = os.path.join(BASE_DIR, "sam_vit_h_4b8939.pth")

# 尝试从 detect_rect.py 导入 process_image，如未找到则使用默认实现
try:
    from detect_rect import process_image
except ImportError:
    def process_image(orig_image_path, config_path, grounded_checkpoint, sam_checkpoint,
                      box_threshold, text_threshold, text_prompt, device, mask_folder, output_folder):
        # 默认实现：将原图复制到 output_folder（仅用于调试）
        shutil.copy(orig_image_path, output_folder)

# 尝试从 create_ppt.py 导入 create_ppt，如未找到则使用默认实现
try:
    from create_ppt import create_ppt
except ImportError:
    def create_ppt(image_paths, ppt_config, compressed_folder="compressed_images"):
        ppt_path = ppt_config.get("ppt_path", "output.pptx")
        with open(ppt_path, "wb") as f:
            f.write(b"Dummy PPT content")
        return ppt_path

# 图像处理与 PPT 生成的其它默认参数
BOX_THRESHOLD = "0.3"
TEXT_THRESHOLD = "0.25"
TEXT_PROMPT = "display content without black borders"
DEVICE = "cpu"


def save_uploaded_files(uploaded_files):
    """
    将用户上传的文件保存到用户专属的 uploads_dir 中。
    上传组件使用 type="filepath"，每个上传对象为文件的本地路径（字符串）。
    为避免文件名冲突，在文件名前添加时间戳。
    返回保存后的文件路径列表。
    """
    saved_paths = []
    for file_path in uploaded_files:
        filename = os.path.basename(file_path)
        timestamp = str(int(time.time()))
        new_filename = f"{timestamp}_{filename}"
        dest_path = os.path.join(uploads_dir, new_filename)
        shutil.copy(file_path, dest_path)
        saved_paths.append(dest_path)
    return saved_paths


def run_processing(uploaded_files):
    """
    1. 将上传文件保存到 uploads_dir；
    2. 针对每张图片调用 process_image() 进行处理（mask 存放于 mask_dir，结果存放于 output_dir）；
    3. 返回每张图片的处理状态。
    """
    saved_paths = save_uploaded_files(uploaded_files)
    if not saved_paths:
        return "没有上传任何文件。"

    status_list = []
    for img_path in saved_paths:
        try:
            process_image(
                img_path,
                CONFIG_PATH,
                GROUNDED_CHECKPOINT,
                SAM_CHECKPOINT,
                BOX_THRESHOLD,
                TEXT_THRESHOLD,
                TEXT_PROMPT,
                DEVICE,
                mask_dir,
                output_dir
            )
            status_list.append(f"{os.path.basename(img_path)} 处理成功")
        except Exception as e:
            status_list.append(f"{os.path.basename(img_path)} 处理失败：{str(e)}")
    return "\n".join(status_list)


def run_create_ppt():
    """
    扫描 output_dir 中的 jpg/jpeg 图片，
    调用 create_ppt() 生成 PPT（保存在 PPT_OUTPUT_PATH 所指位置），
    返回 PPT 文件（相对于 BASE_DIR 的路径）与生成状态信息。
    """
    image_paths = sorted(
        glob.glob(os.path.join(output_dir, "*.jpg")) +
        glob.glob(os.path.join(output_dir, "*.jpeg"))
    )
    if not image_paths:
        return None, "在输出目录中没有找到处理后的图片。"

    ppt_config = {
        "ppt_path": PPT_OUTPUT_PATH,
        "slide_width": 16,
        "slide_height": 9,
        "slide_layout_index": 6
    }
    ppt_path = create_ppt(image_paths, ppt_config, COMPRESSED_FOLDER)
    
    # 若 create_ppt 返回 None，则直接返回错误信息
    if ppt_path is None:
        return None, "PPT 生成失败：create_ppt 返回 None。"
    
    # 将绝对路径转换为相对于 BASE_DIR 的路径，并替换 Windows 反斜杠为正斜杠
    relative_ppt_path = os.path.relpath(ppt_path, BASE_DIR).replace("\\", "/")
    
    if os.path.exists(ppt_path):
        status = f"PPT 已生成：{relative_ppt_path}"
    else:
        status = "PPT 生成失败，未能找到生成的文件。"
        relative_ppt_path = None
    return relative_ppt_path, status


# 构建 Gradio 前端界面
with gr.Blocks() as demo:
    gr.Markdown("# Rect2PPT 网页应用")
    with gr.Column():
        # 文件上传组件（支持多文件上传），type="filepath" 返回文件的本地路径
        uploaded_files = gr.File(
            label="上传图片",
            file_count="multiple",
            type="filepath"
        )
        process_btn = gr.Button("开始处理图片")
        process_status = gr.Textbox(label="处理状态", interactive=False, lines=8)
        ppt_btn = gr.Button("生成 PPT")
        ppt_file_output = gr.File(label="下载 PPT 文件", interactive=False)
        ppt_status = gr.Textbox(label="PPT 生成状态", interactive=False, lines=4)

    # 按钮绑定：点击“开始处理图片”后保存文件并启动图片处理流程
    process_btn.click(
        fn=run_processing,
        inputs=[uploaded_files],
        outputs=process_status
    )

    # 按钮绑定：点击“生成 PPT”后扫描 output_dir 并生成 PPT，返回下载链接和状态
    ppt_btn.click(
        fn=run_create_ppt,
        inputs=[],
        outputs=[ppt_file_output, ppt_status]
    )
    
# 启动 Gradio 服务，监听所有 IP，端口号为 7860
demo.launch(server_name="0.0.0.0", server_port=7860)
