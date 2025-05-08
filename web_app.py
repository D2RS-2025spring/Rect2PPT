import gradio as gr
import os
import glob

# 尝试从 detect_rect.py 导入图片处理相关函数
try:
    # 假定 detect_rect.py 中定义了 process_image 和 ensure_folders
    from detect_rect import process_image, ensure_folders
except ImportError:
    def process_image(orig_image_path, config_path, grounded_checkpoint, sam_checkpoint,
                      box_threshold, text_threshold, text_prompt, device, mask_folder, output_folder):
        import shutil
        shutil.copy(orig_image_path, output_folder)
    def ensure_folders(*folders):
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

# 尝试从 create_ppt.py 导入 PPT 生成函数
try:
    from create_ppt import create_ppt
except ImportError:
    def create_ppt(image_paths, ppt_config, compressed_folder="compressed_images"):
        ppt_path = ppt_config.get("ppt_path", "output.pptx")
        with open(ppt_path, "wb") as f:
            f.write(b"Dummy PPT content")
        return ppt_path

def run_processing(original_folder, mask_folder, output_folder):
    """
    对 original_folder 中的所有 jpg 图片进行处理：
      1. 自动确保原图目录、mask目录与输出目录存在；
      2. 遍历 original_folder 中的 jpg 图片，并调用 process_image() 处理；
      3. 生成的结果（裁剪后的图片）保存在 output_folder 中；
      4. 返回处理状态信息字符串。
    """
    # 确保所有目录存在
    ensure_folders(original_folder, mask_folder, output_folder)
    status_list = []
    image_paths = glob.glob(os.path.join(original_folder, "*.jpg"))
    if not image_paths:
        return "原图目录内没有找到 jpg 文件。"
    
    # 以下为硬编码的默认参数（根据你的实际情况修改默认值）
    config_path = "F:/ONESELF/AI-preject/vscode-preject/Rect2PPT/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "F:/ONESELF/AI-preject/vscode-preject/Rect2PPT/groundingdino_swint_ogc.pth"
    sam_checkpoint = "F:/ONESELF/AI-preject/vscode-preject/Rect2PPT/sam_vit_h_4b8939.pth"
    box_threshold = "0.3"
    text_threshold = "0.25"
    text_prompt = "display content without black borders"
    device = "cpu"
    
    # 针对每张图片调用 process_image
    for img in image_paths:
        try:
            process_image(img, config_path, grounded_checkpoint, sam_checkpoint,
                          box_threshold, text_threshold, text_prompt, device,
                          mask_folder, output_folder)
            status_list.append(f"{os.path.basename(img)} 处理成功")
        except Exception as e:
            status_list.append(f"{os.path.basename(img)} 处理失败：{str(e)}")
    return "\n".join(status_list)

def run_create_ppt(output_folder, ppt_file_path):
    """
    扫描 output_folder 中的所有 jpg 图片，
    调用 PPT 生成函数 create_ppt() 生成 PPT 文件，
    最后保存为 ppt_file_path 并返回提示信息。
    """
    image_paths = sorted(glob.glob(os.path.join(output_folder, "*.jpg")))
    if not image_paths:
        return None, "在输出目录中没有找到处理后的 jpg 图片。"
    
    ppt_config = {
        "ppt_path": ppt_file_path,
        "slide_width": 16,
        "slide_height": 9,
        "slide_layout_index": 6
    }
    ppt_path = create_ppt(image_paths, ppt_config, compressed_folder="compressed_images")
    status = f"PPT 已生成：{ppt_path}"
    return ppt_path, status

# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# Rect2PPT 网页应用")
    
    with gr.Tab("图片处理"):
        with gr.Column():
            orig_folder_input = gr.Textbox(label="原图目录", value="uploads", placeholder="请输入原图目录")
            mask_folder_input = gr.Textbox(label="Mask目录", value="mask_output", placeholder="请输入 mask 保存目录")
            output_folder_input = gr.Textbox(label="输出目录", value="results", placeholder="请输入处理后图片存放目录")
            process_btn = gr.Button("开始处理图片")
            process_status = gr.Textbox(label="处理状态", interactive=False, lines=10)
    
    with gr.Tab("生成 PPT"):
        with gr.Column():
            output_folder_ppt_input = gr.Textbox(label="处理后图片目录", value="results", placeholder="请输入处理后图片目录")
            ppt_file_path_input = gr.Textbox(label="生成 PPT 路径", value="generated_ppt/output.pptx", placeholder="请输入完整 PPT 文件路径")
            ppt_btn = gr.Button("生成 PPT")
            ppt_file_output = gr.File(label="下载 PPT 文件", interactive=False)
            ppt_status = gr.Textbox(label="PPT 生成状态", interactive=False, lines=4)
    
    # 绑定按钮函数并传入用户填写的参数
    process_btn.click(
        fn=run_processing,
        inputs=[orig_folder_input, mask_folder_input, output_folder_input],
        outputs=process_status
    )
    
    ppt_btn.click(
        fn=run_create_ppt,
        inputs=[output_folder_ppt_input, ppt_file_path_input],
        outputs=[ppt_file_output, ppt_status]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
