from pptx import Presentation
from pptx.util import Inches
import os
import glob
import yaml
from PIL import Image

def load_config():
    config_file_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_file_path, "r", encoding="utf-8") as f:
        cfg_data = yaml.safe_load(f)
    return cfg_data

def compress_and_resize_image(input_path, output_path, target_width, target_height, quality=80):
    """
    调整图片大小到目标尺寸，并用指定的 JPEG 质量保存
    """
    with Image.open(input_path) as img:
        # 可选择用 resize() 强制图片变为目标尺寸，也可添加逻辑保持原图片长宽比（如有需要）
        img_resized = img.resize((target_width, target_height), Image.LANCZOS)
        img_resized.save(output_path, "JPEG", quality=quality)

def create_ppt(image_paths, ppt_config, compressed_folder="compressed_images"):
    """
    根据图片列表及 PPT 配置生成幻灯片，每张幻灯片添加一张经过预处理的图片。
    """
    ppt_path = ppt_config.get("ppt_path", "output.pptx")
    slide_width_inches = ppt_config.get("slide_width", 16)
    slide_height_inches = ppt_config.get("slide_height", 9)
    
    prs = Presentation()
    prs.slide_width = Inches(slide_width_inches)
    prs.slide_height = Inches(slide_height_inches)
    
    slide_layout_index = ppt_config.get("slide_layout_index", 6)
    blank_slide_layout = prs.slide_layouts[slide_layout_index]
    
    # 创建存放压缩图片的目录
    if not os.path.exists(compressed_folder):
        os.makedirs(compressed_folder)
    
    # 假设 PPT 为 96 DPI，则幻灯片对应的像素尺寸为：
    dpi = 96
    target_width = int(slide_width_inches * dpi)
    target_height = int(slide_height_inches * dpi)
    
    compressed_image_paths = []
    for img_path in image_paths:
        base = os.path.basename(img_path)
        compressed_path = os.path.join(compressed_folder, base)
        compress_and_resize_image(img_path, compressed_path, target_width, target_height, quality=80)
        compressed_image_paths.append(compressed_path)
    
    # 使用压缩后的图片生成幻灯片
    for img_path in compressed_image_paths:
        slide = prs.slides.add_slide(blank_slide_layout)
        left = Inches(0)
        top = Inches(0)
        width = prs.slide_width
        height = prs.slide_height
        slide.shapes.add_picture(img_path, left, top, width=width, height=height)
    
    prs.save(ppt_path)
    print(f"PPT 已保存到 {ppt_path}")

if __name__ == "__main__":
    # 加载配置文件
    cfg = load_config()
    
    # 从配置文件中获取图片输出目录与 PPT 配置
    output_folder = cfg["Paths"]["output_folder"]
    ppt_config = cfg.get("PPT", {})
    
    # 收集输出目录下所有 jpg/jpeg 图片，并排序（也可以处理其他格式）
    image_paths = glob.glob(os.path.join(output_folder, "*.jpg")) + \
                  glob.glob(os.path.join(output_folder, "*.jpeg"))
    image_paths = sorted(image_paths)
    
    # 生成 PPT，使用预处理好的图片
    create_ppt(image_paths, ppt_config)
