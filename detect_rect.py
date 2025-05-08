import subprocess
import os
import glob
import cv2
import numpy as np
import yaml

def load_config():
    """
    加载 YAML 配置文件，返回配置数据字典。
    假设 config.yaml 和 detect_rect.py 位于同一目录下。
    """
    config_file_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_file_path, 'r', encoding='utf-8') as f:
        cfg_data = yaml.safe_load(f)
    return cfg_data

def ensure_folders(*folders):
    """
    如果传入的目录不存在则创建。    
    """
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def find_minimum_quadrilateral(contour):
    """
    从给定轮廓中获取包裹 PPT 区域的“最小四边形”
    思路：
<<<<<<< HEAD
      1. 先计算该轮廓的凸包，期望 PPT 区域大体为凸形。
      2. 使用 cv2.approxPolyDP 在不同 epsilon 下对凸包进行近似，
         收集那些近似得到 4 个顶点的候选，并选择候选中面积最小的。
      3. 如果在预设范围内未能得到 4 个顶点，则采用轮廓中的 4 个极值点（上、右、下、左），
=======
      1. 先计算该轮廓的凸包（期望 PPT 区域大体为凸形）。
      2. 使用 cv2.approxPolyDP 在不同 epsilon 下对凸包进行近似，
         收集那些近似得到 4 个顶点的候选，并选择候选中面积最小的。
      3. 如果在预设范围内未能得到4个顶点，则采用轮廓中4个极值点（上、右、下、左），
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
         并按照中心角度排序后构成四边形。
    返回值为形状为 (4, 1, 2) 的四边形。
    """
    hull = cv2.convexHull(contour)
    arc_hull = cv2.arcLength(hull, True)
    candidates = []
<<<<<<< HEAD
    for factor in np.linspace(0.01, 0.3, 30):
        approx = cv2.approxPolyDP(hull, factor * arc_hull, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            candidates.append((approx, area))
    if candidates:
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
=======
    # 在一定范围内搜索合适的 epsilon
    for factor in np.linspace(0.01, 0.3, 30):
        approx = cv2.approxPolyDP(hull, factor * arc_hull, True)
        if len(approx) == 4:
            # 记录近似结果及对应面积
            area = cv2.contourArea(approx)
            candidates.append((approx, area))
    if candidates:
        # 选择面积最小、也即最紧贴凸包的四边形
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    # 如果以上均未获得4个顶点，则采用4个极值点方法
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    pts = hull.reshape(-1, 2)
    if pts.shape[0] < 4:
        return None
    top    = pts[np.argmin(pts[:, 1])]
    bottom = pts[np.argmax(pts[:, 1])]
    left   = pts[np.argmin(pts[:, 0])]
    right  = pts[np.argmax(pts[:, 0])]
    quad = np.array([top, right, bottom, left])
<<<<<<< HEAD
    center = np.mean(quad, axis=0)
    def angle(pt):
        return np.arctan2(pt[1] - center[1], pt[0] - center[0])
=======
    # 按照相对于中心的极角（顺时针排列）
    center = np.mean(quad, axis=0)
    def angle(pt):
        return np.arctan2(pt[1]-center[1], pt[0]-center[0])
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    quad = sorted(quad, key=angle)
    quad = np.array(quad).reshape(-1, 1, 2)
    return quad

def process_image(orig_image_path, config_path, grounded_checkpoint, sam_checkpoint,
                  box_threshold, text_threshold, text_prompt, device, mask_folder, output_folder):
    """
<<<<<<< HEAD
    处理单张图片：
      1. 调用外部脚本 grounded_sam_demo.py 生成 mask 文件；
      2. 删除其它辅助文件，仅保留 mask.jpg，并将其重命名为原图同名；
      3. 利用重命名后的 mask 对原图进行四边形检测和透视裁剪，
         裁剪结果保存到 output_folder 中。
=======
    处理单张图片：调用外部脚本生成 mask，重命名文件，
    然后加载原图和 mask 执行透视裁剪，最后写入输出目录。
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    """
    input_filename = os.path.basename(orig_image_path)
    print("\n-------------------------------------------")
    print(f"开始处理: {input_filename}")
<<<<<<< HEAD

    # 设置 grounded_sam_demo.py 的工作目录（假定其与本文件在同一项目中）
    demo_dir = os.path.dirname(os.path.abspath("grounded_sam_demo.py"))

=======
    
    # 第一步：调用 grounded_sam_demo.py 生成 mask 文件
    env = os.environ.copy()
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    command = [
        "python", "grounded_sam_demo.py",
        "--config", config_path,
        "--grounded_checkpoint", grounded_checkpoint,
        "--sam_checkpoint", sam_checkpoint,
        "--input_image", orig_image_path,
<<<<<<< HEAD
        "--output_dir", mask_folder,
=======
        "--output_dir", mask_folder,  # 输出到 mask_folder
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
        "--box_threshold", box_threshold,
        "--text_threshold", text_threshold,
        "--text_prompt", text_prompt,
        "--device", device
    ]
<<<<<<< HEAD
    print("调用的命令：", command)
    result = subprocess.run(command, env=os.environ.copy(), capture_output=True, text=True, cwd=demo_dir)
=======
    result = subprocess.run(command, env=env, capture_output=True, text=True)
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    print("grounded_sam_demo.py 执行完毕。")
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

<<<<<<< HEAD
    # 检查生成的 mask.jpg 是否存在，并只重命名一次
=======
    # 第二步：重命名生成的 mask.jpg 为与原图同名，并删除其它不需要的文件
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    mask_img_path = os.path.join(mask_folder, "mask.jpg")
    if not os.path.exists(mask_img_path):
        print(f"Error: {input_filename} 的 mask.jpg 未生成，跳过此文件。")
        return
<<<<<<< HEAD
=======

>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    new_mask_path = os.path.join(mask_folder, input_filename)
    os.rename(mask_img_path, new_mask_path)
    print(f"重命名: mask.jpg → {input_filename}")

<<<<<<< HEAD
    # 删除其它生成的辅助文件
=======
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    for extra_file in ["grounded_sam_output.jpg", "mask.json", "raw_image.jpg"]:
        extra_path = os.path.join(mask_folder, extra_file)
        if os.path.exists(extra_path):
            os.remove(extra_path)
            print(f"删除文件: {extra_file}")

<<<<<<< HEAD
    # 利用重命名后的 mask 文件对原图进行透视裁剪
=======
    # 第三步：透视裁剪 —— 读取 mask 与原图，对应转换及透视校正
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    mask_img = cv2.imread(new_mask_path, cv2.IMREAD_UNCHANGED)
    raw_img = cv2.imread(orig_image_path, cv2.IMREAD_UNCHANGED)
    if mask_img is None or raw_img is None:
        print(f"Error: 读取 {input_filename} 中的 mask 或原图失败，跳过。")
        return
    print(f"加载成功: mask shape {mask_img.shape}, 原图 shape {raw_img.shape}")

    try:
        hsv = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        print(f"HSV 转换失败: {e}")
        return

<<<<<<< HEAD
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    non_purple_mask = cv2.bitwise_not(purple_mask)
=======
    # 修改部分：提取非紫色区域（阈值可根据实际情况调整）
    # 定义紫色范围（可根据图像调整）
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    # 非紫色区域：反转掩膜（紫色部分为黑色，其它部分为白色）
    non_purple_mask = cv2.bitwise_not(purple_mask)
    
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    if np.all(non_purple_mask == 0):
        print(f"警告: {input_filename} 中未检测到非紫色区域，跳过。")
        return

    kernel = np.ones((5, 5), np.uint8)
    non_purple_mask_morph = cv2.morphologyEx(non_purple_mask, cv2.MORPH_CLOSE, kernel)
    non_purple_mask_morph = cv2.morphologyEx(non_purple_mask_morph, cv2.MORPH_OPEN, kernel)

<<<<<<< HEAD
=======
    # 查找轮廓，选择最大轮廓
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    contours, _ = cv2.findContours(non_purple_mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"未能检测到轮廓: {input_filename}，跳过。")
        return
<<<<<<< HEAD
=======

>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 100:
        print(f"警告: 轮廓面积太小（{cv2.contourArea(largest_contour)}）: {input_filename}，跳过。")
        return

<<<<<<< HEAD
=======
    # 获取包裹 PPT 区域的最小四边形（不要求标准矩形）
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    approx = find_minimum_quadrilateral(largest_contour)
    if approx is None or len(approx) != 4:
        print(f"无法获得合适的四边形，跳过 {input_filename}")
        return
    points = approx.reshape(4, 2)
    print("检测到角点 (mask 坐标系):")
    print(points)

<<<<<<< HEAD
=======
    # 将 mask 坐标转换到原图坐标系
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    scale_x = raw_img.shape[1] / mask_img.shape[1]
    scale_y = raw_img.shape[0] / mask_img.shape[0]
    scaled_points = points.astype(np.float32)
    scaled_points[:, 0] *= scale_x
    scaled_points[:, 1] *= scale_y

<<<<<<< HEAD
=======
    # 定义排序函数：左上、右上、右下、左下
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    ordered_pts = order_points(scaled_points)
    print("排序后的角点 (原图坐标系):")
    print(ordered_pts)

<<<<<<< HEAD
=======
    # 计算透视变换尺寸
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    (tl, tr, br, bl) = ordered_pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(raw_img, M, (maxWidth, maxHeight))
    print(f"{input_filename} 的透视校正完成，输出尺寸：({maxWidth} x {maxHeight})")

<<<<<<< HEAD
=======
    # 保存裁剪后的图片到输出目录
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
    output_path = os.path.join(output_folder, input_filename)
    cv2.imwrite(output_path, warped)
    print(f"保存裁剪图: {output_path}")

<<<<<<< HEAD
def process_images(orig_folder, mask_folder, output_folder):
    """
    遍历 orig_folder 中的所有 jpg 图片，并依次处理。
    """
    ensure_folders(orig_folder, mask_folder, output_folder)
    image_paths = glob.glob(os.path.join(orig_folder, "*.jpg"))
    print(f"检测到 {len(image_paths)} 张图片。")
    for img in image_paths:
        process_image(
            img,
            config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            grounded_checkpoint="groundingdino_swint_ogc.pth",
            sam_checkpoint="sam_vit_h_4b8939.pth",
            box_threshold="0.3",
            text_threshold="0.25",
            text_prompt="display content without black borders",
            device="cpu",
            mask_folder=mask_folder,
            output_folder=output_folder
        )

def run_create_ppt(output_folder, ppt_file_path):
    """
    扫描 output_folder 中的 jpg 图片，调用 PPT 生成函数创建 PPT 文件。
    """
    image_paths = sorted(glob.glob(os.path.join(output_folder, "*.jpg")))
    if not image_paths:
        return None, "在 output 目录中没有找到处理后的 jpg 图片。"
    from create_ppt import create_ppt
    ppt_config = {
        "ppt_path": ppt_file_path,
        "slide_width": 16,
        "slide_height": 9,
        "slide_layout_index": 6,
    }
    ppt_path = create_ppt(image_paths, ppt_config, compressed_folder="compressed_images")
    status = f"PPT 已生成：{ppt_path}"
    return ppt_path, status

if __name__ == "__main__":
    # 供测试时使用：从用户输入中接收四个位置
    orig_folder = input("请输入原图位置：")
    mask_folder = input("请输入 mask 位置：")
    output_folder = input("请输入 output 位置：")
    ppt_file_path = input("请输入生成 PPT 位置（包含文件名）：")
    
    process_images(orig_folder, mask_folder, output_folder)
    run_create_ppt(output_folder, ppt_file_path)
=======
def process_images():
    """
    加载配置后，遍历处理所有需处理的图片。
    """
    # 加载 YAML 配置
    cfg_data = load_config()
    orig_folder = cfg_data['Paths']['orig_folder']
    mask_folder = cfg_data['Paths']['mask_folder']
    output_folder = cfg_data['Paths']['output_folder']

    # 确保输出目录存在
    ensure_folders(mask_folder, output_folder)

    # 固定参数，可以根据需要修改或进一步外部化到配置文件中
    config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "groundingdino_swint_ogc.pth"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    box_threshold = "0.3"
    text_threshold = "0.25"
    text_prompt = "display content without black borders"
    device = "cpu"  # 使用 CPU

    image_paths = glob.glob(os.path.join(orig_folder, "*.jpg"))
    print(f"检测到 {len(image_paths)} 张图片需要处理。")

    for orig_image_path in image_paths:
        process_image(orig_image_path, config_path, grounded_checkpoint, sam_checkpoint,
                      box_threshold, text_threshold, text_prompt, device, mask_folder, output_folder)

def main():
    process_images()
    print("\n所有图片处理完成！")

if __name__ == "__main__":
    main()
>>>>>>> f479cd257306e1f1d7d002b97bd6c88fd4ad8bab
