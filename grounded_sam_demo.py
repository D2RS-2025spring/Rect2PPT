#!/usr/bin/env python
import argparse
import os
import sys
import json
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# 获取脚本所在的绝对路径
base_dir = os.path.dirname(os.path.abspath(__file__))

# 将 GroundingDINO 和 segment_anything 模块所在目录加入搜索路径
sys.path.append(os.path.join(base_dir, "GroundingDINO"))
sys.path.insert(0, os.path.join(base_dir, "segment_anything"))

# ----------------- Grounding DINO 模块 -----------------
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# ----------------- Segment Anything 模块 -----------------
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

def load_image(image_path):
    """加载图片，并对图片进行预处理，返回 PIL 图片和 Tensor 格式的图像。"""
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)
    return image_pil, image_tensor

def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    """
    加载 GroundingDINO 模型：
      - 如果传入的 model_config_path 为目录，则自动拼接默认配置文件相对路径。
      - 检查配置文件是否存在，加载配置并构建模型。
    """
    model_config_path = os.path.abspath(model_config_path)
    if os.path.isdir(model_config_path):
        default_config = os.path.join(model_config_path, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
        print("传入的是目录，自动使用默认配置文件路径:", default_config)
        model_config_path = default_config

    if not os.path.exists(model_config_path):
        print(f"配置文件不存在: {model_config_path}")
        sys.exit(1)
    print("使用的配置文件:", model_config_path)

    args_config = SLConfig.fromfile(model_config_path)
    args_config.device = device
    args_config.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args_config)
    
    if not model_checkpoint_path:
        print("模型检查点路径为空，请检查 --grounded_checkpoint 参数。")
        sys.exit(1)
    model_checkpoint_path = os.path.abspath(model_checkpoint_path)
    if not os.path.exists(model_checkpoint_path):
        print(f"模型检查点文件不存在: {model_checkpoint_path}")
        sys.exit(1)
        
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("模型加载结果:", load_res)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    """利用 GroundingDINO 模型得到目标 box 和预测标签。"""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, num_classes)
    boxes = outputs["pred_boxes"].cpu()[0]              # (nq, 4)

    # 筛选候选框
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    if boxes_filt.shape[0] == 0:
        return boxes_filt, []

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            # 截取前三位数字作为置信度
            pred_phrase += f"({str(logit.max().item())[:4]})"
        pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_box(box, ax, label):
    """在指定轴上绘制边框和标签。"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label, fontsize=8, color='yellow')

def save_mask_data(output_dir, mask_list, box_list, label_list):
    """
    保存 mask 图像和 JSON 文件描述各个 mask 区域。
    此处生成的 mask 文件为背景紫色（RGB: [128, 0, 128]），而检测区域为白色（RGB: [255, 255, 255]）
    """
    # 取 mask_list 的尺寸 (n, 1, H, W)
    H, W = mask_list.shape[-2:]
    # 构造纯紫色背景（RGB）
    colored_mask = np.full((H, W, 3), [128, 0, 128], dtype=np.uint8)
    
    # 合并所有 mask（目标区域设为 True）
    mask_union = np.zeros((H, W), dtype=bool)
    for mask in mask_list:
        mask_np = mask.cpu().numpy()[0].astype(bool)
        mask_union |= mask_np

    # 将目标区域统一填充为白色
    colored_mask[mask_union] = [255, 255, 255]

    plt.figure(figsize=(10, 10))
    plt.imshow(colored_mask)
    plt.axis('off')
    mask_save_path = os.path.join(output_dir, 'mask.jpg')
    plt.savefig(mask_save_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

    # 生成 JSON 数据
    value = 0  # 0 表示背景
    json_data = [{
        'value': 0,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        if '(' in label:
            name, logit = label.split('(')
            logit = logit.rstrip(')')
        else:
            name = label
            logit = "0"
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True,
                        help="路径：配置文件或目录。如果是目录，则会使用默认配置文件路径。")
    parser.add_argument("--grounded_checkpoint", type=str, required=False,
                        default=r"F:\ONESELF\AI-preject\vscode-preject\Rect2PPT\groundingdino_swint_ogc.pth",
                        help="GroundingDINO 模型检查点路径")
    parser.add_argument("--sam_version", type=str, default="vit_h",
                        help="SAM ViT 版本: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False,
                        default=r"F:\ONESELF\AI-preject\vscode-preject\Rect2PPT\sam_vit_h_4b8939.pth",
                        help="SAM 模型检查点路径")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None,
                        help="SAM-HQ 模型检查点路径")
    parser.add_argument("--use_sam_hq", action="store_true",
                        help="是否使用 SAM-HQ 进行预测")
    parser.add_argument("--input_image", type=str, required=True,
                        help="输入图像文件路径")
    parser.add_argument("--text_prompt", type=str, required=True,
                        help="文本提示")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True,
                        help="输出目录")
    parser.add_argument("--box_threshold", type=float, default=0.3,
                        help="box 阈值")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="text 阈值")
    parser.add_argument("--device", type=str, default="cpu",
                        help="运行设备 (cpu 或 cuda)")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False,
                        help="bert_base_uncased 模型路径")
    
    args = parser.parse_args()

    # 转换路径为绝对路径
    args.config = os.path.abspath(args.config)
    print("传入的配置文件路径:", args.config)
    print("使用的模型检查点路径:", os.path.abspath(args.grounded_checkpoint))
    print("使用的 SAM 检查点路径:", os.path.abspath(args.sam_checkpoint))

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载图片
    image_pil, image_tensor = load_image(args.input_image)
    raw_image_path = os.path.join(args.output_dir, "raw_image.jpg")
    image_pil.save(raw_image_path)

    # 加载 GroundingDINO 模型
    model = load_model(args.config, args.grounded_checkpoint, args.bert_base_uncased_path, device=args.device)

    # 获取 GroundingDINO 模型输出
    boxes_filt, pred_phrases = get_grounding_output(
        model, image_tensor, args.text_prompt, args.box_threshold, args.text_threshold, device=args.device
    )
    if boxes_filt.shape[0] == 0:
        print(f"Error: {args.input_image} 未检测到任何候选框，mask.jpg 未生成，跳过此文件。")
        sys.exit(1)

    # 初始化 SAM 模型
    if args.use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[args.sam_version](checkpoint=args.sam_hq_checkpoint).to(args.device))
    else:
        predictor = SamPredictor(sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint).to(args.device))
    
    # 使用 OpenCV加载图片，并转换为 RGB 格式
    img = cv2.imread(args.input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)

    # 将 boxes 转换为图像尺度（从归一化中心坐标转为左上-右下坐标）
    size = image_pil.size  # (width, height)
    W, H = size[0], size[1]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, img.shape[:2]).to(args.device)

    # 使用 SAM 得到 mask
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(args.device),
        multimask_output=False,
    )
    
    if masks is None or masks.shape[0] == 0:
        print(f"Error: {args.input_image} 的 SAM 未生成任何 mask，mask.jpg 未生成，跳过此文件。")
        sys.exit(1)

    # 构建彩色复合图像：背景为紫色，目标区域保持原图颜色
    purple = np.array([128, 0, 128], dtype=np.uint8)
    composite = np.full(img.shape, purple, dtype=np.uint8)
    mask_union = np.zeros(img.shape[:2], dtype=bool)
    for mask in masks:
        mask_np = mask.cpu().numpy()[0].astype(bool)
        mask_union |= mask_np
    composite[mask_union] = img[mask_union]

    # 绘制带边框和标签的输出图像
    plt.figure(figsize=(10, 10))
    plt.imshow(composite)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    grounded_sam_output_path = os.path.join(args.output_dir, "grounded_sam_output.jpg")
    plt.savefig(grounded_sam_output_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

    # 保存 mask 数据，mask 文件为背景紫色，目标区域为白色
    save_mask_data(args.output_dir, masks, boxes_filt, pred_phrases)

    print("处理成功:", args.input_image)
    print("生成的 mask 文件:", os.path.join(args.output_dir, "mask.jpg"))
