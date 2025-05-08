#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
from detect_rect import detect_rectangles, four_point_transform
from create_ppt import create_ppt

def process_image(input_image_path: str, extracted_dir: str) -> list:
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir)
    try:
        rectangles, img = detect_rectangles(input_image_path)
    except Exception as e:
        print(f"检测矩形区域时出错：{e}")
        return []
    if not rectangles:
        print("未检测到符合条件的矩形区域。")
        return []
    extracted_image_paths = []
    for idx, rect in enumerate(rectangles):
        warped = four_point_transform(img, rect)
        output_path = os.path.join(extracted_dir, f"extracted_{idx}.jpg")
        cv2.imwrite(output_path, warped)
        print(f"[INFO] 保存裁剪的图片到 {output_path}")
        extracted_image_paths.append(output_path)
    return extracted_image_paths

def main():
    parser = argparse.ArgumentParser(
        description="自动检测图片中的矩形区域并生成 PPT 文件"
    )
    parser.add_argument("--input", type=str, required=True, help="输入图片文件路径，例如: input.jpg")
    parser.add_argument("--output", type=str, default="output.pptx", help="输出 PPT 文件路径，默认 output.pptx")
    parser.add_argument("--tmp_dir", type=str, default="extracted_images", help="裁剪图片临时存放目录，默认为 extracted_images")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"输入文件 {args.input} 不存在，请检查路径。")
        return

    print("[INFO] 开始处理图片...")
    extracted_paths = process_image(args.input, args.tmp_dir)
    if not extracted_paths:
        print("未提取到任何图片，程序终止。")
        return

    print("[INFO] 开始生成 PPT...")
    create_ppt(extracted_paths, args.output)
    print("[INFO] PPT 已生成。")

if __name__ == "__main__":
    main()
