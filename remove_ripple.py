import os
import cv2
import numpy as np
import yaml

# -------------------------- #
# 1. 读取配置文件获取文件夹路径 #
# -------------------------- #
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 原始未处理图片存放路径（output_folder）与处理后图片存放路径（remove-output）
input_folder = config["Paths"]["output_folder"]
processed_folder = config["Paths"]["remove-output"]

if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

allowed_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# ---------------------------- #
# 2. 遍历输入文件夹中的所有图片 #
# ---------------------------- #
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(allowed_ext):
        print(f"跳过非图片文件: {filename}")
        continue

    input_image_path = os.path.join(input_folder, filename)
    # 以彩色模式读取图片
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"加载图片失败: {input_image_path}")
        continue

    print(f"正在处理: {filename}")

    # ----------------------------------------------
    # 方法2：先用非局部均值去噪消除随机噪声
    # ----------------------------------------------
    denoised_color = cv2.fastNlMeansDenoisingColored(
        img, 
        None, 
        h=10, 
        hColor=10, 
        templateWindowSize=7, 
        searchWindowSize=21
    )

    # ----------------------------------------------
    # 在转换色彩空间的亮度通道上进行频域滤波：
    # 1. 将RGB转换到YCrCb（或者LAB），对Y通道处理
    # 2. 对Y通道做FFT，对频谱应用手动构造的陷波掩码
    # 3. 逆变换后替换原Y通道，再转换回BGR
    # ----------------------------------------------
    # 转换至YCrCb颜色空间
    img_ycrcb = cv2.cvtColor(denoised_color, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img_ycrcb)

    # FFT处理：对Y通道做二维傅里叶变换
    f = np.fft.fft2(Y)
    fshift = np.fft.fftshift(f)

    # 构造陷波掩模，屏蔽周期性波纹对应的频率
    rows, cols = Y.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    # 设定陷波参数（可根据实际频谱调优）
    r = 10      # 陷波半径
    offset = 50 # 频域中噪声条纹相对于中心的偏移量

    # 在左右两个对称位置构造零区
    mask[crow - r:crow + r, ccol + offset - r:ccol + offset + r] = 0
    mask[crow - r:crow + r, ccol - offset - r:ccol - offset + r] = 0

    # 应用掩模
    fshift_filtered = fshift * mask
    # 逆变换
    f_ishift = np.fft.ifftshift(fshift_filtered)
    Y_filtered = np.fft.ifft2(f_ishift)
    Y_filtered = np.abs(Y_filtered).astype(np.uint8)
    # 为防止数值偏差，通过归一化增强一下对比度
    Y_filtered = cv2.normalize(Y_filtered, None, 0, 255, cv2.NORM_MINMAX)

    # 重新组合色彩空间
    img_ycrcb_filtered = cv2.merge([Y_filtered, Cr, Cb])
    img_filtered = cv2.cvtColor(img_ycrcb_filtered, cv2.COLOR_YCrCb2BGR)

    # ----------------------------------------------
    # 额外应用中值滤波，进一步细化残留噪声
    # ----------------------------------------------
    final_output = cv2.medianBlur(img_filtered, 3)

    # ----------------------------------------------
    # 3. 保存处理后的图片，保持原有文件名
    # ----------------------------------------------
    output_image_path = os.path.join(processed_folder, filename)
    cv2.imwrite(output_image_path, final_output)
    print(f"处理完成并保存至: {output_image_path}")
