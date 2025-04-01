from PIL import Image
import numpy as np
import cv2
import math

def analyze_gland_properties(image_path, dpi=96, pixel_area_mm2=4.85786e-4):
    img = np.array(Image.open(image_path))
    
    # 创建掩码
    gland_mask = (img[:, :, 0] == 0) & (img[:, :, 1] == 255) & (img[:, :, 2] == 0)  # gland
    leaf_mask = (img[:, :, 0] == 255) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)  # leaf
    
    # 计算gland和leaf的面积
    gland_area_pixels = np.sum(gland_mask)
    leaf_area_pixels = np.sum(leaf_mask)
    
    # 计算gland数量
    gland_contours, _ = cv2.findContours(gland_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gland_count = len(gland_contours)
    
    # 计算gland的平均直径
    gland_diameters = []
    for contour in gland_contours:
        area = cv2.contourArea(contour)
        if area > 0:
            diameter = math.sqrt(4 * area / math.pi)
            gland_diameters.append(diameter)
    average_diameter_pixels = np.mean(gland_diameters) if gland_diameters else 0
    
    # 计算真实面积（cm^2）
    gland_area_cm2 = gland_area_pixels * pixel_area_mm2 * 1e-2  # 转换为cm²
    leaf_area_cm2 = leaf_area_pixels * pixel_area_mm2 * 1e-2  # 转换为cm²
    
    # 计算腺体面积占叶片面积的百分比
    gland_leaf_ratio = gland_area_cm2 / leaf_area_cm2 if leaf_area_cm2 > 0 else 0
    
    # 计算每平方厘米的腺体密度
    gland_density_per_cm2 = gland_count / leaf_area_cm2 if leaf_area_cm2 > 0 else 0
    
    return {
        "gland_count": gland_count,
        "gland_area_pixels": gland_area_pixels,
        "average_diameter_pixels": average_diameter_pixels,
        "gland_area_cm2": gland_area_cm2,
        "leaf_area_cm2": leaf_area_cm2,
        "gland_leaf_ratio": gland_leaf_ratio,
        "gland_density_per_cm2": gland_density_per_cm2,
        "leaf_area_pixels": leaf_area_pixels  # 添加的测量指标
    }

# 读取图片路径
image_path = r"C:\myfile\data\u-net\date2\Dareconstructed\11-2-1.png"
# 分析图片得到的指标
results = analyze_gland_properties(image_path)

# 保存结果到txt文件
txt_file_path = r"C:\myfile\data\u-net\date2\Count.txt"
with open(txt_file_path, 'w') as f:
    f.write("腺体总个数: {}\n".format(results['gland_count']))
    f.write("腺体像素点总面积: {}\n".format(results['gland_area_pixels']))
    f.write("棉花叶片的像素点总面积: {}\n".format(results['leaf_area_pixels']))  
    f.write("平均腺体直径（像素点）: {:.2f} 像素点\n".format(results['average_diameter_pixels']))
    f.write("棉花叶片面积（cm^2）: {:.4f}\n".format(results['leaf_area_cm2']))
    f.write("腺体面积（cm^2）: {:.4f}\n".format(results['gland_area_cm2']))
    f.write("腺体面积占叶片面积百分比: {:.4f}\n".format(results['gland_leaf_ratio'] * 100))
    f.write("平均每平方厘米的叶片上有多少个色素腺体: {:.2f}\n".format(results['gland_density_per_cm2']))
    
print("结果已保存到:", txt_file_path)
