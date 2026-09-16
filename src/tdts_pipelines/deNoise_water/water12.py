import cv2
import numpy as np
import torch

def process_single_watershed(denoised_img, label, image_index):
    """单张图片的分水岭处理，返回需要收集的结果（和原循环逻辑一致）"""
    left_img, right_img, markers = single_watershed(denoised_img)
    # 生成原逻辑的metadata
    meta = {
        'image_index': image_index,
        'path_idx': 0,
        'label': label,
        'path_id': 8
    }
    return left_img, right_img, meta

def single_watershed(image):
    """
    更健壮的距离变换分割算法
    """
    gray = image.copy()
    
    roi_range = (13, 27)
    height, width = gray.shape
    
    # 二值化
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 增加OTSU自动阈值
      
    # 检查图像是否全黑或全白
    if np.all(binary == 0) or np.all(binary == 255):
        print("警告: 二值图像全黑或全白，使用默认分割")
        return default_split(gray, roi_range)
    
    try:

        # 距离变换
        #参数3表示使用3x3邻域，如果用2会导致距离值过小，系统表现为直接调用默认分割
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5) 

        # 寻找粘连点
        local_min = find_adhesion_points8(dist_transform, roi_range)
        #print(f"找到 {len(local_min)} 个切割点")
        
        # 创建标记
        markers = create_markers(gray, roi_range, local_min)
        
        # 分水岭分割
        return watershed_segmentation(gray, markers)
        
    except Exception as e:
        print(f"距离变换失败: {e}，使用默认分割")
        return default_split(gray, roi_range)

def create_markers(gray, roi_range, adhesion_points):
    """创建分水岭标记"""
    markers = np.zeros_like(gray, dtype=np.int32)
    
    # 在粘连点设置标记
    for y, x in adhesion_points:
        if roi_range[0] <= x <= roi_range[1]:
            markers[y, x] = 1

    # 边界标记
    markers[:, roi_range[0]] = 2
    markers[:, roi_range[1]] = 3
    
    return markers

def watershed_segmentation(gray, markers):
    """执行分水岭分割"""
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((2,2)))
    gradient_3ch = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    watershed_result = cv2.watershed(gradient_3ch, markers)
    
    # 提取分割图像
    left_img = np.zeros_like(gray)
    right_img = np.zeros_like(gray)
    
    left_mask = (watershed_result == 1) | (watershed_result == 2)
    right_mask = (watershed_result == 3)
    
    left_img[left_mask] = gray[left_mask]
    right_img[right_mask] = gray[right_mask]
    
    overlay = overlay_markers_on_image(gray, markers)

    return left_img, right_img, overlay

def default_split(gray, roi_range):
    """默认分割方法（保底）- 使用ROI中间线进行切割"""
    height, width = gray.shape
    
    # 计算ROI中间线位置
    split_x = (roi_range[0] + roi_range[1]) // 2
    
    # 创建真正的分割图像
    left_img = np.zeros_like(gray)
    right_img = np.zeros_like(gray)
    
    # 左部分：从左侧到分割线
    left_img[:, :split_x] = gray[:, :split_x]
    
    # 右部分：从分割线到右侧
    right_img[:, split_x:] = gray[:, split_x:]
    
    # 创建标记图 - 按照分水岭标记规范
    markers = np.zeros_like(gray, dtype=np.int32)
    markers[:, :split_x] = 1  # 分割线左侧标记为1
    markers[:, split_x] = -1  # 中间线标记为-1
    markers[:, split_x+1:] = 3  # 分割线右侧标记为3
    
    #print(f"使用ROI中间线分割: 分割线在x={split_x}")
    
    overlay = overlay_markers_on_image(gray, markers)

    return left_img, right_img, overlay

def default_split1(gray, roi_range):
    """默认分割方法（保底）- 使用垂直投影最小值创建真正的分割"""
    height, width = gray.shape
    
    # 提取ROI区域
    roi_region = gray[:, roi_range[0]:roi_range[1]+1]
    
    # 计算垂直投影（使用反转图像，让间隙更明显）
    inverted_roi = 255 - roi_region
    vertical_proj = np.sum(inverted_roi, axis=0)
    
    # 平滑投影曲线，减少噪声影响
    vertical_proj = cv2.GaussianBlur(vertical_proj, (5, 1), 0)
    
    # 找到投影最小值（间隙位置）
    min_proj_idx = np.argmin(vertical_proj)
    
    # 转换为全局坐标
    split_x = roi_range[0] + min_proj_idx
    
    # 创建真正的分割图像
    left_img = np.zeros_like(gray)
    right_img = np.zeros_like(gray)
    
    # 左部分：从左侧到分割线
    left_img[:, :split_x] = gray[:, :split_x]
    
    # 右部分：从分割线到右侧
    right_img[:, split_x:] = gray[:, split_x:]
    
    # 创建标记图 - 按照分水岭标记规范
    markers = np.zeros_like(gray, dtype=np.int32)
    markers[:, :split_x] = 1  # 分割线左侧标记为1
    markers[:, split_x] = -1  # 垂直投影线标记为-1
    markers[:, split_x+1:] = 3  # 分割线右侧标记为3
    
    print(f"使用投影最小值分割: 分割线在x={split_x}")
    
    overlay = overlay_markers_on_image(gray, markers)

    return left_img, right_img, overlay

def find_adhesion_points(dist_transform, roi_range):
    """寻找粘连点 4邻域"""
    points = []
    height, width = dist_transform.shape
    kernel = np.ones((2,2), np.uint8)
    dist_transform = cv2.morphologyEx(dist_transform, cv2.MORPH_OPEN, kernel)

    for y in range(1, height-1):
        for x in range(max(roi_range[0], 1), min(roi_range[1], width-1)):
            current = dist_transform[y, x]
            neighbors = [
                dist_transform[y-1, x], dist_transform[y+1, x],
                dist_transform[y, x-1], dist_transform[y, x+1]
            ]
            
            if current < min(neighbors) and current > 0:
                points.append((y, x))
    
    return points

def find_adhesion_points8(dist_transform, roi_range):
    """寻找粘连点 - 8邻域版本"""
    points = []
    height, width = dist_transform.shape
    
    for y in range(1, height-1):
        for x in range(max(roi_range[0], 1), min(roi_range[1], width-1)):
            current = dist_transform[y, x]
            
            # 8邻域检查
            neighbors = [
                dist_transform[y-1, x-1], dist_transform[y-1, x], dist_transform[y-1, x+1],
                dist_transform[y, x-1],                               dist_transform[y, x+1],
                dist_transform[y+1, x-1], dist_transform[y+1, x], dist_transform[y+1, x+1]
            ]
            
            if current < min(neighbors) and current > 0:
                points.append((y, x))
    
    return points


def overlay_markers_on_image(original_image, markers, alpha=0.4):
    """
    将标记区域叠加在原始图像上
    
    参数:
        original_image: 原始图像
        markers: 分水岭标记
        alpha: 叠加透明度
        
    返回:
        overlay: 叠加后的图像
    """
    # 确保原始图像是彩色图像
    if len(original_image.shape) == 2:
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original_image.clone()
    
    # 创建彩色标记图像
    colored_markers = np.zeros_like(original_bgr)
    
    # 为不同区域设置不同颜色
    # 左区域 - 红色
    colored_markers[(markers == 1) | (markers == 2)] = [0, 0, 255]  # BGR格式
    # 右区域 - 绿色
    colored_markers[markers == 3] = [0, 255, 0]
    # 分水岭边界 - 蓝色
    colored_markers[markers == -1] = [255, 0, 0]
    
    # 叠加图像
    overlay = cv2.addWeighted(original_bgr, 1-alpha, colored_markers, alpha, 0)
    
    return overlay