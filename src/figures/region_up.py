# region_up.py
# 本文件用于实现区域生长的跟踪功能
import numpy as np

def trace_region_growth_with_steps(image, max_steps=100):
    """
    跟踪区域生长过程，返回每个步骤的图像和点集
    经过后处理过滤，确保每一步都包含新的一行，便于展示垂直切割过程
    
    参数:
    - image: 输入图像
    - max_steps: 最大跟踪步数
    
    返回:
    - image_list: 每个步骤的图像列表
    - points_list: 每个步骤累计的点列表（包含之前所有步骤的点）
    """
    height, width = image.shape
    x_range = (13, 27)
    
    # 运行原始算法（不修改原逻辑）
    found_points = set()
    image_list = []
    points_list = []
    
    # 步骤0: 找到起点
    min_val = float('inf')
    min_x = None
    for x in range(x_range[0], x_range[1] + 1):
        if x < width:
            val = image[0, x]
            if val < min_val:
                min_val = val
                min_x = x
    
    if min_x is not None:
        found_points.add((min_x, 0))
        points_list.append([(min_x, 0)])
        marked_image = image.copy()
        # 标记起点
        marked_image[0, min_x] = 255
        image_list.append(marked_image)
    
    def generate_candidates(points_set):
        candidates = set()
        for (x, y) in points_set:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    new_x = x + dx
                    new_y = y + dy
                    if (new_y >= 0 and new_y < height and 
                        x_range[0] <= new_x <= x_range[1] and 
                        new_x < width):
                        candidates.add((new_x, new_y))
        return candidates
    
    # 执行生长过程，记录每一步
    # 改为while循环，确保能够完整生长
    step = 1
    while step < max_steps:
        candidates = generate_candidates(found_points)
        candidates = {(x, y) for (x, y) in candidates if (x, y) not in found_points}
        
        if not candidates:
            break
        
        zero_candidates = [(x, y) for (x, y) in candidates if image[y, x] == 0]
        
        if zero_candidates:
            new_points = set(zero_candidates)
        else:
            min_val = float('inf')
            min_candidates = []
            for (x, y) in candidates:
                val = image[y, x]
                if val < min_val:
                    min_val = val
                    min_candidates = [(x, y)]
                elif val == min_val:
                    min_candidates.append((x, y))
            
            if not min_candidates:
                break
            new_points = set(min_candidates)
        
        if not new_points:
            break
        
        found_points.update(new_points)
        
        # 记录这一步累计的点集（包含之前所有步骤的点）
        sorted_all_points = sorted(list(found_points), key=lambda p: (p[1], p[0]))
        points_list.append(sorted_all_points)
        
        # 创建标记图像
        marked_image = image.copy()
        # 标记所有已找到的点
        for (x, y) in found_points:
            marked_image[y, x] = 255
        image_list.append(marked_image)
        
        # 检查是否到达底部
        if any(y == height - 1 for (x, y) in new_points):
            break
            
        step += 1
    
    # ========== 后处理过滤：确保每一步都有新的一行 ==========
    filtered_points_list = []
    filtered_image_list = []
    
    if not points_list:
        return [], []
    
    # 第一步总是保留（起点）
    filtered_points_list.append(points_list[0])
    filtered_image_list.append(image_list[0])
    
    # 计算每一步的最大行数
    def get_max_row(points):
        if not points:
            return -1
        return max(y for (x, y) in points)
    
    last_max_row = get_max_row(points_list[0])
    
    # 从第二步开始过滤
    for i in range(1, len(points_list)):
        current_points = points_list[i]
        current_max_row = get_max_row(current_points)
        
        # 只有当最大行数增加时才保留这一步
        if current_max_row > last_max_row:
            filtered_points_list.append(current_points)
            filtered_image_list.append(image_list[i])
            last_max_row = current_max_row
    
    return filtered_image_list, filtered_points_list