#cut_region3.py
import numpy as np

def find_connected_points(image):
    return find_bidirectional_points(image)

#本函数使用了集合，不适合@jit加速
def find_connected_points1(image):
    '''
    FD1 全连通生长切割（8邻域）- 基准算法
    '''
    height, width = image.shape
    x_range = (13, 27)
    
    found_points = set()
    
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
    
    prev_points = set(found_points)
    
    while True:
        candidates = generate_candidates(found_points)
        candidates = {(x, y) for (x, y) in candidates if (x, y) not in found_points}
        
        if not candidates:
            break
        
        zero_candidates = [ (x, y) for (x, y) in candidates if image[y, x] == 0 ]
        
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
        
        if any(y == height - 1 for (x, y) in new_points):
            break
    
    points = sorted(list(found_points), key=lambda p: p[1])
    return points

def find_connected_points1Min(image):
    '''
    FD2 全连通生长切割（8邻域）- 基准算法，但是优先向下,最小值添加时只添加一个点，最小值多时选择y最大且x最接近中心的点
    '''
    height, width = image.shape
    x_range = (13, 27)
    center_x = (x_range[0] + x_range[1]) // 2
    
    found_points = set()
    
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

    prev_points = set(found_points)
    
    while True:
        candidates = generate_candidates(found_points)
        candidates = {(x, y) for (x, y) in candidates if (x, y) not in found_points}
        
        if not candidates:
            break
        
        zero_candidates = [ (x, y) for (x, y) in candidates if image[y, x] == 0 ]
        
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

            # 修改点：从所有最小值点中选择一个最优的点
            # 优先级：1. y最大（向下生长最快） 2. x最接近中心
            min_candidates.sort(key=lambda p: (-p[1], abs(p[0] - center_x)))
            best_point = min_candidates[0]
            new_points = {best_point}  # 只添加一个点

        if not new_points:
            break
        
        found_points.update(new_points)
        
        if any(y == height - 1 for (x, y) in new_points):
            break
    
    points = sorted(list(found_points), key=lambda p: p[1])
    return points

def find_row_optimal_points(image):
    """FD3修正：生成每行所有最小值点的点集
    
    核心机制：每行独立收集所有最小值像素点
    像素选择：每行在x范围内找到值最小的像素，添加所有等于该最小值的点
    连通性：无连通性保证，点集可能不连通
    返回：点集（可能每行多个点）
    """
    height, width = image.shape
    x_range = (13, 27)
    
    point_set = []  # 存储所有点的列表
    
    for y in range(height):
        # 1. 找到当前行的最小值
        min_val = float('inf')
        
        # 首先扫描找到最小值
        for x in range(x_range[0], min(x_range[1] + 1, width)):
            val = image[y, x]
            if val < min_val:
                min_val = val
        
        # 2. 收集所有等于最小值的点
        for x in range(x_range[0], min(x_range[1] + 1, width)):
            if image[y, x] == min_val:
                point_set.append((x, y))
    
    return point_set

#这个应该是错的
def find_background_only_points(image):
    """FD999: 纯背景连通切割（严格零值）
    
    核心机制：只选择零值像素并保持连通性生长
    像素选择：仅添加0像素，忽略非0像素
    连通性：强连通性保证，但可能提前终止
    终止条件：到达底部或没有更多0像素
    回退机制：当0像素路径无法到达底部时，剩余部分使用中间线
    """
    height, width = image.shape
    x_range = (13, 27)
    
    found_points = set()
    
    # 在第一行寻找0像素作为起始点
    zero_points = []
    for x in range(x_range[0], x_range[1] + 1):
        if x < width and image[0, x] == 0:
            zero_points.append((x, 0))
    
    if zero_points:
        # 选择中间的0像素作为起始点
        start_point = zero_points[len(zero_points) // 2]
        found_points.add(start_point)
    else:
        # 如果没有0像素，直接返回中间线
        mid_x = (x_range[0] + x_range[1]) // 2
        return [(mid_x, y) for y in range(height)]
    
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
    
    while True:
        candidates = generate_candidates(found_points)
        candidates = {(x, y) for (x, y) in candidates if (x, y) not in found_points}
        
        if not candidates:
            break
        
        # 只选择0像素
        zero_candidates = [(x, y) for (x, y) in candidates if image[y, x] == 0]
        
        if not zero_candidates:
            break  # 没有更多0像素，停止生长
        
        new_points = set(zero_candidates)
        found_points.update(new_points)
        
        if any(y == height - 1 for (x, y) in new_points):
            break  # 到达底部
    
    # 转换为有序列表
    zero_path = sorted(list(found_points), key=lambda p: p[1])
    
    # 检查是否覆盖了所有行，如果没有，用中间线补全
    covered_y = set(y for x, y in zero_path)
    all_y = set(range(height))
    missing_y = sorted(list(all_y - covered_y))
    
    if missing_y:
        # 确定补全的x坐标：使用最后一个0像素的x坐标或中间值
        if zero_path:
            last_x = zero_path[-1][0]
        else:
            last_x = (x_range[0] + x_range[1]) // 2
        
        for y in missing_y:
            zero_path.append((last_x, y))
        
        # 重新排序
        zero_path.sort(key=lambda p: p[1])
    
    return zero_path

def find_4neighbor_points(image):
    """FD4: 受限连通生长切割（4邻域）
    
    核心机制：4邻域连通生长（上下左右）
    像素选择：优先添加0像素，无0时添加最小值像素
    连通性：受限连通性保证（4邻域而非8邻域）
    终止条件：到达底部或无法继续生长
    """
    height, width = image.shape
    x_range = (13, 27)
    center_x = (x_range[0] + x_range[1]) // 2

    found_points = set()
    
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
    
    def generate_candidates(points_set):
        candidates = set()
        for (x, y) in points_set:
            # 只考虑4邻域（上下左右）
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x = x + dx
                new_y = y + dy
                if (new_y >= 0 and new_y < height and 
                    x_range[0] <= new_x <= x_range[1] and 
                    new_x < width):
                    candidates.add((new_x, new_y))
        return candidates
    
    prev_points = set(found_points)
    
    while True:
        candidates = generate_candidates(found_points)
        candidates = {(x, y) for (x, y) in candidates if (x, y) not in found_points}
        
        if not candidates:
            break
        
        zero_candidates = [ (x, y) for (x, y) in candidates if image[y, x] == 0 ]
        
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
        
        if any(y == height - 1 for (x, y) in new_points):
            break
    
    points = sorted(list(found_points), key=lambda p: p[1])
    return points

def find_bidirectional_points(image):
    """FD5：双向连通生长切割（8邻域双向生长）- 优化版本
    
    核心机制：同时从顶部和底部开始生长，优先0值点，交替执行最小值点生长
    像素选择：优先0值点，其次最小值点
    生长流程：
      1. 从上往下生长down，直到没有新的0值点
      2. 从下往上生长up，直到没有新的0值点
      3. down：添加一批8邻域的最小值点
      4. up：添加一批8邻域的最小值点
      5. 查找down的邻域是否有0值点，如果有，跳转到1执行，否则跳转到3执行
      6. 查找up的邻域是否有0值点，如果有，跳转到2执行，否则跳转到4执行
    终止条件：两个生长区域有重叠行，且重叠点达到阈值
    """
    height, width = image.shape
    x_range = (13, 27)
    
    # 1. 初始化两个生长点集
    top_set = set()  # 从顶部向下生长
    bottom_set = set()  # 从底部向上生长
    
    # 2. 选择顶部种子点（第一行的最小值点）
    top_seed = find_first_row_min_point(image, x_range)
    if top_seed is not None:
        top_set.add(top_seed)
    
    # 3. 选择底部种子点（最后一行的最小值点）
    bottom_seed = find_last_row_min_point(image, x_range, height)
    if bottom_seed is not None:
        bottom_set.add(bottom_seed)
    
    # 4. 定义局部变量跟踪生长状态
    iteration = 0
    max_iterations = height * 4
    
    # 5. 双向生长循环
    while iteration < max_iterations:
        iteration += 1
        
        # 5.1 从上往下生长0值点（down - 0值点）
        top_grew = True
        while top_grew:
            top_new = grow_zero_step(top_set, image, x_range, height)
            if top_new:
                top_set.update(top_new)
                top_grew = True
            else:
                top_grew = False
        
        # 检查终止条件
        if should_merge_bidirectional(top_set, bottom_set, x_range, width):
            break
        
        # 5.2 从下往上生长0值点（up - 0值点）
        bottom_grew = True
        while bottom_grew:
            bottom_new = grow_zero_step(bottom_set, image, x_range, height)
            if bottom_new:
                bottom_set.update(bottom_new)
                bottom_grew = True
            else:
                bottom_grew = False
        
        # 检查终止条件
        if should_merge_bidirectional(top_set, bottom_set, x_range, width):
            break
        
        # 5.3 down：添加一批8邻域的最小值点
        top_new = grow_min_step(top_set, image, x_range, height)
        if top_new:
            top_set.update(top_new)
        
        # 检查终止条件
        if should_merge_bidirectional(top_set, bottom_set, x_range, width):
            break
        
        # 5.4 up：添加一批8邻域的最小值点
        bottom_new = grow_min_step(bottom_set, image, x_range, height)
        if bottom_new:
            bottom_set.update(bottom_new)
        
        # 检查终止条件
        if should_merge_bidirectional(top_set, bottom_set, x_range, width):
            break
        
        # 5.5 检查是否有新的0值点可以生长
        if has_zero_neighbor(top_set, image, x_range, height):
            # 如果down的邻域有0值点，继续循环（会回到步骤1）
            continue
        
        # 5.6 检查up的邻域是否有0值点
        if has_zero_neighbor(bottom_set, image, x_range, height):
            # 如果up的邻域有0值点，继续循环（会回到步骤2）
            continue
        
        # 如果两个方向都没有新的0值点且没有新的最小值点，跳出循环
        if not top_new and not bottom_new:
            break
    
    # 6. 合并两个点集
    all_points = sorted(list(top_set.union(bottom_set)), key=lambda p: p[1])
    return all_points
def grow_zero_step(point_set, image, x_range, height):
    """生长一步：只添加0值点（8邻域）
    
    返回新添加的0值点集合
    """
    if not point_set:
        return set()
    
    # 生成当前点集的所有8邻域候选点
    candidates = set()
    for (x, y) in point_set:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_x = x + dx
                new_y = y + dy
                
                # 检查边界和范围
                if (0 <= new_y < height and 
                    x_range[0] <= new_x <= x_range[1] and 
                    new_x < image.shape[1]):
                    candidates.add((new_x, new_y))
    
    # 移除已有点
    candidates = candidates - point_set
    
    if not candidates:
        return set()
    
    # 只选择0值点
    zero_candidates = [(x, y) for (x, y) in candidates if image[y, x] == 0]
    
    return set(zero_candidates)


def grow_min_step(point_set, image, x_range, height):
    """生长一步：只添加最小值点（8邻域）
    
    如果没有0值点，添加所有最小值点
    返回新添加的最小值点集合
    """
    if not point_set:
        return set()
    
    # 生成当前点集的所有8邻域候选点
    candidates = set()
    for (x, y) in point_set:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_x = x + dx
                new_y = y + dy
                
                # 检查边界和范围
                if (0 <= new_y < height and 
                    x_range[0] <= new_x <= x_range[1] and 
                    new_x < image.shape[1]):
                    candidates.add((new_x, new_y))
    
    # 移除已有点
    candidates = candidates - point_set
    
    if not candidates:
        return set()
    
    # 移除0值点（因为grow_min_step只处理非0的最小值点）
    non_zero_candidates = [(x, y) for (x, y) in candidates if image[y, x] != 0]
    
    if not non_zero_candidates:
        return set()
    
    # 找到最小值
    min_val = float('inf')
    for (x, y) in non_zero_candidates:
        val = image[y, x]
        if val < min_val:
            min_val = val
    
    # 收集所有等于最小值的点
    min_candidates = [(x, y) for (x, y) in non_zero_candidates if image[y, x] == min_val]
    
    return set(min_candidates)


def has_zero_neighbor(point_set, image, x_range, height):
    """检查点集的8邻域中是否有0值点
    
    返回布尔值：True表示有0值点邻域，False表示没有
    """
    if not point_set:
        return False
    
    # 生成当前点集的所有8邻域候选点
    candidates = set()
    for (x, y) in point_set:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_x = x + dx
                new_y = y + dy
                
                # 检查边界和范围
                if (0 <= new_y < height and 
                    x_range[0] <= new_x <= x_range[1] and 
                    new_x < image.shape[1]):
                    candidates.add((new_x, new_y))
    
    # 移除已有点
    candidates = candidates - point_set
    
    if not candidates:
        return False
    
    # 检查是否有0值点
    for (x, y) in candidates:
        if image[y, x] == 0:
            return True
    
    return False

def find_first_row_min_point(image, x_range):
    """找到第一行的最小值点"""
    width = image.shape[1]
    min_val = float('inf')
    min_points = []
    
    for x in range(x_range[0], min(x_range[1] + 1, width)):
        val = image[0, x]
        if val < min_val:
            min_val = val
            min_points = [(x, 0)]
        elif val == min_val:
            min_points.append((x, 0))
    
    if min_points:
        # 如果有多个最小值，选择中间的
        return min_points[len(min_points) // 2]
    return None

def find_last_row_min_point(image, x_range, height):
    """找到最后一行的最小值点"""
    width = image.shape[1]
    min_val = float('inf')
    min_points = []
    
    for x in range(x_range[0], min(x_range[1] + 1, width)):
        val = image[height-1, x]
        if val < min_val:
            min_val = val
            min_points = [(x, height-1)]
        elif val == min_val:
            min_points.append((x, height-1))
    
    if min_points:
        # 如果有多个最小值，选择中间的
        return min_points[len(min_points) // 2]
    return None

def should_merge_bidirectional(top_set, bottom_set, x_range, width):
    """判断两个点集是否应该合并（停止条件）
    
    停止条件：
    1. 两个点集的y范围有重叠
    2. 从向下生长的点集中，取出重叠行的所有点
    3. 检查这些点有多少个也在向上生长的点集中
    4. 如果相同点数量 >= 14，则停止
    """
    if not top_set or not bottom_set:
        return False
    
    # 获取两个点集的y范围
    top_ys = [y for (x, y) in top_set]
    bottom_ys = [y for (x, y) in bottom_set]
    
    top_min_y, top_max_y = min(top_ys), max(top_ys)
    bottom_min_y, bottom_max_y = min(bottom_ys), max(bottom_ys)
    
    # 检查是否有重叠行
    if top_max_y < bottom_min_y or bottom_max_y < top_min_y:
        return False  # 没有重叠行
    
    # 计算重叠行范围
    overlap_start = max(top_min_y, bottom_min_y)
    overlap_end = min(top_max_y, bottom_max_y)
    
    # 从向下生长的点集（top_set）中取出重叠行的所有点
    down_points_in_overlap = {(x, y) for (x, y) in top_set 
                             if overlap_start <= y <= overlap_end}
    
    # 检查这些点有多少在向上生长的点集（bottom_set）中
    common_count = 0
    for point in down_points_in_overlap:
        if point in bottom_set:
            common_count += 1
    
    # 阈值14：基于50%字符覆盖率和2行重叠的假设
    return common_count >= 14