#path_generationSafe9.py
# 10-30日更新 1 迷宫探测法找不到路径或超长时，生成垂直路径 2，添加一条基于投影法的路径 3. 添加路径标识和投票数 
# 11-27 删除没用的路径合并函数
# 12-1日，添加了A*无偏置路径作为第三条路径,实现了A*左，右偏置和无偏置三种路径，完成了穿透、无穿透，偏置无偏置的完整对比
#1-12日发现修改。如果在偏置范围内没有找到目标点，扩大搜索范围
import numpy as np
import heapq


# 添加路径长度限制和内存保护
MAX_PATH_LENGTH = 10000  # 设置最大路径长度限制
PIERCE_COST = 9  # 穿透代价值
BYPASS_RATIO = 0.8  # 绕行阈值比例

def generate_center_path(flag):
    """在图像正中间生成一条垂直路径"""
    height, width = flag.shape
    center_x = width // 2
    path = []
    
    for y in range(height):
        path.append((center_x, y))
    
    return path

def generate_vertical_projection_path(flag):
    """
    基于垂直投影最小值的传统切割方案
    这是前人在字符切割中常用的经典方法
    路径标识 7，投票数 1
    """
    height, width = flag.shape
    
    if height == 0 or width == 0:
        return []
    
    # 定义x_range范围
    x_range = (13, 27)
    
    # 计算垂直投影 - 投影值越小表示该列字符像素越少
    # 注意：flag中1表示可通行区域，0表示障碍
    flag_np = flag.numpy() if hasattr(flag, 'numpy') else flag
    vertical_projection = np.sum(flag_np, axis=0)
    #vertical_projection = np.sum(flag, axis=0)  # 对每列求和，值越大表示可通行像素越多
    
    # 只在指定范围内寻找投影最小值
    restricted_projection = vertical_projection[x_range[0]:x_range[1]+1]
    
    # 找到限制范围内的最小值
    min_value = np.min(restricted_projection)
    
    # 获取限制范围内所有最小值的位置（相对于整个数组的索引）
    min_positions_in_range = np.where(vertical_projection[x_range[0]:x_range[1]+1] == min_value)[0] + x_range[0]
    
    # 选择最靠近中心的位置
    center_x = 20  # 使用固定中心值20
    best_cut_x = min_positions_in_range[np.argmin(np.abs(min_positions_in_range - center_x))]
    
    # 生成垂直切割路径
    path = []
    for y in range(height):
        path.append((best_cut_x, y))
    #print(path)
    #需要加入路径标识和投票数功能
    return {
        "path": path,
        "identifier": 7,
        "vote_value": 1
    }

#该函数不在做路径去重和合并，直接返回7条路径
def generate_cut_lines(flag, denoised_image, max_paths=7):
    """
    生成7条主要切割路径：
    1. 从上向下，靠左优势 (A*) 1                                 Y
    2. 从上向下，靠右优势 (A*) 2                                 Y
    3. 从下向上，靠左优势 (A*) 3 （修改为无偏置的A*路径）          Y
    4. 从下向上，靠右优势 (A*) 4 （不再采用）
    5. 从顶部到底部 (Dijkstra) 5                                 Y  
    6. 从底部到顶部 (Dijkstra) 6 （不再采用）
    7. 基于垂直投影最小的路径 7                                   Y       
    """
    height, width = flag.shape
    
    if height == 0 or width == 0:
        return []
    
    # 计算图像的中心列
    center_x = width // 2
    
    # 生成6条主要路径
    base_paths = []
    
    # 1-4. 4条A*路径
    # 1. 从上向下生长，靠左优势
    path1 = safe_generate_biased_path(flag, "top", "left", center_x)
    if not path1:
        path1 = generate_center_path(flag) #已经确认，找不到路径时返回空列表
    base_paths.append({
        "path": path1,
        "identifier": 1,
        "vote_value": 1
    })
    
    # 2. 从上向下生长，靠右优势
    path2 = safe_generate_biased_path(flag, "top", "right", center_x)
    if not path2:
        path2 = generate_center_path(flag)
    base_paths.append({
        "path": path2,
        "identifier": 2,
        "vote_value": 1
    })
    
    #3.从上向下生长，无左右偏置的A*路径
    path3 = safe_generate_unbiased_path(flag, "top", center_x)
    if not path3:
        path3 = generate_center_path(flag)
    base_paths.append({
        "path": path3,
        "identifier": 3,
        "vote_value": 1
    })
    # 5. 从顶部到底部的Dijkstra路径
    path5 = safe_generate_dijkstra_cut_lines(flag)
    if not path5:
        path5 = generate_center_path(flag)
    base_paths.append({
        "path": path5,
        "identifier": 5,
        "vote_value": 1
    })
    
    path7=generate_vertical_projection_path(denoised_image)  # 添加基于垂直投影的路径
    base_paths.append(path7)
    return base_paths

def safe_generate_biased_path(flag, start_position, bias, center_x):
    """安全的偏置路径生成函数，包含错误处理"""
    try:
        return generate_biased_path(flag, start_position, bias, center_x)
    except (MemoryError, RecursionError) as e:
        print(f"生成偏置路径时发生错误 ({start_position}, {bias}): {e}")
        return None

def safe_generate_dijkstra_cut_lines(flag):
    """安全的Dijkstra路径生成函数，包含错误处理"""
    try:
        return generate_dijkstra_cut_lines(flag)
    except (MemoryError, RecursionError) as e:
        print(f"生成Dijkstra路径时发生错误: {e}")
        return None

def safe_generate_dijkstra_cut_lines_bottom_up(flag):
    """安全的自底向上Dijkstra路径生成函数，包含错误处理"""
    try:
        return generate_dijkstra_cut_lines_bottom_up(flag)
    except (MemoryError, RecursionError) as e:
        print(f"生成自底向上Dijkstra路径时发生错误: {e}")
        return None

# 修改A*算法，添加路径长度检查
def find_biased_path_a_star(flag, start, start_position, bias, center_x):
    """使用A*算法寻找带偏置的路径"""
    height, width = flag.shape
    
    # 确定目标位置
    if start_position == "top":
        goal_y = height - 1
    else:  # bottom
        goal_y = 0
    
    # 根据偏置确定目标x范围
    if bias == "left":
        goal_x_range = range(max(0, center_x-5), min(width, center_x))
    else:  # right
        goal_x_range = range(center_x, min(width, center_x+5))
    
    # 寻找目标点
    goal_points = []
    for x in goal_x_range:
        if flag[goal_y, x] == 1:
            goal_points.append((x, goal_y))
    #原先代码对于4邻域，无法处理
    #1-12日发现修改。如果在偏置范围内没有找到目标点，扩大搜索范围
    if not goal_points:
        # 搜索整个底部行
        for x in range(width):
            if flag[goal_y, x] == 1:
                goal_points.append((x, goal_y))

    if not goal_points:
        # 如果没有找到目标点，使用强制贯穿
        return generate_forced_path(flag, start_position, bias, center_x)
    
    # 选择目标点
    if bias == "left":
        goal = min(goal_points, key=lambda p: p[0])
    else:  # right
        goal = max(goal_points, key=lambda p: p[0])
    
    # A*算法
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    iteration_count = 0
    max_iterations = height * width * 2  # 防止无限循环
    
    while open_set and iteration_count < max_iterations:
        iteration_count += 1
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # 重建路径，添加长度检查
            path = []
            path_set = set()  # 用于检测重复点
            
            while current in came_from:
                if len(path) >= MAX_PATH_LENGTH:
                    print("警告: 路径长度超过限制，截断路径")
                    break
                if current in path_set:
                    print("警告: 检测到路径循环，截断路径")
                    break
                    
                path.append(current)
                path_set.add(current)
                prev_current = current  # 保存当前点
                current = came_from[current]
                if current and (abs(current[1] - prev_current[1]) > 1 or abs(current[0] - prev_current[0]) > 1):
                    mid_x = (current[0] + prev_current[0]) // 2
                    mid_y = (current[1] + prev_current[1]) // 2
                    path.append((mid_x, mid_y))
            path.append(start)
            path.reverse()
            
            # 检查路径是否合理
            if len(path) > MAX_PATH_LENGTH:
                print(f"警告: 路径过长 ({len(path)} points)，返回空路径")
                return []
                
            return path
        
        # 检查是否需要贯穿
        if should_pierce_through(flag, current, start_position, goal_y):
            # 执行贯穿
            pierce_point = get_pierce_point(flag, current, start_position, goal_y)
            if pierce_point:
                # 直接连接到贯穿点
                came_from[pierce_point] = current
                g_score[pierce_point] = g_score[current] + 4  # 贯穿代价稍高
                f_score[pierce_point] = g_score[pierce_point] + heuristic(pierce_point, goal)
                heapq.heappush(open_set, (f_score[pierce_point], pierce_point))
                continue  # 跳过常规邻居检查
        
        # 8个方向
        for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height and flag[neighbor[1], neighbor[0]] == 1:
                # 基础代价
                base_cost = 1
                
                # 根据偏置调整代价
                if start_position == "top":
                    # 从上向下生长
                    if dy > 0:  # 向下移动
                        cost_multiplier = 0.8  # 鼓励向下
                    elif dy < 0:  # 向上移动
                        cost_multiplier = 1.5  # 不鼓励向上，但允许
                    else:  # 水平移动
                        cost_multiplier = 1.0
                else:
                    # 从下向上生长
                    if dy < 0:  # 向上移动
                        cost_multiplier = 0.8  # 鼓励向上
                    elif dy > 0:  # 向下移动
                        cost_multiplier = 1.5  # 不鼓励向下，但允许
                    else:  # 水平移动
                        cost_multiplier = 1.0
                
                # 根据左右偏置调整代价
                if bias == "left":
                    if dx < 0:  # 向左移动
                        cost_multiplier *= 0.9  # 鼓励向左
                    elif dx > 0:  # 向右移动
                        cost_multiplier *= 1.1  # 不鼓励向右
                else:  # right
                    if dx > 0:  # 向右移动
                        cost_multiplier *= 0.9  # 鼓励向右
                    elif dx < 0:  # 向左移动
                        cost_multiplier *= 1.1  # 不鼓励向左
                
                tentative_g_score = g_score[current] + base_cost * cost_multiplier
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # 如果A*找不到路径或超过迭代次数，使用强制贯穿
    if iteration_count >= max_iterations:
        print("警告: A*算法超过最大迭代次数，使用强制贯穿")
    
    return generate_forced_path(flag, start_position, bias, center_x)

# 修改Dijkstra算法，添加迭代限制
def generate_dijkstra_cut_lines(flag):
    """
    基于flag结构生成从顶部到底部的最佳路径（使用Dijkstra算法）
    """
    height, width = flag.shape
    
    # 检查输入是否有效
    if height == 0 or width == 0:
        return []
    
    # 定义起点和终点的y坐标
    start_y = 0
    end_y = height - 1
    
    # 初始化起点集合（所有y=0且flag为1的点）
    start_points = []
    for x in range(width):
        if flag[start_y, x] == 1:
            start_points.append((x, start_y))
    
    # 如果没有起点，无法找到路径
    if not start_points:
        return []
    
    # 初始化终点集合（所有y=height-1且flag为1的点）
    end_points = set()
    for x in range(width):
        if flag[end_y, x] == 1:
            end_points.add((x, end_y))
    
    # 如果没有终点，无法找到路径
    if not end_points:
        return []
    
    # Dijkstra算法实现
    distance = {}  # 记录每个点到起点的最短距离
    previous = {}  # 记录路径
    heap = []  # 优先队列
    
    # 初始化起点的距离为0，并加入优先队列
    for x, y in start_points:
        distance[(x, y)] = 0
        previous[(x, y)] = None
        heapq.heappush(heap, (0, x, y))
    
    # 定义可能的移动方向（8个方向）
    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),          (1, 0),
        (-1, 1),  (0, 1),  (1, 1)
    ]
    
    found = False
    end_node = None
    iteration_count = 0
    max_iterations = height * width * 2  # 防止无限循环
    
    while heap and iteration_count < max_iterations:
        iteration_count += 1
        dist, x, y = heapq.heappop(heap)
        
        # 如果到达终点，结束搜索
        if (x, y) in end_points:
            end_node = (x, y)
            found = True
            break
        
        # 如果当前距离大于已记录的最短距离，跳过
        if dist > distance.get((x, y), float('inf')):
            continue
        
        # 尝试所有可能的移动方向
        for dx, dy in directions:
            new_x = x + dx
            new_y = y + dy
            
            # 检查是否在边界内且是可通行路径
            if 0 <= new_x < width and 0 <= new_y < height and flag[new_y, new_x] == 1:
                new_dist = dist + 1  # 假设所有移动的权重都是1
                
                # 如果找到更短的路径，更新距离和前驱
                if new_dist < distance.get((new_x, new_y), float('inf')):
                    distance[(new_x, new_y)] = new_dist
                    previous[(new_x, new_y)] = (x, y)
                    heapq.heappush(heap, (new_dist, new_x, new_y))
    
    if iteration_count >= max_iterations:
        print("警告: Dijkstra算法超过最大迭代次数")
        return []
    
    # 如果没有找到路径，返回空列表
    if not found:
        return []
    
    # 构建路径，添加长度检查
    path = []
    current = end_node
    path_set = set()
    
    while current is not None and len(path) < MAX_PATH_LENGTH:
        if current in path_set:
            print("警告: 检测到路径循环，截断路径")
            break
        path.append(current)
        path_set.add(current)
        current = previous.get(current)
    
    # 反转路径，使其从起点到终点
    path.reverse()
    
    if len(path) > MAX_PATH_LENGTH:
        print(f"警告: Dijkstra路径过长 ({len(path)} points)，返回空路径")
        return []
    
    return path

# 同样修改 generate_dijkstra_cut_lines_bottom_up 函数，添加相同的保护机制
def generate_dijkstra_cut_lines_bottom_up(flag):
    """
    基于flag结构生成从底部到顶部的最佳路径（使用Dijkstra算法）
    """
    height, width = flag.shape
    
    # 检查输入是否有效
    if height == 0 or width == 0:
        return []
    
    # 定义起点和终点的y坐标（从底部到顶部）
    start_y = height - 1  # 底部作为起点
    end_y = 0  # 顶部作为终点
    
    # 初始化起点集合（所有y=height-1且flag为1的点）
    start_points = []
    for x in range(width):
        if flag[start_y, x] == 1:
            start_points.append((x, start_y))
    
    # 如果没有起点，无法找到路径
    if not start_points:
        return []
    
    # 初始化终点集合（所有y=0且flag为1的点）
    end_points = set()
    for x in range(width):
        if flag[end_y, x] == 1:
            end_points.add((x, end_y))
    
    # 如果没有终点，无法找到路径
    if not end_points:
        return []
    
    # Dijkstra算法实现
    distance = {}  # 记录每个点到起点的最短距离
    previous = {}  # 记录路径
    heap = []  # 优先队列
    
    # 初始化起点的距离为0，并加入优先队列
    for x, y in start_points:
        distance[(x, y)] = 0
        previous[(x, y)] = None
        heapq.heappush(heap, (0, x, y))
    
    # 定义可能的移动方向（8个方向）
    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),          (1, 0),
        (-1, 1),  (0, 1),  (1, 1)
    ]
    
    found = False
    end_node = None
    iteration_count = 0
    max_iterations = height * width * 2  # 防止无限循环
    
    while heap and iteration_count < max_iterations:
        iteration_count += 1
        dist, x, y = heapq.heappop(heap)
        
        # 如果到达终点，结束搜索
        if (x, y) in end_points:
            end_node = (x, y)
            found = True
            break
        
        # 如果当前距离大于已记录的最短距离，跳过
        if dist > distance.get((x, y), float('inf')):
            continue
        
        # 尝试所有可能的移动方向
        for dx, dy in directions:
            new_x = x + dx
            new_y = y + dy
            
            # 检查是否在边界内且是可通行路径
            if 0 <= new_x < width and 0 <= new_y < height and flag[new_y, new_x] == 1:
                # 基础移动代价
                base_cost = 1
                
                # 可以在这里添加偏置逻辑，鼓励向上移动
                if dy < 0:  # 向上移动
                    cost_multiplier = 0.9  # 鼓励向上
                elif dy > 0:  # 向下移动
                    cost_multiplier = 1.1  # 轻微惩罚向下
                else:  # 水平移动
                    cost_multiplier = 1.0
                
                new_dist = dist + base_cost * cost_multiplier
                
                # 如果找到更短的路径，更新距离和前驱
                if new_dist < distance.get((new_x, new_y), float('inf')):
                    distance[(new_x, new_y)] = new_dist
                    previous[(new_x, new_y)] = (x, y)
                    heapq.heappush(heap, (new_dist, new_x, new_y))
    
    if iteration_count >= max_iterations:
        print("警告: 自底向上Dijkstra算法超过最大迭代次数")
        return []
    
    # 如果没有找到路径，返回空列表
    if not found:
        return []
    
    # 构建路径，添加长度检查
    path = []
    current = end_node
    path_set = set()
    
    while current is not None and len(path) < MAX_PATH_LENGTH:
        if current in path_set:
            print("警告: 检测到路径循环，截断路径")
            break
        path.append(current)
        path_set.add(current)
        current = previous.get(current)
    
    # 反转路径，使其从起点（底部）到终点（顶部）
    path.reverse()
    
    if len(path) > MAX_PATH_LENGTH:
        print(f"警告: 自底向上Dijkstra路径过长 ({len(path)} points)，返回空路径")
        return []
    
    return path

# 其余函数保持不变...
# [保持原有的 generate_biased_path, should_pierce_through, get_pierce_point, 
#  generate_forced_path, is_path_connected, is_vertically_connected, 
#  repair_disconnected_path, find_connecting_points, heuristic 函数不变]

def find_connecting_points(point1, point2, denoised_image):
    """在两个点之间寻找连通路径"""
    x1, y1 = point1
    x2, y2 = point2
    
    # 简单的直线插值
    if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
        return []
    
    # 尝试垂直-水平路径
    intermediate1 = [(x1, y2), (x2, y2)]
    if is_path_connected([point1] + intermediate1, denoised_image):
        return intermediate1
    
    # 尝试水平-垂直路径
    intermediate2 = [(x2, y1), (x2, y2)]
    if is_path_connected([point1] + intermediate2, denoised_image):
        return intermediate2
    
    return []

def heuristic(a, b):
    """A*算法的启发式函数"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def repair_disconnected_path(path, denoised_image):
    """修复不连通的路径"""
    if len(path) < 2:
        return path
    
    repaired_path = [path[0]]
    
    for i in range(len(path) - 1):
        current = path[i]
        next_point = path[i+1]
        
        # 如果当前点和下一点不连通，尝试找到连通路径
        if not is_vertically_connected(current[0], current[1], next_point[0], next_point[1], denoised_image):
            # 尝试找到中间点使路径连通
            intermediate_points = find_connecting_points(current, next_point, denoised_image)
            if intermediate_points:
                repaired_path.extend(intermediate_points)
            else:
                # 如果找不到连通路径，保持原路径
                repaired_path.append(next_point)
        else:
            repaired_path.append(next_point)
    
    return repaired_path

def is_vertically_connected(x1, y1, x2, y2, denoised_image):
    """检查两个点是否在垂直方向上连通"""
    if y1 == y2:  # 同一行
        start_x, end_x = min(x1, x2), max(x1, x2)
        for x in range(start_x, end_x + 1):
            if denoised_image[y1, x] != 0:
                return False
        return True
    elif x1 == x2:  # 同一列
        start_y, end_y = min(y1, y2), max(y1, y2)
        for y in range(start_y, end_y + 1):
            if denoised_image[y, x1] != 0:
                return False
        return True
    else:  # 对角线
        # 检查两条可能的路径
        path1_connected = True
        path2_connected = True
        
        # 路径1: 先水平后垂直
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if denoised_image[y1, x] != 0:
                path1_connected = False
                break
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if denoised_image[y, x2] != 0:
                path1_connected = False
                break
        
        # 路径2: 先垂直后水平
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if denoised_image[y, x1] != 0:
                path2_connected = False
                break
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if denoised_image[y2, x] != 0:
                path2_connected = False
                break
        
        return path1_connected or path2_connected

def is_path_connected(path, denoised_image):
    """检查路径是否连通（相邻点之间都是0值）"""
    if len(path) < 2:
        return True
    
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        
        # 检查相邻点之间的像素值
        if abs(x1 - x2) > 1 or abs(y1 - y2) > 1:
            return False
        
        # 检查路径点本身是否在连通域内
        if denoised_image[y1, x1] != 0 or denoised_image[y2, x2] != 0:
            return False
    
    return True

def generate_forced_path(flag, start_position, bias, center_x):
    """生成强制贯穿的路径"""
    height, width = flag.shape
    
    # 确定起点和终点
    if start_position == "top":
        start_y = 0
        end_y = height - 1
    else:  # bottom
        start_y = height - 1
        end_y = 0
    
    # 根据偏置确定x坐标
    if bias == "left":
        x = max(0, center_x - 5)
    else:  # right
        x = min(width - 1, center_x + 5)
    
    # 创建直线路径
    path = []
    if start_position == "top":
        for y in range(start_y, end_y + 1):
            path.append((x, y))
    else:
        for y in range(start_y, end_y - 1, -1):
            path.append((x, y))
    
    return path

def get_pierce_point(flag, current, start_position, goal_y):
    """获取贯穿点"""
    x, y = current
    height, width = flag.shape
    
    if start_position == "top":
        # 向下贯穿
        target_y = y + 2
        if target_y >= height:  # 如果下下一行是最后一行
            target_y = height - 1
        
        # 在目标行寻找最佳贯穿点
        best_point = None
        min_distance = float('inf')
        
        for offset in range(-2, 3):
            target_x = x + offset
            if 0 <= target_x < width and flag[target_y, target_x] == 1:
                # 选择距离当前点最近的点
                distance = abs(offset)
                if distance < min_distance:
                    min_distance = distance
                    best_point = (target_x, target_y)
        
        return best_point
    
    else:  # bottom，向上贯穿
        target_y = y - 2
        if target_y < 0:  # 如果上上一行是第一行
            target_y = 0
        
        # 在目标行寻找最佳贯穿点
        best_point = None
        min_distance = float('inf')
        
        for offset in range(-2, 3):
            target_x = x + offset
            if 0 <= target_x < width and flag[target_y, target_x] == 1:
                # 选择距离当前点最近的点
                distance = abs(offset)
                if distance < min_distance:
                    min_distance = distance
                    best_point = (target_x, target_y)
        
        return best_point

def should_pierce_through(flag, current, start_position, goal_y):
    """判断是否应该贯穿，加入代价比较"""
    x, y = current
    height, width = flag.shape
    
    # 确定移动方向
    if start_position == "top":
        # 向下移动
        next_y = y + 1
        if next_y >= height:  # 已经是最后一行
            return False
        
        # 检查下一行的三个位置是否都是障碍
        positions_to_check = [
            (x-1, next_y),  # 左下方
            (x, next_y),    # 正下方
            (x+1, next_y)   # 右下方
        ]
        
        all_blocked = True
        for pos_x, pos_y in positions_to_check:
            if 0 <= pos_x < width and 0 <= pos_y < height:
                if flag[pos_y, pos_x] == 1:  # 有通路
                    all_blocked = False
                    break
        
        if not all_blocked:
            return False
        
        # =========== 新增：检查绕行代价 ===========
        # 估算向左右绕行的最小代价
        #pierce_cost = 9  # 当前穿透代价
        max_bypass_for_pierce = PIERCE_COST * BYPASS_RATIO    # 绕行阈值：9*0.8=7.2
        
        min_bypass_cost = float('inf')
        
        # 检查左右各4格的绕行可能性
        for dx in range(-4, 5):
            if dx == 0:
                continue
            
            test_x = x + dx
            if 0 <= test_x < width and flag[y, test_x] == 1:
                # 检查从这个点向下是否畅通（检查3行）
                clear_path = True
                for dy in range(1, 4):
                    test_y = y + dy
                    if test_y < height and flag[test_y, test_x] == 0:
                        clear_path = False
                        break
                
                if clear_path:
                    # 绕行代价：水平移动 + 轻微惩罚
                    bypass_cost = abs(dx) * 1.5
                    if bypass_cost < min_bypass_cost:
                        min_bypass_cost = bypass_cost
        
        # 如果找到代价较低的绕行路径，就不穿透
        if min_bypass_cost <= max_bypass_for_pierce:
            return False
        # =========== 新增代码结束 ===========
        
        # 检查下下一行的情况
        next_next_y = y + 2
        if next_next_y >= height:  # 下下一行是最后一行，允许贯穿
            return True
        
        # 检查下下一行的5个位置是否有连通域
        positions_to_check_next = [
            (x-2, next_next_y),
            (x-1, next_next_y),
            (x, next_next_y),
            (x+1, next_next_y),
            (x+2, next_next_y)
        ]
        
        for pos_x, pos_y in positions_to_check_next:
            if 0 <= pos_x < width and 0 <= pos_y < height:
                if flag[pos_y, pos_x] == 1:  # 有连通域
                    return True
        
        return False
    
    else:  # bottom，向上移动
        # ... 类似的逻辑，对称处理 ...
        next_y = y - 1
        if next_y < 0:  # 已经是第一行
            return False
        
        # 检查上一行的三个位置是否都是障碍
        positions_to_check = [
            (x-1, next_y),  # 左上方
            (x, next_y),    # 正上方
            (x+1, next_y)   # 右上方
        ]
        
        all_blocked = True
        for pos_x, pos_y in positions_to_check:
            if 0 <= pos_x < width and 0 <= pos_y < height:
                if flag[pos_y, pos_x] == 1:  # 有通路
                    all_blocked = False
                    break
        
        if not all_blocked:
            return False
        
        # =========== 新增：检查绕行代价（向上） ===========
        #pierce_cost = 9
        #max_bypass_for_pierce = PIERCE_COST  * BYPASS_RATIO 
        
        min_bypass_cost = float('inf')
        
        for dx in range(-4, 5):
            if dx == 0:
                continue
            
            test_x = x + dx
            if 0 <= test_x < width and flag[y, test_x] == 1:
                # 检查从这个点向上是否畅通（检查3行）
                clear_path = True
                for dy in range(1, 4):
                    test_y = y - dy
                    if test_y >= 0 and flag[test_y, test_x] == 0:
                        clear_path = False
                        break
                
                if clear_path:
                    bypass_cost = abs(dx) * 1.5
                    if bypass_cost < min_bypass_cost:
                        min_bypass_cost = bypass_cost
        
        if min_bypass_cost <= max_bypass_for_pierce:
            return False
        # =========== 新增代码结束 ===========
        
        # 检查上上一行的情况
        next_next_y = y - 2
        if next_next_y < 0:  # 上上一行是第一行，允许贯穿
            return True
        
        # 检查上上一行的5个位置是否有连通域
        positions_to_check_next = [
            (x-2, next_next_y),
            (x-1, next_next_y),
            (x, next_next_y),
            (x+1, next_next_y),
            (x+2, next_next_y)
        ]
        
        for pos_x, pos_y in positions_to_check_next:
            if 0 <= pos_x < width and 0 <= pos_y < height:
                if flag[pos_y, pos_x] == 1:  # 有连通域
                    return True
        
        return False

def should_pierce_through1(flag, current, start_position, goal_y):
    """判断是否应该贯穿"""
    x, y = current
    height, width = flag.shape
    
    # 确定移动方向
    if start_position == "top":
        # 向下移动
        next_y = y + 1
        if next_y >= height:  # 已经是最后一行
            return False
        
        # 检查下一行的三个位置是否都是障碍
        positions_to_check = [
            (x-1, next_y),  # 左下方
            (x, next_y),    # 正下方
            (x+1, next_y)   # 右下方
        ]
        
        all_blocked = True
        for pos_x, pos_y in positions_to_check:
            if 0 <= pos_x < width and 0 <= pos_y < height:
                if flag[pos_y, pos_x] == 1:  # 有通路
                    all_blocked = False
                    break
        
        if not all_blocked:
            return False
        
        # 检查下下一行的情况
        next_next_y = y + 2
        if next_next_y >= height:  # 下下一行是最后一行，允许贯穿
            return True
        
        # 检查下下一行的5个位置是否有连通域
        positions_to_check_next = [
            (x-2, next_next_y),
            (x-1, next_next_y),
            (x, next_next_y),
            (x+1, next_next_y),
            (x+2, next_next_y)
        ]
        
        for pos_x, pos_y in positions_to_check_next:
            if 0 <= pos_x < width and 0 <= pos_y < height:
                if flag[pos_y, pos_x] == 1:  # 有连通域
                    return True
        
        return False
    
    else:  # bottom，向上移动
        next_y = y - 1
        if next_y < 0:  # 已经是第一行
            return False
        
        # 检查上一行的三个位置是否都是障碍
        positions_to_check = [
            (x-1, next_y),  # 左上方
            (x, next_y),    # 正上方
            (x+1, next_y)   # 右上方
        ]
        
        all_blocked = True
        for pos_x, pos_y in positions_to_check:
            if 0 <= pos_x < width and 0 <= pos_y < height:
                if flag[pos_y, pos_x] == 1:  # 有通路
                    all_blocked = False
                    break
        
        if not all_blocked:
            return False
        
        # 检查上上一行的情况
        next_next_y = y - 2
        if next_next_y < 0:  # 上上一行是第一行，允许贯穿
            return True
        
        # 检查上上一行的5个位置是否有连通域
        positions_to_check_next = [
            (x-2, next_next_y),
            (x-1, next_next_y),
            (x, next_next_y),
            (x+1, next_next_y),
            (x+2, next_next_y)
        ]
        
        for pos_x, pos_y in positions_to_check_next:
            if 0 <= pos_x < width and 0 <= pos_y < height:
                if flag[pos_y, pos_x] == 1:  # 有连通域
                    return True
        
        return False


def generate_biased_path(flag, start_position, bias, center_x):
    """生成带偏置的路径"""
    height, width = flag.shape
    
    # 确定起点和主要方向
    if start_position == "top":
        start_y = 0
        main_direction = 1  # 向下
    else:  # bottom
        start_y = height - 1
        main_direction = -1  # 向上
    
    # 根据偏置和中心点确定起点
    if bias == "left":
        start_x = max(0, center_x - 3)
    else:  # right
        start_x = min(width - 1, center_x + 3)
    
    # 在起点附近寻找有效的起始点
    start_points = []
    for x in range(max(0, start_x-2), min(width, start_x+3)):
        if flag[start_y, x] == 1:
            start_points.append((x, start_y))
    
    if not start_points:
        # 如果没有找到有效起点，使用强制贯穿
        return generate_forced_path(flag, start_position, bias, center_x)
    
    # 选择起始点
    if bias == "left":
        start_point = min(start_points, key=lambda p: p[0])
    else:  # right
        start_point = max(start_points, key=lambda p: p[0])
    
    # 使用A*算法生成路径，带偏置
    return find_biased_path_a_star(flag, start_point, start_position, bias, center_x)

def generate_unbiased_path_with_piercing(flag, start_position, center_x):
    """生成没有左右方向偏向但带穿透的A*路径"""
    height, width = flag.shape
    
    # 确定起点和主要方向
    if start_position == "top":
        start_y = 0
        main_direction = 1  # 向下
    else:  # bottom
        start_y = height - 1
        main_direction = -1  # 向上
    
    # 使用中心点作为起点
    start_x = center_x
    
    # 在起点附近寻找有效的起始点
    start_points = []
    for x in range(max(0, start_x-2), min(width, start_x+3)):
        if flag[start_y, x] == 1:
            start_points.append((x, start_y))
    
    if not start_points:
        # 如果没有找到有效起点，使用强制贯穿
        return generate_forced_unbiased_path(flag, start_position, center_x)
    
    # 选择最靠近中心点的起始点
    start_point = min(start_points, key=lambda p: abs(p[0] - center_x))
    
    # 使用A*算法生成路径，不带左右偏置但带穿透
    return find_unbiased_path_a_star(flag, start_point, start_position, center_x)

def find_unbiased_path_a_star(flag, start, start_position, center_x):
    """使用A*算法寻找不带左右偏置但带穿透的路径"""
    height, width = flag.shape
    
    # 确定目标位置
    if start_position == "top":
        goal_y = height - 1
    else:  # bottom
        goal_y = 0
    
    # 目标x范围为中心点附近
    goal_x_range = range(max(0, center_x-2), min(width, center_x+3))
    
    # 寻找目标点
    goal_points = []
    for x in goal_x_range:
        if flag[goal_y, x] == 1:
            goal_points.append((x, goal_y))
    
    #原先代码对于4邻域，无法处理
    #1-12日发现修改。如果在偏置范围内没有找到目标点，扩大搜索范围
    if not goal_points:
        # 搜索整个底部行
        for x in range(width):
            if flag[goal_y, x] == 1:
                goal_points.append((x, goal_y))


    if not goal_points:
        # 如果没有找到目标点，使用强制贯穿
        return generate_forced_unbiased_path(flag, start_position, center_x)
    
    # 选择最靠近中心点的目标点
    goal = min(goal_points, key=lambda p: abs(p[0] - center_x))
    
    # A*算法
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    iteration_count = 0
    max_iterations = height * width * 2  # 防止无限循环
    
    while open_set and iteration_count < max_iterations:
        iteration_count += 1
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # 重建路径，添加长度检查
            path = []
            path_set = set()  # 用于检测重复点
            
            while current in came_from:
                if len(path) >= MAX_PATH_LENGTH:
                    print("警告: 路径长度超过限制，截断路径")
                    break
                if current in path_set:
                    print("警告: 检测到路径循环，截断路径")
                    break
                    
                path.append(current)
                path_set.add(current)
                prev_current = current  # 保存当前点
                current = came_from[current]
                if current and (abs(current[1] - prev_current[1]) > 1 or abs(current[0] - prev_current[0]) > 1):
                    mid_x = (current[0] + prev_current[0]) // 2
                    mid_y = (current[1] + prev_current[1]) // 2
                    path.append((mid_x, mid_y))
            path.append(start)
            path.reverse()
            
            # 检查路径是否合理
            if len(path) > MAX_PATH_LENGTH:
                print(f"警告: 路径过长 ({len(path)} points)，返回空路径")
                return []
                
            return path
        
        # 检查是否需要贯穿（使用原有的贯穿逻辑）
        if should_pierce_through(flag, current, start_position, goal_y):
            # 执行贯穿
            pierce_point = get_pierce_point(flag, current, start_position, goal_y)
            if pierce_point:
                # 直接连接到贯穿点
                came_from[pierce_point] = current
                g_score[pierce_point] = g_score[current] + PIERCE_COST   # 贯穿代价稍高
                f_score[pierce_point] = g_score[pierce_point] + heuristic(pierce_point, goal)
                heapq.heappush(open_set, (f_score[pierce_point], pierce_point))
                continue  # 跳过常规邻居检查
        
        # 8个方向
        for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height and flag[neighbor[1], neighbor[0]] == 1:
                # 基础代价
                base_cost = 1
                
                # 根据上下方向调整代价（保持原有的上下方向偏置）
                if start_position == "top":
                    # 从上向下生长
                    if dy > 0:  # 向下移动
                        cost_multiplier = 0.8  # 鼓励向下
                    elif dy < 0:  # 向上移动
                        cost_multiplier = 1.5  # 不鼓励向上，但允许
                    else:  # 水平移动
                        cost_multiplier = 1.0
                else:
                    # 从下向上生长
                    if dy < 0:  # 向上移动
                        cost_multiplier = 0.8  # 鼓励向上
                    elif dy > 0:  # 向下移动
                        cost_multiplier = 1.5  # 不鼓励向下，但允许
                    else:  # 水平移动
                        cost_multiplier = 1.0
                
                # 移除左右偏置调整，保持水平移动代价不变
                
                tentative_g_score = g_score[current] + base_cost * cost_multiplier
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # 如果A*找不到路径或超过迭代次数，使用强制贯穿
    if iteration_count >= max_iterations:
        print("警告: A*算法超过最大迭代次数，使用强制贯穿")
    
    return generate_forced_unbiased_path(flag, start_position, center_x)

def generate_forced_unbiased_path(flag, start_position, center_x):
    """生成不带左右偏置的强制贯穿路径"""
    height, width = flag.shape
    
    # 确定起点和终点
    if start_position == "top":
        start_y = 0
        end_y = height - 1
    else:  # bottom
        start_y = height - 1
        end_y = 0
    
    # 使用中心点作为x坐标
    x = center_x
    
    # 创建直线路径
    path = []
    if start_position == "top":
        for y in range(start_y, end_y + 1):
            path.append((x, y))
    else:
        for y in range(start_y, end_y - 1, -1):
            path.append((x, y))
    
    return path

def safe_generate_unbiased_path(flag, start_position, center_x):
    """安全的无偏置路径生成函数，包含错误处理"""
    try:
        return generate_unbiased_path_with_piercing(flag, start_position, center_x)
    except (MemoryError, RecursionError) as e:
        print(f"生成无偏置路径时发生错误 ({start_position}): {e}")
        return None