"""
resolution_scaling_benchmark.py
================================
R2#3 补充实验：分辨率扩展性 / 运行时间复杂度分析
回应 Reviewer #2 原话："How does the time complexity of the tested algorithms
scale to images with the higher resolution that would be required in
real-world scenarios?"

设计说明（请在跑之前确认，不是"跑了就对"）：
-----------------------------------------------------------------
1. 不改动 cut_line_all 里的原始 cut_region.py / path_generationSafe10.py。
   本脚本自带"可缩放版"的核心函数，逻辑与原版逐行对应，唯一区别是所有
   *空间* 常量（x_range、强制回退偏移±5、穿透探测窗口±2/±4等）按
   scale 倍数放大；*代价* 常量（base_cost=1、方向偏置0.8/1.5/0.9/1.1、
   穿透代价4/9、阈值7.2）保持不变——这些是代价单位，不是像素距离，
   放大分辨率不应该改变它们的语义。这一点如果你们有不同意见，
   请在用之前调整 SPATIAL_SCALE_MODE。

2. 测试图像用最近邻插值从现有 40x32 TDTS 样本放大，不做平滑，
   理由：代价函数依赖精确的 0 值背景判断和灰度大小比较，平滑会
   污染这个语义。

3. 只测 Table 8 已有的三个方法（Dijkstra、Left-Biased A*、Unbiased A*），
   保持和现有 Table 8 可比；Right-Biased/VPM/WA 未纳入，
   因为 Table 8 本身也没有报告它们的效率数据。

4. 每个分辨率跑 N_SAMPLES 张图（默认50，可改），记录：
   - 单图平均运行时间（秒）
   - 峰值内存（MB，用 tracemalloc）
   - 路径循环数（复用原版的循环检测逻辑，粗略计数）
   输出格式对齐现有 Table 8，追加一列 Resolution。

使用方法：
-----------------------------------------------------------------
本文件放在 src/tdts_pipelines/cut_line_all/（与 denoise.py 同目录），
然后修改下面 "===== 请修改这里 =====" 区块里的数据路径，运行：
    python resolution_scaling_benchmark.py
"""

import time
import tracemalloc
import heapq
import numpy as np
import pandas as pd
from PIL import Image

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / 'results' / 'bench'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 复用同目录下的去噪函数（本文件应放在 src/tdts_pipelines/cut_line_all/ 里运行）
try:
    from denoise import remove_salt_pepper_noise
except ImportError:
    print("[警告] 未找到 denoise.py，请把本脚本放在 src/tdts_pipelines/cut_line_all/ 目录下运行。")
    remove_salt_pepper_noise = None


# ============================================================
# ===== 请修改这里：指向你实际的 TDTS 测试图像来源 =====
# ============================================================
# 期望格式：一个函数，返回 N 张 (32, 40) 的 uint8 numpy 数组列表
# （即原始未去噪、未放大的单张 touching-digit 合成图）
def load_baseline_samples(n_samples=50):
    """Load the first n_samples images of TDTS-TS (data/tdts_dataset, MNIST idx format)."""
    import struct
    path = PROJECT_ROOT / 'data' / 'tdts_dataset' / 'MNIST' / 'raw' / 't10k-images-idx3-ubyte'
    with open(path, 'rb') as f:
        _magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return [np.array(img) for img in data[:n_samples]]


# ============================================================
# 分辨率放大（最近邻，保持像素值语义）
# ============================================================
def upscale_nearest(image_32x40, scale):
    if scale == 1:
        return image_32x40
    h, w = image_32x40.shape
    pil_img = Image.fromarray(image_32x40)
    return np.array(pil_img.resize((w * scale, h * scale), Image.NEAREST))


# ============================================================
# 可缩放版 cutting domain 构建（对应 cut_region.py::find_connected_points）
# 与原版逐行一致，唯一区别：x_range 按 scale 放大
# ============================================================
def find_connected_points_scaled(image, scale):
    height, width = image.shape
    x_range = (13 * scale, 27 * scale + (scale - 1))  # 放大走廊，含右边界补偿

    found_points = set()
    min_val = float('inf')
    min_x = None
    for x in range(x_range[0], min(x_range[1] + 1, width)):
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
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_y < height and
                            x_range[0] <= new_x <= x_range[1] and new_x < width):
                        candidates.add((new_x, new_y))
        return candidates

    while True:
        candidates = generate_candidates(found_points)
        candidates = {p for p in candidates if p not in found_points}
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
                    min_val, min_candidates = val, [(x, y)]
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

    return sorted(list(found_points), key=lambda p: p[1])


# ============================================================
# 可缩放版路径搜索：Dijkstra（对应 generate_dijkstra_cut_lines，
# 实测代价函数为 uniform base_cost=1，无方向偏置——与原代码一致，不改）
# ============================================================
def dijkstra_scaled(flag):
    height, width = flag.shape
    start_points = [(x, 0) for x in range(width) if flag[0, x] == 1]
    if not start_points:
        return [], 0
    end_points = {(x, height - 1) for x in range(width) if flag[height - 1, x] == 1}
    if not end_points:
        return [], 0

    distance = {}
    previous = {}
    heap = []
    for p in start_points:
        distance[p] = 0
        previous[p] = None
        heapq.heappush(heap, (0, p[0], p[1]))

    directions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    max_iterations = height * width * 2
    iteration_count = 0
    found = False
    end_node = None

    while heap and iteration_count < max_iterations:
        iteration_count += 1
        dist, x, y = heapq.heappop(heap)
        if (x, y) in end_points:
            end_node, found = (x, y), True
            break
        if dist > distance.get((x, y), float('inf')):
            continue
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and flag[ny, nx] == 1:
                new_dist = dist + 1  # 与原代码一致：均匀代价1，无方向偏置
                if new_dist < distance.get((nx, ny), float('inf')):
                    distance[(nx, ny)] = new_dist
                    previous[(nx, ny)] = (x, y)
                    heapq.heappush(heap, (new_dist, nx, ny))

    if not found:
        return [], iteration_count
    path = []
    current = end_node
    while current is not None:
        path.append(current)
        current = previous.get(current)
    path.reverse()
    return path, iteration_count


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ============================================================
# 可缩放版 Unbiased A*（对应 find_unbiased_path_a_star，
# 代价函数：base_cost=1 × v_mult(0.8/1.5/1.0)，穿透代价=9，
# 触发阈值 = 9*0.8=7.2 —— 全部与原代码一致，不改；
# 只放大：center_x、穿透探测/绕行检查的像素窗口
# ============================================================
PIERCE_COST = 9
BYPASS_RATIO = 0.8


def should_pierce_through_scaled(flag, current, scale):
    x, y = current
    height, width = flag.shape
    next_y = y + 1
    if next_y >= height:
        return False
    win = 1 * scale
    positions = [(x - win, next_y), (x, next_y), (x + win, next_y)]
    all_blocked = all(
        not (0 <= px < width and 0 <= py < height and flag[py, px] == 1)
        for px, py in positions
    )
    if not all_blocked:
        return False

    max_bypass_for_pierce = PIERCE_COST * BYPASS_RATIO  # 7.2，代价单位不缩放
    min_bypass_cost = float('inf')
    bypass_span = 4 * scale
    for dx in list(range(-bypass_span, 0)) + list(range(1, bypass_span + 1)):
        test_x = x + dx
        if 0 <= test_x < width and flag[y, test_x] == 1:
            clear = True
            for dy in range(1, 4 * scale):
                test_y = y + dy
                if test_y < height and flag[test_y, test_x] == 0:
                    clear = False
                    break
            if clear:
                bypass_cost = abs(dx) / scale * 1.5  # 折算回"原始像素距离"再算代价
                min_bypass_cost = min(min_bypass_cost, bypass_cost)
    if min_bypass_cost <= max_bypass_for_pierce:
        return False

    next_next_y = y + 2 * scale
    if next_next_y >= height:
        return True
    recon_span = 2 * scale
    for dx in range(-recon_span, recon_span + 1):
        px, py = x + dx, next_next_y
        if 0 <= px < width and 0 <= py < height and flag[py, px] == 1:
            return True
    return False


def get_pierce_point_scaled(flag, current, scale):
    x, y = current
    height, width = flag.shape
    target_y = min(y + 2 * scale, height - 1)
    best_point, min_distance = None, float('inf')
    win = 2 * scale
    for offset in range(-win, win + 1):
        target_x = x + offset
        if 0 <= target_x < width and flag[target_y, target_x] == 1:
            d = abs(offset)
            if d < min_distance:
                min_distance, best_point = d, (target_x, target_y)
    return best_point


def unbiased_astar_scaled(flag, scale):
    height, width = flag.shape
    center_x = width // 2
    win = 2 * scale
    start_points = [(x, 0) for x in range(max(0, center_x - win), min(width, center_x + win + 1)) if flag[0, x] == 1]
    if not start_points:
        return [], 0
    start = min(start_points, key=lambda p: abs(p[0] - center_x))

    goal_y = height - 1
    goal_points = [(x, goal_y) for x in range(max(0, center_x - win), min(width, center_x + win + 1)) if flag[goal_y, x] == 1]
    if not goal_points:
        goal_points = [(x, goal_y) for x in range(width) if flag[goal_y, x] == 1]
    if not goal_points:
        return [(center_x, y) for y in range(height)], 0
    goal = min(goal_points, key=lambda p: abs(p[0] - center_x))

    open_set = [(0, start)]
    came_from, g_score = {}, {start: 0}
    f_score = {start: heuristic(start, goal)}
    max_iterations = height * width * 2
    iteration_count = 0

    while open_set and iteration_count < max_iterations:
        iteration_count += 1
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, iteration_count

        if should_pierce_through_scaled(flag, current, scale):
            pierce_point = get_pierce_point_scaled(flag, current, scale)
            if pierce_point:
                came_from[pierce_point] = current
                g_score[pierce_point] = g_score[current] + PIERCE_COST
                f_score[pierce_point] = g_score[pierce_point] + heuristic(pierce_point, goal)
                heapq.heappush(open_set, (f_score[pierce_point], pierce_point))
                continue

        for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height and flag[neighbor[1], neighbor[0]] == 1:
                v_mult = 0.8 if dy > 0 else (1.5 if dy < 0 else 1.0)  # 与原代码一致
                tentative_g = g_score[current] + 1 * v_mult
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return [(center_x, y) for y in range(height)], iteration_count  # 强制回退


def left_biased_astar_scaled(flag, scale):
    """对应 find_biased_path_a_star(bias='left')：v_mult(0.8/1.5/1.0) x h_mult(0.9/1.1)，
    穿透代价硬编码为4（与原代码一致，不改为9）"""
    height, width = flag.shape
    center_x = width // 2
    off = 3 * scale
    start_x = max(0, center_x - off)
    win = 2 * scale
    start_points = [(x, 0) for x in range(max(0, start_x - win), min(width, start_x + win + 1)) if flag[0, x] == 1]
    if not start_points:
        return [(center_x, y) for y in range(height)], 0
    start = min(start_points, key=lambda p: p[0])

    goal_y = height - 1
    goal_win = 5 * scale
    goal_points = [(x, goal_y) for x in range(max(0, center_x - goal_win), min(width, center_x)) if flag[goal_y, x] == 1]
    if not goal_points:
        goal_points = [(x, goal_y) for x in range(width) if flag[goal_y, x] == 1]
    if not goal_points:
        return [(center_x, y) for y in range(height)], 0
    goal = min(goal_points, key=lambda p: p[0])

    open_set = [(0, start)]
    came_from, g_score = {}, {start: 0}
    f_score = {start: heuristic(start, goal)}
    max_iterations = height * width * 2
    iteration_count = 0

    while open_set and iteration_count < max_iterations:
        iteration_count += 1
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, iteration_count

        if should_pierce_through_scaled(flag, current, scale):
            pierce_point = get_pierce_point_scaled(flag, current, scale)
            if pierce_point:
                came_from[pierce_point] = current
                g_score[pierce_point] = g_score[current] + 4  # 与原代码一致：硬编码4，非9
                f_score[pierce_point] = g_score[pierce_point] + heuristic(pierce_point, goal)
                heapq.heappush(open_set, (f_score[pierce_point], pierce_point))
                continue

        for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height and flag[neighbor[1], neighbor[0]] == 1:
                v_mult = 0.8 if dy > 0 else (1.5 if dy < 0 else 1.0)
                h_mult = 0.9 if dx < 0 else (1.1 if dx > 0 else 1.0)  # left-biased
                tentative_g = g_score[current] + 1 * v_mult * h_mult
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return [(center_x, y) for y in range(height)], iteration_count


# ============================================================
# Benchmark 主流程
# ============================================================
ALGORITHMS = {
    "Dijkstra": lambda flag, scale: dijkstra_scaled(flag),
    "Unbiased A*": lambda flag, scale: unbiased_astar_scaled(flag, scale),
    "Left-Biased A*": lambda flag, scale: left_biased_astar_scaled(flag, scale),
}


def run_one(image_32x40, scale):
    """跑单张图在给定scale下的一次完整流程（去噪+cutting domain+各算法），
    返回 {算法名: (耗时秒, 峰值内存MB, 迭代数)}"""
    scaled_img = upscale_nearest(image_32x40, scale)
    if remove_salt_pepper_noise is not None:
        import torch
        denoised = remove_salt_pepper_noise(scaled_img.astype(np.float32), True)
        denoised = np.asarray(denoised).astype(np.uint8)
    else:
        denoised = scaled_img

    points = find_connected_points_scaled(denoised, scale)
    flag = np.zeros_like(denoised, dtype=np.uint8)
    for x, y in points:
        flag[y, x] = 1

    results = {}
    for name, fn in ALGORITHMS.items():
        tracemalloc.start()
        t0 = time.perf_counter()
        _, iters = fn(flag, scale)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[name] = (elapsed, peak / (1024 * 1024), iters)
    return results


def main():
    SCALES = [1, 2, 4, 8]  # 对应 40x32(基线), 80x64, 160x128, 320x256
    N_SAMPLES = 50          # 建议正式跑时用和Table 8一致的10000，先小样本验证脚本

    baseline_samples = load_baseline_samples(N_SAMPLES)

    rows = []
    for scale in SCALES:
        res_label = f"{40*scale}x{32*scale}"
        print(f"\n=== 分辨率 {res_label} (scale={scale}) ===")
        per_algo_times = {name: [] for name in ALGORITHMS}
        per_algo_mem = {name: [] for name in ALGORITHMS}
        per_algo_iters = {name: [] for name in ALGORITHMS}

        for i, img in enumerate(baseline_samples):
            r = run_one(img, scale)
            for name, (t, m, it) in r.items():
                per_algo_times[name].append(t)
                per_algo_mem[name].append(m)
                per_algo_iters[name].append(it)
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i+1}/{len(baseline_samples)}")

        for name in ALGORITHMS:
            rows.append({
                "Resolution": res_label,
                "Algorithm": name,
                "Avg Runtime (s)": np.mean(per_algo_times[name]),
                "Avg Memory (MB)": np.mean(per_algo_mem[name]),
                "Avg Iterations": np.mean(per_algo_iters[name]),
                "N Samples": len(baseline_samples),
            })

    df = pd.DataFrame(rows)
    out_path = "resolution_scaling_results.csv"
    df.to_csv(OUT_DIR / "resolution_scaling_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到 {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
