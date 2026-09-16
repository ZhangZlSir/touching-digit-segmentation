"""
corridor_utils.py
==================
走廊计算核心函数，从 corridor_qc_tool.py 中提取出来的纯numpy实现，
不依赖tkinter/PIL.ImageTk，方便在生产pipeline（尤其是ProcessPoolExecutor
多进程场景）中直接import，不需要拖带GUI相关依赖。

corridor_qc_tool.py（人工标注/调参工具）和生产pipeline（u_left_right_recog1.py等）
共用这份文件里的函数，保证"调参时看到的走廊"和"实际切割时用的走廊"是
同一套计算逻辑，不会出现两边代码分叉、参数对不上的问题。
"""

import numpy as np


def find_content_bounds(proj, min_run_length=2):
    """
    在投影序列proj中找非零列构成的所有连续区间，过滤掉长度小于min_run_length
    的区间（视为孤立噪点，不算真实内容），返回过滤后剩余区间里最左端和最右端
    的列坐标 (L, R)。这样得到的内容边界天然对孤立噪点免疫，不需要额外的
    像素密度阈值。如果过滤后没有任何区间满足长度要求，返回 None。
    """
    n = len(proj)
    runs = []
    start = None
    for i in range(n):
        if proj[i] > 0 and start is None:
            start = i
        elif proj[i] == 0 and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))

    runs = [(s, e) for (s, e) in runs if (e - s + 1) >= min_run_length]
    if not runs:
        return None
    return runs[0][0], runs[-1][1]


def compute_corridor(img_array, margin_frac=0.3, min_width=3,
                      min_run_length=2, min_buffer_cols=2,
                      enable_top_guard=False, protect_top_rows=2):
    """
    走廊计算公式（简化版，便于在论文方法部分直接描述）：

        L, R        = 最长有效连续非零投影区间的左右端点（自动过滤孤立噪点列）
        geo_center  = 0.5 * (L + R)                                   内容几何中心
        mass_center = sum(x * proj[x] for x in [L,R]) / sum(proj[x])  投影质量中心
        center      = 0.5 * (geo_center + mass_center)                走廊中心
        r           = margin_frac * (R - L)                           走廊半径（相对宽度的比例，不再是绝对像素数）
        corridor    = clip([center - r, center + r], L, R)

    参数：
        img_array: 2D numpy数组，灰度图，前景为非零像素，背景为0
        margin_frac: 走廊半径占内容总宽度(R-L)的比例（默认0.3）。用相对比例
                     而不是绝对像素数，好处是不需要根据每批数据的字符尺度
                     单独调参，同一个比例对不同大小的内容自动适配。
        min_width: 走廊最终宽度的最小值（防止走廊退化为0宽度导致生长域断裂）
        min_run_length: 判定"一段连续非零列是否算真实内容"所需的最小长度
                     （默认2，即孤立单列噪点会被排除，不参与L/R的计算）
        min_buffer_cols: 走廊左右两侧各自必须排除在外的最少有效列数（默认2）。
                     防止走廊在内容本身很窄时（例如两个紧挨的"1"）把整个
                     字符对全部纳入走廊，保证切割算法的搜索域不会延伸到
                     最外侧字符的外边缘。优先级高于顶行安全网。
        enable_top_guard: 是否启用顶行安全网（默认False，改为显式开关，
                     方便通过GUI勾选框直接对比"开/关"两种情况下走廊的差异，
                     确认这条规则是否真的在起作用）。
        protect_top_rows: 顶行安全网检查的行数（仅在enable_top_guard=True时生效）。
                     检查顶部几行里真实笔画覆盖的列范围，如果超出当前走廊，
                     扩展走廊去包含它（但不能突破侧边缓冲区划定的边界）。

    返回：
        dict，包含：
            L, R: 内容连续区间的左右边界（画布坐标）
            geo_center, mass_center, center: 三个中心坐标
            corridor_left, corridor_right: 最终走廊左右边界（画布坐标）
            top_guard_triggered: 是否触发了顶行安全网扩展（仅enable_top_guard=True时可能为True）
            side_buffer_ok: 是否成功保证了两侧缓冲区
        如果没有满足min_run_length的内容区间，返回 None
    """
    proj = np.sum(img_array > 0, axis=0)
    bounds = find_content_bounds(proj, min_run_length=min_run_length)
    if bounds is None:
        return None
    L, R = bounds

    geo_center = 0.5 * (L + R)
    xs = np.arange(L, R + 1)
    weights = proj[L:R + 1].astype(np.float64)
    total_weight = weights.sum()
    mass_center = float(np.sum(xs * weights) / total_weight) if total_weight > 0 else geo_center

    center = 0.5 * (geo_center + mass_center)
    r = margin_frac * (R - L)

    corridor_left = max(L, int(round(center - r)))
    corridor_right = min(R, int(round(center + r)))
    if (corridor_right - corridor_left) < min_width:
        corridor_left, corridor_right = L, R

    # ---- 侧边缓冲区约束：走廊两侧各留至少min_buffer_cols个有效列 ----
    # 尽力而为策略：从min_buffer_cols开始尝试，如果内容太窄凑不出这么多列，
    # 逐级降低要求（min_buffer_cols-1, min_buffer_cols-2, ... 直到0），
    # 保证始终拿到当前内容条件下能做到的最大缓冲，而不是"要么达标要么完全
    # 放弃、退化成零缓冲"——后者正是之前版本的问题：内容一旦太窄，corridor
    # 会直接退化为整个[L,R]，margin_frac怎么调都不起作用，因为约束整个被跳过了。
    valid_cols = np.where(proj[L:R + 1] > 0)[0] + L
    min_left_boundary = L
    max_right_boundary = R
    achieved_buffer = 0
    if len(valid_cols) > 0:
        for buf in range(min_buffer_cols, -1, -1):
            if buf == 0:
                candidate_left, candidate_right = L, R
            elif len(valid_cols) >= 2 * buf:
                candidate_left = int(valid_cols[buf - 1]) + 1
                candidate_right = int(valid_cols[-buf]) - 1
            else:
                continue
            if candidate_right - candidate_left + 1 >= min_width:
                min_left_boundary = candidate_left
                max_right_boundary = candidate_right
                achieved_buffer = buf
                break
        if corridor_left < min_left_boundary:
            corridor_left = min_left_boundary
        if corridor_right > max_right_boundary:
            corridor_right = max_right_boundary
        if corridor_right < corridor_left:
            corridor_left, corridor_right = min_left_boundary, max_right_boundary
    side_buffer_ok = (achieved_buffer >= min_buffer_cols)

    # ---- 顶行安全网（可选，默认关闭；优先级低于侧边缓冲区）----
    top_guard_triggered = False
    if enable_top_guard:
        height = img_array.shape[0]
        n_top = min(protect_top_rows, height)
        if n_top > 0:
            top_slice = img_array[:n_top, :]
            top_content_cols = np.where(np.any(top_slice > 0, axis=0))[0]
            if len(top_content_cols) > 0:
                top_content_min = int(top_content_cols[0])
                top_content_max = int(top_content_cols[-1])
                allowed_left = min_left_boundary if side_buffer_ok else L
                allowed_right = max_right_boundary if side_buffer_ok else R
                new_left = max(allowed_left, min(corridor_left, top_content_min))
                new_right = min(allowed_right, max(corridor_right, top_content_max))
                if new_left < corridor_left:
                    corridor_left = new_left
                    top_guard_triggered = True
                if new_right > corridor_right:
                    corridor_right = new_right
                    top_guard_triggered = True

    return {
        "L": L, "R": R,
        "geo_center": geo_center, "mass_center": mass_center, "center": center,
        "corridor_left": corridor_left, "corridor_right": corridor_right,
        "top_guard_triggered": top_guard_triggered,
        "side_buffer_ok": side_buffer_ok,
        "achieved_buffer": achieved_buffer,
    }
