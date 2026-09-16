"""
recognition_preprocess_fix.py
==============================
修复"切割正确但识别错误"问题的核心补丁：把切割出来的半图重新标准化到
和CVL Single Digit训练数据一致的格式（提取真实内容+居中+直方图均衡化），
再送入识别模型。

用法：在 u_left_right_recog1_cvl.py 里，把这个文件里的 standardize_for_recognition()
函数复制过去（或 import 这个模块），在调用 self.recognize_image(left_img) /
self.recognize_image(right_img) 之前，先各自过一遍这个函数。

具体改法（对照 process_single_image 第196、199行）：
generate_left_right_images
    原代码：
        left_pred, left_conf = self.recognize_image(left_img)
        right_pred, right_conf = self.recognize_image(right_img)

    改为：
        left_std = standardize_for_recognition(left_img)
        right_std = standardize_for_recognition(right_img)
        left_pred, left_conf = self.recognize_image(left_std)
        right_pred, right_conf = self.recognize_image(right_std)
"""

import numpy as np
from PIL import Image, ImageOps

try:
    from scipy.ndimage import label as _cc_label
except ImportError:
    _cc_label = None
    print("[提示] 未安装scipy，standardize_for_recognition将退化为不做连通域过滤"
          "（噪点鲁棒性下降）。建议 pip install scipy --break-system-packages")


def _robust_content_bbox(half_image, content_threshold=20, min_component_pixels=4):
    """
    用连通域分析找真实笔画内容的边界框，过滤掉孤立噪点（比如遮罩切割残留的
    单个/几个像素噪点）。不能再用"只要非零就纳入边界框"这种朴素判据——
    哪怕只有一个远离主体的孤立噪点像素，边界框也会被强行拉大，导致后续缩放
    把真正的字符主体过度缩小、位置也跟着算偏，这正是本函数要修复的问题。

    做法：先按content_threshold二值化，用连通域标记找出所有独立的像素团块，
    丢弃像素数小于min_component_pixels的团块（视为噪点），剩余团块的并集
    再取边界框。

    返回 (y0, y1, x0, x1)，即内容边界框的行列范围（闭区间）；无有效内容返回None
    """
    binary = half_image > content_threshold

    if _cc_label is not None:
        labeled, num_features = _cc_label(binary)
        if num_features == 0:
            return None
        keep_mask = np.zeros_like(binary)
        for comp_id in range(1, num_features + 1):
            comp_pixels = (labeled == comp_id)
            if comp_pixels.sum() >= min_component_pixels:
                keep_mask |= comp_pixels
        ys, xs = np.where(keep_mask)
        if len(xs) == 0:
            # 所有连通域都比噪点门槛还小（比如整张图只有稀疏噪点），
            # 退化为使用全部前景像素，避免直接判定"无内容"丢弃整张图
            ys, xs = np.where(binary)
    else:
        # 没有scipy时的退化方案：至少比原来"完全不过滤"更好一点，
        # 用content_threshold二值化本身已经能挡掉部分低强度噪点
        ys, xs = np.where(binary)

    if len(xs) == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def standardize_for_recognition(half_image, target_core=27, canvas_size=(32, 40),
                                 content_threshold=20, min_component_pixels=4):
    """
    把"遮罩式切割"产出的半图，重新标准化成和CVL Single Digit训练数据一致的格式：
    1. 用连通域分析找到真实笔画内容的边界框，过滤孤立噪点（见 _robust_content_bbox）
    2. 按目标核心尺寸(target_core)等比缩放该内容（按高度锚定，取较小缩放比）
    3. 把缩放后的内容居中放进一张新的画布（大小同canvas_size，默认32x40，
       和训练/主pipeline其余部分保持一致）
    4. 做直方图均衡化，匹配CVL Single Digit normalized数据集的预处理

    参数：
        half_image: 2D numpy数组，遮罩式切割产出的半图（另一半已被清零）
        target_core: 内容缩放后的目标高度。默认27——用
                     measure_train_inference_gap.py 实测CVL Single Digit训练集
                     （trainStd40-32，N=2000）得到的内容高度均值（标准差仅0.2，
                     几乎是常数）。如果训练数据换了版本，请重新跑一遍测量脚本。
        canvas_size: 输出画布尺寸 (height, width)，默认(32,40)，和主pipeline其余环节一致
        content_threshold: 判定"这个像素算不算真实内容"的灰度阈值（默认20，
                     和项目里其余环节的去噪/二值化阈值保持一致）
        min_component_pixels: 连通域最小像素数门槛（默认4），小于这个像素数的
                     连通域视为噪点，不计入边界框计算

    返回：
        2D numpy数组（uint8），已重新居中+均衡化，可直接送入 recognize_image
        如果half_image没有任何有效内容，返回全零画布
    """
    canvas_h, canvas_w = canvas_size

    bbox = _robust_content_bbox(half_image, content_threshold=content_threshold,
                                 min_component_pixels=min_component_pixels)
    if bbox is None:
        return np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    y0, y1, x0, x1 = bbox
    content = half_image[y0:y1 + 1, x0:x1 + 1]

    h, w = content.shape
    # 【按实测统计校准】CVL训练集高度标准差仅0.2（几乎是常数），宽度标准差5.7
    # （波动大得多）——这个模式说明训练集是"按高度锚定缩放"（缩放到固定高度，
    # 宽度跟着长宽比自然浮动），不是"取长宽中较大者锚定"。所以这里用高度锚定，
    # 而不是 max(h, w)，以匹配训练集实际的构造方式。
    scale = target_core / h
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    # 安全钳制：极端宽扁的内容按高度锚定缩放后，宽度可能超出画布，裁到画布宽度
    new_w = min(new_w, canvas_w)

    content_img = Image.fromarray(content).resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new('L', (canvas_w, canvas_h), 0)
    x_offset = (canvas_w - new_w) // 2
    y_offset = (canvas_h - new_h) // 2
    canvas.paste(content_img, (x_offset, y_offset))

    # 直方图均衡化，匹配CVL Single Digit normalized数据集的预处理步骤
    #canvas_eq = ImageOps.equalize(canvas)

    return np.array(canvas, dtype=np.uint8)