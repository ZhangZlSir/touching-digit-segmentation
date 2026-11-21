#s1_mnist_compact_20x20.py
# Step 1 of the TDTS construction pipeline (paper 3.3.1):
#   raw MNIST (28x28) -> crop the stroke bounding box -> zero-pad to 20x20 -> MNIST idx format
# Output: data/mnist_compact_20x20/MNIST/raw/*
# 取有效笔画外框（去掉外边框）后居中补零到 20x20（20 高、20 宽），按 MNIST idx 格式保存到
# data/mnist_compact_20x20/MNIST/raw/
import os
import numpy as np
import struct
from torchvision.datasets import MNIST
from PIL import Image


#img 要求是ndarray类型
def find_bounding_box(img, threshold=12):
    """找到图像中有效像素的边界框"""
    height, width = img.shape 

    bounding_box= (0, 0, width, height)

    # 初始化边界值
    top = height
    bottom = -1
    left = width
    right = -1

    # 寻找上下边界
    for y in range(height): #显然y是行，x是列
        for x in range(width):
            if img[y, x] >= threshold: #所以这里先行后列没有毛病
                if y < top:
                    top = y
                break  # 找到第一个有效行后退出
        if top != height and y > bottom:
            for x in range(width):
                if img[y, x] >= threshold:
                    bottom = y
                    break

    # 寻找左右边界
    for x in range(width):
        for y in range(height):
            if img[y, x] >= threshold:
                if x < left:
                    left = x
                break  # 找到第一个有效列后退出
        if left != width and x > right:
            for y in range(height):
                if img[y, x] >= threshold:
                    right = x
                    break

    # 处理未找到有效区域的情况
    if left > right or top > bottom:
        bounding_box= (0, 0, width, height)
    else:
        bounding_box= (left, top, right + 1, bottom + 1)
    return bounding_box

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / 'data' / 'mnist_compact_20x20' / 'MNIST' / 'raw'
OUT_DIR.mkdir(parents=True, exist_ok=True)
# 下载数据并获取原始文件路径
def get_mnist_paths(root=None):
    root = str(root or (PROJECT_ROOT / 'data' / '_mnist_source'))
    # 强制下载原始文件
    train_set= MNIST(root=root, train=True, download=True)
    train_set= MNIST(root=root, train=False, download=True)
    
    # 构建原始文件路径
    raw_dir = os.path.join(root, 'MNIST', 'raw')
    return {
        'train_images': os.path.join(raw_dir, 'train-images-idx3-ubyte'),
        'train_labels': os.path.join(raw_dir, 'train-labels-idx1-ubyte'),
        'test_images': os.path.join(raw_dir, 't10k-images-idx3-ubyte'),
        'test_labels': os.path.join(raw_dir, 't10k-labels-idx1-ubyte')
    }

# 获取文件路径
file_paths = get_mnist_paths()

# 验证文件存在
for key, path in file_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"关键文件缺失: {path}")

# 读取IDX文件，返回类型ndarray
def read_idx(filename):
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic == 0x803:
            num = struct.unpack('>I', f.read(4))[0]
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
            return data
        elif magic == 0x801:
            num = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data
        else:
            raise ValueError("无效的IDX文件")

# 处理图像（填充到20×20）
def process_images(images):
    padded = []
    for img in images:
        bounding_box = find_bounding_box(img) #左上角坐标，宽度和高度
        image=Image.fromarray(img)# 转换成pil的img类型，否则crop函数不能使用
        cropped_img = image.crop(bounding_box) #crop是裁剪的意思
        (x0, y0, x1, y1) = bounding_box
        width = x1 - x0
        height = y1 - y0

        # 计算填充量
        left = (20 - width) // 2
        right = 20 - width - left
        top = (20 - height) // 2
        bottom = 20 - height - top

        # 将裁剪后的图像转换为 numpy 数组
        cropped_img_np = np.array(cropped_img)

        # 进行填充操作
        p = np.pad(cropped_img_np, ((top, bottom), (left, right)), mode='constant', constant_values=0)
        padded.append(p)
    return np.array(padded)

# 保存处理后的图像
def save_images(filename, images):
    with open(filename, 'wb') as f:
        f.write(struct.pack('>I', 0x803))  # 魔数
        f.write(struct.pack('>III', *images.shape))  # 数量, 行, 列
        f.write(images.astype(np.uint8).tobytes())

# 保存标签
def save_labels(filename, labels):
    with open(filename, 'wb') as f:
        f.write(struct.pack('>II', 0x801, len(labels)))
        f.write(labels.tobytes())



# 处理训练集
train_images = read_idx(file_paths['train_images'])
train_images_padded = process_images(train_images)
save_images(OUT_DIR / 'train-images-idx3-ubyte', train_images_padded)
save_labels(OUT_DIR / 'train-labels-idx1-ubyte', read_idx(file_paths['train_labels']))

# 处理测试集
test_images = read_idx(file_paths['test_images'])
test_images_padded = process_images(test_images)
save_images(OUT_DIR / 't10k-images-idx3-ubyte', test_images_padded)
save_labels(OUT_DIR / 't10k-labels-idx1-ubyte', read_idx(file_paths['test_labels']))

print(f"处理完成！结果保存在 {OUT_DIR}")