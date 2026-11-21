#new3240-clean.py
#40*32的图片，20*20的单一数字图片叠加在上面
#纯粹的无噪声版本，用于消融实验
#图片的位置也是随机的
import os
import numpy as np
import struct
from torchvision.datasets import MNIST
from PIL import Image
import random

# 字符贴图位置与噪点均为随机采样。原始库生成时未固定随机数种子，因此重新运行得到的是
# 同分布的新实例（如需论文中使用的原始库，请联系作者）；这里固定 seed=42 以保证可复现。
random.seed(42)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_IN = PROJECT_ROOT / 'data' / 'mnist_compact_20x20'
OUT_DIR = PROJECT_ROOT / 'data' / 'tdts_single_char' / 'clean' / 'MNIST' / 'raw'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# 下载数据并获取原始文件路径
def get_mnist_paths(root=None):
    root = str(root or DATA_IN)
    # 输入：data/mnist_compact_20x20 里的 20x20 紧凑字符
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

# 处理图像（填充到32×40）- 只生成不带噪声的版本
def process_images(images):
    clean_images = []  # 不带噪声的图片
    
    for img in images:
        # 创建不带噪声的图像
        image_clean = Image.new('L', (40, 32))
        
        # 把img采用覆盖方式放到图像上
        img_width, img_height = (20,20)
        img_pil = Image.fromarray(img) 
       
        # 随机位置
        x = random.randint(0, 20)
        y = random.randint(0, 12)
        
        # 将img覆盖到图像上
        for i in range(img_height):
            for j in range(img_width):
                if x + j < 40 and y + i < 32:
                    pixel = img_pil.getpixel((j, i))  # 获取原图像素值
                    image_clean.putpixel((x + j, y + i), pixel)

        # 将处理后的图像转换为numpy数组
        p_clean = np.array(image_clean)
        clean_images.append(p_clean)
    
    return np.array(clean_images)

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
print("处理训练集...")
train_images = read_idx(file_paths['train_images'])
train_labels = read_idx(file_paths['train_labels'])

train_clean = process_images(train_images)

# 直接使用干净图像，不需要合并
save_images(OUT_DIR / 'train-images-idx3-ubyte', train_clean)
save_labels(OUT_DIR / 'train-labels-idx1-ubyte', train_labels)

# 处理测试集
print("处理测试集...")
test_images = read_idx(file_paths['test_images'])
test_labels = read_idx(file_paths['test_labels'])

test_clean = process_images(test_images)

# 直接使用干净图像，不需要合并
save_images(OUT_DIR / 't10k-images-idx3-ubyte', test_clean)
save_labels(OUT_DIR / 't10k-labels-idx1-ubyte', test_labels)

print(f"处理完成！结果保存在 {OUT_DIR}")
print(f"训练集大小: {len(train_clean)} 张图片")
print(f"测试集大小: {len(test_clean)} 张图片")