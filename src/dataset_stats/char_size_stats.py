#char_size_stats.py  --  Table 2 的前置输入：统计每张 20x20 紧凑字符的笔画水平范围
# 统计 data/mnist_compact_20x20 中每张 20x20 紧凑字符的笔画尺寸，生成尺寸统计 CSV
# 阈值使用12（5%亮度阈值）

import numpy as np
import pandas as pd
import struct

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_IN = PROJECT_ROOT / 'data' / 'mnist_compact_20x20'          # 20x20 compact MNIST characters
DATA_TDTS = PROJECT_ROOT / 'data' / 'tdts_dataset'               # TDTS two-digit touching set
BUILD_INPUTS = PROJECT_ROOT / 'data' / 'tdts_build_inputs'       # coordinates / pair index CSVs
OUT_DIR = PROJECT_ROOT / 'results' / 'stats'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_idx_images(filename):
    """读取IDX格式的图片文件"""
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 0x803:
            raise ValueError(f"不是IDX图片文件: {filename}")
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        return data

def find_horizontal_bounds(img, threshold=12):
    """找到图片中有效像素的左右边界"""
    height, width = img.shape
    left = width
    right = -1
    
    # 寻找左边界（第一个有效像素列）
    for x in range(width):
        for y in range(height):
            if img[y, x] >= threshold:
                left = x
                break
        if left < width:  # 找到了左边界
            break
    
    # 寻找右边界（最后一个有效像素列）
    for x in range(width-1, -1, -1):
        for y in range(height):
            if img[y, x] >= threshold:
                right = x
                break
        if right >= 0:  # 找到了右边界
            break
    
    # 如果没有找到有效像素
    if left > right:
        return 0, 19, 20  # 返回全图宽度
    
    return left, right, right - left + 1

def process_compact_images(images_path, labels_path, output_csv):
    """处理紧凑字符图片并保存尺寸信息"""
    # 1. 读取图片数据和标签
    images = read_idx_images(images_path)
    labels = read_idx_labels(labels_path)
    
    num_images = images.shape[0]
    
    # 2. 统计每个图片的边界信息
    data = []
    for idx in range(num_images):
        img = images[idx]
        left, right, width = find_horizontal_bounds(img, threshold=12)
        
        data.append({
            'index': idx,
            'digit': labels[idx],  # 数字标签
            'left_bound': left,
            'right_bound': right,
            'width': width
        })
    
    # 3. 保存为CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    # 4. 输出统计信息
    print(f"处理完成: {num_images} 张图片")
    print(f"保存到: {output_csv}")
    
    # 宽度统计
    print("\n=== 宽度分布统计 ===")
    print(df['width'].describe())
    
    # 宽度分布频率
    width_counts = df['width'].value_counts().sort_index()
    print("\n宽度分布频率:")
    for width, count in width_counts.items():
        percentage = count / num_images * 100
        print(f"宽度 {width:2d}: {count:5d} 张 ({percentage:5.2f}%)")
    
    # 按数字统计平均宽度
    print("\n=== 按数字统计平均宽度 ===")
    digit_stats = df.groupby('digit')['width'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(digit_stats)
    
    return df

def read_idx_labels(filename):
    """读取IDX格式的标签文件"""
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 0x801:
            raise ValueError(f"不是IDX标签文件: {filename}")
        num_labels = struct.unpack('>I', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

def main():
    # 训练集分析
    print("=" * 60)
    print("分析训练集...")
    print("=" * 60)
    train_df = process_compact_images(
        DATA_IN / 'MNIST' / 'raw' / 'train-images-idx3-ubyte',
        DATA_IN / 'MNIST' / 'raw' / 'train-labels-idx1-ubyte',
        OUT_DIR / 'train_size_data3.csv'
    )
    
    print("\n" + "=" * 60)
    print("分析测试集...")
    print("=" * 60)
    test_df = process_compact_images(
        DATA_IN / 'MNIST' / 'raw' / 't10k-images-idx3-ubyte',
        DATA_IN / 'MNIST' / 'raw' / 't10k-labels-idx1-ubyte',
        OUT_DIR / 'test_size_data3.csv'
    )
    

if __name__ == "__main__":
    main()