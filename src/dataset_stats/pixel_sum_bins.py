#sum.py
import torchvision.datasets as datasets
import pandas as pd
import numpy as np

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_IN = PROJECT_ROOT / 'data' / 'mnist_compact_20x20'          # 20x20 compact MNIST characters
DATA_TDTS = PROJECT_ROOT / 'data' / 'tdts_dataset'               # TDTS two-digit touching set
BUILD_INPUTS = PROJECT_ROOT / 'data' / 'tdts_build_inputs'       # coordinates / pair index CSVs
OUT_DIR = PROJECT_ROOT / 'results' / 'stats'
OUT_DIR.mkdir(parents=True, exist_ok=True)
#本程序用于计算 MNIST 数据集中每个图片的像素累加和，取中间的99%，并将其分为 20 档，统计每档的数量，最后保存为 CSV 文件。
#单字符：data/mnist_compact_20x20（无噪）与 data/tdts_single_char/noisy（加噪）
#data4 双字符无噪声，data7 双字符加噪  double_original.csv    double_noisy.csv

# 下载并加载 MNIST 数据集
mnist_train = datasets.MNIST(root=str(DATA_TDTS), train=True, download=True)
mnist_test = datasets.MNIST(root=str(DATA_TDTS), train=False, download=True)
print("数据加载完成，处理数据中...")
# 合并训练集和测试集的图片
all_images = []
for image, _ in mnist_train:
    all_images.append(np.array(image))
for image, _ in mnist_test:
    all_images.append(np.array(image))
all_images = np.array(all_images)

# 计算所有图片的像素累加和
pixel_sums = np.sum(all_images, axis=(1, 2))

# 计算 0.5% 和 99.5% 分位数
lower_quantile = np.quantile(pixel_sums, 0.005)
upper_quantile = np.quantile(pixel_sums, 0.995)

# 过滤数据，保留中间 99% 的数据
filtered_pixel_sums = pixel_sums[(pixel_sums >= lower_quantile) & (pixel_sums <= upper_quantile)]

# 找出过滤后数据的最大值和最小值
max_sum = np.max(filtered_pixel_sums)
min_sum = np.min(filtered_pixel_sums)

# 计算每档的范围
bin_size = (max_sum - min_sum) / 20

# 初始化每档的数量
bin_counts = [0] * 20

# 统计每档的数量
for pixel_sum in filtered_pixel_sums:
    bin_index = int((pixel_sum - min_sum) // bin_size)
    if bin_index == 20:
        bin_index = 19
    bin_counts[bin_index] += 1

# 创建 DataFrame
data = {
    'Bin': [f'{min_sum + i * bin_size:.2f}-{min_sum + (i + 1) * bin_size:.2f}' for i in range(20)],
    'Count': bin_counts
}
df = pd.DataFrame(data)

# 保存为 CSV 文件
df.to_csv(OUT_DIR / 'double_noisy.csv', index=False)
print("数据处理完成，结果已保存为 double_noisy.csv")