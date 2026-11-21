import torch
from torchvision.datasets import MNIST
import random
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / 'data' / 'mnist_compact_20x20'
OUT_CSV = PROJECT_ROOT / 'data' / 'tdts_build_inputs' / 'mnist_image_pairs_train.csv'
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# 此代码是正确的,豆包原始方式，用于生成训练集的坐标配对表
# 设置随机数种子
random.seed(42)

# 加载 MNIST 训练集 
# ./data 是数据集下载的路径，train=True 表示加载训练集
# download=True 表示如果数据集不存在则下载数据集
mnist_train = MNIST(str(DATA_ROOT), train=True, download=False)

# 左侧图片索引从 0 到 59999
left_indices = list(range(60000))

# 记录右侧图片每个数字的索引值
#{i:[] for i in range(10)} i:[] 表示这是一个字典；key是整型，value是一个空列表。
right_digit_indices = {i: [] for i in range(10)}
for idx in range(60000):
    label = int(mnist_train[idx][1]) #mnist_train[idx] 是一个元组，[0]是图片，[1]是标签
    right_digit_indices[label].append(idx) #将索引值idx添加到标签如0，1，2的列表中

# 处理右侧图片，使每类图片都有 6000 张
for digit in range(10):
    indices = right_digit_indices[digit] #取出标签为0，1，2的列表
    if len(indices) > 6000:
        # 若某类图片超过 6000 个，随机选择 6000 张
        right_digit_indices[digit] = random.sample(indices, 6000)
    elif len(indices) < 6000:
        # 若某类图片不够 6000 张，循环随机填补重复序号
        while len(indices) < 6000:
            indices.append(random.choice(indices))
        right_digit_indices[digit] = indices

# 合并右侧预选集索引
right_preselected_indices = []
for digit in range(10):
    right_preselected_indices.extend(right_digit_indices[digit])

# 打乱右侧预选集索引顺序
random.shuffle(right_preselected_indices)

# 用于存储配对信息
pairings = []

# 进行配对
for left_index in left_indices:
    left_label = int(mnist_train[left_index][1]) #[1]为mnist图片的标签，[0]为图片
    right_index = right_preselected_indices.pop(0) 
    right_label = int(mnist_train[right_index][1])
    pairings.append((left_index, right_index, left_label, right_label))

# 输出 0 - 9 图片的实际张数
label_counts = {i: {j: 0 for j in range(10)} for i in range(10)}
for _, _, left_label, right_label in pairings:
    label_counts[left_label][right_label] += 1
for i in range(10):
    for j in range(10):
        print(f"标签对 ({i}, {j}) 的数量: {label_counts[i][j]}")

# ---- 类别均衡性指标（对应论文 Table 1）----
_flat = [label_counts[i][j] for i in range(10) for j in range(10)]
_mean = sum(_flat) / len(_flat)
_mad = sum(abs(c - _mean) for c in _flat) / len(_flat)
_var = sum((c - _mean) ** 2 for c in _flat) / len(_flat)
_cv = (_var ** 0.5) / _mean if _mean else 0.0
print(f"标签对平均数量: {_mean:.2f}, MAD: {_mad:.2f}, CV: {_cv:.4f}")

# 输出左侧所有图片的索引值和、右侧图片的索引值和
left_index_sum = sum([pair[0] for pair in pairings])
right_index_sum = sum([pair[1] for pair in pairings])
print(f"左侧图片索引值和: {left_index_sum}")
print(f"右侧图片索引值和: {right_index_sum}")

# 统计各类图片总数
total_images = 0
for digit in range(10):
    total_images += len(right_digit_indices[digit])

print(f"各类图片总数: {total_images}")
if total_images == 60000:
    print("各类图片总数为 60000 张。")
else:
    print("各类图片总数不为 60000 张。")

# 生成 CSV 文件
with open(OUT_CSV, mode='w', newline='') as csv_file:
    fieldnames = ['Left Image Index', 'Right Image Index', 'Left Label', 'Right Label', 'Two - Digit Label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for left_index, right_index, left_label, right_label in pairings:
        two_digit_label = f"{left_label}{right_label}"
        writer.writerow({
            'Left Image Index': left_index,
            'Right Image Index': right_index,
            'Left Label': left_label,
            'Right Label': right_label,
            'Two - Digit Label': two_digit_label
        })

print(f"CSV 文件={OUT_CSV} 已生成。")
    