import torch
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_IN = PROJECT_ROOT / 'data' / 'mnist_compact_20x20'          # 20x20 compact MNIST characters
DATA_TDTS = PROJECT_ROOT / 'data' / 'tdts_dataset'               # TDTS two-digit touching set
BUILD_INPUTS = PROJECT_ROOT / 'data' / 'tdts_build_inputs'       # coordinates / pair index CSVs
OUT_DIR = PROJECT_ROOT / 'results' / 'stats'
OUT_DIR.mkdir(parents=True, exist_ok=True)
# 设置中文字体和样式
rcParams['font.sans-serif'] = ['SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False
# 加载MNIST数据集
train_dataset = datasets.MNIST(root=str(DATA_TDTS), train=True, download=False)
test_dataset = datasets.MNIST(root=str(DATA_TDTS), train=False, download=False)

# 获取标签
train_labels = train_dataset.targets.numpy()
test_labels = test_dataset.targets.numpy()

# 定义标签范围
min_label = 10
max_label = 109
num_labels = max_label - min_label + 1  # 100个标签

# 统计各标签数量
def count_labels(labels, min_label, max_label):
    counts = np.zeros(num_labels, dtype=int)
    for label in labels:
        if min_label <= label <= max_label:
            counts[label - min_label] += 1
    return counts

# 统计训练集和测试集
train_counts = count_labels(train_labels, min_label, max_label)
test_counts = count_labels(test_labels, min_label, max_label)

# 计算总数
train_total = np.sum(train_counts)
test_total = np.sum(test_counts)

# ========== 新增：计算平均绝对偏差（MAD） ==========

def calculate_mad(counts, dataset_name):
    """计算平均绝对偏差"""
    if len(counts) == 0:
        return 0, 0, 0, 0
    
    # 计算平均值
    mean = np.mean(counts)
    
    # 计算绝对偏差
    absolute_deviations = np.abs(counts - mean)
    
    # 计算平均绝对偏差
    mad = np.mean(absolute_deviations)
    
    # 计算MAD相对平均值（用于比较不同规模数据集的偏差）
    mad_relative = (mad / mean) * 100 if mean > 0 else 0
    
    # 计算最大和最小数量
    max_count = np.max(counts)
    min_count = np.min(counts)
    
    # 计算极差
    range_value = max_count - min_count
    
    # 计算标准差（用于对比）
    std_dev = np.std(counts)
    
    print(f"\n{dataset_name} MAD分析:")
    print("-" * 50)
    print(f"类别数量: {len(counts)}")
    print(f"总样本数: {np.sum(counts):,}")
    print(f"每类平均样本数: {mean:.2f}")
    print(f"最大样本数: {max_count:,} (类别 {np.argmax(counts) + min_label})")
    print(f"最小样本数: {min_count:,} (类别 {np.argmin(counts) + min_label})")
    print(f"极差: {range_value:,}")
    print(f"平均绝对偏差(MAD): {mad:.2f}")
    print(f"相对MAD: {mad_relative:.2f}%")
    print(f"标准差: {std_dev:.2f}")
    print(f"变异系数: {(std_dev/mean*100):.2f}%" if mean > 0 else "变异系数: N/A")
    
    return mad, mean, max_count, min_count

def calculate_combined_mad(train_counts, test_counts):
    """计算训练集和测试集合并后的MAD"""
    combined_counts = train_counts + test_counts
    
    if len(combined_counts) == 0:
        return 0, 0, 0, 0
    
    # 计算平均值
    mean = np.mean(combined_counts)
    
    # 计算绝对偏差
    absolute_deviations = np.abs(combined_counts - mean)
    
    # 计算平均绝对偏差
    mad = np.mean(absolute_deviations)
    
    # 计算MAD相对平均值
    mad_relative = (mad / mean) * 100 if mean > 0 else 0
    
    # 计算最大和最小数量
    max_count = np.max(combined_counts)
    min_count = np.min(combined_counts)
    
    return mad, mean, max_count, min_count, combined_counts

def print_label_distribution(counts, min_label, max_label, dataset_name):
    """打印标签分布（每行10个，tab分隔）"""
    print(f"\n{dataset_name}标签分布 (每行显示10个标签):")
    print("-" * 80)
    
    for start_idx in range(0, num_labels, 10):
        end_idx = min(start_idx + 10, num_labels)
        # 构建当前行的标签和数量
        line_labels = range(start_idx + min_label, end_idx + min_label)
        line_counts = counts[start_idx:end_idx]
        
        # 打印标签
        labels_line = "\t".join(f"标签{i:3d}" for i in line_labels)
        print(labels_line)
        
        # 打印数量
        counts_line = "\t".join(f"{count:7d}" for count in line_counts)
        print(counts_line)
        
        # 打印百分比（可选）
        percentages = [(count/np.sum(counts)*100) if np.sum(counts) > 0 else 0 
                      for count in line_counts]
        percent_line = "\t".join(f"{p:6.2f}%" for p in percentages)
        print(percent_line)
        
        print()

# ========== 主程序 ==========

print("自定义MNIST数据集统计 (标签 10-109)")
print("=" * 100)

# 1. 打印标签分布
print_label_distribution(train_counts, min_label, max_label, "训练集")
print_label_distribution(test_counts, min_label, max_label, "测试集")

# 2. 计算并打印MAD
train_mad, train_mean, train_max, train_min = calculate_mad(train_counts, "训练集")
test_mad, test_mean, test_max, test_min = calculate_mad(test_counts, "测试集")

# 3. 计算合并MAD
combined_mad, combined_mean, combined_max, combined_min, combined_counts = calculate_combined_mad(
    train_counts, test_counts
)

print("\n" + "=" * 50)
print("数据集合并MAD分析:")
print("-" * 50)
print(f"总样本数: {np.sum(combined_counts):,}")
print(f"每类平均样本数: {combined_mean:.2f}")
print(f"最大样本数: {combined_max:,}")
print(f"最小样本数: {combined_min:,}")
print(f"平均绝对偏差(MAD): {combined_mad:.2f}")
print(f"相对MAD: {(combined_mad/combined_mean*100):.2f}%" if combined_mean > 0 else "相对MAD: N/A")

# 4. 可视化分析结果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 子图1: 训练集分布条形图
axes[0, 0].bar(range(num_labels), train_counts, alpha=0.7, color='blue')
axes[0, 0].axhline(y=train_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'平均值: {train_mean:.0f}')
axes[0, 0].fill_between(range(num_labels), 
                        train_mean - train_mad, 
                        train_mean + train_mad, 
                        alpha=0.2, color='red', 
                        label=f'MAD范围: ±{train_mad:.0f}')
axes[0, 0].set_title('训练集标签分布 (带MAD)')
axes[0, 0].set_xlabel('标签索引')
axes[0, 0].set_ylabel('样本数量')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 子图2: 测试集分布条形图
axes[0, 1].bar(range(num_labels), test_counts, alpha=0.7, color='green')
axes[0, 1].axhline(y=test_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'平均值: {test_mean:.0f}')
axes[0, 1].fill_between(range(num_labels), 
                        test_mean - test_mad, 
                        test_mean + test_mad, 
                        alpha=0.2, color='red', 
                        label=f'MAD范围: ±{test_mad:.0f}')
axes[0, 1].set_title('测试集标签分布 (带MAD)')
axes[0, 1].set_xlabel('标签索引')
axes[0, 1].set_ylabel('样本数量')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 子图3: 合并数据集分布条形图
axes[0, 2].bar(range(num_labels), combined_counts, alpha=0.7, color='purple')
axes[0, 2].axhline(y=combined_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'平均值: {combined_mean:.0f}')
axes[0, 2].fill_between(range(num_labels), 
                        combined_mean - combined_mad, 
                        combined_mean + combined_mad, 
                        alpha=0.2, color='red', 
                        label=f'MAD范围: ±{combined_mad:.0f}')
axes[0, 2].set_title('合并数据集标签分布 (带MAD)')
axes[0, 2].set_xlabel('标签索引')
axes[0, 2].set_ylabel('样本数量')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 子图4: 偏差分布直方图
all_deviations_train = np.abs(train_counts - train_mean)
all_deviations_test = np.abs(test_counts - test_mean)
all_deviations_combined = np.abs(combined_counts - combined_mean)

axes[1, 0].hist(all_deviations_train, bins=20, alpha=0.7, color='blue', 
                edgecolor='black', label=f'训练集 (MAD={train_mad:.1f})')
axes[1, 0].axvline(x=train_mad, color='red', linestyle='--', linewidth=2, 
                   label=f'MAD值')
axes[1, 0].set_title('训练集偏差分布直方图')
axes[1, 0].set_xlabel('绝对偏差')
axes[1, 0].set_ylabel('类别数量')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(all_deviations_test, bins=20, alpha=0.7, color='green', 
                edgecolor='black', label=f'测试集 (MAD={test_mad:.1f})')
axes[1, 1].axvline(x=test_mad, color='red', linestyle='--', linewidth=2, 
                   label=f'MAD值')
axes[1, 1].set_title('测试集偏差分布直方图')
axes[1, 1].set_xlabel('绝对偏差')
axes[1, 1].set_ylabel('类别数量')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].hist(all_deviations_combined, bins=20, alpha=0.7, color='purple', 
                edgecolor='black', label=f'合并集 (MAD={combined_mad:.1f})')
axes[1, 2].axvline(x=combined_mad, color='red', linestyle='--', linewidth=2, 
                   label=f'MAD值')
axes[1, 2].set_title('合并数据集偏差分布直方图')
axes[1, 2].set_xlabel('绝对偏差')
axes[1, 2].set_ylabel('类别数量')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('MNIST数据集标签分布与MAD分析', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 5. 输出详细统计报告
print("\n" + "=" * 100)
print("详细统计报告:")
print("=" * 100)

# 计算平衡性指标
def calculate_balance_metrics(counts, dataset_name):
    """计算数据集平衡性指标"""
    if len(counts) == 0:
        return
    
    mean = np.mean(counts)
    std = np.std(counts)
    cv = (std / mean * 100) if mean > 0 else 0
    
    # 计算类别分布的不平衡比例
    sorted_counts = np.sort(counts)
    # 前10%和后10%类别的平均数量比
    top_10 = np.mean(sorted_counts[-int(len(counts)*0.1):])
    bottom_10 = np.mean(sorted_counts[:int(len(counts)*0.1)])
    imbalance_ratio = top_10 / bottom_10 if bottom_10 > 0 else float('inf')
    
    print(f"\n{dataset_name}平衡性分析:")
    print(f"  类别数量: {len(counts)}")
    print(f"  总样本数: {np.sum(counts):,}")
    print(f"  每类样本数范围: [{np.min(counts):,} - {np.max(counts):,}]")
    print(f"  变异系数: {cv:.2f}%")
    print(f"  不平衡比例(前10%/后10%): {imbalance_ratio:.2f}")
    
    # 评估平衡性
    if cv < 10:
        balance_status = "非常均衡"
    elif cv < 20:
        balance_status = "较为均衡"
    elif cv < 30:
        balance_status = "轻度不平衡"
    elif cv < 50:
        balance_status = "中度不平衡"
    else:
        balance_status = "严重不平衡"
    
    print(f"  平衡状态: {balance_status}")

calculate_balance_metrics(train_counts, "训练集")
calculate_balance_metrics(test_counts, "测试集")
calculate_balance_metrics(combined_counts, "合并数据集")

# 6. 保存结果到文件
save_to_file = True
if save_to_file:
    timestamp = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
    filename = f"dataset_statistics_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("自定义MNIST数据集统计报告\n")
        f.write("=" * 100 + "\n")
        f.write(f"标签范围: {min_label} - {max_label}\n")
        f.write(f"类别数量: {num_labels}\n\n")
        
        f.write(f"训练集总数: {train_total:,}\n")
        f.write(f"测试集总数: {test_total:,}\n")
        f.write(f"数据集总数: {train_total + test_total:,}\n\n")
        
        f.write(f"训练集MAD: {train_mad:.2f}\n")
        f.write(f"测试集MAD: {test_mad:.2f}\n")
        f.write(f"合并数据集MAD: {combined_mad:.2f}\n\n")
        
        # 保存训练集详细分布
        f.write("训练集标签分布:\n")
        for start_idx in range(0, num_labels, 10):
            end_idx = min(start_idx + 10, num_labels)
            line_labels = range(start_idx + min_label, end_idx + min_label)
            line_counts = train_counts[start_idx:end_idx]
            
            labels_line = "\t".join(f"标签{i:3d}" for i in line_labels)
            counts_line = "\t".join(f"{count:7d}" for count in line_counts)
            
            f.write(labels_line + "\n")
            f.write(counts_line + "\n\n")
    
    print(f"\n详细报告已保存到: {filename}")