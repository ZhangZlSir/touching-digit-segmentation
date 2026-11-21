import torch
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
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
train_dataset = datasets.MNIST(root=str(DATA_IN), train=True, download=True)
test_dataset = datasets.MNIST(root=str(DATA_IN), train=False, download=True)

# 获取标签
train_labels = train_dataset.targets.numpy()
test_labels = test_dataset.targets.numpy()

# 统计各数字数量
def count_digits(labels):
    counts = np.zeros(10, dtype=int)
    for label in labels:
        counts[label] += 1
    return counts

train_counts = count_digits(train_labels)
test_counts = count_digits(test_labels)

# ========== 计算理论MAD ==========

def calculate_theoretical_mad_for_double_digit_mnist(single_digit_counts, total_samples, dataset_type="训练集"):
    """
    计算由单字符MNIST构成双字符MNIST的理论MAD
    
    参数:
        single_digit_counts: 单字符MNIST中每个数字的计数
        total_samples: 双字符MNIST的总样本数
        dataset_type: 数据集类型（"训练集"或"测试集"）
    
    返回:
        theoretical_mad: 理论MAD值
        expected_matrix: 期望数量矩阵 (10x10)
        p: 单字符概率分布
        uniform_expectation: 均匀分布的期望
    """
    # 计算单字符MNIST中每个数字的概率分布
    total_single = np.sum(single_digit_counts)
    p = single_digit_counts / total_single
    
    print(f"\n{dataset_type}单字符MNIST分布信息:")
    print("-" * 50)
    for i in range(10):
        print(f"数字 {i}: {single_digit_counts[i]:6d} 张, 概率: {p[i]:.4f}")
    print(f"总数: {total_single:,}")
    
    # 计算双字符MNIST的期望数量矩阵
    # 对于两个独立字符的组合 (i, j)，期望数量 = 总样本数 * p(i) * p(j)
    expected_matrix = total_samples * np.outer(p, p)
    
    # 计算双字符MNIST的类别数
    num_classes = 100  # 00-99 共100个类别
    
    # 计算每个类别的平均期望（均匀分布下的期望）
    uniform_expectation = total_samples / num_classes
    
    # 计算理论MAD：期望数量与均匀期望之差的绝对值的平均值
    # 使用公式：MAD = mean(abs(E - uniform_expectation))
    theoretical_mad = np.mean(np.abs(expected_matrix.flatten() - uniform_expectation))
    
    # 计算相对MAD
    relative_mad = (theoretical_mad / uniform_expectation) * 100
    
    return theoretical_mad, expected_matrix, p, uniform_expectation, relative_mad

def analyze_double_digit_distribution(expected_matrix, dataset_type, total_samples, uniform_expectation):
    """分析双字符分布"""
    print(f"\n{dataset_type}双字符MNIST理论分布分析:")
    print("-" * 60)
    print(f"双字符类别数: 100 (00-99)")
    print(f"双字符总样本数: {total_samples:,}")
    print(f"均匀分布期望: {uniform_expectation:.2f} 每类")
    
    # 展平期望矩阵
    flat_expected = expected_matrix.flatten()
    
    # 计算统计指标
    max_expected = np.max(flat_expected)
    min_expected = np.min(flat_expected)
    range_expected = max_expected - min_expected
    std_expected = np.std(flat_expected)
    
    print(f"最大期望数量: {max_expected:.2f} (类别 {np.unravel_index(np.argmax(expected_matrix), expected_matrix.shape)})")
    print(f"最小期望数量: {min_expected:.2f} (类别 {np.unravel_index(np.argmin(expected_matrix), expected_matrix.shape)})")
    print(f"期望数量范围: {range_expected:.2f}")
    print(f"期望数量标准差: {std_expected:.2f}")
    
    return flat_expected

def visualize_theoretical_distribution(expected_matrix, dataset_type, uniform_expectation, theoretical_mad):
    """可视化理论分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 期望矩阵热图
    im = axes[0, 0].imshow(expected_matrix, cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'{dataset_type}双字符MNIST期望数量矩阵')
    axes[0, 0].set_xlabel('第二个数字')
    axes[0, 0].set_ylabel('第一个数字')
    axes[0, 0].set_xticks(range(10))
    axes[0, 0].set_yticks(range(10))
    plt.colorbar(im, ax=axes[0, 0], label='期望数量')
    
    # 2. 期望分布直方图
    flat_expected = expected_matrix.flatten()
    axes[0, 1].hist(flat_expected, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=uniform_expectation, color='red', linestyle='--', 
                      label=f'均匀期望: {uniform_expectation:.2f}')
    axes[0, 1].axvline(x=uniform_expectation - theoretical_mad, color='orange', 
                      linestyle=':', label=f'±MAD范围')
    axes[0, 1].axvline(x=uniform_expectation + theoretical_mad, color='orange', 
                      linestyle=':')
    axes[0, 1].set_title(f'{dataset_type}双字符期望分布直方图')
    axes[0, 1].set_xlabel('期望数量')
    axes[0, 1].set_ylabel('类别数量')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 期望数量与均匀期望的偏差
    sorted_expected = np.sort(flat_expected)
    indices = np.arange(len(sorted_expected))
    
    axes[1, 0].plot(indices, sorted_expected, 'b-', linewidth=2)
    axes[1, 0].axhline(y=uniform_expectation, color='red', linestyle='--', 
                      label=f'均匀期望: {uniform_expectation:.2f}')
    axes[1, 0].fill_between(indices, 
                           uniform_expectation - theoretical_mad,
                           uniform_expectation + theoretical_mad,
                           alpha=0.3, color='orange', 
                           label=f'MAD范围: ±{theoretical_mad:.2f}')
    axes[1, 0].set_title(f'{dataset_type}双字符期望数量排序图')
    axes[1, 0].set_xlabel('类别索引（排序后）')
    axes[1, 0].set_ylabel('期望数量')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 单字符概率分布
    # 计算单字符概率
    p = np.sum(expected_matrix, axis=1) / np.sum(expected_matrix)  # 第一个数字的概率
    axes[1, 1].bar(range(10), p, alpha=0.7, color='green')
    axes[1, 1].axhline(y=0.1, color='red', linestyle='--', label='均匀概率: 0.1')
    axes[1, 1].set_title(f'{dataset_type}单字符概率分布')
    axes[1, 1].set_xlabel('数字')
    axes[1, 1].set_ylabel('概率')
    axes[1, 1].set_xticks(range(10))
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset_type}双字符MNIST理论分布分析 (理论MAD={theoretical_mad:.2f})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# ========== 主程序 ==========

print("双字符MNIST理论MAD分析")
print("=" * 80)

# 1. 计算训练集的理论MAD
print("\n" + "="*60)
print("训练集分析")
print("="*60)

train_total_double = 60000  # 双字符训练库数量
train_mad, train_expected, train_p, train_uniform, train_relative_mad = \
    calculate_theoretical_mad_for_double_digit_mnist(
        train_counts, train_total_double, "训练集"
    )

train_flat_expected = analyze_double_digit_distribution(
    train_expected, "训练集", train_total_double, train_uniform
)

print(f"\n训练集理论MAD: {train_mad:.2f}")
print(f"训练集相对MAD: {train_relative_mad:.2f}%")

# 2. 计算测试集的理论MAD
print("\n" + "="*60)
print("测试集分析")
print("="*60)

test_total_double = 10000  # 双字符测试库数量
test_mad, test_expected, test_p, test_uniform, test_relative_mad = \
    calculate_theoretical_mad_for_double_digit_mnist(
        test_counts, test_total_double, "测试集"
    )

test_flat_expected = analyze_double_digit_distribution(
    test_expected, "测试集", test_total_double, test_uniform
)

print(f"\n测试集理论MAD: {test_mad:.2f}")
print(f"测试集相对MAD: {test_relative_mad:.2f}%")

# 3. 比较分析
print("\n" + "="*60)
print("比较分析")
print("="*60)

print(f"\n理论MAD比较:")
print(f"  训练集理论MAD: {train_mad:.2f} (相对: {train_relative_mad:.2f}%)")
print(f"  测试集理论MAD: {test_mad:.2f} (相对: {test_relative_mad:.2f}%)")
print(f"  MAD差异: {abs(train_mad - test_mad):.2f}")

# 计算均匀分布的理论MAD（作为对比）
uniform_mad_train = 0  # 如果完全均匀分布，MAD=0
uniform_mad_test = 0

print(f"\n理想均匀分布:")
print(f"  训练集均匀期望: {train_uniform:.2f}")
print(f"  测试集均匀期望: {test_uniform:.2f}")
print(f"  理论MAD: 0.00")

# 4. 可视化
fig_train = visualize_theoretical_distribution(
    train_expected, "训练集", train_uniform, train_mad
)

fig_test = visualize_theoretical_distribution(
    test_expected, "测试集", test_uniform, test_mad
)

# 5. 创建比较图表
fig_compare, axes_compare = plt.subplots(2, 2, figsize=(14, 10))

# 比较单字符概率分布
axes_compare[0, 0].bar(range(10)-0.2, train_p, width=0.4, alpha=0.7, label='训练集')
axes_compare[0, 0].bar(range(10)+0.2, test_p, width=0.4, alpha=0.7, label='测试集')
axes_compare[0, 0].axhline(y=0.1, color='red', linestyle='--', label='均匀概率')
axes_compare[0, 0].set_title('单字符概率分布比较')
axes_compare[0, 0].set_xlabel('数字')
axes_compare[0, 0].set_ylabel('概率')
axes_compare[0, 0].set_xticks(range(10))
axes_compare[0, 0].legend()
axes_compare[0, 0].grid(True, alpha=0.3)

# 比较理论MAD
categories = ['训练集', '测试集']
mad_values = [train_mad, test_mad]
relative_mad_values = [train_relative_mad, test_relative_mad]

axes_compare[0, 1].bar(categories, mad_values, alpha=0.7, color=['blue', 'orange'])
axes_compare[0, 1].set_title('理论MAD比较')
axes_compare[0, 1].set_ylabel('MAD值')
axes_compare[0, 1].grid(True, alpha=0.3)

# 在柱状图上添加数值标签
for i, v in enumerate(mad_values):
    axes_compare[0, 1].text(i, v + 0.5, f'{v:.2f}', ha='center')

# 比较相对MAD
axes_compare[1, 0].bar(categories, relative_mad_values, alpha=0.7, color=['blue', 'orange'])
axes_compare[1, 0].set_title('相对MAD比较 (%)')
axes_compare[1, 0].set_ylabel('相对MAD (%)')
axes_compare[1, 0].grid(True, alpha=0.3)

# 在柱状图上添加数值标签
for i, v in enumerate(relative_mad_values):
    axes_compare[1, 0].text(i, v + 0.5, f'{v:.2f}%', ha='center')

# 期望分布直方图比较
axes_compare[1, 1].hist(train_flat_expected, bins=20, alpha=0.5, label='训练集', 
                       edgecolor='black')
axes_compare[1, 1].hist(test_flat_expected, bins=20, alpha=0.5, label='测试集', 
                       edgecolor='black')
axes_compare[1, 1].axvline(x=train_uniform, color='blue', linestyle='--', 
                          label='训练集均匀期望')
axes_compare[1, 1].axvline(x=test_uniform, color='orange', linestyle='--', 
                          label='测试集均匀期望')
axes_compare[1, 1].set_title('期望分布直方图比较')
axes_compare[1, 1].set_xlabel('期望数量')
axes_compare[1, 1].set_ylabel('类别数量')
axes_compare[1, 1].legend()
axes_compare[1, 1].grid(True, alpha=0.3)

plt.suptitle('双字符MNIST理论MAD对比分析', fontsize=16, fontweight='bold')
plt.tight_layout()

# 6. 显示所有图表
plt.show()

# 7. 打印总结
print("\n" + "="*80)
print("总结")
print("="*80)

print(f"""
双字符MNIST理论MAD分析总结:

1. 数据规模:
   - 训练集: {train_total_double:,} 个样本，100个类别，均匀期望 {train_uniform:.2f}
   - 测试集: {test_total_double:,} 个样本，100个类别，均匀期望 {test_uniform:.2f}

2. 理论MAD:
   - 训练集: {train_mad:.2f} (相对MAD: {train_relative_mad:.2f}%)
   - 测试集: {test_mad:.2f} (相对MAD: {test_relative_mad:.2f}%)

3. 分析:
   - 理论MAD反映了双字符MNIST中各类别样本数量的不均衡程度
   - MAD值越小，说明分布越接近均匀分布
   - 相对MAD表示MAD相对于均匀期望的比例
   - 理想情况下，如果单字符MNIST完全均匀分布，理论MAD为0

4. 结论:
   - 根据单字符MNIST的分布，可以预测双字符MNIST的理论不均衡程度
   - 训练集和测试集的理论MAD都大于0，说明双字符MNIST存在天然的不均衡性
   - 这种不均衡性会影响模型训练，可能需要数据平衡技术
""")