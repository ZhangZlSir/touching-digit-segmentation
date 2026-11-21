#adhesion_stats.py  --  Table 2：矩形粘连率与实际粘连率分布
# 本程序用于统计双数字图片库中实际x轴投影粘连分布
# 输入：数字尺寸文件、配对表、坐标表
# 输出：训练集和测试集的粘连分布统计

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_IN = PROJECT_ROOT / 'data' / 'mnist_compact_20x20'          # 20x20 compact MNIST characters
DATA_TDTS = PROJECT_ROOT / 'data' / 'tdts_dataset'               # TDTS two-digit touching set
BUILD_INPUTS = PROJECT_ROOT / 'data' / 'tdts_build_inputs'       # coordinates / pair index CSVs
OUT_DIR = PROJECT_ROOT / 'results' / 'stats'
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def load_size_info(train_size_path, test_size_path):
    """加载数字尺寸信息"""
    print("加载数字尺寸信息...")
    
    train_size_df = pd.read_csv(train_size_path)
    train_size_dict = {}
    for _, row in train_size_df.iterrows():
        idx = int(row['index'])
        train_size_dict[idx] = {
            'left_bound': int(row['left_bound']),
            'right_bound': int(row['right_bound']),
            'width': int(row['width'])
        }
    
    test_size_df = pd.read_csv(test_size_path)
    test_size_dict = {}
    for _, row in test_size_df.iterrows():
        idx = int(row['index'])
        test_size_dict[idx] = {
            'left_bound': int(row['left_bound']),
            'right_bound': int(row['right_bound']),
            'width': int(row['width'])
        }
    
    return train_size_dict, test_size_dict

def calculate_actual_overlap(left_size, right_size, x0):
    """计算两个数字的实际重叠像素数"""
    left_left = 0 + left_size['left_bound']
    left_right = 0 + left_size['right_bound']
    
    right_left = x0 + right_size['left_bound']
    right_right = x0 + right_size['right_bound']
    
    overlap_start = max(left_left, right_left)
    overlap_end = min(left_right, right_right)
    
    actual_overlap = max(0, overlap_end - overlap_start + 1)
    
    return actual_overlap

def calculate_rectangle_overlap(x0):
    """计算矩形区域的重叠"""
    overlap_start = max(0, x0)
    overlap_end = min(19, x0 + 19)
    
    rectangle_overlap = max(0, overlap_end - overlap_start + 1)
    
    return rectangle_overlap

def analyze_dataset(pairs_df, coords_df, size_dict):
    """分析单个数据集"""
    if len(pairs_df) != len(coords_df):
        min_len = min(len(pairs_df), len(coords_df))
        pairs_df = pairs_df.head(min_len)
        coords_df = coords_df.head(min_len)
    
    actual_overlaps = []
    rectangle_overlaps = []
    overlap_details = []
    
    for i in range(len(pairs_df)):
        left_idx = pairs_df.iloc[i]['Left Image Index']
        right_idx = pairs_df.iloc[i]['Right Image Index']
        x0 = coords_df.iloc[i]['x']
        
        left_size = size_dict.get(left_idx, {'left_bound': 0, 'right_bound': 19, 'width': 20})
        right_size = size_dict.get(right_idx, {'left_bound': 0, 'right_bound': 19, 'width': 20})
        
        actual_overlap = calculate_actual_overlap(left_size, right_size, x0)
        rectangle_overlap = calculate_rectangle_overlap(x0)
        
        actual_overlaps.append(actual_overlap)
        rectangle_overlaps.append(rectangle_overlap)
        
        overlap_details.append({
            'index': i,
            'left_idx': left_idx,
            'right_idx': right_idx,
            'x0': x0,
            'actual_overlap': actual_overlap,
            'rectangle_overlap': rectangle_overlap
        })
    
    details_df = pd.DataFrame(overlap_details)
    actual_counts = Counter(actual_overlaps)
    rectangle_counts = Counter(rectangle_overlaps)
    
    return details_df, actual_counts, rectangle_counts, actual_overlaps, rectangle_overlaps

def print_percentage_statistics(dataset_name, actual_counts, rectangle_counts, total_samples):
    """以百分比形式打印统计结果"""
    print(f"\n{dataset_name}粘连分布统计 (百分比):")
    print("="*60)
    
    print("实际数字重叠统计 (基于阈值12):")
    print("-"*40)
    for overlap in range(0, 21):
        count = actual_counts.get(overlap, 0)
        percentage = count / total_samples * 100
        print(f"重叠{overlap:2d}像素: {percentage:6.2f}%")
    
    print("\n矩形区域重叠统计 (不考虑数字宽度):")
    print("-"*40)
    for overlap in range(0, 21):
        count = rectangle_counts.get(overlap, 0)
        percentage = count / total_samples * 100
        print(f"重叠{overlap:2d}像素: {percentage:6.2f}%")
    
    print(f"\n重点统计: 0-6像素粘连分布")
    print("-"*40)
    for overlap in range(0, 7):
        actual_count = actual_counts.get(overlap, 0)
        rectangle_count = rectangle_counts.get(overlap, 0)
        actual_pct = actual_count / total_samples * 100
        rectangle_pct = rectangle_count / total_samples * 100
        print(f"重叠{overlap}像素 - 实际: {actual_pct:6.2f}% | 矩形: {rectangle_pct:6.2f}%")

def create_percentage_tables(train_actual_counts, train_rect_counts, train_total,
                             test_actual_counts, test_rect_counts, test_total):
    """创建百分比统计表格"""
    
    # 创建0-20像素的百分比表格
    overlap_range = list(range(0, 21))
    
    # 训练集百分比
    train_actual_pct = []
    train_rect_pct = []
    for overlap in overlap_range:
        train_actual_pct.append(train_actual_counts.get(overlap, 0) / train_total * 100)
        train_rect_pct.append(train_rect_counts.get(overlap, 0) / train_total * 100)
    
    # 测试集百分比
    test_actual_pct = []
    test_rect_pct = []
    for overlap in overlap_range:
        test_actual_pct.append(test_actual_counts.get(overlap, 0) / test_total * 100)
        test_rect_pct.append(test_rect_counts.get(overlap, 0) / test_total * 100)
    
    # 总体百分比
    all_actual_counts = Counter()
    all_rect_counts = Counter()
    
    for overlap, count in train_actual_counts.items():
        all_actual_counts[overlap] += count
    
    for overlap, count in test_actual_counts.items():
        all_actual_counts[overlap] += count
    
    for overlap, count in train_rect_counts.items():
        all_rect_counts[overlap] += count
    
    for overlap, count in test_rect_counts.items():
        all_rect_counts[overlap] += count
    
    all_total = train_total + test_total
    all_actual_pct = []
    all_rect_pct = []
    for overlap in overlap_range:
        all_actual_pct.append(all_actual_counts.get(overlap, 0) / all_total * 100)
        all_rect_pct.append(all_rect_counts.get(overlap, 0) / all_total * 100)
    
    # 创建DataFrame
    pct_df = pd.DataFrame({
        '重叠像素数': overlap_range,
        '训练集_实际(%)': [f"{p:.2f}%" for p in train_actual_pct],
        '训练集_矩形(%)': [f"{p:.2f}%" for p in train_rect_pct],
        '测试集_实际(%)': [f"{p:.2f}%" for p in test_actual_pct],
        '测试集_矩形(%)': [f"{p:.2f}%" for p in test_rect_pct],
        '总体_实际(%)': [f"{p:.2f}%" for p in all_actual_pct],
        '总体_矩形(%)': [f"{p:.2f}%" for p in all_rect_pct]
    })
    
    return pct_df, all_actual_counts, all_rect_counts, all_total

def create_focus_table(train_actual_counts, train_rect_counts, train_total,
                       test_actual_counts, test_rect_counts, test_total):
    """创建0-6像素重点统计表格"""
    
    focus_range = list(range(0, 7))
    
    # 计算百分比
    train_data = []
    test_data = []
    all_data = []
    
    # 训练集
    for overlap in focus_range:
        actual_pct = train_actual_counts.get(overlap, 0) / train_total * 100
        rect_pct = train_rect_counts.get(overlap, 0) / train_total * 100
        train_data.append({
            '重叠像素数': overlap,
            '实际(%)': f"{actual_pct:.2f}%",
            '矩形(%)': f"{rect_pct:.2f}%",
            '差值(%)': f"{actual_pct - rect_pct:+.2f}%"
        })
    
    # 测试集
    for overlap in focus_range:
        actual_pct = test_actual_counts.get(overlap, 0) / test_total * 100
        rect_pct = test_rect_counts.get(overlap, 0) / test_total * 100
        test_data.append({
            '重叠像素数': overlap,
            '实际(%)': f"{actual_pct:.2f}%",
            '矩形(%)': f"{rect_pct:.2f}%",
            '差值(%)': f"{actual_pct - rect_pct:+.2f}%"
        })
    
    # 总体
    all_actual_counts = Counter()
    all_rect_counts = Counter()
    
    for overlap, count in train_actual_counts.items():
        all_actual_counts[overlap] += count
    
    for overlap, count in test_actual_counts.items():
        all_actual_counts[overlap] += count
    
    for overlap, count in train_rect_counts.items():
        all_rect_counts[overlap] += count
    
    for overlap, count in test_rect_counts.items():
        all_rect_counts[overlap] += count
    
    all_total = train_total + test_total
    
    for overlap in focus_range:
        actual_pct = all_actual_counts.get(overlap, 0) / all_total * 100
        rect_pct = all_rect_counts.get(overlap, 0) / all_total * 100
        all_data.append({
            '重叠像素数': overlap,
            '实际(%)': f"{actual_pct:.2f}%",
            '矩形(%)': f"{rect_pct:.2f}%",
            '差值(%)': f"{actual_pct - rect_pct:+.2f}%"
        })
    
    train_focus_df = pd.DataFrame(train_data)
    test_focus_df = pd.DataFrame(test_data)
    all_focus_df = pd.DataFrame(all_data)
    
    return train_focus_df, test_focus_df, all_focus_df

def plot_percentage_comparison(train_actual, train_rect, test_actual, test_rect):
    """绘制百分比对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 训练集实际百分比
    train_actual_counts = Counter(train_actual)
    train_total = len(train_actual)
    train_actual_pct = [train_actual_counts.get(i, 0) / train_total * 100 for i in range(21)]
    
    axes[0, 0].bar(range(21), train_actual_pct, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('训练集 - 实际数字重叠百分比')
    axes[0, 0].set_xlabel('重叠像素数')
    axes[0, 0].set_ylabel('百分比 (%)')
    axes[0, 0].set_xticks(range(0, 21, 2))
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, max(train_actual_pct) * 1.1])
    
    # 训练集矩形百分比
    train_rect_counts = Counter(train_rect)
    train_rect_pct = [train_rect_counts.get(i, 0) / train_total * 100 for i in range(21)]
    
    axes[0, 1].bar(range(21), train_rect_pct, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('训练集 - 矩形区域重叠百分比')
    axes[0, 1].set_xlabel('重叠像素数')
    axes[0, 1].set_ylabel('百分比 (%)')
    axes[0, 1].set_xticks(range(0, 21, 2))
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, max(train_rect_pct) * 1.1])
    
    # 测试集实际百分比
    test_actual_counts = Counter(test_actual)
    test_total = len(test_actual)
    test_actual_pct = [test_actual_counts.get(i, 0) / test_total * 100 for i in range(21)]
    
    axes[1, 0].bar(range(21), test_actual_pct, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('测试集 - 实际数字重叠百分比')
    axes[1, 0].set_xlabel('重叠像素数')
    axes[1, 0].set_ylabel('百分比 (%)')
    axes[1, 0].set_xticks(range(0, 21, 2))
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, max(test_actual_pct) * 1.1])
    
    # 测试集矩形百分比
    test_rect_counts = Counter(test_rect)
    test_rect_pct = [test_rect_counts.get(i, 0) / test_total * 100 for i in range(21)]
    
    axes[1, 1].bar(range(21), test_rect_pct, color='gold', alpha=0.7)
    axes[1, 1].set_title('测试集 - 矩形区域重叠百分比')
    axes[1, 1].set_xlabel('重叠像素数')
    axes[1, 1].set_ylabel('百分比 (%)')
    axes[1, 1].set_xticks(range(0, 21, 2))
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, max(test_rect_pct) * 1.1])
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'overlap_percentage_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_focus_comparison(train_actual, train_rect, test_actual, test_rect):
    """绘制0-6像素重点对比图（百分比）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 训练集0-6像素对比
    train_actual_counts = Counter(train_actual)
    train_rect_counts = Counter(train_rect)
    train_total = len(train_actual)
    
    train_labels = ['0', '1', '2', '3', '4', '5', '6']
    train_actual_pct = [train_actual_counts.get(i, 0) / train_total * 100 for i in range(7)]
    train_rect_pct = [train_rect_counts.get(i, 0) / train_total * 100 for i in range(7)]
    
    x = np.arange(len(train_labels))
    width = 0.35
    
    axes[0].bar(x - width/2, train_actual_pct, width, label='实际数字', color='skyblue')
    axes[0].bar(x + width/2, train_rect_pct, width, label='矩形区域', color='lightcoral')
    axes[0].set_xlabel('粘连像素数')
    axes[0].set_ylabel('百分比 (%)')
    axes[0].set_title('训练集: 实际vs矩形重叠百分比对比 (0-6像素)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(train_labels)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 在柱状图上显示百分比
    for i, v in enumerate(train_actual_pct):
        axes[0].text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    for i, v in enumerate(train_rect_pct):
        axes[0].text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    
    # 测试集0-6像素对比
    test_actual_counts = Counter(test_actual)
    test_rect_counts = Counter(test_rect)
    test_total = len(test_actual)
    
    test_actual_pct = [test_actual_counts.get(i, 0) / test_total * 100 for i in range(7)]
    test_rect_pct = [test_rect_counts.get(i, 0) / test_total * 100 for i in range(7)]
    
    axes[1].bar(x - width/2, test_actual_pct, width, label='实际数字', color='lightgreen')
    axes[1].bar(x + width/2, test_rect_pct, width, label='矩形区域', color='gold')
    axes[1].set_xlabel('粘连像素数')
    axes[1].set_ylabel('百分比 (%)')
    axes[1].set_title('测试集: 实际vs矩形重叠百分比对比 (0-6像素)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(train_labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 在柱状图上显示百分比
    for i, v in enumerate(test_actual_pct):
        axes[1].text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    for i, v in enumerate(test_rect_pct):
        axes[1].text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'overlap_focus_percentage_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # 文件路径
    train_size_path = OUT_DIR / 'train_size_data3.csv'
    test_size_path = OUT_DIR / 'test_size_data3.csv'
    
    train_pairs_path = BUILD_INPUTS / 'mnist_image_pairs_train.csv'
    train_coords_path = BUILD_INPUTS / 'coordinates_train.csv'
    
    test_pairs_path = BUILD_INPUTS / 'mnist_image_pairs_tst.csv'
    test_coords_path = BUILD_INPUTS / 'coordinates_tst.csv'
    
    # 检查文件是否存在
    for path in [train_size_path, test_size_path, train_pairs_path, 
                 train_coords_path, test_pairs_path, test_coords_path]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在 - {path}")
            return
    
    # 1. 加载尺寸信息
    train_size_dict, test_size_dict = load_size_info(train_size_path, test_size_path)
    
    # 2. 加载配对表和坐标表
    print("加载配对表和坐标表...")
    train_pairs = pd.read_csv(train_pairs_path)
    train_coords = pd.read_csv(train_coords_path)
    test_pairs = pd.read_csv(test_pairs_path)
    test_coords = pd.read_csv(test_coords_path)
    
    print(f"训练集: {len(train_pairs)} 个样本")
    print(f"测试集: {len(test_pairs)} 个样本")
    
    # 3. 分析训练集
    print("\n" + "="*60)
    print("分析训练集粘连分布...")
    train_details, train_actual_counts, train_rect_counts, train_actual, train_rect = \
        analyze_dataset(train_pairs, train_coords, train_size_dict)
    
    # 以百分比形式输出训练集统计
    print_percentage_statistics("训练集", train_actual_counts, train_rect_counts, len(train_pairs))
    
    # 4. 分析测试集
    print("\n" + "="*60)
    print("分析测试集粘连分布...")
    test_details, test_actual_counts, test_rect_counts, test_actual, test_rect = \
        analyze_dataset(test_pairs, test_coords, test_size_dict)
    
    # 以百分比形式输出测试集统计
    print_percentage_statistics("测试集", test_actual_counts, test_rect_counts, len(test_pairs))
    
    # 5. 创建百分比统计表格
    print("\n" + "="*60)
    print("创建百分比统计表格...")
    
    # 完整百分比表格 (0-20像素)
    pct_df, all_actual_counts, all_rect_counts, all_total = create_percentage_tables(
        train_actual_counts, train_rect_counts, len(train_pairs),
        test_actual_counts, test_rect_counts, len(test_pairs)
    )
    
    # 0-6像素重点统计表格
    train_focus_df, test_focus_df, all_focus_df = create_focus_table(
        train_actual_counts, train_rect_counts, len(train_pairs),
        test_actual_counts, test_rect_counts, len(test_pairs)
    )
    
    # 6. 打印总体统计
    print("\n" + "="*60)
    print("总体粘连分布统计 (训练集+测试集)")
    print("="*60)
    
    print(f"总样本数: {all_total}")
    print(f"训练集比例: {len(train_pairs)/all_total*100:.2f}%")
    print(f"测试集比例: {len(test_pairs)/all_total*100:.2f}%")
    
    print("\n总体实际数字重叠百分比 (0-6像素):")
    print("-"*40)
    for overlap in range(0, 7):
        count = all_actual_counts.get(overlap, 0)
        percentage = count / all_total * 100
        print(f"重叠{overlap}像素: {percentage:6.2f}%")
    
    print("\n总体矩形区域重叠百分比 (0-6像素):")
    print("-"*40)
    for overlap in range(0, 7):
        count = all_rect_counts.get(overlap, 0)
        percentage = count / all_total * 100
        print(f"重叠{overlap}像素: {percentage:6.2f}%")
    
    # 7. 计算并打印关键指标
    print("\n" + "="*60)
    print("关键指标统计")
    print("="*60)
    
    # 无粘连比例
    train_no_overlap_actual = train_actual_counts.get(0, 0) / len(train_pairs) * 100
    train_no_overlap_rect = train_rect_counts.get(0, 0) / len(train_pairs) * 100
    test_no_overlap_actual = test_actual_counts.get(0, 0) / len(test_pairs) * 100
    test_no_overlap_rect = test_rect_counts.get(0, 0) / len(test_pairs) * 100
    all_no_overlap_actual = all_actual_counts.get(0, 0) / all_total * 100
    all_no_overlap_rect = all_rect_counts.get(0, 0) / all_total * 100
    
    # 1-6像素粘连比例
    train_1_6_actual = sum(train_actual_counts.get(i, 0) for i in range(1, 7)) / len(train_pairs) * 100
    train_1_6_rect = sum(train_rect_counts.get(i, 0) for i in range(1, 7)) / len(train_pairs) * 100
    test_1_6_actual = sum(test_actual_counts.get(i, 0) for i in range(1, 7)) / len(test_pairs) * 100
    test_1_6_rect = sum(test_rect_counts.get(i, 0) for i in range(1, 7)) / len(test_pairs) * 100
    all_1_6_actual = sum(all_actual_counts.get(i, 0) for i in range(1, 7)) / all_total * 100
    all_1_6_rect = sum(all_rect_counts.get(i, 0) for i in range(1, 7)) / all_total * 100
    
    print("无粘连比例 (0像素):")
    print(f"  训练集 - 实际: {train_no_overlap_actual:.2f}%, 矩形: {train_no_overlap_rect:.2f}%")
    print(f"  测试集 - 实际: {test_no_overlap_actual:.2f}%, 矩形: {test_no_overlap_rect:.2f}%")
    print(f"  总体   - 实际: {all_no_overlap_actual:.2f}%, 矩形: {all_no_overlap_rect:.2f}%")
    
    print("\n1-6像素粘连比例:")
    print(f"  训练集 - 实际: {train_1_6_actual:.2f}%, 矩形: {train_1_6_rect:.2f}%")
    print(f"  测试集 - 实际: {test_1_6_actual:.2f}%, 矩形: {test_1_6_rect:.2f}%")
    print(f"  总体   - 实际: {all_1_6_actual:.2f}%, 矩形: {all_1_6_rect:.2f}%")
    
    # 8. 保存统计结果
    print("\n" + "="*60)
    print("保存统计结果...")
    
    # 保存百分比表格
    pct_df.to_csv(OUT_DIR / 'overlap_percentage_all.csv', index=False, encoding='utf-8-sig')
    print("完整百分比表格保存到: overlap_percentage_all.csv")
    
    # 保存重点统计表格
    train_focus_df.to_csv(OUT_DIR / 'train_focus_percentage.csv', index=False, encoding='utf-8-sig')
    test_focus_df.to_csv(OUT_DIR / 'test_focus_percentage.csv', index=False, encoding='utf-8-sig')
    all_focus_df.to_csv(OUT_DIR / 'all_focus_percentage.csv', index=False, encoding='utf-8-sig')
    print("重点统计表格保存到:")
    print("  - train_focus_percentage.csv")
    print("  - test_focus_percentage.csv")
    print("  - all_focus_percentage.csv")
    
    # 保存关键指标汇总
    key_metrics = pd.DataFrame({
        '指标': ['无粘连比例(0像素)', '1-6像素粘连比例'],
        '训练集_实际(%)': [f"{train_no_overlap_actual:.2f}%", f"{train_1_6_actual:.2f}%"],
        '训练集_矩形(%)': [f"{train_no_overlap_rect:.2f}%", f"{train_1_6_rect:.2f}%"],
        '测试集_实际(%)': [f"{test_no_overlap_actual:.2f}%", f"{test_1_6_actual:.2f}%"],
        '测试集_矩形(%)': [f"{test_no_overlap_rect:.2f}%", f"{test_1_6_rect:.2f}%"],
        '总体_实际(%)': [f"{all_no_overlap_actual:.2f}%", f"{all_1_6_actual:.2f}%"],
        '总体_矩形(%)': [f"{all_no_overlap_rect:.2f}%", f"{all_1_6_rect:.2f}%"]
    })
    
    key_metrics.to_csv(OUT_DIR / 'key_metrics_percentage.csv', index=False, encoding='utf-8-sig')
    print("关键指标保存到: key_metrics_percentage.csv")
    
    # 9. 可视化
    print("\n生成可视化图表...")
    plot_percentage_comparison(train_actual, train_rect, test_actual, test_rect)
    plot_focus_comparison(train_actual, train_rect, test_actual, test_rect)
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()