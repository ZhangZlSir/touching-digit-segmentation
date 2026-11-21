import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Pool, cpu_count
import time

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_IN = PROJECT_ROOT / 'data' / 'mnist_compact_20x20'          # 20x20 compact MNIST characters
DATA_TDTS = PROJECT_ROOT / 'data' / 'tdts_dataset'               # TDTS two-digit touching set
BUILD_INPUTS = PROJECT_ROOT / 'data' / 'tdts_build_inputs'       # coordinates / pair index CSVs
OUT_DIR = PROJECT_ROOT / 'results' / 'stats'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 给定的训练集和测试集0-9的数字数量
train_counts = np.array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
test_counts = np.array([980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009])

# 创建标签列表（重复每个标签对应次数）
def create_label_list(counts):
    """根据每个数字的数量创建标签列表"""
    labels = []
    for digit in range(10):
        labels.extend([digit] * counts[digit])
    return np.array(labels)

# 计算MAD和CV
def calculate_stats(class_counts):
    """计算MAD和CV"""
    mean = np.mean(class_counts)
    mad = np.mean(np.abs(class_counts - mean))
    cv = np.std(class_counts) / mean if mean != 0 else 0
    return mad, cv

# 双标签编码规则
def get_dual_label(left_label, right_label):
    """根据左侧和右侧标签计算双标签"""
    dual_label = left_label * 10 + right_label
    if dual_label < 10:
        dual_label += 100
    return dual_label

# 单次模拟任务
def run_simulation_task(args):
    """执行单次模拟的任务函数"""
    left_labels, right_labels, simulation_id = args
    # 设置不同的随机种子确保每次模拟独立
    np.random.seed(simulation_id + int(time.time() * 1000) % 1000000)
    
    # 随机打乱右侧标签（不放回）
    shuffled_right = np.random.permutation(right_labels)
    
    # 配对左侧和右侧标签
    dual_labels = []
    for i in range(len(left_labels)):
        left = left_labels[i]
        right = shuffled_right[i]
        dual_label = get_dual_label(left, right)
        dual_labels.append(dual_label)
    
    # 统计00-99各类别数量
    class_counts = np.zeros(110)  # 索引0-109，实际使用0-109
    for label in dual_labels:
        class_counts[label] += 1
    
    # 提取00-99类别的数量（编码为0-99和100-109）
    valid_counts = []
    for i in range(10):  # 00-09
        valid_counts.append(class_counts[100 + i])
    for i in range(10, 100):  # 10-99
        valid_counts.append(class_counts[i])
    
    # 计算统计量
    mad, cv = calculate_stats(valid_counts)
    return mad, cv

# 并行蒙特卡洛模拟
def parallel_monte_carlo_simulation(left_labels, right_labels, num_simulations=1000, num_processes=4):
    """使用多进程并行执行蒙特卡洛模拟"""
    
    # 准备任务参数
    tasks = [(left_labels.copy(), right_labels.copy(), i) for i in range(num_simulations)]
    
    # 使用进程池
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_simulation_task, tasks)
    
    # 分离MAD和CV结果
    mad_results = [r[0] for r in results]
    cv_results = [r[1] for r in results]
    
    return np.array(mad_results), np.array(cv_results)

# 串行蒙特卡洛模拟（备选）
def monte_carlo_simulation(left_labels, right_labels, num_simulations=1000):
    """执行蒙特卡洛模拟（串行版本）"""
    mad_results = []
    cv_results = []
    
    for sim in range(num_simulations):
        # 随机打乱右侧标签（不放回）
        shuffled_right = np.random.permutation(right_labels)
        
        # 配对左侧和右侧标签
        dual_labels = []
        for i in range(len(left_labels)):
            left = left_labels[i]
            right = shuffled_right[i]
            dual_label = get_dual_label(left, right)
            dual_labels.append(dual_label)
        
        # 统计00-99各类别数量
        class_counts = np.zeros(110)  # 索引0-109，实际使用0-109
        for label in dual_labels:
            class_counts[label] += 1
        
        # 提取00-99类别的数量（编码为0-99和100-109）
        valid_counts = []
        for i in range(10):  # 00-09
            valid_counts.append(class_counts[100 + i])
        for i in range(10, 100):  # 10-99
            valid_counts.append(class_counts[i])
        
        # 计算统计量
        mad, cv = calculate_stats(valid_counts)
        mad_results.append(mad)
        cv_results.append(cv)
    
    return np.array(mad_results), np.array(cv_results)

# 主程序
def main():
    # 设置并行进程数（默认为CPU核心数，但可以指定为4）
    num_processes = 4
    print(f"使用 {num_processes} 个进程进行并行计算...")
    print("=" * 60)
    
    # 训练集模拟
    print("训练集模拟：")
    print("-" * 30)
    train_left = create_label_list(train_counts)
    train_right = create_label_list(train_counts)
    
    start_time = time.time()
    train_mads, train_cvs = parallel_monte_carlo_simulation(
        train_left, train_right, 1000, num_processes
    )
    train_time = time.time() - start_time
    
    print(f"训练集模拟完成，耗时: {train_time:.2f} 秒")
    print(f"训练集MAD: {np.mean(train_mads):.2f} ± {np.std(train_mads):.2f}")
    print(f"训练集MAD范围: [{np.min(train_mads):.2f}, {np.max(train_mads):.2f}]")
    print(f"训练集MAD中位数: {np.median(train_mads):.2f}")
    print()
    print(f"训练集CV: {np.mean(train_cvs):.4f} ± {np.std(train_cvs):.4f}")
    print(f"训练集CV范围: [{np.min(train_cvs):.4f}, {np.max(train_cvs):.4f}]")
    print(f"训练集CV中位数: {np.median(train_cvs):.4f}")
    print()
    
    # 测试集模拟
    print("测试集模拟：")
    print("-" * 30)
    test_left = create_label_list(test_counts)
    test_right = create_label_list(test_counts)
    
    start_time = time.time()
    test_mads, test_cvs = parallel_monte_carlo_simulation(
        test_left, test_right, 1000, num_processes
    )
    test_time = time.time() - start_time
    
    print(f"测试集模拟完成，耗时: {test_time:.2f} 秒")
    print(f"测试集MAD: {np.mean(test_mads):.2f} ± {np.std(test_mads):.2f}")
    print(f"测试集MAD范围: [{np.min(test_mads):.2f}, {np.max(test_mads):.2f}]")
    print(f"测试集MAD中位数: {np.median(test_mads):.2f}")
    print()
    print(f"测试集CV: {np.mean(test_cvs):.4f} ± {np.std(test_cvs):.4f}")
    print(f"测试集CV范围: [{np.min(test_cvs):.4f}, {np.max(test_cvs):.4f}]")
    print(f"测试集CV中位数: {np.median(test_cvs):.4f}")
    print()
    
    # 创建结果汇总表
    results_df = pd.DataFrame({
        'Dataset': ['Training', 'Testing'],
        'Time_seconds': [train_time, test_time],
        'Mean_MAD': [np.mean(train_mads), np.mean(test_mads)],
        'Std_MAD': [np.std(train_mads), np.std(test_mads)],
        'Min_MAD': [np.min(train_mads), np.min(test_mads)],
        'Max_MAD': [np.max(train_mads), np.max(test_mads)],
        'Median_MAD': [np.median(train_mads), np.median(test_mads)],
        'Mean_CV': [np.mean(train_cvs), np.mean(test_cvs)],
        'Std_CV': [np.std(train_cvs), np.std(test_cvs)],
        'Min_CV': [np.min(train_cvs), np.min(test_cvs)],
        'Max_CV': [np.max(train_cvs), np.max(test_cvs)],
        'Median_CV': [np.median(train_cvs), np.median(test_cvs)]
    })
    
    # 显示详细结果
    print("=" * 60)
    print("详细统计结果:")
    print(results_df.to_string(index=False))
    print()
    
    # 计算置信区间（95%）
    print("95%置信区间:")
    print("-" * 30)
    
    # 训练集
    train_mad_ci = stats.norm.interval(0.95, loc=np.mean(train_mads), scale=np.std(train_mads)/np.sqrt(1000))
    train_cv_ci = stats.norm.interval(0.95, loc=np.mean(train_cvs), scale=np.std(train_cvs)/np.sqrt(1000))
    
    print(f"训练集MAD 95%置信区间: [{train_mad_ci[0]:.2f}, {train_mad_ci[1]:.2f}]")
    print(f"训练集CV 95%置信区间: [{train_cv_ci[0]:.4f}, {train_cv_ci[1]:.4f}]")
    print()
    
    # 测试集
    test_mad_ci = stats.norm.interval(0.95, loc=np.mean(test_mads), scale=np.std(test_mads)/np.sqrt(1000))
    test_cv_ci = stats.norm.interval(0.95, loc=np.mean(test_cvs), scale=np.std(test_cvs)/np.sqrt(1000))
    
    print(f"测试集MAD 95%置信区间: [{test_mad_ci[0]:.2f}, {test_mad_ci[1]:.2f}]")
    print(f"测试集CV 95%置信区间: [{test_cv_ci[0]:.4f}, {test_cv_ci[1]:.4f}]")
    
    # 保存结果到文件
    results_df.to_csv(OUT_DIR / 'monte_carlo_results.csv', index=False)
    print(f"\n结果已保存到 'monte_carlo_results.csv'")
    
    # 可选：绘制分布直方图（需要matplotlib）
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 训练集MAD分布
        axes[0, 0].hist(train_mads, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(np.mean(train_mads), color='red', linestyle='--', label=f'Mean: {np.mean(train_mads):.2f}')
        axes[0, 0].set_xlabel('MAD')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Training Set MAD Distribution (1000 simulations)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 训练集CV分布
        axes[0, 1].hist(train_cvs, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(np.mean(train_cvs), color='red', linestyle='--', label=f'Mean: {np.mean(train_cvs):.4f}')
        axes[0, 1].set_xlabel('CV')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Training Set CV Distribution (1000 simulations)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 测试集MAD分布
        axes[1, 0].hist(test_mads, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(np.mean(test_mads), color='red', linestyle='--', label=f'Mean: {np.mean(test_mads):.2f}')
        axes[1, 0].set_xlabel('MAD')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Testing Set MAD Distribution (1000 simulations)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 测试集CV分布
        axes[1, 1].hist(test_cvs, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(np.mean(test_cvs), color='red', linestyle='--', label=f'Mean: {np.mean(test_cvs):.4f}')
        axes[1, 1].set_xlabel('CV')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Testing Set CV Distribution (1000 simulations)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'monte_carlo_simulation_results.png', dpi=300, bbox_inches='tight')
        print("\n分布直方图已保存为 'monte_carlo_simulation_results.png'")
        plt.show()
        
    except ImportError:
        print("\n注意：未安装matplotlib，跳过绘图步骤。")
        print("如需绘图，请运行: pip install matplotlib")

# 备用方案：如果并行有问题，使用串行版本
def main_serial():
    print("使用串行版本进行计算...")
    # ...（使用串行版本的代码）

if __name__ == "__main__":
    # 确保在Windows系统上正确运行多进程
    main()