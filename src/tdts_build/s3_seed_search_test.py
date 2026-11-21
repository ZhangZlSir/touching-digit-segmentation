#s3_seed_search_test.py
# 遍历若干随机种子，逐个重建 TDTS-TS 的测试集配对表并计算 100 个类别对的 MAD，
# 选出使类别分布最均衡的种子。用于评估「右字符均衡采样」在不同随机种子下的稳定性。
#
# 用法：
#   python s3_seed_search_test.py --num-seeds 1000 --start-seed 0
#   python s3_seed_search_test.py --num-seeds 3 --start-seed 115 --generate
#
# 输出（--generate 时）：results/seed_search/mnist_image_pairs_tst_optimized.csv

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / 'data' / 'mnist_compact_20x20'
OUT_DIR = PROJECT_ROOT / 'results' / 'seed_search'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description='Search the seed minimizing the TDTS-TS label-pair MAD')
    p.add_argument('--num-seeds', type=int, default=1000)
    p.add_argument('--start-seed', type=int, default=0)
    p.add_argument('--generate', action='store_true', help='generate the CSV for the best seed without asking')
    p.add_argument('--analyze', action='store_true', help='print MAD distribution analysis')
    return p.parse_args()


args = parse_args()

# 加载 MNIST 测试集（只需要加载一次）
mnist_tst = MNIST(str(DATA_ROOT), train=False, download=False)

def generate_test_set_with_seed(seed, return_mad_only=True):
    """使用指定种子生成测试集，返回MAD和配对信息"""
    # 重置随机种子
    random.seed(seed)
    
    # 左侧图片索引从 0 到 9999
    left_indices = list(range(10000))
    
    # 记录右侧图片每个数字的索引值
    right_digit_indices = {i: [] for i in range(10)}
    for idx in range(10000):
        label = int(mnist_tst[idx][1])
        right_digit_indices[label].append(idx)
    
    # 处理右侧图片，使每类图片都有 1000 张
    for digit in range(10):
        indices = right_digit_indices[digit]
        if len(indices) > 1000:
            right_digit_indices[digit] = random.sample(indices, 1000)
        elif len(indices) < 1000:
            while len(indices) < 1000:
                indices.append(random.choice(indices))
            right_digit_indices[digit] = indices
    
    # 合并右侧预选集索引
    right_preselected_indices = []
    for digit in range(10):
        right_preselected_indices.extend(right_digit_indices[digit])
    
    # 打乱右侧预选集索引顺序
    random.shuffle(right_preselected_indices)
    
    # 进行配对
    pairings = []
    for left_index in left_indices:
        left_label = int(mnist_tst[left_index][1])
        right_index = right_preselected_indices.pop(0) 
        right_label = int(mnist_tst[right_index][1])
        pairings.append((left_index, right_index, left_label, right_label))
    
    # 统计双字符分布并计算MAD
    label_counts = {i: {j: 0 for j in range(10)} for i in range(10)}
    for _, _, left_label, right_label in pairings:
        label_counts[left_label][right_label] += 1
    
    # 将计数转换为numpy数组
    count_matrix = np.zeros((10, 10), dtype=int)
    for i in range(10):
        for j in range(10):
            count_matrix[i][j] = label_counts[i][j]
    
    # 计算MAD
    flat_counts = count_matrix.flatten()
    mean_count = np.mean(flat_counts)
    mad = np.mean(np.abs(flat_counts - mean_count))
    relative_mad = (mad / mean_count) * 100
    
    if return_mad_only:
        return mad, relative_mad, seed
    else:
        return mad, relative_mad, seed, pairings, label_counts

def find_best_seed(num_seeds_to_try=1000, start_seed=0):
    """尝试多个随机种子，找到使MAD最小的种子"""
    print(f"开始寻找最优随机种子...")
    print(f"将尝试 {num_seeds_to_try} 个种子（{start_seed} 到 {start_seed+num_seeds_to_try-1}）")
    print("-" * 60)
    
    best_mad = float('inf')
    best_relative_mad = float('inf')
    best_seed = None
    mad_list = []
    
    start_time = time.time()
    
    for seed in range(start_seed, start_seed + num_seeds_to_try):
        mad, relative_mad, _ = generate_test_set_with_seed(seed, return_mad_only=True)
        mad_list.append((seed, mad, relative_mad))
        
        # 更新最优结果
        if mad < best_mad:
            best_mad = mad
            best_relative_mad = relative_mad
            best_seed = seed
        
        # 每100个种子打印一次进度
        if (seed - start_seed + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"已尝试 {seed - start_seed + 1}/{num_seeds_to_try} 个种子，"
                  f"当前最佳MAD: {best_mad:.2f} (相对: {best_relative_mad:.2f}%)，"
                  f"最佳种子: {best_seed}，耗时: {elapsed:.1f}秒")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 排序查看MAD分布
    mad_list.sort(key=lambda x: x[1])
    
    print("\n" + "=" * 60)
    print("搜索结果汇总:")
    print("=" * 60)
    print(f"尝试种子数: {num_seeds_to_try}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每个种子: {total_time/num_seeds_to_try:.3f}秒")
    print(f"\n最佳种子: {best_seed}")
    print(f"最佳MAD: {best_mad:.2f}")
    print(f"最佳相对MAD: {best_relative_mad:.2f}%")
    
    # 理论MAD
    theoretical_mad = 6.29
    theoretical_relative_mad = 6.29
    
    print(f"\n对比理论值:")
    print(f"  理论MAD: {theoretical_mad:.2f}")
    print(f"  实际最佳MAD: {best_mad:.2f}")
    print(f"  MAD降低: {theoretical_mad - best_mad:.2f}")
    print(f"  相对改善: {((theoretical_mad - best_mad)/theoretical_mad*100):.2f}%")
    
    # 显示前10个最佳种子
    print(f"\n前10个最佳种子:")
    print("种子\tMAD\t相对MAD")
    print("-" * 30)
    for i in range(min(10, len(mad_list))):
        seed, mad, rel_mad = mad_list[i]
        print(f"{seed}\t{mad:.2f}\t{rel_mad:.2f}%")
    
    return best_seed, best_mad, best_relative_mad, mad_list

def generate_final_test_set(seed, output_filename='mnist_image_pairs_tst_optimized.csv'):
    """使用指定种子生成最终的测试集CSV文件"""
    print(f"\n使用最优种子 {seed} 生成最终测试集...")
    
    mad, relative_mad, seed, pairings, label_counts = generate_test_set_with_seed(
        seed, return_mad_only=False
    )
    
    # 输出统计信息
    print(f"双字符分布统计:")
    print(f"  每类平均样本数: {np.mean(list(label_counts[i][j] for i in range(10) for j in range(10))):.2f}")
    print(f"  MAD: {mad:.2f}")
    print(f"  相对MAD: {relative_mad:.2f}%")
    
    # 输出 0 - 9 图片的实际张数
    print("\n双字符标签分布:")
    for i in range(10):
        for j in range(10):
            print(f"标签对 ({i}, {j}) 的数量: {label_counts[i][j]}")
    
    # 输出左侧所有图片的索引值和、右侧图片的索引值和
    left_index_sum = sum([pair[0] for pair in pairings])
    right_index_sum = sum([pair[1] for pair in pairings])
    print(f"\n左侧图片索引值和: {left_index_sum}")
    print(f"右侧图片索引值和: {right_index_sum}")
    
    # 生成 CSV 文件
    with open(output_filename, mode='w', newline='') as csv_file:
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
    
    print(f"\nCSV 文件 '{output_filename}' 已生成。")
    
    return pairings, label_counts

# 主程序
if __name__ == "__main__":
    # 第一步：寻找最优种子
    best_seed, best_mad, best_rel_mad, mad_list = find_best_seed(
        num_seeds_to_try=args.num_seeds,
        start_seed=args.start_seed
    )

    # 第二步：使用最优种子生成最终测试集
    if args.generate:
        generate_final_test_set(best_seed, str(OUT_DIR / 'mnist_image_pairs_tst_optimized.csv'))

    # 第三步：可选 - 分析MAD分布
    if args.analyze:
        print("\nMAD分布分析:")
        print("=" * 60)
        
        # 提取所有MAD值
        mad_values = [item[1] for item in mad_list]
        
        print(f"MAD范围: {min(mad_values):.2f} - {max(mad_values):.2f}")
        print(f"MAD平均值: {np.mean(mad_values):.2f}")
        print(f"MAD标准差: {np.std(mad_values):.2f}")
        print(f"MAD中位数: {np.median(mad_values):.2f}")
        
        # 统计小于理论MAD的比例
        theoretical_mad = 6.29
        below_theoretical = sum(1 for m in mad_values if m < theoretical_mad)
        print(f"\n小于理论MAD({theoretical_mad:.2f})的种子数: {below_theoretical}/{len(mad_values)}")
        print(f"比例: {below_theoretical/len(mad_values)*100:.1f}%")
        
        # 统计小于当前种子(42)MAD的比例
        # 先计算种子42的MAD
        mad_42, rel_mad_42, _ = generate_test_set_with_seed(42, return_mad_only=True)
        below_42 = sum(1 for m in mad_values if m < mad_42)
        print(f"\n小于当前种子(42)MAD({mad_42:.2f})的种子数: {below_42}/{len(mad_values)}")
        print(f"比例: {below_42/len(mad_values)*100:.1f}%")
        
        # 建议
        print(f"\n建议:")
        if best_mad < theoretical_mad:
            print(f"✅ 找到了MAD小于理论值的种子({best_seed})!")
            print(f"   理论MAD: {theoretical_mad:.2f}, 最优MAD: {best_mad:.2f}")
        else:
            print(f"⚠️  未找到MAD小于理论值的种子，但找到了当前最优种子({best_seed})")
            print(f"   理论MAD: {theoretical_mad:.2f}, 最优MAD: {best_mad:.2f}")