#evaluate.py - batched TDTS evaluation
import torch
from torchvision import datasets, transforms
from pipeline import LeftRightRecognizer
import pandas as pd
import os
from inputimeout import inputimeout, TimeoutOccurred
import gc
import datetime
from pathlib import Path
import argparse

def process_batch(recognizer, img_data, start_index, batch_size, output_suffix):
    """处理指定批次并保存结果 - 使用批量处理优化版本"""
    
    # 处理当前批次
    end_index = min(start_index + batch_size, len(img_data))
    
    # 提取当前批次的图片数据
    batch_data = []
    for image_index in range(start_index, end_index):
        batch_data.append(img_data[image_index])
    
    # 使用批量处理方法处理当前批次
    batch_results = recognizer.process_batch_images(batch_data, len(batch_data))
    
    # 调整图片ID，确保与原始索引一致
    for i, row in enumerate(batch_results):
        # 将相对索引转换为绝对索引
        row[0] = start_index + i + 1
    
    # 保存当前批次结果
    save_batch_results(batch_results, output_suffix)
    return end_index

def save_batch_results(batch_results, suffix):
    """保存批次结果到CSV文件"""
    if not batch_results:
        return
    
    num_cuts = 5#只剩下5条路径了
    
    # 创建列名
    columns = ['ImgID', 'TrueLbl']  # 图片序号和真实标签
    for i in range(num_cuts):
        columns.extend([
            f'P{i+1}ID',  # 路径标识符 (Path ID)
            f'P{i+1}OK',  # 是否正确 (OK)
            f'P{i+1}VT',  # 投票数 (Vote)
            f'P{i+1}LC',  # 左侧置信度 (Left Confidence)
            f'P{i+1}RC'   # 右侧置信度 (Right Confidence)
        ])
    
    # 创建DataFrame并保存
    df = pd.DataFrame(batch_results, columns=columns)
    SCRIPT_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = SCRIPT_DIR / 'outPut'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)     # 不存在则创建
    output_file = os.path.join(OUTPUT_DIR, f'tdts_{suffix}.csv')#保存到outPut文件夹
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存到 {output_file}")
    
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / 'data' / 'tdts_dataset'
DIGIT_MODEL_PATH = PROJECT_ROOT / 'models' / 'tdts_mrm.pth'   # default, kept for backward compatibility
# 识别模型路径：CRM/NRM/MRM，由 src/recognizer_training/train_tdts_cnn.py 训练后写入 models/
MODEL_PATHS = {name: PROJECT_ROOT / 'models' / f'tdts_{name}.pth' for name in ('crm', 'nrm', 'mrm')}


def parse_args():
    p = argparse.ArgumentParser(description='TDTS end-to-end evaluation: cutting domain + recognition')
    p.add_argument('--model', choices=('crm', 'nrm', 'mrm'), default='mrm', help='recognition model (default mrm)')
    p.add_argument('--split', choices=('tr', 'ts'), default='tr', help='tr=TDTS-TR (train source), ts=TDTS-TS (test source)')
    p.add_argument('--start', type=int, default=None, help='start index (ask interactively if omitted)')
    p.add_argument('--batch-size', dest='batch_size', type=int, default=None, help='batch size (ask interactively if omitted)')
    p.add_argument('--yes', action='store_true', help='do not prompt between batches')
    return p.parse_args()



def main():
    """主程序：分批处理MNIST图片""" 
    args = parse_args()
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor()])
    img_data = datasets.MNIST(DATA_ROOT, download=False, transform=transform, train=(args.split == 'tr'))
    total_samples = len(img_data)
    
    print(f"MNIST训练集总样本数: {total_samples}")
    
    # 初始化识别器
    recognizer = LeftRightRecognizer(MODEL_PATHS[args.model])
    print("模型加载完成")
    
    # 用户输入处理参数
    print("\n请输入处理参数:")
    start_index = args.start if args.start is not None else int(input(f"Start index (0-{total_samples - 1}, default 0): ") or 0)
    batch_size = args.batch_size if args.batch_size is not None else int(input("批次大小 (默认1500): ") or 1500)
    
    start_index = max(0, min(start_index, total_samples - 1))
    batch_size = min(batch_size, total_samples - start_index)
    
    print(f"\n开始处理: 从索引 {start_index} 开始, 处理 {batch_size} 个样本")
    
    # 处理批次y
    current_index = start_index
    batch_number = 1
    
    while current_index < total_samples and batch_size > 0:
        remaining = total_samples - current_index
        current_batch_size = min(batch_size, remaining)
        
        suffix = f"{args.model}_{args.split}_{current_index:05d}_{current_index + current_batch_size - 1:05d}"
        current_index = process_batch(recognizer, img_data, current_index, current_batch_size, suffix)
        
        batch_number += 1
        
        gc.collect()  # 清理内存

        if current_index < total_samples:
            try:
                continue_processing = 'y' if args.yes else inputimeout(prompt=f"\n已处理到索引 {current_index}, 是否继续? (输入n结束, 其他任意键或2秒后自动继续): ", timeout=2)
                if continue_processing.lower() == 'n':
                    print("用户选择结束处理。")
                    break
                else:
                    print("继续处理...")
            except TimeoutOccurred:
                print("\n超时未输入，自动继续...")
        else:
            print("所有样本处理完成!")
            
            print("处理结束")

if __name__ == "__main__":
    main()