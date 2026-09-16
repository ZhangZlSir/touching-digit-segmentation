# batch_main_recog2.py - 分批处理版本
import torch
from torchvision import datasets, transforms
from u_left_right_recog import LeftRightRecognizer
import pandas as pd
import os
from inputimeout import inputimeout, TimeoutOccurred
import gc
import datetime
from pathlib import Path


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
    
    num_cuts = 1#只剩下1条路径了
    
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
    output_file = os.path.join(OUTPUT_DIR, f'mrm_test_{suffix}.csv')#保存到outPut文件夹
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存到 {output_file}")
    
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / 'data' / 'tdts_dataset'
DIGIT_MODEL_PATH = PROJECT_ROOT / 'models' / 'tdts_mrm.pth'

def main():
    """主程序：分批处理MNIST图片""" 
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor()])
    img_data = datasets.MNIST(DATA_ROOT, download=True, transform=transform, train=False)
    total_samples = len(img_data)
    
    print(f"MNIST训练集总样本数: {total_samples}")
    
    # 初始化识别器
    recognizer = LeftRightRecognizer(DIGIT_MODEL_PATH)
    print("模型加载完成")
    
    # 用户输入处理参数
    print("\n请输入处理参数:")
    start_index = int(input(f"起始索引 (0-{total_samples - 1}, 默认0): ") or 0)
    batch_size = int(input("批次大小 (默认1500): ") or 1500)
    
    start_index = max(0, min(start_index, total_samples - 1))
    batch_size = min(batch_size, total_samples - start_index)
    
    print(f"\n开始处理: 从索引 {start_index} 开始, 处理 {batch_size} 个样本")
    
    # 处理批次y
    current_index = start_index
    batch_number = 1
    
    while current_index < total_samples and batch_size > 0:
        remaining = total_samples - current_index
        current_batch_size = min(batch_size, remaining)
        
        suffix = f"{current_index:05d}_{current_index + current_batch_size - 1:05d}"
        current_index = process_batch(recognizer, img_data, current_index, current_batch_size, suffix)
        
        batch_number += 1
        
        gc.collect()  # 清理内存

        if current_index < total_samples:
            try:
                continue_processing = inputimeout(prompt=f"\n已处理到索引 {current_index}, 是否继续? (输入n结束, 其他任意键或2秒后自动继续): ", timeout=2)
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