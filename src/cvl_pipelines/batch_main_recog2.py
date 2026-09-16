# batch_main_recog2.py - 分批处理版本（适配CVL数据集）
import torch
from torchvision import transforms
import pandas as pd
import os
from inputimeout import inputimeout, TimeoutOccurred
import gc
import datetime
import glob
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

# -------------------- 自定义数据集类（从CVL改进版复制） --------------------
class PNGDigitDataset(Dataset):
    """从PNG文件读取数字图片，标签从文件名解析（第一个'-'前的数字）"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, '*.png'))
        if not self.image_paths:
            raise RuntimeError(f"在 {root_dir} 中没有找到PNG图片")
        self.labels = []
        for path in self.image_paths:
            fname = os.path.basename(path)
            label_str = fname.split('-')[0]
            if not label_str.isdigit():
                raise ValueError(f"无法从文件名 {fname} 解析数字标签")
            self.labels.append(int(label_str))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------- 原有功能函数（微调） --------------------
def process_batch1(recognizer, img_data, start_index, batch_size, output_suffix):
    """处理指定批次并保存结果"""
    start_time = datetime.datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] 开始处理批次 {start_index}-{start_index+batch_size}")
    print(f"开始处理批次: 起始索引={start_index}, 批次大小={batch_size}")
    
    end_index = min(start_index + batch_size, len(img_data))
    batch_results = []
    
    for image_index in range(start_index, end_index):
        img, label = img_data[image_index]
        cut_results, path_id, path_lbl, true_label = recognizer.process_single_image(img, label)
        
        row = [image_index + 1,  true_label]
        for i in range(len(path_id)):
            start_idx = i * 3
            row.extend([
                path_id[i],
                cut_results[start_idx],
                path_lbl[i],
                cut_results[start_idx + 1],
                cut_results[start_idx + 2]
            ])
        batch_results.append(row)
    
    save_batch_results(batch_results, output_suffix)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"[{end_time.strftime('%H:%M:%S')}] 批次完成, 耗时: {duration:.2f}秒")
    print(f"批次处理完成: 处理了 {len(batch_results)} 个样本")
    return end_index

def process_batch(recognizer, img_data, start_index, batch_size, output_suffix):
    """处理指定批次并保存结果 - 使用批量处理优化版本"""
    end_index = min(start_index + batch_size, len(img_data))
    batch_data = []
    for image_index in range(start_index, end_index):
        batch_data.append(img_data[image_index])
    
    batch_results = recognizer.process_batch_images(batch_data, len(batch_data))
    
    # 调整图片ID，确保与原始索引一致
    for i, row in enumerate(batch_results):
        row[0] = start_index + i + 1
    
    save_batch_results(batch_results, output_suffix)
    return end_index

def save_batch_results(batch_results, suffix):
    """保存批次结果到CSV文件"""
    if not batch_results:
        return
    
    # 【改动】动态计算切割数量，不再固定为5
    num_cuts = (len(batch_results[0]) - 2) // 5  # 每路径5个字段：ID, OK, VT, LC, RC
    
    columns = ['ImgID', 'TrueLbl']
    for i in range(num_cuts):
        columns.extend([
            f'P{i+1}ID',
            f'P{i+1}OK',
            f'P{i+1}VT',
            f'P{i+1}LC',
            f'P{i+1}RC'
        ])
    
    df = pd.DataFrame(batch_results, columns=columns)
    output_file = os.path.join('outPutCVlTest', f'mix22_test_{suffix}.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存到 {output_file}")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_img  = PROJECT_ROOT / 'data' / 'cvl_double_char' 
DATA_model = PROJECT_ROOT / 'models' / 'cvl_cnn_improved.pth'

def main():
    """主程序：分批处理CVL数字图片"""
    # 【改动】使用PNGDigitDataset加载数据（路径可调整）
    data_root = DATA_img  # 请根据实际测试集路径修改
    transform = transforms.Compose([transforms.ToTensor()])
    img_data = PNGDigitDataset(root_dir=data_root, transform=transform)
    total_samples = len(img_data)
    print("检查前5个样本的标签和文件名：")
    for i in range(min(5, len(img_data))):
        img, label = img_data[i]
        img_path = img_data.image_paths[i]
        print(f"i={i}, label={label}, path={img_path}")
    
    print(f"数据集总样本数: {total_samples}")
    
    # 初始化识别器（需使用改进后的模型权重）
    from u_left_right_recog1_cvl import LeftRightRecognizer
    recognizer = LeftRightRecognizer(DATA_model)  # 使用改进版模型
    print("模型加载完成")
    
    print("\n请输入处理参数:")
    start_index = int(input("起始索引 (0-{total_samples-1}, 默认0): ") or 0)
    batch_size = int(input("批次大小 (默认1500): ") or 1500)
    
    start_index = max(0, min(start_index, total_samples - 1))
    batch_size = min(batch_size, total_samples - start_index)
    
    print(f"\n开始处理: 从索引 {start_index} 开始, 处理 {batch_size} 个样本")
    
    current_index = start_index
    batch_number = 1
    
    while current_index < total_samples and batch_size > 0:
        remaining = total_samples - current_index
        current_batch_size = min(batch_size, remaining)
        
        suffix = f"{current_index:05d}_{current_index + current_batch_size - 1:05d}"
        current_index = process_batch(recognizer, img_data, current_index, current_batch_size, suffix)
        
        batch_number += 1
        gc.collect()

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