#waterMain.py
# 本程序用于产生分水岭算法路径的左右识别结果，主要用于产生基线数据
#11-19日修订， 代码修订来源：C:\Users\Administrator\Desktop\稳定版本代码\error\切割域干扰
# 1  把识别和前期处理分离，一次识别全部数据
# 2  把去噪算法修改为多进程模型，提高cpu利用率
# 11-21日修订， 代码修订来源目录：WaterRegn-1，修订目标：
# 1  删除无用代码，便于后期升级 
#   waterMain.py 删除process_batch1，相同功能使用process_batch替代
# 11-21日修改，经测试，效果和之前相同。
import torch
from torchvision import datasets, transforms
from u_left_right_recog import LeftRightRecognizer
import pandas as pd
import os
from inputimeout import inputimeout, TimeoutOccurred
import gc

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
    
    num_cuts = 1
    
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
    output_file = os.path.join('outPutTrain', f'water23_train_{suffix}.csv')#保存到outPut文件夹
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存到 {output_file}")
    

def main():
    """主程序：分批处理MNIST图片"""
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor()])
    img_data = datasets.MNIST('./data17', download=False, transform=transform, train=True)
    total_samples = len(img_data)
    
    print(f"MNIST训练集总样本数: {total_samples}")
    
    # 初始化识别器
    recognizer = LeftRightRecognizer('digit1_modelCln_state_epoch_23.pth')
    print("模型加载完成")
    
    # 用户输入处理参数
    print("\n请输入处理参数:")
    start_index = int(input("起始索引 (0-59999, 默认0): ") or 0)
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