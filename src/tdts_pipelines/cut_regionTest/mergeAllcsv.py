#mergeAllcsv.py
import os
import re
import pandas as pd
from pathlib import Path

def merge_cutting_recognition_files(file_pattern=None, input_dir="fd5\\outPutTrain", output_file="fd5_mix22_train.csv"):
    """
    合并切割识别结果文件
    
    参数:
    file_pattern: 文件模式，如果为None则自动匹配所有相关文件
    input_dir: 输入目录
    output_file: 输出文件名
    """
    
    # 如果未提供文件模式，则使用默认的文件名模式
    if file_pattern is None:
        file_pattern = r"mix22_train_\d+_\d+\.csv"
    
    # 获取所有匹配的文件
    input_path = Path(input_dir)
    all_files = [f for f in input_path.iterdir() if f.is_file() and re.match(file_pattern, f.name)]
    
    if not all_files:
        print("未找到匹配的文件")
        return None
    
    # 从文件名中提取起始序号并排序
    def extract_start_number(filename):
        match = re.search(r"batch_(\d+)_\d+\.csv", filename.name)
        return int(match.group(1)) if match else 0
    
    # 按起始序号排序文件
    sorted_files = sorted(all_files, key=extract_start_number)
    
    print(f"找到 {len(sorted_files)} 个文件，按顺序合并:")
    for f in sorted_files:
        print(f"  - {f.name}")
    
    # 读取并合并所有文件
    dfs = []
    for i, file_path in enumerate(sorted_files):
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 如果是第一个文件，保留标题；否则跳过标题
            if i == 0:
                dfs.append(df)
            else:
                dfs.append(df)
            
            print(f"已加载: {file_path.name} ({len(df)} 行)")
            
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            continue
    
    if not dfs:
        print("没有成功读取任何文件")
        return None
    
    # 合并所有DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 保存合并后的文件
    try:
        merged_df.to_csv(output_file, index=False)
        print(f"\n合并完成! 总共 {len(merged_df)} 行数据")
        print(f"输出文件: {output_file}")
        
        # 显示统计信息
        start_img = merged_df.iloc[0]['图片序号'] if '图片序号' in merged_df.columns else "未知"
        end_img = merged_df.iloc[-1]['图片序号'] if '图片序号' in merged_df.columns else "未知"
        print(f"图片序号范围: {start_img} - {end_img}")
        
        return output_file
        
    except Exception as e:
        print(f"保存合并文件时出错: {e}")
        return None

def merge_specific_files(file_list, output_file="merged_cutting_recognition_results.csv"):
    """
    合并指定的文件列表
    
    参数:
    file_list: 文件路径列表
    output_file: 输出文件名
    """
    if not file_list:
        print("文件列表为空")
        return None
    
    # 从文件名中提取起始序号并排序
    def extract_start_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r"batch_(\d+)_\d+\.csv", filename)
        return int(match.group(1)) if match else 0
    
    # 按起始序号排序文件
    sorted_files = sorted(file_list, key=extract_start_number)
    
    print(f"合并 {len(sorted_files)} 个文件:")
    for f in sorted_files:
        print(f"  - {os.path.basename(f)}")
    
    # 读取并合并所有文件
    dfs = []
    for i, file_path in enumerate(sorted_files):
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"已加载: {os.path.basename(file_path)} ({len(df)} 行)")
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            continue
    
    if not dfs:
        print("没有成功读取任何文件")
        return None
    
    # 合并所有DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 保存合并后的文件
    try:
        merged_df.to_csv(output_file, index=False,encoding='utf-8-sig')
        print(f"\n合并完成! 总共 {len(merged_df)} 行数据")
        print(f"输出文件: {output_file}")
        return output_file
    except Exception as e:
        print(f"保存合并文件时出错: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    # 方法1: 自动查找并合并所有匹配的文件
    print("方法1: 自动合并当前目录下所有匹配的文件")
    result_file = merge_cutting_recognition_files()
