import os
from collections import Counter

def count_categories(folder_path):
    """
    统计文件夹内图片文件名前两位（数字类别）的分布
    仅统计文件名前两位均为数字的文件
    """
    all_items = os.listdir(folder_path)
    
    # 筛选文件（可根据需要保留特定图片扩展名，这里保留所有文件）
    files = [f for f in all_items if os.path.isfile(os.path.join(folder_path, f))]
    
    prefixes = []
    for f in files:
        basename = os.path.splitext(f)[0]  # 去除扩展名
        if len(basename) >= 2:
            prefix = basename[:2]
            # 关键：只统计前两位为数字的情况
            if prefix.isdigit():
                prefixes.append(prefix)
            # 若前两位不是数字，则忽略（不统计）
    
    counter = Counter(prefixes)
    return counter

if __name__ == "__main__":
    folder = input("请输入文件夹路径：").strip()
    if not os.path.isdir(folder):
        print("路径不存在或不是文件夹")
    else:
        result = count_categories(folder)
        print(f"共有 {len(result)} 种类别（前两位数字组合）")
        print("各类别数量如下：")
        for prefix, count in sorted(result.items()):
            print(f"{prefix}: {count}")