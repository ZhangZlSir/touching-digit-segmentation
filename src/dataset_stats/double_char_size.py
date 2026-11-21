#sizeAll.py
#本程序用于将图片的空白部分裁剪掉,并计算图片的宽度和高度，保存到csv文件中
#处理的是data4文件夹下的图片，双数字图片，外边界大小没有实际意义了，这个是没有加噪的图片
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd #数据处理和csv文件读写的库

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_IN = PROJECT_ROOT / 'data' / 'mnist_compact_20x20'          # 20x20 compact MNIST characters
DATA_TDTS = PROJECT_ROOT / 'data' / 'tdts_dataset'               # TDTS two-digit touching set
BUILD_INPUTS = PROJECT_ROOT / 'data' / 'tdts_build_inputs'       # coordinates / pair index CSVs
OUT_DIR = PROJECT_ROOT / 'results' / 'stats'
OUT_DIR.mkdir(parents=True, exist_ok=True)
  #20对应7.8%  12对应4.7%
def find_bounding_box(img, threshold=12):
    """找到图像中有效像素的边界框"""
    height, width = img.shape 

    bounding_box= (0, 0, width, height)

    # 初始化边界值
    top = height
    bottom = -1
    left = width
    right = -1

    # 寻找上下边界
    for y in range(height):
        for x in range(width):
            if img[y, x] >= threshold:
                if y < top:
                    top = y
                break  # 找到第一个有效行后退出
        if top != height and y > bottom:
            for x in range(width):
                if img[y, x] >= threshold:
                    bottom = y
                    break

    # 寻找左右边界
    for x in range(width):
        for y in range(height):
            if img[y, x] >= threshold:
                if x < left:
                    left = x
                break  # 找到第一个有效列后退出
        if left != width and x > right:
            for y in range(height):
                if img[y, x] >= threshold:
                    right = x
                    break

    # 处理未找到有效区域的情况
    if left > right or top > bottom:
        bounding_box= (0, 0, width, height)
    else:
        bounding_box= (left, top, right + 1, bottom + 1)
    return bounding_box

transform = transforms.Compose([transforms.ToTensor()])
img_data = MNIST(str(DATA_TDTS), download=False,transform=transform)#第二次运行时，不再下载
img_loader  = DataLoader(img_data, batch_size=600,shuffle=False) #上一行默认参数是train=True,没写表示默认值，所有这里是测试集

img0,label0 = img_loader.dataset[0]
img0 = img0.numpy().squeeze(axis=0) 
imgnp0 = (img0 * 255).astype('uint8')
print(imgnp0)

size=[]
i=0;
for batch_imgs, a_label in img_loader:
    #imgs_np = batch_imgs  # [128, 28, 28]
    imgs_np = batch_imgs.numpy().squeeze(axis=1)  # [128, 28, 28]
    imgs_np = (imgs_np * 255).astype('uint8')
    
    print(i)
    i+=1
    for img in imgs_np:
        left, top, right, bottom= find_bounding_box(img)
        size.append((right-left,bottom-top))


index_column = list(range(len(size)))
# 提取宽度和高度列
widths = [item[0] for item in size]
heights = [item[1] for item in size]

# 创建包含序号、宽度和高度的 DataFrame
data = {
    'No': index_column,
    'width': widths,
    'height': heights
}
df = pd.DataFrame(data)
csv_file_path = 'AllSize4032.csv'
df.to_csv(csv_file_path, index=False)
print(len(size))