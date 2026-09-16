#new3240-noise2img.py
#本程序生成左右叠加的双数字图片，并添加噪声
#修改完成，对于一位数字的情况，添加100
#论文分析重要的文件
#本代码采用最大值方式
from matplotlib import pyplot as plt
import pandas as pd
from torchvision.datasets import MNIST
from PIL import Image
import numpy as np
import struct
import random
from pathlib import Path

def process_data(df, coordinates, dataset):
    LeftIdx = df['Left Image Index'].tolist()
    rightIdx = df['Right Image Index'].tolist()
    twoLabel = df['Two - Digit Label'].tolist()
    xList = coordinates['x'].tolist()
    yList = coordinates['y'].tolist()

    num_images = len(LeftIdx)
    images = []
    labels = []

    for i in range(num_images):
        result = Image.new("L", (40, 32), 0)
        num_points = random.randint(20, 120) #噪声点数20-120
        for _ in range(num_points):
            # 随机选择像素位置
            x = random.randint(0, 40-1)
            y = random.randint(0, 32-1)
            # 随机生成噪声值
            noise_value = random.randint(12, 255)
            result.putpixel((x, y), noise_value)
        # 把img采用覆盖方式放到上面的image里面

        lidx = LeftIdx[i]
        ridx = rightIdx[i]
        left_img, _ = dataset[lidx]
        right_img, _ = dataset[ridx]

        result.paste(left_img, (0, 6))

        right_array = np.array(right_img)
        result_array = np.array(result)

        x0 = xList[i]
        y0 = yList[i]

        for ii in range(20):
            for j in range(20):
                x = x0 + ii
                y = y0 + j
                if 0 <= x < 40 and 0 <= y < 32:
                    if 0 <= x < 20 and 6 <= y < 26:                       
                        sum_value = max(result_array[y, x], right_array[j, ii])
                        result_array[y, x] = sum_value
                    else:
                        result_array[y, x] = right_array[j, ii]

        result_img = Image.fromarray(result_array)
        images.append(result_img)
        twolabel1= twoLabel[i] if twoLabel[i]>9 else twoLabel[i] +100
        labels.append(twolabel1)

    return images, labels


def save_mnist_data(images, labels, data_path, label_path):
    num_images = len(images)
    image_size = 40 * 32

    with open(data_path, 'wb') as data_file, open(label_path, 'wb') as label_file:
        data_file.write(struct.pack('>IIII', 2051, num_images, 32, 40))
        label_file.write(struct.pack('>II', 2049, num_images))

        for image, label in zip(images, labels):
            for row in image.getdata():
                data_file.write(struct.pack('B', row))
            label_file.write(struct.pack('B', label))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / 'data' / 'mnist_compact_20x20'
DATA_input = PROJECT_ROOT / 'data' / 'tdts_build_inputs'
DATA_output  = PROJECT_ROOT / 'data' / 'tdts_dataset'/ 'MNIST' / 'RAW' 
DATA_output.mkdir(parents=True, exist_ok=True)     # 不存在则创建

output_files = {
    'train_images': DATA_output / 'train-images-idx3-ubyte',
    'train_labels': DATA_output / 'train-labels-idx1-ubyte',
    'test_images':  DATA_output / 't10k-images-idx3-ubyte',
    'test_labels':  DATA_output / 't10k-labels-idx1-ubyte',
}

# 加载MNIST数据集
train_set = MNIST(str(DATA_ROOT), train=True, download=False)
test_set = MNIST(str(DATA_ROOT), train=False, download=False) 

# 读取训练集文件
train_df = pd.read_csv(DATA_input / 'mnist_image_pairs_train.csv') 
train_coordinates = pd.read_csv(DATA_input / 'coordinates_train.csv')

# 读取测试集文件
test_df = pd.read_csv(DATA_input / 'mnist_image_pairs_tst.csv')
test_coordinates = pd.read_csv(DATA_input / 'coordinates_tst.csv')

print("数据加载完成，处理数据中...")
# 检测是否已存在，全部存在则跳过
already_generated = all(p.exists() for p in output_files.values())
if already_generated:
    print(f"Dataset already exists at {DATA_output}, skipping generation.")
else:
    # 处理训练集数据
    train_images, train_labels = process_data(train_df, train_coordinates, train_set)
    # 处理测试集数据
    test_images, test_labels = process_data(test_df, test_coordinates, test_set)
    # 保存训练集数据
    save_mnist_data(train_images, train_labels,
                    str(output_files['train_images']),
                    str(output_files['train_labels']))
    # 保存测试集数据
    save_mnist_data(test_images, test_labels,
                    str(output_files['test_images']),
                    str(output_files['test_labels']))
    print(f"Dataset generated at {DATA_output}")
