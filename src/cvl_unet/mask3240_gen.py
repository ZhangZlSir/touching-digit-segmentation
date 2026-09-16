#mask3240_gen.py
#
#目的：只生成"逐像素真值分割掩码"，并且严格对齐已有的 TDTS 图像（data17里已经生成好的
#train/test idx-ubyte 文件），不重新生成图像本身——这样不会因为噪声点是随机撒的
#而导致掩码和已经用于论文实验的那批图像对不上。
#
#关键观察：噪声点的位置/取值是随机的，但字符区域(哪个像素属于左字符/右字符)完全由
#  - LeftIdx / rightIdx （决定具体是哪张MNIST图）
#  - coordinates 里的 (x0, y0) （决定右字符贴的位置）
#这两组信息决定，和噪声无关。所以只要用和生成 data17 时同一份
#mnist_image_pairs_*.csv / coordinates_*.csv（按同样的行顺序），就能精确复原出
#与 data17 里每一张图一一对应的真值掩码，不需要重新跑一遍图像生成、也不用担心随机种子。
#
#掩码取值: 0=背景, 1=左字符, 2=右字符
#归属规则和 new3240-noise2img.py 里的像素融合规则完全一致：
#   - 左字符落笔区域 (paste 到 (0,6)) 内，只要左字符灰度>0，标为1（会被右字符按下面规则覆盖/合并）
#   - 与右字符可能重叠的子区域 (0<=x<20 且 6<=y<26)：取 max(左,右)，谁的灰度大掩码就标谁
#   - 右字符落笔区域内其余部分：右字符直接覆盖，标为2
#
#用法：
#   python mask3240_gen.py
#输出：
#   ./data18/train-masks.npy   形状 (N_train, 32, 40)，与 data17 的训练图像逐条对应
#   ./data18/test-masks.npy    形状 (N_test, 32, 40)， 与 data17 的测试图像逐条对应

import os
import numpy as np
import pandas as pd
from torchvision.datasets import MNIST


def build_masks(df, coordinates, dataset):
    LeftIdx = df['Left Image Index'].tolist()
    rightIdx = df['Right Image Index'].tolist()
    xList = coordinates['x'].tolist()
    yList = coordinates['y'].tolist()

    num_images = len(LeftIdx)
    masks = np.zeros((num_images, 32, 40), dtype=np.uint8)

    for i in range(num_images):
        left_img, _ = dataset[LeftIdx[i]]
        right_img, _ = dataset[rightIdx[i]]
        left_array = np.array(left_img)   # 20x20
        right_array = np.array(right_img)  # 20x20

        mask = masks[i]

        # 左字符先落笔 (paste 到 (0,6))
        for jj in range(20):
            for ii in range(20):
                if left_array[jj, ii] > 0:
                    mask[6 + jj, ii] = 1

        x0, y0 = xList[i], yList[i]
        for ii in range(20):
            for j in range(20):
                x = x0 + ii
                y = y0 + j
                if 0 <= x < 40 and 0 <= y < 32:
                    rv = right_array[j, ii]
                    if 0 <= x < 20 and 6 <= y < 26:
                        # 与左字符共享的重叠子区域：max融合规则决定归属
                        lv = left_array[y - 6, x] if (0 <= y - 6 < 20 and 0 <= x < 20) else 0
                        if rv > 0 and rv >= lv:
                            mask[y, x] = 2
                        # 否则保留左字符标注（若lv>0已在上面置1），或保持背景
                    else:
                        if rv > 0:
                            mask[y, x] = 2

    return masks


if __name__ == '__main__':
    train_set = MNIST('../tdts_build/data3', train=True, download=False)
    test_set = MNIST('../tdts_build/data3', train=False, download=False)

    train_df = pd.read_csv('../tdts_build/csv/mnist_image_pairs_train.csv')
    train_coordinates = pd.read_csv('../tdts_build/csv/coordinates_train.csv')
    test_df = pd.read_csv('../tdts_build/csv/mnist_image_pairs_tst.csv') 
    test_coordinates = pd.read_csv('../tdts_build/csv/coordinates_tst.csv') 

    print("正在根据坐标文件复原真值分割掩码（与 data17 已有图像逐条对应）...") 
    train_masks = build_masks(train_df, train_coordinates, train_set)
    test_masks = build_masks(test_df, test_coordinates, test_set)

    os.makedirs('./data18', exist_ok=True) 
    np.save('./data18/train-masks.npy', train_masks)
    np.save('./data18/test-masks.npy', test_masks)
    print(f"完成：train_masks {train_masks.shape}, test_masks {test_masks.shape} 已保存到 ./data18")
