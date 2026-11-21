#fig_watershed.py  --  论文 Fig 9：约束分水岭算法的分割结果
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2
from denoising1 import remove_salt_pepper_noise
from matplotlib.lines import Line2D
from water12 import robust_distance_transform_split

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = str(PROJECT_ROOT / 'data' / 'tdts_dataset')
OUT_DIR = PROJECT_ROOT / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

#本程序绘制带图例的切割线图像
# ============== 设置符合PLOS要求的字体 ==============
# 设置字体为Arial，字号10点（符合PLOS 8-12点要求）
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

def display_images(images, labels=None, indices=None, save_path=None):
    """
    符合PLOS格式的图片显示函数
    只支持5张图片显示
    标题格式: idx: [索引]
    
    参数:
    images: 图片列表，每个元素是张量或numpy数组
    labels: 标签列表 (可选)
    indices: 索引列表 (可选)
    save_base_name: 保存文件的基础名 (不含扩展名)
    """
    n_images = len(images)
    
    # 只支持5张图片
    if n_images != 5:
        raise ValueError("只支持显示5张图片")
    
    # 布局：1行5列
    n_rows, n_cols = 1, 5
    figsize = (7.5, 2.0)  # PLOS最大宽度7.5英寸，高度适当
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=300)
    axes_flat = axes.flatten()
    
    for i, (ax, img) in enumerate(zip(axes_flat, images)):
        # 处理图片格式
        if hasattr(img, 'squeeze'):
            img_display = img.squeeze()
        else:
            img_display = img

        # 如果是彩色图像 (通道 x 高度 x 宽度)，需要转置为 (高度 x 宽度 x 通道)
        if len(img_display.shape) == 3 and img_display.shape[0] == 3:
            img_display = img_display.permute(1, 2, 0)
        
        # 显示图片
        if len(img_display.shape) == 3:  # 彩色图像
            ax.imshow(img_display)
        else:  # 灰度图像
            ax.imshow(img_display, cmap='gray')    
        
        # 设置标题 - 只显示索引
        if indices and i < len(indices):
            ax.set_title(f"idx: {indices[i]}", fontsize=10, fontfamily='Arial')
        else:
            ax.set_title(f"Image {i+1}", fontsize=10, fontfamily='Arial')
        
        # 移除坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # 保存图片为PDF和TIFF格式
    if save_path:
        # 保存为PDF
        pdf_path = f'{save_path}.pdf'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"PDF saved: {pdf_path}")
        
        # 保存为TIFF（使用LZW压缩，符合PLOS要求）
        tiff_path = f'{save_path}.tiff'
        plt.savefig(tiff_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', 
                   format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
        print(f"TIFF saved: {tiff_path}")
    
    plt.show()
    return fig, axes

transform = transforms.Compose([transforms.ToTensor()])
img_data = MNIST(DATA_ROOT, download=False, transform=transform, train=True)

# 测试多个样本
test_indices = [1,181, 223, 243, 1091, 1108, 1274,1334, 3082, 3300]

# 准备图片列表
# 准备图片和标签
image_list = []
label_list = []
for idx in test_indices[:10]:  # 取前10张
    img, label = img_data[idx]
    image_list.append(img)
    label_list.append(label)

# 1 显示原始图片
#fig, axes = display_images(image_list,label_list,test_indices[:10], save_path='mnist1_2x5.png')
def deNoise(image_list):
    processed_images = []
    for img,  in image_list:
        # 您的处理逻辑
        img_np = (img.numpy().squeeze() * 255).astype(np.uint8) # 转为numpy数组，范围0-255，后续要保存这个类型，直到识别
        processed_img = remove_salt_pepper_noise(img_np) 
        processed_images.append(processed_img)
    return processed_images
#test_indices1 = [0, 1, 2, 3, 4] # 只测试5张，10张显示不下
test_indices1 = [181, 243, 1091, 1108, 3300] # 只测试5张，10张显示不下
#test_indices1 = [4, 5,1274,1334,509] # 只测试5张，10张显示不下

image_list.clear()
label_list.clear()
for idx in test_indices1[:5]:  # 取前5张
    img, label = img_data[idx]
    image_list.append(img)
    label_list.append(label)
noNoiseImgs = deNoise(image_list)

def do_waterCuts(image_list):
    """
    对图像列表中的每张图像生成切割线并绘制
    """
    processed_images = []
    for img in image_list:

        L_,R_,dt_img=robust_distance_transform_split(img)
        processed_images.append(dt_img)
    return processed_images
cutImages = do_waterCuts(noNoiseImgs)
fig, axes = display_images(cutImages,label_list,test_indices1[:5], save_path=str(OUT_DIR / 'mnist_waterCut5'))
