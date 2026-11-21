#fig_denoise.py  --  论文 Fig 4/5：原始粘连图与去噪结果（PLOS 版，300 dpi TIFF）
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2
from denoising1 import remove_salt_pepper_noise
import sys
import matplotlib
from matplotlib import rcParams

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = str(PROJECT_ROOT / 'data' / 'tdts_dataset')
OUT_DIR = PROJECT_ROOT / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ====================== PLOS ONE 格式设置 ======================
# 设置符合PLOS的字体
PLOS_FONTS = ['Arial', 'Times New Roman', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = PLOS_FONTS
matplotlib.rcParams['font.size'] = 10  # 8-12点范围内
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9

# 设置DPI符合PLOS要求 (300-600 dpi)
PLOS_DPI = 300
PLOS_PAD = 0.1033  # 与投稿版 2220x899 对齐的 tight-bbox 边距（英寸）
# 是否复刻投稿版版式：python fig_denoise.py --plos-exact
PLOS_EXACT = '--plos-exact' in sys.argv

def display_images(images, labels=None, indices=None, save_path=None, title=None):
    """
    显示图片网格，支持多种布局
    符合PLOS ONE期刊要求
    
    参数:
    images: 图片列表，每个元素是张量或numpy数组
    labels: 标签列表 (可选)
    indices: 索引列表 (可选)
    save_path: 保存路径前缀 (不带扩展名)
    title: 图形标题 (可选)
    """
    n_images = len(images)
    
    # 确定布局
    if n_images == 9:
        n_rows, n_cols = 3, 3
        # PLOS要求：宽度在2.63-7.5英寸之间
        fig_width_inches = 7.5  # 最大宽度
        fig_height_inches = 7.5  # 确保不超过8.75英寸
    elif n_images == 10:
        n_rows, n_cols = 2, 5
        if PLOS_EXACT:
            fig_width_inches = 7.75   # tight bbox 后 2220 像素（与投稿版一致）
            fig_height_inches = 3.8437  # tight bbox 后 900 像素（投稿版 899，差 1px）
        else:
            fig_width_inches = 7.5    # 7.5英寸 = 2250像素 (300 dpi)
            fig_height_inches = 3.5   # 3.5英寸 = 1050像素
    elif n_images == 6:
        n_rows, n_cols = 2, 3
        fig_width_inches = 6.0  # 6.0英寸 = 1800像素
        fig_height_inches = 4.0  # 4.0英寸 = 1200像素
    elif n_images == 5:
        n_rows, n_cols = 1, 6  # 改为6列，最后一列用于图注
        fig_width_inches = 7.5  # 7.5英寸
        fig_height_inches = 2.5  # 2.5英寸
        width_ratios = [1, 1, 1, 1, 1, 0.6]  # 前5列宽度为1，最后一列为0.6
    else:
        raise ValueError(f"不支持显示 {n_images} 张图片")
    
    # 创建图形，根据PLOS要求设置尺寸
    if n_images == 5:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width_inches, fig_height_inches), 
                                dpi=PLOS_DPI,
                                gridspec_kw={'width_ratios': width_ratios})
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width_inches, fig_height_inches),
                                dpi=PLOS_DPI)
    
    # 对于5张图片，只取前5个子图用于显示图片
    if n_images == 5:
        axes_flat = axes[:5]  # 只取前5个轴显示图片
    else:
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
        
        # 设置标题 (使用符合PLOS的字体大小)
        if PLOS_EXACT:
            # 投稿版版式：标题为 "idx | lbl"，字号 11
            if indices and i < len(indices) and labels and i < len(labels):
                ax.set_title(f"{indices[i]} | {labels[i]}", fontsize=11)
            elif indices and i < len(indices):
                ax.set_title(f"{indices[i]}", fontsize=11)
            else:
                ax.set_title(f"Image {i+1}", fontsize=11)
        else:
            title_parts = []
            if indices and i < len(indices):
                title_parts.append(f"idx: {indices[i]}")
            if labels and i < len(labels):
                title_parts.append(f"lbl: {labels[i]}")
            if title_parts:
                ax.set_title(", ".join(title_parts), fontsize=10)
            else:
                ax.set_title(f"Image {i+1}", fontsize=10)
        
        # 移除坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 对于5张图片的情况，生成图例图片并添加到第6个子图
    if n_images == 5:
        legend_ax = axes[5]
        
        # 生成图例图片
        legend_fig, legend_ax_temp = plt.subplots(figsize=(2, 3.2), dpi=PLOS_DPI)
        legend_fig.patch.set_facecolor('white')
        
        # 定义图注内容
        legend_items = [
            ((1.0, 0, 0), "A* Left"),
            ((0, 1.0, 0), "A* Right"), 
            ((1.0, 0, 1.0), "A* Center"),
            ((0, 0, 1.0), "Dijkstra"),
            ((1.0, 1.0, 0), "Projection")
        ]
        
        # 添加图注
        y_pos = 0.90  # 起始y位置
        y_step = 0.18  # 每个图注项的垂直间距
        
        for color, label in legend_items:
            # 添加颜色方块
            legend_ax_temp.add_patch(plt.Rectangle((0.05, y_pos-0.03), 0.15, 0.05, 
                                                facecolor=color, transform=legend_ax_temp.transAxes))
            # 添加文本标签
            legend_ax_temp.text(0.25, y_pos, label, fontsize=12,  # 符合PLOS 8-12点要求
                             transform=legend_ax_temp.transAxes, verticalalignment='center')
            y_pos -= y_step
        
        # 设置图注属性
        legend_ax_temp.set_xlim(0, 1)
        legend_ax_temp.set_ylim(0, 1)
        legend_ax_temp.set_xticks([])
        legend_ax_temp.set_yticks([])
        legend_ax_temp.set_title("Path Legend", fontsize=12, pad=2, fontweight='bold')
        
        # 将图注转换为图像数组
        legend_fig.canvas.draw()
        legend_img = np.frombuffer(legend_fig.canvas.tostring_rgb(), dtype=np.uint8)
        legend_img = legend_img.reshape(legend_fig.canvas.get_width_height()[::-1] + (3,))
        
        # 关闭临时图注图像
        plt.close(legend_fig)
        
        # 在主图中显示图注
        legend_ax.imshow(legend_img)
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])
        legend_ax.set_title("")  # 移除原有标题
    
    # 如果有主标题，添加主标题
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    
    # 调整布局，增加留白符合PLOS要求
    plt.tight_layout(pad=2.0)
    
    # 保存图片为TIFF和PDF格式（PLOS要求的格式）
    if save_path:
        # 保存为TIFF格式（LZW压缩）
        tiff_path = f"{save_path}.tiff"
        plt.savefig(tiff_path, dpi=PLOS_DPI, bbox_inches='tight', 
                   pad_inches=(PLOS_PAD if PLOS_EXACT else 0.1),
                   facecolor='white', edgecolor='none',
                   format='tiff',
                   pil_kwargs={'compression': 'tiff_lzw'})
        
        # 保存为PDF格式
        pdf_path = f"{save_path}.pdf"
        plt.savefig(pdf_path, dpi=PLOS_DPI, bbox_inches='tight',
                   pad_inches=(PLOS_PAD if PLOS_EXACT else 0.1),
                   facecolor='white', edgecolor='none',
                   format='pdf')
        
        # 计算并显示图片尺寸信息
        width_pixels = int(fig_width_inches * PLOS_DPI)
        height_pixels = int(fig_height_inches * PLOS_DPI)
        
        print(f"✓ 图片已保存，符合PLOS ONE格式要求")
        print(f"✓ TIFF文件: {tiff_path} (LZW压缩)")
        print(f"✓ PDF文件: {pdf_path}")
        print(f"✓ 尺寸: {fig_width_inches:.2f}×{fig_height_inches:.2f} 英寸")
        print(f"✓ 分辨率: {PLOS_DPI} dpi")
        print(f"✓ 像素尺寸: {width_pixels}×{height_pixels} 像素")
        print(f"✓ 宽度检查: {width_pixels}像素 - ", end="")
        if 789 <= width_pixels <= 2250:
            print("✓ 在PLOS要求的789-2250像素范围内")
        else:
            print("✗ 超出PLOS要求范围")
        
        print(f"✓ 高度检查: {height_pixels}像素 - ", end="")
        if height_pixels <= 2625:
            print("✓ 低于PLOS最大高度2625像素")
        else:
            print("✗ 超过PLOS最大高度")
        
        print(f"✓ 字体检查: Arial/Times, 8-12点")
    
    plt.show()
    return fig, axes

# ====================== 主程序 ======================

if __name__ == "__main__":
    print("=" * 60)
    print("PLOS ONE期刊格式 - MNIST去噪图像展示")
    print("=" * 60)
    
    # 数据转换
    transform = transforms.Compose([transforms.ToTensor()])
    img_data = MNIST(DATA_ROOT, download=False, transform=transform, train=True)
    
    # 测试多个样本
    test_indices = [1, 181, 223, 243, 1091, 1108, 1274, 1334, 3082, 3300]
    
    # 准备图片和标签
    image_list = []
    label_list = []
    for idx in test_indices[:10]:  # 取前10张
        img, label = img_data[idx]
        image_list.append(img)
        label_list.append(label)
    
    # 1. 显示原始图片 (2×5布局)
    print("\n1. 显示原始MNIST图片 (10张, 2×5布局)")
    fig1, axes1 = display_images(
        image_list, 
        labels=label_list, 
        indices=test_indices[:10], 
        save_path=str(OUT_DIR / 'mnist_original_2x5'),
        #title="Original MNIST Images"
    )
    
    # 去噪函数
    def deNoise(image_list):
        processed_images = []
        for img in image_list:
            img_np = (img.numpy().squeeze() * 255).astype(np.uint8) # 转为numpy数组，范围0-255
            processed_img = remove_salt_pepper_noise(img_np) 
            processed_images.append(processed_img)
        return processed_images
    
    # 应用去噪
    noNoiseImgs = deNoise(image_list)
    
    # 2. 显示去噪后的图片 (2×5布局)
    print("\n2. 显示去噪后的MNIST图片 (10张, 2×5布局)")
    fig2, axes2 = display_images(
        noNoiseImgs,
        labels=label_list,
        indices=test_indices[:10],
        save_path=str(OUT_DIR / 'mnist_denoised_2x5'),
        #title="Denoised MNIST Images"
    )
    
    # 3. 显示对比图 (3×3布局，前9张)
    print("\n3. 显示对比图 (前9张, 3×3布局)")
    # 创建对比图片列表：原始和去噪交替
    comparison_images = []
    comparison_labels = []
    comparison_indices = []
    
    for i in range(9):  # 前9张
        # 原始图片
        img, label = img_data[test_indices[i]]
        img_np = (img.numpy().squeeze() * 255).astype(np.uint8)
        comparison_images.append(img_np)
        comparison_labels.append(f"Original: {label}")
        comparison_indices.append(test_indices[i])
        
        # 去噪图片
        denoised_img = remove_salt_pepper_noise(img_np)
        comparison_images.append(denoised_img)
        comparison_labels.append(f"Denoised: {label}")
        comparison_indices.append(test_indices[i])
    
    # 由于我们现在有18张图片(9对)，我们可以选择显示前6对(12张)或者前9对(18张)
    # 这里我们显示前6对(12张，3×4布局)
    print("\n4. 显示原始vs去噪对比图 (12张, 3×4布局)")
    
    # 创建3×4布局
    fig_width_inches = 7.5
    fig_height_inches = 6.0
    fig3, axes3 = plt.subplots(3, 4, figsize=(fig_width_inches, fig_height_inches), dpi=PLOS_DPI)
    
    for i in range(6):  # 6对
        row = i // 2
        col = (i % 2) * 2
        
        # 原始图片
        img, label = img_data[test_indices[i]]
        img_np = (img.numpy().squeeze() * 255).astype(np.uint8)
        axes3[row, col].imshow(img_np, cmap='gray')
        axes3[row, col].set_title(f"Original (idx: {test_indices[i]}, lbl: {label})", fontsize=9)
        axes3[row, col].set_xticks([])
        axes3[row, col].set_yticks([])
        
        # 去噪图片
        denoised_img = remove_salt_pepper_noise(img_np)
        axes3[row, col+1].imshow(denoised_img, cmap='gray')
        axes3[row, col+1].set_title(f"Denoised (idx: {test_indices[i]}, lbl: {label})", fontsize=9)
        axes3[row, col+1].set_xticks([])
        axes3[row, col+1].set_yticks([])
    
    # 设置主标题
    #fig3.suptitle("MNIST Images: Original vs Denoised", fontsize=12, fontweight='bold')
    
    # 调整布局
    plt.tight_layout(pad=2.0)
    
    # 保存对比图
    save_path_base = str(OUT_DIR / 'mnist_comparison_3x4')
    tiff_path = f"{save_path_base}.tiff"
    pdf_path = f"{save_path_base}.pdf"
    
    plt.savefig(tiff_path, dpi=PLOS_DPI, bbox_inches='tight',
               pad_inches=0.1, facecolor='white', edgecolor='none',
               format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    plt.savefig(pdf_path, dpi=PLOS_DPI, bbox_inches='tight',
               pad_inches=0.1, facecolor='white', edgecolor='none',
               format='pdf')
    
    # 显示图片信息
    width_pixels = int(fig_width_inches * PLOS_DPI)
    height_pixels = int(fig_height_inches * PLOS_DPI)
    
    print(f"✓ 对比图已保存:")
    print(f"  TIFF文件: {tiff_path}")
    print(f"  PDF文件: {pdf_path}")
    print(f"  尺寸: {fig_width_inches:.2f}×{fig_height_inches:.2f} 英寸")
    print(f"  像素: {width_pixels}×{height_pixels}")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("所有图片已按PLOS ONE格式要求保存")
    print("=" * 60)