#fig_cut_lines.py  --  论文 Fig 8：各算法的切割线对比（PLOS 版，300 dpi TIFF）
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import cv2
from cut_region import find_connected_points
from denoising1 import remove_salt_pepper_noise
from path_generationSafe10 import generate_cut_lines

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = str(PROJECT_ROOT / 'data' / 'tdts_dataset')
OUT_DIR = PROJECT_ROOT / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============== 设置符合PLOS要求的字体 ==============
# 设置字体为Arial，字号10点（符合PLOS 8-12点要求）
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# ============== 数据处理函数 ==============
def deNoise(image_list):
    processed_images = []
    for img in image_list:
        img_np = (img.numpy().squeeze() * 255).astype(np.uint8) # 转为numpy数组，范围0-255
        processed_img = remove_salt_pepper_noise(img_np) 
        processed_images.append(processed_img)
    return processed_images

def clear_points_on_image(original_image, points, color=None):
    """
    在uint8 numpy图像上标记点
    Args:
        original_image: numpy array, uint8类型, 形状为 (H, W) 或 (H, W, 1)
        points: 要标记的点列表 [(x, y), ...]
        color: RGB颜色元组, 值范围0-255
    Returns:
        processed_image: 标记后的三通道图像, uint8类型
    """
    # 将单通道灰度图像转换为三通道图像
    if len(original_image.shape) == 2:  # 如果是二维数组 (高度 x 宽度)
        # 复制到三个通道
        processed_image = np.stack([original_image] * 3, axis=2)  # 形状变为 (H, W, 3)
    elif len(original_image.shape) == 3 and original_image.shape[2] == 1:  # 如果是 (高度, 宽度, 1)
        processed_image = np.repeat(original_image, 3, axis=2)  # 复制到三个通道
    else:
        processed_image = original_image.copy()
    
    # 设置默认颜色（蓝色）
    if color is None:
        color = (51, 51, 153)  # R, G, B (0-255范围)
    
    # 标记点
    for x, y in points:
        if 0 <= x < processed_image.shape[1] and 0 <= y < processed_image.shape[0]:
            processed_image[y, x, 0] = color[0]  # 红色通道
            processed_image[y, x, 1] = color[1]  # 绿色通道  
            processed_image[y, x, 2] = color[2]  # 蓝色通道
    
    return processed_image

def draw_cut_lines_on_image(original_image, paths_info, colors=None):
    """
    在单张图像上绘制多条切割线，每条线使用不同颜色
    
    参数:
    - original_image: 原始图像
    - paths_info: 切割线信息列表，每个元素包含"path"、"identifier"和"vote_value"
    - colors: 可选的颜色列表，如果没有提供则使用默认颜色
    
    返回:
    - 绘制了切割线的图像
    """
    # 将单通道灰度图像转换为三通道"伪彩色"图像
    if len(original_image.shape) == 2:  # 如果是二维数组 (高度 x 宽度)
        processed_image = np.stack([original_image] * 3, axis=2)  # 复制到三个通道，形状变为 (H, W, 3)
    elif len(original_image.shape) == 3 and original_image.shape[2] == 1:  # 如果是 (高度, 宽度, 1)
        processed_image = np.repeat(original_image, 3, axis=2)  # 复制到三个通道
    else:
        processed_image = original_image.copy()
    
    # 默认颜色方案
    if colors is None:
        colors = [
            (255, 0, 0),      # 红色 - 路径1
            (0, 255, 0),      # 绿色 - 路径2
            (255, 0, 255),    # 紫色 - 路径3
            (0, 0, 255),      # 蓝色 - 路径4
            (0, 0, 255),      # 蓝色 - 路径5
            (0, 255, 255),    # 青色 - 路径6
            (255, 255, 0),    # 黄色 - 路径7
        ]
    
    # 获取图像尺寸
    image_height = processed_image.shape[0]  # 高度
    image_width = processed_image.shape[1]   # 宽度
    
    # 创建一个字典来存储每个位置的所有路径标识符
    positions = {}
    
    # 首先，收集所有位置上的路径标识符
    for path_info in paths_info:
        identifier = path_info["identifier"]
        path = path_info["path"]
        
        for point in path:
            x, y = point
            # 确保坐标在图像范围内
            if 0 <= x < image_width and 0 <= y < image_height:
                pos_key = (x, y)
                if pos_key not in positions:
                    positions[pos_key] = []
                positions[pos_key].append(identifier)
    
    # 然后，对于每个位置，根据某种规则选择要显示的颜色
    for (x, y), identifiers in positions.items():
        if not identifiers:
            continue
            
        # 方法1：根据y坐标的奇偶性交替显示
        # 这样可以确保在垂直方向上颜色交替
        if len(identifiers) > 1:
            # 如果有多个路径通过这个点，我们交替显示
            # 使用y坐标来决定显示哪个
            index = y % len(identifiers)
            identifier = identifiers[index]
        else:
            identifier = identifiers[0]
        
        color_index = (identifier - 1) % len(colors)
        color = colors[color_index]
        
        # 修正：使用正确的索引
        processed_image[y, x, 0] = color[0]  # 红色通道
        processed_image[y, x, 1] = color[1]  # 绿色通道
        processed_image[y, x, 2] = color[2]  # 蓝色通道
    
    return processed_image

def do_cut_lines(image_list):
    """
    对图像列表中的每张图像生成切割线并绘制
    """
    processed_images = []
    for img in image_list:
        points = find_connected_points(img)
        flag = np.zeros((32, 40), dtype=np.uint8)
        for x, y in points:
            flag[y, x] = 1
        pointInfo = generate_cut_lines(flag, img)
        dt_img = draw_cut_lines_on_image(img, pointInfo)
        processed_images.append(dt_img)
    return processed_images

# ============== 符合PLOS格式的切割线显示函数 ==============
def display_cut_lines_plos(images, labels=None, indices=None, save_base_name=None):
    """
    符合PLOS格式的切割线图片显示函数
    专为5张切割线图片设计，包含图例
    保存为PDF和TIFF格式
    
    参数:
    images: 图片列表，每个元素是numpy数组
    labels: 标签列表 (可选)
    indices: 索引列表 (可选)
    save_base_name: 保存文件的基础名 (不含扩展名)
    """
    n_images = len(images)
    
    # 只支持5张图片
    if n_images != 5:
        raise ValueError("切割线显示只支持5张图片")
    
    # 布局：1行6列，前5列用于图片，第6列用于图例
    n_rows, n_cols = 1, 6
    figsize = (7.5, 1.5)  # PLOS最大宽度7.5英寸，高度适当
    width_ratios = [1, 1, 1, 1, 1, 0.5]  # 前5列宽度为1，最后一列用于图例宽度为0.5
    
    # 创建图形，设置DPI为300（PLOS要求）
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=300,
                            gridspec_kw={'width_ratios': width_ratios})
    
    # 显示前5张图片
    for i in range(5):
        ax = axes[i]
        img = images[i]
        
        # 处理图片格式
        if hasattr(img, 'squeeze'):
            img_display = img.squeeze()
        else:
            img_display = img
        
        # 显示图片
        if len(img_display.shape) == 3:  # 彩色图像
            ax.imshow(img_display)
        else:  # 灰度图像
            ax.imshow(img_display, cmap='gray')
        
        # 设置标题（简化格式，符合PLOS要求）
        title_parts = []
        if indices and i < len(indices):
            title_parts.append(f"idx: {indices[i]}")
        
        if title_parts:
            ax.set_title(", ".join(title_parts), fontsize=10, fontfamily='Arial')
        
        # 移除坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 在第6个子图中生成图例
    legend_ax = axes[5]
    
    # 定义图注内容（字体大小调整为10点，符合PLOS要求）
    legend_items = [
        ((1.0, 0, 0), "A* Left"),
        ((0, 1.0, 0), "A* Right"), 
        ((1.0, 0, 1.0), "A* Center"),
        ((0, 0, 1.0), "Dijkstra"),
        ((1.0, 1.0, 0), "Projection")
    ]
    
    # 直接在图例子图中绘制图例
    legend_ax.clear()
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    
    y_pos = 0.75  # 起始y位置
    y_step = 0.16  # 每个图注项的垂直间距
    
    for color, label in legend_items:
        # 添加颜色方块
        legend_ax.add_patch(plt.Rectangle((0.05, y_pos-0.025), 0.15, 0.05, 
                                        facecolor=color, transform=legend_ax.transAxes))
        # 添加文本标签（使用Arial字体，10点大小）
        legend_ax.text(0.25, y_pos, label, fontsize=10, fontfamily='Arial',
                     transform=legend_ax.transAxes, verticalalignment='center')
        y_pos -= y_step
    
    # 设置图注标题（使用Arial字体，10点大小）
    legend_ax.text(0.5, 0.95, "Path Legend", fontsize=10, fontfamily='Arial',
                   transform=legend_ax.transAxes, horizontalalignment='center')
    
    # 移除坐标轴
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_frame_on(False)
    
    plt.tight_layout()
    
    # 保存图片为PDF和TIFF格式
    if save_base_name:
        # 保存为PDF
        pdf_path = f'{save_base_name}.pdf'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"PDF saved: {pdf_path}")
        
        # 保存为TIFF（使用LZW压缩，符合PLOS要求）
        tiff_path = f'{save_base_name}.tiff'
        plt.savefig(tiff_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', 
                   format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
        print(f"TIFF saved: {tiff_path}")
    
    plt.show()
    return fig, axes

# ============== 主程序 ==============
if __name__ == "__main__":
    # 数据加载
    transform = transforms.Compose([transforms.ToTensor()])
    img_data = MNIST(DATA_ROOT, download=False, transform=transform, train=True)
    
    # 测试样本 - 只测试5张
    test_indices1 = [181, 243, 1091, 1108, 3300]
    
    # 准备图片和标签
    image_list = []
    label_list = []
    for idx in test_indices1[:5]:
        img, label = img_data[idx]
        image_list.append(img)
        label_list.append(label)
    
    # 1. 去噪处理
    noNoiseImgs = deNoise(image_list)
    
    # 2. 切割线处理
    processed_images3 = do_cut_lines(noNoiseImgs)
    
    # 3. 显示切割线图片（使用PLOS格式）
    fig, axes = display_cut_lines_plos(processed_images3, label_list, test_indices1, 
                                       save_base_name=str(OUT_DIR / 'mnist_cut_lines_plos'))
    
    print("切割线图片已按PLOS格式要求保存为PDF和TIFF格式")